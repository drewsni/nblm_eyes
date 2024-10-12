import os
import sys
import time
import subprocess
import json
import re
import google.generativeai as genai
from google.api_core import exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import yt_dlp
import whisper
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# User-configurable variables
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=vYyUb_MI7to"
VIDEO_NAME = "goprocat"
WHISPER_MODEL = "turbo"
GEMINI_MODEL = "gemini-1.5-flash-002"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MIN_CHUNK_DURATION = 30  # Minimum chunk duration in seconds
TARGET_CHUNKS = 10  # Target number of chunks
TIMESTAMP_MODE = "model"  # Options: "chunk", "model", "none"

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# Configure your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

def download_video(url, output_path):
    if os.path.exists(output_path):
        print(f"Video already downloaded: {output_path}")
        return output_path
    
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    print(f"Downloaded video: {filename}")
    return filename

def get_video_duration(video_file):
    duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_file}"'
    try:
        duration = float(subprocess.check_output(duration_cmd, shell=True).decode('utf-8').strip())
        return duration
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        raise

def split_video(video_file, output_folder):
    duration = get_video_duration(video_file)
    print(f"Total video duration: {duration:.2f} seconds")
    
    chunk_duration = max(duration / TARGET_CHUNKS, MIN_CHUNK_DURATION)
    num_chunks = min(TARGET_CHUNKS, int(duration // chunk_duration) + (1 if duration % chunk_duration > 0 else 0))
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        chunk_filename = f"chunk{i+1}_{start:.2f}_{end:.2f}.mp4"
        chunk_path = os.path.join(output_folder, chunk_filename)
        if not os.path.exists(chunk_path):
            cmd = f'ffmpeg -i "{video_file}" -ss {start:.2f} -to {end:.2f} -c copy "{chunk_path}" -y'
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Created chunk: {chunk_filename}")
        chunks.append((chunk_path, start, end))
        print(f"Chunk {i+1}: {start:.2f} - {end:.2f} (duration: {end - start:.2f}s)")
    
    return chunks

def process_with_gemini(video_file, model, prompt):
    try:
        uploaded_file = genai.upload_file(video_file)
        print(f"Processing with Gemini: {video_file}")
        
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(5)
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            print(f"Processing failed: {uploaded_file.state.name}")
            return None

        response = model.generate_content([prompt, uploaded_file], request_options={"timeout": 300})
        
        if response.prompt_feedback.block_reason:
            print(f"Prompt blocked. Reason: {response.prompt_feedback.block_reason}")
            return None
        
        if response.candidates[0].finish_reason == "SAFETY":
            print("Response blocked due to safety concerns.")
            for rating in response.candidates[0].safety_ratings:
                print(f"Category: {rating.category}, Probability: {rating.probability}")
            return None
        
        return response.text

    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return None
    finally:
        if 'uploaded_file' in locals():
            uploaded_file.delete()

def process_chunk_with_whisper(chunk_file, whisper_model):
    print(f"Processing chunk with Whisper: {chunk_file}")
    try:
        result = whisper_model.transcribe(chunk_file)
        formatted_result = ""
        for segment in result["segments"]:
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            formatted_result += f"[{start_time} - {end_time}] {segment['text']}\n"
        return formatted_result
    except Exception as e:
        print(f"Error transcribing chunk {chunk_file} with Whisper: {e}")
        return None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes}:{seconds:02}"

def parse_timestamp(timestamp):
    parts = timestamp.split(':')
    parts = [part.replace(',', '.') for part in parts]  # Replace commas with dots if necessary
    try:
        parts = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

def clean_transcript(results, timestamp_mode):
    sorted_chunks = sorted(results.items(), key=lambda x: int(x[0]))
    
    full_transcript = ""
    total_duration = 0

    for i, (_, chunk) in enumerate(sorted_chunks):
        # Validate chunk data
        if not all(key in chunk for key in ('transcription', 'start', 'end')):
            print(f"Chunk {i+1} is missing required data. Skipping.")
            continue  # Skip this chunk
        
        transcription = chunk['transcription']
        start_time = chunk['start']
        end_time = chunk['end']
        
        if start_time >= end_time:
            print(f"Chunk {i+1} has invalid start and end times. Skipping.")
            continue  # Skip this chunk
        
        if not transcription:
            print(f"Chunk {i+1} has an empty transcription. Skipping.")
            continue  # Skip empty transcriptions
        
        chunk_duration = end_time - start_time
        
        print(f"Processing Chunk {i+1}: {start_time:.2f} - {end_time:.2f} (Duration: {chunk_duration:.2f}s)")
        
        # Remove asterisks only around timestamps
        transcription = re.sub(r'\*\[(.*?)\]\*', r'[\1]', transcription)
        
        # Adjust timestamps using updated regex
        def adjust_timestamp(match):
            ts_start, ts_end = match.groups()
            try:
                new_start = format_time(parse_timestamp(ts_start) + start_time)
                new_end = format_time(parse_timestamp(ts_end) + start_time)
                return f"[{new_start} - {new_end}]"
            except ValueError:
                return match.group(0)  # Return the original match if parsing fails
        
        # Updated regex to allow one or two digits for minutes and handle optional milliseconds
        transcription = re.sub(
            r'\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?) - (\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]',
            adjust_timestamp,
            transcription
        )
        
        # Build the full transcript based on the chosen timestamp mode
        if timestamp_mode == "chunk":
            full_transcript += f"[{format_time(total_duration)} - {format_time(total_duration + chunk_duration)}]\n{transcription.strip()}\n\n"
        elif timestamp_mode == "model":
            full_transcript += transcription + "\n\n"
        else:  # timestamp_mode == "none"
            # Remove all timestamps
            clean_text = re.sub(r'\[.*?\]', '', transcription).strip()
            full_transcript += clean_text + "\n\n"
        
        total_duration += chunk_duration
    
    print(f"Total duration from chunks: {total_duration:.2f} seconds")
    print(f"Final timestamp: {format_time(total_duration)}")
    
    return full_transcript.strip()

def get_next_file_number(folder, base_name):
    existing_files = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith('.txt')]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        match = re.search(r'_(\d+)\.txt$', f)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1

def main():
    data_folder = "data"
    video_folder = os.path.join(data_folder, VIDEO_NAME)
    chunks_folder = os.path.join(video_folder, "chunks")
    transcript_folder = os.path.join(video_folder, "transcripts")
    os.makedirs(chunks_folder, exist_ok=True)
    os.makedirs(transcript_folder, exist_ok=True)

    video_filename = f"{VIDEO_NAME}.mp4"
    json_output_filename = f"{VIDEO_NAME}_transcription.json"
    json_path = os.path.join(transcript_folder, json_output_filename)

    if os.path.exists(json_path):
        print(f"JSON transcription file already exists: {json_path}")
        print("Skipping video download and transcription steps.")
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        print("Downloading video from YouTube...")
        try:
            video_file = download_video(YOUTUBE_VIDEO_URL, os.path.join(video_folder, video_filename))
        except Exception as e:
            print(f"Error downloading video: {e}")
            sys.exit(1)

        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        if TIMESTAMP_MODE == "model":
            prompt = "Transcribe the audio with timestamps, and provide visual descriptions for each scene. Format the output as [MM:SS - MM:SS] Transcription (Visual description)"
        else:
            prompt = "Transcribe the audio and provide visual descriptions for each scene. Do not include timestamps in your output."

        print("Attempting to process entire video with Gemini...")
        full_video_result = process_with_gemini(video_file, gemini_model, prompt)
        
        if full_video_result is not None:
            print("Successfully processed entire video with Gemini.")
            video_duration = get_video_duration(video_file)
            results = {1: {'file': video_file, 'transcription': full_video_result, 'method': 'gemini', 'start': 0, 'end': video_duration}}
        else:
            print("Gemini failed to process the entire video. Falling back to chunking...")
            chunks = split_video(video_file, chunks_folder)
            results = {}
            whisper_model = None  # Initialize Whisper model only if needed

            for i, (chunk, start, end) in enumerate(chunks, 1):
                print(f"Processing chunk {i}/{len(chunks)}...")
                result = process_with_gemini(chunk, gemini_model, prompt)
                if result is None:
                    print(f"Gemini failed to process chunk {i}. Falling back to Whisper for this chunk.")
                    if whisper_model is None:
                        whisper_model = whisper.load_model(WHISPER_MODEL)
                    result = process_chunk_with_whisper(chunk, whisper_model)
                    method = 'whisper'
                else:
                    method = 'gemini'
                results[i] = {'file': chunk, 'transcription': result, 'method': method, 'start': start, 'end': end}

        # Save results to JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON transcription file saved: {json_path}")

    # Clean and consolidate transcript
    cleaned_transcript = clean_transcript(results, TIMESTAMP_MODE)

    # Generate a new numbered filename for the transcript
    base_name = f"{VIDEO_NAME}_full_transcript"
    file_number = get_next_file_number(transcript_folder, base_name)
    txt_output_filename = f"{base_name}_{file_number}.txt"
    txt_path = os.path.join(transcript_folder, txt_output_filename)

    # Write cleaned transcript to text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned_transcript)

    print(f"Cleaned transcript saved to: {txt_path}")
    print("Chunks have been preserved for investigation.")
    print(f"Chunk files are located in: {chunks_folder}")

if __name__ == "__main__":
    main()
