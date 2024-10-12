# YouTube Video Transcription Script

## Overview
This script is used to download YouTube videos, split them into manageable chunks, and generate transcripts using either Gemini AI or Whisper models. It automates the transcription process, providing a consolidated and cleaned transcript as the final output.

## Features
- Downloads YouTube videos via URL.
- Splits videos into chunks to manage longer durations.
- Uses Gemini AI for transcription with fallback to Whisper when necessary.
- Provides scene descriptions along with transcription.
- Saves results to both JSON and text files.

## Prerequisites
- Python 3.x
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Whisper](https://github.com/openai/whisper)
- [Google Generative AI](https://github.com/google/generative-ai-python)
- [ffmpeg](https://ffmpeg.org/) (must be installed and available in your PATH)
- [dotenv](https://pypi.org/project/python-dotenv/) for loading environment variables

## Setup
1. Clone the repository or copy the script to your local machine.
2. Install the required dependencies:
   ```sh
   pip install yt-dlp whisper google-generativeai python-dotenv
   ```
3. Ensure you have `ffmpeg` installed and in your system PATH.
4. Create a `.env` file in the same directory as the script and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage
1. Update the script with your desired YouTube video URL:
   ```python
   YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=vYyUb_MI7to"
   ```
2. Configure other parameters such as:
   - `VIDEO_NAME`: Name for the video file and output directory.
   - `WHISPER_MODEL`: Whisper model to use for fallback transcription.
   - `MIN_CHUNK_DURATION`: Minimum duration of video chunks.
   - `TARGET_CHUNKS`: Target number of chunks to split the video into.
   - `TIMESTAMP_MODE`: Controls how timestamps are handled ("chunk", "model", "none").

3. Run the script:
   ```sh
   python your_script_name.py
   ```

## Output
- **Video File**: The YouTube video will be downloaded and saved locally.
- **Chunks**: The video will be split into chunks and saved in the `chunks` folder.
- **JSON File**: Contains transcriptions for each chunk.
- **Transcript File**: A cleaned and consolidated transcript will be saved as a text file in the `transcripts` folder.

## Error Handling
- If Gemini API fails to process the video, the script will fall back to using Whisper for transcription.
- The script will attempt to download the video and retry operations if an error occurs.

## License
This script is open-source and available for use under the MIT License.

## Troubleshooting
- Ensure that `ffmpeg` is installed and accessible from the command line.
- Make sure the `.env` file is properly set up with your API key.
- If the YouTube download fails, check the video URL and ensure it is accessible.

## Contributions
Feel free to contribute by creating a pull request or opening an issue on GitHub to enhance the functionality of this script.

