# 🎤 Whisper Transcriber App

A simple Streamlit application that uses OpenAI's Whisper model to transcribe audio files into text. Upload your `.mp3`, `.wav`, or `.m4a` audio files and get high-quality transcriptions instantly!

## 🚀 Demo

✅ Live app: [Whisper Transcriber on Streamlit](https://audio-text-converter-varshini.streamlit.app/)

## 📦 Features

- Upload audio files (`.mp3`, `.wav`, `.m4a`)
- View and download the transcribed text
- Uses OpenAI's powerful `whisper` model
- Lightweight and simple Streamlit interface
- Automatically deletes uploaded audio after processing

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/) – for the frontend UI
- [OpenAI Whisper](https://github.com/openai/whisper) – for speech-to-text transcription
- [PyTorch](https://pytorch.org/) – required for Whisper
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) – for audio processing

## 📁 Project Structure

whisper-transcriber/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation

bash
Copy
Edit

## 📥 Installation (Local Setup)

1. **Clone the repository**
```bash
git clone https://github.com/varshinivarma16/whisper-transcriber.git
cd whisper-transcriber
Create and activate a virtual environment

bash
Copy
Edit
python -m venv whisperenv
whisperenv\Scripts\activate        # On Windows
# source whisperenv/bin/activate  # On macOS/Linux
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Install FFmpeg (required)

Download FFmpeg

Extract the zip

Add the bin/ folder to your system's environment PATH

Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
🌐 Deployment on Streamlit Cloud
Push your code to GitHub:
https://github.com/varshinivarma16/whisper-transcriber

Visit Streamlit Cloud, connect your GitHub repo, and deploy the app
