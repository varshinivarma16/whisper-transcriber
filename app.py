# 📦 Install required dependencies first:
# pip install streamlit openai-whisper ffmpeg-python torch

import streamlit as st
import whisper
import tempfile
import os

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# App UI
st.title("🎤 Upload Audio and Convert to Text")
st.markdown("Upload a .wav, .mp3 or .m4a audio file to generate transcription using OpenAI Whisper.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    st.audio(uploaded_file, format='audio/wav')
    st.info("Transcribing audio...")

    # Transcribe the audio
    try:
        result = model.transcribe(temp_file_path)
        st.subheader("📝 Transcription")
        st.write(result["text"])

        # Option to download
        st.download_button("📥 Download Transcription", data=result["text"], file_name="transcription.txt")
    except Exception as e:
        st.error(f"❌ Failed to transcribe: {e}")

    # Clean up
    os.remove(temp_file_path)