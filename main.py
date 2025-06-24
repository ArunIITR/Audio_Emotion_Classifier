import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import streamlit.components.v1 as components
from classifier import predict_emotion

# Apply CSS
st.markdown("""
<style>
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
.uploadedFile {
    display: none;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎙️ Emotion Detection from Audio")
st.markdown("Upload an audio file, then click **Submit** to detect emotions.")

# File Upload Only
audio_file = st.file_uploader("📁 Choose an audio file", type=["wav", "mp3", "m4a"])

# Submit button
submit = st.button("🚀 Submit")

# Anchor for result scroll
st.markdown("<div id='prediction-result'></div>", unsafe_allow_html=True)

# Run prediction only if file uploaded and submitted
# In your main.py, replace the plotting section with:
if audio_file is not None and submit:
    with st.spinner("Analyzing emotion..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getbuffer())
            temp_path = tmp.name

        # Predict
        emotion_label, confidence_array = predict_emotion(temp_path)
