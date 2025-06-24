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

        # Show prediction
        st.markdown(f"### 🎯 Predicted Emotion: **{emotion_label.title()}**")

        # Plot confidence chart for all 8 emotions
        st.markdown("### Confidence per Emotion")
        fig, ax = plt.subplots(figsize=(10, 6))
        emotions = ["Angry", "Calm", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        colors = ["#FF6384", "#36A2EB", "#FF9F40", "#FFCD56", "#4BC0C0", "#9966FF", "#FF6384", "#C9CBCF"]
        
        bars = ax.bar(
            emotions,
            [val * 100 for val in confidence_array],
            color=colors
        )
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidence_array):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{conf*100:.1f}%', ha='center', va='bottom')
        
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Emotion Detection Confidence Scores")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Smooth scroll to result
        # components.html("""
        # <script>
        #     const anchor = window.parent.document.getElementById("prediction-result");
        #     if (anchor) {
        #         anchor.scrollIntoView({behavior: 'smooth'});
        #     }
        # </script>
        # """, height=0)
