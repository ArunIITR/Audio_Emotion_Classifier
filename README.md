# 🎧 Audio Emotion Classifier

**Live Demo:** [https://audio-emotion-classifier-1.onrender.com](https://audio-emotion-classifier-1.onrender.com)

---

## 🌟 Overview

The **Audio Emotion Classifier** is an interactive web app that predicts emotions from speech audio files. Simply upload an audio file, and the app will analyze the sound to determine the most likely emotion being expressed, along with confidence scores for all possible emotions.

---

## ⚙️ How It Works

### 1. 🎵 Feature Extraction

The backend (`classifier.py`) uses the powerful [Librosa](https://librosa.org/) library to extract key audio features:

* **Zero Crossing Rate (ZCR):** Measures how frequently the audio signal changes sign.
* **Chroma STFT:** Captures the energy across different pitch classes.
* **MFCC (Mel-Frequency Cepstral Coefficients):** Represents the timbral texture of the audio.
* **Root Mean Square (RMS):** Measures signal energy.
* **Mel Spectrogram:** Visualizes the frequency spectrum over time.

All these features are combined into a single feature vector of length **162** for each audio file.

### 2. 🤖 Emotion Prediction

* The extracted features are reshaped and passed to a pre-trained deep learning model: `Emotion_Audio.keras`.

* The model outputs probabilities for the following emotions:

  `Angry` | `Calm` | `Disgust` | `Fear` | `Happy` | `Neutral` | `Sad` | `Surprise`

* The emotion with the **highest probability** is displayed as the prediction, along with a full confidence score breakdown for all emotions.

### 3. 🖥️ Web Interface

The frontend (`main.py`) is built using [Streamlit](https://streamlit.io/) for a smooth and interactive user experience:

* Upload supported audio files: `.wav`, `.mp3`, `.m4a`
* Click **Submit** to run the analysis
* View:

  * The predicted emotion in bold
  * A bar chart showing confidence scores for each emotion, with percentage labels

---

## 📁 File Structure

```
├── classifier.py         # Handles feature extraction and prediction
├── main.py                # Streamlit web interface
├── Emotion_Audio.keras    # Pre-trained emotion classification model
├── requirements.txt       # Project dependencies
```

---

## 🚀 Usage

### 1. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the App Locally

```bash
streamlit run main.py
```

### 3. Open the App

Visit `http://localhost:8501` in your browser.

### 4. Upload & Predict

* Upload your audio file (e.g., someone saying *"I'm so happy today!"*)
* Click **Submit**
* The app predicts *"Happy"* and shows a confidence bar chart 🎯

---

## 🛠 Example

1. Upload a speech audio file.
2. Click **Submit**.
3. Get the predicted emotion and visual confidence scores.

---

## 🙏 Acknowledgements

* Built with ❤️ using:

  * [Streamlit](https://streamlit.io/)
  * [Librosa](https://librosa.org/)
  * [TensorFlow / Keras](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)

---

## 📬 Contributor

Arun Kumar (IIT ROORKEE)
