import numpy as np
import librosa
from keras.models import load_model

# Load model (ensure Emotion_Audio.keras is trained with 162 features)
emotion_classifier = load_model("Emotion_Audio.keras", compile=False)

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def predict_emotion(file_path):
    data, sample_rate = librosa.load(file_path, sr=None)
    features = extract_features(data, sample_rate)

    features = np.resize(features,(162,1))
    features = np.reshape(features, (1, 162, 1)) 

    prediction = emotion_classifier(features)
    print(prediction)

    predicted_label_index = np.argmax(prediction)
    
    # Update this list to reflect actual label order from training
    labels = ["Angry", "Calm", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    
    return labels[predicted_label_index], prediction[0]
