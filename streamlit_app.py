import joblib
import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

# Load the trained model and label encoder
model = joblib.load('best_model.joblib')
label_encoder = joblib.load('label_encoder.pkl')

# Feature extraction function
def extract_features(file):
    y, sr = librosa.load(file, sr=None, duration=30)  # Ensure 30 seconds duration
    pitch, mag = librosa.core.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch)
    pitch_variance = np.var(pitch)
    power = np.mean(mag)

    features = pd.DataFrame({
        'Pitch Mean': [pitch_mean],
        'Pitch Variance': [pitch_variance],
        'Power': [power]
    })
    return features

# Streamlit App
def app():
    st.title("Deception Detection Model")

    st.write("""
    Upload an audio recording of a story, and the model will predict if the story is True or False based on the extracted features.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Extract features from the uploaded file
        features = extract_features(uploaded_file)

        # Predict using the model
        prediction = model.predict(features)

        # Decode the prediction
        label = label_encoder.inverse_transform(prediction)

        # Display the result
        st.write(f"Prediction: The story is **{label[0]}**.")

# Run the app
if __name__ == "__main__":
    app()
