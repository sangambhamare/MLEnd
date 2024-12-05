import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib

# Load the pre-trained model and label encoder
model = joblib.load('best_model.joblib')  # Load your pre-trained model
label_encoder = joblib.load('label_encoder.pkl')  # Load the label encoder

# Define the feature names in the order used during training
feature_columns = ['Power', 'Pitch Mean', 'Pitch Variance', 'Voiced Fraction']

# Feature extraction function
def extract_features(file):
    # Load the audio file with librosa (ensure 30-second duration)
    y, sr = librosa.load(file, sr=None, duration=30)  # Duration fixed to 30 seconds
    
    # Extract pitch and magnitude using librosa's piptrack
    pitch, mag = librosa.core.piptrack(y=y, sr=sr)
    
    # Calculate pitch mean and variance
    pitch_mean = np.mean(pitch)
    pitch_variance = np.var(pitch)
    
    # Calculate power (mean of magnitude)
    power = np.mean(mag)
    
    # Voiced fraction: the proportion of speech frames where voice is present
    voiced_frames = librosa.effects.split(y)  # Find voiced segments
    voiced_fraction = sum(np.diff(frames) for frames in voiced_frames) / len(y)
    
    # Create a DataFrame with the features in the same order as during training
    features = pd.DataFrame({
        'Power': [power],
        'Pitch Mean': [pitch_mean],
        'Pitch Variance': [pitch_variance],
        'Voiced Fraction': [voiced_fraction]  # Added Voiced Fraction
    })

    # Ensure the columns are in the same order as the model expects
    features = features[feature_columns]  # Reorder the columns if necessary
    
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
        # Display the uploaded audio
        st.audio(uploaded_file, format="audio/wav")
        
        # Extract features from the uploaded file
        features = extract_features(uploaded_file)

        # Display the extracted features
        st.write("Extracted Features:")
        st.write(features)

        # Predict using the model
        prediction = model.predict(features)

        # Decode the prediction (map 0 to 'False' and 1 to 'True')
        if prediction[0] == 1:
            label = "True"
        else:
            label = "False"

        # Display the result
        st.write(f"Prediction: The story is **{label}**.")

        # Disclaimer
        st.write("**Disclaimer**: Predictions can be wrong sometimes. The model is based on audio features and may not always be accurate.")

    # Copyright notice
    st.markdown("""
    ---
    **All Rights Reserved to Sangam S Bhamare 2024**
    """)

# Run the app
if __name__ == "__main__":
    app()
