import os
import librosa
import numpy as np
import joblib

# Function to extract features from an audio file
def extract_features(file_path, n_fft=2048):
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Calculate the next power of 2 greater than or equal to the length of the audio
    if len(audio) < n_fft:
        next_pow2 = 2 ** np.ceil(np.log2(n_fft))
        audio = np.pad(audio, (0, int(next_pow2) - len(audio)), mode='constant')

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, fmin=300, fmax=5000, n_fft=n_fft)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

if __name__ == "__main__":
    # Load the trained models, scaler and max_length
    classifier_chars = joblib.load('trained_morse_classifier_chars.joblib')
    classifier_takts = joblib.load('trained_morse_classifier_takts.joblib')
    classifier_tones = joblib.load('trained_morse_classifier_tones.joblib')
    scaler = joblib.load('trained_morse_scaler.joblib')
    max_length = joblib.load('trained_morse_max_length.joblib')
    
    # Process the test file
    test_feature_vector = extract_features('training_data/c_takt90_tone800.wav')
    test_feature_vector_padded = np.pad(test_feature_vector, (0, max_length - len(test_feature_vector)), 'constant')
    test_feature_vector_scaled = scaler.transform([test_feature_vector_padded])

    # Make predictions
    predicted_char = classifier_chars.predict(test_feature_vector_scaled)[0]
    predicted_takt = classifier_takts.predict(test_feature_vector_scaled)[0]
    predicted_tone = classifier_tones.predict(test_feature_vector_scaled)[0]

    # Output the predictions
    print(f"Predicted Morse code character: {predicted_char}")
    print(f"Predicted Takt (WPM): {predicted_takt}")
    print(f"Predicted Tone Frequency: {predicted_tone}")
