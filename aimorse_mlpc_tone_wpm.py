import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool

def extract_features2n(file_path, n_fft=2048):
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Calculate the next power of 2 greater than or equal to the length of the audio
    if len(audio) < n_fft:
        next_pow2 = 2 ** np.ceil(np.log2(n_fft))
        audio = np.pad(audio, (0, int(next_pow2) - len(audio)), mode='constant')

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, fmin=300, fmax=5000, n_fft=n_fft)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Function to get file data from a folder
def get_file_data(folder_path):
    file_names = os.listdir(folder_path)
    data = []
    for file_name in file_names:
        if file_name.startswith('a') or file_name.startswith('b') or file_name.startswith('c'):
            print(f"Processing file: {file_name}")
            file_path = os.path.join(folder_path, file_name)
            parts = file_name.split('_')
            label_char = parts[0]
            label_takt = int(parts[1][4:])
            label_tone = int(parts[2].split('.')[0][4:])
            data.append({'char': label_char, 'takt': label_takt, 'tone': label_tone, 'filename': file_path})
    return data

# Function to process each file (for multiprocessing)
def process_file(file_info):
    feature_vector = extract_features2n(file_info['filename'])
    return feature_vector, file_info['char'], file_info['takt'], file_info['tone']

# Main script
if __name__ == "__main__":
    folder_path = 'training_data/'
    audio_files = get_file_data(folder_path)

    # Use multiprocessing to process files in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_file, audio_files)

    # Unpack results
    features, chars, takts, tones = zip(*results)

    # Padding feature vectors to the same length
    max_length = max(len(f) for f in features)
    padded_features = [np.pad(f, (0, max_length - len(f)), 'constant') for f in features]

    joblib.dump(max_length, 'trained_morse_max_length.joblib')

    X_train = np.array(padded_features)
    y_train_chars = np.array(chars)
    y_train_takts = np.array(takts)
    y_train_tones = np.array(tones)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Save the scaler
    joblib.dump(scaler, 'trained_morse_scaler.joblib')

    # Train separate models for each attribute
    # Model for characters
    classifier_chars = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1024, random_state=42)
    classifier_chars.fit(X_train_scaled, y_train_chars)

    # Model for takts
    classifier_takts = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1024, random_state=42)
    classifier_takts.fit(X_train_scaled, y_train_takts)

    # Model for tones
    classifier_tones = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1024, random_state=42)
    classifier_tones.fit(X_train_scaled, y_train_tones)

    # Save the models to files
    joblib.dump(classifier_chars, 'trained_morse_classifier_chars.joblib')
    joblib.dump(classifier_takts, 'trained_morse_classifier_takts.joblib')
    joblib.dump(classifier_tones, 'trained_morse_classifier_tones.joblib')

    # Save the training data
    joblib.dump(X_train_scaled, 'X_train_scaled.joblib')
    joblib.dump(y_train_chars, 'y_train_chars.joblib')
    joblib.dump(y_train_takts, 'y_train_takts.joblib')
    joblib.dump(y_train_tones, 'y_train_tones.joblib')

    # Example of predicting on a test file
    test_feature_vector = extract_features2n('training_data/a_takt93_tone801.wav')
    test_feature_vector_padded = np.pad(test_feature_vector, (0, max_length - len(test_feature_vector)), 'constant')
    test_feature_vector_scaled = scaler.transform([test_feature_vector_padded])

    predicted_char = classifier_chars.predict(test_feature_vector_scaled)[0]
    predicted_takt = classifier_takts.predict(test_feature_vector_scaled)[0]
    predicted_tone = classifier_tones.predict(test_feature_vector_scaled)[0]

    print(f"Predicted Morse code character: {predicted_char}")
    print(f"Predicted Takt (WPM): {predicted_takt}")
    print(f"Predicted Tone Frequency: {predicted_tone}")
