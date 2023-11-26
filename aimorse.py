import os
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool

# Function to estimate WPM based on a known Morse code sequence
def estimate_wpm(morse_code_sequence, duration):
    wpm = len(morse_code_sequence) / duration
    return wpm

# Function to extract features from an audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, fmin=20, fmax=8000)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Function to get file data from a folder
def get_file_data(folder_path):
    file_names = os.listdir(folder_path)
    data = []
    for file_name in file_names:
        if file_name.startswith('a') or file_name.startswith('b') or file_name.startswith('c'):
            file_path = os.path.join(folder_path, file_name)
            label_char = file_name[0]
            label_takt = int(file_name.split('_')[1][4:])
            label_tone = int(file_name.split('_')[2].split('.')[0][4:])
            data.append({'char': label_char, 'takt': label_takt, 'tone': label_tone, 'filename': file_path})
    return data

# Function to process each file (for multiprocessing)
def process_file(file_info):
    feature_vector = extract_features(file_info['filename'])
    return feature_vector, file_info['char']

# Main script
if __name__ == "__main__":
    folder_path = 'training_data/'
    audio_files = get_file_data(folder_path)

    # Use multiprocessing to process files in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_file, audio_files)

    # Unpack results
    features, labels = zip(*results)

    # Find the maximum length of feature vectors
    max_length = max(len(f) for f in features)

    # Padding feature vectors to the same length
    padded_features = [np.pad(f, (0, max_length - len(f)), 'constant') for f in features]

    X_train = np.array(padded_features)
    y_train = np.array(labels)

    # Training the Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=300, random_state=42)
    classifier.fit(X_train, y_train)

    # Save the model to a file (optional)
    model_filename = 'trained_morse_classifier.joblib'
    joblib.dump(classifier, model_filename)

    # Predicting on a test file (example)
    test_feature_vector = extract_features('morse_code_output.wav')
    test_feature_vector_padded = np.pad(test_feature_vector, (0, max_length - len(test_feature_vector)), 'constant')
    predicted_label = classifier.predict([test_feature_vector_padded])[0]

    print(f"Predicted Morse code for the test audio file: {' '.join(predicted_label)}")
