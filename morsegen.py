import os
import threading
import numpy as np
import wave
import array
import sounddevice as sd
from scipy.io.wavfile import read

# Morse code dictionary and other functions should be here
# Dictionary of morse code incluing special characters like swedish åäö /,. etc.
# .: short, -: long, 2 : space between letters, 3: space between words and sentences for example '.--' is 'a'
morsecodedict = {
    'a': '.-',
    'b': '-...',
    'c': '-.-.',
    'd': '-..',
    'e': '.',
    'f': '..-.',
    'g': '--.',
    'h': '....',
    'i': '..',
    'j': '.---',
    'k': '-.-',
    'l': '.-..',
    'm': '--',
    'n': '-.',
    'o': '---',
    'ö': '---.',
    'p': '.--.',
    'q': '--.-',
    'r': '.-.',
    's': '...',
    't': '-',
    'u': '..-',
    'ü': '..--',
    'v': '...-',
    'w': '.--',
    'x': '-..-',
    'y': '-.--',
    'z': '--..',

    'å': '.--.-',
    'ä': '.-.-',
    'æ': '.-.-',
    'é': '..-..',
    'ñ': '--.--',
    'ö': '---.',
    'ü': '..--',

    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    '0': '-----',

    '.': '.-.-.-',
    ',': '--..--',
    ':': '---...',
    ';': '-.-.-.',
    '?': '..--..',
    '-': '-....-',
    '/': '-..-.',
    "'": '.----.',
    '(': '-.--.',
    ')': '-.--.-',
    '=': '-...-',
    '+': '.-.-.',
    '"': '.-..-.',
    '@': '.--.-.',
    '!': '-.-.--',
}

# Function to generate sine wave samples for the specified duration
def generate_tone_samples(frequency, duration, volume, sample_rate):
    t = np.arange(0, duration, 1 / sample_rate)
    wave_samples = volume * np.sin(2 * np.pi * frequency * t)
    return wave_samples

# Function to generate silence
def generate_silence(duration, sample_rate):
    return np.zeros(int(duration * sample_rate), dtype=np.float32)

# Count morse units for a character
def count_morse_units(char):
    if char.lower() in morsecodedict:
        morse_sequence = morsecodedict[char.lower()]
        #print(f"morse_sequence: {morse_sequence} : {len(morse_sequence)}")
        #print(f"sum: {sum(1 if symbol == '.' else 3 for symbol in morse_sequence)}")
        return sum(1 if symbol == '.' else 3 for symbol in morse_sequence) + len(morse_sequence) - 1
    return 0

# Function to save audio data to a WAV file
def save_wav_file(file_path, audio_data, sample_rate):
    if os.path.exists(file_path):
        os.remove(file_path)
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(array.array('h', (audio_data * 32767).astype(np.int16)).tobytes())

# Count morse units in a string
def count_morse_units_in_string(s):
    total_units = 0

    for char in s:
        # Count units for the current character
        total_units += count_morse_units(char)

        # Add 7 units for space
        if char.isspace():
            total_units += 7

    if (s[len(s) - 1].isspace()):
        total_units += 3 * (len(s) - 2)  # Add 3 units for space between characters, except for the last character if it is a space
    else:
        total_units += 3 * (len(s) - 1) # Add 3 units for space between characters

    return total_units

# Calculate morse durations with the specified takt value
def calculate_morse_durations(swe_takt=50):
    # Calculate no of units in PARIS from dictionary when dot is 1, dash is 3, space between elements is 1, space between letters is 3 and space between words is 7
    # P = .--. 11
    # PAUS 3
    # A = .- 5
    # PAUS 3
    # R = .-. 7
    # PAUS 3
    # I = .. 3
    # PAUS 3
    # S = ... 5
    # PAUS 7
    # Total 11 + 3 + 5 + 3 + 7 + 3 + 3 + 3 + 5 + 7 = 50

    # Calculate no of units in PARIS from dictionary when dot is 1, dash is 3, space between elements is 1, space between letters is 3 and space between words is 7
    no_units_in_paris_inc_space_between_words = count_morse_units_in_string('PARIS ')

    # Ten PARIS is 500 units and must be sent in 60 seconds to get 10 WPM
    ten_units_in_paris_inc_space_between_words = no_units_in_paris_inc_space_between_words * 10

    # Calculate baud from ten PARIS (500 / 60 = 8.333) and modify for given swe_takt
    bd = (ten_units_in_paris_inc_space_between_words / 60.0) * (1 / 50 * swe_takt)

    # Calculate dot duration in seconds from bd (8.333) (1 / 8.333 = 0.12)
    dot_duration_in_seconds = 1.0 / bd

    # Calculate dash duration in seconds from bd (8.333) (3 / 8.333 = 0.36)
    dash_duration_in_seconds = 3.0 / bd

    # Calculate space between elements in seconds from bd (8.333) (1 / 8.333 = 0.12)
    space_between_elements_in_seconds = 1.0 / bd

    # Calculate space between characters in seconds from bd (8.333) (3 / 8.333 = 0.36)
    space_between_characters_in_seconds = 3.0 / bd

    # Calculate space between words in seconds from bd (8.333) (7 / 8.333 = 0.84)
    space_between_words_in_seconds = 7.0 / bd

    # Return calculated durations
    return dot_duration_in_seconds, dash_duration_in_seconds, space_between_elements_in_seconds, space_between_characters_in_seconds, space_between_words_in_seconds

# Generate audio data for the specified message with the specified parameters
def generate_audio_morse_code(message, swe_takt = 50, frequency=800, volume=0.5, sample_rate=44100, add_space_between_words=False, add_space_between_characters=False, add_space_between_elements=False):
    dot_duration, dash_duration, space_between_elements, space_between_characters, space_between_words = calculate_morse_durations(swe_takt)

    audio_data = []

    for word in message.split():
        for i, char in enumerate(word):
            # Add silence for inter-character gap if not the first character
            if i > 0:
                audio_data.append(generate_silence(space_between_characters, sample_rate))

            if char.lower() in morsecodedict:
                morse_sequence = morsecodedict[char.lower()]
                for j, symbol in enumerate(morse_sequence):
                    # Add silence for inter-element gap if not the first element
                    if j > 0:
                        audio_data.append(generate_silence(space_between_elements, sample_rate))

                    # Generate tone samples for dot or dash
                    if symbol == '.':
                        audio_data.append(generate_tone_samples(frequency, dot_duration, volume, sample_rate))
                    elif symbol == '-':
                        audio_data.append(generate_tone_samples(frequency, dash_duration, volume, sample_rate))

        # Add silence at the end of the generated audio data
        if add_space_between_words:
            audio_data.append(generate_silence(space_between_words, sample_rate))
        elif add_space_between_characters:
            audio_data.append(generate_silence(space_between_characters, sample_rate))  
        elif add_space_between_elements:
            audio_data.append(generate_silence(space_between_elements, sample_rate))    


    if not audio_data:
        return np.array([])

    return np.concatenate(audio_data)

# Function to add external noise to an audio signal
def add_external_noise(audio_signal, noise_file_path, noise_level=0.5):
    # Read the noise WAV file
    _, noise_data = read(noise_file_path)

    # Repeat the noise data to match or exceed the length of the audio signal
    repetitions = len(audio_signal) // len(noise_data) + 1
    repeated_noise = np.tile(noise_data, repetitions)
    
    # Trim the repeated noise data to the length of the audio signal
    repeated_noise = repeated_noise[:len(audio_signal)]

    # Normalize the repeated noise data to the desired level
    normalized_noise = noise_level * repeated_noise / np.max(np.abs(repeated_noise))

    # Combine the noise with the original audio signal
    audio_signal_with_external_noise = audio_signal + normalized_noise

    return audio_signal_with_external_noise

# Function to generate and save WAV file for a single Morse code character
def generate_and_save_wav(key, takt, tone, volume, sample_rate):
    try:
        audio_data = generate_audio_morse_code(key, swe_takt=takt, frequency=tone, volume=volume, sample_rate=sample_rate)
        audio_data = generate_audio_morse_code(key, swe_takt=takt, frequency=tone, volume=volume, sample_rate=sample_rate, add_space_between_elements=True)
        audio_data = add_external_noise(audio_data, noise_file_path='noise.wav', noise_level=0.2)
        wav_file_path = f"training_data/{key}_takt{takt}_tone{tone}.wav"
        save_wav_file(wav_file_path, audio_data, sample_rate)
        print(f"WAV file '{wav_file_path}' generated successfully")
    except Exception as e:
        print(f"Error generating WAV file for character '{key}': {e}")

# Modified function for generating WAV files using threading
def gen_dictonary_wav_files_threaded(volume=0.5, sample_rate=22050):
    threads = []
    for key, value in morsecodedict.items():
        for takt in range(20, 151, 5):
            for tone in range(550, 851, 5):
                thread = threading.Thread(target=generate_and_save_wav, args=(key, takt, tone, volume, sample_rate))
                threads.append(thread)
                thread.start()
    for thread in threads:
        thread.join()

# Function to delete WAV files in a given directory
def delete_wav_files(folder_path):
    for wav_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, wav_file)
        if file_path.endswith(".wav"):
            os.remove(file_path)

def generat_test_wav_files():
    generate_and_save_wav('a', 93, 801, 0.5, 22050)
    generate_and_save_wav('a', 91, 802, 0.5, 22050)

# Main function
def main():
    delete_wav_files("training_data")
    gen_dictonary_wav_files_threaded()

if __name__ == "__main__":
    generat_test_wav_files()
