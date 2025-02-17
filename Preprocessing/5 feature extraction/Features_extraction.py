import librosa
from librosa import feature
import numpy as np
import os

from glob import glob
import pandas as pd

# Define the genre directory where your audio files are stored
genre_directory = "D:\\R021\\project\\Songs"


# List all directories within the genre directory
genre_folders = os.listdir(genre_directory)

# Define feature extraction functions
fn_list_i = [
    librosa.onset.onset_strength,
    feature.chroma_stft,
    feature.chroma_cqt,
    feature.chroma_cens,
    feature.melspectrogram,
    feature.mfcc,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_contrast,
    feature.spectral_rolloff,
    feature.tonnetz
]

fn_list_ii = [
    feature.zero_crossing_rate
]

# Define columns for the DataFrame
columns = ["Song_Name", "onset_strength", "chroma_stft", "chroma_cqt", "chroma_cens", "melspectrogram", "mfcc",
           "spectral_centroid", "spectral_bandwidth", "spectral_contrast", "spectral_rolloff", "tonnetz", "zero_crossing_rate"]

# Iterate through each genre folder
for genre in genre_folders:
    print(f"Processing genre: {genre}")

    # Construct the directory path for the current genre
    genre_dir = os.path.join(genre_directory, genre)

    # Get a list of audio files in the genre directory
    audio_files = glob(os.path.join(genre_dir, '*.opus'))

    # print(f'Number of audio files in {genre}: {len(audio_files)}')

    # Initialize list to store features of all audio files
    song_features = []

    # Iterate through each audio file in the genre folder
    for file in audio_files:
        # Load audio file
        y, sr = librosa.load(file, sr=None)
        
        # Extract feature vector
        feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
        feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
        feature_vector = feat_vect_i + feat_vect_ii

        # Append file name along with feature vector to song_features list
        song_features.append([file] + feature_vector)

    # Create a DataFrame from song_features list
    df = pd.DataFrame(song_features, columns=columns)

    # Save the DataFrame as a CSV file
    file_name = genre + '_features.csv'
    df.to_csv(file_name, index=False)  # Set index=False to avoid saving row indices

    print(f"CSV file saved for {genre} as: {file_name}")

print("All genres processed.")
