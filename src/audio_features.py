import librosa
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def extract_mfcc(audio_dir, max_files=200):
    features = []
    filenames = []

    for i, file in enumerate(os.listdir(audio_dir)):
        if file.endswith(".mp3") or file.endswith(".wav"):
            path = os.path.join(audio_dir, file)

            y, sr = librosa.load(path, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)

            features.append(mfcc_mean)
            filenames.append(file)

            if i + 1 >= max_files:
                break

    X = np.array(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
 

    return X, filenames
