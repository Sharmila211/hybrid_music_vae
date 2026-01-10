from audio_features import extract_mfcc

X, files = extract_mfcc(
    "data/raw/gtzan-dataset-music-genre-classification/Data/genres_original/blues",
    max_files=10
)

print("MFCC shape:", X.shape)
print("Mean (should be ~0):", X.mean(axis=0)[:5])
print("Std (should be ~1):", X.std(axis=0)[:5])
