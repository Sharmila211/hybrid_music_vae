import os
import numpy as np

base_dir = "data/raw/gtzan-dataset-music-genre-classification/Data/genres_original"

genres = sorted(os.listdir(base_dir))
genre_to_id = {g: i for i, g in enumerate(genres)}

labels = []

for genre in genres:
    genre_path = os.path.join(base_dir, genre)
    files = os.listdir(genre_path)
    labels.extend([genre_to_id[genre]] * len(files))

labels = np.array(labels)

np.save("results/gtzan_genre_labels.npy", labels)

print("Saved GTZAN labels:", labels.shape)
print("Genre mapping:", genre_to_id)
