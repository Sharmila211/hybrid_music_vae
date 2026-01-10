import numpy as np
from sklearn.cluster import KMeans

Z = np.load("results/text_latents.npy")

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(Z)

np.save("results/vae_clusters.npy", labels)

print("Clustering done. Cluster counts:", np.bincount(labels))
