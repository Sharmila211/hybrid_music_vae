import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

Z = np.load("results/fused_latents.npy")

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(Z)

print("Silhouette (Fused):", silhouette_score(Z, labels))
