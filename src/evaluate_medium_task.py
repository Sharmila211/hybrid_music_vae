import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

Z = np.load("results/fused_latents.npy")

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(Z)

score = silhouette_score(Z, labels)
print("Medium Task – Silhouette (Fused):", score)
