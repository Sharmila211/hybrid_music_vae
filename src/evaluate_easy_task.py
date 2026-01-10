import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

Z = np.load("results/text_latents.npy")
vae_labels = np.load("results/vae_clusters.npy")
pca_labels = np.load("results/pca_clusters.npy")

print("VAE + KMeans")
print("Silhouette:", silhouette_score(Z, vae_labels))
print("Calinski-Harabasz:", calinski_harabasz_score(Z, vae_labels))

print("\nPCA + KMeans")
print("Silhouette:", silhouette_score(Z, pca_labels))
print("Calinski-Harabasz:", calinski_harabasz_score(Z, pca_labels))
