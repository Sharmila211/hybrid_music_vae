import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Z = np.load("results/text_latents.npy")
labels = np.load("results/vae_clusters.npy")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z_2d = tsne.fit_transform(Z)

plt.figure(figsize=(6,5))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap="viridis", s=10)
plt.title("t-SNE of VAE Latent Space (Lyrics)")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("results/tsne_vae.png")
plt.show()
