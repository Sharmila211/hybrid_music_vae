import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Z = np.load("results/beta_latents.npy")
labels = np.load("results/gtzan_genre_labels.npy")

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
Z_2d = tsne.fit_transform(Z)

plt.figure(figsize=(6,5))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap="tab10", s=10)
plt.title("t-SNE of β-VAE Latent Space (GTZAN Audio)")
plt.colorbar(label="Genre ID")
plt.tight_layout()
plt.savefig("results/tsne_beta.png")
plt.show()
