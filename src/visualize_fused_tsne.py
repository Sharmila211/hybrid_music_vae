import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

Z = np.load("results/fused_latents.npy")

tsne = TSNE(n_components=2, random_state=42)
Z_2d = tsne.fit_transform(Z)

plt.figure(figsize=(6,5))
plt.scatter(Z_2d[:,0], Z_2d[:,1], s=10)
plt.title("t-SNE of Fused Audio + Text Latent Space")
plt.tight_layout()
plt.savefig("results/tsne_fused.png")
plt.show()
