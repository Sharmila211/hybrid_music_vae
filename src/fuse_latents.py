import numpy as np

Z_text = np.load("results/text_latents.npy")
Z_audio = np.load("results/audio_latents.npy")

# Match sample size
n = min(len(Z_text), len(Z_audio))

Z_fused = np.concatenate([Z_text[:n], Z_audio[:n]], axis=1)
np.save("results/fused_latents.npy", Z_fused)

print("Fused latent shape:", Z_fused.shape)
