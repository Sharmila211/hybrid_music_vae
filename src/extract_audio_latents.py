import torch
import numpy as np
from audio_data_loader import load_audio_features
from audio_vae import AudioVAE

# Load audio features
X = load_audio_features("data/raw/bangla_audio/dataset.csv")
X = torch.tensor(X, dtype=torch.float32)

# Load trained audio VAE
model = AudioVAE(input_dim=X.shape[1], latent_dim=16)
model.load_state_dict(torch.load("results/audio_vae.pt"))
model.eval()

# Extract latent means
with torch.no_grad():
    h = model.encoder(X)
    mu = model.mu(h)

Z_audio = mu.numpy()

# Save
np.save("results/audio_latents.npy", Z_audio)

print("Audio latent shape:", Z_audio.shape)
print("Saved to results/audio_latents.npy")
