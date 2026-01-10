import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from beta_vae import BetaVAE

# Load GTZAN features
df = pd.read_csv(
    "data/raw/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv"
)

X = df.filter(regex="mfcc").values
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)

# Load trained Beta-VAE
model = BetaVAE(input_dim=X.shape[1], latent_dim=16, beta=4.0)
model.load_state_dict(torch.load("results/beta_vae.pt"))
model.eval()

# Extract latent means
with torch.no_grad():
    h = model.encoder(X)
    mu = model.mu(h)

Z = mu.numpy()
np.save("results/beta_latents.npy", Z)

print("Saved beta latents:", Z.shape)
