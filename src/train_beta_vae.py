import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from beta_vae import BetaVAE

# Load GTZAN audio features (30-sec MFCCs)
df = pd.read_csv(
    "data/raw/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv"
)

# Keep only MFCC features
X = df.filter(regex="mfcc").values
X = StandardScaler().fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)

# Beta-VAE
model = BetaVAE(input_dim=X.shape[1], latent_dim=16, beta=4.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def beta_vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    recon, mu, logvar = model(X)
    loss = beta_vae_loss(recon, X, mu, logvar, model.beta)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.2f}")

# Save model
torch.save(model.state_dict(), "results/beta_vae.pt")
print("Beta-VAE trained on GTZAN audio and saved.")
