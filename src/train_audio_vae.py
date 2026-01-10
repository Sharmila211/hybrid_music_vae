import torch
import torch.nn.functional as F
from audio_data_loader import load_audio_features
from audio_vae import AudioVAE

X = load_audio_features("data/raw/bangla_audio/dataset.csv")
X = torch.tensor(X, dtype=torch.float32)

model = AudioVAE(input_dim=X.shape[1], latent_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    recon, mu, logvar = model(X)
    loss = vae_loss(recon, X, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.2f}")

torch.save(model.state_dict(), "results/audio_vae.pt")
print("Audio VAE trained and saved.")
