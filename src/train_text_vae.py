import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from text_data_loader import load_lyrics
from text_features import extract_tfidf
from text_vae import TextVAE

#  Load text data
texts, labels = load_lyrics()
X = extract_tfidf(texts)

# Convert sparse matrix to tensor
X = torch.tensor(X.toarray(), dtype=torch.float32)

dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

#  Create VAE model
model = TextVAE(input_dim=X.shape[1], latent_dim=32)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#  Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

#  Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for (x_batch,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(x_batch)
        loss = vae_loss(recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f}")

#  Save model
torch.save(model.state_dict(), "results/text_vae.pt")
print("Text VAE training finished and model saved.")
