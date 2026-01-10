import torch
from text_data_loader import load_lyrics
from text_features import extract_tfidf
from text_vae import TextVAE

# Load data
texts, labels = load_lyrics()
X = extract_tfidf(texts)
X = torch.tensor(X.toarray(), dtype=torch.float32)

# Load trained model
model = TextVAE(input_dim=X.shape[1], latent_dim=32)
model.load_state_dict(torch.load("results/text_vae.pt"))
model.eval()

# Extract latent vectors (mean)
with torch.no_grad():
    mu, _ = model.encode(X)

Z = mu.numpy()

print("Latent shape:", Z.shape)

# Save
import numpy as np
np.save("results/text_latents.npy", Z)
