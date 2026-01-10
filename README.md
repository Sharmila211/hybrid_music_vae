# Hybrid Music Representation Learning using Variational Autoencoders (VAE)

This project implements a hybrid music representation learning framework using Variational Autoencoders (VAE).  
The system learns latent representations from **lyrics (text)**, **audio features**, and their **fusion**, and evaluates clustering performance for music analysis tasks.

The project is divided into **Easy**, **Medium**, and **Hard** tasks as specified in the course project guidelines.

---

## Task Breakdown

### Easy Task – Text VAE
- TF-IDF feature extraction from Bangla and English lyrics  
- Text-based Variational Autoencoder  
- K-Means clustering on latent space  
- PCA + K-Means baseline comparison  
- Evaluation using Silhouette Score and Calinski–Harabasz Index  
- t-SNE visualization of text latent space  

---

### Medium Task – Audio + Text Fusion
- MFCC feature extraction from audio  
- Audio VAE training  
- Fusion of audio and text latent representations  
- Clustering on fused latent space  
- t-SNE visualization of fused representations  
- Comparison with text-only baseline  

---

### Hard Task – β-VAE with Ground-Truth Evaluation
- β-VAE trained on GTZAN audio dataset  
- Ground-truth genre labels extracted from folder structure  
- Latent space clustering using K-Means  
- Quantitative evaluation using:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Purity score  
- t-SNE visualization of β-VAE latent space  

---

## Key Results

| Task | Representation | Evaluation Metrics |
|----|----|----|
| Easy | Text VAE | Silhouette, Calinski–Harabasz |
| Medium | Audio + Text Fusion | Silhouette ≈ 0.26 |
| Hard | β-VAE Audio | ARI ≈ 0.09, NMI ≈ 0.17, Purity ≈ 0.29 |

---

## How to Run

### Activate Environment

venv\Scripts\activate

## Install Dependencies

py -m pip install -r requirements.txt


# Train Models

python src/train_text_vae.py
python src/train_audio_vae.py
python src/train_beta_vae.py

# Extract Latent Representations

python src/extract_text_latents.py
python src/extract_audio_latents.py
python src/extract_beta_latents.py

## Evaluation

# Easy Task

python src/evaluate_easy_task.py

# Medium Task

python src/fuse_latents.py
python src/cluster_fused_latents.py
python src/visualize_fused_tsne.py

# Hard Task

python src/evaluate_hard_task.py

