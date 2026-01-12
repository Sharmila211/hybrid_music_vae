# Hybrid Music Representation Learning using Variational Autoencoders (VAE)

This project implements a hybrid music representation learning framework using Variational Autoencoders (VAE).  
The system learns latent representations from **lyrics (text)**, **audio features**, and their **fusion**, and evaluates clustering performance for music analysis tasks.

The project is divided into **Easy**, **Medium**, and **Hard** tasks as specified in the project guidelines. The codes are in `src` folder, the datasets will be on data folder and the results (.png, .npy, .pt) will be on results folder.

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

### Hard Task – Beta-VAE with Ground-Truth Evaluation
- Beta-VAE trained on GTZAN audio dataset  
- Ground-truth genre labels extracted from folder structure  
- Latent space clustering using K-Means  
- Quantitative evaluation using:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Purity score  
- t-SNE visualization of Beta-VAE latent space  

---
## Dataset Description

Datasets are NOT included in this repository due to size and licensing restrictions.

This repository only contains code, scripts, and instructions to reproduce the results.


## Audio Datasets Used
GTZAN Music Genre Dataset for Hard Task.

Description: 1000 audio tracks categorized into 10 music genres

Used for: Audio VAE and Beta-VAE training, ground-truth clustering evaluation

Link:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification


Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Bangla Audio Dataset (Medium Task)

Description: Bangla music audio features stored in CSV format

Used for: Audio VAE training and multimodal fusion

Content: Pre-extracted spectral and MFCC-based audio features

Note: Audio files were processed offline and features saved as dataset.csv

Location: data/raw/bangla_audio/dataset.csv

Lyrics Datasets Used for Easy & Medium Tasks.

# Bangla Lyrics Dataset

Format: CSV

Language: Bangla

Used for: Text VAE and hybrid fusion

Location:  data/raw/bangla_lyrics.csv

# English Lyrics Dataset

Format: CSV

Language: English

Used for: Text VAE and hybrid fusion

Location: data/raw/english_lyrics.csv


Lyrics datasets were collected from publicly available sources and Kaggle lyric datasets.


## Project Structure

hybrid_music_vae/
├── src/               ( # All source code )
├── data/               ( # Dataset directory (not included here) )
│   └── raw/
├── results/            ( # Generated outputs (ignored in Git) )
├── README.md
├── requirements.txt
└── .gitignore


## Key Results

| Task | Representation | Evaluation Metrics |
|----|----|----|
| Easy | Text VAE | Silhouette, Calinski–Harabasz |
| Medium | Audio + Text Fusion | Silhouette ≈ 0.26 |
| Hard | Beta-VAE Audio | ARI = 0.09, NMI = 0.17, Purity = 0.29 |

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

