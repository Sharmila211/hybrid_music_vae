import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from text_data_loader import load_lyrics
from text_features import extract_tfidf

texts, _ = load_lyrics()
X = extract_tfidf(texts).toarray()

pca = PCA(n_components=32, random_state=42)
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_pca)

np.save("results/pca_clusters.npy", labels)
print("Baseline PCA + KMeans done.")
