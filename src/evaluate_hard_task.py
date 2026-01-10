import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load data
Z = np.load("results/beta_latents.npy")
true_labels = np.load("results/gtzan_genre_labels.npy")

# KMeans clustering
kmeans = KMeans(n_clusters=len(set(true_labels)), random_state=42)
pred_labels = kmeans.fit_predict(Z)

# Metrics
print("ARI:", adjusted_rand_score(true_labels, pred_labels))
print("NMI:", normalized_mutual_info_score(true_labels, pred_labels))

# Purity
def purity_score(y_true, y_pred):
    contingency = np.zeros((len(set(y_pred)), len(set(y_true))))
    for i in range(len(y_true)):
        contingency[y_pred[i], y_true[i]] += 1
    return np.sum(np.max(contingency, axis=1)) / np.sum(contingency)

print("Purity:", purity_score(true_labels, pred_labels))
