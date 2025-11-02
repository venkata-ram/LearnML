"""
K-Means Clustering - Unsupervised Learning Example
===================================================

Clustering is used to find natural groupings in data without labels.
This example demonstrates K-Means clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample clustering data
print("=" * 60)
print("K-Means Clustering Example")
print("=" * 60)

np.random.seed(42)
X, true_labels = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42
)

print(f"\nGenerated {len(X)} samples")
print(f"Features: {X.shape[1]}")
print(f"True number of clusters: 3")

# Apply K-Means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(X)

print(f"\nK-Means with {n_clusters} clusters")
print(f"Cluster centers:")
print(kmeans.cluster_centers_)
print(f"\nInertia (sum of squared distances): {kmeans.inertia_:.2f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Original data with true labels
axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.6)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Original Data (True Labels)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Data with predicted clusters
axes[1].scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', alpha=0.6)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='X', s=300, c='red', edgecolors='black', linewidths=2,
                label='Centroids')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('K-Means Clustering Results')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/04_clustering/kmeans_clustering.png')
print("\nVisualization saved to 'kmeans_clustering.png'")

# Elbow method to find optimal number of clusters
print("\n" + "=" * 60)
print("Finding Optimal Number of Clusters (Elbow Method)")
print("=" * 60)

inertias = []
K_range = range(1, 8)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.savefig('examples/04_clustering/elbow_method.png')
print("\nElbow method visualization saved to 'elbow_method.png'")
