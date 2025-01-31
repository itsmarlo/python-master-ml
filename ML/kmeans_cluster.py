import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Create sample data (you can replace this with your own dataset)
# Here we're creating a synthetic dataset with 2 features (you can ignore this part if using your own data)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Standardize the features to ensure all features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4)  # We are choosing 4 clusters here
kmeans.fit(X_scaled)

# Predict the cluster labels for each data point
y_kmeans = kmeans.predict(X_scaled)

# Visualize the results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering")
plt.show()
