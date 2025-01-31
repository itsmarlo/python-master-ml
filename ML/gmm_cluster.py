import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Create sample data (you can replace this with your own dataset)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=4)  # We are choosing 4 components here
gmm.fit(X_scaled)

# Predict the cluster labels for each data point
y_gmm = gmm.predict(X_scaled)

# Visualize the results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_gmm, cmap='viridis')
plt.title("GMM Clustering")
plt.show()
