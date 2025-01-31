import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize  # Correct import for normalize function
import scipy.cluster.hierarchy as shc  # Importing hierarchical clustering module
from typing import Literal  # No need for 'typing_extensions'

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('Wholesale customers data.csv')

# Display the first few rows of the dataset to check its structure
print(data.head())

# Normalizing the dataset so that all features are on the same scale
# This helps with algorithms like clustering that rely on distance calculations
# We use the 'normalize' function from sklearn.preprocessing
data_scaled = normalize(data.drop(['Channel', 'Region'], axis=1))  # Dropping 'Channel' and 'Region' columns
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[2:])  # Reassign column names to the scaled data

# Display the first few rows of the scaled data to verify that normalization worked
print(data_scaled.head())

# Visualizing the dendrogram to find the optimal number of clusters for hierarchical clustering
plt.figure(figsize=(10, 7))  # Set the size of the plot
plt.title("Dendrogram")  # Set the title of the plot
# We perform hierarchical/agglomerative clustering using the 'ward' linkage method
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()  # Display the dendrogram

