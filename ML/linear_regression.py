# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# Set plot style for better visuals
plt.style.use('ggplot')

# Load the Boston housing dataset
# fetch_openml is used as load_boston() is deprecated in newer sklearn versions
bostonData = datasets.fetch_openml(data_id=531, as_frame=True)

# Extract the feature (average number of rooms per house) and target (house prices)
Xb = bostonData.data['RM'].values.reshape(-1, 1)  # 'RM' is the column for the number of rooms
yb = bostonData.target.values.reshape(-1, 1)     # Target: house prices in $1000s

# Create a scatter plot to visualize the relationship between the number of rooms and house prices
plt.scatter(Xb, yb, color='blue', edgecolor='k')  # Scatter plot with blue points
plt.ylabel('Value of house / $1000')             # Label for the y-axis
plt.xlabel('Number of rooms')                    # Label for the x-axis
plt.title('Number of Rooms vs House Prices')     # Plot title
plt.show()                                       # Display the plot
