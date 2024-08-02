import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib for plotting
from sklearn.cluster import KMeans  # Import the KMeans class from sklearn for clustering
from sklearn.datasets import make_blobs  # Import the make_blobs function from sklearn for generating synthetic data
from sklearn.metrics import confusion_matrix  # Import the confusion_matrix function from sklearn for evaluating classification (not used in this code)

# Generate synthetic dataset with 2500 samples, 4 centers (clusters), and 2 features
# random_state ensures reproducibility
X, y = make_blobs(n_samples=2500, centers=4, n_features=2, random_state=10)

# Apply K-Means clustering with 4 clusters
# random_state ensures reproducibility
y_pred = KMeans(n_clusters=4, random_state=0).fit_predict(X)
cmat = confusion_matrix(y,y_pred)
# Create a new figure for the first plot
plt.figure()

# Scatter plot of the data points with true cluster labels
# X[:, 0] and X[:, 1] are the coordinates of the data points
# c=y uses the true cluster labels to color the points
# cmap='jet' specifies the colormap
# s=10 sets the size of the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=10)

# Set the title of the plot
plt.suptitle('K-Means Clusters')

# Add grid lines to the plot
plt.grid(1, which='both')

# Adjust the axis limits to be tight around the data points
plt.axis('tight')

# Display the plot
plt.show()

# Create a new figure for the second plot
plt.figure()

# Scatter plot of the data points with predicted cluster labels
# c=y_pred uses the predicted cluster labels to color the points
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='jet', s=10)

# Set the title of the plot
plt.suptitle('K-Means Clusters')

# Add grid lines to the plot
plt.grid(1, which='both')

# Adjust the axis limits to be tight around the data points
plt.axis('tight')

# Display the plot
plt.show()
