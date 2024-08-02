import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

# Load the iris dataset
irisset = datasets.load_iris()
X = irisset.data[:100, :2]  # Use the first 100 samples and the first two features
z = irisset.target[:100]    # Use the first 100 labels

# Create and train the SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X, z)

# Get the coefficients of the decision boundary
w = clf.coef_[0]

# Create points for plotting the decision boundary
xpoints = np.linspace(4, 7)
ypoints = -w[0] / w[1] * xpoints - clf.intercept_[0] / w[1]

# Plot the decision boundary and the data points
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X[:, 0], X[:, 1], c=z, cmap=plt.cm.bwr)
plt.suptitle('SVM IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1, which='both')
plt.tight_layout()
plt.show()
