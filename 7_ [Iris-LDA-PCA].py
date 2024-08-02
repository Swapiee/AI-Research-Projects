from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the iris dataset
irisset = datasets.load_iris()
X = irisset.data  # Features
Y = irisset.target  # Labels

# LDA for classification
lda = LinearDiscriminantAnalysis(n_components=2)  # Initialize LDA with 2 components
Ypred = lda.fit(X, Y).predict(X)  # Fit LDA model and predict labels

# Evaluate the LDA model
cmat = confusion_matrix(Y, Ypred)  # Confusion matrix
ldascore = accuracy_score(Y, Ypred)  # Accuracy score

# Print the evaluation results
print('Confusion matrix =\n', cmat)
print('Accuracy score of LDA is', 100 * ldascore, '%\n')

# LDA for dimensionality reduction and visualization
Xl = lda.transform(X)  # Transform the data into LDA space
plt.figure(1)  # Create a new figure
plt.scatter(Xl[:, 0], Xl[:, 1], c=Y)  # Scatter plot of LDA-transformed data
plt.suptitle('LDA IRIS Data')  # Title of the plot
plt.xlabel('LDA 1')  # Label for x-axis
plt.ylabel('LDA 2')  # Label for y-axis
plt.grid(True, which='both')  # Show grid lines
plt.axis('tight')  # Fit axes tightly around the data
plt.show()  # Display the plot

# PCA for dimensionality reduction and visualization
pca = PCA(n_components=2)  # Initialize PCA with 2 components
Xp = pca.fit(X).transform(X)  # Fit PCA and transform the data
plt.figure(2)  # Create a new figure
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y)  # Scatter plot of PCA-transformed data
plt.suptitle('PCA IRIS Data')  # Title of the plot
plt.xlabel('PCA 1')  # Label for x-axis
plt.ylabel('PCA 2')  # Label for y-axis
plt.grid(True, which='both')  # Show grid lines
plt.axis('tight')  # Fit axes tightly around the data
plt.show()  # Display the plot
