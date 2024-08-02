import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Sigmoid function to calculate probabilities
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Load the dataset
purchaseData = pd.read_csv('Purchase_Logistic.csv')

# Extract features (columns 2 and 3) and target (column 4)
X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets (75% train, 25% test)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# Initialize and train the logistic regression model
logr = LogisticRegression(random_state=0)
logr.fit(Xtrain, Ytrain)

# Predict the target values for the test set
Ypred = logr.predict(Xtest)

# Compute the confusion matrix to evaluate the model
cmat = confusion_matrix(Ytest, Ypred)

# Plot the original data points, color-coded by target value
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(True, which='both')
plt.axis('tight')
plt.show()

# Calculate the predicted probabilities using the sigmoid function
col = sigmoid(np.dot(X, np.transpose(logr.coef_)) + logr.intercept_)

# Get the logistic regression coefficients
cf = logr.coef_

# Generate x-values for plotting the decision boundary
xplot = np.arange(-1.0, 1.2, 0.01)

# Calculate y-values for the decision boundary
yplot = -(cf[0, 0] * xplot + logr.intercept_) / cf[0, 1]

# Plot the data points color-coded by predicted probabilities and the decision boundary
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=col)
plt.plot(xplot, yplot, 'g')
plt.suptitle('Logistic Regression Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(True, which='both')
plt.axis('tight')
plt.show()
