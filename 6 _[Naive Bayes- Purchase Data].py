# Import necessary libraries
import pandas as pd  # for handling data in DataFrames
from sklearn.model_selection import train_test_split  # for splitting data into training and testing sets
from sklearn.metrics import confusion_matrix  # for evaluating model performance using a confusion matrix
from sklearn.naive_bayes import GaussianNB  # for the Naive Bayes classifier
from sklearn.preprocessing import StandardScaler  # for feature scaling
import matplotlib.pyplot as plt  # for plotting data

# Load the dataset from a CSV file
purchaseData = pd.read_csv('Purchase_Logistic.csv')

# Extract features and target variable from the dataset
# X will contain the features (Age and Estimated Salary)
# Y will contain the target variable (Purchased or not)
X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
# 75% of the data will be used for training and 25% for testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# Initialize the Gaussian Naive Bayes classifier
cf = GaussianNB()

# Train the classifier on the training data
cf.fit(Xtrain, Ytrain)

# Make predictions on the test set using the trained classifier
Ypred = cf.predict(Xtest)

# Evaluate the model performance using a confusion matrix
cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix =\n', cmat)

# Plot the original data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=Y)  # Scatter plot of Age vs. Estimated Salary, colored by Purchased status
plt.suptitle('Purchase Data')  # Title of the plot
plt.xlabel('Scaled Age')  # X-axis label
plt.ylabel('Scaled Income')  # Y-axis label
plt.grid(True, which='both')  # Show grid lines
plt.axis('tight')  # Tight layout
plt.show()  # Display the plot

# Predict the entire dataset using the trained classifier for visualization
col = cf.predict(X)

# Plot the predicted data
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=col)  # Scatter plot of Age vs. Estimated Salary, colored by predicted Purchased status
plt.suptitle('Naive Bayes Purchase Data')  # Title of the plot
plt.xlabel('Scaled Age')  # X-axis label
plt.ylabel('Scaled Income')  # Y-axis label
plt.grid(True, which='both')  # Show grid lines
plt.axis('tight')  # Tight layout
plt.show()  # Display the plot
