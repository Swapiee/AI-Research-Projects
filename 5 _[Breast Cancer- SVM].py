# Import necessary libraries and modules
from sklearn import datasets  # for loading datasets
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.metrics import accuracy_score  # for evaluating model performance
from sklearn.svm import SVC  # for Support Vector Classifier

# Load the breast cancer dataset
bcancer = datasets.load_breast_cancer()
X = bcancer.data  # Feature data: measurements of cell nuclei
Y = bcancer.target  # Target labels: 0 for benign, 1 for malignant

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Fit to data, then transform it

# Split the dataset into training and testing sets
# 75% of the data will be used for training and 25% for testing
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=10)

# Train a Support Vector Machine (SVM) with a linear kernel
svmc = SVC(kernel='linear')  # Initialize the SVM classifier with a linear kernel
svmc.fit(Xtrain, Ytrain)  # Train the classifier on the training data

# Make predictions on the test set using the trained linear SVM
Ypred = svmc.predict(Xtest)

# Calculate the accuracy of the linear SVM classifier
svmcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Linear SVM Classifier is', 100 * svmcscore, '%\n')

# Train a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel
ksvmc = SVC(kernel='rbf')  # Initialize the SVM classifier with an RBF kernel
ksvmc.fit(Xtrain, Ytrain)  # Train the classifier on the training data

# Make predictions on the test set using the trained RBF SVM
Ypred = ksvmc.predict(Xtest)

# Calculate the accuracy of the RBF SVM classifier
svmcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Kernel SVM Classifier with RBF is', 100 * svmcscore, '%\n')
