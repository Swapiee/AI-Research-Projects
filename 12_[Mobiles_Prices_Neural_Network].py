# Import necessary libraries and modules
from keras.models import Sequential  # Sequential model type from Keras
from keras.layers import Dense  # Dense layer type from Keras
import pandas as pd  # Pandas for data manipulation
from sklearn.preprocessing import StandardScaler  # StandardScaler for data standardization
from sklearn.preprocessing import OneHotEncoder  # OneHotEncoder for encoding categorical labels
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
import numpy as np  # NumPy for numerical operations
from sklearn.metrics import accuracy_score  # Function to compute the accuracy score

# Load the dataset from a CSV file into a Pandas DataFrame
df = pd.read_csv("mobile_prices.csv")

# Separate features (X) and labels (y) from the DataFrame
X = df.iloc[:, :20].values  # Features: all rows, first 20 columns
y = df.iloc[:, 20:21].values  # Labels: all rows, 21st column (index 20)

# Create an instance of StandardScaler and fit it to the features to standardize them
ss = StandardScaler()
X = ss.fit_transform(X)  # Standardize features by removing mean and scaling to unit variance

# Split the dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)  # 10% of data is used for testing

# Create an instance of OneHotEncoder
# OneHotEncoder converts categorical labels into a format that can be provided to ML algorithms
# Each category is converted into a binary vector with only one element set to 1
oh = OneHotEncoder()
ytrain = oh.fit_transform(ytrain).toarray()  # Fit and transform the labels for the training set

# Determine the number of unique classes in the labels
num_classes = len(np.unique(y))  # This will be used to define the output layer size

# Create a Sequential model
model = Sequential()
# Add a dense (fully connected) layer with 16 neurons, ReLU activation function, and input dimension of 20
model.add(Dense(16, input_dim=20, activation="relu"))
# Add another dense layer with 12 neurons and ReLU activation function
model.add(Dense(12, activation="relu"))
# Add an output dense layer with number of neurons equal to the number of classes and softmax activation function
model.add(Dense(num_classes, activation="softmax"))#Use sigmoid in place of softmax if 2 classes, i.e, binary , for multiclass classification use this softmax

# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(loss="categorical_crossentropy" , optimizer="adam", metrics=["accuracy"])#use binary_crossentropy if needed

# Train the model on the training data
history = model.fit(Xtrain, ytrain, epochs=100, batch_size=64)  # Train for 100 epochs with batch size of 64

# Make predictions on the test data
ypred = model.predict(Xtest)
# Convert the predictions from one-hot encoded format to label format
ypred = np.argmax(ypred, axis=1)

# Compute the accuracy score by comparing predicted labels with true labels
score = accuracy_score(ypred, ytest)
# Print the accuracy score
print('Accuracy score is', 100*score, '%')
