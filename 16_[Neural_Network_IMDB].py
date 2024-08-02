# The aim of this module is to build, train, and evaluate a neural network model to classify movie reviews as positive or negative using the IMDB dataset.
from keras.datasets import imdb  # for loading the IMDB dataset
import numpy as np  # for numerical operations
from keras.models import Sequential  # for creating a sequential neural network
from keras.layers import Dense  # for adding densely-connected neural network layers
from sklearn.metrics import accuracy_score  # for evaluating the accuracy of the model

# Define a function to vectorize sequences of integers into binary matrix format
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # create an all-zero matrix of shape (number of sequences, dimension)
    for i in range(len(sequences)):  # iterate over each sequence
        results[i, sequences[i]] = 1  # set specific indices of each sequence to 1
    return results

# Load the IMDB dataset, keeping only the top 10,000 most frequently occurring words
(Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words=10000)

# Vectorize the training data
Xtrain = vectorize(Xtrain)

# Create a sequential neural network model
model = Sequential()
# Add the first dense layer with 50 neurons and ReLU activation function
model.add(Dense(50, input_dim=10000, activation="relu"))
# Add the second dense layer with 50 neurons and ReLU activation function
model.add(Dense(50, activation="relu"))
# Add the third dense layer with 50 neurons and ReLU activation function
model.add(Dense(50, activation="relu"))
# Add the output layer with 1 neuron and sigmoid activation function for binary classification
model.add(Dense(1, activation="sigmoid"))

# Compile the model specifying the loss function, optimizer, and evaluation metric
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the training data for 10 epochs with a batch size of 550
history = model.fit(Xtrain, ytrain, epochs=10, batch_size=550)

# Vectorize the test data
Xtest = vectorize(Xtest)

# Make predictions on the test data
ypred = model.predict(Xtest)
# Round the predictions to get binary output
ypred = np.round(ypred)

# Calculate the accuracy score of the model on the test data
score = accuracy_score(ypred, ytest)
# Print the accuracy score in percentage
print('Accuracy score is', 100 * score, '%')
