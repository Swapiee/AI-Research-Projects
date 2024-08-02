import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Flatten, Dense
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import numpy as np

# Number of classes in the dataset
nc = 10

# Load the Fashion MNIST dataset
# This returns two tuples: (training data, training labels) and (test data, test labels)
(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

# Show a random sample image from the training set
plt.figure(1)
imgplot1 = plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

# Show a random sample image from the test set
plt.figure(2)
imgplot2 = plt.imshow(Xtest[nr.randint(10000)])
plt.show()

# Reshape the training and test data to include a single channel
# This is necessary for the Conv2D layer which expects a 4D input
Xtrain = Xtrain.reshape(60000, 28, 28, 1)
Xtest = Xtest.reshape(10000, 28, 28, 1)

# One-hot encode the labels for the training and test data
# This converts the labels into a binary matrix representation
ytrainEnc = tf.one_hot(ytrain, depth=nc)
ytestEnc = tf.one_hot(ytest, depth=nc)

# Create a sequential model
model = Sequential()

# Add a Conv2D layer with 64 filters, a kernel size of 3, ReLU activation, and an input shape of (28, 28, 1)
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))

# Add another Conv2D layer with 32 filters and ReLU activation
model.add(Conv2D(32, kernel_size=3, activation="relu"))

# Flatten the output of the previous layer to create a single long feature vector
model.add(Flatten())

# Add a Dense layer with 10 units (one for each class) and softmax activation
# Softmax activation is used to obtain the probabilities for each class
model.add(Dense(10, activation="softmax"))

# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy as the evaluation metric
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model using the training data and validation data
# Train for 3 epochs
model.fit(Xtrain, ytrainEnc, validation_data=(Xtest, ytestEnc), epochs=3)

# Predict the labels for the test data
ypred = model.predict(Xtest)

# Convert the predicted probabilities to class labels by taking the argmax (index of the highest probability)
ypred = np.argmax(ypred, axis=1)

# Calculate the accuracy score by comparing the predicted labels to the true test labels
score = accuracy_score(ypred, ytest)

# Print the accuracy score as a percentage
print('Accuracy score is', 100 * score, '%')
