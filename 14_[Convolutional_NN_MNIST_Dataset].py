import tensorflow as tf  # Import TensorFlow for building and training the neural network
from keras.datasets import fashion_mnist  # Import Fashion MNIST dataset from Keras
from keras.layers import Conv2D, Flatten, Dense  # Import necessary layers from Keras
import numpy.random as nr  # Import numpy random for generating random numbers
import matplotlib.pyplot as plt  # Import Matplotlib for plotting images
from keras.models import Sequential  # Import Sequential model from Keras
from sklearn.metrics import accuracy_score  # Import accuracy_score from scikit-learn for evaluating the model
import numpy as np  # Import numpy for numerical operations

# Number of classes in the Fashion MNIST dataset
nc = 10 

# Load the Fashion MNIST dataset
# Xtrain and Xtest are the images, ytrain and ytest are the corresponding labels
(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

# Display a random image from the training set
plt.figure(1)  # Create a new figure
imgplot1 = plt.imshow(Xtrain[nr.randint(60000)])  # Select a random image from the training set and display it
plt.show()  # Show the figure

# Display a random image from the test set
plt.figure(2)  # Create another new figure
imgplot2 = plt.imshow(Xtest[nr.randint(10000)])  # Select a random image from the test set and display it
plt.show()  # Show the figure

# Reshape the training and test images to include the channel dimension (grayscale image has 1 channel)
Xtrain = Xtrain.reshape(60000, 28, 28, 1)  # Reshape training data
Xtest = Xtest.reshape(10000, 28, 28, 1)  # Reshape test data

# Convert the training and test labels to one-hot encoded format
# This is necessary for using categorical crossentropy loss
ytrainEnc = tf.one_hot(ytrain, depth=nc)  # One-hot encode training labels
ytestEnc = tf.one_hot(ytest, depth=nc)  # One-hot encode test labels

# Create a Sequential model
model = Sequential()

# Add a Conv2D layer with 64 filters, a kernel size of 3x3, ReLU activation, and input shape (28, 28, 1)
# This layer will learn 64 different filters
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))

# Add another Conv2D layer with 32 filters and ReLU activation
# This layer will learn 32 different filters
model.add(Conv2D(32, kernel_size=3, activation="relu"))

# Flatten the output of the previous layer to create a single long feature vector
# This step is necessary before feeding the data into a fully connected (Dense) layer
model.add(Flatten())

# Add a Dense (fully connected) layer with 10 units and softmax activation
# Each unit represents a class, and softmax converts the outputs to probability distributions
model.add(Dense(10, activation="softmax"))

# Compile the model
# Use categorical crossentropy loss for multi-class classification
# Use Adam optimizer for training
# Track accuracy as the performance metric
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the training data
# Use the validation data for evaluating the model after each epoch
# Train for 3 epochs
model.fit(Xtrain, ytrainEnc, validation_data=(Xtest, ytestEnc), epochs=3)

# Predict the class probabilities for the test data
ypred = model.predict(Xtest)

# Convert the predicted probabilities to class labels by taking the index of the highest probability
ypred = np.argmax(ypred, axis=1)

# Calculate the accuracy score by comparing the predicted labels to the true labels
score = accuracy_score(ypred, ytest)

# Print the accuracy score as a percentage
print('Accuracy score is', 100 * score, '%')
