import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.optimizers import SGD
nc = 10 # Number of classes

#CIFAR is an acronym that stands for the Canadian 
#Institute For Advanced Research and the CIFAR-10 
#dataset was developed along with the CIFAR-100 
#dataset by researchers at the CIFAR institute.
#The dataset is comprised of 50,000 32×32 
#pixel color photographs of objects from 10 classes, 
#such as frogs, birds, cats, ships, etc. 
#The class labels and their standard associated 
#integer values are listed below.
#0: airplane
#1: automobile
#2: bird
#3: cat
#4: deer
#5: dog
#6: frog
#7: horse
#8: ship
#9: truck


# Number of classes in CIFAR-10 dataset
nc = 10 

# Load the CIFAR-10 dataset
(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()

# Show a random sample image from the training set
plt.figure(1)
imgplot1 = plt.imshow(Xtrain[nr.randint(50000)])
plt.show()

# Show a random sample image from the test set
plt.figure(2)
imgplot2 = plt.imshow(Xtest[nr.randint(10000)])
plt.show()

# Normalize the training data to the range [0, 1]
Xtrain = Xtrain.astype('float32')
Xtrain = Xtrain[0:20000,:] / 255.0

# Normalize the test data to the range [0, 1]
Xtest = Xtest.astype('float32')
Xtest = Xtest / 255.0

# One-hot encode the training labels
ytrainEnc = tf.one_hot(ytrain[0:20000,0], depth=nc)

# One-hot encode the test labels
ytestEnc = tf.one_hot(ytest[:,0], depth=nc)

# Define the Sequential model
model = Sequential()

# Add a 2D convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and He uniform initialization
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

# Add another 2D convolutional layer with the same configuration
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add a 2D convolutional layer with 64 filters, 3x3 kernel size, ReLU activation, and He uniform initialization
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# Add another 2D convolutional layer with the same configuration
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Add a 2D convolutional layer with 128 filters, 3x3 kernel size, ReLU activation, and He uniform initialization
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# Add another 2D convolutional layer with the same configuration
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# Add a max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers to feed into fully connected layers
model.add(Flatten())

# Add a fully connected layer with 128 units, ReLU activation, and He uniform initialization
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

# Add the output layer with 10 units (one for each class) and softmax activation
model.add(Dense(10, activation='softmax'))

# Define the optimizer (Stochastic Gradient Descent with a learning rate of 0.001 and momentum of 0.9)
opt = SGD(learning_rate=0.001, momentum=0.9)

# Compile the model with categorical crossentropy loss and accuracy as a metric
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data for 10 epochs, batch size of 200, and validate with test data
history = model.fit(Xtrain, ytrainEnc, epochs=10, batch_size=200, validation_data=(Xtest, ytestEnc))

# Predict the labels for the test data
ypred = model.predict(Xtest)

# Convert the predicted probabilities to class labels
ypred = np.argmax(ypred, axis=1)

# Calculate the accuracy of the predictions
score = accuracy_score(ypred, ytest)

# Print the accuracy score
print('Accuracy score is', 100*score, '%')
