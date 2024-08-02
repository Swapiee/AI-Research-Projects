from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load and preprocess the data
BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:, 0:13]
y = BosData.iloc[:, 13]  # MEDV: Median value of owner-occupied homes in $1000s

# Standardize the features
ss = StandardScaler()
X = ss.fit_transform(X)

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# Build the model
model = Sequential()
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
history = model.fit(Xtrain, ytrain, epochs=150, batch_size=10)

# Make predictions
ypred = model.predict(Xtest)
ypred = ypred[:, 0]

# Calculate the prediction error
error = np.sum(np.abs(ytest - ypred)) / np.sum(np.abs(ytest)) * 100
print('Prediction Error is', error, '%')
