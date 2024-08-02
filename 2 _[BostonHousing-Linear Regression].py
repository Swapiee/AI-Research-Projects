import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the dataset
BosData = pd.read_csv('BostonHousing.csv')

# Extract features (columns 0 to 11) and target (column 13)
X = BosData.iloc[:, 0:11]
y = BosData.iloc[:, 13]  # MEDV: Median value of owner-occupied homes in $1000s

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Initialize and train the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict the target values for the training set
y_train_predict = reg.predict(X_train)

# Calculate and print RMSE and R2 score for the training set
rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2 = r2_score(y_train, y_train_predict)
print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")

# Predict the target values for the testing set
y_test_predict = reg.predict(X_test)

# Calculate and print RMSE and R2 score for the testing set
rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2 = r2_score(y_test, y_test_predict)
print('Test RMSE =', rmse)
print('Test R2 score =', r2)

