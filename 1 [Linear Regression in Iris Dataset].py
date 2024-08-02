import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

# Load the iris dataset
irisset = datasets.load_iris()
x = irisset.data[:50, 0].reshape(-1, 1)  # Sepal length (feature)
y = irisset.data[:50, 1]  # Sepal width (target)

# Train the linear regression model
reg = LinearRegression().fit(x, y)
w = reg.coef_
c = reg.intercept_

# Generate points for the regression line
xpoints = np.linspace(4, 6).reshape(-1, 1)
ypoints = reg.predict(xpoints)

# Plot the regression line and the data points
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(x, y, s=10)
plt.suptitle('Linear Regression IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

# Make predictions
yPredict = reg.predict(x)

# Calculate and print RMSE and R2 score
rmse = np.sqrt(mean_squared_error(y, yPredict))
r2 = r2_score(y, yPredict)
print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")
