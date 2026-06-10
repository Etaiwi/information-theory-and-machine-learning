# Imports
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Diabetes dataset
diabetes = datasets.load_diabetes() # Load diabetes dataset
diabetes_X = diabetes.data # Input matrix of dimensions 442 x 10
diabetes_y = diabetes.target # Target values

# Split the data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the target values into training and testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(diabetes_X_train)
X_test_scaled = scaler.transform(diabetes_X_test)

# Reference Model Using scikit-learn
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train_scaled, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(X_test_scaled)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error: %.2f" % mean_squared_error)
print("="*80)

# Implementing Linear Regression with Gradient Descent
# train
X = X_train_scaled
y = diabetes_y_train

# train: init
n_samples, n_features = X.shape

W = np.zeros(n_features) # Initialize weights to zeros
b = 0 # Initialize bias to zero

learning_rate = 1e-2
epochs = 10000

# train: gradient descent
# Note: Save the mean squared error at each iteration in a list (e.g.,mse_history)
# for later visualization (MSE vs. iteration plot).
mse_history = []

for i in range(epochs):
    # calculate predictions
    y_train_pred = X @ W + b

    # calculate error and cost (mean squared error)
    error = y_train_pred - y
    current_mse = np.mean(error**2)
    mse_history.append(current_mse)

    # calculate gradients
    dW = (2/n_samples) * (X.T @ error) # Gradient with respect to weights
    db = (2/n_samples) * np.sum(error) # Gradient with respect to bias

    # update parameters
    W -= learning_rate * dW
    b -= learning_rate * db

    # diagnostic output
    if i % 1000 == 0:
        print("Epoch %d: %f" % (i, current_mse))

# Evaluate manual model on the test set
y_test_pred = X_test_scaled @ W + b

manual_mse = metrics.mean_squared_error(diabetes_y_test, y_test_pred)

print("Manual coefficients:\n", W)
print("Manual bias:", b)
print("Manual mean squared error: %.2f" % manual_mse)
print("=" * 80)

# Compare manual model with sckikit-learn model
print("Scikit-learn MSE: %.2f" % mean_squared_error)
print("Manual GD MSE: %.2f" % manual_mse)

relative_difference = abs(manual_mse - mean_squared_error) / mean_squared_error * 100
print("Relative MSE difference: %.2f%%" % relative_difference)

# Plot 1: MSE vs. Iteration
plt.figure(figsize=(8, 5))
plt.plot(mse_history)
plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs. Iteration (Logarithmic X-Axis)")
plt.grid(True)
plt.savefig("images/linear_regression/mse_vs_iteration.png")
plt.show()

# Plot 2: True vs. Predicted Values
min_value = min(diabetes_y_test.min(), y_test_pred.min())
max_value = max(diabetes_y_test.max(), y_test_pred.max())

plt.figure(figsize=(8,5))
plt.scatter(diabetes_y_test, y_test_pred)
plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs. Predicted Values")
plt.grid(True)
plt.savefig("images/linear_regression/true_vs_predicted.png")
plt.show()

# Plot 3: Feature Importance
feature_names = diabetes.feature_names
feature_importance = np.abs(W)

plt.figure(figsize=(8,5))
plt.bar(feature_names, feature_importance)
plt.xlabel("Feature")
plt.ylabel("Absolute Coefficient Value")
plt.title("Feature Importance")
plt.grid(True, axis="y")
plt.savefig("images/linear_regression/feature_importance.png")
plt.show()