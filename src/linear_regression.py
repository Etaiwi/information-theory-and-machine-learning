# Imports
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Print the shapes of the training and testing sets and the feature names
print_toggle = False
if print_toggle:
    print(diabetes_X_train.shape) # (422, 10)
    print(diabetes_X_test.shape) # (20, 10)
    print(diabetes_y_train.shape) # (422,)
    print(diabetes_y_test.shape) # (20,)
    print(diabetes.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Feature Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(diabetes_X_train)
X_test_scaled = scaler.transform(diabetes_X_test)

# Reference Model Using scikit-learn
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
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
W = ...
b = ...
learning_rate = 1e-3
epochs = 10
# train: gradient descent
# Note: Save the mean squared error at each iteration in a list (e.g.,mse_history)
# for later visualization (MSE vs. iteration plot).
mse_history = []
for i in range(epochs):
    # calculate predictions
    # TODO
    # calculate error and cost (mean squared error)
    # TODO
    # mse_history.append(current_mse)
    # calculate gradients
    # TODO
    # update parameters
    # TODO
    # diagnostic output
    if i % 1000 == 0:
    print("Epoch %d: %f" % (i, current_mse))