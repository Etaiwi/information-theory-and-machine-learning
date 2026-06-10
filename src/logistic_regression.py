import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

# Filter digits 0 and 1
mask_train = (y_train_full == 0) | (y_train_full == 1)
mask_test = (y_test_full == 0) | (y_test_full == 1)

X_train = X_train_full[mask_train].reshape(-1, 28*28) / 255.0
y_train = y_train_full[mask_train]

X_test = X_test_full[mask_test].reshape(-1, 28*28) / 255.0
y_test = y_test_full[mask_test]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("Training labels:", np.unique(y_train))
print("Test labels:", np.unique(y_test))

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train scikit-learn logistic regression
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Standardize again and add intercept
X_train_manual = scaler.transform(X_train)
X_manual = np.hstack([np.ones((X_train_manual.shape[0], 1)), X_train_manual])
y_manual = y_train.reshape(-1, 1)

# Prepare test data with intercept
X_test_manual = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Initialize weights
W = np.zeros((X_manual.shape[1], 1))

# Set hyperparameters
lr = 1e-1
epochs = 100

train_loss_history = []
test_loss_history = []

W_after_5 = None
W_after_100 = None

# Gradient descent loop
for i in range(epochs):
    z = X_manual @ W
    y_hat = sigmoid(z)

    # Compute loss (binary cross-entropy)
    loss = -np.mean(y_manual * np.log(y_hat + 1e-10) + (1 - y_manual) * np.log(1 - y_hat + 1e-10))
    train_loss_history.append(loss)

    # Compute test loss
    y_hat_test = sigmoid(X_test_manual @ W)
    test_loss = -np.mean(
        y_test.reshape(-1, 1) * np.log(y_hat_test + 1e-10) +
        (1 - y_test.reshape(-1, 1)) * np.log(1 - y_hat_test + 1e-10)
    )
    test_loss_history.append(test_loss)

    # Compute gradient
    grad = (X_manual.T @ (y_hat - y_manual)) / X_manual.shape[0]

    # Update weights
    W -= lr * grad

    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1}, Loss: {loss:.4f}")

    if i + 1 == 5:
        W_after_5 = W.copy()
    
    if i + 1 == 100:
        W_after_100 = W.copy()

# Evaluation and Visualization
# Accuracy and Log Loss at 5 and 100 epochs
def evaluate_manual_model(W_model, name):
    y_prob_manual = sigmoid(X_test_manual @ W_model)
    y_pred_manual = (y_prob_manual >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_manual)
    loss = log_loss(y_test, y_prob_manual)

    print(f"{name} Accuracy:", accuracy)
    print(f"{name} Log Loss:", loss)
    print("=" * 80)

    return y_prob_manual, y_pred_manual

# Evaluate manual model after 5 and 100 epochs
y_prob_manual_5, y_pred_manual_5 = evaluate_manual_model(W_after_5, "Manual model after 5 epochs")
y_prob_manual, y_pred_manual = evaluate_manual_model(W_after_100, "Manual model after 100 epochs")

# Compare with scikit-learn
print("scikit-learn accuracy:", accuracy_score(y_test, y_pred))
print("scikit-learn log loss:", log_loss(y_test, y_prob))
print("=" * 80)


os.makedirs("images/logistic_regression", exist_ok=True)
# Plot 1: Loss vs Iteration
epoch_numbers = np.arange(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epoch_numbers, train_loss_history, label="Train Loss")
plt.plot(epoch_numbers, test_loss_history, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss vs. Iteration")
plt.legend()
plt.grid(True)
plt.savefig("images/logistic_regression/loss_vs_iteration.png")
plt.show()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_manual.ravel())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(8, 5))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig("images/logistic_regression/confusion_matrix.png")
plt.show()

# Plot 3: Important Pixels
pixel_weights = W[1:].reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(pixel_weights)
plt.colorbar()
plt.title("Important Pixels")
plt.axis("off")
plt.savefig("images/logistic_regression/important_pixels.png", bbox_inches="tight")
plt.show()

# Plot 4: Misclassified Examples - Manual Logistic Regression
misclassified_idx = np.where(y_pred_manual.ravel() != y_test)[0]
num_examples = min(10, len(misclassified_idx))

plt.figure(figsize=(8 * num_examples, 8))

for i in range(num_examples):
    idx = misclassified_idx[i]

    pred_label = y_pred_manual[idx, 0]
    prob_1 = y_prob_manual[idx, 0]
    confidence = prob_1 if pred_label == 1 else 1 - prob_1

    plt.subplot(1, num_examples, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_test[idx]}\nPred: {pred_label}\nConf: {confidence:.3f}")
    plt.axis("off")

plt.suptitle("Misclassified Examples - Manual Logistic Regression")
plt.savefig("images/logistic_regression/misclassified_examples_manual.png", bbox_inches="tight")
plt.show()

# Misclassified Examples - scikit-learn Logistic Regression
misclassified_idx_sklearn = np.where(y_pred != y_test)[0]
num_examples_sklearn = min(10, len(misclassified_idx_sklearn))

plt.figure(figsize=(8 * num_examples_sklearn, 8))

for i in range(num_examples_sklearn):
    idx = misclassified_idx_sklearn[i]

    pred_label = y_pred[idx]
    prob_1 = y_prob[idx]
    confidence = prob_1 if pred_label == 1 else 1 - prob_1

    plt.subplot(1, num_examples_sklearn, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"True: {y_test[idx]}\nPred: {pred_label}\nConf: {confidence:.3f}")
    plt.axis("off")

plt.suptitle("Misclassified Examples - scikit-learn Logistic Regression")
plt.savefig("images/logistic_regression/misclassified_examples_sklearn.png", bbox_inches="tight")
plt.show()