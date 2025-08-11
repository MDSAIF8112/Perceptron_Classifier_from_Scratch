import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
np.random.seed(0)
num_samples = 100

# Class 0: centered around (1,1)
X0 = np.random.randn(num_samples, 2) + [1, 1]
y0 = np.zeros(num_samples)

# Class 1: centered around (3,3)
X1 = np.random.randn(num_samples, 2) + [3, 3]
y1 = np.ones(num_samples)

# Combine
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# 2. Add bias term to input
X_bias = np.c_[np.ones(X.shape[0]), X]  # Shape: (200, 3)

# 3. Initialize weights randomly
weights = np.random.randn(3)
learning_rate = 0.1
epochs = 100

# 4. Train using Perceptron update rule
for epoch in range(epochs):
    for i in range(len(X_bias)):
        z = np.dot(weights, X_bias[i])
        prediction = 1 if z > 0 else 0
        error = y[i] - prediction
        weights += learning_rate * error * X_bias[i]

print("Final weights:", weights)

# 5. Plot data and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], label="Class 0")
plt.scatter(X1[:, 0], X1[:, 1], label="Class 1")

# Plot decision boundary
x_vals = np.array([0, 5])
y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Classification")
plt.legend()
plt.grid(True)
plt.show()
