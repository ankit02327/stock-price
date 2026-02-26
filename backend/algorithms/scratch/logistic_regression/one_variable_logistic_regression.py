"""
Single Variable Logistic Regression Implementation from Scratch

This module implements logistic regression for binary classification with one feature.
It includes:
- Data preprocessing and visualization
- Gradient descent optimization for logistic regression
- Sigmoid activation function
- Binary cross-entropy loss function
- Model evaluation and prediction

Mathematical Foundation:
- Hypothesis: h(x) = 1 / (1 + e^(-(θ₀ + θ₁x)))
- Cost Function: J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
- Gradient Descent: θ = θ - α * ∇J(θ)
"""

import numpy as np
import pandas as pd

# Read number of data points
n = int(input())
data = []

# Read exam score and admission status pairs
for z in range(n):
    line = input().strip().split()
    data.append([int(line[0]), int(line[1])])

# Convert to DataFrame for easier manipulation
data = pd.DataFrame(data, columns=['exam_score', 'admitted'])
print("First 5 rows:")
print(data.head(5))

print("Shape (N, d):", data.shape)
print()
print("Summary statistics for exam_score:")

# Calculate summary statistics
examScoreMin = np.min(data['exam_score'])
examScoreMax = np.max(data['exam_score'])
examScoreMean = np.mean(data['exam_score'])
examScoreStd = np.std(data['exam_score'])

print("Min:", examScoreMin)
print("Max:", examScoreMax)
print(f"Mean: {examScoreMean:.2f}")
print(f"Std: {examScoreStd:.2f}")
print()
# Model Training Parameters
epochs = 1000
learning_rate = 0.01
theta0 = 0  # Intercept (bias term)
theta1 = 0  # Slope (weight for exam score)

# Extract features and target
x = data['exam_score'].values
y = data['admitted'].values

# Gradient Descent Algorithm for Logistic Regression
for i in range(epochs):
    # Forward pass: compute z = θ₀ + θ₁x
    z = theta0 + theta1 * x
    
    # Apply sigmoid activation: h(x) = 1 / (1 + e^(-z))
    pred = 1 / (1 + np.exp(-z))
    
    # Compute gradients for binary cross-entropy loss
    theta0_gradient = np.mean(pred - y)  # ∂J/∂θ₀
    theta1_gradient = np.mean((pred - y) * x)  # ∂J/∂θ₁
    
    # Update parameters using gradient descent
    theta0 -= learning_rate * theta0_gradient
    theta1 -= learning_rate * theta1_gradient

print()

# Final model evaluation
z = theta0 + theta1 * x
pred = 1 / (1 + np.exp(-z))

# Compute binary cross-entropy loss
loss = -np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10))

print(f"Final theta0: {theta0:.2f}")
print(f"Final theta1: {theta1:.2f}")
print(f"Final Loss: {loss:.2f}")
print()

# Make predictions for specific exam scores
threshold = 0.5
print()

# Prediction for exam score = 65
pred65 = 1 / (1 + np.exp(-(theta0 + theta1 * 65)))
print(f"Prediction for exam_score=65: {pred65:.2f}")

# Prediction for exam score = 155
pred155 = 1 / (1 + np.exp(-(theta0 + theta1 * 155)))
print(f"Prediction for exam_score=155: {pred155:.2f}")