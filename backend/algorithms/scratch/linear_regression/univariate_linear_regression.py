"""
Univariate Linear Regression Implementation from Scratch

This module implements linear regression for a single feature (univariate case).
It includes:
- Data preprocessing and normalization
- Gradient descent optimization
- Model evaluation with MSE
- Prediction for new data points

Mathematical Foundation:
- Hypothesis: h(x) = θ₀ + θ₁x
- Cost Function: J(θ) = (1/2m) * Σ(h(x) - y)²
- Gradient Descent: θ = θ - α * ∇J(θ)
"""

import numpy as np
import pandas as pd
import statistics

# Read number of data points
N = int(input())
data = []

# Read feature-target pairs
for i in range(N):
    inp = input()
    temp = inp.split(" ")
    data.append(temp)
    
# Display first 5 rows of dataset
if (N < 5):
    for i in data:
        for j in i:
            print(float(j), end=" ")
        print()
else:
    for i in range(5):
        print(float(data[i][0]), end=" ")
        print(float(data[i][1]))
        
# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=["feature", "target"])

# Display dataset shape
print(f"({N},2)")

# Convert to numeric types
df["feature"] = df["feature"].astype(float)
df["target"] = df["target"].astype(float)
feature = df['feature'].tolist()
target = df['target'].tolist()

# Calculate and display summary statistics
Mean = df.mean()
# Use population standard deviation (N denominator)
target_pstd = statistics.pstdev(target)
feature_pstd = statistics.pstdev(feature)
Std = df.std()
Min = df.min()
Max = df.max()

# Display feature statistics: mean, std, min, max
print(f"{Mean.iloc[0]:.2f} {feature_pstd:.2f} {Min.iloc[0]:.2f} {Max.iloc[0]:.2f}")
# Display target statistics: mean, std, min, max
print(f"{Mean.iloc[1]:.2f} {target_pstd:.2f} {Min.iloc[1]:.2f} {Max.iloc[1]:.2f}")

# Feature normalization using z-score: (x - μ) / σ
normalized_feature = [(x - Mean.iloc[0]) / feature_pstd for x in feature]

# Model Training using Gradient Descent
# Initialize parameters
theta0 = 0  # Intercept (bias term)
theta1 = 0  # Slope (weight for feature)
learning_rate = 0.01  # Step size for gradient descent
epochs = 1000  # Number of training iterations

# Number of data points
N2 = len(feature)

# Gradient Descent Algorithm
for epoch in range(epochs):
    # Forward pass: compute predictions h(x) = θ₀ + θ₁x
    predictions = [theta0 + theta1 * x for x in normalized_feature]
    
    # Compute gradients using chain rule
    grad_theta0 = 0  # ∂J/∂θ₀
    grad_theta1 = 0  # ∂J/∂θ₁
    
    for i in range(N):
        error = predictions[i] - target[i]  # h(x) - y
        grad_theta0 += error  # Sum of errors for intercept
        grad_theta1 += error * normalized_feature[i]  # Sum of errors × feature
        
    # Average gradients over all samples
    grad_theta0 /= N
    grad_theta1 /= N
    
    # Update parameters using gradient descent
    theta0 = theta0 - learning_rate * grad_theta0
    theta1 = theta1 - learning_rate * grad_theta1

# Model Evaluation
# Compute final predictions and Mean Squared Error
final_predictions = [theta0 + theta1 * x for x in normalized_feature]
mse = 0
for i in range(N):
    error = final_predictions[i] - target[i]
    mse += error * error
    
# MSE formula: (1/2m) * Σ(h(x) - y)²
mse = mse / (2 * N)

# Display final model parameters and performance
print(f"Final theta0={theta0:.3f} | theta1={theta1:.3f} | Final MSE={mse:.2f}")

# Make predictions for new data points (150 and 200)
# First normalize the new inputs using training data statistics
pred_150_normalized = (150 - Mean.iloc[0]) / feature_pstd
pred_200_normalized = (200 - Mean.iloc[0]) / feature_pstd

# Apply the learned model: h(x) = θ₀ + θ₁x
pred_150 = theta0 + theta1 * pred_150_normalized
pred_200 = theta0 + theta1 * pred_200_normalized

# Display predictions
print(f"{pred_150:.2f}")
print(f"{pred_200:.2f}")