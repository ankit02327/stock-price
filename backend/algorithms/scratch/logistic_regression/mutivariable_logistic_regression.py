"""
Multivariable Logistic Regression Implementation from Scratch

This module implements logistic regression for binary classification with multiple features.
It includes:
- Data preprocessing and normalization
- Gradient descent optimization for logistic regression
- Sigmoid activation function
- Binary cross-entropy loss function
- Model evaluation and prediction

Mathematical Foundation:
- Hypothesis: h(x) = 1 / (1 + e^(-(θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃)))
- Cost Function: J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
- Gradient Descent: θ = θ - α * ∇J(θ)
"""

import numpy as np
import pandas as pd

# Read number of data points
n = int(input())
data = []

# Read feature-target pairs (3 features + 1 target)
for z in range(n):
    line = input().strip().split()
    data.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])

# Convert to DataFrame with meaningful column names
data = pd.DataFrame(data, columns=['exam1', 'exam2', 'hours_study', 'admitted'])
print("First 5 rows:")
print(data.head(5))

print("Shape (N, d):", data.shape)
print()
print("Summary statistics:")

# Extract features and target
x1 = data['exam1'].values
x2 = data['exam2'].values
x3 = data['hours_study'].values
y = data['admitted'].values

# Display summary statistics for each feature
print(f"exam1 -> Min: {np.min(x1)}, Max: {np.max(x1)}, Mean: {np.mean(x1):.2f}, Std: {np.std(x1, ddof=1):.2f}")
print(f"exam2 -> Min: {np.min(x2)}, Max: {np.max(x2)}, Mean: {np.mean(x2):.2f}, Std: {np.std(x2, ddof=1):.2f}")
print(f"hours_study -> Min: {np.min(x3)}, Max: {np.max(x3)}, Mean: {np.mean(x3):.2f}, Std: {np.std(x3, ddof=1):.2f}")
print()

# Feature normalization using z-score: (x - μ) / σ
# Using population standard deviation (ddof=0) for consistency
mean_x1, std_x1 = np.mean(x1), np.std(x1)
mean_x2, std_x2 = np.mean(x2), np.std(x2)
mean_x3, std_x3 = np.mean(x3), np.std(x3)

# Normalize features
x1 = (x1 - mean_x1) / std_x1
x2 = (x2 - mean_x2) / std_x2
x3 = (x3 - mean_x3) / std_x3

# Create feature matrix with bias term: [1, x1, x2, x3]
X = np.c_[np.ones(x1.shape[0]), x1, x2, x3]

# Initialize parameters: [θ₀, θ₁, θ₂, θ₃] for [bias, exam1, exam2, hours_study]
theta = np.zeros(X.shape[1])

# Training parameters
learningRate = 0.01
epochs = 1500

# Gradient Descent Algorithm for Multivariable Logistic Regression
for epoch in range(epochs):
    # Forward pass: compute z = Xθ
    z = X.dot(theta)
    
    # Apply sigmoid activation: h(x) = 1 / (1 + e^(-z))
    prediction = 1 / (1 + np.exp(-z))
    
    # Clip predictions to prevent log(0) in loss calculation
    prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
    
    # Compute binary cross-entropy loss
    loss = -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
    
    # Compute gradient: ∇J(θ) = (1/m) * X^T * (h(x) - y)
    gradient = X.T.dot(prediction - y) / len(y)
    
    # Update parameters: θ = θ - α * ∇J(θ)
    theta -= learningRate * gradient

# Display final results
FormTheta = [f"{t:.2f}" for t in theta]
print(f"Final theta: [{(', '.join(FormTheta))}]")
print(f"Final Loss: {round(loss, 2)}")
print()

print()

def predictAdmission(exam1, exam2, hours_study, theta, mean_x1, std_x1, mean_x2, std_x2, mean_x3, std_x3):
    """
    Predict admission probability for new data points
    
    Parameters:
    - exam1, exam2, hours_study: raw feature values
    - theta: learned model parameters
    - mean_x1, std_x1, etc.: normalization statistics from training data
    
    Returns:
    - prediction: probability of admission (0-1)
    """
    # Normalize new inputs using training data statistics
    exam1_std = (exam1 - mean_x1) / std_x1
    exam2_std = (exam2 - mean_x2) / std_x2
    hours_study_std = (hours_study - mean_x3) / std_x3

    # Create feature vector with bias term
    X_new = np.array([1, exam1_std, exam2_std, hours_study_std])

    # Apply learned model: z = Xθ, then sigmoid
    z = X_new.dot(theta)
    prediction = 1 / (1 + np.exp(-z))
    return prediction

# Make predictions for specific test cases
predictionA = predictAdmission(72, 80, 11, theta, mean_x1, std_x1, mean_x2, std_x2, mean_x3, std_x3)
print(f"Prediction for (exam1=72, exam2=80, hours study=11): {predictionA:.2f}")

predictionB = predictAdmission(150, 118, 20, theta, mean_x1, std_x1, mean_x2, std_x2, mean_x3, std_x3)
print(f"Prediction for (exam1=150, exam2=118, hours study=20): {predictionB:.2f}")