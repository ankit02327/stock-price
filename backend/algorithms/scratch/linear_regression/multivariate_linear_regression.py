"""
Multivariate Linear Regression Implementation from Scratch

This module implements linear regression for multiple features (multivariate case).
It includes:
- Data preprocessing and normalization for multiple features
- Vectorized gradient descent optimization
- Model evaluation with MSE
- Prediction for new data points

Mathematical Foundation:
- Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
- Cost Function: J(θ) = (1/2m) * Σ(h(x) - y)²
- Gradient Descent: θ = θ - α * ∇J(θ)
"""

import numpy as np
import pandas as pd
import statistics

# Read number of data points
N = int(input())
data = []

# Read feature-target pairs (3 features + 1 target)
for i in range(N):
    inp = input()
    temp = inp.split()
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
        print(float(data[i][1]), end=" ")
        print(float(data[i][2]), end=" ")
        print(float(data[i][3]))
        
# Convert to DataFrame with meaningful column names
df = pd.DataFrame(data, columns=["s", "b", "a", "p"])  # s, b, a are features, p is target
print(f"({N},4)")

# Convert to numeric types
df["s"] = df["s"].astype(float)
df["b"] = df["b"].astype(float)
df["a"] = df["a"].astype(float)
df["p"] = df["p"].astype(float)

# Extract features and target
s = df['s'].tolist()
b = df['b'].tolist()
a = df['a'].tolist()
p = df['p'].tolist()

# Calculate statistics for each feature
m_s = df["s"].mean()
m_b = df["b"].mean()
m_a = df["a"].mean()
m = df.mean()
s_std = statistics.pstdev(s)
b_std = statistics.pstdev(b)
a_std = statistics.pstdev(a)
p_std = statistics.pstdev(p)

mi = df.min()
ma = df.max()

# Display summary statistics for each feature
print(f"{m.iloc[0]:.2f} {s_std:.2f} {mi.iloc[0]:.2f} {ma.iloc[0]:.2f}")
print(f"{m.iloc[1]:.2f} {b_std:.2f} {mi.iloc[1]:.2f} {ma.iloc[1]:.2f}")
print(f"{m.iloc[2]:.2f} {a_std:.2f} {mi.iloc[2]:.2f} {ma.iloc[2]:.2f}")
print(f"{m.iloc[3]:.2f} {p_std:.2f} {mi.iloc[3]:.2f} {ma.iloc[3]:.2f}")

# Feature normalization using z-score: (x - μ) / σ
ns = [(x - m_s) / s_std for x in s]
nb = [(x - m_b) / b_std for x in b]
na = [(x - m_a) / a_std for x in a]

# Prepare feature matrix and target vector
X = np.column_stack([ns, nb, na])  # Normalized features
X2 = np.column_stack([np.ones(N), X])  # Add bias column (intercept)
y = np.array(p)  # Target values

# Initialize parameters: [θ₀, θ₁, θ₂, θ₃] for [bias, s, b, a]
t = np.zeros(4)
lr = 0.01  # Learning rate
ep = 300   # Number of epochs

# Gradient Descent Algorithm
for i in range(ep):
    # Forward pass: compute predictions h(x) = Xθ
    pred = X2 @ t
    
    # Compute gradient: ∇J(θ) = (1/m) * X^T * (h(x) - y)
    grad = (1/N) * X2.T @ (pred - y)
    
    # Update parameters: θ = θ - α * ∇J(θ)
    t = t - lr * grad
    
# Final predictions and MSE calculation
f_pred = X2 @ t
mse1 = (1/(2*N)) * np.sum((f_pred - y)**2)

# Compare with analytical solution (Normal Equation)
t2 = np.linalg.pinv(X2.T @ X2) @ X2.T @ y
pred2 = X2 @ t2
mse2 = (1/(2*N)) * np.sum((pred2 - y)**2)
diff = abs(mse1 - mse2)

# Display results
print(f"Final theta=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}, {t[3]:.3f}]")
print(f"Final MSE={mse1:.2f}")
print(f"MSE Difference={diff:.5f}")

# Make predictions for new data points
# First normalize the new inputs using training data statistics
h1 = [
    (150 - m_s) / s_std,  # Normalize first feature
    (3 - m_b) / b_std,    # Normalize second feature
    (5 - m_a) / a_std     # Normalize third feature
]

h2 = [
    (200 - m_s) / s_std,  # Normalize first feature
    (4 - m_b) / b_std,    # Normalize second feature
    (2 - m_a) / a_std     # Normalize third feature
]

# Add bias term (intercept) to feature vectors
h1_f = np.array([1] + h1)  # [1, s_norm, b_norm, a_norm]
h2_f = np.array([1] + h2)  # [1, s_norm, b_norm, a_norm]

# Apply the learned model: h(x) = θ₀ + θ₁s + θ₂b + θ₃a
p1 = h1_f @ t  # Prediction for first test case
p2 = h2_f @ t  # Prediction for second test case

# Display predictions
print(f"{p1:.2f}")
print(f"{p2:.2f}")

