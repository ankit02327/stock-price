"""
Gaussian Naive Bayes Classifier Implementation from Scratch

This module implements a Gaussian Naive Bayes classifier using:
- Gaussian probability density function
- Maximum likelihood estimation
- Prior probability calculation
- Binary classification

Mathematical Foundation:
- Gaussian PDF: P(x|μ,σ) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
- Bayes' Theorem: P(y|x) = P(x|y) * P(y) / P(x)
- Naive Assumption: P(x|y) = ∏ P(xᵢ|y)
"""

import math

def gaussian_probability(x, mean, std):
    """
    Calculate Gaussian probability density function
    
    Parameters:
    - x: input value
    - mean: mean of the Gaussian distribution
    - std: standard deviation of the Gaussian distribution
    
    Returns:
    - float: probability density value
    """
    if std == 0:  
        return 1 if x == mean else 0
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    coefficient = 1 / (math.sqrt(2 * math.pi) * std)
    return coefficient * math.exp(exponent)

def calculate_metrics(predictions, actual):
    """
    Calculate classification metrics
    
    Parameters:
    - predictions: predicted class labels
    - actual: true class labels
    
    Returns:
    - accuracy, precision, recall, f1: classification metrics
    """
    # Calculate confusion matrix components
    tp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 1)  # True Positives
    fp = sum(1 for p, a in zip(predictions, actual) if p == 1 and a == 0)  # False Positives
    tn = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 0)  # True Negatives
    fn = sum(1 for p, a in zip(predictions, actual) if p == 0 and a == 1)  # False Negatives
    
    # Calculate metrics
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


# Read input data
n = int(input())
data = []

for _ in range(n):
    line = input().split()
    row = [float(x) for x in line]
    data.append(row)

# Separate features and labels
X = []
y = []

for row in data:
    features = row[:8]   # First 8 values are features
    label = int(row[8])  # Last value is the class label
    X.append(features)
    y.append(label)

# Split data into training and testing sets
train_size = int(0.7 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# Calculate prior probabilities
class_0_count = sum(1 for label in y_train if label == 0)
class_1_count = sum(1 for label in y_train if label == 1)
total_train = len(y_train)

prior_0 = class_0_count / total_train
prior_1 = class_1_count / total_train

print(f"Class 0 Prior: {prior_0:.2f}")
print(f"Class 1 Prior: {prior_1:.2f}")

# Initialize arrays for class statistics
n_features = len(X_train[0])
means_0 = [0] * n_features
means_1 = [0] * n_features
stds_0 = [0] * n_features
stds_1 = [0] * n_features

# Separate training data by class
features_class_0 = []
features_class_1 = []

for i, label in enumerate(y_train):
    if label == 0:
        features_class_0.append(X_train[i])
    else:
        features_class_1.append(X_train[i])


# Calculate means for each class
for j in range(n_features):
    if features_class_0:
        means_0[j] = sum(row[j] for row in features_class_0) / len(features_class_0)
    if features_class_1:
        means_1[j] = sum(row[j] for row in features_class_1) / len(features_class_1)

# Calculate standard deviations for each class
for j in range(n_features):
    if len(features_class_0) > 1:
        variance_0 = sum((row[j] - means_0[j])**2 for row in features_class_0) / len(features_class_0)
        stds_0[j] = math.sqrt(variance_0)
    if len(features_class_1) > 1:
        variance_1 = sum((row[j] - means_1[j])**2 for row in features_class_1) / len(features_class_1)
        stds_1[j] = math.sqrt(variance_1)


# Make predictions on test set
predictions = []

for test_sample in X_test:
    # Calculate likelihood for class 0: P(x|y=0) * P(y=0)
    likelihood_0 = prior_0
    for j in range(n_features):
        prob = gaussian_probability(test_sample[j], means_0[j], stds_0[j])
        likelihood_0 *= prob
    
    # Calculate likelihood for class 1: P(x|y=1) * P(y=1)
    likelihood_1 = prior_1
    for j in range(n_features):
        prob = gaussian_probability(test_sample[j], means_1[j], stds_1[j])
        likelihood_1 *= prob
    
    # Choose class with higher likelihood
    if likelihood_1 > likelihood_0:
        predictions.append(1)
    else:
        predictions.append(0)

# Display results
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test}")

# Calculate and display metrics
accuracy, precision, recall, f1 = calculate_metrics(predictions, y_test)

print(f"Accuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1:.2f}")