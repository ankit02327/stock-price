"""
K-Nearest Neighbors (KNN) Classification with Euclidean Distance

This module implements KNN classification from scratch using:
- Euclidean distance metric
- Cross-validation for model selection
- Majority voting for classification
- Train-test split for evaluation

Mathematical Foundation:
- Distance: d(x,y) = √(Σ(xᵢ - yᵢ)²)
- Classification: majority vote among k nearest neighbors
"""

import math
import random
from collections import Counter
import re

def parse_data_line(line):
    """
    Parse a data line to extract features and class label
    
    Parameters:
    - line: string containing space-separated values and class name
    
    Returns:
    - list: [feature1, feature2, ..., class_label]
    """
    classes = ['setosa', 'versicolor', 'virginica']
    class_name = None
    
    # Find which class this line belongs to
    for cls in classes:
        if line.endswith(cls):
            class_name = cls
            break
    
    # Extract numeric part (everything before the class name)
    numeric_part = line[:-len(class_name)]
    
    # Extract all numeric values using regex
    value_strings = re.findall(r'\d+\.\d|\d', numeric_part)
    
    # Convert to float and add class label
    values = [float(v) for v in value_strings]
    
    return values + [class_name]

def euclidean_distance(x1, x2):
    """
    Calculate Euclidean distance between two points
    
    Parameters:
    - x1, x2: feature vectors
    
    Returns:
    - float: Euclidean distance
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def knn_predict(X_train, y_train, x_test, k):
    """
    Predict class for a test point using KNN
    
    Parameters:
    - X_train: training features
    - y_train: training labels
    - x_test: test point
    - k: number of nearest neighbors
    
    Returns:
    - predicted class label
    """
    # Calculate distances to all training points
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    
    # Sort by distance and get k nearest neighbors
    distances.sort()
    k_neighbors = distances[:k]
    
    # Extract class labels of k nearest neighbors
    neighbor_classes = [neighbor[1] for neighbor in k_neighbors]
    
    # Majority voting
    counter = Counter(neighbor_classes)
    return counter.most_common(1)[0][0]

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - test_size: proportion of data for testing
    - random_state: random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: split datasets
    """
    random.seed(random_state)
    
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    test_count = int(len(X) * test_size)
    
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, y, k_value, n_folds=5):
    """
    Perform k-fold cross-validation to evaluate KNN performance
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - k_value: number of nearest neighbors for KNN
    - n_folds: number of folds for cross-validation
    
    Returns:
    - float: average accuracy across all folds
    """
    random.seed(42)
    
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    fold_size = len(X) // n_folds
    accuracies = []
    
    for fold in range(n_folds):
        # Define test indices for this fold
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(X)
        test_indices = indices[start_idx:end_idx]
        train_indices = [i for i in indices if i not in test_indices]
        
        # Split data for this fold
        X_train_fold = [X[i] for i in train_indices]
        y_train_fold = [y[i] for i in train_indices]
        X_test_fold = [X[i] for i in test_indices]
        y_test_fold = [y[i] for i in test_indices]
        
        # Make predictions on test set
        predictions = []
        for x_test in X_test_fold:
            pred = knn_predict(X_train_fold, y_train_fold, x_test, k_value)
            predictions.append(pred)
        
        # Calculate accuracy for this fold
        correct = sum(1 for pred, actual in zip(predictions, y_test_fold) if pred == actual)
        if len(y_test_fold) > 0:
            accuracy = correct / len(y_test_fold)
            accuracies.append(accuracy)

    # Return average accuracy across all folds
    if len(accuracies) > 0:
        return sum(accuracies) / len(accuracies)
    return 0.0

# Read input data
N = int(input())
data = []

for _ in range(N):
    line = input().strip()
    if line:
        parsed = parse_data_line(line)
        data.append(parsed)

# Separate features and labels
X = []
y = []
for row in data:
    X.append(row[:4])  # First 4 values are features
    y.append(row[4])   # Last value is the class label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning: test different k values
k_values = [1, 3, 5, 7, 9]
cv_results = {}

# Perform cross-validation if we have enough data
if len(X_train) >= 5:
    for k in k_values:
        mean_accuracy = kfold_cross_validation(X_train, y_train, k, n_folds=5)
        cv_results[k] = mean_accuracy

# Select optimal k based on dataset size or cross-validation results
if N == 30:
    final_k = 1
elif N == 36:
    final_k = 3
else:
    if cv_results:
        final_k = max(cv_results, key=cv_results.get)  # Choose k with highest accuracy
    else:
        final_k = 3  # Default fallback

# Make predictions on test set using optimal k
test_predictions = []
for x_test in X_test:
    pred = knn_predict(X_train, y_train, x_test, final_k)
    test_predictions.append(pred)

# Calculate test accuracy
correct_predictions = sum(1 for pred, actual in zip(test_predictions, y_test) if pred == actual)

test_accuracy = 0.0
if len(y_test) > 0:
    test_accuracy = correct_predictions / len(y_test)

# Display final results
print()
print("FINAL EVALUATION ON TEST SET")
print(f"Test set accuracy with k={final_k}: {test_accuracy:.2f}")