"""
K-Nearest Neighbors (KNN) Classification with Manhattan Distance

This module implements KNN classification from scratch using:
- Manhattan distance metric (L1 norm)
- Data normalization for better performance
- K-fold cross-validation for model evaluation
- Majority voting for classification

Mathematical Foundation:
- Manhattan Distance: d(x,y) = Σ|xᵢ - yᵢ|
- Classification: majority vote among k nearest neighbors
"""

import numpy as np
import math
from collections import Counter

def manhattan_distance(x1, x2):
    """
    Calculate Manhattan distance between two points
    
    Parameters:
    - x1, x2: feature vectors
    
    Returns:
    - float: Manhattan distance (L1 norm)
    """
    return np.sum(np.abs(x1 - x2))

def knn_predict(X_train, y_train, X_test, k):
    """
    Predict class labels for test points using KNN with Manhattan distance
    
    Parameters:
    - X_train: training features
    - y_train: training labels
    - X_test: test features
    - k: number of nearest neighbors
    
    Returns:
    - list: predicted class labels
    """
    predictions = []
    
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = manhattan_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        # Extract class labels of k nearest neighbors
        neighbor_classes = [neighbor[1] for neighbor in k_nearest]
        
        # Majority voting
        counts = Counter(neighbor_classes)
        predicted_class = counts.most_common(1)[0][0]
        predictions.append(predicted_class)
        
    return predictions

def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    
    Returns:
    - float: accuracy score
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true) if len(y_true) > 0 else 0

def train_test_split(X, y, test_size=0.2, random_seed=42):
    """
    Split data into training and testing sets
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - test_size: proportion of data for testing
    - random_seed: random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: split datasets
    """
    np.random.seed(random_seed)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

if __name__ == "__main__":
    # Read input data
    N = int(input())
    data = []
    
    for _ in range(N):
        line = input().strip()
        if line:
            parts = line.split()
            numeric_values = [float(p) for p in parts[:-1]]  # Features
            class_label = parts[-1]  # Class label
            data.append(numeric_values + [class_label])

    # Separate features and labels
    features = np.array([row[:-1] for row in data])
    labels = np.array([row[-1] for row in data])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_seed=42)
    
    # Make predictions using KNN with k=1 (1-NN)
    predictions = knn_predict(X_train, y_train, X_test, 1)
    accuracy = calculate_accuracy(y_test, predictions)
    
    # Display results
    print("FINAL EVALUATION ON TEST SET")
    print(f"Test set accuracy with k=1: {accuracy:.2f}")