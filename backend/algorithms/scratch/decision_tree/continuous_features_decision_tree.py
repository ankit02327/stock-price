"""
Decision Tree Implementation from Scratch

This module implements a decision tree classifier for continuous features.
It includes:
- Entropy and Gini impurity measures
- Recursive tree building with stopping criteria
- Hyperparameter tuning with grid search
- Train/validation/test split evaluation

Mathematical Foundation:
- Entropy: H(S) = -Σ pᵢ log₂(pᵢ)
- Gini: G(S) = 1 - Σ pᵢ²
- Information Gain: IG(S,A) = H(S) - Σ(|Sᵥ|/|S|) * H(Sᵥ)
"""

import math
import random

class DecisionTreeNode:
    """
    Node class for decision tree
    
    Each node can be either:
    - Internal node: has feature_idx, threshold, left/right children
    - Leaf node: has prediction value, is_leaf=True
    """
    
    def __init__(self):
        self.feature_idx = None    # Feature index for splitting
        self.threshold = None      # Threshold value for splitting
        self.left = None           # Left child (≤ threshold)
        self.right = None          # Right child (> threshold)
        self.prediction = None     # Predicted class (for leaf nodes)
        self.is_leaf = False       # Whether this is a leaf node

class DecisionTree:
    """
    Decision Tree Classifier for continuous features
    
    Features:
    - Supports both entropy and Gini impurity criteria
    - Configurable max depth and minimum samples per leaf
    - Handles continuous features with threshold-based splits
    """
    
    def __init__(self, criterion='entropy', max_depth=5, min_samples_leaf=1):
        """
        Initialize decision tree
        
        Parameters:
        - criterion: 'entropy' or 'gini' for impurity measure
        - max_depth: maximum depth of the tree
        - min_samples_leaf: minimum samples required in a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def calculate_entropy(self, y):
        """
        Calculate entropy of a set of labels
        
        Entropy: H(S) = -Σ pᵢ log₂(pᵢ)
        where pᵢ is the proportion of class i in set S
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        class_counts = {}
        for label in y:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        # Calculate entropy
        entropy = 0
        total = len(y)
        for count in class_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_gini(self, y):
        """
        Calculate Gini impurity of a set of labels
        
        Gini: G(S) = 1 - Σ pᵢ²
        where pᵢ is the proportion of class i in set S
        """
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        class_counts = {}
        for label in y:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
        
        # Calculate Gini impurity
        gini = 1.0
        total = len(y)
        for count in class_counts.values():
            p = count / total
            gini -= p * p
        
        return gini
    
    def calculate_impurity(self, y):
        """
        Calculate impurity using the specified criterion
        
        Parameters:
        - y: list of class labels
        
        Returns:
        - float: impurity value
        """
        if self.criterion == 'entropy':
            return self.calculate_entropy(y)
        else:
            return self.calculate_gini(y)
    
    def find_best_split(self, X, y):
        """
        Find the best split for the current node
        
        Parameters:
        - X: feature matrix
        - y: target vector
        
        Returns:
        - best_feature: index of best feature to split on
        - best_threshold: threshold value for the split
        - best_gain: information gain of the best split
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0])
        current_impurity = self.calculate_impurity(y)
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get all values for this feature
            feature_values = []
            for i in range(len(X)):
                feature_values.append(X[i][feature_idx])
            
            # Sort unique values
            unique_values = sorted(list(set(feature_values)))
            
            # Try all possible thresholds (midpoints between consecutive values)
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data based on threshold
                left_indices = []
                right_indices = []
                
                for j in range(len(X)):
                    if X[j][feature_idx] <= threshold:
                        left_indices.append(j)
                    else:
                        right_indices.append(j)
                
                # Skip if split doesn't meet minimum samples requirement
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity after split
                left_y = [y[j] for j in left_indices]
                right_y = [y[j] for j in right_indices]
                
                left_impurity = self.calculate_impurity(left_y)
                right_impurity = self.calculate_impurity(right_y)
                
                # Weighted average of child impurities
                weighted_impurity = (len(left_y) * left_impurity + len(right_y) * right_impurity) / len(y)
                
                # Information gain = parent impurity - weighted child impurities
                gain = current_impurity - weighted_impurity
                
                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        node = DecisionTreeNode()
        
        # Check stopping conditions
        unique_labels = list(set(y))
        if len(unique_labels) == 1 or depth >= self.max_depth or len(y) <= self.min_samples_leaf:
            # Make this a leaf node
            node.is_leaf = True
            # Find most common class
            class_counts = {}
            for label in y:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            node.prediction = max(class_counts, key=class_counts.get)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            # No good split found, make leaf
            node.is_leaf = True
            class_counts = {}
            for label in y:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            node.prediction = max(class_counts, key=class_counts.get)
            return node
        
        # Split the data
        left_X, left_y, right_X, right_y = [], [], [], []
        
        for i in range(len(X)):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)
        
        return node
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
    
    def predict_single(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction
    
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return predictions
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for i in range(len(y)) if predictions[i] == y[i])
        return correct / len(y)

def split_data(X, y):
    """
    Split data into train/validation/test sets with 70:15:15 ratio
    
    Parameters:
    - X: feature matrix
    - y: target vector
    
    Returns:
    - X_train, X_val, X_test: feature matrices for each split
    - y_train, y_val, y_test: target vectors for each split
    """
    n = len(X)
    
    # 70% train, 15% validation, 15% test
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Read input data
n = int(input())
X = []
y = []

for _ in range(n):
    line = input().split()
    X.append([float(line[0]), float(line[1])])  # Two features
    y.append(int(line[2]))  # Binary class label

# Split data into train/validation/test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Hyperparameter search for both entropy and Gini criteria
max_depths = [2, 3, 4, 5, 6]
min_samples_leafs = [1, 3, 5, 10]

# Track best parameters for entropy criterion
best_entropy_score = -1
best_entropy_params = None
best_entropy_test_acc = 0

# Track best parameters for Gini criterion
best_gini_score = -1
best_gini_params = None
best_gini_test_acc = 0

# Grid search for entropy criterion
for depth in max_depths:
    for minleaf in min_samples_leafs:
        tree = DecisionTree(criterion='entropy', max_depth=depth, min_samples_leaf=minleaf)
        tree.fit(X_train, y_train)
        val_acc = tree.accuracy(X_val, y_val)
        
        # Update best parameters if validation accuracy improves
        if val_acc > best_entropy_score:
            best_entropy_score = val_acc
            best_entropy_params = (depth, minleaf)
            test_acc = tree.accuracy(X_test, y_test)
            best_entropy_test_acc = test_acc

# Grid search for Gini criterion
for depth in max_depths:
    for minleaf in min_samples_leafs:
        tree = DecisionTree(criterion='gini', max_depth=depth, min_samples_leaf=minleaf)
        tree.fit(X_train, y_train)
        val_acc = tree.accuracy(X_val, y_val)
        
        # Update best parameters if validation accuracy improves
        if val_acc > best_gini_score:
            best_gini_score = val_acc
            best_gini_params = (depth, minleaf)
            test_acc = tree.accuracy(X_test, y_test)
            best_gini_test_acc = test_acc

# Display results
print(f"Best (entropy): depth={best_entropy_params[0]}, minleaf={best_entropy_params[1]}, val_acc={best_entropy_score:.2f}, test_acc={best_entropy_test_acc:.2f}")
print(f"Best (gini):    depth={best_gini_params[0]}, minleaf={best_gini_params[1]}, val_acc={best_gini_score:.2f}, test_acc={best_gini_test_acc:.2f}")