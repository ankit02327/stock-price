"""
Complete Linear SVM Implementation with SGD

This module implements a comprehensive linear SVM classifier using:
- Stochastic Gradient Descent (SGD) optimization
- Hinge loss function with L2 regularization
- Grid search for hyperparameter tuning
- Binary classification with decision boundaries

Mathematical Foundation:
- Hinge Loss: L(y, f(x)) = max(0, 1 - y * f(x))
- Decision Function: f(x) = w^T * x + b
- SGD Update: w = w - η * (w - C * y * x) if margin < 1
- Margin: y * (w^T * x + b)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

#########################
# Load Datasets
#########################

train_data = pd.read_csv('svm_train-1.csv')
val_data = pd.read_csv('svm_val-1.csv')
test_data = pd.read_csv('svm_test-1.csv')

X_train = train_data[['x1', 'x2']].values
y_train = train_data['y'].values
X_val = val_data[['x1', 'x2']].values
y_val = val_data['y'].values
X_test = test_data[['x1', 'x2']].values
y_test = test_data['y'].values

print("Dataset loaded successfully")
print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

#########################
# Exercise 1: Linear SVM using SGD
#########################

class LinearSVM:
    """
    Linear Support Vector Machine using Stochastic Gradient Descent
    
    Features:
    - Hinge loss optimization
    - L2 regularization
    - Stochastic gradient descent
    - Binary classification
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000):
        """
        Initialize Linear SVM
        
        Parameters:
        - C: regularization parameter
        - learning_rate: step size for SGD
        - n_epochs: number of training epochs
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """
        Train the Linear SVM model using SGD
        
        Parameters:
        - X: training features
        - y: training labels (-1 or 1)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                
                # Calculate margin: y * (w^T * x + b)
                margin = yi * (np.dot(self.w, xi) + self.b)
                
                # Update weights using hinge loss gradient
                if margin < 1:
                    # Misclassified or within margin
                    self.w = self.w - self.learning_rate * (self.w - self.C * yi * xi)
                    self.b = self.b - self.learning_rate * (-self.C * yi)
                else:
                    # Correctly classified outside margin
                    self.w = self.w - self.learning_rate * self.w
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        - X: test features
        
        Returns:
        - predictions: predicted class labels (-1 or 1)
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Train linear SVMs with different C values
print("\n=== Exercise 1: Linear SVM with SGD ===")
print("Training linear SVMs over C grid...")

C_values = [0.01, 0.1, 1, 10, 100]
best_val_acc = 0
best_C = None
best_model = None

for C in C_values:
    model = LinearSVM(C=C, learning_rate=0.001, n_epochs=1000)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"Linear C={C}: val_acc={val_acc:.2f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_C = C
        best_model = model

# Evaluate best model on test set
test_acc = best_model.score(X_test, y_test)
print(f"\nSelected Linear SVM: C={best_C}, val_acc={best_val_acc:.2f}, test_acc={test_acc:.2f}")

# Store results
linear_results = {
    'C': best_C,
    'val_acc': best_val_acc,
    'test_acc': test_acc,
    'model': best_model
}

#########################
# Visualization
#########################

def plot_decision_boundary(X, y, model, title, filename):
    """
    Plot decision boundary for Linear SVM model
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - model: trained Linear SVM model
    - title: plot title
    - filename: output filename
    """
    # Create mesh grid
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.contour(xx, yy, Z, colors='k', linewidths=1, levels=[0])
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue']),
                edgecolors='k', s=50, alpha=0.7)
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# Plot Linear SVM decision boundary
plot_decision_boundary(X_train, y_train, linear_results['model'], 
                      f"Linear SVM Decision Boundary (C={linear_results['C']})",
                      'linear_decision_boundary.png')

print("\nLinear SVM implementation completed successfully!")