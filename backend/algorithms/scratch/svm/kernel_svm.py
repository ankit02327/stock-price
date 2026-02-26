"""
Complete Kernel SVM Implementation with SMO

This module implements a comprehensive kernel SVM classifier using:
- Sequential Minimal Optimization (SMO) algorithm
- Multiple kernel functions (RBF, Polynomial, Linear)
- Grid search for hyperparameter tuning
- Support vector identification
- Binary classification with decision boundaries

Mathematical Foundation:
- RBF Kernel: K(x, x') = exp(-||x - x'||² / (2σ²))
- Polynomial Kernel: K(x, x') = (x^T * x' + 1)^d
- Decision Function: f(x) = Σ αᵢ yᵢ K(x, xᵢ) + b
- SMO Algorithm: Optimize dual problem with KKT conditions
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
# Exercise 2: Kernel SVM using Simplified SMO
#########################

class KernelSVM:
    """
    Kernel Support Vector Machine using Simplified SMO
    
    Features:
    - Multiple kernel functions (RBF, Polynomial, Linear)
    - Sequential Minimal Optimization (SMO)
    - Support vector identification
    - Binary classification
    """
    
    def __init__(self, C=1.0, kernel='rbf', sigma=1.0, degree=3, max_passes=10, tol=1e-3):
        """
        Initialize Kernel SVM
        
        Parameters:
        - C: regularization parameter
        - kernel: kernel type ('rbf', 'polynomial', 'linear')
        - sigma: RBF kernel parameter
        - degree: polynomial kernel degree
        - max_passes: maximum SMO passes
        - tol: convergence tolerance
        """
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.max_passes = max_passes
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None
    
    def kernel_function(self, xi, xj):
        """
        Compute kernel function between two points
        
        Parameters:
        - xi, xj: feature vectors
        
        Returns:
        - kernel value
        """
        if self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(xi - xj)**2 / (2 * self.sigma**2))
        elif self.kernel == 'polynomial':
            return (np.dot(xi, xj) + 1)**self.degree
        else:  # linear
            return np.dot(xi, xj)
    
    def compute_kernel_matrix(self, X1, X2):
        """
        Compute kernel matrix between two sets of points
        
        Parameters:
        - X1, X2: feature matrices
        
        Returns:
        - kernel matrix
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_function(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        """
        Train the Kernel SVM model using SMO
        
        Parameters:
        - X: training features
        - y: training labels (-1 or 1)
        """
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        self.K = self.compute_kernel_matrix(X, X)
        
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Compute Ei = f(xi) - yi
                Ei = np.sum(self.alpha * y * self.K[:, i]) + self.b - y[i]
                
                # Check KKT conditions
                if ((y[i] * Ei < -self.tol and self.alpha[i] < self.C) or 
                    (y[i] * Ei > self.tol and self.alpha[i] > 0)):
                    
                    # Select j randomly (simplified SMO)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Compute Ej
                    Ej = np.sum(self.alpha * y * self.K[:, j]) + self.b - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] = alpha_j_old - y[j] * (Ei - Ej) / (eta + 1e-12)
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Compute b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        - X: test features
        
        Returns:
        - predictions: predicted class labels (-1 or 1)
        """
        K_test = self.compute_kernel_matrix(X, self.X)
        predictions = np.dot(K_test, self.alpha * self.y) + self.b
        return np.sign(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_support_vectors(self):
        """Get indices of support vectors"""
        support_indices = np.where(self.alpha > 1e-5)[0]
        return support_indices

# Train RBF kernel SVMs with grid search
print("\n=== Exercise 2: RBF Kernel SVM with SMO ===")
print("Training RBF SVMs over C and sigma grid...")

C_values = [0.01, 0.1, 1, 10, 100]
sigma_values = [0.1, 0.3, 1, 3]

best_rbf_val_acc = 0
best_rbf_C = None
best_rbf_sigma = None
best_rbf_model = None

for C in C_values:
    for sigma in sigma_values:
        model = KernelSVM(C=C, kernel='rbf', sigma=sigma, max_passes=5)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        print(f"RBF C={C}, sigma={sigma}: val_acc={val_acc:.2f}")
        
        if val_acc > best_rbf_val_acc:
            best_rbf_val_acc = val_acc
            best_rbf_C = C
            best_rbf_sigma = sigma
            best_rbf_model = model

# Evaluate best RBF model on test set
rbf_test_acc = best_rbf_model.score(X_test, y_test)
print(f"\nSelected RBF SVM: C={best_rbf_C}, sigma={best_rbf_sigma:.1f}, val_acc={best_rbf_val_acc:.2f}, test_acc={rbf_test_acc:.2f}")

# Store RBF results
rbf_results = {
    'C': best_rbf_C,
    'sigma': best_rbf_sigma,
    'val_acc': best_rbf_val_acc,
    'test_acc': rbf_test_acc,
    'model': best_rbf_model
}

# Train polynomial kernel SVMs
print("\n=== Polynomial Kernel SVM (degree=3) ===")
print("Training polynomial (degree=3) SVMs over C grid...")

C_values = [0.01, 0.1, 1, 10, 100]

best_poly_val_acc = 0
best_poly_C = None
best_poly_model = None

for C in C_values:
    model = KernelSVM(C=C, kernel='polynomial', degree=3, max_passes=5)
    model.fit(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"Poly C={C}: val_acc={val_acc:.2f}")
    
    if val_acc > best_poly_val_acc:
        best_poly_val_acc = val_acc
        best_poly_C = C
        best_poly_model = model

# Evaluate best polynomial model on test set
poly_test_acc = best_poly_model.score(X_test, y_test)
print(f"\nSelected polynomial SVM (degree=3): C={best_poly_C}, val_acc={best_poly_val_acc:.2f}, test_acc={poly_test_acc:.2f}")

# Store polynomial results
poly_results = {
    'C': best_poly_C,
    'val_acc': best_poly_val_acc,
    'test_acc': poly_test_acc,
    'model': best_poly_model
}

#########################
# Final Summary
#########################

print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"RBF SVM: C={rbf_results['C']}, sigma={rbf_results['sigma']:.2f}, val_acc={rbf_results['val_acc']:.2f}, test_acc={rbf_results['test_acc']:.2f}")
print(f"Poly SVM (deg=3): C={poly_results['C']}, val_acc={poly_results['val_acc']:.2f}, test_acc={poly_results['test_acc']:.2f}")
print("="*50)

#########################
# Visualization
#########################

def plot_decision_boundary(X, y, model, title, filename, is_kernel=True):
    """
    Plot decision boundary for Kernel SVM model
    
    Parameters:
    - X: feature matrix
    - y: target vector
    - model: trained Kernel SVM model
    - title: plot title
    - filename: output filename
    - is_kernel: whether model is kernel SVM
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
    
    # Highlight support vectors if kernel SVM
    if is_kernel:
        support_indices = model.get_support_vectors()
        if len(support_indices) > 0:
            plt.scatter(X[support_indices, 0], X[support_indices, 1],
                       s=200, facecolors='none', edgecolors='green',
                       linewidths=2, label=f'Support vectors ({len(support_indices)})')
    
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    if is_kernel:
        plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# Plot RBF SVM decision boundary
plot_decision_boundary(X_train, y_train, rbf_results['model'],
                      f"RBF SVM Decision Boundary (C={rbf_results['C']}, σ={rbf_results['sigma']:.1f})",
                      'rbf_decision_boundary.png', is_kernel=True)

print("\nKernel SVM implementation completed successfully!")