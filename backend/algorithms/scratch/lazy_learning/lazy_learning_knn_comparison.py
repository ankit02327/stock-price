"""
Lazy Learning vs Eager Learning Comparison

This module compares K-Nearest Neighbors (lazy learner) with Logistic Regression (eager learner).
It demonstrates the fundamental differences between these two learning paradigms:

Lazy Learning (KNN):
- No explicit training phase
- All computation deferred to prediction time
- Stores all training data
- Decision boundary adapts to local data density

Eager Learning (Logistic Regression):
- Explicit training phase to learn parameters
- Fast prediction after training
- Discards training data after learning
- Global decision boundary
"""

import numpy as np
import time

#########################
# K-Nearest Neighbors (Lazy Learner)
#########################
def euclidean_dist(A, B):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(A - B)

class KNN:
    """
    K-Nearest Neighbors classifier (Lazy Learner)
    
    Lazy learning characteristics:
    - No explicit training phase
    - All computation happens during prediction
    - Stores all training data
    """
    
    def __init__(self, K=5):
        """
        Initialize KNN classifier
        
        Parameters:
        - K: number of nearest neighbors to consider
        """
        self.K = K

    def fit(self, X, y):
        """
        'Training' phase - just store the data (lazy learning)
        
        Parameters:
        - X: training features
        - y: training labels
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict class labels for test points
        
        Parameters:
        - X: test features
        
        Returns:
        - y_pred: predicted class labels
        """
        n = X.shape[0]
        y_pred = np.zeros(n, dtype=int)
        
        for i in range(n):
            # Compute distances to all training points
            dists = np.array([euclidean_dist(X[i], xt) for xt in self.X_train])
            
            # Find K nearest neighbors
            idx = np.argsort(dists)[:self.K]
            
            # Majority voting among k nearest neighbors
            labels, counts = np.unique(self.y_train[idx], return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
            
        return y_pred

#########################
# Logistic Regression (Eager Learner)
#########################
class LogisticRegressionScratch:
    """
    Logistic Regression classifier (Eager Learner)
    
    Eager learning characteristics:
    - Explicit training phase to learn parameters
    - Fast prediction after training
    - Discards training data after learning
    - Global decision boundary
    """
    
    def __init__(self, lr=0.1, epochs=1000):
        """
        Initialize Logistic Regression classifier
        
        Parameters:
        - lr: learning rate for gradient descent
        - epochs: number of training iterations
        """
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Parameters:
        - X: training features
        - y: training labels
        """
        m, n = X.shape
        
        # Add intercept term (bias)
        Xb = np.hstack([np.ones((m, 1)), X])
        
        # Initialize weights
        self.w = np.zeros(n + 1)
        
        # Gradient descent optimization
        for _ in range(self.epochs):
            # Forward pass: compute predictions
            z = Xb.dot(self.w)
            h = self.sigmoid(z)
            
            # Compute gradient
            grad = (1/m) * Xb.T.dot(h - y)
            
            # Update weights
            self.w -= self.lr * grad

    def predict(self, X):
        """
        Predict class labels for test points
        
        Parameters:
        - X: test features
        
        Returns:
        - y_pred: predicted class labels (0 or 1)
        """
        m = X.shape[0]
        Xb = np.hstack([np.ones((m, 1)), X])
        probs = self.sigmoid(Xb.dot(self.w))
        return (probs >= 0.5).astype(int)

#########################
# Main Comparison
#########################
if __name__ == "__main__":
    # Generate synthetic 2D binary classification data
    np.random.seed(0)
    N = 500
    
    # Create two clusters: positive class around [2,2], negative class around [-2,-2]
    X_pos = np.random.randn(N//2, 2) + [2, 2]
    X_neg = np.random.randn(N//2, 2) + [-2, -2]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*(N//2) + [0]*(N//2))

    # Shuffle and split data
    idx = np.random.permutation(N)
    X, y = X[idx], y[idx]
    split = int(0.7 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("==== Eager Learner: Logistic Regression ====")
    lr = LogisticRegressionScratch(lr=0.1, epochs=500)
    
    # Measure training time
    start = time.time()
    lr.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Measure prediction time
    start = time.time()
    y_pred_lr = lr.predict(X_test)
    pred_time = time.time() - start
    
    # Calculate accuracy
    acc_lr = np.mean(y_pred_lr == y_test)
    print(f"Train time: {train_time:.4f}s, Predict time: {pred_time:.4f}s, Accuracy: {acc_lr*100:.2f}%\n")

    print("==== Lazy Learner: K-Nearest Neighbors ====")
    knn = KNN(K=5)
    
    # Measure 'training' time (just storing data)
    start = time.time()
    knn.fit(X_train, y_train)
    train_time_knn = time.time() - start
    
    # Measure prediction time (all computation happens here)
    start = time.time()
    y_pred_knn = knn.predict(X_test)
    pred_time_knn = time.time() - start
    
    # Calculate accuracy
    acc_knn = np.mean(y_pred_knn == y_test)
    print(f"Train time: {train_time_knn:.4f}s, Predict time: {pred_time_knn:.4f}s, Accuracy: {acc_knn*100:.2f}%\n")
    
    # Display comparison summary
    print("==== Comparison Summary ====")
    print(f"Logistic Regression: Fast training ({train_time:.4f}s), Fast prediction ({pred_time:.4f}s)")
    print(f"KNN: Instant training ({train_time_knn:.4f}s), Slow prediction ({pred_time_knn:.4f}s)")
    print(f"Accuracy: LR={acc_lr*100:.2f}%, KNN={acc_knn*100:.2f}%")
