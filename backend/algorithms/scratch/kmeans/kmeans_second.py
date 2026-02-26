"""
K-Means++ Clustering Implementation from Scratch

This module implements the K-Means++ clustering algorithm using:
- K-Means++ initialization (better than random)
- Euclidean distance metric
- Iterative centroid updates
- Elbow method for optimal K selection

Mathematical Foundation:
- K-Means++: Choose centroids with probability ∝ distance² to nearest centroid
- Distance: d(x, c) = √(Σ(xᵢ - cᵢ)²)
- Centroid Update: c = (1/|S|) * Σ xᵢ for xᵢ ∈ S
- Elbow Method: Find K where WCSS decrease rate slows down
"""

import numpy as np
import pandas as pd

#########################
# K-Means++ Implementation
#########################

class KMeansPlusPlus:
    """
    K-Means++ Clustering Algorithm
    
    Features:
    - K-Means++ initialization for better starting points
    - Euclidean distance metric
    - Iterative centroid updates
    - Convergence detection
    """
    
    def __init__(self, K=3, max_iters=100, random_state=0):
        """
        Initialize K-Means++ algorithm
        
        Parameters:
        - K: number of clusters
        - max_iters: maximum number of iterations
        - random_state: random seed for reproducibility
        """
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def kmeans_plus_plus_init(self, X):
        """
        K-Means++ initialization for better starting centroids
        
        Algorithm:
        1. Choose first centroid randomly
        2. For each remaining centroid:
           - Compute distance to nearest existing centroid
           - Choose with probability ∝ distance²
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.randint(n_samples)]]
        
        # Choose remaining K-1 centroids
        for _ in range(1, self.K):
            # Compute distances to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            
            # Clip small negative values to avoid numeric errors
            distances = np.clip(distances, 0, None)
            
            # Choose next centroid with probability proportional to distance squared
            if distances.sum() > 0:
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                
                for idx, cum_prob in enumerate(cumulative_probs):
                    if r < cum_prob:
                        centroids.append(X[idx])
                        break
            else:
                # All points are already centroids, pick random
                centroids.append(X[np.random.randint(n_samples)])
        
        return np.array(centroids)
    
    def assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Re-seed empty cluster to a random training point
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def fit(self, X):
        """Train K-Means on dataset X"""
        # K-Means++ initialization
        self.centroids = self.kmeans_plus_plus_init(X)
        
        prev_labels = None
        for iteration in range(self.max_iters):
            # Assignment step
            labels = self.assign_clusters(X, self.centroids)
            
            # Check for convergence (unchanged cluster assignments)
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                self.labels = labels
                break
            
            prev_labels = labels.copy()
            
            # Update step
            self.centroids = self.update_centroids(X, labels)
            self.labels = labels
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)
    
    def compute_wcss(self, X):
        """Compute Within-Cluster Sum of Squares"""
        labels = self.predict(X)
        wcss = 0
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                squared_dists = np.linalg.norm(cluster_points - self.centroids[k], axis=1)**2
                squared_dists = np.clip(squared_dists, 0, None)
                wcss += np.sum(squared_dists)
        return wcss
    
    def compute_J(self, X):
        """Compute average cost J = (1/m) * WCSS"""
        wcss = self.compute_wcss(X)
        return wcss / X.shape[0]

def find_elbow_kneedle(wcss_values):
    """
    Find elbow point using kneedle rule
    Detects the point where the rate of WCSS decrease slows down most
    (maximum change in slope = maximum second derivative)
    """
    if len(wcss_values) < 3:
        return 1
    
    # Compute first derivative (rate of change)
    first_deriv = np.diff(wcss_values)
    
    # Compute second derivative (change in slope)
    second_deriv = np.diff(first_deriv)
    
    # Find K with maximum absolute change in slope
    elbow_idx = np.argmax(np.abs(second_deriv))
    
    # Return K (accounting for index offset)
    return elbow_idx + 2

#########################
# Main Program
#########################

# Read input
line1 = input().strip()
line2 = input().strip()

Kmax = int(line1.split('=')[1])
iterations = int(line2.split('=')[1])

# Load datasets
train_data = pd.read_csv('train.csv', header=None)
val_data = pd.read_csv('val.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)

X_train = train_data.values
X_val = val_data.values
X_test = test_data.values

# Compute WCSS for all K values on validation set
wcss_val = []

for K in range(1, Kmax + 1):
    kmeans = KMeansPlusPlus(K=K, max_iters=iterations, random_state=0)
    kmeans.fit(X_train)
    wcss_k = kmeans.compute_wcss(X_val)
    wcss_val.append(wcss_k)

# Find elbow point
elbow_K = find_elbow_kneedle(wcss_val)

# Retrain with elbow K and evaluate on all datasets
kmeans_final = KMeansPlusPlus(K=elbow_K, max_iters=iterations, random_state=0)
kmeans_final.fit(X_train)

# Compute metrics for all datasets
train_J = kmeans_final.compute_J(X_train)
train_WCSS = kmeans_final.compute_wcss(X_train)
val_J = kmeans_final.compute_J(X_val)
val_WCSS = kmeans_final.compute_wcss(X_val)
test_J = kmeans_final.compute_J(X_test)
test_WCSS = kmeans_final.compute_wcss(X_test)

# Format centroids
centroids_str = ";".join([f"({c[0]:.2f},{c[1]:.2f})" for c in kmeans_final.centroids])

# Output (exact format as required by VPL)
print(f"Elbow_K={elbow_K}")
print(f"WCSS_Val={[round(w, 2) for w in wcss_val]}")
print(f"Train_J={train_J:.2f}")
print(f"Train_WCSS={train_WCSS:.2f}")
print(f"Val_J={val_J:.2f}")
print(f"Val_WCSS={val_WCSS:.2f}")
print(f"Test_J_Elbow={test_J:.2f}")
print(f"Test_WCSS_Elbow={test_WCSS:.2f}")
print(f"Centroids=[{centroids_str}]")