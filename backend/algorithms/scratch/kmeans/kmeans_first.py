"""
K-Means Clustering Implementation from Scratch

This module implements the K-Means clustering algorithm using:
- Euclidean distance metric
- Random centroid initialization
- Iterative centroid updates
- Convergence detection
- Elbow method for optimal K selection
- Silhouette score evaluation

Mathematical Foundation:
- Distance: d(x, c) = √(Σ(xᵢ - cᵢ)²)
- Centroid Update: c = (1/|S|) * Σ xᵢ for xᵢ ∈ S
- Objective: minimize Σ Σ d(xᵢ, cⱼ)²
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

#########################
# Load Datasets
#########################

train_data = pd.read_csv('train.csv', header=None, names=['x1', 'x2'])
val_data = pd.read_csv('val.csv', header=None, names=['x1', 'x2'])
test_data = pd.read_csv('test.csv', header=None, names=['x1', 'x2'])

X_train = train_data.values
X_val = val_data.values
X_test = test_data.values

print("=" * 60)
print("K-MEANS CLUSTERING LAB ASSIGNMENT")
print("=" * 60)
print(f"Training data: {X_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Test data: {X_test.shape}")

#########################
# K-Means Implementation
#########################

class KMeans:
    def __init__(self, K=3, max_iters=100, random_state=42):
        """
        K-Means clustering implementation from scratch
        
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
        self.J_history = []
    
    def initialize_centroids(self, X):
        """Randomly select K data points as initial centroids"""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        return X[indices].copy()
    
    def compute_distances(self, X, centroids):
        """Compute Euclidean distance from each point to each centroid"""
        distances = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return distances
    
    def assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid"""
        distances = self.compute_distances(X, centroids)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            else:
                # Re-seed empty cluster to a random point
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids
    
    def compute_cost(self, X, labels, centroids):
        """
        Compute K-Means objective J
        J = (1/m) * sum(||x(i) - mu_c(i)||^2)
        """
        cost = 0
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                cost += np.sum(np.linalg.norm(cluster_points - centroids[k], axis=1)**2)
        return cost / X.shape[0]
    
    def fit(self, X):
        """Train K-Means on dataset X"""
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        self.J_history = []
        
        for iteration in range(self.max_iters):
            # Assignment step
            labels = self.assign_clusters(X, self.centroids)
            
            # Compute cost
            cost = self.compute_cost(X, labels, self.centroids)
            self.J_history.append(cost)
            
            # Update step
            new_centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                self.labels = labels
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids = new_centroids
            self.labels = labels
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)
    
    def get_wcss(self, X):
        """Compute Within-Cluster Sum of Squares"""
        labels = self.predict(X)
        wcss = 0
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum(np.linalg.norm(cluster_points - self.centroids[k], axis=1)**2)
        return wcss

def compute_silhouette_score(X, labels, centroids):
    """
    Compute average silhouette score for clustering
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
    - a(i) = mean distance to points in same cluster
    - b(i) = min mean distance to points in other clusters
    """
    n = X.shape[0]
    K = len(centroids)
    silhouette_scores = np.zeros(n)
    
    for i in range(n):
        cluster_i = labels[i]
        
        # Points in same cluster
        same_cluster_mask = labels == cluster_i
        same_cluster_points = X[same_cluster_mask]
        
        # a(i): mean distance to points in same cluster
        if len(same_cluster_points) > 1:
            a_i = np.mean(np.linalg.norm(same_cluster_points - X[i], axis=1))
        else:
            a_i = 0
        
        # b(i): min mean distance to points in other clusters
        b_i = np.inf
        for k in range(K):
            if k != cluster_i:
                other_cluster_mask = labels == k
                other_cluster_points = X[other_cluster_mask]
                if len(other_cluster_points) > 0:
                    mean_dist = np.mean(np.linalg.norm(other_cluster_points - X[i], axis=1))
                    b_i = min(b_i, mean_dist)
        
        # Silhouette coefficient
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)

#########################
# Train and Evaluate
#########################

# Try different values of K
K_values = range(2, 11)
train_wcss = []
val_wcss = []
silhouette_scores = []

print("\\n" + "=" * 60)
print("TRAINING K-MEANS FOR DIFFERENT K VALUES")
print("=" * 60)

for K in K_values:
    kmeans = KMeans(K=K, max_iters=100, random_state=42)
    kmeans.fit(X_train)
    
    train_wcss_k = kmeans.get_wcss(X_train)
    val_wcss_k = kmeans.get_wcss(X_val)
    
    train_wcss.append(train_wcss_k)
    val_wcss.append(val_wcss_k)
    
    # Compute silhouette score on validation set
    val_labels = kmeans.predict(X_val)
    silhouette = compute_silhouette_score(X_val, val_labels, kmeans.centroids)
    silhouette_scores.append(silhouette)
    
    print(f"K={K}: Train WCSS={train_wcss_k:.2f}, Val WCSS={val_wcss_k:.2f}, Silhouette={silhouette:.4f}")

# Find optimal K
optimal_K_wcss = K_values[np.argmin(val_wcss)]
optimal_K_silhouette = K_values[np.argmax(silhouette_scores)]

print(f"\\n{'='*60}")
print(f"Optimal K by Validation WCSS: {optimal_K_wcss}")
print(f"Optimal K by Silhouette Score: {optimal_K_silhouette}")
print(f"Recommended K: {optimal_K_silhouette}")
print(f"{'='*60}")

#########################
# Train Final Model
#########################

print(f"\\nTraining final model with K={optimal_K_silhouette}...")
kmeans_final = KMeans(K=optimal_K_silhouette, max_iters=100, random_state=42)
kmeans_final.fit(X_train)

# Evaluate on all datasets
train_labels = kmeans_final.labels
val_labels = kmeans_final.predict(X_val)
test_labels = kmeans_final.predict(X_test)

train_wcss_final = kmeans_final.get_wcss(X_train)
val_wcss_final = kmeans_final.get_wcss(X_val)
test_wcss_final = kmeans_final.get_wcss(X_test)

train_silhouette = compute_silhouette_score(X_train, train_labels, kmeans_final.centroids)
val_silhouette = compute_silhouette_score(X_val, val_labels, kmeans_final.centroids)
test_silhouette = compute_silhouette_score(X_test, test_labels, kmeans_final.centroids)

print("\\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"\\nNumber of clusters: {optimal_K_silhouette}")
print(f"Final training cost J: {kmeans_final.J_history[-1]:.2f}")
print(f"Convergence iterations: {len(kmeans_final.J_history)}")

print(f"\\nTraining WCSS: {train_wcss_final:.2f}")
print(f"Validation WCSS: {val_wcss_final:.2f}")
print(f"Test WCSS: {test_wcss_final:.2f}")

print(f"\\nTraining Silhouette Score: {train_silhouette:.4f}")
print(f"Validation Silhouette Score: {val_silhouette:.4f}")
print(f"Test Silhouette Score: {test_silhouette:.4f}")

print("\\nFinal Centroids:")
for k in range(kmeans_final.K):
    print(f"Cluster {k+1}: ({kmeans_final.centroids[k, 0]:.2f}, {kmeans_final.centroids[k, 1]:.2f})")

unique, counts = np.unique(train_labels, return_counts=True)
print(f"\\nCluster sizes: {counts.tolist()}")

# Visualization code removed for cleaner implementation

print("\\n" + "=" * 60)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 60)