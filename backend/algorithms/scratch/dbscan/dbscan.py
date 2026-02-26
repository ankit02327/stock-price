"""
DBSCAN (Density-Based Spatial Clustering) Implementation from Scratch

This module implements the DBSCAN clustering algorithm using:
- Density-based clustering
- Core points, border points, and noise points
- Epsilon-neighborhood and minimum points criteria
- Cluster expansion through density-reachable points

Mathematical Foundation:
- Epsilon-neighborhood: N_eps(p) = {q ∈ D | dist(p,q) ≤ eps}
- Core point: |N_eps(p)| ≥ min_pts
- Density-reachable: q is density-reachable from p if there exists a chain
- Cluster: maximal set of density-connected points
"""

import numpy as np

#########################
# DBSCAN Implementation
#########################

class DBSCAN:
    """
    DBSCAN Clustering Algorithm
    
    Features:
    - Density-based clustering
    - Handles noise points
    - No need to specify number of clusters
    - Finds clusters of arbitrary shape
    """
    
    def __init__(self, eps=0.5, min_pts=5):
        """
        Initialize DBSCAN algorithm
        
        Parameters:
        - eps: neighborhood radius (epsilon)
        - min_pts: minimum points to form a dense region
        """
        self.eps = eps
        self.min_pts = min_pts
        self.labels = None
        self.core_points = None
        
    def compute_distance_matrix(self, X):
        """Compute pairwise Euclidean distance matrix"""
        n = X.shape[0]
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def get_neighbors(self, point_idx, dist_matrix):
        """Get epsilon-neighborhood of a point"""
        neighbors = np.where(dist_matrix[point_idx] <= self.eps)[0]
        return neighbors
    
    def fit(self, X):
        """
        Perform DBSCAN clustering
        
        Returns:
        - labels: cluster assignments (0 = noise, 1+ = cluster ID)
        """
        n_points = X.shape[0]
        
        # Compute distance matrix once
        dist_matrix = self.compute_distance_matrix(X)
        
        # Initialize labels (0 = unvisited/noise)
        self.labels = np.zeros(n_points, dtype=int)
        
        # Find all neighbors for each point
        neighbors_list = [self.get_neighbors(i, dist_matrix) for i in range(n_points)]
        
        # Identify core points
        self.core_points = []
        for i in range(n_points):
            if len(neighbors_list[i]) >= self.min_pts:
                self.core_points.append(i)
        
        # Cluster assignment
        cluster_id = 0
        visited = np.zeros(n_points, dtype=bool)
        
        for point_idx in range(n_points):
            # Skip if already visited
            if visited[point_idx]:
                continue
            
            visited[point_idx] = True
            
            # Check if it's a core point
            if point_idx not in self.core_points:
                continue
            
            # Start a new cluster
            cluster_id += 1
            self.labels[point_idx] = cluster_id
            
            # Expand cluster using BFS
            seeds = list(neighbors_list[point_idx])
            
            i = 0
            while i < len(seeds):
                neighbor_idx = seeds[i]
                
                # Mark as visited
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    
                    # If it's a core point, add its neighbors to seeds
                    if neighbor_idx in self.core_points:
                        neighbor_neighbors = neighbors_list[neighbor_idx]
                        for nn in neighbor_neighbors:
                            if nn not in seeds:
                                seeds.append(nn)
                
                # Assign to cluster if not yet assigned
                if self.labels[neighbor_idx] == 0:
                    self.labels[neighbor_idx] = cluster_id
                
                i += 1
        
        return self.labels
    
    def get_neighborhood_counts(self, X):
        """Get neighborhood count for each point"""
        dist_matrix = self.compute_distance_matrix(X)
        counts = []
        for i in range(X.shape[0]):
            neighbors = self.get_neighbors(i, dist_matrix)
            counts.append(len(neighbors))
        return counts
    
    def get_noise_ratio(self):
        """Compute ratio of noise points"""
        if self.labels is None:
            return 0.0
        noise_count = np.sum(self.labels == 0)
        return noise_count / len(self.labels)

#########################
# Main Program
#########################

# Read input
n = int(input())
X = []
for _ in range(n):
    coords = list(map(float, input().split()))
    X.append(coords)
X = np.array(X)

eps = float(input())
min_pts = int(input())

# Run DBSCAN
dbscan = DBSCAN(eps=eps, min_pts=min_pts)
labels = dbscan.fit(X)

# Output core points
core_points_str = " ".join([str(i+1) for i in dbscan.core_points])
print(f"Core points: {core_points_str}")

# Output cluster assignments
cluster_str = " ".join([str(label) for label in labels])
print(f"Cluster Assignments: {cluster_str}")

# Output noise ratio
noise_ratio = dbscan.get_noise_ratio()
print(f"Noise Ratio = {noise_ratio:.2f}")