"""
Principal Component Analysis (PCA) Implementation from Scratch

This module implements PCA for dimensionality reduction using:
- Data centering and covariance matrix computation
- Eigenvalue decomposition for principal components
- Variance explained calculation
- Dimensionality reduction with optimal component selection

Mathematical Foundation:
- Covariance Matrix: C = (1/n) * X^T * X
- Eigenvalue Decomposition: C = V * Λ * V^T
- Principal Components: PC = X * V
- Explained Variance: λᵢ / Σλⱼ
"""

import numpy as np

#########################
# PCA Implementation
#########################
class PCA:
    """
    Principal Component Analysis for Dimensionality Reduction
    
    Features:
    - Data centering and standardization
    - Covariance matrix computation
    - Eigenvalue decomposition
    - Variance explained calculation
    - Dimensionality reduction
    """
    
    def __init__(self, n_components):
        """
        Initialize PCA
        
        Parameters:
        - n_components: number of principal components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Fit PCA on the data X
        
        Algorithm:
        1. Center data by subtracting mean
        2. Compute covariance matrix
        3. Perform eigenvalue decomposition
        4. Sort eigenvalues/eigenvectors by variance
        5. Select top n_components
        
        Parameters:
        - X: input data matrix (n_samples, n_features)
        """
        # 1. Center data by subtracting mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute covariance matrix: C = (1/n) * X^T * X
        cov = np.cov(X_centered, rowvar=False)

        # 3. Eigenvalue decomposition: C = V * Λ * V^T
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 4. Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # 5. Keep top n_components with highest variance
        self.explained_variance_ = eigvals[:self.n_components]
        self.components_ = eigvecs[:, :self.n_components].T
        return self

    def transform(self, X):
        """
        Project X onto principal components
        
        Parameters:
        - X: input data matrix (n_samples, n_features)
        
        Returns:
        - X_transformed: projected data (n_samples, n_components)
        """
        # Center the data using training mean
        X_centered = X - self.mean_
        # Project onto principal components: Y = X * V^T
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        """
        Fit PCA and return transformed data in one step
        
        Parameters:
        - X: input data matrix (n_samples, n_features)
        
        Returns:
        - X_transformed: projected data (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

#########################
# Example Usage
#########################
if __name__ == "__main__":
    print("=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA) - FROM SCRATCH")
    print("=" * 60)

    # Generate synthetic data (3 clusters in 3D)
    np.random.seed(0)
    C1 = np.random.randn(100, 3) + np.array([5, 5, 5])
    C2 = np.random.randn(100, 3) + np.array([-5, -5, 0])
    C3 = np.random.randn(100, 3) + np.array([5, -5, 5])
    X = np.vstack([C1, C2, C3])

    print(f"Original data shape: {X.shape}")

    # Perform PCA, reduce to 2D
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print(f"Transformed data shape: {X_transformed.shape}")

    # Calculate explained variance ratio
    total_variance = np.sum(pca.explained_variance_)
    all_eigenvalues = np.linalg.eigvalsh(np.cov(X - pca.mean_, rowvar=False))
    total_all_variance = np.sum(all_eigenvalues)
    explained_variance_ratio = total_variance / total_all_variance

    print(f"Explained variance by {pca.n_components} components: {explained_variance_ratio*100:.2f}%")
    print(f"Individual component variances: {pca.explained_variance_}")
    print(f"Principal components shape: {pca.components_.shape}")

    print("\nPCA implementation complete!")
