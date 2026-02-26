"""
Singular Value Decomposition (SVD) Implementation from Scratch

This module implements SVD for matrix factorization using:
- Eigenvalue decomposition of covariance matrices
- Singular value computation and sorting
- Matrix reconstruction and approximation
- Dimensionality reduction capabilities

Mathematical Foundation:
- SVD: A = U * Σ * V^T
- U: left singular vectors (m x m)
- Σ: singular values diagonal matrix (m x n)
- V^T: right singular vectors (n x n)
- Truncated SVD: A ≈ U_k * Σ_k * V_k^T
"""

import numpy as np

#########################
# SVD Implementation
#########################
class SVD:
    """
    Singular Value Decomposition for Matrix Factorization
    
    Features:
    - Full and truncated SVD computation
    - Matrix reconstruction and approximation
    - Dimensionality reduction
    - Numerical stability optimization
    """
    
    def __init__(self, k=None):
        """
        Initialize SVD
        
        Parameters:
        - k: number of singular values/vectors to keep (for truncated SVD)
        """
        self.k = k  # number of singular values/vectors to keep

    def fit(self, A):
        """
        Compute SVD decomposition of matrix A
        
        Algorithm:
        1. Choose smaller covariance matrix for numerical stability
        2. Compute eigenvalues and eigenvectors
        3. Extract singular values and vectors
        4. Sort by singular values (descending)
        5. Truncate to k components if specified
        
        Parameters:
        - A: input matrix (m x n)
        """
        m, n = A.shape
        
        # For numerical stability, work on the smaller of A A^T or A^T A
        if m >= n:
            # Compute V from eigen of A^T A (n x n)
            C = A.T @ A
            eigvals, V = np.linalg.eigh(C)  # ascending order
            
            # Singular values are sqrt(eigvals)
            singular_vals = np.sqrt(np.clip(eigvals, 0, None))
            
            # Sort in descending order
            idx = np.argsort(singular_vals)[::-1]
            singular_vals = singular_vals[idx]
            V = V[:, idx]
            
            # Compute U = A V Σ^{-1}
            U = A @ V / singular_vals
        else:
            # Compute U from eigen of A A^T (m x m)
            C = A @ A.T
            eigvals, U = np.linalg.eigh(C)
            singular_vals = np.sqrt(np.clip(eigvals, 0, None))
            
            # Sort in descending order
            idx = np.argsort(singular_vals)[::-1]
            singular_vals = singular_vals[idx]
            U = U[:, idx]
            
            # Compute V = A^T U Σ^{-1}
            V = (A.T @ U) / singular_vals

        # Keep top k components if specified (truncated SVD)
        if self.k is not None:
            singular_vals = singular_vals[:self.k]
            U = U[:, :self.k]
            V = V[:, :self.k]

        # Build Σ matrix (diagonal matrix of singular values)
        Sigma = np.zeros((m, n))
        np.fill_diagonal(Sigma, singular_vals)
        
        # Store decomposition components
        self.U = U
        self.Sigma = Sigma
        self.VT = V.T
        self.singular_values_ = singular_vals
        return self

    def reconstruct(self):
        """
        Reconstruct original matrix from SVD decomposition
        
        Returns:
        - reconstructed matrix: U * Σ * V^T
        """
        return self.U @ self.Sigma @ self.VT

#########################
# Example Usage
#########################
if __name__ == "__main__":
    print("="*60)
    print("SINGULAR VALUE DECOMPOSITION (SVD) - FROM SCRATCH")
    print("="*60)

    # Create a synthetic matrix (e.g., rank-3 plus noise)
    np.random.seed(0)
    m, n = 50, 30
    # True low-rank factors
    U_true = np.random.randn(m, 3)
    V_true = np.random.randn(n, 3)
    A = U_true @ V_true.T + 0.1 * np.random.randn(m, n)

    print(f"Original matrix shape: {A.shape}")

    # Fit SVD, keep full rank
    svd = SVD()
    svd.fit(A)
    A_rec_full = svd.reconstruct()
    err_full = np.linalg.norm(A - A_rec_full) / np.linalg.norm(A)
    print(f"Reconstruction error (full rank): {err_full:.6f}")

    # Truncate to k=3
    k = 3
    svd_k = SVD(k=k)
    svd_k.fit(A)
    A_rec_k = svd_k.reconstruct()
    err_k = np.linalg.norm(A - A_rec_k) / np.linalg.norm(A)
    print(f"Reconstruction error (k={k}): {err_k:.6f}")

    # Display singular values
    print(f"Singular values: {svd.singular_values_[:10]}...")  # Show first 10
    print(f"Number of non-zero singular values: {np.sum(svd.singular_values_ > 1e-10)}")

    print("\nSVD demonstration complete!")
