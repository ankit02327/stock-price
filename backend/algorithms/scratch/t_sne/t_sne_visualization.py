"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) Implementation from Scratch

This module implements t-SNE for non-linear dimensionality reduction using:
- Gaussian similarities in high-dimensional space
- t-distribution similarities in low-dimensional space
- Gradient descent optimization
- Perplexity-based neighborhood selection

Mathematical Foundation:
- High-dim similarities: p_ij = exp(-||x_i - x_j||²/2σ²) / Σ_k exp(-||x_i - x_k||²/2σ²)
- Low-dim similarities: q_ij = (1 + ||y_i - y_j||²)⁻¹ / Σ_k≠l (1 + ||y_k - y_l||²)⁻¹
- Cost function: C = KL(P||Q) = Σ p_ij log(p_ij/q_ij)
- Gradient: ∂C/∂y_i = 4Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i - y_j||²)⁻¹
"""

import numpy as np

#########################
# Helper Functions
#########################
def pairwise_distances(X):
    """Compute squared Euclidean distances matrix."""
    sum_X = np.sum(np.square(X), axis=1)
    return -2 * X.dot(X.T) + sum_X[:, None] + sum_X[None, :]

def compute_perplexity_probabilities(D, perplexity=30.0, tol=1e-5):
    """
    Compute high-dimensional (P) affinities with fixed perplexity via binary search.
    Returns symmetric P matrix.
    """
    (n, _) = D.shape
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        betamin, betamax = -np.inf, np.inf
        beta = 1.0
        Di = np.delete(D[i], i)
        # Binary search for beta to match perplexity
        for _ in range(50):
            Pi = np.exp(-Di * beta)
            sumPi = np.sum(Pi)
            if sumPi == 0:
                break
            Hi = np.log(sumPi) + beta * np.sum(Di * Pi) / sumPi
            Hdiff = Hi - logU
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2
        Pi = Pi / sumPi
        P[i, np.arange(n) != i] = Pi

    # Symmetrize
    P = (P + P.T) / (2 * n)
    return P

#########################
# t-SNE Algorithm
#########################
def tsne(X, no_dims=2, perplexity=30.0, max_iter=1000, learning_rate=200.0, momentum=0.5):
    """
    Basic (vanilla) t-SNE using gradient descent.
    """
    (n, d) = X.shape
    Y = np.random.randn(n, no_dims) * 1e-4
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P matrix
    D = pairwise_distances(X)
    P = compute_perplexity_probabilities(D, perplexity)
    P = np.maximum(P, 1e-12)

    for iter in range(max_iter):
        # Compute low-dim affinities Q
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + (-2 * Y.dot(Y.T) + sum_Y[:, None] + sum_Y[None, :]))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i] = 4 * np.sum((PQ[:, i][:, None] * num[:, i][:, None]) * (Y[i] - Y), axis=0)

        # Update gains
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01

        # Update Y
        iY = momentum * iY - learning_rate * (gains * dY)
        Y += iY
        Y -= np.mean(Y, axis=0)  # zero-mean

        # Increase momentum after 250 iterations
        if iter == 250:
            momentum = 0.8

        # Print cost every 100 iters
        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print(f"Iteration {iter+1}: error = {C:.5f}")

    return Y

#########################
# Example Usage & Visualization
#########################
if __name__ == "__main__":
    print("="*60)
    print("t-SNE DIMENSIONALITY REDUCTION (from scratch)")
    print("="*60)

    # Generate synthetic 3D data with clusters
    np.random.seed(0)
    C1 = np.random.randn(100, 3) + np.array([0,0,0])
    C2 = np.random.randn(100, 3) + np.array([5,5,5])
    C3 = np.random.randn(100, 3) + np.array([-5,5,-5])
    X = np.vstack([C1, C2, C3])
    labels = np.array([0]*100 + [1]*100 + [2]*100)

    # Run t-SNE
    Y = tsne(X, no_dims=2, perplexity=30.0, max_iter=500, learning_rate=200.0)

    # Display results
    print(f"Original data shape: {X.shape}")
    print(f"t-SNE projection shape: {Y.shape}")
    print(f"Number of clusters: {len(np.unique(labels))}")
    
    # Calculate cluster separation in 2D space
    cluster_centers = []
    for lab in np.unique(labels):
        center = np.mean(Y[labels==lab], axis=0)
        cluster_centers.append(center)
        print(f"Cluster {lab} center: ({center[0]:.3f}, {center[1]:.3f})")
    
    # Calculate inter-cluster distances
    if len(cluster_centers) > 1:
        distances = []
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                distances.append(dist)
        print(f"Average inter-cluster distance: {np.mean(distances):.3f}")
    
    print("\nt-SNE implementation complete!")
