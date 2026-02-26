# general_clustering_methods.py

import numpy as np
import matplotlib.pyplot as plt


#########################
# K-Means Clustering
#########################
def k_means(X, K, max_iters=100, tol=1e-4):
    """Simple K-Means from scratch."""
    n, d = X.shape
    # Initialize centroids by sampling K points
    centroids = X[np.random.choice(n, K, replace=False)]
    labels = np.zeros(n, dtype=int)
    for it in range(max_iters):
        # Assign labels
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # Update centroids
        new_centroids = np.vstack([X[new_labels == k].mean(axis=0) if np.any(new_labels==k) else centroids[k]
                                   for k in range(K)])
        # Check convergence
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            break
        centroids, labels = new_centroids, new_labels
    return labels, centroids

#########################
# Hierarchical Agglomerative Clustering
#########################
def hac_single_linkage(X, K):
    """Agglomerative clustering with single linkage."""
    n = X.shape[0]
    clusters = {i: [i] for i in range(n)}
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    while len(clusters) > K:
        # Find closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        # Merge j into i
        clusters[i].extend(clusters[j])
        del clusters[j]
        # Update distances
        for k in clusters:
            if k == i: continue
            d_ik = min(dists[p, q] for p in clusters[i] for q in clusters[k])
            dists[i, k] = dists[k, i] = d_ik
        dists[:, j] = dists[j, :] = np.inf
    # Assign labels
    labels = np.zeros(n, dtype=int)
    for label, pts in enumerate(clusters.values()):
        labels[pts] = label
    return labels

#########################
# DBSCAN
#########################
def dbscan(X, eps, min_pts):
    """DBSCAN from scratch."""
    n = X.shape[0]
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    labels = np.zeros(n, dtype=int)
    visited = np.zeros(n, bool)
    cluster_id = 0

    def region_query(p):
        return np.where(dists[p] <= eps)[0]

    for p in range(n):
        if visited[p]: continue
        visited[p] = True
        neigh = region_query(p)
        if len(neigh) < min_pts:
            labels[p] = 0  # noise
        else:
            cluster_id += 1
            labels[p] = cluster_id
            seeds = list(neigh)
            i = 0
            while i < len(seeds):
                q = seeds[i]
                if not visited[q]:
                    visited[q] = True
                    q_neigh = region_query(q)
                    if len(q_neigh) >= min_pts:
                        seeds += [m for m in q_neigh if m not in seeds]
                if labels[q] == 0:
                    labels[q] = cluster_id
                i += 1
    return labels

#########################
# Gaussian Mixture via EM
#########################
def gmm_em(X, K, max_iters=100, tol=1e-4):
    """Gaussian Mixture Model via Expectation–Maximization."""
    n, d = X.shape
    # Initialize parameters
    np.random.seed(0)
    means = X[np.random.choice(n, K, replace=False)]
    covs = [np.eye(d)] * K
    pis = np.ones(K) / K
    log_likelihood_old = None

    def gaussian(x, mean, cov):
        denom = np.sqrt((2*np.pi)**d * np.linalg.det(cov))
        diff = x - mean
        return np.exp(-0.5 * diff.T @ np.linalg.inv(cov) @ diff) / denom

    for it in range(max_iters):
        # E-step: responsibilities
        resp = np.zeros((n, K))
        for k in range(K):
            for i in range(n):
                resp[i, k] = pis[k] * gaussian(X[i], means[k], covs[k])
        resp /= resp.sum(axis=1, keepdims=True)
        # M-step: update parameters
        Nk = resp.sum(axis=0)
        pis = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        covs = []
        for k in range(K):
            diff = X - means[k]
            cov = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
            covs.append(cov + 1e-6*np.eye(d))
        # Check log-likelihood
        log_likelihood = np.sum(np.log(resp.sum(axis=1)))
        if log_likelihood_old is not None and abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood
    labels = resp.argmax(axis=1)
    return labels, means, covs, pis

#########################
# Example Usage
#########################
if __name__ == "__main__":
    print("="*60)
    print("GENERAL CLUSTERING METHODS - DEMONSTRATION")
    print("="*60)

    # Generate sample data: 2D blobs
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    print("\n--- K-Means ---")
    labels_km, centroids = k_means(X, K=4)
    print("K-Means labels unique:", np.unique(labels_km))

    print("\n--- HAC (Single Linkage) ---")
    labels_hac = hac_single_linkage(X, K=4)
    print("HAC labels unique:", np.unique(labels_hac))

    print("\n--- DBSCAN (eps=0.5, min_pts=5) ---")
    labels_db = dbscan(X, eps=0.5, min_pts=5)
    print("DBSCAN labels unique:", np.unique(labels_db))

    print("\n--- GMM via EM (K=4) ---")
    labels_gmm, means, covs, pis = gmm_em(X, K=4)
    print("GMM labels unique:", np.unique(labels_gmm))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    for ax, lbls, title in zip(axes.ravel(),
                               [labels_km, labels_hac, labels_db, labels_gmm],
                               ["K-Means", "HAC", "DBSCAN", "GMM"]):
        ax.scatter(X[:,0], X[:,1], c=lbls, cmap='tab10', s=30)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("general_clustering_methods.png")
    print("\nSaved plot: general_clustering_methods.png")
    print("\nGENERAL CLUSTERING DEMO COMPLETE!")