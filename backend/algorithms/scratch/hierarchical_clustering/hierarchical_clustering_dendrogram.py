# hierarchical_clustering_dendrogram.py

import numpy as np
import matplotlib.pyplot as plt

#########################
# Distance Function
#########################
def euclidean(a, b):
    return np.linalg.norm(a - b)

#########################
# Linkage Methods
#########################
def single_linkage(D, c1, c2):
    return np.min([D[i, j] for i in c1 for j in c2])

def complete_linkage(D, c1, c2):
    return np.max([D[i, j] for i in c1 for j in c2])

def average_linkage(D, c1, c2):
    vals = [D[i, j] for i in c1 for j in c2]
    return np.mean(vals)

#########################
# Agglomerative Clustering & Dendrogram Data
#########################
def hac_linkage_matrix(X, linkage='single'):
    n = X.shape[0]
    # Precompute distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = D[j, i] = euclidean(X[i], X[j])
    # Initial clusters
    clusters = [[i] for i in range(n)]
    # Linkage list: [idx1, idx2, dist, new_size]
    Z = []
    link_fn = {'single': single_linkage,
               'complete': complete_linkage,
               'average': average_linkage}[linkage]
    # Agglomerate
    while len(clusters) > 1:
        min_dist = np.inf
        to_merge = (None, None)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = link_fn(D, clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    to_merge = (i, j)
        i, j = to_merge
        size = len(clusters[i]) + len(clusters[j])
        Z.append([i, j, min_dist, size])
        # Merge
        new = clusters[i] + clusters[j]
        # Remove higher index first
        clusters.pop(j), clusters.pop(i)
        clusters.append(new)
    return np.array(Z)

#########################
# Plotting Dendrogram from Z
#########################
def plot_dendrogram(Z, labels=None, ax=None):
    """
    Plot a dendrogram from a linkage matrix Z (as returned above).
    Simple bottom-up tree plot.
    """
    if ax is None:
        ax = plt.gca()

    # Number of original points
    m = Z.shape[0] + 1
    # Track cluster positions (x-coordinates)
    cluster_pos = {i: i for i in range(m)}
    current_cluster_id = m

    for idx, (i, j, dist, size) in enumerate(Z):
        i, j = int(i), int(j)
        xi, xj = cluster_pos[i], cluster_pos[j]
        x_new = (xi + xj) / 2
        y = dist
        # Draw vertical lines
        ax.plot([xi, xi], [0, y], 'k-')
        ax.plot([xj, xj], [0, y], 'k-')
        # Draw horizontal connector
        ax.plot([xi, xj], [y, y], 'k-')
        # Assign new cluster position
        cluster_pos[current_cluster_id] = x_new
        current_cluster_id += 1

    ax.set_xlabel('Sample index' if labels is None else 'Sample label')
    ax.set_ylabel('Distance')
    ax.set_xticks(range(m))
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)
    ax.set_title('Hierarchical Clustering Dendrogram')

#########################
# Main Demo
#########################
if __name__ == "__main__":
    print("="*60)
    print("HIERARCHICAL AGGLOMERATIVE CLUSTERING (PURE SCRATCH)")
    print("="*60)

    # Generate sample data
    from numpy.random import default_rng
    rng = default_rng(0)
    # Three clusters
    C1 = rng.normal([1,1], 0.2, size=(20, 2))
    C2 = rng.normal([5,5], 0.2, size=(20, 2))
    C3 = rng.normal([1,5], 0.2, size=(20, 2))
    X = np.vstack([C1, C2, C3])
    labels = np.array([0]*20 + [1]*20 + [2]*20)

    # Compute linkage matrix
    linkage_method = 'average'  # 'single', 'complete', or 'average'
    Z = hac_linkage_matrix(X, linkage=linkage_method)

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_dendrogram(Z, labels=labels, ax=ax)
    plt.tight_layout()
    plt.savefig("hierarchical_clustering_dendrogram.png", dpi=150)
    print("Dendrogram saved as hierarchical_clustering_dendrogram.png")
    print("Demo complete!")
