from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os


"""Module for Clustering Operations

This module provides functions for performing clustering operations, including
calculating the optimal number of clusters using silhouette score and obtaining
cluster labels using the KMeans algorithm.

Functions:
- calculate_cluster_number: Calculates the optimal number of clusters using 
    silhouette score.
- get_cluster_labels: Performs clustering and returns cluster labels.
"""


def calculate_cluster_number(vectors):
    """
    Calculates the optimal number of clusters using silhouette score.

    Parameters:
    vectors (array-like): Input vectors.

    Returns:
    int: Optimal number of clusters.
    """
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=3, n_init=20, max_iter=1000)
        kmeans.fit(vectors)
        score = silhouette_score(vectors, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k


def get_cluster_labels(vectors):
    """
    Performs clustering and returns cluster labels.

    Parameters:
    vectors (array-like): Input vectors.

    Returns:
    array: Cluster labels.
    """
    os.environ["LOKY_MAX_CPU_COUNT"] = "2"
    cluster_no = calculate_cluster_number(vectors)
    print(f"Clustering to {cluster_no} Clusters")
    kmeans = KMeans(n_clusters=cluster_no, random_state=3, n_init=20, max_iter=1000)
    clustering = kmeans.fit(vectors)
    return clustering.labels_
