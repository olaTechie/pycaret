"""Clustering model zoo."""
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


def get_zoo(n_clusters: int = 4) -> dict:
    return {
        "kmeans":    {"estimator": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                      "requires_n_clusters": True},
        "dbscan":    {"estimator": DBSCAN(),
                      "requires_n_clusters": False},
        "agglom":    {"estimator": AgglomerativeClustering(n_clusters=n_clusters),
                      "requires_n_clusters": True},
        "gmm":       {"estimator": GaussianMixture(n_components=n_clusters, random_state=42),
                      "requires_n_clusters": True},
    }
