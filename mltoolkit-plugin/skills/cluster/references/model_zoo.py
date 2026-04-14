"""Clustering model zoo."""
import sys
from pathlib import Path

from sklearn.cluster import (
    AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans,
    MeanShift, OPTICS, SpectralClustering,
)
from sklearn.mixture import GaussianMixture

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


def get_zoo(n_clusters: int = 4) -> dict:
    zoo = {
        "kmeans":   {"estimator": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                     "requires_n_clusters": True},
        "dbscan":   {"estimator": DBSCAN(),
                     "requires_n_clusters": False},
        "agglom":   {"estimator": AgglomerativeClustering(n_clusters=n_clusters),
                     "requires_n_clusters": True},
        "gmm":      {"estimator": GaussianMixture(n_components=n_clusters, random_state=42),
                     "requires_n_clusters": True},
        "ap":       {"estimator": AffinityPropagation(random_state=42),
                     "requires_n_clusters": False},
        "meanshift": {"estimator": MeanShift(bin_seeding=True),
                      "requires_n_clusters": False},
        "spectral": {"estimator": SpectralClustering(n_clusters=n_clusters, random_state=42,
                                                     assign_labels="discretize"),
                     "requires_n_clusters": True},
        "optics":   {"estimator": OPTICS(min_samples=5),
                     "requires_n_clusters": False},
        "birch":    {"estimator": Birch(n_clusters=n_clusters),
                     "requires_n_clusters": True},
    }
    if deps._check("kmodes"):
        from kmodes.kmodes import KModes  # type: ignore
        zoo["kmodes"] = {"estimator": KModes(n_clusters=n_clusters, random_state=42),
                         "requires_n_clusters": True}
    return zoo
