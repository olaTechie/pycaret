"""Anomaly detection model zoo."""
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


class PCAReconstructionDetector:
    """Reconstruction-error anomaly scorer via sklearn PCA.

    Score is the squared reconstruction error; higher = more anomalous.
    Mirrors pyod's pca detector's API (fit, decision_function, predict).
    """

    def __init__(self, contamination: float = 0.05, n_components: int = 5,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        self._pca = PCA(n_components=min(self.n_components, X.shape[1]),
                        random_state=self.random_state)
        self._pca.fit(X)
        self._mean = X.mean(axis=0)
        scores = self.decision_function(X)
        # threshold at (1 - contamination) quantile — above = anomaly
        self._threshold = float(np.quantile(scores, 1 - self.contamination))
        return self

    def decision_function(self, X):
        X = np.asarray(X)
        recon = self._pca.inverse_transform(self._pca.transform(X))
        return np.sum((X - recon) ** 2, axis=1)

    def predict(self, X):
        return np.where(self.decision_function(X) >= self._threshold, -1, 1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)


def get_zoo(contamination: float = 0.05) -> dict:
    zoo = {
        "iforest":  {"estimator": IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)},
        "lof":      {"estimator": LocalOutlierFactor(contamination=contamination, novelty=False)},
        "elliptic": {"estimator": EllipticEnvelope(contamination=contamination, random_state=42)},
        "ocsvm":    {"estimator": OneClassSVM(nu=contamination)},
        "pca":      {"estimator": PCAReconstructionDetector(contamination=contamination)},
        "mcd":      {"estimator": MinCovDet(random_state=42)},
    }
    if deps._check("pyod"):
        try:
            from pyod.models.abod import ABOD  # type: ignore
            from pyod.models.hbos import HBOS  # type: ignore
            from pyod.models.cof import COF  # type: ignore
            from pyod.models.sod import SOD  # type: ignore
            from pyod.models.sos import SOS  # type: ignore
            zoo["abod"] = {"estimator": ABOD(contamination=contamination)}
            zoo["hbos"] = {"estimator": HBOS(contamination=contamination)}
            zoo["cof"] = {"estimator": COF(contamination=contamination)}
            zoo["sod"] = {"estimator": SOD(contamination=contamination)}
            zoo["sos"] = {"estimator": SOS(contamination=contamination)}
        except ImportError:
            pass
    return zoo
