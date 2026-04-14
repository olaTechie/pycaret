"""Anomaly detection model zoo."""
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def get_zoo(contamination: float = 0.05) -> dict:
    return {
        "iforest":   {"estimator": IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)},
        "lof":       {"estimator": LocalOutlierFactor(contamination=contamination, novelty=False)},
        "elliptic":  {"estimator": EllipticEnvelope(contamination=contamination, random_state=42)},
        "ocsvm":     {"estimator": OneClassSVM(nu=contamination)},
    }
