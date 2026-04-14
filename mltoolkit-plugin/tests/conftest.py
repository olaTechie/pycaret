"""Shared pytest fixtures: synthetic datasets for reference-script smoke tests."""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs


@pytest.fixture
def classification_data(tmp_path):
    """Binary classification dataset with mixed numeric/categorical features."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    df["cat_a"] = np.random.RandomState(42).choice(["a", "b", "c"], size=500)
    df["target"] = y
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "target": "target", "df": df}


@pytest.fixture
def regression_data(tmp_path):
    """Regression dataset."""
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    df["cat_a"] = np.random.RandomState(42).choice(["a", "b", "c"], size=500)
    df["target"] = y
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "target": "target", "df": df}


@pytest.fixture
def cluster_data(tmp_path):
    """Unsupervised clustering dataset (no target)."""
    X, _ = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    path = tmp_path / "cluster.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "df": df}


@pytest.fixture
def anomaly_data(tmp_path):
    """Dataset with injected outliers."""
    rng = np.random.RandomState(42)
    normal = rng.normal(0, 1, size=(480, 5))
    outliers = rng.normal(8, 1, size=(20, 5))
    X = np.vstack([normal, outliers])
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    path = tmp_path / "anomaly.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "df": df}
