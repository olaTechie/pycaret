"""bootstrap.py — bootstrap CI for any metric_fn(y_true, y_pred) -> float."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from references._shared.bootstrap import bootstrap_ci


def test_bootstrap_ci_point_estimate_matches_plain_metric():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    res = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=200, random_state=0)
    assert abs(res["point"] - accuracy_score(y_true, y_pred)) < 1e-12


def test_bootstrap_ci_has_lower_and_upper():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    res = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=200, random_state=0, alpha=0.05)
    assert res["lower"] <= res["point"] <= res["upper"]
    assert 0 <= res["lower"] and res["upper"] <= 1


def test_bootstrap_ci_works_with_probability_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_proba = rng.random(300)
    res = bootstrap_ci(roc_auc_score, y_true, y_proba, n_boot=200, random_state=0)
    assert "lower" in res and "upper" in res


def test_bootstrap_ci_deterministic_with_random_state():
    y_true = np.array([0, 1] * 100)
    y_pred = np.array([0, 1] * 100)
    a = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=100, random_state=42)
    b = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=100, random_state=42)
    assert a == b
