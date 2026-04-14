"""fairness.py — per-group metrics + disparity ratios."""
import numpy as np
import pandas as pd

from references._shared.fairness import group_metrics, disparity_ratios


def test_group_metrics_returns_row_per_group():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_proba = rng.random(n)
    groups = pd.Series(rng.choice(["A", "B", "C"], size=n))
    df = group_metrics(y_true, y_pred, y_proba, groups)
    assert set(df["group"]) == {"A", "B", "C"}
    for col in ["n", "prevalence", "accuracy", "precision", "recall",
                "f1", "roc_auc", "tpr", "fpr"]:
        assert col in df.columns


def test_group_metrics_handles_single_class_group():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])
    groups = pd.Series(["A"] * 3 + ["B"] * 3)
    df = group_metrics(y_true, y_pred, y_proba, groups)
    a = df[df["group"] == "A"].iloc[0]
    assert np.isnan(a["roc_auc"])


def test_disparity_ratios_returns_max_min_ratio_per_metric():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_proba = rng.random(n)
    groups = pd.Series(rng.choice(["A", "B"], size=n))
    per = group_metrics(y_true, y_pred, y_proba, groups)
    ratios = disparity_ratios(per)
    assert "tpr" in ratios and ratios["tpr"] >= 1.0
    assert "fpr" in ratios
