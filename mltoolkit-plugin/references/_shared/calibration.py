"""Calibration diagnostics for binary classifiers.

    calibration_summary(y_true, y_proba, n_bins=10) -> dict
        brier, ECE, calibration_intercept, calibration_slope
        (the latter two come from logistic regression of y on logit(p)).

    reliability_diagram(y_true, y_proba, n_bins=10) -> matplotlib.Figure
        Standard bin-means plot with y=x reference line.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_EPS = 1e-12


def _ece(y_true, y_proba, n_bins):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    weights = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        weights[i] = mask.sum()
    weights = weights / max(weights.sum(), 1)
    used = len(prob_true)
    w_used = weights[:used]
    return float(np.sum(w_used * np.abs(prob_true - prob_pred)))


def _calibration_intercept_slope(y_true, y_proba):
    p = np.clip(y_proba, _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def calibration_summary(y_true, y_proba, *, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    brier = float(brier_score_loss(y_true, y_proba))
    ece = _ece(y_true, y_proba, n_bins)
    intercept, slope = _calibration_intercept_slope(y_true, y_proba)
    return {"brier": brier, "ece": ece,
            "intercept": intercept, "slope": slope,
            "n_bins": n_bins, "bins": n_bins}


def reliability_diagram(y_true, y_proba, *, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(prob_pred, prob_true, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram")
    ax.legend()
    return fig
