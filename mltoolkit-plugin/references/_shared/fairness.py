"""Group-fairness metrics for binary classification outputs.

    group_metrics(y_true, y_pred, y_proba, groups) -> DataFrame
        One row per group with n, prevalence, accuracy, precision, recall,
        f1, roc_auc, tpr, fpr, ppv, npv.

    disparity_ratios(per_group_df) -> dict
        For each rate metric (tpr, fpr, ppv, npv, prevalence), returns
        max-over-groups / min-over-groups. Ratio of 1.0 = perfect parity.

Single-class-in-group edge case: roc_auc becomes NaN. All rate metrics
are still reported; callers decide how to handle NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def _safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return tpr, fpr, ppv, npv


def group_metrics(y_true, y_pred, y_proba, groups) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba); groups = pd.Series(groups).reset_index(drop=True)
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        yt, yp, yr = y_true[idx], y_pred[idx], y_proba[idx]
        tpr, fpr, ppv, npv = _rates(yt, yp)
        rows.append({
            "group": g,
            "n": len(idx),
            "prevalence": float(np.mean(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall":    float(recall_score(yt, yp, zero_division=0)),
            "f1":        float(f1_score(yt, yp, zero_division=0)),
            "roc_auc":   _safe_roc_auc(yt, yr),
            "tpr": tpr, "fpr": fpr, "ppv": ppv, "npv": npv,
        })
    return pd.DataFrame(rows)


_DISPARITY_METRICS = ("tpr", "fpr", "ppv", "npv", "prevalence")


def disparity_ratios(per_group: pd.DataFrame) -> dict:
    out = {}
    for m in _DISPARITY_METRICS:
        col = per_group[m].dropna()
        if col.empty or col.min() == 0:
            out[m] = float("nan")
        else:
            out[m] = float(col.max() / col.min())
    return out
