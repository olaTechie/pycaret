"""Sample-size audit: events-per-variable + minority-class prevalence.

    audit_epv(X, y, low_epv_threshold=10, rare_outcome_threshold=0.05)
      -> dict with keys:
         n_rows, n_features, n_events (minority-class count),
         minority_prevalence, epv (events / features),
         low_epv_warning (bool), rare_outcome_warning (bool)

Callers print the warnings at EDA time so users see them before training.
"""
from __future__ import annotations

import pandas as pd


def audit_epv(
    X: pd.DataFrame,
    y,
    *,
    low_epv_threshold: float = 10.0,
    rare_outcome_threshold: float = 0.05,
) -> dict:
    y = pd.Series(y)
    n = len(y)
    counts = y.value_counts(dropna=False)
    n_events = int(counts.min())
    prev = float(n_events / n) if n else float("nan")
    n_features = int(X.shape[1])
    epv = float(n_events / n_features) if n_features else float("inf")
    return {
        "n_rows": n,
        "n_features": n_features,
        "n_events": n_events,
        "minority_prevalence": prev,
        "epv": epv,
        "low_epv_warning": epv < low_epv_threshold,
        "rare_outcome_warning": prev < rare_outcome_threshold,
    }
