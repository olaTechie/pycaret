"""Percentile bootstrap confidence intervals for any metric_fn(y_true, y_pred).

Primary API:
    bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, alpha=0.05,
                 random_state=42) -> dict{point, lower, upper, alpha, n_boot, n_valid}
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    metric_fn,
    y_true,
    y_pred,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    point = float(metric_fn(y_true, y_pred))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot[i] = float(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {"point": point, "lower": lo, "upper": hi,
            "alpha": alpha, "n_boot": n_boot, "n_valid": int(len(boot))}
