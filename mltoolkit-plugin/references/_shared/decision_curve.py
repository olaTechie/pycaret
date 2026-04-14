"""Decision-curve analysis (Vickers 2006) for binary classifiers.

    net_benefit_curve(y_true, y_proba, thresholds) -> DataFrame
        Columns: threshold, model, treat_all, treat_none.
        Net benefit = TP/n - FP/n * (p_t / (1 - p_t))

    decision_curve_figure(y_true, y_proba, thresholds=None) -> matplotlib.Figure
        Plots all three curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _nb(y_true, y_hat, p_t):
    n = len(y_true)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    if n == 0 or p_t >= 1:
        return 0.0
    return tp / n - (fp / n) * (p_t / (1 - p_t))


def net_benefit_curve(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)
    rows = []
    prev = float(np.mean(y_true))
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        model = _nb(y_true, y_hat, t)
        treat_all = prev - (1 - prev) * (t / (1 - t)) if t < 1 else 0.0
        rows.append({"threshold": float(t), "model": model,
                     "treat_all": float(treat_all), "treat_none": 0.0})
    return pd.DataFrame(rows)


def decision_curve_figure(y_true, y_proba, thresholds=None):
    df = net_benefit_curve(y_true, y_proba, thresholds=thresholds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["threshold"], df["model"], label="Model")
    ax.plot(df["threshold"], df["treat_all"], "--", label="Treat all")
    ax.plot(df["threshold"], df["treat_none"], ":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend()
    return fig
