"""Parity test: per-model prediction arrays vs. frozen 3.4.0 baseline.

Uses Spearman rank-correlation for continuous outputs and exact-match-rate
for classification labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from .baseline import load_predictions
from .datasets import load_reference


@pytest.mark.parametrize(
    "dataset_name", ["iris", "diabetes", "california_housing", "credit"]
)
def test_predict_parity(dataset_name, parity_config):
    baseline_dir = parity_config.baseline_root / dataset_name
    if not baseline_dir.exists():
        pytest.skip(f"Baseline dir missing: {baseline_dir}")

    pred_files = sorted(baseline_dir.glob("predictions_*.npz"))
    if not pred_files:
        pytest.skip(f"No per-model predictions in {baseline_dir}")

    X, y, task = load_reference(dataset_name)

    for pred_path in pred_files:
        baseline = load_predictions(pred_path)
        current_preds, current_probs = _predict_current(X, y, task, baseline.model)

        if task == "classification":
            match_rate = float(np.mean(current_preds == baseline.predictions))
            assert match_rate >= 0.99, (
                f"{dataset_name}/{baseline.model}: "
                f"label match rate {match_rate:.4f} < 0.99"
            )
            if baseline.probabilities is not None and current_probs is not None:
                for col in range(baseline.probabilities.shape[1]):
                    rho, _ = spearmanr(
                        current_probs[:, col], baseline.probabilities[:, col]
                    )
                    assert rho >= parity_config.rank_corr_min, (
                        f"{dataset_name}/{baseline.model}/proba[{col}]: "
                        f"rank-corr {rho:.4f} < {parity_config.rank_corr_min}"
                    )
        elif task == "regression":
            rho, _ = spearmanr(current_preds, baseline.predictions)
            assert rho >= parity_config.rank_corr_min, (
                f"{dataset_name}/{baseline.model}: "
                f"regression rank-corr {rho:.4f} < {parity_config.rank_corr_min}"
            )


def _predict_current(X, y, task, model_name):
    """Re-fit and predict with named model on current HEAD."""
    if task == "classification":
        from pycaret.classification import ClassificationExperiment

        exp = ClassificationExperiment()
    else:
        from pycaret.regression import RegressionExperiment

        exp = RegressionExperiment()

    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    exp.setup(data=df, target=target, session_id=42, verbose=False, html=False)
    model = exp.create_model(model_name, verbose=False)
    preds_df = exp.predict_model(model, data=df, verbose=False)
    preds = (
        preds_df["prediction_label"].values
        if "prediction_label" in preds_df.columns
        else preds_df["Label"].values
    )
    probs = None
    if task == "classification":
        score_cols = [
            c
            for c in preds_df.columns
            if c.startswith("prediction_score_") or c.startswith("Score_")
        ]
        if score_cols:
            probs = preds_df[sorted(score_cols)].values
    return preds, probs
