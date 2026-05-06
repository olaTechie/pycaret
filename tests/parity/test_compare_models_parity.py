"""Parity test: compare_models() leaderboard vs. frozen 3.4.0 baseline.

This test will SKIP cleanly until Task 12 builds the baseline artifacts.
Once artifacts exist, it asserts metric absolute deltas are within tolerance.
"""
from __future__ import annotations

import pytest

from .baseline import load_leaderboard
from .datasets import load_reference


@pytest.mark.parametrize(
    "dataset_name", ["iris", "diabetes", "california_housing", "credit"]
)
def test_compare_models_leaderboard_parity(dataset_name, parity_config):
    baseline_path = parity_config.baseline_root / dataset_name / "leaderboard.json"
    if not baseline_path.exists():
        pytest.skip(f"Baseline missing: {baseline_path}")

    baseline = load_leaderboard(baseline_path)
    X, y, task = load_reference(dataset_name)

    current = _run_compare_models(X, y, task)

    baseline_by_model = {r["Model"]: r for r in baseline.rows}
    current_by_model = {r["Model"]: r for r in current}

    missing = set(baseline_by_model) - set(current_by_model)
    assert not missing, (
        f"Models present in 3.4.0 but absent in current: {sorted(missing)}"
    )

    for model, base_row in baseline_by_model.items():
        cur_row = current_by_model[model]
        tol = parity_config.overrides.get(model, {})
        for metric, base_val in base_row.items():
            if metric == "Model" or not isinstance(base_val, (int, float)):
                continue
            cur_val = cur_row.get(metric)
            assert cur_val is not None, f"Metric {metric} missing for model {model}"
            metric_tol = tol.get(metric, parity_config.metric_abs_tol)
            assert abs(cur_val - base_val) <= metric_tol, (
                f"{dataset_name}/{model}/{metric}: "
                f"current={cur_val}, baseline={base_val}, "
                f"delta={abs(cur_val - base_val)}, tol={metric_tol}"
            )


def _run_compare_models(X, y, task):
    """Run compare_models on current HEAD and return leaderboard as list of dicts."""
    if task == "classification":
        from pycaret.classification import ClassificationExperiment
        exp = ClassificationExperiment()
        exp.setup(
            data=_to_frame(X, y),
            target=y.name or "target",
            session_id=42,
            verbose=False,
            html=False,
        )
    elif task == "regression":
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment()
        exp.setup(
            data=_to_frame(X, y),
            target=y.name or "target",
            session_id=42,
            verbose=False,
            html=False,
        )
    else:
        pytest.skip(f"Task {task} handled in test_predict_parity")

    _ = exp.compare_models(n_select=1, verbose=False)
    leaderboard_df = exp.pull()
    leaderboard_df = leaderboard_df.reset_index().rename(columns={"index": "Model"})
    leaderboard_df = leaderboard_df.sort_values("Model")
    return leaderboard_df.to_dict(orient="records")


def _to_frame(X, y):
    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    return df
