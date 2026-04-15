"""Build frozen PyCaret 3.4.0 baseline artifacts for parity testing.

USAGE:
    # In an ISOLATED env with pycaret==3.4.0 on Python 3.11:
    uv venv --python 3.11 /tmp/pycaret-baseline
    source /tmp/pycaret-baseline/bin/activate
    uv pip install "pycaret[full]==3.4.0"
    cd <repo-root>
    python scripts/build_parity_baseline.py

The script writes to tests/parity/baselines/3.4.0/<dataset>/ and must be
re-run only if the reference dataset set changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.parity.baseline import (  # noqa: E402
    LeaderboardBaseline,
    PredictionBaseline,
    save_leaderboard,
    save_predictions,
)
from tests.parity.datasets import REFERENCE_DATASETS, load_reference  # noqa: E402

BASELINE_VERSION = "3.4.0"
OUT_ROOT = REPO_ROOT / "tests" / "parity" / "baselines" / BASELINE_VERSION

TOP_N_MODELS = 3  # persist predictions for the top-N leaderboard rows


def build_classification_or_regression(dataset_name: str, X, y, task: str):
    from pycaret.classification import ClassificationExperiment
    from pycaret.regression import RegressionExperiment

    exp_cls = ClassificationExperiment if task == "classification" else RegressionExperiment
    exp = exp_cls()
    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    exp.setup(data=df, target=target, session_id=42, verbose=False, html=False)

    _ = exp.compare_models(n_select=TOP_N_MODELS, verbose=False)
    leaderboard_df = exp.pull().reset_index().rename(columns={"index": "Model"})
    leaderboard_df = leaderboard_df.sort_values("Model")

    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_leaderboard(
        LeaderboardBaseline(
            dataset=dataset_name,
            version=BASELINE_VERSION,
            task=task,
            rows=leaderboard_df.to_dict(orient="records"),
        ),
        out_dir / "leaderboard.json",
    )

    top_models = leaderboard_df["Model"].head(TOP_N_MODELS).tolist()
    for model_name in top_models:
        model = exp.create_model(model_name, verbose=False)
        preds_df = exp.predict_model(model, data=df, verbose=False)
        preds = (
            preds_df["prediction_label"].values
            if "prediction_label" in preds_df.columns
            else preds_df["Label"].values
        )
        probs = None
        if task == "classification":
            score_cols = sorted(
                [
                    c
                    for c in preds_df.columns
                    if c.startswith("prediction_score_") or c.startswith("Score_")
                ]
            )
            if score_cols:
                probs = preds_df[score_cols].values
        save_predictions(
            PredictionBaseline(
                dataset=dataset_name,
                version=BASELINE_VERSION,
                model=model_name,
                predictions=np.asarray(preds),
                probabilities=probs,
            ),
            out_dir / f"predictions_{model_name}.npz",
        )

    print(f"[ok] {dataset_name}: leaderboard + {len(top_models)} prediction files")


def build_time_series(dataset_name: str, X, y):
    from pycaret.time_series import TSForecastingExperiment

    exp = TSForecastingExperiment()
    exp.setup(data=y, fh=12, session_id=42, verbose=False)
    _ = exp.compare_models(n_select=TOP_N_MODELS, verbose=False)
    leaderboard_df = exp.pull().reset_index().rename(columns={"index": "Model"})
    leaderboard_df = leaderboard_df.sort_values("Model")

    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_leaderboard(
        LeaderboardBaseline(
            dataset=dataset_name,
            version=BASELINE_VERSION,
            task="time_series",
            rows=leaderboard_df.to_dict(orient="records"),
        ),
        out_dir / "leaderboard.json",
    )

    top_models = leaderboard_df["Model"].head(TOP_N_MODELS).tolist()
    for model_name in top_models:
        model = exp.create_model(model_name, verbose=False)
        preds = exp.predict_model(model, fh=12, verbose=False)
        preds_arr = np.asarray(
            preds["y_pred"].values
            if "y_pred" in preds.columns
            else preds.iloc[:, 0].values,
            dtype=float,
        )
        save_predictions(
            PredictionBaseline(
                dataset=dataset_name,
                version=BASELINE_VERSION,
                model=model_name,
                predictions=preds_arr,
                probabilities=None,
            ),
            out_dir / f"predictions_{model_name}.npz",
        )
    print(f"[ok] {dataset_name}: ts leaderboard + {len(top_models)} prediction files")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    import pycaret
    if not pycaret.__version__.startswith("3.4.0"):
        raise SystemExit(
            f"Expected pycaret==3.4.0 but got {pycaret.__version__}. "
            f"Activate the isolated baseline env before running this script."
        )
    for name in REFERENCE_DATASETS:
        X, y, task = load_reference(name)
        if task == "time_series":
            build_time_series(name, X, y)
        else:
            build_classification_or_regression(name, X, y, task)


if __name__ == "__main__":
    main()
