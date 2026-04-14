"""Publication-quality classification pipeline — standalone, no PyCaret.

Stages (via --stage):
    eda       Data overview + exploratory figures
    compare   Cross-validated comparison of all available models
    tune      RandomizedSearchCV on the best model from compare
    evaluate  Holdout evaluation with confusion matrix, ROC, PR curves
    all       Runs eda -> compare -> tune -> evaluate

Usage:
    python classify_reference.py --data data.csv --target y --output-dir out/ --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


# ----- Stages ---------------------------------------------------------------

def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in columns: {list(df.columns)}")
    if df[target].nunique() < 2:
        raise SystemExit(f"Target '{target}' has only one unique value.")
    return df


def run_eda(df: pd.DataFrame, target: str, out: Path):
    plotting.set_style()
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "nunique": df.nunique(),
    })
    summary.to_csv(out / "results/schema.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Class distribution — {target}")
    ax.set_ylabel("count")
    plotting.save_fig(fig, out / "artifacts/class_distribution")
    plt.close(fig)

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap")
        plt.close(fig)


def compare_models(X_train, y_train, out: Path, cv: int = 5) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = preprocessing.build_preprocessor(X_train)

    n_classes = len(np.unique(y_train))
    scorers = {
        "accuracy": "accuracy",
        "f1": "f1_macro" if n_classes > 2 else "f1",
    }
    if n_classes == 2:
        scorers["roc_auc"] = "roc_auc"

    rows = []
    for mid, entry in zoo.items():
        pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])
        row = {"model": mid}
        try:
            for name, scoring in scorers.items():
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                row[name] = float(scores.mean())
                row[f"{name}_std"] = float(scores.std())
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    sort_key = "roc_auc" if "roc_auc" in scorers else "f1"
    leaderboard = pd.DataFrame(rows).sort_values(
        by=sort_key, ascending=False, na_position="last",
    ).reset_index(drop=True)
    leaderboard.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(leaderboard))
    return leaderboard


def tune_model(model_id: str, X_train, y_train, out: Path,
               n_iter: int = 20, search_library: str = "sklearn"):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = preprocessing.build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])
    grid = {f"clf__{k}": v for k, v in entry["param_grid"].items()}
    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, None

    # Resolve shared modules in both in-place and staged layouts.
    try:
        from _shared import deps as _deps
        from _shared.tuning_optuna import optuna_search as _optuna_search
    except ImportError:
        from references._shared import deps as _deps
        from references._shared.tuning_optuna import optuna_search as _optuna_search

    if search_library == "optuna" and not _deps.has_optuna():
        print("WARNING: --search-library optuna requested but optuna is "
              "not installed; falling back to RandomizedSearchCV.", flush=True)
        search_library = "sklearn"

    if search_library == "optuna":
        best, best_score, best_params = _optuna_search(
            pipe, grid, X_train, y_train,
            scoring="accuracy", cv=5, n_iter=n_iter, random_state=42,
        )
        with open(out / "results/best_params.json", "w") as f:
            json.dump({k: str(v) for k, v in best_params.items()}, f, indent=2)
        return best, best_score

    # Default: RandomizedSearchCV.
    max_combos = 1
    for v in grid.values():
        max_combos *= len(v)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, max_combos),
        cv=5, scoring="accuracy", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    with open(out / "results/best_params.json", "w") as f:
        json.dump({k: str(v) for k, v in search.best_params_.items()}, f, indent=2)
    return search.best_estimator_, search.best_score_


def evaluate(model, X_test, y_test, out: Path):
    plotting.set_style()
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / "results/classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plotting.save_fig(fig, out / "artifacts/confusion_matrix"); plt.close(fig)

    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("ROC Curve")
        plotting.save_fig(fig, out / "artifacts/roc_curve"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Precision-Recall Curve")
        plotting.save_fig(fig, out / "artifacts/pr_curve"); plt.close(fig)

    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "results/permutation_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(imp))))
        top = imp.head(20)[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("Feature importance (permutation)")
        plotting.save_fig(fig, out / "artifacts/feature_importance"); plt.close(fig)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "tune", "evaluate", "all"], default="all")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--model", default=None, help="Model id for --stage tune (defaults to top of leaderboard)")
    ap.add_argument("--search-library", choices=["sklearn", "optuna"], default="sklearn",
                    help="Tuning backend. 'optuna' requires the optuna package.")
    ap.add_argument("--n-iter", type=int, default=20,
                    help="Number of tuning iterations.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.target)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        stratify=y if y.nunique() < 20 else None,
        random_state=42,
    )

    if args.stage in ("eda", "all"):
        run_eda(df, args.target, out)

    best_model_id = None
    if args.stage in ("compare", "all"):
        lb = compare_models(X_train, y_train, out, cv=args.cv)
        best_model_id = lb.iloc[0]["model"]
    else:
        best_model_id = args.model

    best_model = None
    if args.stage in ("tune", "all") and best_model_id:
        best_model, score = tune_model(
            best_model_id, X_train, y_train, out,
            n_iter=args.n_iter, search_library=args.search_library,
        )
        print(f"Tuned {best_model_id}: CV score = {score}")

    if args.stage in ("evaluate", "all") and best_model is not None:
        best_model.fit(X_train, y_train)
        evaluate(best_model, X_test, y_test, out)
        joblib.dump(best_model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
