"""Publication-quality regression pipeline — standalone."""
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in columns: {list(df.columns)}")
    return df


def run_eda(df: pd.DataFrame, target: str, out: Path):
    plotting.set_style()
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "nunique": df.nunique(),
    }).to_csv(out / "results/schema.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[target], kde=True, ax=ax)
    ax.set_title(f"Target distribution — {target}")
    plotting.save_fig(fig, out / "artifacts/target_distribution"); plt.close(fig)

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap"); plt.close(fig)


def compare_models(X_train, y_train, out: Path, cv: int = 5) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = preprocessing.build_preprocessor(X_train)

    rows = []
    for mid, entry in zoo.items():
        pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
        row = {"model": mid}
        try:
            r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            neg_rmse = cross_val_score(pipe, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error", n_jobs=-1)
            neg_mae = cross_val_score(pipe, X_train, y_train, cv=cv,
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
            row["r2"] = float(r2.mean()); row["r2_std"] = float(r2.std())
            row["rmse"] = float(-neg_rmse.mean()); row["mae"] = float(-neg_mae.mean())
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    lb = pd.DataFrame(rows).sort_values(by="r2", ascending=False, na_position="last").reset_index(drop=True)
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    return lb


def tune_model(model_id: str, X_train, y_train, out: Path, n_iter: int = 20):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = preprocessing.build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
    grid = {f"reg__{k}": v for k, v in entry["param_grid"].items()}
    if not grid:
        pipe.fit(X_train, y_train); return pipe, None
    max_combos = 1
    for v in grid.values():
        max_combos *= len(v)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, max_combos),
        cv=5, scoring="r2", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    with open(out / "results/best_params.json", "w") as f:
        json.dump({k: str(v) for k, v in search.best_params_.items()}, f, indent=2)
    return search.best_estimator_, search.best_score_


def evaluate(model, X_test, y_test, out: Path):
    plotting.set_style()
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    with open(out / "results/holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residual"); ax.set_title("Residuals vs Fitted")
    plotting.save_fig(fig, out / "artifacts/residuals_vs_fitted"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Prediction vs Actual")
    plotting.save_fig(fig, out / "artifacts/prediction_vs_actual"); plt.close(fig)

    from scipy import stats as scipy_stats
    fig, ax = plt.subplots(figsize=(6, 5))
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of residuals")
    plotting.save_fig(fig, out / "artifacts/qq_plot"); plt.close(fig)

    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({"feature": X_test.columns,
                            "importance_mean": r.importances_mean,
                            "importance_std": r.importances_std}
                          ).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "results/permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(imp))))
        top = imp.head(20)[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_xlabel("Permutation importance"); ax.set_title("Feature importance")
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
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.target)
    X = df.drop(columns=[args.target]); y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.stage in ("eda", "all"):
        run_eda(df, args.target, out)

    best_id = None
    if args.stage in ("compare", "all"):
        lb = compare_models(X_train, y_train, out, cv=args.cv)
        best_id = lb.iloc[0]["model"]
    else:
        best_id = args.model

    best_model = None
    if args.stage in ("tune", "all") and best_id:
        best_model, score = tune_model(best_id, X_train, y_train, out)
        print(f"Tuned {best_id}: CV R² = {score}")

    if args.stage in ("evaluate", "all") and best_model is not None:
        best_model.fit(X_train, y_train)
        evaluate(best_model, X_test, y_test, out)
        joblib.dump(best_model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
