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
from scipy import stats as scipy_stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


def _imp():
    try:
        from _shared import deps, splits, bootstrap, epv, table1  # type: ignore
        from _shared.tuning_optuna import optuna_search  # type: ignore
    except ImportError:
        from references._shared import deps, splits, bootstrap, epv, table1  # noqa: F401
        from references._shared.tuning_optuna import optuna_search  # noqa: F401
    return dict(deps=deps, splits=splits, bootstrap=bootstrap, epv=epv,
                table1=table1, optuna_search=optuna_search)


def _preprocessor(X, args, sensitive):
    return preprocessing.build_preprocessor(
        X, imputation=args.imputation, sensitive=sensitive,
        allow_te_on_sensitive=args.allow_target_encode_on_sensitive,
    )


def _groups_and_times(df, X, args):
    groups = df[args.group_col].loc[X.index] if args.group_col else None
    time_order = (pd.to_datetime(df[args.time_col]).loc[X.index]
                  if args.time_col else None)
    return groups, time_order


def _make_cv(y_train, *, cv: int, groups=None, time_order=None):
    return _imp()["splits"].make_splitter(
        y_train, n_splits=cv, groups=groups, time_order=time_order,
    )


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

    # Table 1 (no stratification for continuous targets).
    mods = _imp()
    t1 = mods["table1"].table1(df.drop(columns=[target]))
    t1.to_csv(out / "results/table1.csv", index=False)
    (out / "results/table1.md").write_text(reporting.df_to_markdown(t1))

    # Missingness figure.
    miss = df.isna().mean().sort_values(ascending=False)
    if (miss > 0).any():
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * (miss > 0).sum())))
        miss[miss > 0].plot(kind="barh", ax=ax)
        ax.set_xlabel("Missing fraction"); ax.set_title("Missingness by column")
        plotting.save_fig(fig, out / "artifacts/missingness"); plt.close(fig)


def compare_models(X_train, y_train, df, out: Path, args, sensitive) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = _preprocessor(X_train, args, sensitive)

    scorers = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "explained_variance": "explained_variance",
        "neg_mape": "neg_mean_absolute_percentage_error",
    }
    # RMSLE requires non-negative targets.
    if (y_train >= 0).all():
        scorers["neg_rmsle"] = "neg_mean_squared_log_error"

    groups, time_order = _groups_and_times(df, X_train, args)
    cv_splitter = _make_cv(y_train, cv=args.cv, groups=groups, time_order=time_order)

    rows, fold_rows = [], []
    for mid, entry in zoo.items():
        pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
        row = {"model": mid}
        try:
            res = cross_validate(
                pipe, X_train, y_train, cv=cv_splitter, scoring=scorers,
                groups=groups, n_jobs=-1, return_train_score=False,
            )
            for name in scorers:
                arr = res[f"test_{name}"]
                row[name] = float(np.mean(arr))
                row[f"{name}_std"] = float(np.std(arr))
                for k, v in enumerate(arr):
                    fold_rows.append({"model": mid, "metric": name,
                                      "fold": k, "score": float(v)})
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    lb = pd.DataFrame(rows).sort_values(
        by="r2", ascending=False, na_position="last").reset_index(drop=True)
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    pd.DataFrame(fold_rows).to_csv(out / "results/leaderboard_folds.csv", index=False)
    return lb


def tune_model(model_id, X_train, y_train, df, out: Path, args, sensitive):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = _preprocessor(X_train, args, sensitive)
    pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
    grid = {f"reg__{k}": v for k, v in entry["param_grid"].items()}
    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, None

    mods = _imp()
    n_iter, search_library = args.n_iter, args.search_library

    if search_library == "optuna" and not mods["deps"].has_optuna():
        print("WARNING: --search-library optuna requested but optuna is "
              "not installed; falling back to RandomizedSearchCV.", flush=True)
        search_library = "sklearn"

    if search_library == "optuna":
        best, best_score, best_params = mods["optuna_search"](
            pipe, grid, X_train, y_train,
            scoring="r2", cv=5, n_iter=n_iter, random_state=42,
        )
        (out / "results/best_params.json").write_text(
            json.dumps({k: str(v) for k, v in best_params.items()}, indent=2))
        return best, best_score

    groups, time_order = _groups_and_times(df, X_train, args)
    cv_splitter = _make_cv(y_train, cv=5, groups=groups, time_order=time_order)

    max_combos = 1
    for v in grid.values():
        max_combos *= len(v)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, max_combos),
        cv=cv_splitter, scoring="r2", n_jobs=-1,
        random_state=42, refit=True,
    )
    if groups is not None:
        search.fit(X_train, y_train, groups=groups)
    else:
        search.fit(X_train, y_train)
    (out / "results/best_params.json").write_text(
        json.dumps({k: str(v) for k, v in search.best_params_.items()}, indent=2))
    return search.best_estimator_, search.best_score_


def evaluate(model, X_test, y_test, out: Path, args):
    plotting.set_style()
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
    }
    (out / "results/holdout_metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.bootstrap:
        mods = _imp()
        ci = {
            "r2":   mods["bootstrap"].bootstrap_ci(
                r2_score, y_test, y_pred, n_boot=args.bootstrap, random_state=42),
            "mae":  mods["bootstrap"].bootstrap_ci(
                mean_absolute_error, y_test, y_pred,
                n_boot=args.bootstrap, random_state=42),
        }
        (out / "results/holdout_metrics_ci.json").write_text(
            json.dumps(ci, indent=2))

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

    fig, ax = plt.subplots(figsize=(6, 5))
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of residuals")
    plotting.save_fig(fig, out / "artifacts/qq_plot"); plt.close(fig)

    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5,
                                   random_state=42, n_jobs=-1)
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
    ap.add_argument("--stage", choices=["eda", "compare", "tune", "evaluate", "all"],
                    default="all")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--search-library", choices=["sklearn", "optuna"], default="sklearn")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--sensitive-features", default="")
    ap.add_argument("--allow-target-encode-on-sensitive", action="store_true")
    ap.add_argument("--imputation", choices=["simple", "iterative", "knn", "drop"],
                    default="simple")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--ensemble", choices=["none", "voting", "stacking"], default="none")
    ap.add_argument("--ensemble-k", type=int, default=3)
    args = ap.parse_args()
    sensitive = [c.strip() for c in args.sensitive_features.split(",") if c.strip()]

    out = Path(args.output_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.target)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Skew-aware stratified split (LEAD-039).
    stratify = None
    if abs(float(scipy_stats.skew(y))) > 1.0 and len(y) > 100:
        try:
            stratify = pd.qcut(y, 10, duplicates="drop")
        except ValueError:
            stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify,
    )

    if args.stage in ("eda", "all"):
        run_eda(df, args.target, out)

    best_id = None
    if args.stage in ("compare", "all"):
        lb = compare_models(X_train, y_train, df, out, args, sensitive)
        best_id = lb.iloc[0]["model"]
    else:
        best_id = args.model

    best_model = None
    if args.stage in ("tune", "all") and best_id:
        best_model, score = tune_model(
            best_id, X_train, y_train, df, out, args, sensitive,
        )
        print(f"Tuned {best_id}: CV R² = {score}")

    if args.stage in ("evaluate", "all") and best_model is not None:
        best_model.fit(X_train, y_train)
        evaluate(best_model, X_test, y_test, out, args)
        joblib.dump(best_model, out / "model.joblib")

        if args.ensemble != "none":
            lb_path = out / "results/leaderboard.csv"
            if lb_path.exists():
                lb_df = pd.read_csv(lb_path)
                top_ids = lb_df["model"].head(args.ensemble_k).tolist()
                zoo = model_zoo.get_zoo()
                pre = _preprocessor(X_train, args, sensitive)
                base = [
                    (mid, Pipeline([("pre", pre), ("reg", zoo[mid]["estimator"])]))
                    for mid in top_ids if mid in zoo
                ]
                if len(base) >= 2:
                    if args.ensemble == "voting":
                        from sklearn.ensemble import VotingRegressor
                        ens = VotingRegressor(estimators=base, n_jobs=-1)
                    else:
                        from sklearn.ensemble import StackingRegressor
                        from sklearn.linear_model import Ridge
                        ens = StackingRegressor(
                            estimators=base, final_estimator=Ridge(), n_jobs=-1,
                        )
                    try:
                        ens.fit(X_train, y_train)
                        joblib.dump(ens, out / "model_ensemble.joblib")
                        (out / "results/ensemble_score.json").write_text(
                            json.dumps({"ensemble_kind": args.ensemble,
                                        "members": top_ids,
                                        "holdout_r2": float(r2_score(
                                            y_test, ens.predict(X_test)))},
                                       indent=2))
                    except Exception as e:
                        print(f"WARNING: ensemble failed: {e}", flush=True)

    try:
        from _shared.run_manifest import build_manifest, write_manifest
    except ImportError:
        from references._shared.run_manifest import build_manifest, write_manifest
    write_manifest(out / "results", build_manifest(
        stage=args.stage, args_dict=vars(args),
        extra={"best_model_id": best_id, "sensitive": sensitive},
    ))

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
