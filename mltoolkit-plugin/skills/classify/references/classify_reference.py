"""Publication-quality classification pipeline — standalone, no PyCaret.

Stages (via --stage):
    eda       Data overview + EDA figures + EPV audit + Table 1
    compare   Cross-validated comparison of all available models
    tune      RandomizedSearchCV / Optuna on the best model from compare
    evaluate  Holdout metrics + calibration + subgroup + decision curve + SHAP
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
    RocCurveDisplay, PrecisionRecallDisplay, classification_report,
    cohen_kappa_score, confusion_matrix, make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    RandomizedSearchCV, cross_validate, train_test_split, learning_curve,
)
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


# ----- Shared-module import shims (in-place vs staged) ----------------------

def _imp():
    try:
        from _shared import deps, splits, fairness, calibration, bootstrap, \
            decision_curve, epv, table1  # type: ignore
        from _shared.tuning_optuna import optuna_search  # type: ignore
    except ImportError:
        from references._shared import deps, splits, fairness, calibration, \
            bootstrap, decision_curve, epv, table1  # noqa: F401
        from references._shared.tuning_optuna import optuna_search  # noqa: F401
    return dict(deps=deps, splits=splits, fairness=fairness, calibration=calibration,
                bootstrap=bootstrap, decision_curve=decision_curve, epv=epv,
                table1=table1, optuna_search=optuna_search)


# ----- Wrappers ------------------------------------------------------------

def _preprocessor(X, args, sensitive):
    return preprocessing.build_preprocessor(
        X,
        imputation=args.imputation,
        sensitive=sensitive,
        allow_te_on_sensitive=args.allow_target_encode_on_sensitive,
    )


def _pipeline(pre, estimator, *, resample: str):
    if resample == "none":
        return Pipeline([("pre", pre), ("clf", estimator)])
    mods = _imp()
    if not mods["deps"].has_imblearn():
        print(f"WARNING: --resample {resample} requested but imblearn is not installed; "
              "running without resampling.", flush=True)
        return Pipeline([("pre", pre), ("clf", estimator)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    if resample == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
    else:
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=42)
    return ImbPipeline([("pre", pre), ("resample", sampler), ("clf", estimator)])


def _make_cv(y_train, *, cv: int, groups=None, time_order=None):
    mods = _imp()
    return mods["splits"].make_splitter(
        y_train, n_splits=cv, groups=groups, time_order=time_order,
    )


def _groups_and_times(df, X, args):
    """Align group/time columns to the X index."""
    groups = df[args.group_col].loc[X.index] if args.group_col else None
    time_order = (pd.to_datetime(df[args.time_col]).loc[X.index]
                  if args.time_col else None)
    return groups, time_order


# ----- Clinical scorers -----------------------------------------------------

def _ppv_at_recall(y_true, y_proba, target_recall=0.80):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ok = rec >= target_recall
    return float(prec[ok].max() if ok.any() else 0.0)


def _spec_at_recall(y_true, y_proba, target_recall=0.80):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ok = tpr >= target_recall
    return float((1 - fpr[ok]).max() if ok.any() else 0.0)


def _pick_threshold(y_true, y_proba, method, *, fixed_recall=0.80):
    from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve
    thresholds = np.linspace(0.01, 0.99, 99)
    if method == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return {"threshold": float(thr[idx]), "criterion": "youden",
                "score": float(j[idx])}
    if method == "f1":
        scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(scores))
        return {"threshold": float(thresholds[idx]), "criterion": "f1",
                "score": float(scores[idx])}
    if method == "mcc":
        scores = [matthews_corrcoef(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(scores))
        return {"threshold": float(thresholds[idx]), "criterion": "mcc",
                "score": float(scores[idx])}
    if method == "fixed-recall":
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        ok = rec >= fixed_recall
        if not ok.any():
            return {"threshold": 0.5, "criterion": "fixed-recall", "score": float("nan")}
        best = int(np.argmax(prec * ok))
        t = float(thr[best]) if best < len(thr) else 0.5
        return {"threshold": t, "criterion": "fixed-recall",
                "score": float(prec[best])}
    raise ValueError(f"unknown threshold method {method}")


# ----- Stages --------------------------------------------------------------

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

    # Sample-size audit.
    mods = _imp()
    audit = mods["epv"].audit_epv(df.drop(columns=[target]), df[target])
    (out / "results/epv_audit.json").write_text(json.dumps(audit, indent=2))
    if audit["low_epv_warning"]:
        print(f"WARNING: low events-per-variable (EPV={audit['epv']:.2f}).", flush=True)
    if audit["rare_outcome_warning"]:
        print(f"WARNING: rare outcome (prevalence={audit['minority_prevalence']:.3f}).",
              flush=True)

    # Table 1.
    strat = df[target] if df[target].nunique() <= 10 else None
    t1 = mods["table1"].table1(df.drop(columns=[target]), group=strat)
    t1.to_csv(out / "results/table1.csv", index=False)
    (out / "results/table1.md").write_text(reporting.df_to_markdown(t1))

    # Missingness figure.
    miss = df.isna().mean().sort_values(ascending=False)
    if (miss > 0).any():
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * (miss > 0).sum())))
        miss[miss > 0].plot(kind="barh", ax=ax)
        ax.set_xlabel("Missing fraction")
        ax.set_title("Missingness by column")
        plotting.save_fig(fig, out / "artifacts/missingness")
        plt.close(fig)


def compare_models(X_train, y_train, df, out: Path, args, sensitive) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = _preprocessor(X_train, args, sensitive)

    n_classes = len(np.unique(y_train))
    scorers: dict = {
        "accuracy": "accuracy",
        "f1": "f1_macro" if n_classes > 2 else "f1",
    }
    scorers["balanced_accuracy"] = "balanced_accuracy"
    scorers["mcc"] = make_scorer(matthews_corrcoef)
    scorers["kappa"] = make_scorer(cohen_kappa_score)
    if n_classes == 2:
        scorers["roc_auc"] = "roc_auc"
        scorers["average_precision"] = "average_precision"
        scorers["neg_log_loss"] = "neg_log_loss"
        scorers["ppv_at_recall_80"] = make_scorer(_ppv_at_recall, needs_proba=True)
        scorers["spec_at_recall_80"] = make_scorer(_spec_at_recall, needs_proba=True)

    groups, time_order = _groups_and_times(df, X_train, args)
    cv_splitter = _make_cv(y_train, cv=args.cv, groups=groups, time_order=time_order)

    rows = []
    fold_rows = []
    for mid, entry in zoo.items():
        pipe = _pipeline(pre, entry["estimator"], resample=args.resample)
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

    sort_key = "roc_auc" if "roc_auc" in scorers else "f1"
    leaderboard = pd.DataFrame(rows).sort_values(
        by=sort_key, ascending=False, na_position="last",
    ).reset_index(drop=True)
    leaderboard.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(leaderboard))
    pd.DataFrame(fold_rows).to_csv(out / "results/leaderboard_folds.csv", index=False)
    return leaderboard


def tune_model(model_id, X_train, y_train, df, out: Path, args, sensitive):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = _preprocessor(X_train, args, sensitive)
    pipe = _pipeline(pre, entry["estimator"], resample=args.resample)
    grid = {f"clf__{k}": v for k, v in entry["param_grid"].items()}
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
            scoring="accuracy", cv=5, n_iter=n_iter, random_state=42,
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
        cv=cv_splitter, scoring="accuracy", n_jobs=-1,
        random_state=42, refit=True,
    )
    search.fit(X_train, y_train, groups=groups) if groups is not None else search.fit(X_train, y_train)
    (out / "results/best_params.json").write_text(
        json.dumps({k: str(v) for k, v in search.best_params_.items()}, indent=2))
    return search.best_estimator_, search.best_score_


def evaluate(model, X_train, y_train, X_test, y_test, df, out: Path, args, sensitive):
    plotting.set_style()
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / "results/classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plotting.save_fig(fig, out / "artifacts/confusion_matrix"); plt.close(fig)

    binary = len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test)[:, 1] if binary else None

    if binary:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("ROC Curve")
        plotting.save_fig(fig, out / "artifacts/roc_curve"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Precision-Recall Curve")
        plotting.save_fig(fig, out / "artifacts/pr_curve"); plt.close(fig)

    mods = _imp()

    # Calibration.
    if binary:
        cal = mods["calibration"].calibration_summary(y_test, y_proba, n_bins=10)
        (out / "results/calibration.json").write_text(json.dumps(cal, indent=2))
        fig = mods["calibration"].reliability_diagram(y_test, y_proba, n_bins=10)
        plotting.save_fig(fig, out / "artifacts/reliability"); plt.close(fig)

    # Bootstrap CIs.
    if binary and args.bootstrap:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        ci = {
            "accuracy": mods["bootstrap"].bootstrap_ci(
                accuracy_score, y_test, y_pred, n_boot=args.bootstrap, random_state=42),
            "f1": mods["bootstrap"].bootstrap_ci(
                f1_score, y_test, y_pred, n_boot=args.bootstrap, random_state=42),
            "roc_auc": mods["bootstrap"].bootstrap_ci(
                roc_auc_score, y_test, y_proba, n_boot=args.bootstrap, random_state=42),
        }
        (out / "results/holdout_metrics_ci.json").write_text(json.dumps(ci, indent=2))

    # Threshold optimization.
    if binary and args.optimize_threshold != "none":
        picked = _pick_threshold(y_test, y_proba, args.optimize_threshold,
                                 fixed_recall=args.fixed_recall)
        (out / "results/threshold.json").write_text(json.dumps(picked, indent=2))

    # Subgroup + fairness.
    if binary and args.group_col:
        groups_test = df[args.group_col].loc[X_test.index]
        per = mods["fairness"].group_metrics(y_test, y_pred, y_proba, groups_test)
        per.to_csv(out / "results/subgroup_metrics.csv", index=False)
        ratios = mods["fairness"].disparity_ratios(per)
        (out / "results/fairness_disparities.json").write_text(
            json.dumps(ratios, indent=2))

    # Decision curve.
    if binary and args.decision_curve:
        fig = mods["decision_curve"].decision_curve_figure(y_test, y_proba)
        plotting.save_fig(fig, out / "artifacts/decision_curve"); plt.close(fig)

    # Permutation importance.
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5,
                                   random_state=42, n_jobs=-1)
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

    # Learning curve.
    try:
        train_sizes, tr, te = learning_curve(
            model, X_train, y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5), random_state=42,
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, tr.mean(axis=1), "o-", label="Train")
        ax.plot(train_sizes, te.mean(axis=1), "o-", label="CV")
        ax.set_xlabel("Training examples"); ax.set_ylabel("Score")
        ax.set_title("Learning curve"); ax.legend()
        plotting.save_fig(fig, out / "artifacts/learning_curve"); plt.close(fig)
    except Exception:
        pass

    # Class-report heatmap.
    try:
        cr = pd.DataFrame(report).T.drop(columns="support", errors="ignore")
        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(cr))))
        sns.heatmap(cr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        plotting.save_fig(fig, out / "artifacts/classification_report_heatmap")
        plt.close(fig)
    except Exception:
        pass

    # SHAP (optional).
    if binary and mods["deps"]._check("shap"):
        try:
            import shap
            sample = X_test.iloc[:min(200, len(X_test))]
            explainer = shap.Explainer(model.predict, sample)
            sv = explainer(sample)
            fig = plt.figure(figsize=(7, 6))
            shap.plots.beeswarm(sv, show=False)
            plotting.save_fig(fig, out / "artifacts/shap_beeswarm"); plt.close(fig)
        except Exception as e:
            print(f"SHAP failed (continuing): {e}", flush=True)


# ----- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "tune", "evaluate", "all"],
                    default="all")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--model", default=None,
                    help="Model id for --stage tune (defaults to top of leaderboard)")
    ap.add_argument("--search-library", choices=["sklearn", "optuna"], default="sklearn")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--sensitive-features", default="")
    ap.add_argument("--allow-target-encode-on-sensitive", action="store_true")
    ap.add_argument("--imputation", choices=["simple", "iterative", "knn", "drop"],
                    default="simple")
    ap.add_argument("--resample", choices=["none", "smote", "adasyn"], default="none")
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--optimize-threshold",
                    choices=["none", "youden", "f1", "mcc", "fixed-recall"],
                    default="none")
    ap.add_argument("--fixed-recall", type=float, default=0.80)
    ap.add_argument("--decision-curve", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--ensemble", choices=["none", "voting", "stacking"], default="none",
                    help="Build an ensemble from the top-K leaderboard models (LEAD-037).")
    ap.add_argument("--ensemble-k", type=int, default=3)
    ap.add_argument("--finalize", action="store_true",
                    help="Refit best model on X_train ∪ X_test (LEAD-019).")
    ap.add_argument("--track", choices=["none", "mlflow"], default="none",
                    help="Experiment tracking backend (LEAD-033).")
    args = ap.parse_args()
    sensitive = [c.strip() for c in args.sensitive_features.split(",") if c.strip()]

    out = Path(args.output_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    if args.track == "mlflow":
        try:
            import mlflow
            mlflow.set_experiment("mltoolkit")
            mlflow.autolog()
        except ImportError:
            print("WARNING: --track mlflow requested but mlflow is not installed.",
                  flush=True)

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
        lb = compare_models(X_train, y_train, df, out, args, sensitive)
        best_model_id = lb.iloc[0]["model"]
    else:
        best_model_id = args.model

    best_model = None
    if args.stage in ("tune", "all") and best_model_id:
        best_model, score = tune_model(
            best_model_id, X_train, y_train, df, out, args, sensitive,
        )
        print(f"Tuned {best_model_id}: CV score = {score}")

    # Calibration wrap before holdout evaluation.
    if args.calibrate != "none" and best_model is not None and best_model_id is not None:
        from sklearn.calibration import CalibratedClassifierCV
        zoo_entry = model_zoo.get_zoo()[best_model_id]
        pre = _preprocessor(X_train, args, sensitive)
        base = _pipeline(pre, zoo_entry["estimator"], resample=args.resample)
        best_model = CalibratedClassifierCV(base, method=args.calibrate, cv=5)
        best_model.fit(X_train, y_train)

    if args.stage in ("evaluate", "all") and best_model is not None:
        if not hasattr(best_model, "classes_"):
            best_model.fit(X_train, y_train)
        evaluate(best_model, X_train, y_train, X_test, y_test, df, out, args, sensitive)
        joblib.dump(best_model, out / "model.joblib")

        if args.ensemble != "none":
            lb_path = out / "results/leaderboard.csv"
            if lb_path.exists():
                lb_df = pd.read_csv(lb_path)
                top_ids = lb_df["model"].head(args.ensemble_k).tolist()
                zoo = model_zoo.get_zoo()
                pre = _preprocessor(X_train, args, sensitive)
                base_estimators = [
                    (mid, _pipeline(pre, zoo[mid]["estimator"], resample=args.resample))
                    for mid in top_ids if mid in zoo
                ]
                if len(base_estimators) >= 2:
                    if args.ensemble == "voting":
                        from sklearn.ensemble import VotingClassifier
                        ens = VotingClassifier(
                            estimators=base_estimators, voting="soft", n_jobs=-1)
                    else:
                        from sklearn.ensemble import StackingClassifier
                        from sklearn.linear_model import LogisticRegression
                        ens = StackingClassifier(
                            estimators=base_estimators,
                            final_estimator=LogisticRegression(max_iter=1000),
                            n_jobs=-1,
                        )
                    try:
                        ens.fit(X_train, y_train)
                        joblib.dump(ens, out / "model_ensemble.joblib")
                        ens_preds = ens.predict(X_test)
                        ens_score = float((ens_preds == y_test).mean())
                        (out / "results/ensemble_score.json").write_text(
                            json.dumps({"ensemble_kind": args.ensemble,
                                        "members": top_ids,
                                        "holdout_accuracy": ens_score}, indent=2))
                    except Exception as e:
                        print(f"WARNING: ensemble failed: {e}", flush=True)

    # Finalize: refit on full dataset.
    if args.finalize and best_model is not None:
        import datetime as _dt
        final_model = best_model
        final_model.fit(X, y)
        joblib.dump(final_model, out / "model_final.joblib")
        (out / "results/finalize_note.json").write_text(json.dumps({
            "note": "Fit on X_train ∪ X_test (full dataset). Do NOT re-evaluate on holdout.",
            "n_rows": int(len(X)),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }, indent=2))

    # Reproducibility manifest.
    try:
        from _shared.run_manifest import build_manifest, write_manifest
    except ImportError:
        from references._shared.run_manifest import build_manifest, write_manifest
    write_manifest(out / "results", build_manifest(
        stage=args.stage, args_dict=vars(args),
        extra={"best_model_id": best_model_id, "sensitive": sensitive},
    ))

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
