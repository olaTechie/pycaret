"""Publication-quality clustering pipeline — standalone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pickle import PicklingError

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402


def _prep(df: pd.DataFrame, categorical: str = "drop"):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols and categorical == "drop":
        print(f"WARNING: dropping non-numeric columns {cat_cols}. "
              "Pass --categorical one-hot to include them.", flush=True)
    if cat_cols and categorical == "one-hot":
        ct = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("scl", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), cat_cols),
        ], remainder="drop")
        X = ct.fit_transform(df)
        feat_names = list(num_cols) + [f"{c}={v}" for c in cat_cols for v in sorted(df[c].dropna().unique())]
        return X, feat_names
    if not num_cols:
        raise SystemExit("Clustering requires at least one numeric feature (or --categorical one-hot).")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return pipe.fit_transform(df[num_cols]), num_cols


def elbow_plot(X: np.ndarray, out: Path, max_k: int = 10):
    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, "o-")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow plot (KMeans)")
    plotting.save_fig(fig, out / "artifacts/elbow"); plt.close(fig)


def _fit_labels(model, X):
    if isinstance(model, GaussianMixture):
        return model.fit(X).predict(X)
    return model.fit_predict(X)


def compare(X, out: Path, n_clusters: int):
    zoo = model_zoo.get_zoo(n_clusters=n_clusters)
    rows = []
    labelings = {}
    fitted = {}
    for mid, entry in zoo.items():
        try:
            model = entry["estimator"]
            labels = _fit_labels(model, X)
            labelings[mid] = labels
            fitted[mid] = model
            unique = set(labels) - {-1}
            sil = silhouette_score(X, labels) if len(unique) > 1 else float("nan")
            rows.append({"model": mid, "n_clusters": len(unique),
                         "silhouette": sil, "noise_points": int((labels == -1).sum())})
        except Exception as e:
            rows.append({"model": mid, "error": str(e)})
    lb = pd.DataFrame(rows).sort_values(by="silhouette", ascending=False, na_position="last")
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    return lb, labelings, fitted


def plot_pca(X, labels, out: Path, name: str):
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap="tab10", s=18, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"PCA projection — {name}")
    plotting.save_fig(fig, out / "artifacts/pca_scatter"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "assign", "all"], default="all")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--categorical", choices=["drop", "one-hot"], default="drop",
                    help="How to handle non-numeric columns (LEAD-032).")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True); (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    X, feat_names = _prep(df, categorical=args.categorical)

    if args.stage in ("eda", "all"):
        elbow_plot(X, out)

    if args.stage in ("compare", "all"):
        lb, labelings, fitted = compare(X, out, args.n_clusters)
        best = lb.iloc[0]["model"]
        if args.stage == "all":
            plot_pca(X, labelings[best], out, best)
            df_out = df.copy(); df_out["cluster"] = labelings[best]
            df_out.to_csv(out / "results/assigned.csv", index=False)
            try:
                joblib.dump(fitted[best], out / "model.joblib")
            except (TypeError, PicklingError) as e:
                print(f"WARNING: serialization failed for {best}: {e}", flush=True)

    try:
        from _shared.run_manifest import build_manifest, write_manifest
    except ImportError:
        from references._shared.run_manifest import build_manifest, write_manifest
    write_manifest(out / "results", build_manifest(
        stage=args.stage, args_dict=vars(args),
    ))

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
