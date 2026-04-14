"""Publication-quality clustering pipeline — standalone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402


def _prep(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.empty:
        raise SystemExit("Clustering requires at least one numeric feature.")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return pipe.fit_transform(num), num.columns.tolist()


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
    """Dispatch: GMM uses fit().predict(); others use fit_predict()."""
    if isinstance(model, GaussianMixture):
        return model.fit(X).predict(X)
    return model.fit_predict(X)


def compare(X, out: Path, n_clusters: int):
    zoo = model_zoo.get_zoo(n_clusters=n_clusters)
    rows = []
    labelings = {}
    for mid, entry in zoo.items():
        try:
            model = entry["estimator"]
            labels = _fit_labels(model, X)
            labelings[mid] = labels
            unique = set(labels) - {-1}  # exclude DBSCAN noise
            if len(unique) > 1:
                sil = silhouette_score(X, labels)
            else:
                sil = float("nan")
            rows.append({"model": mid, "n_clusters": len(unique),
                         "silhouette": sil, "noise_points": int((labels == -1).sum())})
        except Exception as e:
            rows.append({"model": mid, "error": str(e)})
    lb = pd.DataFrame(rows).sort_values(by="silhouette", ascending=False, na_position="last")
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    return lb, labelings


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
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True); (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    X, feat_names = _prep(df)

    if args.stage in ("eda", "all"):
        elbow_plot(X, out)

    if args.stage in ("compare", "all"):
        lb, labelings = compare(X, out, args.n_clusters)
        best = lb.iloc[0]["model"]
        if args.stage == "all":
            plot_pca(X, labelings[best], out, best)
            df_out = df.copy(); df_out["cluster"] = labelings[best]
            df_out.to_csv(out / "results/assigned.csv", index=False)
            # Refit the chosen model cleanly for serialization
            fresh = model_zoo.get_zoo(args.n_clusters)[best]["estimator"]
            _fit_labels(fresh, X)
            if not isinstance(fresh, (type(labelings[best]),)):
                pass  # just a type guard
            try:
                joblib.dump(fresh, out / "model.joblib")
            except Exception:
                pass  # DBSCAN etc. may not serialize cleanly

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
