"""Publication-quality anomaly detection pipeline — standalone."""
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
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
        raise SystemExit("Anomaly detection requires numeric features.")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return pipe.fit_transform(num), num.columns.tolist()


def score(X, model_id: str, contamination: float):
    entry = model_zoo.get_zoo(contamination)[model_id]
    est = entry["estimator"]
    if isinstance(est, LocalOutlierFactor):
        labels = est.fit_predict(X)
        scores = est.negative_outlier_factor_
    else:
        est.fit(X)
        labels = est.predict(X)
        scores = est.score_samples(X) if hasattr(est, "score_samples") else est.decision_function(X)
    is_anomaly = (labels == -1).astype(int)
    return est, scores, is_anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "assign", "all"], default="all")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--model", default="iforest")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True); (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    X, feat_names = _prep(df)

    if args.stage in ("compare", "all"):
        rows = []
        for mid in model_zoo.get_zoo(args.contamination):
            try:
                _, _, is_anom = score(X, mid, args.contamination)
                rows.append({"model": mid, "n_anomalies": int(is_anom.sum()),
                             "anomaly_rate": float(is_anom.mean())})
            except Exception as e:
                rows.append({"model": mid, "error": str(e)})
        lb = pd.DataFrame(rows)
        lb.to_csv(out / "results/leaderboard.csv", index=False)
        (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))

    model, scores, is_anom = score(X, args.model, args.contamination)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=50, alpha=0.7)
    ax.set_xlabel("Anomaly score"); ax.set_title(f"Score distribution — {args.model}")
    plotting.save_fig(fig, out / "artifacts/score_histogram"); plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.where(is_anom == 1, "red", "steelblue")
    ax.scatter(Xp[:, 0], Xp[:, 1], c=colors, s=18, alpha=0.7)
    ax.set_title(f"Anomalies (red) vs normal — {args.model}")
    plotting.save_fig(fig, out / "artifacts/pca_anomaly_scatter"); plt.close(fig)

    out_df = df.copy(); out_df["anomaly_score"] = scores; out_df["is_anomaly"] = is_anom
    out_df.to_csv(out / "results/scores.csv", index=False)
    top = out_df.sort_values("anomaly_score").head(20)
    top.to_csv(out / "results/top_anomalies.csv", index=False)

    if not isinstance(model, LocalOutlierFactor):
        joblib.dump(model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
