"""Generic data-loading + EDA helper. Task-agnostic — run first in a session."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
from references._shared import plotting  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--output-dir", default=".mltoolkit")
    args = ap.parse_args()

    out = Path(args.output_dir); (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns from {args.data}")
    print(df.head())

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    summary.to_csv(out / "results/schema.csv")
    print("\nSchema:\n", summary)

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap"); plt.close(fig)

    if args.target and args.target in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        if df[args.target].dtype.kind in "iufc" and df[args.target].nunique() > 20:
            sns.histplot(df[args.target], kde=True, ax=ax)
            ax.set_title(f"Target distribution — {args.target} (continuous)")
        else:
            df[args.target].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Target distribution — {args.target} (categorical)")
        plotting.save_fig(fig, out / "artifacts/target_distribution"); plt.close(fig)

    print(f"\nDone. Artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
