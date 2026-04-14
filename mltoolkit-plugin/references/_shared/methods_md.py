"""TRIPOD+AI-aligned methods-section scaffold.

    render_methods(context) -> str

context is a dict with optional keys:
    data_source, n_rows, n_features, prevalence, epv,
    preprocessing, imputation, resampling,
    split_strategy, cv,
    zoo_ids, tuning,
    holdout_metrics, holdout_ci,
    fairness_disparities, calibration,
    packages (version dict), python_version, timestamp_utc
"""
from __future__ import annotations

from typing import Any


def _val(d: dict, key: str, default: Any = "—") -> Any:
    v = d.get(key)
    return default if v is None else v


def render_methods(ctx: dict) -> str:
    lines = [
        "# Methods",
        "",
        "## 1. Data source and cohort",
        f"- Source: {_val(ctx, 'data_source')}",
        f"- Sample size: {_val(ctx, 'n_rows')} rows × {_val(ctx, 'n_features')} features",
        f"- Outcome prevalence: {_val(ctx, 'prevalence')}",
        f"- Events-per-variable (EPV): {_val(ctx, 'epv')}",
        "",
        "## 2. Preprocessing",
        f"- Imputation: {_val(ctx, 'imputation')}",
        f"- Encoding: {_val(ctx, 'preprocessing')}",
        f"- Resampling: {_val(ctx, 'resampling')}",
        "",
        "## 3. Split strategy",
        f"- Holdout: 20% (random seed 42)",
        f"- CV during training: {_val(ctx, 'cv')}-fold ({_val(ctx, 'split_strategy')})",
        "",
        "## 4. Model search space",
        f"- Zoo members: {', '.join(ctx.get('zoo_ids') or []) or '—'}",
        "",
        "## 5. Tuning procedure",
        f"- {_val(ctx, 'tuning')}",
        "",
        "## 6. Performance metrics (holdout)",
    ]
    for k, v in (ctx.get("holdout_metrics") or {}).items():
        lines.append(f"- {k}: {v}")
    if ctx.get("holdout_ci"):
        lines.append("")
        lines.append("### 95% bootstrap CIs")
        for metric, rec in ctx["holdout_ci"].items():
            lines.append(f"- {metric}: {rec.get('point', '—')} "
                         f"[{rec.get('lower', '—')}, {rec.get('upper', '—')}]")

    lines += ["", "## 7. Fairness analysis"]
    fd = ctx.get("fairness_disparities") or {}
    if fd:
        for m, r in fd.items():
            lines.append(f"- {m} disparity ratio (max/min): {r}")
    else:
        lines.append("- Not performed in this run.")

    lines += ["", "## 8. Calibration"]
    cal = ctx.get("calibration") or {}
    if cal:
        lines.append(f"- Brier: {cal.get('brier')}")
        lines.append(f"- ECE: {cal.get('ece')}")
        lines.append(f"- Calibration intercept: {cal.get('intercept')}")
        lines.append(f"- Calibration slope: {cal.get('slope')}")
    else:
        lines.append("- Not performed in this run.")

    lines += ["", "## 9. Software"]
    lines.append(f"- Python: {_val(ctx, 'python_version')}")
    for k, v in (ctx.get("packages") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- Run timestamp (UTC): {_val(ctx, 'timestamp_utc')}")

    return "\n".join(lines) + "\n"
