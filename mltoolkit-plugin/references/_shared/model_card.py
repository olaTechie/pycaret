"""Model-card scaffold (Mitchell et al. 2019).

    render_model_card(context) -> str
"""
from __future__ import annotations


def render_model_card(ctx: dict) -> str:
    g = ctx.get
    sections = [
        "# Model card",
        "",
        "## Model details",
        f"- Name: {g('name', '—')}",
        f"- Version: {g('version', '1.0.0')}",
        f"- Type: {g('task', '—')}",
        f"- Primary algorithm: {g('algorithm', '—')}",
        f"- Trained on: {g('timestamp_utc', '—')}",
        "",
        "## Intended use",
        "- <Primary use case>",
        "- <Primary users>",
        "- <Out-of-scope uses>",
        "",
        "## Training data",
        f"- Source: {g('data_source', '—')}",
        f"- Size: {g('n_rows', '—')} rows × {g('n_features', '—')} features",
        f"- Outcome prevalence: {g('prevalence', '—')}",
        f"- Consent / ethics: {g('consent', '—')}",
        "",
        "## Evaluation data",
        "- Holdout (20% stratified random split)",
        "",
        "## Performance",
    ]
    for k, v in (ctx.get("holdout_metrics") or {}).items():
        sections.append(f"- {k}: {v}")
    sections += ["", "## Fairness"]
    fd = ctx.get("fairness_disparities") or {}
    if fd:
        for m, r in fd.items():
            sections.append(f"- {m} disparity ratio (max/min): {r}")
    else:
        sections.append("- Not assessed in this run (no --group-col).")

    sections += [
        "",
        "## Ethical considerations",
        "- <Potential harms and mitigations>",
        "- <Biases detected in data or model>",
        "- <Fairness caveats>",
        "",
        "## Caveats and recommendations",
        "- <Known limitations>",
        "- <Conditions under which model should NOT be used>",
        "",
        "## Contact",
        "- <Maintainer name / email>",
    ]
    return "\n".join(sections) + "\n"
