"""Reporting-guideline checklists: TRIPOD+AI, STARD, CONSORT-AI scaffolds.

Each function returns a markdown table. Items marked `?` by default —
the author fills in "yes/no/partial" + a location pointer.
"""
from __future__ import annotations


def _render(rows, title: str) -> str:
    out = [f"# {title}", "",
           "| # | Item | Reported? | Where |",
           "|---|---|---|---|"]
    for i, (item, desc) in enumerate(rows, 1):
        out.append(f"| {i} | **{item}** — {desc} | ? | ? |")
    return "\n".join(out) + "\n"


def tripod_ai_checklist() -> str:
    items = [
        ("Title", "identify AI/ML prediction model"),
        ("Abstract", "structured summary of methods + performance"),
        ("Source of data", "design, data sources, dates"),
        ("Participants", "eligibility, setting"),
        ("Outcome", "definition, prediction horizon"),
        ("Predictors", "how measured, data types"),
        ("Sample size", "rationale, events-per-variable"),
        ("Missing data", "handling strategy"),
        ("Model development", "algorithm(s), hyperparameters"),
        ("Model evaluation", "split, resampling, metrics"),
        ("Calibration", "method, plots"),
        ("Fairness", "subgroups, disparity metrics"),
        ("Interpretability", "feature importance, SHAP, limitations"),
        ("Risk of bias (PROBAST)", "per-domain assessment"),
        ("Reproducibility", "code, data, environment available"),
    ]
    return _render(items, "TRIPOD+AI checklist")


def stard_checklist() -> str:
    items = [
        ("Study design", "cross-sectional / longitudinal"),
        ("Reference standard", "how applied, time relative to index test"),
        ("Index test", "how performed, blinding"),
        ("Participants", "eligibility, recruitment, setting"),
        ("Sample size", "rationale"),
        ("Analysis", "2x2 table, sensitivity, specificity, LR+, LR-"),
        ("Uncertainty", "95% CIs for estimates"),
        ("Flow diagram", "participants from eligibility through analysis"),
    ]
    return _render(items, "STARD 2015 checklist")


def consort_ai_checklist() -> str:
    items = [
        ("AI intervention", "version, inputs, outputs, integration"),
        ("Handling of errors", "performance, failure modes"),
        ("Inclusion/exclusion", "population, setting"),
        ("Randomization", "sequence generation, allocation concealment"),
        ("Blinding", "of participants/outcome assessors"),
        ("Outcomes", "primary + secondary, how measured"),
        ("Statistical methods", "intent-to-treat, per-protocol"),
        ("Participants flow", "enrollment through analysis"),
        ("Harms", "adverse events, off-target actions of AI"),
    ]
    return _render(items, "CONSORT-AI checklist")
