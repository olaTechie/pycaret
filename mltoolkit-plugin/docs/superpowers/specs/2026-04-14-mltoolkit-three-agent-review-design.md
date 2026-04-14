# Three-Agent Review of the mltoolkit Plugin — Design

**Date:** 2026-04-14
**Status:** Design approved, pending spec review
**Scope:** One-shot orchestration. Produces a review only — no plugin code changes.

## Goal

Produce a prioritized, evidence-backed gap analysis of the `mltoolkit-plugin` from three independent Opus reviewers, so a follow-up session can plan enhancements to:

1. Close spec drift in what the plugin already claims to do (sklearn-native ML toolkit with skills for setup / classify / regress / cluster / anomaly / compare / tune / eda / package, plus the `ml-pipeline` orchestrator agent).
2. Reach PyCaret-equivalent functionality for the same task surface.
3. Add ML-paper reporting capability (methods/results scaffolding, reporting-guideline alignment, reproducibility artifacts).

This session intentionally stops at the gap analysis. Acting on the findings is a follow-up session's job — the review must drive the plan, not the other way around.

## Deliverables

All files land under `docs/superpowers/reviews/` and are git-trackable:

| Path | Produced by | Purpose |
|---|---|---|
| `2026-04-14-pycaret-capability-reference.md` | Main agent (pre-flight) | Shared "ruler" — PyCaret capability surface fetched from Context7. Keeps reviewer measurements consistent. |
| `2026-04-14-review-data-scientist.md` | DS reviewer (Opus) | Structured report, technical-rigor lens. |
| `2026-04-14-review-public-health-analyst.md` | PH analyst reviewer (Opus) | Structured report, applied / equity / reporting-guideline lens. |
| `2026-04-14-lead-synthesis.md` | Lead reviewer (Opus) | Deduped, prioritized backlog + executive summary. |
| `2026-04-14-findings.jsonl` | Lead reviewer (merged) | Machine-readable finding records, deduped across DS + PH — input for the follow-up enhancement plan. |

**Findings flow.** DS and PH reviewers each emit their own findings as a JSONL block inside section 8 of their report. The lead extracts both blocks, dedupes by evidence + affected skill, resolves disagreements, re-tags, re-assigns severity, and writes the single consolidated `2026-04-14-findings.jsonl`. Reviewer findings are traceable via the `reviewer` field on each record.

## Orchestration Flow

```
Step 1 — Pre-fetch PyCaret reference (main agent)
    Context7 resolve-library-id → pycaret
    Context7 query-docs for:
        setup, compare_models, create_model, tune_model,
        plot_model, interpret_model, calibrate_model,
        finalize_model, predict_model, save_model, deploy_model,
        check_fairness, dashboard, get_leaderboard,
        + classification / regression / clustering / anomaly module surfaces
    Write 2026-04-14-pycaret-capability-reference.md

Step 2 — Parallel domain reviewers (single message, 2× Agent calls)
    Agent: Data scientist reviewer    (subagent_type=general-purpose, model=opus)
    Agent: Public-health analyst      (subagent_type=general-purpose, model=opus)
    Each reads plugin + capability reference, writes its own report to disk,
    returns a ≤50-word confirmation.

Step 3 — Lead reviewer (sequential, after both return)
    Agent: Lead reviewer              (subagent_type=general-purpose, model=opus)
    Reads both reviewer reports + capability reference.
    Writes lead-synthesis.md and findings.jsonl.

Step 4 — Verification + summary (main agent)
    Verify all five files exist and are non-empty.
    Parse findings.jsonl, confirm schema.
    Print summary: paths, counts by severity and tag, disagreements resolved.
    Suggest next step: /superpowers:writing-plans against lead-synthesis.md.
```

**Why this shape.** Parallel independence keeps the two domain reviewers from anchoring on each other — the lead's job is exactly to resolve disagreements. Sequential lead is unavoidable: it needs both inputs. A pre-fetched shared reference means both reviewers measure PyCaret gaps against the same ruler, not their own training-data recall.

## Reviewer Contracts

Every reviewer receives a self-contained prompt containing: role, lens, required reads, output path, report template, findings-record schema, and read-only constraint.

### Data scientist — lens

- **Technical correctness:** leakage, CV strategy, preprocessing order inside pipelines, metric selection, stratification, seed handling.
- **Model-zoo completeness** vs. PyCaret per task (classification / regression / clustering / anomaly).
- **Tuning coverage:** search strategies, early stopping, budget controls, Optuna backend claim verification.
- **Interpretability:** SHAP, permutation importance, PDP/ICE, calibration curves, learning curves.
- **Deliverable packaging quality:** reproducibility, env pinning, inference parity between session and packaged artifact.

### Public-health data analyst — lens

- **Applied workflow fit:** Table 1 generation, stratified / subgroup metrics, missing-data handling (MAR/MCAR/MNAR awareness), rare-outcome handling, class imbalance for low-prevalence outcomes, calibration in the small.
- **Ethics / bias / equity auditing (explicit, first-class section):** group fairness metrics, disparate performance across protected attributes, representational harms, documentation of protected-attribute handling, dataset provenance / consent signals.
- **Reporting-guideline alignment:** TRIPOD+AI (prediction models), STARD 2015 (diagnostic accuracy), CONSORT-AI (interventional), PROBAST (risk-of-bias).
- **Paper-ready outputs:** methods-section scaffolding, results tables, calibration + decision-curve plots, supplementary reproducibility appendix.

### Lead reviewer — job

- **Dedupe** findings across DS + PH streams (merge by evidence + affected skill).
- **Resolve disagreements** — note them explicitly, pick a recommendation, justify.
- **Tag** every finding with ≥1 of `current-spec-gap`, `pycaret-target-gap`, `paper-reporting-gap` (a finding can carry multiple).
- **Assign severity** (P0 / P1 / P2) using blast radius × effort × strategic fit with the new target.
- **Write** executive summary (≤500 words) + prioritized backlog table.

### Shared report template (all three reviewers)

```
1. Summary (≤200 words)
2. Current-spec gaps
3. PyCaret-target gaps
4. Paper-reporting gaps
5. [PH analyst only] Ethics / bias / equity audit
6. Prioritized recommendations (P0 / P1 / P2)
7. Open questions for follow-up session
8. Findings (JSONL block — appended to findings.jsonl by lead)
```

### Findings record schema (JSONL)

```json
{
  "id": "DS-001",
  "reviewer": "data-scientist",
  "category": "tuning",
  "severity": "P1",
  "tags": ["pycaret-target-gap"],
  "affected_skill": "mltoolkit:tune",
  "evidence": "skills/tune/SKILL.md:42 — no Optuna backend despite README claim",
  "recommendation": "Add optuna search path gated on import, fallback to RandomizedSearchCV"
}
```

Required fields: `id`, `reviewer`, `category`, `severity`, `tags`, `affected_skill`, `evidence`, `recommendation`. Severity ∈ `{P0, P1, P2}`. Tags ⊆ `{current-spec-gap, pycaret-target-gap, paper-reporting-gap}`.

## Required Reads

**All three reviewers:**

- `README.md` (plugin root)
- `agents/ml-pipeline.md`
- Every `skills/*/SKILL.md` (nine total)
- All files under `skills/*/references/` and `skills/*/scripts/` — reference scripts hold the real behavior
- `tests/` — to see what's actually covered
- `scripts/check-env.sh`
- `pytest.ini`
- `docs/superpowers/reviews/2026-04-14-pycaret-capability-reference.md`

**Lead only, additional:**

- `docs/superpowers/reviews/2026-04-14-review-data-scientist.md`
- `docs/superpowers/reviews/2026-04-14-review-public-health-analyst.md`

**Read-only constraint.** Reviewer prompts explicitly forbid editing plugin files. Reviewers may only write their own report file under `docs/superpowers/reviews/`.

## Error Handling & Verification

### Pre-flight (main agent, before spawning)

- Ensure `docs/superpowers/reviews/` exists (create if missing).
- If Context7 `resolve-library-id` for pycaret fails, abort with a clear message. Do not spawn reviewers against a missing ruler.
- Enumerate required plugin files. If any are missing, abort and report which.

### Per-reviewer robustness

- Hard output contract in every reviewer prompt: "if you cannot complete the review, write a report whose Summary explains why and return — do NOT silently exit." Prevents empty deliverables.
- Reviewer agents return only a short confirmation (path + ≤50-word status). Content lives on disk. Keeps main context light.

### Post-flight (main agent, after lead returns)

- All five files exist and are non-empty.
- `findings.jsonl` parses as valid JSONL; every record has required fields; severities and tags within enums.
- Lead synthesis references both reviewer reports by path.
- Print terminal summary: paths, finding counts (total, by severity, by tag), count of disagreements resolved.
- Suggest next step: run `/superpowers:writing-plans` against `lead-synthesis.md`.

### What we explicitly don't do

- No retries on failed reviewers. Surface the failure, let the user decide.
- No auto-promotion to enhancement implementation. Review drives the plan; the plan is a separate session.

## Success Criteria

- All five deliverable files exist, non-empty, committable.
- `findings.jsonl` valid JSONL with all required fields.
- Every DS and PH finding is either merged into or explicitly superseded by a lead finding — no orphans.
- Lead synthesis tags every finding with ≥1 allowed tag.
- PH analyst report contains a non-empty Ethics / bias / equity section.
- Executive summary ≤500 words; prioritized backlog shows P0 / P1 / P2 counts.

## Out of Scope

- Acting on the findings (follow-up session).
- Any plugin code changes this session.
- Reviewer retries / multi-round adjudication.
- Implementing the PyCaret-equivalent or paper-reporting enhancements themselves.
