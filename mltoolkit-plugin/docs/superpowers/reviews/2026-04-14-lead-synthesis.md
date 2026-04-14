# Lead Synthesis — mltoolkit-plugin review (2026-04-14)

Lead reviewer synthesis of the data-scientist (DS, 36 findings) and public-health-analyst (PH, 27 findings) parallel reviews of `mltoolkit-plugin` at `master` (6f40561f). 63 source findings consolidated into 41 lead findings (22 merges, 12 disagreements resolved). All severities re-assigned using blast-radius × effort × strategic fit with the PyCaret-parity + clinical-prediction-paper target.

## Executive summary

**Three clusters dominate.** (1) **Correctness / packaging** — the copy-to-`.mltoolkit/session.py` workflow every SKILL claims is silently broken by sibling-module imports (`preprocessing`, `model_zoo`, `_shared`) and by `_PLUGIN_ROOT = _HERE.parents[3]` resolving outside the plugin once copied. Tests run in place so CI never hits it. README and `skills/tune/SKILL.md` advertise an `optuna` backend with zero code. (2) **Clinical / equity reporting** — no group-fairness surface, no calibration, no bootstrap CIs, no decision-curve, no Table 1, no subgroup stratification, no TRIPOD+AI checklist, no run manifest, no ethics/provenance prompts, and a random 80/20 split with no `GroupKFold` / `StratifiedGroupKFold` / `TimeSeriesSplit` option. Protected attributes (`race`, `zip_code`) flow through `TargetEncoder`, leaking outcome into proxy-discrimination vectors. (3) **PyCaret parity** — narrower zoos in all four tasks (esp. regression robust regressors and anomaly PyOD), no `interpret_model` / SHAP, no `calibrate_model`, no `finalize_model`, no `optimize_threshold`, no `check_fairness`, no experiment-tracking hook, no ensembling, no `get_leaderboard` / per-fold scores.

**Biggest risks to current spec.** LEAD-001 (broken copy-to-session.py) and LEAD-002 (phantom optuna backend) are both P0: users following the documented workflow get a failure or a missing feature with no diagnostic. LEAD-023 (tests don't cover the copy path) is the reason these slipped through. LEAD-021 (cluster refit-for-serialization drift + silent exception swallowing) risks publishing irreproducible cohort labels.

**Biggest gaps vs. PyCaret.** The strategic gap is not any single missing function but the absence of a **fairness/interpretability/calibration/finalize quartet** that PyCaret ships as first-class: `check_fairness`, `interpret_model` (SHAP), `calibrate_model`, `finalize_model`, plus `optimize_threshold`. These map to LEAD-003, LEAD-010, LEAD-004, LEAD-019, LEAD-024. Secondary zoo/metric breadth (LEAD-013..LEAD-017, LEAD-035..LEAD-036) is lower-risk but increases the count of PyCaret gaps and the count of plots/metrics a clinical paper cannot currently cite.

**Biggest gaps vs. ML-paper reporting.** A TRIPOD+AI / STARD / CONSORT-AI-grade paper cannot be written from current `.mltoolkit/results/` output: no calibration (LEAD-004), no 95% CIs (LEAD-005), no Table 1 (LEAD-027), no subgroup metrics (LEAD-026), no methods scaffold (LEAD-028), no decision-curve (LEAD-029), no reproducibility manifest (LEAD-018), no EPV check (LEAD-012), no missingness audit (LEAD-025), no model card (LEAD-040), no STARD/CONSORT checklist (LEAD-041). The plugin is a capable modelling tool today but a thin reporting tool.

**Top three recommended next actions.**

1. **Unblock the P0s before any user adoption.** Fix LEAD-001 by inlining `preprocessing.py` + `model_zoo.py` (or stage-copying `references/`) and add the missing CI coverage in LEAD-023. Either implement optuna (LEAD-002) or delete the claim. These are ~1–2 days each and restore truth-in-advertising.
2. **Ship the equity + calibration + CI P0 cluster.** LEAD-003 (fairness helper), LEAD-004 (calibration), LEAD-005 (bootstrap CIs + per-fold), LEAD-006 (group/time splits), LEAD-007 (setup ethics prompt), LEAD-008 (target-encoder guard). This is one bounded feature workstream (~1 week) that converts the plugin from "generic ML demo" to "paper-grade clinical-prediction toolkit" and closes the biggest PyCaret + TRIPOD+AI gaps simultaneously.
3. **Make Tier-B/C paper-ready.** Combine LEAD-018 (run manifest), LEAD-019 (`finalize_model`), LEAD-027 (Table 1), LEAD-028 (methods + TRIPOD+AI checklist), LEAD-034 (pinned requirements), LEAD-040 (model card). This lets Tier C land a supplementary-appendix-complete bundle that is the realistic differentiator vs. a notebook workflow.

## Prioritized backlog

| ID | severity | tags | affected_skill | one-line recommendation |
|---|---|---|---|---|
| LEAD-001 | P0 | current-spec-gap | mltoolkit:package | Inline sibling modules (or stage-copy `references/`) so copied session.py runs standalone; add CI that copies + executes. |
| LEAD-002 | P0 | current-spec-gap, pycaret-target-gap | mltoolkit:tune | Implement an optuna tuner gated on `deps.has_optuna()` or remove the advertising copy. |
| LEAD-003 | P0 | current-spec-gap, pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add `references/_shared/fairness.py` + `--group-col` producing per-group metrics and disparities. |
| LEAD-004 | P0 | current-spec-gap, pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add calibration helper (reliability, Brier, ECE, intercept/slope) + `--calibrate` CalibratedClassifierCV wrap. |
| LEAD-005 | P0 | current-spec-gap, paper-reporting-gap | mltoolkit:classify | Emit bootstrap 95% CIs + per-fold scores; write `holdout_metrics_ci.json` and `leaderboard_folds.csv`. |
| LEAD-006 | P0 | current-spec-gap | mltoolkit:classify | Add `--group-col` / `--time-col` routing to GroupKFold / StratifiedGroupKFold / TimeSeriesSplit. |
| LEAD-007 | P0 | current-spec-gap | mltoolkit:setup | Prompt for IRB/consent, protected-attribute list, data provenance; emit `.mltoolkit/datasheet.md`. |
| LEAD-008 | P0 | current-spec-gap | mltoolkit:classify | Refuse TargetEncoder on protected-attribute / high-cardinality geographic columns without explicit override. |
| LEAD-009 | P1 | current-spec-gap | mltoolkit:tune | Promote `n_iter` / `scoring` / `cv-tune` / `time-budget` to CLI flags; default `--tune-metric` to compare's ranking metric. |
| LEAD-010 | P1 | current-spec-gap, pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add SHAP (gated) + PartialDependenceDisplay + calibration_curve + learning_curve to evaluate. |
| LEAD-011 | P1 | current-spec-gap | mltoolkit:classify | Implement `--resample {smote,adasyn,none}` via imblearn.pipeline.Pipeline. |
| LEAD-012 | P1 | current-spec-gap, paper-reporting-gap | mltoolkit:eda | Compute minority prevalence + events-per-variable; warn when EPV<10 or prevalence<5%. |
| LEAD-013 | P1 | pycaret-target-gap | mltoolkit:classify | Expand classification zoo with qda, lda, dummy, par, MultinomialNB, BernoulliNB. |
| LEAD-014 | P1 | pycaret-target-gap | mltoolkit:regress | Expand regression zoo with robust regressors and dummy baseline (huber, ransac, theilsen, br, ard, ..., dummy). |
| LEAD-015 | P1 | pycaret-target-gap | mltoolkit:cluster | Add AffinityPropagation, MeanShift, SpectralClustering, OPTICS, Birch; gate kmodes. |
| LEAD-016 | P1 | pycaret-target-gap | mltoolkit:anomaly | Add pca (reconstruction) and MinCovDet sklearn-natively; gate abod/hbos/cof/sod/sos via pyod. |
| LEAD-017 | P1 | pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add learning_curve, calibration_curve, class_report heatmap, cumulative gain/lift to evaluate. |
| LEAD-018 | P1 | current-spec-gap, paper-reporting-gap | mltoolkit:package | Write `run_manifest.json` (versions/seed/split/CV/hyperparams, JSON-typed) per stage; pin requirements via importlib.metadata. |
| LEAD-019 | P1 | current-spec-gap, pycaret-target-gap | mltoolkit:package | Add `--finalize` stage refitting on X_train ∪ X_test; emit `model_final.joblib`. |
| LEAD-020 | P1 | current-spec-gap | mltoolkit:anomaly | Fix LOF non-serialization (novelty=True or load_hint.json refit-on-load); sync Tier-B/C predict docs. |
| LEAD-021 | P1 | current-spec-gap | mltoolkit:cluster | Save the actually-fitted estimator, log serialization exceptions to stderr, honor random_state on any re-fit path. |
| LEAD-022 | P1 | current-spec-gap | mltoolkit:package | Tier-C transform should emit the chosen estimator into `src/train.py` so inference parity is code-enforced. |
| LEAD-023 | P1 | current-spec-gap | mltoolkit:package | Add CI that copies session.py to a tmp dir and runs it; add a Tier-A package-and-execute test. |
| LEAD-024 | P1 | pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add `--optimize-threshold {youden,f1,mcc,cost,fixed-recall}` writing `threshold.json` and annotated curves. |
| LEAD-025 | P1 | current-spec-gap, paper-reporting-gap | mltoolkit:classify | Add `--imputation {simple,iterative,knn,drop,multiple}`; emit missingness pattern figure + audit. |
| LEAD-026 | P1 | current-spec-gap, paper-reporting-gap | mltoolkit:classify | Emit `subgroup_metrics.csv` (AUC/PR/PPV/NPV + n/prevalence per group) when `--group-col` is passed. |
| LEAD-027 | P1 | paper-reporting-gap | mltoolkit:eda | Generate `results/table1.csv` (mean±SD / N(%) + SMD, by outcome) from `run_eda`. |
| LEAD-028 | P1 | paper-reporting-gap | mltoolkit:package | Tier B/C: emit `reports/methods.md` + `reports/tripod_ai_checklist.md`. |
| LEAD-029 | P1 | pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add decision-curve / net-benefit plot via `--decision-curve`. |
| LEAD-030 | P1 | current-spec-gap | mltoolkit:anomaly | Add ethics warning + `--group-col` anomaly-rate disparity report to the anomaly skill. |
| LEAD-031 | P1 | paper-reporting-gap | mltoolkit:classify | Extend leaderboard with PPV@recall-0.8, specificity@recall-0.8, Brier; expose `--leaderboard-sort`. |
| LEAD-032 | P1 | current-spec-gap | mltoolkit:cluster | Warn + `--categorical {drop,one-hot}` so cluster/anomaly do not silently drop indicator-encoded protected attributes. |
| LEAD-033 | P2 | pycaret-target-gap | mltoolkit:setup | Add optional `--log-mlflow` / `--track {none,mlflow}` hook. |
| LEAD-034 | P2 | current-spec-gap | mltoolkit:package | Pin packaged requirements via `importlib.metadata.version`; add `--pin {exact,compatible,none}`. |
| LEAD-035 | P2 | pycaret-target-gap, paper-reporting-gap | mltoolkit:classify | Add MCC, kappa, balanced_accuracy, logloss, average_precision to leaderboard. |
| LEAD-036 | P2 | pycaret-target-gap, paper-reporting-gap | mltoolkit:regress | Add MAPE, RMSLE (guarded), explained_variance to leaderboard. |
| LEAD-037 | P2 | pycaret-target-gap | mltoolkit:classify | Add `--ensemble {voting,stacking}` consuming top-k from leaderboard. |
| LEAD-038 | P2 | current-spec-gap | mltoolkit:eda | Wire `reporting.summary_report` as the evaluate-stage default; emit `report.html`. |
| LEAD-039 | P2 | current-spec-gap | mltoolkit:regress | Auto-switch to `stratify=pd.qcut(y, 10)` when `|skew|>1` in EDA. |
| LEAD-040 | P2 | current-spec-gap, paper-reporting-gap | mltoolkit:package | Extend Tier B README with Intended use / Out-of-scope / Re-id / Harms / Provenance (minimal model card). |
| LEAD-041 | P2 | paper-reporting-gap | mltoolkit:package | Tier B/C: emit STARD (`--diagnostic`) / CONSORT-AI (`--interventional`) checklist stubs. |

## Disagreements resolved

- **LEAD-003 (fairness helper)** — DS P2 vs. PH P0. Chose P0: the plugin's stated audience (clinical-prediction / public-health) makes fairness a first-class requirement, not a nice-to-have, and the same feature closes the largest PyCaret-target gap simultaneously.
- **LEAD-004 (calibration)** — DS P1 vs. PH P0. Chose P0: calibration is both a PyCaret parity item and a non-negotiable TRIPOD+AI reporting item; postponing it to P1 undercuts the paper-ready claim.
- **LEAD-005 (bootstrap CIs + per-fold)** — DS P2 vs. PH P0. Chose P0: paper-ready reporting is impossible without CIs on holdout metrics; effort is moderate and unblocks every downstream reporting claim.
- **LEAD-006 (group / time splits)** — DS P1 vs. PH P0. Chose P0: random-split leakage directly falsifies holdout numbers for longitudinal / multi-site clinical data, which is the plugin's target domain.
- **LEAD-019 (finalize_model)** — DS P1 vs. PH P2. Chose P1: small effort, needed for Tier-C deployment-grade artifacts and expected by TRIPOD+AI.
- **LEAD-021 (cluster serialization + reproducibility)** — DS P2 vs. PH P1. Chose P1: PH's reproducibility framing (labels-may-differ-on-reload) outweighs DS's pure exception-swallowing read for a cohort-assignment study.
- **LEAD-024 (optimize_threshold)** — DS P2 vs. PH P1. Chose P1: threshold moving is standard for screening/triage papers in the plugin's target audience.
- **LEAD-025 (imputation options)** — DS P2 vs. PH P1. Chose P1: missing-data handling is a mandatory paper-reporting field in PROBAST / TRIPOD+AI.
- **LEAD-026 (subgroup metrics)** — DS P2 vs. PH P1. Chose P1: subgroup stratification is required for equity-aware ML reporting; closes a load-bearing PH gap.
- **LEAD-027 (Table 1)** — DS P2 vs. PH P1. Chose P1: Table 1 is required in clinical ML manuscripts.
- **LEAD-032 (categorical handling in cluster/anomaly)** — DS P2 vs. PH P1. Chose P1: silently dropping indicator-encoded protected attributes is an audit-failure path in a PH audit.
- **Tag-level merges** (LEAD-011, LEAD-013, LEAD-014, LEAD-018, LEAD-020) where one reviewer added only `current-spec-gap` and the other added `pycaret-target-gap` or `paper-reporting-gap` — the consolidated record carries the union of tags without a separate disagreement note, because severity was unchanged.

## Superseded findings

(No findings were dropped without absorption — every DS-xxx and PH-xxx source ID appears in at least one LEAD-xxx `source_ids` array per the Cross-references table.)

## Cross-references

| Source ID | Absorbed into |
|---|---|
| DS-001 | LEAD-001 |
| DS-002 | LEAD-001 |
| DS-003 | LEAD-001 |
| DS-004 | LEAD-001 |
| DS-005 | LEAD-002 |
| DS-006 | LEAD-009 |
| DS-007 | LEAD-009 |
| DS-008 | LEAD-006 |
| DS-009 | LEAD-010 |
| DS-010 | LEAD-004 |
| DS-011 | LEAD-011 |
| DS-012 | LEAD-013 |
| DS-013 | LEAD-014 |
| DS-014 | LEAD-015 |
| DS-015 | LEAD-016 |
| DS-016 | LEAD-017 |
| DS-017 | LEAD-018 |
| DS-018 | LEAD-019 |
| DS-019 | LEAD-020 |
| DS-020 | LEAD-021 |
| DS-021 | LEAD-022 |
| DS-022 | LEAD-034 |
| DS-023 | LEAD-035 |
| DS-024 | LEAD-036 |
| DS-025 | LEAD-005 |
| DS-026 | LEAD-024 |
| DS-027 | LEAD-003 |
| DS-028 | LEAD-037 |
| DS-029 | LEAD-025 |
| DS-030 | LEAD-026 |
| DS-031 | LEAD-038 |
| DS-032 | LEAD-039 |
| DS-033 | LEAD-032 |
| DS-034 | LEAD-023 |
| DS-035 | LEAD-033 |
| DS-036 | LEAD-027 |
| PH-001 | LEAD-003 |
| PH-002 | LEAD-004 |
| PH-003 | LEAD-005 |
| PH-004 | LEAD-007 |
| PH-005 | LEAD-006 |
| PH-006 | LEAD-025 |
| PH-007 | LEAD-027 |
| PH-008 | LEAD-026 |
| PH-009 | LEAD-028 |
| PH-010 | LEAD-029 |
| PH-011 | LEAD-013, LEAD-014 |
| PH-012 | LEAD-012 |
| PH-013 | LEAD-024 |
| PH-014 | LEAD-008 |
| PH-015 | LEAD-010 |
| PH-016 | LEAD-018 |
| PH-017 | LEAD-030 |
| PH-018 | LEAD-021 |
| PH-019 | LEAD-031 |
| PH-020 | LEAD-019 |
| PH-021 | LEAD-040 |
| PH-022 | LEAD-033 |
| PH-023 | LEAD-014 |
| PH-024 | LEAD-016 |
| PH-025 | LEAD-041 |
| PH-026 | LEAD-032 |
| PH-027 | LEAD-012 |
