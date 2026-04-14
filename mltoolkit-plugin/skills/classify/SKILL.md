---
name: classify
description: Build a classification pipeline with native sklearn code. Runs inline, shows leaderboard + figures, then packages into a deliverable.
triggers:
  - classify
  - classification
  - predict category
  - binary classification
  - multiclass
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Classification Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py` (in user's CWD)
- Reference: `{SKILL_DIR}/references/classify_reference.py`
- Plugin-wide helpers: `{PLUGIN_ROOT}/references/_shared/`
- Confirm the user's data path and target column before running anything.

## Workflow

1. **Read the reference script** at `{SKILL_DIR}/references/classify_reference.py` so you know its stages.
2. **Create `.mltoolkit/` if missing** in the user's CWD and add to `.gitignore`.
3. **Stage the reference bundle** into `.mltoolkit/`:
   `python {PLUGIN_ROOT}/scripts/stage_session.py --task classify --dest .mltoolkit`
   This copies `classify_reference.py` as `session.py`, plus the sibling modules (`preprocessing.py`, `model_zoo.py`) and the `_shared/` package, all co-located so the script runs standalone.
4. **Run EDA stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`
5. **Read and present** `.mltoolkit/results/schema.csv` and the generated figures in `.mltoolkit/artifacts/`.
6. **Run compare stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
7. **Read and present** `.mltoolkit/results/leaderboard.csv` as a markdown table.
8. **Ask the user** which model(s) to tune. If unsure, propose the top 1.
9. **Run tune stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID>`
10. **Run evaluate stage**:
    `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage evaluate`
11. **Present holdout metrics** and figures (confusion matrix, ROC, PR, feature importance).
12. **Ask the user** if they want to package the pipeline (invoke `mltoolkit:package`).

## Adaptation rules (invariants)

- **Never import pycaret** in any generated code.
- **Never hardcode the target column** — always route it through `--target`.
- If rows > 100k: pass `--cv 3` to reduce fold count; warn the user.
- If the target has >20 classes, skip stratified splitting (script handles this automatically).
- If a class imbalance >4:1 is detected in EDA and `imblearn` is installed, mention SMOTE as an option for the user.
- All artifacts land under `.mltoolkit/artifacts/` and `.mltoolkit/results/`. Do not write elsewhere without asking.

## Iteration prompts

When user asks to:
- **"add a model"** → edit `.mltoolkit/session.py`, extend `get_zoo()` import with the new model, re-run compare.
- **"remove a model"** → add `--exclude` filter to the session copy and re-run.
- **"tune more"** → re-run tune with higher `--n-iter` (may require editing the script).
- **"handle imbalance"** → edit `session.py` to wrap estimator in `imblearn.pipeline.Pipeline` with SMOTE.

## Paper-mode flags

These flags turn the plugin into a TRIPOD+AI-grade reporting toolkit. Defaults preserve original behavior.

| Flag | Effect | Output |
|---|---|---|
| `--group-col <col>` | Route CV through GroupKFold / StratifiedGroupKFold. | per-fold scores + subgroup_metrics.csv |
| `--time-col <col>` | TimeSeriesSplit. | per-fold |
| `--sensitive-features a,b,c` | Plugin refuses to target-encode these. Use with `--group-col` for subgroup metrics when the group is a protected attribute. | |
| `--allow-target-encode-on-sensitive` | Override the refusal (record rationale in `datasheet.md`). | |
| `--imputation {simple,iterative,knn,drop}` | Imputer class. | missingness.png |
| `--resample {smote,adasyn}` | imblearn over-sampling before fit. | |
| `--calibrate {sigmoid,isotonic}` | Wrap final model in CalibratedClassifierCV. | calibration.json, reliability.png |
| `--optimize-threshold {youden,f1,mcc,fixed-recall} --fixed-recall 0.80` | Pick operating point. | threshold.json |
| `--decision-curve` | Vickers 2006 net-benefit plot. | decision_curve.png |
| `--bootstrap N` | N-sample percentile CI on holdout metrics. | holdout_metrics_ci.json |

Fill out `.mltoolkit/datasheet.md` (from setup) with protected-attribute column names and pass them via `--sensitive-features`.

## When to hand off to package

Prompt the user to package once:
- They've picked a winning model
- Holdout metrics look acceptable
- Iteration has stabilized

Then invoke `mltoolkit:package` and pass through the task type (`classification`).
