# mltoolkit — Claude Code ML Plugin

Standalone machine learning plugin for Claude Code. Generates native Python code (scikit-learn + optional XGBoost/LightGBM/CatBoost) for classification, regression, clustering, and anomaly detection. **No PyCaret dependency.**

## Install

1. Clone the repo.
2. Load the plugin: `claude --plugin-dir ./mltoolkit-plugin`
3. Verify: `bash mltoolkit-plugin/scripts/check-env.sh`

## Required

- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

## Optional

- xgboost, lightgbm, catboost — adds more models to the zoo
- imblearn — SMOTE for class imbalance
- category_encoders — TargetEncoder for high-cardinality features
- optuna — alternative hyperparameter search backend (enable via `--search-library optuna` in the tune stage)
- plotly — interactive exploration plots

## Skills

| Skill | Purpose |
|-------|---------|
| `mltoolkit:setup` | Load data, run EDA, identify task type |
| `mltoolkit:classify` | Binary/multiclass classification |
| `mltoolkit:regress` | Continuous-value regression |
| `mltoolkit:cluster` | Unsupervised clustering |
| `mltoolkit:anomaly` | Outlier / anomaly detection |
| `mltoolkit:compare` | Re-run model comparison |
| `mltoolkit:tune` | Hyperparameter tuning |
| `mltoolkit:eda` | Regenerate EDA figures |
| `mltoolkit:package` | Package session into A/B/C tier deliverable |

## How it works

1. You ask Claude to classify/regress/cluster/etc. your data.
2. Claude copies a pre-tested reference script into `.mltoolkit/session.py` in your CWD.
3. Claude runs stages (`eda`, `compare`, `tune`, `evaluate`) and shows you results inline.
4. You iterate — try different models, tweak preprocessing, tune further.
5. When you're happy, ask Claude to "package this." Claude emits a single file, mini project, or full scaffold depending on your choice.

The packaged output is pure scikit-learn (+ optional boosters) — you can drop it anywhere.

## Tests

```bash
bash mltoolkit-plugin/tests/test_references.sh
```

## Architecture

See `docs/superpowers/specs/2026-04-14-mltoolkit-plugin-design.md` for the full design and `docs/superpowers/plans/2026-04-14-mltoolkit-plugin.md` for the implementation plan.
