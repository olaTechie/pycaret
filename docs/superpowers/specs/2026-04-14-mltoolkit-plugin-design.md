# mltoolkit — Standalone ML Plugin for Claude Code

**Date:** 2026-04-14
**Status:** Draft — awaiting user approval
**Supersedes:** The `pycaret-plugin/` built earlier in this session (which wraps PyCaret at runtime)

---

## 1. Motivation

The earlier `pycaret-plugin/` delegates ML work to the PyCaret library. That approach has three problems:

1. **Runtime dependency on PyCaret** — users must install a large, fast-moving package; their generated code cannot live without it.
2. **PyCaret evolves and breaks** — its model IDs, setup parameters, and internal APIs shift between versions. Skills written today may silently fail tomorrow.
3. **Output isn't publication-ready** — PyCaret's plots and tables are serviceable but not tuned for reports or deployable artifacts.

The replacement plugin generates **standalone native Python code** (sklearn + optional boosters) that the user can run, edit, and ship without any dependency on the plugin or PyCaret. Claude executes the code inline during the session, shows results/tables/figures, iterates with the user, and then packages the final deliverable in the tier the user chooses.

## 2. Goals

- Zero PyCaret dependency in either the plugin runtime or the generated output.
- Claude runs code inline and produces tables + figures as it works (like a notebook).
- Tight, maintainable set of reference scripts that define the "publication-quality standard."
- User-controlled final deliverable: single file, mini project, or full scaffold.
- Graceful degradation when optional packages (xgboost/lightgbm/catboost/imblearn/optuna) are missing.

## 3. Non-Goals (v1)

- Time-series forecasting (deferred to v2).
- NLP/vision-specific pipelines.
- Experiment tracking integrations (MLflow, W&B, CometML).
- GPU-specific model backends (cuML, sklearnex).
- Automated hyperparameter search beyond `RandomizedSearchCV` / `optuna` (if installed).

## 4. User Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Primary use case | ML engineer (quick prototyping → production-ready code) |
| Workflow model | Hybrid — Claude runs inline with live results, then packages on request |
| Task scope (v1) | Classification, Regression, Clustering, Anomaly Detection |
| Plotting | Matplotlib + seaborn for final output; Plotly for interactive exploration |
| Model libraries | sklearn core + optional XGBoost/LightGBM/CatBoost (graceful fallback) |
| Location | New standalone directory (not inside `pycaret-plugin/`) |
| Deliverable shape | Tiered: Tier A (single file) / Tier B (mini project) / Tier C (full scaffold); default Tier B |
| Architecture | Approach 1 — template-based reference implementations + adaptation |

## 5. Architecture

### 5.1 Directory layout

```
mltoolkit-plugin/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   ├── setup/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── setup_reference.py
│   ├── classify/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── classify_reference.py
│   │       ├── preprocessing.py
│   │       └── model_zoo.py
│   ├── regress/       (same shape as classify)
│   ├── cluster/       (same shape)
│   ├── anomaly/       (same shape)
│   ├── compare/
│   │   └── SKILL.md
│   ├── tune/
│   │   └── SKILL.md
│   ├── eda/
│   │   └── SKILL.md
│   └── package/
│       ├── SKILL.md
│       └── references/
│           ├── tier_a_template.py
│           ├── tier_b_readme_template.md
│           ├── tier_c_scaffold/
│           │   ├── src/
│           │   │   ├── preprocess.py
│           │   │   ├── train.py
│           │   │   └── predict.py
│           │   ├── tests/
│           │   │   └── test_pipeline.py
│           │   ├── api.py
│           │   ├── Dockerfile
│           │   └── requirements.txt
├── agents/
│   └── ml-pipeline.md
├── references/
│   └── _shared/
│       ├── plotting.py
│       ├── reporting.py
│       └── deps.py
├── scripts/
│   └── check-env.sh
└── tests/
    └── test_references.sh
```

### 5.2 Core principles

1. **Reference scripts are the source of truth.** Each skill ships a complete, tested, publication-quality `.py` script. Claude adapts these scripts rather than writing code from scratch.
2. **Generated code imports nothing from this plugin.** All output uses only the user's Python environment (sklearn, pandas, matplotlib, + optional boosters).
3. **Skills are playbooks, not docs.** Short (~150 lines), workflow-shaped, tell Claude exactly what to do and when.
4. **Session scratchpad holds in-progress work.** `.mltoolkit/session.py` in the user's CWD, edited incrementally as the user iterates.

## 6. Workflow (Inline → Iterate → Package)

### Phase 1: Setup & Exploration (inline)

1. User says *"classify this dataset"* (or similar trigger).
2. Claude invokes `mltoolkit:setup` — reads `setup_reference.py`, writes a copy to `.mltoolkit/session.py` with the user's data path and target column substituted in, runs `python .mltoolkit/session.py --stage=eda`.
3. Claude invokes `mltoolkit:classify` — reads `classify_reference.py`, merges it into `session.py`, runs `--stage=compare`.
4. Artifacts land in `.mltoolkit/artifacts/` (figures) and `.mltoolkit/results/` (tables). Claude reads them back to display to the user.

### Phase 2: Iteration

User requests changes: *"try tuning the top 3"*, *"remove outliers"*, *"handle class imbalance"*. Claude:

- Uses `Edit` on `session.py` to toggle flags, add models, or insert preprocessing steps (preserves prior edits).
- Re-runs the appropriate stage.
- Shows updated tables/figures.

### Phase 3: Packaging

When the user says *"package this"* (or Claude proactively asks after several iterations), Claude invokes `mltoolkit:package`:

- **Tier A** (single file): strip argparse/stages from `session.py`, flatten to one linear script → `classification_pipeline.py`.
- **Tier B** (mini project): Tier A + auto-generated `requirements.txt` (derived from actual imports) + `README.md` with run instructions. **Default.**
- **Tier C** (full scaffold): split into `src/{preprocess,train,predict}.py`, add `tests/test_pipeline.py` (pytest), `requirements.txt`, `README.md`. Ask user whether to add `api.py` (FastAPI) and `Dockerfile`.

## 7. Reference Scripts — Content Specification

### 7.1 `classify_reference.py`

Stages, each a clearly-separated function:

1. **`load_data(path, target)`** — read CSV/parquet, validate schema, report dtypes + missing.
2. **`run_eda(df, target)`** — shape, dtypes, missing, class distribution, correlation heatmap, pairplot; saves artifacts.
3. **`build_preprocessor(df, target)`** — `sklearn.compose.ColumnTransformer`:
   - Numeric: `SimpleImputer(strategy='median')` + `StandardScaler()`
   - Categorical (≤10 unique): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
   - Categorical (>10 unique): `TargetEncoder` from `category_encoders` (fall back to `OrdinalEncoder` if not installed)
   - Optional outlier removal: `IsolationForest`
   - Optional class imbalance: `imblearn.over_sampling.SMOTE` if installed and imbalance ratio >4:1
4. **`split(df, target)`** — `train_test_split(stratify=y, test_size=0.2, random_state=42)`.
5. **`compare_models(X_train, y_train, cv=5)`** — iterate through model zoo; `cross_val_score` for each; record Accuracy, F1, AUC (when applicable), Precision, Recall. Returns sorted leaderboard DataFrame.
6. **`tune_model(model_id, X_train, y_train, n_iter=20)`** — `RandomizedSearchCV` with the model's param grid from `model_zoo.py`; falls back to `optuna` if installed and enabled.
7. **`evaluate(model, X_test, y_test)`** — holdout metrics, confusion matrix, classification report, ROC curve, PR curve, calibration plot.
8. **`explain(model, X_test)`** — permutation importance (always) + built-in feature importance for tree models.
9. **`save(model, path)`** — `joblib.dump` of the full pipeline (preprocessor + model).
10. **`main()`** — argparse with `--stage` (one of: `eda`, `compare`, `tune`, `evaluate`, `all`) + `--data`, `--target`, `--output-dir`. Stage `all` runs the full pipeline end-to-end; individual stages are independently re-runnable and share state via `.mltoolkit/config.json`.

### 7.2 `regress_reference.py`

Same structure as classify. Differences:

- Metrics: R², RMSE, MAE, MAPE.
- Models: LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, SVR, KNeighborsRegressor, MLPRegressor + optional XGBoost/LightGBM/CatBoost regressors.
- Diagnostic plots: residuals vs fitted, Q-Q plot, prediction vs actual, error distribution histogram.
- No stratified split, no class imbalance handling.

### 7.3 `cluster_reference.py`

- Models: KMeans (with elbow + silhouette), DBSCAN, AgglomerativeClustering, GaussianMixture.
- Diagnostics: silhouette plot per model, PCA 2D scatter colored by cluster, cluster summary statistics table (size, feature means per cluster).
- No target column; no train/test split.

### 7.4 `anomaly_reference.py`

- Models: IsolationForest (default), LocalOutlierFactor, EllipticEnvelope, OneClassSVM.
- Diagnostics: anomaly score histogram with chosen threshold, PCA/t-SNE scatter with anomalies highlighted, top-K anomaly table (most anomalous rows).
- No target; no class imbalance concerns.

### 7.5 Model zoo (`model_zoo.py` per task)

Dict keyed by model ID. Each entry:

```python
{
    "estimator": <sklearn-compatible estimator with sensible defaults>,
    "param_grid": <dict of 5–15 hyperparameters, tight ranges>,
    "requires": <optional package name, or None>,
}
```

**Tight param grids** — intentionally smaller than PyCaret's. Example for RandomForestClassifier:

```python
"rf": {
    "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
    "param_grid": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "requires": None,
}
```

Optional entries (xgboost/lightgbm/catboost) check `deps.has_xgboost()` etc. and are skipped in comparison loops when unavailable.

### 7.6 Shared helpers (`references/_shared/`)

- **`plotting.py`** — `set_style()` (seaborn whitegrid, 300 DPI, consistent fontsize), `save_fig(fig, name)` → PNG + PDF.
- **`reporting.py`** — `df_to_markdown(df)`, `df_to_latex(df)`, `summary_report(results, output='report.html')` that emits a one-page HTML summary with all artifacts embedded.
- **`deps.py`** — `has_xgboost()`, `has_lightgbm()`, `has_catboost()`, `has_imblearn()`, `has_category_encoders()`, `has_optuna()`. Each returns bool without raising.

## 8. Skill Content Specification

### 8.1 Common SKILL.md structure

```markdown
---
name: <skill-name>
description: <one line>
triggers: [...]
allowed-tools: [Bash(python*), Read, Write, Edit]
---

# <Task> Playbook

## Prerequisites
- Session scratchpad: .mltoolkit/session.py (in user's CWD)
- Skill-owned reference: {SKILL_DIR}/references/<name>_reference.py
- Plugin-wide shared helpers: {PLUGIN_ROOT}/references/_shared/ (plotting, reporting, deps)

## Workflow
<numbered steps>

## Adaptation rules
<invariants Claude must follow>
```

### 8.2 Invariants enforced in every skill

- **Never import pycaret** in generated code.
- **Target column is always a parameter**, never hardcoded.
- **If dataset >100k rows**: reduce CV folds to 3 and warn the user.
- **If class imbalance >4:1**: enable SMOTE (when `imblearn` installed) or class-weight balancing.
- **All figures saved to `.mltoolkit/artifacts/`** with descriptive filenames (e.g., `roc_curve_randomforest.png`).
- **All tables saved to `.mltoolkit/results/`** as CSV + rendered markdown.

### 8.3 `package` skill details

The packaging skill is a pure transformation — it takes `.mltoolkit/session.py` and the artifacts produced during iteration and emits a deliverable in the chosen tier. It does not run any new ML; it only reshapes the working code.

- **Tier A output:** `classification_pipeline.py` (or `regression_pipeline.py` etc.)
- **Tier B output:** adds `requirements.txt` (from actual imports used) + `README.md` (how to install, how to run, what it does)
- **Tier C output:** splits into `src/preprocess.py`, `src/train.py`, `src/predict.py`; adds `tests/test_pipeline.py` (smoke test against synthetic data); optionally adds `api.py` + `Dockerfile` if user confirms

## 9. Session Scratchpad Protocol

- **Location:** `.mltoolkit/` in user's CWD.
- **Layout:**
  - `.mltoolkit/session.py` — the working Python script (modified via `Edit` tool, not rewritten).
  - `.mltoolkit/artifacts/` — figures (PNG + PDF).
  - `.mltoolkit/results/` — tables (CSV + Markdown).
  - `.mltoolkit/config.json` — user-chosen hyperparameters, target column, task type (so subsequent skill invocations pick up state).
- **Git hygiene:** the `setup` skill writes a `.gitignore` entry for `.mltoolkit/` on first use.
- **Stage-based CLI:** every reference script accepts `--stage={eda,compare,tune,evaluate,all}` so individual stages are independently re-runnable.

## 10. Error Handling & Edge Cases

| Condition | Behavior |
|-----------|----------|
| Optional package missing (xgboost/lightgbm/catboost/imblearn/optuna) | Log warning via `deps.py`, skip that model, continue. |
| Target column not found in data | Fail fast with error message naming available columns. |
| All-NaN column | Drop with warning, log the dropped columns. |
| Single-class target in classification | Fail with clear message suggesting regression or data check. |
| Dataset >1M rows | Warn and ask user to confirm or subsample. |
| No numeric/categorical columns after preprocessing | Fail fast with diagnostic output. |
| Reference script crashes | Surface the stderr directly; do not swallow. |

## 11. Testing Strategy

- Each reference ships with a smoke test embedded: running with `--stage=all` against a synthetic dataset (`make_classification` / `make_regression` / `make_blobs`) must complete and save artifacts.
- `tests/test_references.sh` runs every reference end-to-end using synthetic data; exit code is the pass/fail signal.
- CI-friendly: references emit no interactive prompts when run directly (prompts happen via Claude during sessions, not in the scripts themselves).

## 12. Dependencies

### Required (user environment)
- Python ≥3.9
- pandas, numpy
- scikit-learn ≥1.3
- matplotlib, seaborn
- joblib

### Optional (graceful fallback)
- xgboost, lightgbm, catboost — additional model options
- imblearn — SMOTE for class imbalance
- category_encoders — TargetEncoder for high-cardinality features
- optuna — alternative hyperparameter search
- plotly — interactive exploration plots (if user wants them during iteration)

### Not required
- PyCaret (explicitly excluded)

`scripts/check-env.sh` reports status of all required and optional packages. Non-fatal — plugin works with the core.

## 13. Migration from Existing `pycaret-plugin/`

- The existing `pycaret-plugin/` remains in the repo as historical reference.
- The new `mltoolkit-plugin/` is created as a separate top-level directory.
- No code is shared between them. User chooses which plugin to load.

## 14. Open Questions

None remaining — all design points approved during brainstorming.

## 15. Success Criteria

The v1 plugin is complete when:

1. `check-env.sh` reports clean status on a minimal sklearn environment.
2. `tests/test_references.sh` passes on synthetic datasets for all four task types.
3. A user can load a CSV, invoke `/classify`, and get a leaderboard + ROC curve + confusion matrix rendered inline within ~3 minutes.
4. `mltoolkit:package` emits a Tier B project that runs standalone (`python classification_pipeline.py --data test.csv --target y`) and reproduces the same result the user saw during iteration.
5. None of the generated deliverables import `pycaret`.
