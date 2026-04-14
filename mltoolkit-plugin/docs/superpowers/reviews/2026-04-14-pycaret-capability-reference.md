# PyCaret Capability Reference (fetched 2026-04-14)

**Source:** Context7 MCP — library ID `/pycaret/pycaret` (reputation: High, 243 code snippets indexed).
**Purpose:** Shared ruler for the three-agent review of `mltoolkit-plugin`. Reviewers measure gaps against this document, not their own recall.
**Status note:** Context7 surfaces PyCaret as a snippet-indexed repo (README, docs/api/*.rst, tutorials) rather than as a complete API reference. Where a specific capability slot was not directly covered by the snippets Context7 returned, it is explicitly flagged `**Not returned by Context7 query.**` so reviewers know it is absent-from-reference, not absent-from-PyCaret. Reviewers should treat unreturned slots as "likely exists upstream — verify before calling it a gap."

---

## 1. Top-level API

PyCaret exposes both a **functional API** (from `pycaret.<module> import *`) and an **OOP API** (`ClassificationExperiment`, `RegressionExperiment`, `ClusteringExperiment`, `AnomalyExperiment`, `TimeSeriesExperiment`). The two APIs are mirror images — every functional call has an OOP method with the same signature. Both are idiomatic; tutorials use functional, the OOP API is recommended for multi-experiment workflows in the same process.

Source: `README.md` — demonstrates both for classification side-by-side (same arguments, same return types).

### 1.1 `setup(data, target, session_id, ...)`

- **Purpose:** Initialize the experiment. Infers feature types, creates train/holdout split, configures preprocessing pipeline (imputation, encoding, scaling, feature selection, transformation), configures metrics, and establishes the global experiment state consumed by every downstream call.
- **Signature highlights (from snippets):** `data`, `target`, `session_id` (reproducibility), `log_experiment` (`'mlflow'` / `'wandb'` / `'comet_ml'` / `True`), `experiment_name`, plus a large set of preprocessing flags not individually enumerated in the snippets returned.
- **Returns:** an experiment handle (functional mode returns the module state; OOP mode returns the experiment object).
- **Available in:** classification, regression, clustering, anomaly, time_series.
- **Evidence:** `tutorials/Tutorial - Anomaly Detection.ipynb`, `README.md`, per-module `.rst` files.

### 1.2 `compare_models(...)`

- **Purpose:** Train and cross-validate every model in the task's zoo, rank them on the primary metric, return the top model (or a list if `n_select > 1`).
- **Returns:** best trained estimator(s).
- **Available in:** classification, regression. Clustering/anomaly lack a directly analogous call (they use `create_model` per algorithm).
- **Evidence:** `README.md` functional example — `best = compare_models()`.

### 1.3 `create_model(estimator_id, ...)`

- **Purpose:** Train and cross-validate a single model by its PyCaret ID (e.g. `'dt'`, `'knn'`, `'rf'`, `'kmeans'`, `'iforest'`).
- **Signature highlights:** `estimator_id` (string), `fold` (CV folds), additional kwargs forwarded to the underlying estimator.
- **Returns:** trained estimator object.
- **Available in:** all task modules.
- **Evidence:** `tutorials/translations/chinese/Binary Classification Tutorial Level Beginner (中文) - CLF101.ipynb` — `dt = create_model('dt')`; clustering tutorial — `km = create_model('kmeans')`.

### 1.4 `tune_model(estimator, ...)`

- **Purpose:** Hyperparameter tuning. Multiple backends supported (scikit-learn RandomizedSearchCV, Optuna, tune-sklearn, scikit-optimize) — selection via `search_library` argument.
- **Returns:** tuned estimator.
- **Available in:** classification, regression, clustering (limited). Anomaly module: **Not returned by Context7 query** for `tune_model` — anomaly is unsupervised, tuning surface may be absent.
- **Evidence:** `README.md`, clustering tutorial (commented `# tuned_km = tune_model(km)`).

### 1.5 `ensemble_model`, `blend_models`, `stack_models`

- **Purpose:** Boosting/bagging wrapper, soft/hard voting blend, meta-learner stacking.
- **Available in:** classification, regression.
- **Evidence:** `**Not returned by Context7 query** as individual snippets`, but listed implicitly via the `.rst` members directive in `docs/source/api/classification.rst` and `regression.rst`.

### 1.6 `calibrate_model(estimator, ...)`

- **Purpose:** Isotonic / sigmoid calibration of classifier probabilities.
- **Available in:** classification.
- **Evidence:** **Not returned by Context7 query** (confirmed to exist per PyCaret's `.rst` member lists but no snippet was returned).

### 1.7 `plot_model(estimator, plot=...)`

- **Purpose:** Render a single analytical plot. The plot catalog is task-dependent.
- **Classification — "15 different plots available"** per the Chinese tutorial snippet. Named plots seen in snippets: `auc`, `pr`, `confusion_matrix` (implied via "confusion matrix" text), `feature`. The other ~11 are **Not returned by Context7 query** but canonically include: `threshold`, `error`, `class_report`, `boundary`, `rfe`, `learning`, `manifold`, `calibration`, `vc` (validation curve), `dimension`, `feature_all`, `parameter`, `lift`, `gain`, `tree`, `ks`.
- **Regression — named plots seen in snippets:** default (residuals), `error`, `feature`. Canonical additions **not returned by Context7 query** include: `cooks`, `rfe`, `learning`, `manifold`, `vc`, `feature_all`, `parameter`, `tree`.
- **Clustering — named plots seen in snippets:** `elbow`. Canonical additions **not returned by Context7 query**: `silhouette`, `distribution`, `cluster`, `tsne`, `umap`, `distance`.
- **Anomaly — named plots seen in snippets:** `tsne`, `umap` (umap requires the `umap` library installed separately).
- **Evidence:** `tutorials/translations/chinese/.../CLF101.ipynb`, regression tutorials, clustering tutorial, `Tutorial - Anomaly Detection.ipynb`.

### 1.8 `evaluate_model(estimator)`

- **Purpose:** Interactive (Jupyter) UI wrapping `plot_model` — widget lets user flip between every available plot without calling `plot_model` N times.
- **Available in:** classification, regression (confirmed); likely clustering/anomaly, **Not returned by Context7 query**.
- **Evidence:** `README.md`, JA Binary Classification / JA Regression tutorials.

### 1.9 `interpret_model(estimator, plot=...)`

- **Purpose:** SHAP-based interpretability.
- **Plot surface:** **Not returned directly by Context7 query**. Canonical plots include `summary`, `correlation`, `reason`, `pdp`, `msa`, `pfi`. Requires `shap` installed.
- **Available in:** classification, regression.
- **Evidence:** Referenced indirectly by the `.rst` members directive, not by any returned snippet.

### 1.10 `predict_model(estimator, data=...)`

- **Purpose:** Run the full preprocessing pipeline + model on new data, return predictions (+ probabilities for classifiers, + Anomaly/Anomaly_Score for anomaly).
- **Returns:** DataFrame with prediction columns appended.
- **Available in:** all task modules.
- **Evidence:** `README.md`, anomaly tutorial — `iforest_pred = predict_model(iforest, data=data)` with `Anomaly` and `Anomaly_Score` columns.

### 1.11 `finalize_model(estimator)`

- **Purpose:** Retrain on the full dataset (train + holdout) before deployment.
- **Available in:** classification, regression.
- **Evidence:** **Not returned by Context7 query** as a named snippet, but present in `.rst` members directive.

### 1.12 `save_model(estimator, path)` / `load_model(path)`

- **Purpose:** Serialize/deserialize the full pipeline (preprocessing + model) to disk.
- **Available in:** all task modules.
- **Evidence:** `README.md` — `save_model(best, 'best_pipeline')`.

### 1.13 `deploy_model(estimator, model_name, platform, authentication)`

- **Purpose:** Push a saved pipeline to AWS S3 / GCP / Azure storage. `load_model` with matching `platform` argument pulls it back.
- **Available in:** classification, regression, clustering, anomaly.
- **Evidence:** `tutorials/Tutorial - Anomaly Detection.ipynb` — commented deploy-to-S3 example.

### 1.14 `get_leaderboard()` / `pull()`

- **Purpose:** `pull()` returns the most recent score grid as a DataFrame. `get_leaderboard()` returns the full leaderboard after `compare_models`.
- **Available in:** classification, regression.
- **Evidence:** **Not returned by Context7 query** as named snippets; present in the module `.rst` members directive.

### 1.15 `check_fairness(estimator, sensitive_features)`

- **Purpose:** Group-fairness audit. Evaluates disparity metrics across protected attribute groups.
- **Disparity metrics surface:** **Not returned by Context7 query**. Per PyCaret's design this wraps Fairlearn-style metrics — demographic parity, equalized odds, per-group metric tables.
- **Available in:** classification, regression.

### 1.16 `dashboard(estimator)`

- **Purpose:** Launch an `explainerdashboard`-based interactive report (per-observation explanations, feature importance, what-if, decision trees when applicable).
- **Available in:** classification, regression.
- **Evidence:** **Not returned by Context7 query** as a named snippet.

### 1.17 Config / metric management

- `get_config(variable)` / `set_config(variable, value)` — access the global experiment state (e.g., `get_config('X_train')`).
- `get_metrics()` / `add_metric()` / `remove_metric()` — inspect and mutate the metric catalog for the active experiment.
- **Evidence:** **Not returned by Context7 query** as named snippets.

### 1.18 Experiment tracking integration

- `setup(..., log_experiment='mlflow', experiment_name='...')` — autologs params, metrics, artifacts to MLflow. Other supported backends (per PyCaret docs, **not individually confirmed by Context7**): `'wandb'`, `'comet_ml'`, `'dagshub'`.
- **Evidence:** `tutorials/Tutorial - Anomaly Detection.ipynb` — commented `log_experiment='mlflow'`.

---

## 2. Classification module (`pycaret.classification`)

### 2.1 Model zoo

- Enumerable at runtime via `models()`. Context7 snippets confirm `'dt'` (Decision Tree), `'knn'`, `'rf'` (Random Forest) as IDs.
- **Full zoo not returned by Context7 query.** Canonical PyCaret classification zoo (per upstream docs, to verify if cited in a gap): `lr`, `knn`, `nb`, `dt`, `svm`, `rbfsvm`, `gpc`, `mlp`, `ridge`, `rf`, `qda`, `ada`, `gbc`, `lda`, `et`, `xgboost`, `lightgbm`, `catboost`, `dummy`.

### 2.2 Plots available via `plot_model`

15 plots total per the Chinese tutorial's documentation string. Named in snippets: `auc`, `pr`, `feature`. Remaining 12 **not returned by Context7 query** (canonical: `confusion_matrix`, `threshold`, `error`, `class_report`, `boundary`, `rfe`, `learning`, `manifold`, `calibration`, `vc`, `dimension`, `feature_all`, `parameter`, `lift`, `gain`, `tree`, `ks`).

### 2.3 Tuning backends

**Not returned by Context7 query.** Canonical: `scikit-learn` (default, RandomizedSearchCV/GridSearchCV), `scikit-optimize`, `tune-sklearn`, `optuna`. Selection via `search_library` argument.

### 2.4 Default metrics

**Not returned by Context7 query.** Canonical: Accuracy, AUC, Recall, Precision (Prec.), F1, Kappa, MCC. `get_metrics()` returns the live metric table.

### 2.5 Task-specific helpers

**Not returned by Context7 query** as individual snippets. Canonical helpers (likely present; to verify before citing as a gap): `optimize_threshold`, `automl`, `create_api`, `create_docker`, `create_app`, `convert_model`, `check_drift`, `models()`.

---

## 3. Regression module (`pycaret.regression`)

### 3.1 Model zoo

- Enumerable via `models()`. **Full zoo not returned by Context7 query.** Canonical: `lr`, `lasso`, `ridge`, `en`, `lar`, `llar`, `omp`, `br`, `ard`, `par`, `ransac`, `tr`, `huber`, `kr`, `svm`, `knn`, `dt`, `rf`, `et`, `ada`, `gbr`, `mlp`, `xgboost`, `lightgbm`, `catboost`, `dummy`.

### 3.2 Plots available via `plot_model`

Named in snippets: default (residuals), `error`, `feature`. Remaining **not returned by Context7 query**. Canonical: `cooks`, `rfe`, `learning`, `manifold`, `vc`, `feature_all`, `parameter`, `tree`, `residuals`.

### 3.3 Tuning backends

Same as classification (see 2.3). **Not returned by Context7 query.**

### 3.4 Default metrics

**Not returned by Context7 query.** Canonical: MAE, MSE, RMSE, R2, RMSLE, MAPE.

### 3.5 Task-specific helpers

**Not returned by Context7 query.** Canonical: same deployment helpers as classification (`create_api`, `create_docker`, `create_app`, `convert_model`, `automl`, `check_drift`, `models()`).

---

## 4. Clustering module (`pycaret.clustering`)

### 4.1 Model zoo

Named in snippets: `kmeans`. **Full zoo not returned by Context7 query.** Canonical: `kmeans`, `ap` (Affinity Propagation), `meanshift`, `sc` (Spectral), `hclust` (Agglomerative), `dbscan`, `optics`, `birch`, `kmodes`.

### 4.2 Plots available via `plot_model`

Named in snippets: `elbow`. Remaining **not returned by Context7 query**. Canonical: `silhouette`, `distribution`, `cluster`, `tsne`, `umap`, `distance`.

### 4.3 Tuning support

`tune_model` exists for clustering (snippet shows a commented call). Primary tuning target is typically `n_clusters`. Supervised tuning (via a reference target) is supported when `supervised_target` is provided at setup or tune time — **Not returned by Context7 query.**

### 4.4 Clustering-specific helpers

- `assign_model(model)` — append cluster labels to the training dataset.
- `models()` — list available algorithms.

### 4.5 Default metrics

**Not returned by Context7 query.** Canonical internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin. External/supervised metrics (when a reference target is available): Homogeneity, Rand Index, Completeness.

---

## 5. Anomaly detection module (`pycaret.anomaly`)

### 5.1 Model zoo

**Full zoo not returned by Context7 query.** Canonical: `abod` (ABOD), `cluster` (Cluster-based LOF), `cof`, `iforest` (Isolation Forest), `histogram` (HBOS), `knn`, `lof` (Local Outlier Factor), `svm` (One-Class SVM), `pca`, `mcd`, `sod`, `sos`.

### 5.2 Plots available via `plot_model`

Named in snippets: `tsne`, `umap` (umap requires separate `umap-learn` install). Both 2-D projections of anomaly scores.

### 5.3 Anomaly-specific helpers

- `assign_model(model)` — append `Anomaly` (0/1) and `Anomaly_Score` columns to the training dataset.
- `predict_model(model, data=...)` — same columns on new data.
- `models()` — list available detectors.
- `tune_model` presence and semantics in anomaly: **Not returned by Context7 query** — anomaly is unsupervised; if present, likely relies on a reference target.

### 5.4 Scoring / threshold behavior

Default contamination / anomaly fraction is settable via setup or create_model; concrete argument name **Not returned by Context7 query.** Canonical name: `fraction`.

### 5.5 Fairness / drift / interpretability in anomaly

**Not returned by Context7 query.** Canonical: anomaly does not expose `interpret_model`, `check_fairness`, or `check_drift` on par with classification/regression.

---

## 6. Cross-cutting capabilities

### 6.1 Fairness / bias auditing

- **`check_fairness(estimator, sensitive_features)`** in classification/regression. See 1.15.
- No fairness helper surfaced for clustering or anomaly.

### 6.2 Interpretability

- **`interpret_model`** (SHAP) in classification/regression. See 1.9.
- **`plot_model(plot='feature'|'feature_all')`** — permutation-importance style feature bars, available across classification/regression.
- **`dashboard(estimator)`** — explainerdashboard-based interactive interpretability. See 1.16.

### 6.3 Experiment tracking

- `setup(..., log_experiment='mlflow', experiment_name=...)`. MLflow is first-class; wandb, comet_ml, dagshub named in PyCaret docs but **not returned by Context7 query** as individual snippets.

### 6.4 Deployment

- `deploy_model` → S3/GCS/Azure.
- `create_api(estimator, api_name, host, port)` — FastAPI scaffold. **Not returned by Context7 query.**
- `create_docker(api_name)` — Dockerfile + requirements.txt scaffold. **Not returned by Context7 query.**
- `create_app(estimator)` — Gradio app for interactive inference. **Not returned by Context7 query.**
- `convert_model(estimator, language='python'|'java'|'c'|'onnx'|...)` — cross-target model export. **Not returned by Context7 query.**

### 6.5 Dashboards & leaderboards

- `dashboard(estimator)` — interactive model explorer.
- `evaluate_model(estimator)` — interactive plot widget.
- `get_leaderboard()` / `pull()` — tabular ranking.

---

## 7. Notable gaps / caveats

1. **Time-series module is separate** (`pycaret.time_series`) with its own API — out of scope for this review's four target tasks.
2. **Interactive calls (`evaluate_model`, `dashboard`) require a notebook / display context.** In a CLI/skill-driven workflow they would need a headless substitute.
3. **Many capability slots were not directly returned by Context7 query.** See the `**Not returned by Context7 query.**` markers throughout. Reviewers should treat those as "likely present upstream" — if a finding claims mltoolkit is missing feature X and X is in the unreturned set, the reviewer should cite the upstream PyCaret `.rst` member directive as evidence rather than relying on this reference alone.
4. **Snippet-based reference.** Context7 surfaces the PyCaret corpus as code examples, not a structured API table. This document is structured by this reviewer using those snippets plus module `.rst` members directives visible in the snippet set.
5. **Version alignment.** Context7 returned no version-specific results; snippets span tutorial translations of different vintages. Reviewers should not assume a specific PyCaret minor version unless a finding depends on it — if one does, re-query Context7 with a pinned version ID (`/pycaret/pycaret/v3.x.x`).
