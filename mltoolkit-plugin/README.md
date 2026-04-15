# mltoolkit — Claude Code ML Plugin

A paper-grade machine-learning plugin for Claude Code. Generates native Python (scikit-learn + optional XGBoost/LightGBM/CatBoost/Optuna/SHAP) for classification, regression, clustering, and anomaly detection — **no PyCaret dependency** — with TRIPOD+AI / STARD / CONSORT-AI reporting scaffolds built in.

Every run emits: leaderboard, per-fold scores, calibration, bootstrap CIs, subgroup metrics, fairness disparities, decision-curve analysis, reliability diagram, SHAP, learning curve, Table 1, EPV audit, `datasheet.md`, `methods.md`, `model_card.md`, TRIPOD+AI checklist, `run_manifest.json`, and a packaged deliverable you can drop into a repo.

---

## Install

### Option A — Claude Code plugin marketplace (recommended)

```bash
# Inside a Claude Code session:
/plugin marketplace add olaTechie/mltoolkit-plugin
/plugin install mltoolkit@olaTechie
```

After install, verify the skills are registered:

```
/plugin list
```

You should see `mltoolkit:setup`, `mltoolkit:classify`, `mltoolkit:regress`, `mltoolkit:cluster`, `mltoolkit:anomaly`, `mltoolkit:compare`, `mltoolkit:tune`, `mltoolkit:eda`, `mltoolkit:package`.

### Option B — Local clone (for development / forking)

```bash
git clone https://github.com/olaTechie/mltoolkit-plugin.git
cd mltoolkit-plugin
claude --plugin-dir .
```

### Option C — Per-repo pin

Inside a project that will use the plugin, create `.claude/plugins.json`:

```json
{
  "plugins": [
    {
      "name": "mltoolkit",
      "source": "github:olaTechie/mltoolkit-plugin"
    }
  ]
}
```

### Verify

```bash
bash scripts/check-env.sh
bash tests/test_references.sh   # optional: 67-test smoke suite
```

---

## Requirements

**Required:** Python ≥ 3.9, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib`.

**Optional (additive features when installed):**

| Package | Unlocks |
|---|---|
| `xgboost`, `lightgbm`, `catboost` | Extra models in the classify/regress zoos |
| `imbalanced-learn` | `--resample {smote,adasyn}` |
| `category_encoders` | TargetEncoder for non-sensitive high-cardinality columns |
| `optuna` | `--search-library optuna` (TPE sampler) |
| `shap` | SHAP beeswarm plot in evaluate |
| `mlflow` | `--track mlflow` experiment logging |
| `pyod` | anomaly zoo: `abod`, `hbos`, `cof`, `sod`, `sos` |
| `kmodes` | cluster zoo: `kmodes` |

All optional deps are gracefully skipped when absent (the plugin prints a warning and falls back).

---

## Skills at a glance

| Skill | Purpose | Primary outputs |
|---|---|---|
| `mltoolkit:setup` | Load data, EDA, task detection, ethics datasheet | `schema.csv`, `datasheet.md`, `correlation_heatmap.png` |
| `mltoolkit:classify` | Binary/multiclass classification (full paper-mode) | leaderboard, calibration, subgroup, SHAP, reports/ |
| `mltoolkit:regress` | Regression with robust estimators + skew-aware CV | leaderboard, residuals, Q-Q, bootstrap CIs |
| `mltoolkit:cluster` | KMeans/DBSCAN/Agglom/GMM/AP/MeanShift/Spectral/OPTICS/Birch | leaderboard, elbow, PCA scatter, `assigned.csv` |
| `mltoolkit:anomaly` | iForest/LOF/Elliptic/OCSVM/PCA/MCD (+pyod) | `scores.csv`, `top_anomalies.csv`, subgroup rates |
| `mltoolkit:compare` | Re-run model comparison with new flags | leaderboard + per-fold |
| `mltoolkit:tune` | Hyperparameter search (sklearn or optuna) | `best_params.json` |
| `mltoolkit:eda` | Regenerate EDA figures (Table 1, missingness, EPV) | `table1.csv`, `epv_audit.json` |
| `mltoolkit:package` | Tier A (single file) / B (mini project) / C (full scaffold) | deliverable + pinned requirements + reports |

---

## Sample prompts

Copy-paste any of these at the Claude Code prompt. Claude will invoke the right skill and generate native Python in your CWD.

### Quickstart — binary classifier on a CSV

```
Use mltoolkit:setup on data/diabetes.csv with target "outcome".
Then classify it and package the result as a mini project called "diabetes_model".
```

### Paper-grade clinical-prediction run

```
I have a cohort at data/patients.csv with target "readmitted_30d".
The columns "race", "sex", and "zip_code" are protected attributes.
I want:
  - group-fairness metrics by race
  - calibration + reliability diagram
  - 95% bootstrap CIs on holdout
  - decision-curve analysis
  - TRIPOD+AI reporting scaffold
  - finalized model refit on the full dataset

Use mltoolkit:classify.
```

Claude will generate a staged `.mltoolkit/session.py` and run it with:

```bash
python .mltoolkit/session.py \
  --data data/patients.csv --target readmitted_30d \
  --output-dir .mltoolkit --stage all \
  --sensitive-features race,sex,zip_code \
  --group-col race \
  --calibrate sigmoid --bootstrap 1000 \
  --decision-curve --optimize-threshold youden \
  --finalize
```

### Regression with time-based CV and robust estimators

```
Forecast "price_usd" in data/sales.csv using mltoolkit:regress.
Data has a date column "sale_date" — use time-series CV.
Include robust regressors (Huber, RANSAC, TheilSen).
Emit bootstrap CIs on holdout.
Package the winner as Tier C with a FastAPI endpoint.
```

### Anomaly detection with an ethics check

```
Find anomalies in data/transactions.csv using mltoolkit:anomaly.
The column "customer_segment" encodes a protected attribute.
I want per-segment anomaly rates and LOF serialized for reuse
(--novelty). Default contamination 3%.
```

### Clustering with categorical columns

```
Segment customers in data/customers.csv into 5 groups using
mltoolkit:cluster. Keep categorical columns via one-hot encoding.
```

### Re-tune a chosen model with Optuna

```
From the leaderboard, lightgbm was best. Tune it more aggressively
with Optuna over 100 trials.
```

Claude will invoke `mltoolkit:tune` and run:

```bash
python .mltoolkit/session.py --stage tune --model lgbm \
  --search-library optuna --n-iter 100
```

### Package a finished session into a GitHub-ready project

```
Package this session as Tier C with a FastAPI app and Dockerfile.
```

---

## Paper-mode flags (classify + regress)

Defaults preserve the zero-config behavior; every flag below is opt-in.

| Flag | Effect |
|---|---|
| `--group-col <col>` | GroupKFold / StratifiedGroupKFold |
| `--time-col <col>` | TimeSeriesSplit |
| `--sensitive-features a,b,c` | Refuse to target-encode these; feed subgroup metrics |
| `--allow-target-encode-on-sensitive` | Explicit override (document in `datasheet.md`) |
| `--imputation {simple,iterative,knn,drop}` | Imputer class |
| `--resample {smote,adasyn}` | Over-sampling via imblearn |
| `--calibrate {sigmoid,isotonic}` | Wrap in `CalibratedClassifierCV` |
| `--optimize-threshold {youden,f1,mcc,fixed-recall}` | Pick operating point |
| `--fixed-recall 0.80` | Target recall for `fixed-recall` |
| `--decision-curve` | Vickers 2006 net-benefit plot |
| `--bootstrap N` | N-sample percentile CI on holdout |
| `--ensemble {voting,stacking}` + `--ensemble-k 3` | Ensemble top-K |
| `--finalize` | Refit on X_train ∪ X_test |
| `--search-library {sklearn,optuna}` + `--n-iter N` | Tuning backend |
| `--track mlflow` | MLflow autolog |

---

## How it works

1. You ask Claude to classify / regress / cluster / detect anomalies in your data.
2. Claude runs `scripts/stage_session.py --task <task> --dest .mltoolkit/` which copies a pre-tested reference script into `.mltoolkit/session.py` along with its sibling modules (`preprocessing.py`, `model_zoo.py`) and the `_shared/` primitives (fairness, calibration, bootstrap, splits, Table 1, decision curve, EPV, run manifest, methods.md, model_card.md, checklists).
3. Claude runs stages (`eda` → `compare` → `tune` → `evaluate`) and shows you results inline.
4. Every stage appends to `.mltoolkit/results/run_manifest.json` (versions, seed, args, timestamp).
5. At the end of evaluate, Claude emits `reports/methods.md`, `reports/model_card.md`, `reports/tripod_ai_checklist.md`, and `results/summary_report.html`.
6. You iterate — swap models, change flags, tune harder.
7. When you're happy, ask Claude to **"package this"**. You get one of three tiers:
   - **Tier A** — single `deliverable.py` (session copy, renamed).
   - **Tier B** — script + pinned `requirements.txt` + README + harvested reports.
   - **Tier C** — full `src/`/`tests/` scaffold; `src/train.py` is patched with the chosen estimator + tuned params; optional FastAPI + Dockerfile.

The packaged output is pure scikit-learn (+ optional boosters) — drop it into any repo.

---

## Outputs reference

All outputs land under `.mltoolkit/` in your CWD.

```
.mltoolkit/
├── session.py                     # staged reference (+ siblings + _shared/)
├── preprocessing.py
├── model_zoo.py
├── _shared/                       # bootstrap, splits, fairness, …
├── datasheet.md                   # setup → fill in before training
├── model.joblib                   # tuned model
├── model_final.joblib             # --finalize output
├── model_ensemble.joblib          # --ensemble output
├── load_hint.json                 # LOF without --novelty
├── artifacts/                     # PNG + PDF figures
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── missingness.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── reliability.png            # calibration
│   ├── decision_curve.png
│   ├── learning_curve.png
│   ├── classification_report_heatmap.png
│   ├── feature_importance.png
│   └── shap_beeswarm.png
├── results/
│   ├── schema.csv
│   ├── table1.csv + table1.md
│   ├── epv_audit.json
│   ├── leaderboard.csv + leaderboard.md
│   ├── leaderboard_folds.csv      # per-fold scores
│   ├── best_params.json
│   ├── classification_report.csv
│   ├── permutation_importance.csv
│   ├── calibration.json
│   ├── holdout_metrics_ci.json
│   ├── threshold.json
│   ├── subgroup_metrics.csv
│   ├── fairness_disparities.json
│   ├── ensemble_score.json
│   ├── finalize_note.json
│   ├── run_manifest.json          # appended at every stage
│   └── summary_report.html
└── reports/
    ├── methods.md                  # TRIPOD+AI methods scaffold
    ├── model_card.md               # Mitchell et al. 2019 template
    └── tripod_ai_checklist.md
```

Pass `--diagnostic` or `--interventional` at package time (future) to additionally emit `stard_checklist.md` / `consort_ai_checklist.md`.

---

## Testing

```bash
bash tests/test_references.sh
```

Expected: five steps, 67 tests green.

Individual test files:

```bash
pytest tests/test_shared_bootstrap.py
pytest tests/test_shared_fairness.py
pytest tests/test_shared_calibration.py
pytest tests/test_shared_run_manifest.py
pytest tests/test_stage_and_run.py   # end-to-end copy + execute
```

---

## Architecture

- **Design spec:** `docs/superpowers/specs/2026-04-14-mltoolkit-three-agent-review-design.md`
- **Review reports** (data-scientist, public-health analyst, lead synthesis): `docs/superpowers/reviews/`
- **Implementation plans 1–5:** `docs/superpowers/plans/` — cover staging fix, shared backbone, classify wire-up, parity for regress/cluster/anomaly, and reproducibility + paper scaffolding respectively.

Each plan's commit trail tags every LEAD-### finding it closes. Full 41-finding backlog → 67 tests — closed.
