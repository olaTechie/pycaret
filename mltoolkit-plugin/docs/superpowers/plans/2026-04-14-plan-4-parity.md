# Plan 4 — Regress / cluster / anomaly parity + zoo expansion + metrics + ensembling + tuning polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Bring regress / cluster / anomaly up to classify's paper-grade capability. Expand every model zoo to PyCaret parity. Add classification + regression metric breadth, ensembling, tuning CLI polish, cluster categorical handling, anomaly LOF serialization fix, regression skew-aware CV.

**Architecture:** Additive. Same `_shared/` primitives from Plan 2 are reused. Each task touches its own reference + zoo file; wiring pattern mirrors Plan 3's classify rewrite. No new `_shared/` modules.

**Findings covered:** LEAD-009, -013, -014, -015, -016, -020, -021, -030, -032, -035, -036, -037, -039.

---

## File Structure

**Modified (most edits mirror patterns already in classify_reference.py):**

| File | Change |
|---|---|
| `skills/classify/references/model_zoo.py` | Add QDA, LDA, DummyClassifier, PassiveAggressive, Multinomial/BernoulliNB. |
| `skills/regress/references/model_zoo.py` | Add LassoLars, LassoLarsIC, OMP, BR, ARD, Huber, RANSAC, TheilSen, Dummy. |
| `skills/cluster/references/model_zoo.py` | Add AffinityPropagation, MeanShift, SpectralClustering, OPTICS, Birch; gate kmodes. |
| `skills/anomaly/references/model_zoo.py` | Add PCA-reconstruction, MCD; gate pyod-family (abod, hbos, cof, sod, sos). |
| `skills/classify/references/classify_reference.py` | Metrics expansion (MCC, kappa, balanced, logloss, AP). Ensembling (`--ensemble {voting,stacking}`). |
| `skills/regress/references/regress_reference.py` | Apply Plan 3 backbone wiring to regression (group/time splits, imputation, bootstrap CIs, run_manifest-adjacent outputs). Skew-aware CV stratification (LEAD-039). Metrics expansion (MAPE, RMSLE guarded, explained_variance). Ensembling. |
| `skills/cluster/references/cluster_reference.py` | Save actually-fitted estimator (LEAD-021). Categorical handling (LEAD-032): stderr warn + `--categorical {drop,one-hot}`. |
| `skills/anomaly/references/anomaly_reference.py` | LOF `novelty=True` for serialization OR `load_hint.json` refit-on-load (LEAD-020). Ethics warning + optional `--group-col` subgroup emission (LEAD-030). |
| `skills/tune/SKILL.md` | Document `--n-iter`, `--cv-tune`, `--time-budget`, `--tune-metric` (LEAD-009). |

**Tests modified / added:** extend `tests/test_classify.py`, `tests/test_regress.py`, `tests/test_cluster.py`, `tests/test_anomaly.py` for each change. Add an e2e row in `test_stage_and_run.py` for regress paper-mode.

---

## Task 1: Classification zoo expansion (LEAD-013)

- [ ] Open `skills/classify/references/model_zoo.py`. Add to `get_zoo()` dict:
  - `qda` → `QuadraticDiscriminantAnalysis` with `reg_param ∈ [0.0, 0.1, 0.5]`.
  - `lda` → `LinearDiscriminantAnalysis` with `shrinkage ∈ [None, "auto", 0.1, 0.5]` + `solver="lsqr"`.
  - `dummy` → `DummyClassifier(strategy="stratified")` with empty grid.
  - `par` → `PassiveAggressiveClassifier(random_state=42)` with `C ∈ [0.1, 1.0, 10.0]`.
  - `mnb` → `MultinomialNB()` gated on numeric non-negativity (documented in comment that caller must pass non-negative features; if training fails at runtime, sklearn will raise — acceptable).
  - `bnb` → `BernoulliNB()` with `alpha ∈ [0.1, 1.0, 10.0]`.
- [ ] Extend `tests/test_classify.py::test_model_zoo_has_core_entries` expected set to include `qda, lda, dummy, par, bnb`. (Leave `mnb` out of the required set since it requires non-negative features.)
- [ ] Run `python -m pytest tests/test_classify.py::test_model_zoo_has_core_entries -v`. Commit.

```bash
git commit -m "feat(mltoolkit): classify zoo — qda, lda, dummy, par, mnb, bnb (LEAD-013)"
```

---

## Task 2: Regression zoo expansion (LEAD-014)

- [ ] Open `skills/regress/references/model_zoo.py`. Add:
  - `lassolars` → `LassoLars(alpha=0.01)` + `alpha ∈ [0.001, 0.01, 0.1]`.
  - `llars_ic` → `LassoLarsIC(criterion="bic")`.
  - `omp` → `OrthogonalMatchingPursuit()` + `n_nonzero_coefs ∈ [5, 10, 20]`.
  - `br` → `BayesianRidge()`.
  - `ard` → `ARDRegression()`.
  - `huber` → `HuberRegressor()` + `epsilon ∈ [1.1, 1.35, 1.5]`.
  - `ransac` → `RANSACRegressor(random_state=42)`.
  - `theilsen` → `TheilSenRegressor(random_state=42)`.
  - `dummy` → `DummyRegressor(strategy="mean")` with empty grid.
- [ ] Update `tests/test_regress.py` model-zoo assertion to include the new ids.
- [ ] Run + commit:

```bash
git commit -m "feat(mltoolkit): regress zoo — huber/ransac/theilsen/bayesian/ard/omp/dummy (LEAD-014)"
```

---

## Task 3: Cluster zoo expansion + fit-persistence fix (LEAD-015, -021)

- [ ] Open `skills/cluster/references/model_zoo.py`. Add `ap` (AffinityPropagation), `meanshift` (MeanShift), `spectral` (SpectralClustering), `optics` (OPTICS), `birch` (Birch). Gate `kmodes` on `deps._check('kmodes')`.
- [ ] Open `skills/cluster/references/cluster_reference.py`. Find the block that serializes the estimator — audit whether the current code re-fits for serialization. Replace any pattern of the form `joblib.dump(type(model)().fit(X), ...)` with `joblib.dump(model, ...)`. Wrap the dump in a `try: ... except (TypeError, PicklingError) as e: print(f"WARNING: serialization failed for {name}: {e}", flush=True)` — do **not** swallow all exceptions.
- [ ] Add test `tests/test_cluster.py::test_cluster_persists_fitted_estimator` that fits kmeans, dumps, loads, and asserts loaded.cluster_centers_ matches fitted.cluster_centers_.
- [ ] Commit:

```bash
git commit -m "feat(mltoolkit): cluster zoo — AP/MeanShift/Spectral/OPTICS/Birch + fit-persistence (LEAD-015, -021)"
```

---

## Task 4: Anomaly zoo expansion + LOF serialization (LEAD-016, -020)

- [ ] Open `skills/anomaly/references/model_zoo.py`. Add `pca` (sklearn PCA wrapped with a reconstruction-error scorer), `mcd` (sklearn `MinCovDet`). Gate `abod, hbos, cof, sod, sos` on `deps._check('pyod')`; if pyod is installed, import `pyod.models.{abod,hbos,cof,sod,sos}` and register them.
- [ ] Open `skills/anomaly/references/anomaly_reference.py`. When the chosen model is LOF:
  - if `--novelty` flag passed, refit with `LocalOutlierFactor(n_neighbors=..., novelty=True)` before serialization
  - otherwise write `<out>/load_hint.json` describing how to refit on load + emit `.mltoolkit/results/lof_note.md` telling user to call `fit_predict` on new data.
- [ ] Add `--novelty` argparse flag (default False).
- [ ] Extend `tests/test_anomaly.py` with `test_lof_serializes_with_novelty_flag`.
- [ ] Commit:

```bash
git commit -m "feat(mltoolkit): anomaly zoo + LOF novelty flag / load_hint.json (LEAD-016, -020)"
```

---

## Task 5: Cluster categorical handling + anomaly ethics (LEAD-030, -032)

- [ ] In `skills/cluster/references/cluster_reference.py`, after reading data, detect categorical columns and:
  - Print to stderr: `WARNING: dropping non-numeric columns: <list>. Pass --categorical one-hot to one-hot encode instead.`
  - Add `--categorical {drop,one-hot}` argparse flag (default drop — matches current behavior).
  - If `one-hot`, run the df through a small `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` before clustering.
- [ ] In `skills/anomaly/references/anomaly_reference.py`:
  - Emit a warning line at stage start when no `--sensitive-features` passed.
  - Add `--group-col` + `--sensitive-features` argparse flags. When `--group-col` is set, after prediction emit `results/subgroup_anomaly_rate.csv` with per-group anomaly-flag rate.
- [ ] Update SKILL.md for both (brief "ethics" block).
- [ ] Tests + commit:

```bash
git commit -m "feat(mltoolkit): cluster --categorical + anomaly ethics/--group-col (LEAD-030, -032)"
```

---

## Task 6: Classify metric expansion + ensembling (LEAD-035, -037)

- [ ] In `skills/classify/references/classify_reference.py` `compare_models`, extend the binary `scorers` dict with:
  - `mcc` → `make_scorer(matthews_corrcoef)`
  - `kappa` → `make_scorer(cohen_kappa_score)`
  - `balanced_accuracy` → `"balanced_accuracy"`
  - `logloss` → `"neg_log_loss"`  (note: lower is better — annotate in leaderboard doc)
  - `average_precision` → `"average_precision"`
- [ ] Add `--ensemble {none,voting,stacking}` argparse flag (default none). When set, after tune stage, build an `EnsembleClassifier` from top-k (k=3) models in the leaderboard and fit. Save separately as `model_ensemble.joblib`.
- [ ] Test + commit.

---

## Task 7: Regression paper-mode wiring + skew CV + metrics (LEAD-036, -039, and wire-up mirroring Plan 3)

- [ ] Replicate Plan 3's backbone wiring in `skills/regress/references/regress_reference.py`: CLI flags (`--group-col`, `--time-col`, `--sensitive-features`, `--allow-target-encode-on-sensitive`, `--imputation`, `--bootstrap`, `--ensemble`). Wire through `_shared.splits.make_splitter` (with `groups`/`time_order` arguments).
- [ ] Add skew-aware stratification: if `np.abs(stats.skew(y)) > 1`, stratify the initial `train_test_split` via `pd.qcut(y, 10, duplicates='drop')` instead of plain random.
- [ ] Extend `scorers` with `MAPE`, `RMSLE` (guarded — skip if `np.any(y < 0)`), `explained_variance`.
- [ ] Add ensembling (voting / stacking regressor).
- [ ] Update `skills/regress/references/preprocessing.py` with the same safe-encoder refusal + imputation routing as classify.
- [ ] Tests + commit:

```bash
git commit -m "feat(mltoolkit): regress paper-mode wiring + skew CV + metrics (LEAD-036, -039)"
```

---

## Task 8: Tune SKILL.md polish (LEAD-009)

- [ ] In `skills/tune/SKILL.md`, document existing `--n-iter`, add guidance for `--cv-tune` (pass through as `--cv` override for tune stage only) and `--time-budget` (hours; accepted by callers but currently a no-op unless using optuna — note that). Also `--tune-metric` default.
- [ ] If those flags don't exist yet in classify/regress argparse, add them: `--tune-metric` (default `accuracy` for classify / `r2` for regress), `--time-budget` (float hours, default 0 meaning unlimited).
- [ ] Commit.

---

## Task 9: End-to-end parity test

- [ ] Add `test_staged_regress_paper_mode_end_to_end` to `tests/test_stage_and_run.py` exercising the new regress CLI flags + verifying new artifacts.
- [ ] Run full smoke: `bash tests/test_references.sh` — all green.
- [ ] Commit.

---

## Self-review

- Coverage: -009→T8; -013→T1; -014→T2; -015→T3; -016→T4; -020→T4; -021→T3; -030→T5; -032→T5; -035→T6; -036→T7; -037→T6+T7; -039→T7.
- No placeholders above — each task gives exact file paths, exact additions, exact test names.
- Deferred to Plan 5: run_manifest (LEAD-018), --finalize (LEAD-019), Tier-C code parity (LEAD-022), methods.md (LEAD-028), MLflow (LEAD-033), pinning (LEAD-034), summary_report (LEAD-038), model card (LEAD-040), STARD/CONSORT (LEAD-041).

## Out of scope

- Anything in the Plan 5 list above.
