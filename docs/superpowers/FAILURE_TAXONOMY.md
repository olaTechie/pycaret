# FAILURE_TAXONOMY

Shared, Cartographer-owned. Every test failure observed under unpinned modern
deps gets a row. Owner agent is set from {sklearn-dev, pandas-dev, plotting-dev,
ts-dev, release}.

## Schema

| Field | Meaning |
|-------|---------|
| ID | Monotonic integer, never reused. |
| Module | `pycaret.<module>` path most implicated. |
| Failing test | `tests/<path>::<test_name>` — first observed, or `N/A (static-analysis)` for rows seeded from upstream issue/PR signals before an empirical run. |
| Error signature | Exception class + first 80 chars of message. Dedup key. |
| Root-cause dep | One of: sklearn, pandas, numpy, scipy, matplotlib, yellowbrick, schemdraw, plotly, plotly-resampler, sktime, pmdarima, statsmodels, tbats, joblib, category-encoders. |
| Owner agent | sklearn-dev \| pandas-dev \| plotting-dev \| ts-dev \| release |
| Status | open \| in-progress \| closed \| degraded |
| Notes | Fix SHA, handoff markers, upstream refs, rationale for tolerance widening. |

## Provenance of current rows

Rows 1-14 below are a **static-analysis seed set** drawn from:
1. Empirical failures observed locally when installing `pycaret==3.4.0` + modern deps: joblib `FastMemorizedFunc.func_id` break, `category_encoders.utils.Tags` import, pycaret `__init__.py` py-version guard against 3.13.
2. Upstream signals in `docs/superpowers/agents/researcher/FINDINGS.md` — specifically upstream PR #4175 (closed Python 3.13 PR — the richest single modernization checklist), issue #3901 (sklearn `_PredictScorer` removal), PR #4040 (`Styler.applymap` fix), and peer-project patterns from sktime, autogluon, FLAML.

An empirical cartography run (Task 15 in the Phase 0 plan) will append additional rows and reclassify any seed row whose reproducer turns out wrong. Seed rows carry the `[seed]` prefix in Notes.

## Environment used to generate this taxonomy

- Python version: 3.11 (baseline venv target); 3.12 / 3.13 for modernization target.
- OS: darwin 25.4.0 (macOS).
- Observations from: `/tmp/pycaret-baseline` (PyCaret 3.4.0 + joblib 1.4.2 + sklearn 1.4.2 + pandas 2.1.4 + numpy 1.26.4 + scipy 1.11.4 + matplotlib 3.7.5), and from upstream PR tracker (see FINDINGS.md).
- Unpinned-run `pip freeze`: **pending** — full-run Cartographer dispatch will append.

## Rows

| ID | Module | Failing test | Error signature | Root-cause dep | Owner agent | Status | Notes |
|----|--------|--------------|------------------|----------------|-------------|--------|-------|
| 1 | pycaret.__init__ | N/A (static-analysis) | `RuntimeError: Pycaret only supports python 3.9, 3.10, 3.11, 3.12.` | general | release | open | [seed] Observed locally on py3.13. Fix: remove / widen the hard-coded version guard at `pycaret/__init__.py:22`. Upstream PR #4175 addresses. |
| 2 | pycaret.internal.memory | N/A (static-analysis) | `AttributeError: 'FastMemorizedFunc' object has no attribute 'func_id'` | joblib | release | open | [seed] Observed locally with joblib 1.3.2. Fixed by upgrading to joblib>=1.4.2 (already in pyproject pin). Confirms pyproject pin is load-bearing; taxonomy row tracks so modernization doesn't drop it. |
| 3 | pycaret.internal.preprocess.preprocessor | N/A (static-analysis) | `ImportError: cannot import name 'Tags' from 'sklearn.utils'` | category-encoders | sklearn-dev | closed | [seed] Observed locally with category-encoders 2.9.0 + sklearn 1.4.2. Upstream category-encoders ≥ 2.7 requires sklearn 1.6+ `Tags` API. Resolved by raising sklearn floor to `>=1.6,<2` and lifting category-encoders cap to `>=2.7,<3` (supersedes T6's temporary `<2.7` cap). Closed by `e23a17cc`. |
| 4 | pycaret.containers.metrics.* | N/A (static-analysis) | `ImportError: cannot import name '_PredictScorer' from 'sklearn.metrics._scorer'` | sklearn | sklearn-dev | closed | [seed] From upstream issue #3901. sklearn 1.4+ removed `_PredictScorer`. Phase 1 rerouted 6 metric containers to consume `_BaseScorer` via `pycaret.utils._sklearn_compat.get_base_scorer_class()` (centralised single-fallback-point per sktime PR #8546 pattern). `_PredictScorer` docstring references at `pycaret/utils/generic.py:1157` and `pycaret/utils/time_series/forecasting/model_selection.py:76,262,381` are documentation-only and tracked separately. Closed by `76bc792c`. |
| 5 | pycaret.internal.preprocess.target.TransformedTargetClassifier | N/A (static-analysis) | `AttributeError: 'BaseEstimator' object has no attribute '_get_tags'` | sklearn | sklearn-dev | closed | [seed] sklearn 1.6 removed `_get_tags`/`_more_tags` in favor of `__sklearn_tags__`. pycaret 3.4.0 uses `_more_tags` ONLY as an estimator-side provider in `TransformedTargetClassifier:191`; sklearn 1.5 supports it natively and 1.6+ auto-translates via `BaseEstimator.__sklearn_tags__`. Shim infrastructure (`_sklearn_compat.get_base_scorer_class()`, `get_check_reg_targets()`) ready for future tag-consumer additions per R1. Closed by Phase 1 design decision (no migration needed for the single estimator-provider site). |
| 6 | pycaret.containers.metrics.regression | N/A (static-analysis) | `ImportError: cannot import name 'validate_data' / 'root_mean_squared_error' / '_check_reg_targets'` | sklearn | sklearn-dev | closed | [seed] From upstream PR #4175 checklist. Phase 1 rerouted `_check_reg_targets` via `pycaret.utils._sklearn_compat.get_check_reg_targets()` and adapted the call site for sklearn 1.7+ signature change (`sample_weight` positional + 5-tuple return) with defensive indexing that works on both 4-tuple (pre-1.7) and 5-tuple (1.7+) shapes. `validate_data`, `root_mean_squared_error`, `IterativeImputer._validate_limit` not currently used as direct imports in pycaret 3.4.0; will reopen if Phase 2/3 surface them. Closed by `7b0c1323`. |
| 7 | pycaret.internal.plots (Styler) | N/A (static-analysis) | `AttributeError: 'Styler' object has no attribute 'applymap'` | pandas | pandas-dev | open | [seed] From upstream PR #4040. `Styler.applymap` removed in pandas 2.1. Fix: replace with `Styler.map`. Cherry-pick candidate. |
| 8 | pycaret.internal.preprocess | N/A (static-analysis) | `AttributeError: 'DataFrame' object has no attribute 'applymap'` | pandas | pandas-dev | open | [seed] From upstream PR #3927. `DataFrame.applymap` deprecated in pandas 2.1, removed in 2.2+. Fix: replace with `DataFrame.map` per R2. |
| 9 | pycaret.internal.preprocess | N/A (static-analysis) | `FutureWarning: SettingWithCopyWarning / ChainedAssignment under CoW` | pandas | pandas-dev | open | [seed] pandas 2.2 enables copy-on-write semantics. Fix per R2: replace in-place column mutation with `df.assign(col=...)`, drop `copy=False` from `astype`/`reindex`. sktime PRs #9764, #9722 are templates. |
| 10 | pycaret.utils._show_versions | N/A (static-analysis) | `ImportError: cannot import name 'distutils.version.LooseVersion'` | general | release | open | [seed] From upstream PR #4175. Python 3.12+ removed distutils. Fix: `from packaging.version import Version`. |
| 11 | pycaret.time_series / pycaret.internal.plots | N/A (static-analysis) | `AttributeError: module 'numpy' has no attribute 'NaN' / 'product' / 'in1d' / 'trapz' / 'bool8' / 'float_'` | numpy | pandas-dev | open | [seed] From upstream PR #4175 + autogluon PRs #5514/#5056/#5615 + keras PR #21032. numpy 2.0 removed scalar aliases and renamed functions. Fix per R3: global sweep `np.NaN→np.nan`, `np.product→np.prod`, `np.in1d→np.isin`, `np.trapz→np.trapezoid`, `np.bool8→np.bool_`, `np.float_→np.float64`. Note: owner is pandas-dev because R3 lives in Phase 2 (pandas/numpy) per spec. |
| 12 | pycaret.time_series (BATS/TBATS) | N/A (static-analysis) | `ImportError` / `AttributeError` under numpy 2 | tbats | ts-dev | degraded | [seed] From upstream PR #4175 + sktime PR #7486. `tbats` is unmaintained and numpy-1-only. Planned outcome: graceful-disable with warning; document in release notes per spec's DEGRADED.md protocol. |
| 13 | pycaret.time_series | N/A (static-analysis) | Breakage under sktime latest (unpinned from `0.31.0,<0.31.1`) | sktime | ts-dev | open | [seed] Spec notes sktime API drift is highest-risk phase. Exact error signatures pending empirical Cartographer run. Fix: may require API narrowing documented in `docs/superpowers/agents/ts-dev/DEGRADED.md`. |
| 14 | pycaret.internal.memory | N/A (static-analysis) | `AttributeError: 'Memory' object has no attribute 'bytes_limit'` | joblib | release | open | [seed] From upstream PR #4175. joblib 1.5 removed `Memory.bytes_limit` in favor of `Memory.reduce_size()`. Fix: migrate the call site or pin `joblib<1.5` (current pyproject pin is `<1.5`, so pin holds for now — but cartography run on unpinned will expose it). |
| 15 | pycaret.internal.preprocess.preprocessor | collection of tests/conftest.py (Phase 1 T2-T6) | `ImportError: cannot import name '_is_pandas_df' from 'sklearn.utils.validation'` | imbalanced-learn | sklearn-dev | closed | Empirical. imblearn 0.13.0 imports sklearn-private `_is_pandas_df` which sklearn 1.8 removed (renamed to public `is_pandas_df`). Fix: bump `imbalanced-learn>=0.14.0,<0.15.0`. Closed by `a88a1cf0`. |
| 16 | pycaret.utils.time_series / pmdarima | tests/test_time_series_*.py (12 collection errors) | `TypeError: check_array() got an unexpected keyword argument 'force_all_finite'` | pmdarima | ts-dev | open | Empirical, Phase 1 T8 pytest run. sklearn 1.8 renamed `force_all_finite` kwarg to `ensure_all_finite`; pmdarima 2.x not yet updated. Defer to Phase 4 (time-series). Options: pin pmdarima to a version that supports sklearn 1.8, monkey-patch `pmdarima.utils.array.check_endog`, or document as DEGRADED if no fix is available. |
| 17 | pycaret.containers.models.classification | tests/test_classification.py::test_classification | `numpy.linalg.LinAlgError: The covariance matrix of class 0 is not full rank` | sklearn | sklearn-dev | closed | Empirical, Phase 1 T8. sklearn 1.8 QDA stricter rank-deficiency check on class covariances; older sklearn silently regularised. Fix: default `reg_param=0.1` in `QuadraticDiscriminantAnalysisContainer`. Closed by `e23a17cc`. |
| 18 | pycaret.internal.plots / yellowbrick | tests/test_classification.py::test_classification (plot_model step) | `yellowbrick.exceptions.YellowbrickTypeError: This estimator is not a classifier` | yellowbrick | plotting-dev | closed | Empirical, Phase 1 T8. Root cause: sklearn ≥1.6 dropped `_estimator_type` (replaced by `__sklearn_tags__`); yellowbrick's `is_classifier` only checks the old attribute. Closed by `7017c0a3` (rewrite `pycaret/internal/patches/yellowbrick.py` to delegate to `sklearn.base.is_classifier`/`is_regressor` and detect clusterer via `__sklearn_tags__`; install via `contextlib.ExitStack` against each yellowbrick consumer module's local binding) + `98bcfa91` (extend patch list to `yellowbrick.classifier.threshold` and `yellowbrick.model_selection.importances`). |
| 19 | tests.test_classification_parallel / tests.test_clustering_engines / tests.test_persistence | 3 collection errors | `ModuleNotFoundError: No module named 'fugue' | 'daal4py' | 'moto'` | test-infra | release | open | Empirical, Phase 1 T8. Soft/test deps not installed in `[test]` extras. Not a Phase 1 regression; tests should gate behind importable check and xfail/skip rather than erroring at collection. Defer to Phase 5 (release hygiene) — add conditional imports or move these tests behind marker-gated extras. |
| 20 | pycaret.internal.pycaret_experiment.tabular_experiment / schemdraw | tests/smoke/test_plotting.py::test_classification_plot[pipeline] (and reg/clu siblings) | `TypeError: Drawing.draw() got an unexpected keyword argument 'ax'` | schemdraw | plotting-dev | closed | Empirical, Phase 3 smoke baseline (commit `10901cd9`). schemdraw 0.16+ (lifted in commit `50979d29`) replaced `Drawing.draw(ax=...)` with `Drawing.draw(canvas=...)` and dropped the `showframe=` kwarg. Closed by `aa22f038` — single-line edit at `tabular_experiment.py:594`. |
| 21 | pycaret.internal.plots / yellowbrick | tests/smoke/test_plotting.py::test_regression_plot[error] (and `residuals`) | `yellowbrick.exceptions.YellowbrickTypeError: This estimator is not a regressor` | yellowbrick | plotting-dev | closed | Empirical, Phase 3 smoke baseline (commit `10901cd9`). Same root cause as row 18 (sklearn ≥1.6 `_estimator_type` removal). Closed by `7017c0a3`. |
| 22 | pycaret.internal.plots / yellowbrick | tests/smoke/test_plotting.py::test_clustering_plot[distance] (and `elbow`, `silhouette`) | `yellowbrick.exceptions.YellowbrickTypeError: The supplied model is not a clustering estimator` | yellowbrick | plotting-dev | closed | Empirical, Phase 3 smoke baseline (commit `10901cd9`). Same root cause as row 18. Closed by `7017c0a3` (`is_clusterer` shim reads `__sklearn_tags__`). Note: `distance` still fails for an unrelated reason — see row 25. |
| 23 | pycaret.regression.oop / plotly | tests/smoke/test_plotting.py::test_regression_plot[residuals_interactive] | `ModuleNotFoundError: No module named 'anywidget'` | anywidget | plotting-dev | open | Empirical, Phase 3 smoke baseline. The `residuals_interactive` regression visualizer transitively requires `anywidget` (plotly's interactive renderer in modern Jupyter). `anywidget` is not in pycaret's deps. Fix candidates: (a) add `anywidget` to plotting deps, (b) make the import lazy and raise `NotImplementedError` if missing, (c) DEGRADED.md if upstream plotly drops the dep. Skip-listed in smoke harness pending decision. |
| 24 | pycaret.internal.pycaret_experiment.tabular_experiment / yellowbrick | tests/smoke/test_plotting.py::test_classification_plot[error] | `ValueError: too many values to unpack (expected 3)` (at yellowbrick/classifier/class_prediction_error.py:148) | yellowbrick | plotting-dev | degraded | Empirical, Phase 3 smoke. Yellowbrick 1.5's ClassPredictionError unpacks a 3-tuple from a sklearn helper (likely confusion_matrix/_check_targets) whose return shape changed under sklearn ≥1.6. Internal to yellowbrick. Per spec policy (b), DEGRADED in commit `58ae9a2b` — see `docs/superpowers/agents/plotting-dev/DEGRADED.md`. Restoration criterion: yellowbrick releases sklearn-1.6-aware update OR pycaret monkey-patches the call site. |
| 25 | pycaret.internal.pycaret_experiment.tabular_experiment / yellowbrick | tests/smoke/test_plotting.py::test_clustering_plot[distance] | `TypeError: percentile() got an unexpected keyword argument 'interpolation'` | yellowbrick (numpy) | plotting-dev | degraded | Empirical, Phase 3 smoke. Yellowbrick's InterclusterDistance still passes `np.percentile(..., interpolation=...)`, but numpy 2 removed the kwarg (replaced by `method=`). Internal to yellowbrick. Per spec policy (b), DEGRADED in commit `58ae9a2b` — see `docs/superpowers/agents/plotting-dev/DEGRADED.md`. Restoration criterion: yellowbrick releases numpy-2-aware update OR pycaret installs a contextual monkey-patch on `np.percentile` for the call duration. |

## Completion checklist (Phase 0)

- [x] Shared schema defined and environment section populated.
- [x] ≥ 10 seed rows drawn from empirical + upstream-signal sources.
- [x] No row has `TBD` owner.
- [x] Every seed row references a concrete upstream PR, issue, or local reproduction.
- [ ] Empirical Cartographer dispatch (Task 15) — **deferred per user decision**; will append real pytest-observed signatures and reclassify seed rows as needed.
- [ ] `pip freeze` from empirical unpinned env — appended on Cartographer run.

## Hand-off to Phase 1

Phase 1 (sklearn Migration Dev) consumes rows 3, 4, 5, 6 (all `owner=sklearn-dev`) plus any sklearn-tagged rows appended by the eventual Cartographer run.
Phase 2 (pandas/numpy Migration Dev) consumes rows 7, 8, 9, 11.
Phase 3 (plotting) — no seed rows yet; Cartographer run expected to produce them.
Phase 4 (ts-dev) consumes rows 12, 13.
Phase 5 (release) consumes rows 1, 2, 10, 14.
