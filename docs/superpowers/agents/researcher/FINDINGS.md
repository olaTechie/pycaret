# Ecosystem Researcher — FINDINGS

**Author:** Ecosystem Researcher agent (pycaret-ng Phase 0, Task 14)
**Date:** 2026-04-15
**Inputs consulted:** `pyproject.toml`, upstream `pycaret/pycaret` issue/PR tracker, peer-project issue/PR trackers for keras, ludwig, sktime, autogluon, FLAML.
**Scope:** Research only — no source code, CI, or taxonomy changes.

---

## 1. Upstream cherry-pick candidates (pycaret/pycaret)

Each reference is tagged `[sklearn]`, `[pandas]`, `[numpy]`, `[py313]`, `[joblib]`, or `[general]` for the dependency it addresses. "Cherry-pick-friendly" flags signal PRs whose diff we can likely lift with minimal modification. "Adapt-only" flags signal closed/stale PRs whose approach we should study, not merge verbatim.

### Merged upstream fixes to diff-mine for pycaret-ng

1. **PR #4009 — "Support scikit-learn 1.5"** (merged). `[sklearn][pandas]` — Canonical upstream bump that also moved the pandas floor to 2.2.x. This is the closest diff to our target state and is the **primary cherry-pick candidate** for the sklearn 1.5 migration. https://github.com/pycaret/pycaret/pull/4009
2. **PR #3857 — "Support Scikit-learn 1.4"** (merged). `[sklearn]` — Previous-generation sklearn compat bump. Useful for understanding upstream's layered fix strategy (feature_names_in_, tag plumbing). Cherry-pick-friendly for incremental context. https://github.com/pycaret/pycaret/pull/3857
3. **PR #3830 — "Support Pandas 2.0"** (merged). `[pandas]` — Upstream's initial pandas 2.x migration (fixes #3722). Core reference for iloc/loc, groupby, and categorical changes. Cherry-pick-friendly. https://github.com/pycaret/pycaret/pull/3830
4. **PR #4040 — "Fix: AttributeError: 'Styler' object has no attribute 'applymap'"** (merged). `[pandas]` — Replaces `Styler.applymap` → `Styler.map` for pandas ≥ 2.1. Small, surgical, must-include. https://github.com/pycaret/pycaret/pull/4040
5. **PR #3987 — "Workflows and dependencies update + python 3.12 support"** (merged). `[py312][general]` — Python 3.12 enablement; gives the CI/dep pattern we will extend to 3.13. https://github.com/pycaret/pycaret/pull/3987

### Closed-but-unmerged PRs with harvestable diffs (adapt-only)

6. **PR #4175 — "feat: Add Python 3.13 support"** (closed, unmerged). `[py313][numpy][sklearn][joblib]` — **Richest single reference for our work.** PR body enumerates every modernization pain point we face: `np.NaN→np.nan`, `np.product→np.prod`, `distutils.version.LooseVersion→packaging.version.Version`, sklearn 1.6+ `validate_data`/`root_mean_squared_error`/`_check_reg_targets`/`IterativeImputer._validate_limit`, joblib 1.5 `Memory.bytes_limit`→`reduce_size()`, `mock_s3→mock_aws` in tests, evidently legacy import paths, and BATS/TBATS graceful degradation. Treat the PR body as an annotated migration checklist. Diff is closed-unmerged so adapt rather than straight cherry-pick. https://github.com/pycaret/pycaret/pull/4175
7. **PR #4164 — "Change to make it compatible with numpy>=2.0"** (open). `[numpy]` — Open draft from an external contributor; check diff for concrete `np.*` replacements before we write our own. https://github.com/pycaret/pycaret/pull/4164
8. **PR #4157 — "Update numpy requirement … to >=1.21,<2.1"** (open dependabot). `[numpy]` — Signals where upstream was willing to cap. We go further (≥ 2.0 uncapped); the PR's CI log is a useful failure catalog. https://github.com/pycaret/pycaret/pull/4157
9. **PR #3927 — "applymap deprecated since pandas 2.1"** (closed). `[pandas]` — Targeted diff for `DataFrame.applymap`→`DataFrame.map`. Stale but surgically useful. https://github.com/pycaret/pycaret/pull/3927
10. **PR #3832 — "Support pandas 2.1"** (closed). `[pandas]` — Bridges 2.0 (merged in #3830) and 2.2 (pulled in via #4009). Useful diff for FutureWarning hunts. https://github.com/pycaret/pycaret/pull/3832
11. **PR #3824 — "Initial phase to support pandas 2"** (closed). `[pandas]` — Earliest pandas-2 scouting PR; good for historical context on what broke first. https://github.com/pycaret/pycaret/pull/3824
12. **PR #3683 — "Relax pandas version to pandas<2.1"** (closed). `[pandas]` — Shows upstream's intermediate ceiling; ignore the cap, study the tests. https://github.com/pycaret/pycaret/pull/3683
13. **PR #3921 — "support polars"** (open). `[general]` — Tangential but signals upstream's dataframe-abstraction appetite; pycaret-ng should keep pandas-first but note the direction. https://github.com/pycaret/pycaret/pull/3921

### Open issues we must treat as regression-test seeds

14. **Issue #4123 — "[ENH]: Support for pandas 2.2"** (open). `[pandas]` — Tracker for the exact gap pycaret-ng closes. https://github.com/pycaret/pycaret/issues/4123
15. **Issue #3908 — "[BUG]: Incompatible with Pandas 2.0"** (open). `[pandas]` — Concrete reproducer seeds. https://github.com/pycaret/pycaret/issues/3908
16. **Issue #4148 — "[BUG]: Cannot cast object dtype to float64"** (open). `[pandas]` — Likely PDEP-14 (dtype inference) fallout; keep as parity test candidate. https://github.com/pycaret/pycaret/issues/4148
17. **Issue #3901 — "ImportError: cannot import name '_PredictScorer' from 'sklearn.metrics._scorer'"** (open). `[sklearn]` — sklearn 1.4+ removed private import; exact symbol our patches must substitute. https://github.com/pycaret/pycaret/issues/3901
18. **Issue #4133 — "[INSTALL]:"** (open). `[general]` — User reports around resolver thrash with modern deps; worth scanning for user-facing symptoms. https://github.com/pycaret/pycaret/issues/4133

---

## 2. Peer migration patterns

Each peer is summarized under three headings: **sklearn ≥ 1.5**, **pandas ≥ 2.2**, **numpy ≥ 2.0**. PR numbers cited where available.

### keras (keras-team/keras)

- **sklearn ≥ 1.5:** Keras ships a first-party sklearn wrapper (`SKLearnClassifier`/`SKLearnRegressor`). The wrapper was added in PR #20599 ("FEAT add scikit-learn wrappers", merged) and hardened in PR #21387 ("Fix missing and fragile scikit-learn imports in Keras sklearn wrappers", merged), which moved to tolerant imports and the public `sklearn.utils._tags` surface. PR #20657 ("Make sklearn dependency optional", merged) demonstrates **conditional-import pattern** — sklearn is imported lazily and behind `_is_sklearn_available()` guards so failures are isolated. PR #21843 ("Fix failing sklearn tests following release of pytest 9.0", merged) keeps the test suite current.
  - URLs: https://github.com/keras-team/keras/pull/20599 · https://github.com/keras-team/keras/pull/20657 · https://github.com/keras-team/keras/pull/21387 · https://github.com/keras-team/keras/pull/21843
- **pandas ≥ 2.2:** Keras does not depend on pandas in the core package; not directly instructive.
- **numpy ≥ 2.0:** PR #21032 ("Make code compatible with Numpy >= 2.1", merged) is the canonical diff — replaces `np.product`, `np.cumproduct`, `np.NaN`, scalar type aliases, `np.in1d`, and tightens dtype promotion. PR #22141 ("Fix sparse reshape test with Numpy 2.4", merged) is the follow-up for NumPy 2.4 sparse array semantics. PR #20725 ("Patch tf2onnx to ensure compatibility with numpy>=2.0.0", merged) shows the downstream-patching pattern keras uses for deps that lag.
  - URLs: https://github.com/keras-team/keras/pull/21032 · https://github.com/keras-team/keras/pull/22141 · https://github.com/keras-team/keras/pull/20725

### ludwig (ludwig-ai/ludwig)

- **sklearn ≥ 1.5:** PR #1684 and PR #3185 ("Unpin scikit-learn", both merged) document ludwig's **"unpin-then-fix-forward"** approach — lift the ceiling, let CI break, then patch. Historically they had pinned via PR #2850 ("Pin scikit-learn<1.2.0") when regressions landed; the pattern is "unpin by default, pin defensively only on confirmed breakage."
  - URLs: https://github.com/ludwig-ai/ludwig/pull/1684 · https://github.com/ludwig-ai/ludwig/pull/2850 · https://github.com/ludwig-ai/ludwig/pull/3185
- **pandas ≥ 2.2 + numpy ≥ 2.0:** PR #4041 ("Add Python 3.10, 3.11, 3.12 compatibility", closed) and PR #4059 ("Modernize Ludwig for v0.11 release", merged) bundle dep bumps behind a versioned release. Ludwig's pattern is to **gate modernization behind a minor release** (`v0.11`) rather than dripping patches — good governance precedent for our pycaret-ng cutover.
  - URLs: https://github.com/ludwig-ai/ludwig/pull/4041 · https://github.com/ludwig-ai/ludwig/pull/4059

### sktime (sktime/sktime)

- **sklearn ≥ 1.5:** PR #6462 ("Update scikit-learn requirement from <1.5.0,>=0.24 to >=0.24,<1.6.0", merged) lifted the ceiling. PR #8546 ("[BUG] fix sklearn tag inspection in scikit-learn < 1.6", merged) is the **key pattern**: sktime introduces a compatibility shim that calls `__sklearn_tags__` on 1.6+ and falls back to the legacy `_more_tags`/`_get_tags` on older versions. This is the cleanest way to support a range of sklearn versions in one codebase.
  - URLs: https://github.com/sktime/sktime/pull/6462 · https://github.com/sktime/sktime/pull/8546
- **pandas ≥ 2.2 (CoW / FutureWarning hygiene):** PR #9764 ("fix: use assign() in _calendar_dummies to silence pandas CoW FutureWarning", open) and PR #9722 ("fix: remove deprecated copy=False from astype in WindowSummarizer", open) show sktime's **concrete CoW migration tactics**: replace in-place column sets with `.assign()`, drop `copy=False` from `astype`. These are the exact idioms pycaret-ng will need.
  - URLs: https://github.com/sktime/sktime/pull/9764 · https://github.com/sktime/sktime/pull/9722
- **numpy ≥ 2.0:** PR #7486 ("[MNT] enable pmdarima under numpy 2", merged) is directly relevant — pycaret-ng also depends on pmdarima; sktime's approach (constrain to pmdarima ≥ 2.0.4 wheel that ships numpy-2-compatible ABI) is a template. PR #6627 ("DIAGNOSTIC - numpy 2 and no soft dependencies, test all estimators", open) is the **diagnostic harness pattern** — run all estimators against a stripped-down numpy-2-only env to flag issues. PR #9218 ("Bump numpy from 1.21.0 to 2.4.0", closed dependabot) tracks the upper bound.
  - URLs: https://github.com/sktime/sktime/pull/7486 · https://github.com/sktime/sktime/pull/6627 · https://github.com/sktime/sktime/pull/9218

### autogluon (autogluon/autogluon)

- **sklearn ≥ 1.5:** PR #4420 ("Upgrade scikit-learn to 1.5.1", merged) is the canonical single-commit bump — touches estimator HTML repr, feature-name plumbing, and `set_output` containers. Clean diff, excellent reference.
  - URL: https://github.com/autogluon/autogluon/pull/4420
- **pandas ≥ 2.2:** autogluon tracks pandas continuously via dependabot; no single flagship PR, but the `_check_pandas_version` guards throughout `autogluon.common` show a **defensive-gating pattern** where optional features (e.g., `pyarrow`-backed dtypes) are probed at runtime.
- **numpy ≥ 2.0:** PR #4538 ("Upgrade to numpy 2.0", merged) is the top-level bump. PR #5615 ("[tabular] support NumPy 2.x and fix compatibility issues", merged) is the **surgical follow-up** that patches the tabular trainer's dtype handling for numpy-2 scalar-promotion changes. PR #5514 ("Replace np.in1d with np.isin for NumPy 2.4 compatibility", merged) and PR #5056 ("fix numpy deprecation warning (np.trapz)", merged) are line-level references for exact symbol replacements.
  - URLs: https://github.com/autogluon/autogluon/pull/4538 · https://github.com/autogluon/autogluon/pull/5615 · https://github.com/autogluon/autogluon/pull/5514 · https://github.com/autogluon/autogluon/pull/5056

### FLAML (microsoft/FLAML)

- **sklearn ≥ 1.5:** No single flagship PR found, but FLAML relies on sklearn's public estimator API and tags, keeping direct exposure minimal. The pattern of interest is FLAML's **version probe** in `flaml.automl.ml` (conditional import on `sklearn.__version__`) — a useful lightweight alternative to the sktime shim.
- **pandas ≥ 2.2:** PR #1527 ("Fix pandas 3.0 compatibility: StringDtype, datetime resolution, deprecated APIs, pyspark.pandas import", merged) is **far-sighted** — already prepping for pandas 3.0. Patterns covered: `pd.StringDtype()` plumbing, `datetime64[ns]→datetime64[us]` resolution changes, `pd.Index.applymap→pd.Index.map`. Directly reusable.
  - URL: https://github.com/microsoft/FLAML/pull/1527
- **numpy ≥ 2.0:** PR #1424 ("Numpy 2.x is not supported yet", merged) added a pin; PR #1426 ("Revert …", merged) removed it once downstreams caught up — a case study in **conservative pin-then-revert** governance. PR #1485 ("Add NumPy 2.0 compatibility test suite", closed) contains the test harness that would have codified support.
  - URLs: https://github.com/microsoft/FLAML/pull/1424 · https://github.com/microsoft/FLAML/pull/1426 · https://github.com/microsoft/FLAML/pull/1485

---

## 3. Recommended adoption patterns for pycaret-ng

Three concrete, non-generic recommendations. Each is phrased as "When fixing X in PyCaret's Y module, apply pattern Z as seen in <peer project reference>."

### R1 — sklearn estimator tags

**When fixing** `__sklearn_tags__` / `_more_tags` / `_get_tags` breakage in `pycaret/internal/pipeline.py`, `pycaret/internal/preprocess/transformers.py`, and every custom-estimator site under `pycaret/containers/models/`, **apply the dual-API shim pattern as seen in sktime PR #8546** (https://github.com/sktime/sktime/pull/8546). Create a single helper — e.g. `pycaret/utils/_sklearn_compat.py::get_tags(est)` — that returns tags via `__sklearn_tags__()` when available (sklearn ≥ 1.6) and falls back to `_get_tags()` / `_more_tags()` for 1.5 and earlier. Route every `hasattr(..., '_get_tags')` / direct `_more_tags` call through this helper. Rationale: we already target sklearn ≥ 1.5 and want forward-compat with 1.6/1.7 without a second migration; sktime's shim is battle-tested across thousands of estimators.

### R2 — pandas CoW & applymap migration

**When fixing** pandas-2.2 `SettingWithCopyWarning`, `FutureWarning`, and `applymap` deprecations in `pycaret/internal/preprocess/`, `pycaret/internal/plots/`, and all Styler-formatted output in `pycaret/classification.py` / `regression.py`, **apply the "assign + map + no-copy" pattern as seen in sktime PRs #9764 and #9722** (https://github.com/sktime/sktime/pull/9764, https://github.com/sktime/sktime/pull/9722) **combined with PyCaret upstream PR #4040's `Styler.applymap→Styler.map` fix** (https://github.com/pycaret/pycaret/pull/4040). Concretely: (a) replace in-place column mutations (`df[col] = ...`) with `df = df.assign(col=...)` when the source is not guaranteed to be a unique owner; (b) remove `copy=False` arguments from `astype`, `reindex`, and `to_numpy`; (c) replace every `DataFrame.applymap` with `DataFrame.map` and every `Styler.applymap` with `Styler.map`. FLAML PR #1527 (https://github.com/microsoft/FLAML/pull/1527) gives the template for the datetime-resolution follow-up we will inevitably hit.

### R3 — numpy 2.0 scalar/API migration with pmdarima constraint

**When fixing** numpy-2.0 regressions in `pycaret/time_series/`, `pycaret/internal/plots/`, and any module touching `np.NaN`, `np.product`, `np.in1d`, `np.trapz`, or deprecated scalar aliases (`np.bool8`, `np.float_`), **apply the line-level replacement pattern as seen in autogluon PRs #5514, #5056, and #5615** (https://github.com/autogluon/autogluon/pull/5514, /pull/5056, /pull/5615) **and lift the pmdarima constraint approach from sktime PR #7486** (https://github.com/sktime/sktime/pull/7486). Concretely: (a) global sweep replacing `np.NaN→np.nan`, `np.product→np.prod`, `np.in1d→np.isin`, `np.trapz→np.trapezoid`, `np.float_→np.float64`, `np.bool8→np.bool_`; (b) require `pmdarima>=2.0.4` in `pyproject.toml` so the numpy-2-ABI wheel is picked up; (c) borrow PyCaret PR #4175's BATS/TBATS graceful-fallback block (https://github.com/pycaret/pycaret/pull/4175) since `tbats` remains numpy-1-only and unmaintained — disable with a warning on numpy 2. This avoids forking `tbats` and keeps the `compare_models` surface stable minus two forecasters clearly documented in release notes.

---

## Appendix: method & provenance

- All URLs resolved against github.com on 2026-04-15.
- `gh search issues` / `gh search prs` used with repo scoping; no WebFetch required.
- Upstream (`pycaret/pycaret`) queried for: "scikit-learn", "pandas", "numpy", "numpy 2", and targeted PR numbers (3830, 4009, 4040, 4164, 4175).
- Peers queried per charter spec: keras-team/keras, ludwig-ai/ludwig, sktime/sktime, autogluon/autogluon, microsoft/FLAML.
- No source code was modified during this research. FAILURE_TAXONOMY.md not touched. No CI workflows authored.
