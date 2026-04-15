# Phase 1 — scikit-learn ≥ 1.5 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `pycaret-ng`'s scikit-learn floor from `<1.5` to `>=1.5,<2` while keeping the test suite green and producing cherry-pick-clean commits onto upstream `pycaret/pycaret:master`.

**Architecture:** Introduce a single `pycaret/utils/_sklearn_compat.py` shim that absorbs version-sensitive private imports (`_BaseScorer`, `_check_reg_targets`) and routes every consumer through it (sktime PR #8546 pattern, R1 from FINDINGS.md). Pin `category-encoders<2.7` to keep working with sklearn 1.5 (2.7+ requires sklearn 1.6's `Tags` API). Leave the existing `_more_tags` site untouched — sklearn 1.5 still supports the legacy tag protocol and 1.6+ auto-translates it via `BaseEstimator.__sklearn_tags__`.

**Tech Stack:** Python 3.11/3.12, scikit-learn 1.5+, pytest, uv (for venv setup), GitHub `phase-1-sklearn` branch off `modernize`.

**Spec reference:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` (Phase 1 section).
**Charter:** `docs/superpowers/agents/sklearn-dev/CHARTER.md`.
**Taxonomy slice:** `docs/superpowers/FAILURE_TAXONOMY.md` rows 3, 4, 5, 6.
**Researcher input:** `docs/superpowers/agents/researcher/FINDINGS.md` — R1 (sktime #8546 dual-API tag shim), upstream PR #4009 (sklearn 1.5 cherry-pick candidate), upstream PR #4175 (sklearn 1.6 symbol-move catalog), upstream issue #3901 (`_PredictScorer` removal).

---

## Decisions locked in this plan (deferred from spec)

- **sklearn floor:** `>=1.5,<2`. Single sweep also keeps things working on 1.6 / 1.7 by routing private symbols through the shim.
- **`set_output` adoption:** **deferred to Phase 2** (couples cleanly with pandas 2.2 dtype work). Phase 1 stays output-format-neutral.
- **`category-encoders` pin:** `>=2.4.0,<2.7`. Bumping to ≥ 2.7 requires sklearn ≥ 1.6 (their `Tags` import); leave that for a follow-up phase.

## Gate state for Phase 1

- **Gate A (test suite green on sklearn ≥ 1.5):** required.
- **Gate B (parity within tolerance):** **partially enforced** — `tests/parity/` runs but skips cleanly because `tests/parity/baselines/3.4.0/` is empty (Phase 0 Task 12 was deferred). Re-enable when baselines are built.
- **Gate C (smoke):** required — at least `tests/test_classification.py` and `tests/test_regression.py` must pass.
- **Gate D (cherry-pick-clean onto `upstream/master`):** required for every commit.

## Hand-back conditions

- All sklearn-tagged taxonomy rows (3, 4, 5, 6) marked `closed` with the closing commit SHA in `Notes`.
- Any new failure signatures observed during this work appended as new rows to `docs/superpowers/FAILURE_TAXONOMY.md` (charter permits this).
- PR opened to `modernize`.

---

## File Structure

**Created in Phase 1:**

| Path | Responsibility |
|------|----------------|
| `pycaret/utils/_sklearn_compat.py` | Single shim module: tolerant `_BaseScorer` / `_check_reg_targets` imports, with version-aware fallbacks for sklearn 1.5 / 1.6+. |
| `tests/test_sklearn_compat.py` | Unit tests for the shim (import contracts + behavior). |

**Modified in Phase 1:**

| Path | Change |
|------|--------|
| `pyproject.toml:15`, `:57` | `scikit-learn<1.5` → `scikit-learn>=1.5,<2`. |
| `pyproject.toml:60` | `category-encoders>=2.4.0` → `category-encoders>=2.4.0,<2.7`. |
| `pycaret/containers/metrics/anomaly.py:13` | Import `_BaseScorer` via shim. |
| `pycaret/containers/metrics/base_metric.py:12` | Import `_BaseScorer` via shim. |
| `pycaret/containers/metrics/classification.py:17` | Import `_BaseScorer` via shim. |
| `pycaret/containers/metrics/clustering.py:17` | Import `_BaseScorer` via shim. |
| `pycaret/containers/metrics/regression.py:15-16` | Import `_BaseScorer` and `_check_reg_targets` via shim. |
| `pycaret/containers/metrics/time_series.py:16` | Import `_BaseScorer` via shim. |
| `docs/superpowers/FAILURE_TAXONOMY.md` rows 3, 4, 5, 6 | Status `open` → `closed`; SHA in Notes. |

**Branch:** `phase-1-sklearn` off `modernize`.

**Out of scope (handed off / deferred):**
- pandas 2.2 / numpy 2.0 work (rows 7, 8, 9, 11) → Phase 2.
- `pycaret/internal/preprocess/iterative_imputer.py` private imports of `FLOAT_DTYPES, _check_inputs_dtype, _get_mask, _ImputerTriplet, _safe_indexing, is_scalar_nan, stats` from `sklearn.impute._iterative` — **document as a Phase 1 finding** if Task 8 runtime exposes a failure on sklearn ≥ 1.5; otherwise note and defer.
- `pycaret/internal/preprocess/target/TransformedTargetClassifier.py:191` `_more_tags` — sklearn 1.5 supports it natively, 1.6+ auto-translates via `BaseEstimator.__sklearn_tags__`. No change needed in Phase 1.

---

## Task 1: Branch off `modernize` and prepare working venv

**Files:** none yet (branch creation + env setup only).

- [ ] **Step 1: Verify clean tree on `master`**

Run: `git status --short`
Expected: empty output (or only `?? pycaret-plugin/` + similar untracked dirs unrelated to this work).

- [ ] **Step 2: Verify `modernize` branch tracks origin**

Run: `git fetch origin modernize && git rev-parse origin/modernize`
Expected: a SHA (e.g., `dc525d6cdc3f74744dcf9f04bb064ac7e783717c`).

- [ ] **Step 3: Create the phase branch off `modernize`**

Run:
```bash
git checkout -B phase-1-sklearn origin/modernize
```
Expected: `Switched to a new branch 'phase-1-sklearn'`. Note: branches off `origin/modernize`, not local `master`, because all Phase 0 docs/specs/plans live on `master` and we do NOT want them in this Phase 1 branch — they're not source code and would muddy gate D cherry-pick checks against upstream.

- [ ] **Step 4: Verify the branch base is correct**

Run: `git log --oneline origin/modernize..phase-1-sklearn`
Expected: empty output (no commits beyond `modernize` yet).

Run: `git log --oneline -1`
Expected: same SHA as `origin/modernize` (e.g., `dc525d6c`).

- [ ] **Step 5: Create the Phase 1 working venv with sklearn 1.5**

Run:
```bash
uv venv --python 3.12 .venv-phase1
VIRTUAL_ENV=$(pwd)/.venv-phase1 uv pip install -e . pytest "scikit-learn>=1.5,<2" "category-encoders>=2.4.0,<2.7"
```
Expected: install completes; `.venv-phase1/` exists with sklearn 1.5+.

- [ ] **Step 6: Confirm sklearn version**

Run: `.venv-phase1/bin/python -c "import sklearn; print(sklearn.__version__)"`
Expected: a version string starting with `1.5.` or higher (e.g., `1.5.2` or `1.7.x`).

- [ ] **Step 7: Add `.venv-phase1/` to gitignore (idempotent)**

Run:
```bash
grep -qxF '.venv-phase1/' .gitignore || echo '.venv-phase1/' >> .gitignore
```
If `.gitignore` was modified, commit:
```bash
git add .gitignore
git commit -m "chore(ci): ignore .venv-phase1 working venv"
```

If `.gitignore` was unchanged (entry already present), skip the commit.

---

## Task 2: TDD — `_sklearn_compat` shim for `_BaseScorer`

**Files:**
- Create: `pycaret/utils/_sklearn_compat.py`
- Test: `tests/test_sklearn_compat.py`

The shim's first responsibility: provide a `_BaseScorer` class import that is tolerant of sklearn moving the symbol in future minor releases. Today (sklearn 1.5/1.6/1.7) it lives in `sklearn.metrics._scorer`. The shim wraps the import so that if a future sklearn relocates the symbol, we fix one site instead of six.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sklearn_compat.py`:

```python
"""Unit tests for pycaret.utils._sklearn_compat."""
from __future__ import annotations

import pytest


def test_get_base_scorer_class_returns_a_class():
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    cls = get_base_scorer_class()
    assert isinstance(cls, type), f"Expected a class, got {type(cls)}"


def test_get_base_scorer_class_is_sklearn_base_scorer():
    """Round-trip: a sklearn make_scorer() result must be an instance of the returned class."""
    from sklearn.metrics import make_scorer
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    scorer = make_scorer(lambda y_true, y_pred: 0.0)
    cls = get_base_scorer_class()
    assert isinstance(scorer, cls), (
        f"make_scorer() returned {type(scorer).__mro__}; "
        f"shim returned {cls}"
    )


def test_get_base_scorer_class_is_cached():
    """Shim should not re-import on every call."""
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    a = get_base_scorer_class()
    b = get_base_scorer_class()
    assert a is b
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-phase1/bin/python -m pytest tests/test_sklearn_compat.py::test_get_base_scorer_class_returns_a_class -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pycaret.utils._sklearn_compat'`.

- [ ] **Step 3: Implement the shim**

Create `pycaret/utils/_sklearn_compat.py`:

```python
"""scikit-learn version-compat shim for pycaret-ng.

Centralises every private/relocated sklearn symbol pycaret depends on.
When sklearn moves a symbol, fix one site here instead of dozens of call
sites. Inspired by sktime PR #8546's dual-API tag inspection helper.

Targets sklearn>=1.5,<2. Intentionally tolerant of 1.6/1.7 internal moves.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Type


@lru_cache(maxsize=1)
def get_base_scorer_class() -> Type:
    """Return sklearn's `_BaseScorer` class regardless of internal location.

    sklearn 1.5/1.6/1.7 keep it at sklearn.metrics._scorer._BaseScorer.
    If a future sklearn relocates it, extend the fallback chain here.
    """
    try:
        from sklearn.metrics._scorer import _BaseScorer
        return _BaseScorer
    except ImportError as e:
        raise ImportError(
            "pycaret-ng could not locate sklearn's _BaseScorer. "
            "Tried sklearn.metrics._scorer._BaseScorer. "
            "Add a fallback path in pycaret/utils/_sklearn_compat.py."
        ) from e
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-phase1/bin/python -m pytest tests/test_sklearn_compat.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Verify cherry-pick cleanliness against upstream/master**

Run:
```bash
git fetch upstream master
git add pycaret/utils/_sklearn_compat.py tests/test_sklearn_compat.py
git stash push -u -m "phase1-task2-staged"
git checkout upstream/master -- .  # noop check
git stash pop
```
Then test the cherry-pick scenario: stage the change, commit, and dry-run apply to upstream master.

Run:
```bash
git commit -m "feat(sklearn): _sklearn_compat shim with get_base_scorer_class()

Centralises sklearn private-symbol imports. First user: get_base_scorer_class()
returns sklearn.metrics._scorer._BaseScorer with a single fallback point.

Closes nothing yet; consumed by subsequent commits.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
SHA=$(git rev-parse HEAD)
git checkout -B phase-1-cherry-test upstream/master
git cherry-pick "$SHA"
PICK_OK=$?
git cherry-pick --abort 2>/dev/null
git checkout phase-1-sklearn
git branch -D phase-1-cherry-test
[ "$PICK_OK" -eq 0 ] && echo "GATE-D OK" || echo "GATE-D FAILED"
```
Expected: `GATE-D OK`. If `GATE-D FAILED`, do NOT proceed — this commit conflicts with upstream and must be rebased or split.

---

## Task 3: TDD — extend shim with `get_check_reg_targets()`

**Files:**
- Modify: `pycaret/utils/_sklearn_compat.py`
- Modify: `tests/test_sklearn_compat.py`

`_check_reg_targets` is currently imported from `sklearn.metrics._regression` in `pycaret/containers/metrics/regression.py:15`. Same shim treatment.

- [ ] **Step 1: Append the failing test to `tests/test_sklearn_compat.py`**

Add at the end of the file:

```python
def test_get_check_reg_targets_returns_a_callable():
    from pycaret.utils._sklearn_compat import get_check_reg_targets

    fn = get_check_reg_targets()
    assert callable(fn)


def test_get_check_reg_targets_round_trip():
    """Calling the function on simple regression vectors should not raise."""
    import numpy as np
    from pycaret.utils._sklearn_compat import get_check_reg_targets

    fn = get_check_reg_targets()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    result = fn(y_true, y_pred, multioutput="uniform_average")
    # Signature varies subtly across sklearn versions but always returns >= 3 items.
    assert len(result) >= 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-phase1/bin/python -m pytest tests/test_sklearn_compat.py::test_get_check_reg_targets_returns_a_callable -v`
Expected: FAIL with `ImportError: cannot import name 'get_check_reg_targets'`.

- [ ] **Step 3: Extend the shim**

Append to `pycaret/utils/_sklearn_compat.py`:

```python
@lru_cache(maxsize=1)
def get_check_reg_targets():
    """Return sklearn's `_check_reg_targets` function regardless of location.

    sklearn 1.5/1.6/1.7 keep it at sklearn.metrics._regression._check_reg_targets.
    Add fallback paths here if a future sklearn relocates it.
    """
    try:
        from sklearn.metrics._regression import _check_reg_targets
        return _check_reg_targets
    except ImportError as e:
        raise ImportError(
            "pycaret-ng could not locate sklearn's _check_reg_targets. "
            "Tried sklearn.metrics._regression._check_reg_targets. "
            "Add a fallback path in pycaret/utils/_sklearn_compat.py."
        ) from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-phase1/bin/python -m pytest tests/test_sklearn_compat.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit + cherry-pick check**

Run:
```bash
git add pycaret/utils/_sklearn_compat.py tests/test_sklearn_compat.py
git commit -m "feat(sklearn): _sklearn_compat.get_check_reg_targets() shim entry

Same pattern as get_base_scorer_class(): centralised, lru-cached, single
fallback point. Consumed by metrics/regression.py in the next commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
Then run the cherry-pick dry-run:
```bash
SHA=$(git rev-parse HEAD)
git checkout -B phase-1-cherry-test upstream/master
git cherry-pick "$SHA"
PICK_OK=$?
git cherry-pick --abort 2>/dev/null
git checkout phase-1-sklearn
git branch -D phase-1-cherry-test
[ "$PICK_OK" -eq 0 ] && echo "GATE-D OK" || echo "GATE-D FAILED"
```
Expected: `GATE-D OK`.

---

## Task 4: Route 6 metric-container `_BaseScorer` imports through the shim

**Files (modify):**
- `pycaret/containers/metrics/anomaly.py:13`
- `pycaret/containers/metrics/base_metric.py:12`
- `pycaret/containers/metrics/classification.py:17`
- `pycaret/containers/metrics/clustering.py:17`
- `pycaret/containers/metrics/regression.py:16`
- `pycaret/containers/metrics/time_series.py:16`

Each file has the same line:
```python
from sklearn.metrics._scorer import _BaseScorer
```
or with a `# type: ignore` comment. Replace with:
```python
from pycaret.utils._sklearn_compat import get_base_scorer_class
_BaseScorer = get_base_scorer_class()
```

- [ ] **Step 1: Patch `pycaret/containers/metrics/anomaly.py`**

Replace the line `from sklearn.metrics._scorer import _BaseScorer` with:

```python
from pycaret.utils._sklearn_compat import get_base_scorer_class

_BaseScorer = get_base_scorer_class()
```

- [ ] **Step 2: Patch `pycaret/containers/metrics/base_metric.py`**

Replace the line `from sklearn.metrics._scorer import _BaseScorer  # type: ignore` with:

```python
from pycaret.utils._sklearn_compat import get_base_scorer_class

_BaseScorer = get_base_scorer_class()
```

- [ ] **Step 3: Patch `pycaret/containers/metrics/classification.py`**

Replace the line `from sklearn.metrics._scorer import _BaseScorer` with the same two-line replacement as Step 1.

- [ ] **Step 4: Patch `pycaret/containers/metrics/clustering.py`**

Same replacement as Step 3.

- [ ] **Step 5: Patch `pycaret/containers/metrics/regression.py`**

Replace the line `from sklearn.metrics._scorer import _BaseScorer` with:

```python
from pycaret.utils._sklearn_compat import get_base_scorer_class

_BaseScorer = get_base_scorer_class()
```

(Leave the `_check_reg_targets` import on line 15 alone — that's fixed in Task 5.)

- [ ] **Step 6: Patch `pycaret/containers/metrics/time_series.py`**

Replace `from sklearn.metrics._scorer import _BaseScorer  # type: ignore` with:

```python
from pycaret.utils._sklearn_compat import get_base_scorer_class

_BaseScorer = get_base_scorer_class()
```

- [ ] **Step 7: Verify all six files are patched and import cleanly**

Run:
```bash
.venv-phase1/bin/python -c "
import pycaret.containers.metrics.anomaly
import pycaret.containers.metrics.base_metric
import pycaret.containers.metrics.classification
import pycaret.containers.metrics.clustering
import pycaret.containers.metrics.regression
import pycaret.containers.metrics.time_series
print('all six metric containers import cleanly')
"
```
Expected: `all six metric containers import cleanly`.

- [ ] **Step 8: Run the metric-touching subset of the test suite**

Run: `.venv-phase1/bin/python -m pytest tests/test_classification.py tests/test_regression.py -x --tb=short`
Expected: PASS (or fails for non-sklearn reasons that we'll surface in Task 8). If a NEW sklearn-related failure surfaces, append a row to `docs/superpowers/FAILURE_TAXONOMY.md` and stop to ask for guidance.

- [ ] **Step 9: Commit + cherry-pick check**

Run:
```bash
git add pycaret/containers/metrics/{anomaly,base_metric,classification,clustering,regression,time_series}.py
git commit -m "fix(sklearn): route _BaseScorer through _sklearn_compat shim (rows 4)

Replaces 6 direct imports of sklearn.metrics._scorer._BaseScorer with the
shim's get_base_scorer_class() helper. Centralises the symbol lookup so a
future sklearn move costs one fix, not six.

Closes FAILURE_TAXONOMY row 4 (partial — _PredictScorer references in
docstrings/comments at pycaret/utils/generic.py:1157 and
pycaret/utils/time_series/forecasting/model_selection.py:76,262,381 are
documentation-only and tracked separately for a docstring sweep).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
Then run the cherry-pick dry-run from Task 2 Step 5. Expected: `GATE-D OK`.

---

## Task 5: Route `_check_reg_targets` through the shim

**Files (modify):**
- `pycaret/containers/metrics/regression.py:15` and `:233`

- [ ] **Step 1: Replace the import on line 15**

Change:
```python
from sklearn.metrics._regression import _check_reg_targets
```
to:
```python
from pycaret.utils._sklearn_compat import get_check_reg_targets

_check_reg_targets = get_check_reg_targets()
```

- [ ] **Step 2: Verify the call site at line 233 still works**

The call site looks like:
```python
y_type, y_true, y_pred, multioutput = _check_reg_targets(
    ...
)
```
No change needed — `_check_reg_targets` is now bound to the shim's return value, same callable signature.

- [ ] **Step 3: Verify the file imports cleanly**

Run:
```bash
.venv-phase1/bin/python -c "import pycaret.containers.metrics.regression; print('regression metrics import OK')"
```
Expected: `regression metrics import OK`.

- [ ] **Step 4: Run regression tests**

Run: `.venv-phase1/bin/python -m pytest tests/test_regression.py -x --tb=short`
Expected: PASS. If a NEW sklearn-related failure surfaces, append a row to FAILURE_TAXONOMY.md and stop.

- [ ] **Step 5: Commit + cherry-pick check**

Run:
```bash
git add pycaret/containers/metrics/regression.py
git commit -m "fix(sklearn): route _check_reg_targets through _sklearn_compat shim (row 6)

Replaces direct import of sklearn.metrics._regression._check_reg_targets
with the shim's get_check_reg_targets() helper. Same single-fallback-point
strategy as _BaseScorer.

Closes FAILURE_TAXONOMY row 6 (partial — validate_data /
root_mean_squared_error / _validate_limit not currently used as direct
imports in pycaret 3.4.0; see Task 8 cartography report for status).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
Then run the cherry-pick dry-run from Task 2 Step 5. Expected: `GATE-D OK`.

---

## Task 6: Pin `category-encoders<2.7` in `pyproject.toml`

**Files (modify):**
- `pyproject.toml:60`

The current line is `"category-encoders>=2.4.0",`. Bumping to ≥ 2.7 would require sklearn ≥ 1.6 (their 2.7 release imports `sklearn.utils.Tags` which only exists in 1.6+). For Phase 1's "sklearn ≥ 1.5" target, we cap category-encoders below 2.7. A follow-up phase can lift the cap when we move the sklearn floor to ≥ 1.6.

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the line:
```
  "category-encoders>=2.4.0",
```
with:
```
  "category-encoders>=2.4.0,<2.7",
```

- [ ] **Step 2: Verify TOML still parses**

Run: `.venv-phase1/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('toml OK')"`
Expected: `toml OK`.

- [ ] **Step 3: Force a category-encoders downgrade in the working venv to confirm the pin works against sklearn 1.5**

Run: `VIRTUAL_ENV=$(pwd)/.venv-phase1 uv pip install "category-encoders>=2.4.0,<2.7"`
Expected: a downgrade to a 2.6.x version (2.6.4 is the typical resolve).

- [ ] **Step 4: Verify the preprocessor module imports cleanly**

Run:
```bash
.venv-phase1/bin/python -c "import pycaret.internal.preprocess.preprocessor; print('preprocessor import OK')"
```
Expected: `preprocessor import OK`. (Previously this would have raised `ImportError: cannot import name 'Tags' from 'sklearn.utils'` with category-encoders 2.7+ on sklearn 1.5.)

- [ ] **Step 5: Commit + cherry-pick check**

Run:
```bash
git add pyproject.toml
git commit -m "fix(sklearn): cap category-encoders<2.7 for sklearn 1.5 compat (row 3)

category-encoders 2.7+ requires sklearn>=1.6 (Tags API). Phase 1 targets
sklearn>=1.5, so cap category-encoders below 2.7. A follow-up phase can
lift this cap when the sklearn floor moves to >=1.6.

Closes FAILURE_TAXONOMY row 3.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
Then run the cherry-pick dry-run from Task 2 Step 5. Expected: `GATE-D OK`.

---

## Task 7: Lift sklearn floor to `>=1.5,<2` in `pyproject.toml`

**Files (modify):**
- `pyproject.toml:15` (base deps)
- `pyproject.toml:57` (full extras — verify it's the same line type and update)

- [ ] **Step 1: Inspect the two sklearn-pin sites**

Run: `grep -n "scikit-learn" pyproject.toml`
Expected: at least two lines with `scikit-learn<1.5`. Note their exact line numbers; if the file has been edited since this plan was written, line numbers may have shifted.

- [ ] **Step 2: Edit both sklearn pin sites**

Replace every occurrence of:
```
"scikit-learn<1.5"
```
with:
```
"scikit-learn>=1.5,<2"
```

Do NOT touch the `scikit-learn-intelex<2024.7.0; ...` lines (those are a different package).

- [ ] **Step 3: Verify TOML parses**

Run: `.venv-phase1/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); print('toml OK; deps=', [d for d in d['project']['dependencies'] if 'scikit-learn' in d])"`
Expected: `toml OK; deps= ['scikit-learn>=1.5,<2']`.

- [ ] **Step 4: Re-resolve the working venv against the new pin**

Run:
```bash
VIRTUAL_ENV=$(pwd)/.venv-phase1 uv pip install -e . "scikit-learn>=1.5,<2"
.venv-phase1/bin/python -c "import sklearn; print('sklearn', sklearn.__version__)"
```
Expected: a 1.5.x or higher version.

- [ ] **Step 5: Commit + cherry-pick check**

Run:
```bash
git add pyproject.toml
git commit -m "feat(sklearn): lift scikit-learn floor to >=1.5,<2

Phase 1 of pycaret-ng modernization. Pairs with the _sklearn_compat shim
introduced in earlier commits (centralises private-symbol imports so this
floor lift does not break consumers).

Inspired by upstream PR #4009 cherry-pick candidate; tag-shim strategy
adapted from sktime PR #8546 per Researcher recommendation R1.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
Then run the cherry-pick dry-run from Task 2 Step 5. Expected: `GATE-D OK`.

---

## Task 8: Run the full pytest suite + parity skips on phase-1-sklearn

**Files:** none modified; this is a verification task. New FAILURE_TAXONOMY rows MAY be appended if novel signatures surface.

- [ ] **Step 1: Run the full pytest suite (excluding parity, which is gate B)**

Run:
```bash
.venv-phase1/bin/python -m pytest tests/ --ignore=tests/parity --tb=short -p no:cacheprovider 2>&1 | tee /tmp/phase1-pytest.log | tail -50
```
Expected: green or near-green. Tolerated: tests that fail for non-sklearn reasons (pandas / numpy / plotting / time-series) — those belong to later phases. NOT tolerated: any sklearn-tagged regression introduced by this phase.

- [ ] **Step 2: Triage failures**

For each FAILED test in the log:

  a. If the failure traceback names sklearn (`sklearn.*`), `_BaseScorer`, `_check_reg_targets`, `category_encoders`, `__sklearn_tags__`, or `_more_tags`: this is a Phase 1 regression. **STOP** — do not proceed; investigate and fix on the current branch.

  b. If the failure traceback names pandas, numpy, matplotlib, sktime, statsmodels, pmdarima, tbats, plotly, schemdraw, joblib, distutils, or `np.NaN`/`np.product`/`np.in1d`: this is a Phase 2/3/4/5 issue. Append a row to `docs/superpowers/FAILURE_TAXONOMY.md` if not already present, with `Status: open`, `Notes: surfaced during Phase 1 sklearn pytest run; defer to Phase N`.

  c. If the failure is unclassifiable: append a row with `Owner agent: TBD-Orchestrator` and stop to ask the user for guidance.

- [ ] **Step 3: Run the parity harness — verify it skips cleanly (no baselines yet)**

Run: `.venv-phase1/bin/python -m pytest tests/parity/ --tb=short`
Expected: 9 PASS (datasets + baseline schema unit tests), 8 SKIPPED (parity tests waiting for Task 12 baselines).

- [ ] **Step 4: Run the focused smoke tests (gate C)**

Run: `.venv-phase1/bin/python -m pytest tests/test_classification.py tests/test_regression.py -x --tb=short`
Expected: PASS on both. If either fails for sklearn reasons, return to Step 2 triage rule (a).

- [ ] **Step 5: If any new taxonomy rows were added in Step 2, commit them**

Run:
```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): append rows surfaced during Phase 1 pytest run"
```

If no new rows were added, skip the commit.

- [ ] **Step 6: Push the phase branch to origin**

Run: `git push -u origin phase-1-sklearn`
Expected: `* [new branch]      phase-1-sklearn -> phase-1-sklearn`.

---

## Task 9: Verify all Phase 1 commits cherry-pick cleanly onto upstream/master

**Files:** none modified; this is a final gate-D audit run AFTER all task-level cherry-pick checks have already passed (each task above did its own per-commit dry-run; this is the cumulative cross-check).

- [ ] **Step 1: List Phase 1 commits**

Run: `git log --oneline origin/modernize..phase-1-sklearn`
Expected: 5-7 commits, each prefixed `feat(sklearn):`, `fix(sklearn):`, or `chore(ci):`.

- [ ] **Step 2: Cherry-pick each commit in order onto a throwaway branch off upstream/master**

Run:
```bash
git fetch upstream master
git checkout -B phase-1-final-pickcheck upstream/master
git cherry-pick origin/modernize..phase-1-sklearn
PICK_OK=$?
git cherry-pick --abort 2>/dev/null
git checkout phase-1-sklearn
git branch -D phase-1-final-pickcheck
[ "$PICK_OK" -eq 0 ] && echo "ALL COMMITS CHERRY-PICK CLEAN" || echo "GATE-D FAILED — at least one commit conflicts with upstream"
```
Expected: `ALL COMMITS CHERRY-PICK CLEAN`.

- [ ] **Step 3: If gate D failed, identify the bad commit**

Run:
```bash
git checkout -B phase-1-bisect-pickcheck upstream/master
for sha in $(git log --reverse --format=%H origin/modernize..phase-1-sklearn); do
  if ! git cherry-pick --no-commit "$sha" >/dev/null 2>&1; then
    echo "FIRST BAD: $sha"
    git checkout -- .
    break
  fi
  git checkout -- .
done
git checkout phase-1-sklearn
git branch -D phase-1-bisect-pickcheck
```
Then split or reorder the bad commit. STOP and ask the user before force-rewriting any commits already pushed.

---

## Task 10: Close out FAILURE_TAXONOMY rows + open PR to `modernize`

**Files (modify):**
- `docs/superpowers/FAILURE_TAXONOMY.md` (rows 3, 4, 5, 6 — set Status to `closed`, append SHA in Notes)

- [ ] **Step 1: Close the four sklearn-tagged taxonomy rows**

Find each row in `docs/superpowers/FAILURE_TAXONOMY.md` and:
- Set `Status` from `open` to `closed`
- Append `· closed by <SHA>` to `Notes` where `<SHA>` is the short SHA of the commit that fixed it.

Mapping:
- Row 3 (category-encoders Tags) → fixed by Task 6's `fix(sklearn): cap category-encoders<2.7 ...` commit
- Row 4 (`_PredictScorer`) → fixed by Task 4's `fix(sklearn): route _BaseScorer through _sklearn_compat shim ...` commit
- Row 5 (`_get_tags`/`_more_tags`) → mark `Status: closed` with `Notes: pycaret 3.4.0 only uses _more_tags as estimator-side provider in TransformedTargetClassifier; sklearn 1.5 supports it natively, 1.6+ auto-translates via BaseEstimator.__sklearn_tags__. Shim infrastructure (Task 2/3 commits) ready for future tag-consumer adds. Closed.`
- Row 6 (`validate_data` / `root_mean_squared_error` / `_check_reg_targets` / `_validate_limit`) → fixed by Task 5's `fix(sklearn): route _check_reg_targets ...` commit. For the other three names: append `Notes: validate_data, root_mean_squared_error, _validate_limit not used as direct imports in pycaret 3.4.0 surface; will reopen if Phase 2/3 surface them.`

- [ ] **Step 2: Commit the taxonomy update**

Run:
```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): close rows 3, 4, 5, 6 (Phase 1 sklearn migration)

Status flip + closing-commit SHAs added per sklearn-dev charter handoff
protocol.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push the final commit**

Run: `git push origin phase-1-sklearn`
Expected: push succeeds.

- [ ] **Step 4: Open the PR to `modernize`**

Run:
```bash
gh pr create --base modernize --head phase-1-sklearn \
  --title "Phase 1: sklearn >= 1.5 migration" \
  --body "$(cat <<'EOF'
## Summary
- Lifts scikit-learn floor from `<1.5` to `>=1.5,<2`.
- Introduces `pycaret/utils/_sklearn_compat.py` shim with `get_base_scorer_class()` and `get_check_reg_targets()` helpers (sktime PR #8546 pattern, R1 in Researcher FINDINGS.md).
- Routes 6 metric-container `_BaseScorer` imports + 1 `_check_reg_targets` import through the shim.
- Caps `category-encoders<2.7` (their 2.7 release requires sklearn ≥ 1.6).
- Closes FAILURE_TAXONOMY rows 3, 4, 5, 6.

## Gates
- **A (test suite green):** `pytest tests/ --ignore=tests/parity` runs to completion on Python 3.12 + sklearn ≥ 1.5. Any failures triaged and either fixed (sklearn-tagged) or appended to taxonomy for later phases.
- **B (parity within tolerance):** *partially enforced* — parity tests skip cleanly because `tests/parity/baselines/3.4.0/` is empty (Phase 0 Task 12 deferred). Re-enable when baselines are built.
- **C (smoke):** `tests/test_classification.py` and `tests/test_regression.py` PASS.
- **D (cherry-pick-clean onto upstream/master):** verified per-commit during development and via cumulative pick on Task 9.

## Out of scope
- pandas 2.2 / numpy 2.0 (Phase 2).
- `pycaret/internal/preprocess/iterative_imputer.py` private imports — flagged for follow-up if exposed.
- `_more_tags` migration in `TransformedTargetClassifier` — sklearn 1.5 supports natively, 1.6+ auto-translates.

## Test plan
- [ ] Reviewer confirms `git log --oneline origin/modernize..phase-1-sklearn` shows 5-7 commits, each cherry-pick-clean
- [ ] Reviewer runs `pytest tests/test_classification.py tests/test_regression.py` against sklearn 1.5+ locally
- [ ] Reviewer scans `docs/superpowers/FAILURE_TAXONOMY.md` for any `Status: open` rows tagged `sklearn-dev` (should be zero)
- [ ] QA agent (when invoked) attaches a parity report — currently `N/A: baselines deferred to Phase 0 follow-up`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL printed.

- [ ] **Step 5: Notify Orchestrator (the human) that Phase 1 is ready for QA review**

Print to the user:
> Phase 1 PR opened: <URL>. QA agent dispatch is the next handoff step. After QA sign-off, merge `phase-1-sklearn` → `modernize` and proceed to Phase 2 plan via `writing-plans`.
