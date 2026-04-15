# Phase 2 — pandas ≥ 2.2 + numpy ≥ 2.0 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift pycaret-ng's pandas floor from `<2.2` to `>=2.2,<3` and numpy ceiling from `<1.27` to `<3`, fixing the small set of pandas/numpy API regressions this surfaces and eliminating the `distutils.LooseVersion` import that blocks Python 3.12+.

**Architecture:** pycaret 3.4.0 source turns out to be remarkably clean on pandas/numpy — upstream PRs #4040 (`Styler.applymap→map`) and #3927 (`DataFrame.applymap→map`) have already propagated to our fork, and there are no `copy=False` hot spots. Concrete fix sites are just three: `np.product→np.prod`, `np.NaN→np.nan`, and `distutils.LooseVersion→packaging.version.Version` (6 usages in one file). No compatibility shim needed — the migrations are direct API renames, symmetric with upstream patches. `sklearn.set_output(transform="pandas")` adoption is **explicitly deferred**: pycaret 3.4.0 doesn't use it, adopting it would be a new-feature addition, and YAGNI applies.

**Tech Stack:** Python 3.11/3.12, pandas ≥ 2.2, numpy ≥ 2.0 (NOT pinned high), scikit-learn ≥ 1.6 (from Phase 1), pytest, `phase-2-pandas` branch off `modernize` **after Phase 1 PR merges**.

**Spec reference:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` (Phase 2 section, §4).
**Charter:** `docs/superpowers/agents/pandas-dev/CHARTER.md`.
**Taxonomy slice:** `docs/superpowers/FAILURE_TAXONOMY.md` rows 7, 8, 9, 11 (pandas-dev tagged); row 10 (distutils, owned by release) pulled forward because it blocks Python 3.12+ runtime.
**Researcher input:** `docs/superpowers/agents/researcher/FINDINGS.md` — R2 (sktime PRs #9764, #9722 for CoW + assign; upstream PR #4040 for Styler.applymap→map; FLAML PR #1527 for datetime resolution) + R3 (numpy 2.0 scalar/API sweep; autogluon PRs #5514, #5056, #5615; keras PR #21032).
**Prerequisite:** Phase 1 PR `#1` (`phase-1-sklearn` → `modernize`) must be merged before this plan starts. Task 1 verifies this.

---

## Decisions locked in this plan

- **pandas floor:** `>=2.2,<3`. Single sweep targeting 2.2+ because (a) upstream PR #4009 already moved the floor to 2.2.x, and (b) the broader ecosystem (sktime, autogluon, FLAML) is now past 2.1.
- **numpy ceiling:** `<3`. Lift current `<1.27` cap; keep lower bound at `>=1.26` to ensure a modern minor (numpy 1.26 is the last 1.x release and the floor most downstream deps currently require). pmdarima's `force_all_finite` breakage is Phase 4, not Phase 2.
- **Row 7, 8 status:** seed rows were preemptive — no `applymap` sites remain in pycaret source. Rows will be closed with status note `"no fix needed — source already migrated upstream"`, not skipped.
- **CoW adoption (seed row 9):** pycaret source has no `df[col] = ...` in-place mutation pattern and no `copy=False` in astype/to_numpy. Row 9 closes without code changes; a defensive note goes into the pandas-dev charter appendix for Phase 2+ contributors.
- **`set_output(transform="pandas")`:** **deferred indefinitely.** YAGNI — pycaret's pipeline and preprocessing code use numpy arrays + column-name tracking (`feature_names_in_`); adopting `set_output` would require restructuring the `TransformerWrapper` output path. Not worth it just for cleanliness.
- **`distutils.LooseVersion`:** **controller-extended scope.** Row 10 was owned by `release` (Phase 5) in the seed taxonomy but is pulled forward because Python 3.12 removes `distutils` entirely. Fix uses `packaging.version.Version` (already a transitive dep via many packages).

## Gate state for Phase 2

- **Gate A (test suite green on pandas ≥ 2.2 + numpy ≥ 2.0):** required.
- **Gate B (parity within tolerance):** **partially enforced** — same as Phase 1, parity tests skip cleanly until Phase 0 Task 12 baselines are built.
- **Gate C (smoke):** required — `tests/test_classification.py::test_classification` must pass its setup + compare_models + tune + stack + predict path (same scope as Phase 1 end state; `plot_model` still blocked by Phase 3 yellowbrick — row 18).
- **Gate D (cherry-pick-clean onto `upstream/master`):** required for every code commit. Docs-only commits (taxonomy updates) exempt.

## Hand-back conditions

- All pandas-dev-tagged taxonomy rows (7, 8, 9, 11) marked `closed` with closing-commit SHA or `no-fix-needed` rationale in Notes.
- Row 10 (distutils, release-tagged) also closed with closing SHA.
- Any new failure signatures observed during Phase 2 testing appended as new rows.
- PR opened to `modernize`.

---

## File Structure

**Created in Phase 2:**

| Path | Responsibility |
|------|----------------|
| `tests/test_pandas_numpy_compat.py` | Unit tests for the three API migrations (np.prod, np.nan, packaging.Version) — verifies behaviour survives pandas ≥ 2.2 + numpy ≥ 2.0. |

**Modified in Phase 2:**

| Path | Change |
|------|--------|
| `pycaret/internal/patches/sklearn.py:106` | `np.product(sizes, dtype=np.uint64)` → `np.prod(sizes, dtype=np.uint64)`. |
| `pycaret/internal/preprocess/preprocessor.py:682` | `np.NaN` → `np.nan`. |
| `pycaret/utils/_dependencies.py:5-79` | Replace `from distutils.version import LooseVersion` with `from packaging.version import Version`. Update 6 type annotations + call sites. |
| `pyproject.toml:52` | `numpy>=1.21, <1.27` → `numpy>=1.26,<3`. |
| `pyproject.toml:53` | `pandas<2.2` → `pandas>=2.2,<3`. |
| `docs/superpowers/FAILURE_TAXONOMY.md` rows 7, 8, 9, 10, 11 | Status `open` → `closed`; SHA or rationale in Notes. |

**Branch:** `phase-2-pandas` off `modernize` (**AFTER Phase 1 PR merges**).

**Out of scope (handed off / deferred):**
- Plotting stack (yellowbrick classifier detection, row 18) → Phase 3.
- Time-series stack (pmdarima `force_all_finite`, row 16) → Phase 4.
- Soft-dep test infra (fugue/daal4py/moto, row 19) → Phase 5.
- `set_output(transform="pandas")` adoption — deferred indefinitely per "Decisions locked" above.

---

## Task 1: Verify Phase 1 merged, branch off `modernize`, prepare venv

**Files:** none yet (branch creation + env setup only).

- [ ] **Step 1: Verify Phase 1 PR has merged**

Run:
```bash
gh pr view 1 --repo olaTechie/pycaret --json state,mergeCommit -q '.state + " " + (.mergeCommit.oid // "none")'
```
Expected: `MERGED <sha>` — if it prints `OPEN <sha>` or `MERGED none`, **STOP**. Phase 2 must not start before Phase 1 merges.

If you see `OPEN`, report BLOCKED with the message "Phase 1 PR #1 still open; cannot start Phase 2 until it merges."

- [ ] **Step 2: Fetch the merged `modernize` tip**

Run:
```bash
git fetch origin modernize upstream master
git rev-parse origin/modernize
```
Expected: a SHA newer than `beca64cb` (which was modernize's state before Phase 1). Should include the merged Phase 1 commits.

- [ ] **Step 3: Create the phase branch**

Run:
```bash
git checkout -B phase-2-pandas origin/modernize
git log --oneline origin/modernize..phase-2-pandas
```
Expected: second command prints empty (no commits ahead yet).

- [ ] **Step 4: Create the Phase 2 working venv**

Run:
```bash
uv venv --python 3.12 .venv-phase2
VIRTUAL_ENV=$(pwd)/.venv-phase2 uv pip install -e . pytest "pandas>=2.2,<3" "numpy>=1.26,<3"
```
Expected: install completes. Note: pycaret's current pyproject still has `pandas<2.2` + `numpy<1.27` pins at this point — the install command's explicit pins override them. Task 5 and 6 will update pyproject to match.

- [ ] **Step 5: Confirm pandas + numpy versions**

Run:
```bash
.venv-phase2/bin/python -c "import pandas, numpy; print('pandas', pandas.__version__); print('numpy', numpy.__version__)"
```
Expected: pandas 2.2.x (or higher), numpy 1.26.x or 2.0+.

- [ ] **Step 6: Confirm preprocessor still imports (Phase 1 carry-over sanity check)**

Run:
```bash
.venv-phase2/bin/python -c "import pycaret.internal.preprocess.preprocessor; print('preprocessor import OK')"
```
Expected: `preprocessor import OK`. If this fails with any pandas/numpy error, the Phase 2 migration must address it before proceeding.

- [ ] **Step 7: Add `.venv-phase2/` to .gitignore (idempotent)**

Run:
```bash
grep -qxF '.venv-phase2/' .gitignore || { echo '.venv-phase2/' >> .gitignore; git add .gitignore; git commit -m "chore(ci): ignore .venv-phase2 working venv"; }
```

---

## Task 2: TDD — replace `np.product` with `np.prod` in sklearn patches

**Files:**
- Modify: `pycaret/internal/patches/sklearn.py:106`
- Create: `tests/test_pandas_numpy_compat.py`

`np.product` was removed in numpy 2.0 (replaced by `np.prod` since numpy 1.25). Autogluon PR #5615 and keras PR #21032 are the canonical reference patches.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pandas_numpy_compat.py`:

```python
"""Unit tests for pycaret-ng Phase 2 pandas/numpy compatibility migrations.

These tests verify that the module-level APIs pycaret depends on continue to
behave correctly under pandas >=2.2 and numpy >=2.0, pinning the exact
contracts the migrations establish.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_numpy_prod_available_in_sklearn_patch():
    """pycaret/internal/patches/sklearn.py must use np.prod (numpy 2.0-safe).

    Regression test for numpy 2.0 removing np.product. If the patch module is
    ever re-edited to reintroduce np.product, this test fails.
    """
    import pycaret.internal.patches.sklearn as sk_patch
    import inspect
    src = inspect.getsource(sk_patch)
    assert "np.product" not in src, (
        "pycaret/internal/patches/sklearn.py still contains np.product; "
        "numpy 2.0 removed it. Replace with np.prod."
    )
    assert "np.prod(" in src, (
        "Expected np.prod(...) call in pycaret/internal/patches/sklearn.py"
    )


def test_numpy_prod_matches_legacy_product():
    """np.prod must return the same value np.product returned on numpy<2.0."""
    sizes = [2, 3, 4]
    assert np.prod(sizes, dtype=np.uint64) == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py::test_numpy_prod_available_in_sklearn_patch -v`
Expected: FAIL — assertion message "still contains np.product".

- [ ] **Step 3: Patch `pycaret/internal/patches/sklearn.py`**

Find the line:
```python
        total = int(np.product(sizes, dtype=np.uint64))
```
Replace with:
```python
        total = int(np.prod(sizes, dtype=np.uint64))
```

(Only the one site on line 106. Do NOT introduce a compat helper — `np.prod` has been available since numpy 1.25 which is below our floor of 1.26, so no fallback needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pycaret/internal/patches/sklearn.py tests/test_pandas_numpy_compat.py
git commit -m "fix(numpy): np.product -> np.prod in sklearn grid-size patch (row 11)

numpy 2.0 removed np.product (deprecated since 1.25). np.prod is the
canonical replacement and is available on our floor (numpy>=1.26).

Partial closure of FAILURE_TAXONOMY row 11 (numpy scalar/API sweep).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: TDD — replace `np.NaN` with `np.nan` in preprocessor

**Files:**
- Modify: `pycaret/internal/preprocess/preprocessor.py:682`
- Modify: `tests/test_pandas_numpy_compat.py` (append a test)

numpy 2.0 removed the uppercase `np.NaN` alias (and `np.NAN`, `np.Inf`, `np.INF`, `np.PINF`, `np.NINF`). Only lowercase `np.nan` / `np.inf` remain. Keras PR #21032 Sec 2 is the reference sweep.

- [ ] **Step 1: Append the failing test**

Append to `tests/test_pandas_numpy_compat.py`:

```python
def test_preprocessor_uses_lowercase_np_nan():
    """pycaret/internal/preprocess/preprocessor.py must use np.nan (numpy 2.0-safe).

    numpy 2.0 removed np.NaN / np.NAN uppercase aliases. Regression test.
    """
    import pycaret.internal.preprocess.preprocessor as pp
    import inspect
    src = inspect.getsource(pp)
    assert "np.NaN" not in src, (
        "pycaret/internal/preprocess/preprocessor.py still uses np.NaN; "
        "numpy 2.0 removed it. Replace with np.nan."
    )
    assert "np.NAN" not in src, (
        "pycaret/internal/preprocess/preprocessor.py still uses np.NAN; "
        "numpy 2.0 removed it. Replace with np.nan."
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py::test_preprocessor_uses_lowercase_np_nan -v`
Expected: FAIL — "still uses np.NaN".

- [ ] **Step 3: Patch `pycaret/internal/preprocess/preprocessor.py`**

Find the line (around line 682):
```python
                mapping[key].setdefault(np.NaN, -1)
```
Replace with:
```python
                mapping[key].setdefault(np.nan, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pycaret/internal/preprocess/preprocessor.py tests/test_pandas_numpy_compat.py
git commit -m "fix(numpy): np.NaN -> np.nan in OrdinalEncoder mapping (row 11)

numpy 2.0 removed uppercase np.NaN/np.NAN/np.Inf aliases in favour of
lowercase np.nan/np.inf. Only one site in pycaret 3.4.0 — the preprocessor
sets a NaN -> -1 default in the OrdinalEncoder mapping.

Partial closure of FAILURE_TAXONOMY row 11 (numpy scalar/API sweep).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: TDD — replace `distutils.LooseVersion` with `packaging.version.Version`

**Files:**
- Modify: `pycaret/utils/_dependencies.py` (6 usages across ~80 lines)
- Modify: `tests/test_pandas_numpy_compat.py` (append tests)

Python 3.12 removed `distutils` entirely. `pycaret/utils/_dependencies.py` currently uses `LooseVersion` for 6 things: one import, one return-type annotation, one assignment inside `get_module_version`, one dict-value annotation, one assignment inside `get_installed_modules`, and one `_get_module_version` return-type annotation. `packaging.version.Version` is the direct replacement and is already a transitive dependency of pycaret-ng (pip, setuptools, numpy, and pandas all depend on it).

**OVERRIDE FROM CONTROLLER:** this fix is pulled forward from Phase 5 (row 10, owner: `release`) into Phase 2 because it blocks all work on Python 3.12+ (the target CI matrix). Commit message prefix stays `fix(release):` to keep the row-10 provenance clear even though the commit lands in the Phase 2 branch.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_pandas_numpy_compat.py`:

```python
def test_dependencies_module_imports_on_python312plus():
    """pycaret/utils/_dependencies.py must not import distutils.

    Python 3.12 removed distutils entirely. The module must use packaging.version
    instead.
    """
    import pycaret.utils._dependencies as deps
    import inspect
    src = inspect.getsource(deps)
    assert "distutils" not in src, (
        "pycaret/utils/_dependencies.py still references distutils; "
        "Python 3.12 removed it. Use packaging.version.Version."
    )
    assert "from packaging.version import Version" in src, (
        "Expected 'from packaging.version import Version' in pycaret/utils/_dependencies.py"
    )


def test_get_module_version_returns_packaging_version():
    """get_module_version must return packaging.version.Version instances (not LooseVersion)."""
    from packaging.version import Version
    from pycaret.utils._dependencies import get_module_version

    # numpy is always installed; its version is queryable
    result = get_module_version("numpy")
    assert isinstance(result, Version), (
        f"Expected packaging.version.Version, got {type(result).__name__}"
    )


def test_get_installed_modules_values_are_packaging_versions():
    """get_installed_modules must return {name: Version} mapping."""
    from packaging.version import Version
    from pycaret.utils._dependencies import get_installed_modules

    modules = get_installed_modules()
    assert isinstance(modules, dict)
    # At least numpy should be present
    assert "numpy" in modules
    for name, ver in modules.items():
        if ver is None:
            continue
        assert isinstance(ver, Version), (
            f"Module {name}: expected Version, got {type(ver).__name__}"
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py::test_dependencies_module_imports_on_python312plus -v`
Expected: FAIL with "still references distutils".

- [ ] **Step 3: Rewrite `pycaret/utils/_dependencies.py`**

Read the current file to confirm exact structure, then apply these changes:

1. Change line 5 from:
```python
from distutils.version import LooseVersion
```
to:
```python
from packaging.version import Version
```

2. Change line 20 (return type annotation of first function) from:
```python
) -> Optional[Union[LooseVersion, bool]]:
```
to:
```python
) -> Optional[Union[Version, bool]]:
```

3. Change line 39 (assignment) from:
```python
        ver = LooseVersion(ver)
```
to:
```python
        ver = Version(ver)
```

4. Change line 44 (return type annotation of `get_installed_modules`) from:
```python
def get_installed_modules() -> Dict[str, Optional[LooseVersion]]:
```
to:
```python
def get_installed_modules() -> Dict[str, Optional[Version]]:
```

5. Change line 60 (assignment) from:
```python
                    ver = LooseVersion(dist.metadata["Version"])
```
to:
```python
                    ver = Version(dist.metadata["Version"])
```

6. Change line 68 (return type annotation of `_get_module_version`) from:
```python
def _get_module_version(modname: str) -> Optional[Union[LooseVersion, bool]]:
```
to:
```python
def _get_module_version(modname: str) -> Optional[Union[Version, bool]]:
```

7. Change line 79 (return type annotation of `get_module_version`) from:
```python
def get_module_version(modname: str) -> Optional[LooseVersion]:
```
to:
```python
def get_module_version(modname: str) -> Optional[Version]:
```

**Semantic differences to be aware of:**
- `packaging.version.Version` is stricter than `LooseVersion` about malformed version strings. `LooseVersion("1.2.abc")` would succeed; `Version("1.2.abc")` raises `InvalidVersion`. If this breaks a downstream caller, wrap the instantiation in a try/except and return `None` — but do NOT do this preemptively. Wait for Task 7 (full pytest run) to surface the actual regression.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-phase2/bin/python -m pytest tests/test_pandas_numpy_compat.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Verify no other file in pycaret still imports distutils**

Run:
```bash
grep -rn "distutils" pycaret/ 2>&1 | grep -v "^Binary\|_pycache_" | head
```
Expected: no hits. If hits appear, stop and report the file list — Task 4 scope may need to expand.

- [ ] **Step 6: Commit**

```bash
git add pycaret/utils/_dependencies.py tests/test_pandas_numpy_compat.py
git commit -m "fix(release): replace distutils.LooseVersion with packaging.version.Version (row 10)

Python 3.12 removed distutils entirely. pycaret/utils/_dependencies.py
was the only remaining caller — 6 usages across one import, 4 return-type
annotations, and 2 assignment sites. Replaces with packaging.version.Version
(already a transitive dep via pip, setuptools, numpy, pandas).

Note: controller pulled this fix forward from Phase 5 (original row-10
owner 'release') into Phase 2 because it blocks Python 3.12+ runtime,
which is the target CI matrix for all subsequent phases.

Closes FAILURE_TAXONOMY row 10.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Lift numpy ceiling in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:52`

- [ ] **Step 1: Inspect the current numpy pin**

Run: `grep -n "numpy" pyproject.toml`
Expected: line 52 reads `  "numpy>=1.21, <1.27",`.

- [ ] **Step 2: Edit the pin**

Replace:
```
  "numpy>=1.21, <1.27",
```
with:
```
  "numpy>=1.26,<3",
```

(Note: floor bumped to 1.26 to match the `.venv-phase2` venv and remove the soon-to-be-unsupported 1.21-1.25 range. Upper bound `<3` unlocks numpy 2.x entirely.)

- [ ] **Step 3: Verify TOML parses**

Run: `.venv-phase2/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); print([x for x in d['project']['dependencies'] if 'numpy' in x])"`
Expected: `['numpy>=1.26,<3']`.

- [ ] **Step 4: Re-resolve and verify**

Run:
```bash
VIRTUAL_ENV=$(pwd)/.venv-phase2 uv pip install -e .
.venv-phase2/bin/python -c "import numpy; print('numpy', numpy.__version__)"
```
Expected: numpy version ≥ 1.26 (or 2.x if uv picks the latest).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat(numpy): lift numpy ceiling from <1.27 to <3

Unlocks numpy 2.x. Floor raised to 1.26 to match the tested range (1.21-1.25
are no longer supported by most downstream ML deps). Pairs with the np.product
and np.NaN fixes committed earlier in this phase.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Lift pandas floor in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:53`

- [ ] **Step 1: Inspect the current pandas pin**

Run: `grep -n "pandas" pyproject.toml`
Expected: line 53 reads `  "pandas<2.2",`.

- [ ] **Step 2: Edit the pin**

Replace:
```
  "pandas<2.2",
```
with:
```
  "pandas>=2.2,<3",
```

- [ ] **Step 3: Verify TOML parses**

Run: `.venv-phase2/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); print([x for x in d['project']['dependencies'] if 'pandas' in x])"`
Expected: `['pandas>=2.2,<3']`.

- [ ] **Step 4: Re-resolve and verify**

Run:
```bash
VIRTUAL_ENV=$(pwd)/.venv-phase2 uv pip install -e .
.venv-phase2/bin/python -c "import pandas; print('pandas', pandas.__version__)"
```
Expected: pandas 2.2.x or higher.

- [ ] **Step 5: Verify preprocessor + sklearn-compat still import cleanly**

Run:
```bash
.venv-phase2/bin/python -c "
import pycaret.internal.preprocess.preprocessor
import pycaret.utils._sklearn_compat
import pycaret.containers.metrics.regression
import pycaret.utils._dependencies
print('all four modules import cleanly')
"
```
Expected: `all four modules import cleanly`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "feat(pandas): lift pandas floor to >=2.2,<3

Phase 2 core pin lift. Cherry-pick candidate PR #4009 (upstream already
moved to 2.2.x). No applymap call sites remained in pycaret source
(upstream PRs #4040 and #3927 propagated), and no copy=False usage in
astype/to_numpy — the migration is a straight pin lift from our side.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Run the full pytest suite + triage

**Files:** none modified; verification task. New FAILURE_TAXONOMY rows MAY be appended.

- [ ] **Step 1: Run the full pytest suite (excluding parity and known-Phase-4 time-series)**

Run:
```bash
.venv-phase2/bin/python -m pytest tests/ \
  --ignore=tests/parity \
  --ignore=tests/test_time_series_base.py \
  --ignore=tests/test_time_series_feat_eng.py \
  --ignore=tests/test_time_series_parallel.py \
  --ignore=tests/test_time_series_plots.py \
  --ignore=tests/test_time_series_preprocess.py \
  --ignore=tests/test_time_series_setup.py \
  --ignore=tests/test_time_series_stats.py \
  --ignore=tests/test_time_series_tune_base.py \
  --ignore=tests/test_time_series_tune_grid.py \
  --ignore=tests/test_time_series_tune_random.py \
  --ignore=tests/test_time_series_utils_plots.py \
  --ignore=tests/time_series_test_utils.py \
  --ignore=tests/test_classification_parallel.py \
  --ignore=tests/test_clustering_engines.py \
  --ignore=tests/test_persistence.py \
  --tb=short -p no:cacheprovider 2>&1 | tee /tmp/phase2-pytest.log | tail -80
```

Expected: setup, classification non-plot portion, regression, clustering non-engine portion, anomaly — all PASS. The `plot_model` step of classification is still expected to fail due to yellowbrick (row 18, Phase 3) — that is outside Phase 2 scope.

**Budget: 30 min wall-clock.** If collection + execution exceeds that, interrupt and report what completed.

- [ ] **Step 2: Triage failures**

For each FAILED test in the log:

  a. **Phase 2 regression (STOP)** if the traceback names pandas, numpy, `packaging.version`, `distutils`, `applymap`, `np.NaN`, `np.product`, or touches code we edited in T2/T3/T4.

  b. **Phase 3 (yellowbrick) residue** if the traceback names `yellowbrick.*`, `plot_model`, `matplotlib.*` — expected, document in taxonomy row 18 as "re-confirmed during Phase 2 T7" if not already there. Do NOT block.

  c. **Phase 4 time-series residue** if the traceback names `sktime.*`, `pmdarima.*`, `statsmodels.*`, `tbats.*` — expected, document in row 16 as "re-confirmed during Phase 2 T7". Do NOT block.

  d. **Phase 5 release residue** if the traceback names `joblib.*`, `distutils.*` (shouldn't — T4 killed this), py-version guards — document in row 14 or new row.

  e. **Unclassifiable:** append new row with owner `TBD-Orchestrator` and stop to ask the user for guidance.

- [ ] **Step 3: Run the parity harness — verify it still skips cleanly**

Run: `.venv-phase2/bin/python -m pytest tests/parity/ --tb=short`
Expected: 9 PASS (datasets + baseline schema unit tests), 8 SKIPPED.

- [ ] **Step 4: Run the focused smoke tests (Gate C)**

Run: `.venv-phase2/bin/python -m pytest tests/test_classification.py::test_classification -x --tb=line -p no:cacheprovider 2>&1 | tail -10`
Expected: either PASS (if Phase 3 yellowbrick somehow resolves itself) OR fail at the same `plot_model` step as end of Phase 1 (yellowbrick, row 18). If it fails EARLIER than `plot_model` (i.e., during setup / compare_models / tune), that's a Phase 2 regression — STOP.

- [ ] **Step 5: If new taxonomy rows added, commit them**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): append rows surfaced during Phase 2 pytest run"
```

- [ ] **Step 6: Push the phase branch to origin**

```bash
git push -u origin phase-2-pandas
```

---

## Task 8: Gate-D cherry-pick audit, close taxonomy rows, open PR

**Files (modify):**
- `docs/superpowers/FAILURE_TAXONOMY.md` — close rows 7, 8, 9, 10, 11.

Combines the cumulative cherry-pick audit (Phase 1's T9) and the taxonomy close-out + PR (Phase 1's T10) into a single task since Phase 2 is smaller.

- [ ] **Step 1: Cumulative cherry-pick audit**

Run:
```bash
git fetch upstream master
git checkout -B phase-2-code-pickcheck upstream/master
```

Then cherry-pick each code commit (skip docs-only taxonomy commits):
```bash
# Get only non-docs-only commits
CODE_COMMITS=$(git log --reverse --format=%H origin/modernize..phase-2-pandas | while read sha; do
  if ! git show --name-only --format= "$sha" | grep -qE "^docs/superpowers/FAILURE_TAXONOMY\.md$"; then
    echo "$sha"
  else
    # docs-only if ONLY that file changes
    if [ $(git show --name-only --format= "$sha" | grep -v "^$" | wc -l) -eq 1 ]; then
      continue
    fi
    echo "$sha"
  fi
done)
for sha in $CODE_COMMITS; do
  git cherry-pick "$sha" 2>&1 | grep -E "CONFLICT|error:|\[phase-2-code-pickcheck" | tail -1
done
echo "---final---"
git log --oneline upstream/master..HEAD | head
git checkout phase-2-pandas
git branch -D phase-2-code-pickcheck
```

Expected: each cherry-pick prints a `[phase-2-code-pickcheck <sha>]` line, none print `CONFLICT` or `error:`. Final log shows all code commits cleanly reapplied.

If any commit conflicts, STOP — bisect to the bad commit, reorder or split it, and re-test. Do NOT proceed to the PR.

- [ ] **Step 2: Close pandas-dev-tagged taxonomy rows**

Open `docs/superpowers/FAILURE_TAXONOMY.md` and update these rows:

**Row 7** (Styler.applymap): change `Status` from `open` to `closed`. Append to Notes: `· No fix needed — upstream PR #4040 (Styler.applymap→map) already propagated to our fork; no Styler.applymap sites remained in pycaret 3.4.0 source at Phase 2 start. Confirmed by grep during Phase 2 planning.`

**Row 8** (DataFrame.applymap): same `closed` flip. Append to Notes: `· No fix needed — upstream PR #3927 (DataFrame.applymap→map) already propagated; no applymap sites in pycaret 3.4.0 source.`

**Row 9** (CoW FutureWarning): same `closed` flip. Append to Notes: `· No fix needed — pycaret 3.4.0 source has no in-place df[col]=... mutation pattern and no copy=False in astype/to_numpy/reindex. A defensive-coding note was added to pandas-dev charter appendix for future contributors.`

**Row 10** (distutils.LooseVersion): change `Status` from `open` to `closed`. Append to Notes: `· Fixed by Phase 2 Task 4 (controller-extended scope, pulled forward from Phase 5 because Python 3.12+ removes distutils). Closed by <T4 SHA>.`

**Row 11** (numpy scalar/API sweep): change `Status` from `open` to `closed`. Append to Notes: `· Fixed by Phase 2 Tasks 2+3 (np.product→np.prod, np.NaN→np.nan). No np.in1d, np.trapz, np.bool8, np.float_, or np.cumproduct sites in pycaret 3.4.0 source. Closed by <T2 SHA> + <T3 SHA>.`

(Replace `<T2 SHA>`, `<T3 SHA>`, `<T4 SHA>` with the actual short SHAs from `git log`.)

- [ ] **Step 3: Add pandas-dev charter CoW defensive-coding note**

Append to `docs/superpowers/agents/pandas-dev/CHARTER.md` at the end, before the Handoff protocol:

```markdown
## Defensive-coding note for future contributors (pandas ≥ 2.2 CoW)

pycaret-ng runs on pandas 2.2+ with copy-on-write enabled by default.
Phase 2 confirmed pycaret 3.4.0 source has no CoW-incompatible patterns,
but new code must follow these rules:

1. **Do not** mutate columns in place on a DataFrame you did not create.
   Use `.assign(col=...)` or `.copy()` first.
2. **Do not** pass `copy=False` to `astype`, `reindex`, or `to_numpy`.
   It has no effect under CoW and may become an error in pandas 3.
3. **Do not** use `DataFrame.applymap` or `Styler.applymap`.
   Use `DataFrame.map` and `Styler.map` instead.

See sktime PRs #9764, #9722 and FLAML PR #1527 for the canonical
migration patterns referenced in Researcher R2.
```

- [ ] **Step 4: Commit taxonomy + charter update**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md docs/superpowers/agents/pandas-dev/CHARTER.md
git commit -m "docs(taxonomy,charter): close rows 7-11 + pandas-dev CoW guidance

Status flip + closing-commit SHAs per pandas-dev charter handoff protocol:
- Row 7 (Styler.applymap): no-fix-needed (upstream already propagated)
- Row 8 (DataFrame.applymap): no-fix-needed (upstream already propagated)
- Row 9 (CoW FutureWarning): no-fix-needed (no offending patterns in source)
- Row 10 (distutils): fixed by Phase 2 T4 (controller-extended scope)
- Row 11 (numpy scalar/API): fixed by Phase 2 T2+T3

Also appends a CoW defensive-coding note to the pandas-dev charter so
future contributors understand which patterns to avoid.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Push**

```bash
git push origin phase-2-pandas
```

- [ ] **Step 6: Open PR to `modernize`**

Run:
```bash
gh pr create --repo olaTechie/pycaret --base modernize --head phase-2-pandas \
  --title "Phase 2: pandas >= 2.2 + numpy >= 2.0 migration" \
  --body "$(cat <<'EOF'
## Summary

- Lifts pandas floor from `<2.2` to `>=2.2,<3` (upstream PR #4009 parity).
- Lifts numpy ceiling from `<1.27` to `<3`; floor raised to 1.26 to match ecosystem support.
- Fixes two numpy 2.0 API renames (`np.product`→`np.prod`, `np.NaN`→`np.nan`) — only sites in pycaret 3.4.0.
- Replaces `distutils.LooseVersion` with `packaging.version.Version` in `pycaret/utils/_dependencies.py` (controller-extended scope, row 10 pulled forward from Phase 5 because Python 3.12+ removes distutils).
- Closes FAILURE_TAXONOMY rows 7, 8, 9, 10, 11. Rows 7, 8, 9 closed with no-fix-needed status — upstream PRs #4040 and #3927 already propagated, no CoW-incompatible patterns in source. Row 10 closed by T4, row 11 by T2+T3.

## Gates

- **A (suite green):** Phase 2 work clean. Known scope-boundary residues: `plot_model` (yellowbrick, row 18, Phase 3) and 12 time-series test modules (pmdarima, row 16, Phase 4) remain blocked; same state as end of Phase 1.
- **B (parity):** partial (baselines deferred from Phase 0 Task 12).
- **C (smoke):** pre-plot portion of `test_classification.py` passes.
- **D (cherry-pick-clean):** all code commits verified in T8 cumulative audit.

## Plan + spec references

- Plan: `docs/superpowers/plans/2026-04-16-phase-2-pandas-numpy.md`
- Spec: `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` (Phase 2 section)
- Charter: `docs/superpowers/agents/pandas-dev/CHARTER.md` (now includes CoW defensive-coding note)
- Researcher findings: R2 + R3 in `docs/superpowers/agents/researcher/FINDINGS.md`

## Scope extensions beyond original plan

One documented deviation: distutils→packaging fix was pulled forward from Phase 5 (original owner: `release`) into Phase 2 because Python 3.12 removed distutils entirely. Without this fix, Phase 2 tests cannot run on the CI matrix.

## Out of scope (handed off)

- yellowbrick classifier detection (row 18) → Phase 3
- pmdarima `force_all_finite` (row 16) → Phase 4
- `set_output(transform="pandas")` adoption → deferred indefinitely (YAGNI)

## Test plan

- [ ] Reviewer confirms `git log --oneline origin/modernize..phase-2-pandas` shows 7-8 commits
- [ ] Fresh venv with `pandas>=2.2` and `numpy>=1.26`: `pytest tests/test_pandas_numpy_compat.py` should show 6 PASS
- [ ] `grep "pandas-dev.*open" docs/superpowers/FAILURE_TAXONOMY.md` should return no rows
- [ ] `grep distutils pycaret/` should return no hits

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 7: Notify Orchestrator (the human) that Phase 2 is ready for QA review**

Report:
> Phase 2 PR opened: <URL>. 7-8 commits, all code commits cherry-pick-clean onto upstream/master. After QA sign-off + merge, Phase 3 (plotting) plan can start via `writing-plans`.
