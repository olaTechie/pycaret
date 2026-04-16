# Phase 3 — Plotting Stack Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift pycaret-ng's matplotlib cap from `<3.8.0` to `>=3.8`, update schemdraw from `==0.15` to `>=0.16`, and handle yellowbrick 1.5's two known incompatibilities (distutils on Python 3.12+, `use_line_collection` on matplotlib ≥ 3.8) so the plotting stack works on the modernized dep set.

**Architecture:** yellowbrick 1.5 is the latest release and is effectively unmaintained — no fix is coming upstream. Two specific breakages: (a) internal `from distutils.version import LooseVersion` (Python 3.12 removed distutils) — solved by ensuring `setuptools` is installed as a runtime dep (provides the distutils compatibility shim); (b) internal `stem(..., use_line_collection=True)` in `yellowbrick/regressor/influence.py:184` (matplotlib 3.8 removed this kwarg) — solved by monkey-patching `matplotlib.axes.Axes.stem` to silently drop the `use_line_collection` kwarg when present, via a small compat module that loads before yellowbrick.

**Tech Stack:** Python 3.12, matplotlib ≥ 3.8, yellowbrick 1.5, schemdraw ≥ 0.16, pytest.

**Spec reference:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` (Phase 3 section).
**Charter:** `docs/superpowers/agents/plotting-dev/CHARTER.md`.
**Taxonomy:** Row 18 (yellowbrick classifier detection). Gate B (parity) is **WAIVED per spec** — visual output is not numerically comparable.
**Prerequisite:** Phase 2 PR `#2` merged into `modernize`.

---

## Decisions locked in this plan

- **matplotlib floor:** `>=3.8` (no upper cap). The old `<3.8.0` pin existed solely because yellowbrick 1.5 uses `stem(use_line_collection=True)` which matplotlib 3.8 removed. We monkey-patch around it.
- **schemdraw:** `>=0.16` (drop the `==0.15` pin). schemdraw 0.22 works on matplotlib ≥ 3.8 and Python 3.12+. The old pin comment ("0.16 only supports Python >3.8") is exactly our target, so the pin is no longer needed.
- **yellowbrick:** stay at `>=1.4` (no version bump available — 1.5 is the latest). Fix both breakages via shim/patch in pycaret, not by upgrading.
- **plotly-resampler, kaleido, mljar-scikit-plot:** no changes needed — their current pins (`>=0.8.3.1`, `>=0.2.1`, unpinned respectively) work on the modernized stack.
- **setuptools:** already in pyproject as `"setuptools; python_version>='3.12'"`. Phase 3 verifies this is sufficient for yellowbrick's distutils shim and adds a code comment documenting the dependency chain.

## Gate state for Phase 3

- **Gate A:** tests pass. Specifically `tests/test_classification_plots.py` and `tests/test_regression_plots.py`.
- **Gate B:** **WAIVED** per spec (visual output not numerically comparable).
- **Gate C:** `plot_model(lr, save=True)` runs without error for at least classification and regression.
- **Gate D:** cherry-pick-clean onto `upstream/master`.

---

## File Structure

**Created in Phase 3:**

| Path | Responsibility |
|------|----------------|
| `pycaret/internal/patches/matplotlib_compat.py` | Monkey-patch `Axes.stem` to silently drop `use_line_collection` kwarg. Loaded before any yellowbrick import. |
| `tests/test_plotting_compat.py` | Unit tests for the matplotlib stem patch + schemdraw import. |

**Modified in Phase 3:**

| Path | Change |
|------|--------|
| `pycaret/internal/patches/__init__.py` | Import `matplotlib_compat` at module load to apply the stem patch early. |
| `pyproject.toml:75` | `matplotlib<3.8.0` → `matplotlib>=3.8`. |
| `pyproject.toml:80` | `schemdraw==0.15` → `schemdraw>=0.16`. |
| `docs/superpowers/FAILURE_TAXONOMY.md` row 18 | Status `open` → `closed`. |

**Branch:** `phase-3-plotting` off `modernize` (after Phase 2 PR merges).

---

## Task 1: Branch off `modernize` and prepare venv

**Files:** none yet.

- [ ] **Step 1: Fetch + create branch**

```bash
git fetch origin modernize upstream master
git checkout -B phase-3-plotting origin/modernize
```

- [ ] **Step 2: Create the Phase 3 venv**

```bash
uv venv --python 3.12 .venv-phase3
VIRTUAL_ENV=$(pwd)/.venv-phase3 uv pip install -e . pytest setuptools \
  "matplotlib>=3.8" "schemdraw>=0.16" "yellowbrick>=1.4" "mljar-scikit-plot" \
  "joblib>=1.4.2,<1.5"
```

Note: explicitly pin `joblib<1.5` because pyproject has that cap and joblib 1.5+ breaks `pycaret/internal/memory.py` (row 14, Phase 5 scope). Also explicitly install `setuptools` (yellowbrick needs it on Python 3.12).

- [ ] **Step 3: Verify versions**

```bash
.venv-phase3/bin/python -c "
import matplotlib, schemdraw, yellowbrick
print('matplotlib', matplotlib.__version__)
print('schemdraw', schemdraw.__version__)
print('yellowbrick', yellowbrick.__version__)
"
```
Expected: matplotlib ≥ 3.8, schemdraw ≥ 0.16, yellowbrick 1.5.

- [ ] **Step 4: Add `.venv-phase3/` to .gitignore (idempotent) + commit if changed**

```bash
grep -qxF '.venv-phase3/' .gitignore || { echo '.venv-phase3/' >> .gitignore; git add .gitignore; git commit -m "chore(ci): ignore .venv-phase3 working venv"; }
```

---

## Task 2: TDD — matplotlib stem monkey-patch for yellowbrick compat

**Files:**
- Create: `pycaret/internal/patches/matplotlib_compat.py`
- Create: `tests/test_plotting_compat.py`
- Modify: `pycaret/internal/patches/__init__.py`

yellowbrick 1.5's `regressor/influence.py:184` calls `ax.stem(..., use_line_collection=True)`. matplotlib 3.8 removed the `use_line_collection` parameter from `Axes.stem`. Our patch wraps `Axes.stem` to silently drop the kwarg.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plotting_compat.py`:

```python
"""Unit tests for pycaret-ng Phase 3 plotting compatibility patches."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_stem_accepts_use_line_collection_without_error():
    """After the patch, ax.stem() must silently accept use_line_collection.

    yellowbrick 1.5 passes this kwarg; matplotlib 3.8+ removed it.
    The patch wraps Axes.stem to drop it silently.
    """
    import pycaret.internal.patches.matplotlib_compat  # applies the patch

    fig, ax = plt.subplots()
    # This would raise TypeError on matplotlib >=3.8 without the patch
    ax.stem([1, 2, 3], [4, 5, 6], use_line_collection=True)
    plt.close(fig)


def test_stem_still_works_without_use_line_collection():
    """Normal stem() calls must not be affected by the patch."""
    import pycaret.internal.patches.matplotlib_compat  # applies the patch

    fig, ax = plt.subplots()
    ax.stem([1, 2, 3], [4, 5, 6])
    plt.close(fig)


def test_schemdraw_imports_on_modern_matplotlib():
    """schemdraw >=0.16 must import cleanly on matplotlib >=3.8."""
    from schemdraw import Drawing
    from schemdraw.flow import Arrow, Data, RoundBox, Subroutine
    assert Drawing is not None
```

- [ ] **Step 2: Run to verify first test fails**

Run: `.venv-phase3/bin/python -m pytest tests/test_plotting_compat.py::test_stem_accepts_use_line_collection_without_error -v --noconftest`
Expected: FAIL with `TypeError: stem() got an unexpected keyword argument 'use_line_collection'` (because the patch module doesn't exist yet).

Actually it may fail with `ModuleNotFoundError` first. Either failure mode confirms RED.

- [ ] **Step 3: Implement the stem monkey-patch**

Create `pycaret/internal/patches/matplotlib_compat.py`:

```python
"""Matplotlib compatibility patches for pycaret-ng.

yellowbrick 1.5 (latest release, effectively unmaintained) calls
matplotlib's Axes.stem() with use_line_collection=True, which matplotlib
3.8 removed. This module monkey-patches Axes.stem to silently drop the
kwarg, keeping yellowbrick's CooksDistance and other stem-based plots
functional on modern matplotlib.

Loaded early via pycaret/internal/patches/__init__.py.
"""
from __future__ import annotations

import matplotlib.axes

_original_stem = matplotlib.axes.Axes.stem


def _patched_stem(self, *args, **kwargs):
    kwargs.pop("use_line_collection", None)
    return _original_stem(self, *args, **kwargs)


matplotlib.axes.Axes.stem = _patched_stem
```

- [ ] **Step 4: Ensure the patch loads early**

Read `pycaret/internal/patches/__init__.py`. Add the import at the top (after any existing imports):

```python
import pycaret.internal.patches.matplotlib_compat  # noqa: F401
```

If the file doesn't exist or is empty, create it with just that line.

- [ ] **Step 5: Run tests to verify GREEN**

Run: `.venv-phase3/bin/python -m pytest tests/test_plotting_compat.py -v --noconftest`
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pycaret/internal/patches/matplotlib_compat.py pycaret/internal/patches/__init__.py tests/test_plotting_compat.py
git commit -m "fix(plot): monkey-patch Axes.stem for yellowbrick 1.5 compat on matplotlib >=3.8

yellowbrick 1.5 (latest release, unmaintained) passes use_line_collection
to stem(), which matplotlib 3.8 removed. Patch drops the kwarg silently.
Loaded early via pycaret/internal/patches/__init__.py.

Partial closure of FAILURE_TAXONOMY row 18 (yellowbrick plotting).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Lift matplotlib and schemdraw pins in pyproject.toml

**Files:**
- Modify: `pyproject.toml:75` (matplotlib pin)
- Modify: `pyproject.toml:80` (schemdraw pin)

- [ ] **Step 1: Edit matplotlib pin**

Replace:
```
  "matplotlib<3.8.0",  # stem(..., use_line_collection=False) is no longer supported.
```
with:
```
  "matplotlib>=3.8",
```

- [ ] **Step 2: Edit schemdraw pin**

Replace:
```
  "schemdraw==0.15",  # 0.16 only supports Python >3.8
```
with:
```
  "schemdraw>=0.16",
```

- [ ] **Step 3: Verify TOML parses**

Run: `.venv-phase3/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); deps = d['project']['dependencies']; print([x for x in deps if 'matplotlib' in x or 'schemdraw' in x])"`
Expected: `['matplotlib>=3.8', 'schemdraw>=0.16']`.

- [ ] **Step 4: Re-resolve venv and verify**

```bash
VIRTUAL_ENV=$(pwd)/.venv-phase3 uv pip install -e . "joblib>=1.4.2,<1.5"
.venv-phase3/bin/python -c "import matplotlib, schemdraw; print('mpl', matplotlib.__version__, 'schemdraw', schemdraw.__version__)"
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat(plot): lift matplotlib to >=3.8, schemdraw to >=0.16

Drops the matplotlib<3.8 cap (was held by yellowbrick's use_line_collection
usage, now monkey-patched in matplotlib_compat.py). Drops schemdraw==0.15
pin (0.16+ supports Python >3.8, which is our target).

plotly-resampler, kaleido, and mljar-scikit-plot pins unchanged (already
compatible with the modern stack).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Smoke-test the yellowbrick classifier detection (row 18)

**Files:**
- Modify: `tests/test_plotting_compat.py` (append test)

The Phase 1 T8 yellowbrick `YellowbrickTypeError` was triggered by `plot_model` because yellowbrick's classifier validators don't recognize pycaret's Pipeline-wrapped estimator. pycaret already has a `pycaret/internal/patches/yellowbrick.py` that wraps `get_model_name` to unwrap meta-estimators. The classifier detection issue may already be fixed by the stem patch + setuptools (which were missing during Phase 1 T8).

- [ ] **Step 1: Append a smoke test**

Append to `tests/test_plotting_compat.py`:

```python
def test_yellowbrick_classifier_detection_with_pycaret_pipeline():
    """yellowbrick must recognise a pycaret pipeline-wrapped classifier.

    Phase 1 T8 saw YellowbrickTypeError because the estimator type wasn't
    detected through pycaret's Pipeline wrapper. This test verifies the
    existing yellowbrick patch + matplotlib_compat patch together handle it.
    """
    pytest.importorskip("pycaret.classification")
    from pycaret.datasets import get_data
    from pycaret.classification import ClassificationExperiment

    exp = ClassificationExperiment()
    df = get_data("juice", verbose=False)
    exp.setup(data=df, target="Purchase", session_id=42, verbose=False, html=False)
    lr = exp.create_model("lr", verbose=False)
    # plot_model with save=True avoids opening a GUI window
    exp.plot_model(lr, save=True, scale=5)
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv-phase3/bin/python -m pytest tests/test_plotting_compat.py::test_yellowbrick_classifier_detection_with_pycaret_pipeline -v --noconftest`

Expected outcomes:
- **PASS:** the classifier detection issue was caused by missing setuptools + matplotlib compat; both are now fixed. Mark row 18 as closed.
- **FAIL with YellowbrickTypeError:** the issue persists. Report BLOCKED — the yellowbrick classifier detection needs a deeper patch (likely extending `pycaret/internal/patches/yellowbrick.py`'s `is_estimator` function to set `_estimator_type` on the wrapper). The plan can add a Task 4b to fix it.
- **FAIL with `bytes_limit` TypeError:** the venv has wrong joblib version. Fix by re-pinning `joblib<1.5`.

- [ ] **Step 3: If PASS, commit the test**

```bash
git add tests/test_plotting_compat.py
git commit -m "test(plot): smoke test for yellowbrick classifier detection (row 18)

Verifies that pycaret's pipeline-wrapped estimator is recognised as a
classifier by yellowbrick's visualization classes after the matplotlib
stem patch and setuptools-backed distutils shim are in place.

Closes FAILURE_TAXONOMY row 18.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

If FAIL, do NOT commit — report BLOCKED with the error and ask for guidance.

---

## Task 5: Full pytest smoke (plot-related tests) + triage

**Files:** none modified; verification task.

- [ ] **Step 1: Run plot-related test files**

```bash
.venv-phase3/bin/python -m pytest tests/test_classification_plots.py tests/test_regression_plots.py -v --tb=short --noconftest -p no:cacheprovider 2>&1 | tee /tmp/phase3-pytest.log | tail -30
```

If conftest loads cleanly (setuptools distutils shim works), drop `--noconftest` and re-run.

Expected: most plot tests PASS. Some may fail due to Phase 4 (time-series deps) or Phase 5 (joblib) residues — triage those as existing taxonomy rows.

- [ ] **Step 2: Triage any failures**

For each FAILED test:
  a. **Phase 3 regression:** traceback names matplotlib, yellowbrick, schemdraw, `use_line_collection`, or code edited in T2/T3. **STOP** and fix.
  b. **Phase 4/5 residue:** traceback names sktime, pmdarima, joblib, or distutils. Append to taxonomy if not already present.
  c. **Unclassifiable:** report with `TBD-Orchestrator`.

- [ ] **Step 3: Commit any new taxonomy rows**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): append rows surfaced during Phase 3 plotting pytest"
```

Skip if no new rows.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin phase-3-plotting
```

---

## Task 6: Gate-D audit + taxonomy close + PR

**Files (modify):**
- `docs/superpowers/FAILURE_TAXONOMY.md` — close row 18.

- [ ] **Step 1: Cumulative cherry-pick audit (code commits only)**

```bash
git fetch upstream master
git checkout -B phase-3-code-pickcheck upstream/master
for sha in $(git log --reverse --format=%H origin/modernize..phase-3-plotting | while read s; do
  FILES=$(git show --name-only --format= "$s" | grep -v "^$")
  echo "$FILES" | grep -qE "\.py$|pyproject\.toml$|\.gitignore$" && echo "$s"
done); do
  git cherry-pick "$sha" 2>&1 | grep -E "CONFLICT|error:|\[phase-3" | tail -1
done
echo "---final---"
git log --oneline upstream/master..HEAD
git checkout phase-3-plotting
git branch -D phase-3-code-pickcheck
```

Expected: all code commits cherry-pick cleanly.

- [ ] **Step 2: Close taxonomy row 18**

In `docs/superpowers/FAILURE_TAXONOMY.md`, change row 18:
- Status: `open` → `closed`
- Append to Notes: `· Fixed by Phase 3: matplotlib_compat.py stem patch (removes use_line_collection kwarg), setuptools distutils shim already in pyproject, classifier detection confirmed working via smoke test. Closed by <T2 SHA> + <T4 SHA>.`

- [ ] **Step 3: Commit + push**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): close row 18 (yellowbrick plotting, Phase 3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push origin phase-3-plotting
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --repo olaTechie/pycaret --base modernize --head phase-3-plotting \
  --title "Phase 3: plotting stack migration (matplotlib >= 3.8)" \
  --body "$(cat <<'EOF'
## Summary

- Lifts matplotlib from `<3.8.0` to `>=3.8`, schemdraw from `==0.15` to `>=0.16`.
- Adds `pycaret/internal/patches/matplotlib_compat.py` monkey-patch for yellowbrick 1.5's deprecated `stem(use_line_collection=True)` call.
- yellowbrick 1.5 (latest, unmaintained) works via: (a) setuptools distutils shim (already in pyproject for py3.12+), (b) stem kwarg patch.
- Closes FAILURE_TAXONOMY row 18. Gate B (parity) waived per spec.

## Gates

- **A:** Plot-related tests pass on matplotlib ≥ 3.8. Known residues: Phase 4 (pmdarima) and Phase 5 (joblib).
- **B:** Waived (visual output not numerically comparable).
- **C:** `plot_model(lr, save=True)` runs cleanly for classification.
- **D:** Code commits cherry-pick-clean onto upstream/master.

## Test plan

- [ ] `pytest tests/test_plotting_compat.py --noconftest` → 4 PASS
- [ ] `grep "matplotlib" pyproject.toml` → `matplotlib>=3.8`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.
