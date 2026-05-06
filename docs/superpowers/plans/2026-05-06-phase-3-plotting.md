# Phase 3 Plotting Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the pycaret plotting stack on `phase-3-plotting` to run cleanly under matplotlib≥3.8, yellowbrick≥1.5, schemdraw≥0.16, plotly-resampler latest, and mljar-scikit-plot latest, while preserving cherry-pick discipline (Gate D) and shipping a tiny local smoke harness.

**Architecture:** Build the smoke harness first so subsequent fixes have a verification target. Then close taxonomy row 18 by extending `pycaret/internal/patches/yellowbrick.py` with classifier/regressor detection shims. Then sweep matplotlib 3.8 / pandas-Styler / numpy-2 deprecations. Lift `plotly-resampler` and audit `mljar-scikit-plot`. Disable any visualizer that resists fixing under the spec's fix-or-disable fallback policy with a clear `NotImplementedError` and a `DEGRADED.md` row.

**Tech Stack:** Python 3.11+, pytest, matplotlib (Agg backend), yellowbrick, plotly, plotly-resampler, mljar-scikit-plot, schemdraw, pycaret 3.4.0 codebase.

**Spec:** `docs/superpowers/specs/2026-05-06-phase-3-plotting-design.md`

**Branch:** `phase-3-plotting` (off `modernize`)

**Verification floor (local):** `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` green or skips-only. Aggregate <90 s, per-plot <30 s. **Do not run the full `pytest tests/` locally** — it hangs the user's machine. Gate A is enforced by CI on PR push.

**Working venv:** `.venv-phase3` (already exists at repo root, Python 3.12.13).

**Important — invocation pattern.** Do NOT use `source .venv-phase3/bin/activate`. The activate script's PATH was baked in at venv-creation time using a Dropbox CloudStorage path alias (`00Todo/00_ToReview` vs `00_ToReview`) that no longer resolves, so after activate `which python` falls back to anaconda's 3.13 (which violates pycaret's 3.9–3.12 guard). Always invoke the binary directly:

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/...
```

The `--confcutdir=tests/smoke` flag is required because the root `tests/conftest.py` eagerly imports `pycaret`, which transitively imports `sktime`. Sktime is not installed yet (Phase 4 owns its modernization). `--confcutdir` tells pytest to ignore conftests above `tests/smoke/`, so only the smoke harness's lightweight conftest loads.

Canonical smoke invocation used throughout this plan:

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v
```

---

## File Structure

**Create:**
- `tests/smoke/__init__.py` — empty package marker
- `tests/smoke/conftest.py` — sets matplotlib `Agg` backend, suppresses plotly browser open, sets up cached `setup()` fixtures per task type
- `tests/smoke/test_plotting.py` — parametrized smoke tests for classification, regression, clustering plots
- `docs/superpowers/agents/plotting-dev/CHARTER.md` — agent charter (mirrors other agents under `docs/superpowers/agents/`)
- `docs/superpowers/agents/plotting-dev/LOG.md` — append-only progress log
- `docs/superpowers/agents/plotting-dev/DEGRADED.md` — degraded-visualizer registry (initially empty schema)

**Modify:**
- `pycaret/internal/patches/yellowbrick.py` — add `is_classifier`, `is_regressor` shims that unwrap pycaret's pipeline before delegating to yellowbrick's originals
- `pycaret/internal/pycaret_experiment/tabular_experiment.py:526–538` — extend the `mock.patch.object` block to install the new shims
- `pyproject.toml` — lift `plotly-resampler` floor (W3); possibly lift `mljar-scikit-plot` floor (W4); possibly lift `yellowbrick` floor if W1 forces it
- `pycaret/internal/plots/helper.py` and any other matched files — fix matplotlib 3.8 deprecations surfaced by W2 sweep
- `docs/superpowers/FAILURE_TAXONOMY.md` — close row 18, append rows 20+ for any new empirical findings
- `docs/superpowers/MIGRATION_BACKLOG.md` — refresh row counts at Phase 3 close

**Reference (read, don't modify unless required):**
- `pycaret/classification/oop.py:64–86` — classification `_available_plots` dict (21 keys)
- `pycaret/regression/oop.py:50–64` — regression `_available_plots` dict (13 keys)
- `pycaret/clustering/oop.py:26–34` — clustering `_available_plots` dict (7 keys)
- `pycaret/internal/pycaret_experiment/tabular_experiment.py:1046+` — yellowbrick dispatch sites
- `pycaret/internal/plots/yellowbrick.py` — `show_yellowbrick_plot`

---

## Task 1: DEGRADED.md scaffold

**Files:**
- Create: `docs/superpowers/agents/plotting-dev/DEGRADED.md`
- Create: `docs/superpowers/agents/plotting-dev/CHARTER.md`
- Create: `docs/superpowers/agents/plotting-dev/LOG.md`

- [ ] **Step 1: Write `DEGRADED.md` schema-only**

```markdown
# DEGRADED.md — Plotting Visualizer Registry

Visualizers that pycaret-ng explicitly disables under modernized deps. A
disabled visualizer raises `NotImplementedError` from its dispatch site
in `tabular_experiment.py` or `clustering/oop.py`. The corresponding
smoke entry in `tests/smoke/test_plotting.py` is skip-marked.

## Schema

| Plot key | Task | Disabled because | Tracking | Restoration criterion |
|----------|------|-------------------|----------|------------------------|

## Rows

(none — populate as Phase 3 fixes uncover unfixable visualizers)
```

- [ ] **Step 2: Write `CHARTER.md` (concise, ~25 lines)**

```markdown
# Plotting Migration Dev — Charter

**Phase:** 3 (Plotting Stack)
**Branch:** `phase-3-plotting` (off `modernize`)
**Spec:** `docs/superpowers/specs/2026-05-06-phase-3-plotting-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-phase-3-plotting.md`

## Inputs
- `FAILURE_TAXONOMY.md` rows tagged `matplotlib | yellowbrick | schemdraw | plotly | plotly-resampler | mljar-scikit-plot` (currently row 18; expect 20+ on empirical sweep).
- Master spec § 4 Phase 3 paragraph.

## Outputs
- Cherry-pickable commits on `phase-3-plotting` (one logical change per commit, conventional message).
- New `tests/smoke/test_plotting.py` (pycaret-ng infra, exempt from Gate D).
- Updated `FAILURE_TAXONOMY.md` and `MIGRATION_BACKLOG.md`.
- `DEGRADED.md` rows for any visualizer disabled under fallback policy (b).

## Stop criteria
- All in-scope taxonomy rows closed or moved to DEGRADED.md.
- Smoke harness green locally (skips-only ok for degraded entries).
- PR open from `phase-3-plotting → modernize` with CI green.

## Out-of-scope handoffs
- Time-series plot failures → tag `ts-dev`, defer to Phase 4.
- Test-infra failures (missing soft deps) → tag `release`, defer to Phase 5.
- Plot dispatch refactoring → not in Phase 3.

## Authority
- May add taxonomy rows. May not edit closed rows owned by other agents.
- May edit `pyproject.toml` for plotting-stack dep floors only.
```

- [ ] **Step 3: Write `LOG.md` (header only)**

```markdown
# Plotting Migration Dev — Log

Append-only progress log.

## 2026-05-06 — Phase 3 kickoff
- Plan committed: `docs/superpowers/plans/2026-05-06-phase-3-plotting.md`.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/agents/plotting-dev/
git commit -m "$(cat <<'EOF'
docs(plotting-dev): scaffold agent charter, log, DEGRADED registry

Mirrors the existing per-agent doc layout under docs/superpowers/agents/.
DEGRADED.md is empty (schema only); rows populated as Phase 3 fixes
encounter unfixable visualizers under fallback policy (b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Smoke harness — package skeleton

**Files:**
- Create: `tests/smoke/__init__.py`
- Create: `tests/smoke/conftest.py`

- [ ] **Step 1: Create `tests/smoke/__init__.py`**

Empty file. One blank line content is fine; `pytest` doesn't strictly need it under modern conftest discovery, but include it so the directory is unambiguously a package.

```python
```

- [ ] **Step 2: Create `tests/smoke/conftest.py`**

```python
"""Shared fixtures and backend setup for the Phase 3 plotting smoke harness.

This conftest forces a non-interactive matplotlib backend BEFORE any
pycaret import in the test module, so headless CI/local runs don't try
to open a display.
"""
import os

# Headless matplotlib — must run before pycaret pulls matplotlib in.
import matplotlib

matplotlib.use("Agg")

# Suppress plotly's browser-open on `fig.show()`.
os.environ.setdefault("PLOTLY_RENDERER", "json")

import pytest
```

- [ ] **Step 3: Smoke-import to make sure conftest loads**

Run:

```bash
python -c "import pytest; pytest.main(['--collect-only', 'tests/smoke/'])"
```

Expected: pytest collects 0 tests but exits cleanly (no errors). If `matplotlib.use('Agg')` errors, fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/__init__.py tests/smoke/conftest.py
git commit -m "$(cat <<'EOF'
test(smoke): scaffold tests/smoke/ package with Agg-backend conftest

Phase 3 introduces a tiny local smoke harness for plotting (separate
from tests/ proper, not pulled into the default pytest target). The
conftest forces matplotlib's Agg backend before any pycaret import so
the harness runs headless on dev laptops and CI alike.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Smoke harness — classification path (failing first)

**Files:**
- Create: `tests/smoke/test_plotting.py`

- [ ] **Step 1: Write the classification smoke test**

```python
"""Phase 3 plotting smoke harness.

Minimal local-only coverage: one estimator per task type, parametrized
over each `experiment._available_plots.keys()`. Pass condition is "no
exception raised" — no image diff, no SHAP, no interpret_model.

Per-plot timeout 30 s, aggregate target <90 s on a developer laptop.

This file is NOT discovered by `pytest tests/` (it lives under
`tests/smoke/`). Run explicitly:

    .venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v
"""
from __future__ import annotations

import pytest

from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

# --- Fixtures ----------------------------------------------------------

@pytest.fixture(scope="module")
def clf_setup():
    data = get_data("iris", verbose=False)
    exp = ClassificationExperiment()
    exp.setup(
        data=data,
        target="species",
        session_id=42,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("rf", n_estimators=5, max_depth=2, verbose=False)
    return exp, model


# --- Classification plots ----------------------------------------------

# Skip-list is empty initially; entries added as DEGRADED.md grows or as
# we identify visualizers that are pathologically slow on iris.
CLF_DEGRADED: set[str] = set()

CLF_PLOTS = sorted(
    [
        "pipeline", "parameter", "auc", "confusion_matrix", "threshold",
        "pr", "error", "class_report", "rfe", "learning", "manifold",
        "calibration", "vc", "dimension", "feature", "feature_all",
        "boundary", "lift", "gain", "tree", "ks",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", CLF_PLOTS)
def test_classification_plot(clf_setup, plot, tmp_path):
    if plot in CLF_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = clf_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)
```

- [ ] **Step 2: Confirm `pytest-timeout` is available; if not, install it locally for the dev venv only**

Run:

```bash
python -c "import pytest_timeout" || pip install pytest-timeout
```

Note: do NOT add `pytest-timeout` to `pyproject.toml`. It is a developer-side convenience, not a runtime dep. If you prefer not to install it, replace the `@pytest.mark.timeout(30)` decorator with a manual `signal.alarm(30)` guard in a fixture (see Step 3 fallback).

- [ ] **Step 3 (optional, fallback if pytest-timeout unavailable): Replace the timeout marker with a signal-based guard**

Skip if Step 2 succeeded. Otherwise, in `tests/smoke/conftest.py`, add:

```python
import signal


@pytest.fixture(autouse=True)
def _per_test_timeout():
    def _handler(signum, frame):
        raise TimeoutError("smoke per-plot timeout (30s)")
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(30)
    yield
    signal.alarm(0)
```

And remove the `@pytest.mark.timeout(30)` decorator from `tests/smoke/test_plotting.py`. Note: signal-based timeouts only work on POSIX; Windows users would need `pytest-timeout`.

- [ ] **Step 4: Run only the classification path; observe failures**

Run:

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py::test_classification_plot -v --tb=short -x
```

Expected: at least the yellowbrick-backed plots (`auc`, `pr`, `error`, `class_report`, `confusion_matrix`, `threshold`, `learning`, `vc`, `manifold`, `feature`, `feature_all`, `dimension`, `boundary`, `calibration`, `rfe`) FAIL with the row-18 signature (`yellowbrick.exceptions.YellowbrickTypeError: This estimator is not a classifier`) or similar. Some plots (`pipeline`, `parameter`, `tree`, `lift`, `gain`, `ks`) may pass since they don't go through yellowbrick.

Capture the exact failure signatures into a temporary note for Task 5 (taxonomy refresh).

- [ ] **Step 5: Commit**

```bash
git add tests/smoke/test_plotting.py
git commit -m "$(cat <<'EOF'
test(smoke): plotting harness — classification path (parametrized)

Iterates ClassificationExperiment._available_plots on iris with a tiny
random-forest. Per-plot timeout 30 s; aggregate target <90 s. CLF_DEGRADED
is the runtime skip-list mirroring DEGRADED.md.

This commit is expected to leave failing tests on phase-3-plotting HEAD
until the yellowbrick patch in the next commit lands. The harness is
pycaret-ng-only infra, exempt from Gate D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Smoke harness — regression path

**Files:**
- Modify: `tests/smoke/test_plotting.py`

- [ ] **Step 1: Append regression fixture and parametrized test**

Add to the bottom of `tests/smoke/test_plotting.py`:

```python
# --- Regression --------------------------------------------------------

@pytest.fixture(scope="module")
def reg_setup():
    data = get_data("diabetes", verbose=False)
    # Diabetes from pycaret.datasets has a numeric target column;
    # confirm by inspecting columns and pick the appropriate one.
    target = "Class variable" if "Class variable" in data.columns else data.columns[-1]
    exp = RegressionExperiment()
    exp.setup(
        data=data,
        target=target,
        session_id=42,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("rf", n_estimators=5, max_depth=2, verbose=False)
    return exp, model


REG_DEGRADED: set[str] = set()

REG_PLOTS = sorted(
    [
        "pipeline", "parameter", "residuals", "error", "cooks", "rfe",
        "learning", "manifold", "vc", "feature", "feature_all", "tree",
        "residuals_interactive",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", REG_PLOTS)
def test_regression_plot(reg_setup, plot, tmp_path):
    if plot in REG_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = reg_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)
```

- [ ] **Step 2: Confirm the diabetes target name**

Run:

```bash
python -c "from pycaret.datasets import get_data; d = get_data('diabetes', verbose=False); print(list(d.columns))"
```

Expected: a list of column names. The fixture's `target = "Class variable" if ...` line falls back to `data.columns[-1]`, which is the standard pycaret convention. If the dataset's actual target is named differently, edit the fixture's literal accordingly.

- [ ] **Step 3: Run regression smoke**

Run:

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py::test_regression_plot -v --tb=short
```

Expected: yellowbrick-backed regression plots (`residuals`, `error`, `cooks`, `learning`, `manifold`, `vc`, `feature`, `feature_all`, `rfe`) likely fail with the same row-18 signature.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_plotting.py
git commit -m "$(cat <<'EOF'
test(smoke): plotting harness — regression path

Iterates RegressionExperiment._available_plots on diabetes with a tiny
random-forest. Same skip-list / timeout discipline as the classification
path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Smoke harness — clustering path

**Files:**
- Modify: `tests/smoke/test_plotting.py`

- [ ] **Step 1: Append clustering fixture and parametrized test**

Add to the bottom of `tests/smoke/test_plotting.py`:

```python
# --- Clustering --------------------------------------------------------

@pytest.fixture(scope="module")
def clu_setup():
    data = get_data("iris", verbose=False).drop(columns=["species"])
    exp = ClusteringExperiment()
    exp.setup(
        data=data,
        session_id=42,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("kmeans", num_clusters=3, verbose=False)
    return exp, model


CLU_DEGRADED: set[str] = set()

CLU_PLOTS = sorted(
    [
        "pipeline", "cluster", "tsne", "elbow", "silhouette", "distance",
        "distribution",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", CLU_PLOTS)
def test_clustering_plot(clu_setup, plot, tmp_path):
    if plot in CLU_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = clu_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)
```

- [ ] **Step 2: Run clustering smoke**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py::test_clustering_plot -v --tb=short
```

Expected: yellowbrick-backed `elbow` and `silhouette` likely fail with the row-18 signature; the rest may pass.

- [ ] **Step 3: Run the full smoke once to capture full failure landscape**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --tb=short --no-header 2>&1 | tee /tmp/phase3-smoke-baseline.txt
```

The `/tmp/phase3-smoke-baseline.txt` file is the input for Task 6 (taxonomy refresh) and Task 7 (yellowbrick fix). It's not committed.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_plotting.py
git commit -m "$(cat <<'EOF'
test(smoke): plotting harness — clustering path

Completes the smoke harness with ClusteringExperiment._available_plots
on iris features. Smoke is now full-coverage across the three task
families.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Append empirical taxonomy rows

**Files:**
- Modify: `docs/superpowers/FAILURE_TAXONOMY.md`

- [ ] **Step 1: Read `/tmp/phase3-smoke-baseline.txt` and group failures by error signature**

Inspect the captured failures. Each distinct exception class + first 80 chars of message is one taxonomy row.

Expected groups:
- Yellowbrick `YellowbrickTypeError: This estimator is not a classifier` (row 18 — already exists, do not re-add).
- Yellowbrick `YellowbrickTypeError: This estimator is not a regressor` (likely a sibling failure on regression plots).
- Possibly other matplotlib 3.8 deprecations triggered inside yellowbrick or scikitplot.

- [ ] **Step 2: Append rows starting at ID 20**

Open `docs/superpowers/FAILURE_TAXONOMY.md` and append rows (one per distinct error signature) to the table. Example template:

```markdown
| 20 | pycaret.internal.plots / yellowbrick | tests/smoke/test_plotting.py::test_regression_plot[residuals] | `yellowbrick.exceptions.YellowbrickTypeError: This estimator is not a regressor` | yellowbrick | plotting-dev | open | Empirical, Phase 3 smoke baseline. Sibling of row 18 on the regression side. Same expected fix path: pipeline-unwrap shim in pycaret/internal/patches/yellowbrick.py. |
```

Add as many rows as distinct signatures (typically 1–4 new rows).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "$(cat <<'EOF'
docs(taxonomy): append rows 20+ from Phase 3 smoke baseline

Empirical signatures captured from tests/smoke/test_plotting.py on
phase-3-plotting HEAD before any plotting fix lands. These rows are
the input to Task 7 (yellowbrick patch extension).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Yellowbrick W1 — extend the patch module

**Files:**
- Modify: `pycaret/internal/patches/yellowbrick.py`

The existing module shims `is_estimator` and `get_model_name`. We add `is_classifier` and `is_regressor` shims that unwrap pycaret's pipeline before delegating, plus an `_unwrap` helper that consolidates the "get the inner estimator" logic.

- [ ] **Step 1: Read the existing patch module to confirm helpers**

Run:

```bash
cat pycaret/internal/patches/yellowbrick.py
```

Expected: 15-line file with `is_estimator`, `get_model_name`, and the `get_estimator_from_meta_estimator` import.

- [ ] **Step 2: Extend the patch module**

Replace the entire contents of `pycaret/internal/patches/yellowbrick.py` with:

```python
"""Pycaret-side shims for yellowbrick API drift.

Yellowbrick probes a model's estimator type via `is_classifier`/`is_regressor`
on the *pipeline* pycaret hands it. Modern sklearn (>=1.6) reads
`__sklearn_tags__` / `_estimator_type` directly off the wrapped object,
so yellowbrick's checks fail on a Pipeline that doesn't expose those
attributes from its outermost step.

These shims unwrap pycaret's meta-estimator/pipeline first, then delegate
to yellowbrick's original detector. They are installed via mock.patch
in pycaret/internal/pycaret_experiment/tabular_experiment.py.
"""
from __future__ import annotations

from yellowbrick.utils.helpers import get_model_name as _get_model_name_original
from yellowbrick.utils.types import (
    is_classifier as _is_classifier_original,
    is_regressor as _is_regressor_original,
)

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator


def _unwrap(model):
    """Return the innermost estimator pycaret would dispatch on."""
    return get_estimator_from_meta_estimator(model)


def is_estimator(model):
    try:
        return callable(getattr(model, "fit"))
    except Exception:
        return False


def is_classifier(model):
    return _is_classifier_original(_unwrap(model))


def is_regressor(model):
    return _is_regressor_original(_unwrap(model))


def get_model_name(model):
    return _get_model_name_original(_unwrap(model))
```

- [ ] **Step 3: Confirm the new symbols import without error**

Run:

```bash
python -c "from pycaret.internal.patches.yellowbrick import is_classifier, is_regressor, _unwrap, get_model_name, is_estimator; print('ok')"
```

Expected: `ok`. If `ImportError: cannot import name 'is_classifier'`, yellowbrick has reorganized — search `python -c "import yellowbrick.utils.types as t; print(dir(t))"` and adjust the import path. If `is_classifier`/`is_regressor` live in `yellowbrick.utils.helpers` instead of `.types`, switch the import.

- [ ] **Step 4: Commit**

```bash
git add pycaret/internal/patches/yellowbrick.py
git commit -m "$(cat <<'EOF'
fix(plot): yellowbrick patch — pipeline-aware classifier/regressor detection

Yellowbrick's is_classifier / is_regressor probe the outer estimator,
which for pycaret is a Pipeline that does not expose _estimator_type or
__sklearn_tags__ from the inner model. The new shims unwrap via
pycaret.internal.meta_estimators.get_estimator_from_meta_estimator
before delegating, mirroring the existing get_model_name shim.

Closes the patch module side of FAILURE_TAXONOMY row 18 (and the
regression sibling appended by the empirical smoke baseline). The
mock.patch installation is updated in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Yellowbrick W1 — wire the new shims into the dispatch site

**Files:**
- Modify: `pycaret/internal/pycaret_experiment/tabular_experiment.py:526–545`

- [ ] **Step 1: Confirm the current block shape**

Run:

```bash
sed -n '530,545p' pycaret/internal/pycaret_experiment/tabular_experiment.py
```

Expected output (this is the verbatim shape — match it exactly before editing):

```python
        with patch(
            "yellowbrick.utils.types.is_estimator",
            pycaret.internal.patches.yellowbrick.is_estimator,
        ):
            with patch(
                "yellowbrick.utils.helpers.is_estimator",
                pycaret.internal.patches.yellowbrick.is_estimator,
            ):
                _base_dpi = 100
```

If the block has changed shape since this plan was written, abort this step and re-plan the patch insertion (the file may have been refactored by an earlier task).

- [ ] **Step 2: Wrap the existing block in two more `with patch(...)` context managers**

Replace the block from Step 1 with the following — the two new `is_classifier` / `is_regressor` patches go on the *outside* (innermost-to-outermost order doesn't matter for context managers, but outer placement keeps the diff smaller):

```python
        with patch(
            "yellowbrick.utils.types.is_classifier",
            pycaret.internal.patches.yellowbrick.is_classifier,
        ):
            with patch(
                "yellowbrick.utils.types.is_regressor",
                pycaret.internal.patches.yellowbrick.is_regressor,
            ):
                with patch(
                    "yellowbrick.utils.types.is_estimator",
                    pycaret.internal.patches.yellowbrick.is_estimator,
                ):
                    with patch(
                        "yellowbrick.utils.helpers.is_estimator",
                        pycaret.internal.patches.yellowbrick.is_estimator,
                    ):
                        _base_dpi = 100
```

Note: every line of the original `with patch(...)` body (everything after `_base_dpi = 100`, including the `def pipeline():` and the rest of the `plot_model` body that lives inside the original two-level `with` chain) must be re-indented one further level (4 more spaces) because we added two outer `with` blocks. Use your editor's block-indent feature, not hand re-typing. Verify with `git diff` that only indentation changed in the body — no logic changed.

- [ ] **Step 3: Re-run the smoke harness**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --tb=short 2>&1 | tee /tmp/phase3-smoke-w1.txt
```

Expected: yellowbrick-backed plots that previously failed with `YellowbrickTypeError` now pass. Any remaining failures fall into one of three buckets:
1. Matplotlib 3.8 deprecation (handled by W2 in Tasks 9–10).
2. Pathologically slow plot (handled by per-plot timeout → DEGRADED.md).
3. Genuinely broken visualizer that resists fixing (DEGRADED.md row).

If `YellowbrickTypeError` still appears for some entries, yellowbrick may probe via a *different* function (e.g., `yellowbrick.utils.types.isclassifier` without underscore, or `yellowbrick.base._is_classifier`). Trace by inspecting the failing visualizer's source: `python -c "import yellowbrick.classifier; import inspect; print(inspect.getsourcefile(yellowbrick.classifier.ROCAUC))"` and grep that file for `is_classifier` — patch the precise symbol it imports.

- [ ] **Step 4: Commit**

```bash
git add pycaret/internal/pycaret_experiment/tabular_experiment.py
git commit -m "$(cat <<'EOF'
fix(plot): install pipeline-aware is_classifier/is_regressor shims

Extends the existing mock.patch block around show_yellowbrick_plot to
install pycaret.internal.patches.yellowbrick.is_classifier and
.is_regressor over yellowbrick.utils.types. Together with the patch
module change in the previous commit, this unblocks yellowbrick-backed
visualizers that probe a pycaret Pipeline directly.

Closes FAILURE_TAXONOMY row 18 (final closure docs land at end of
Phase 3).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: W2 sweep — matplotlib 3.8 deprecations (cm.get_cmap, set_tight_layout)

**Files:**
- Modify (potentially): `pycaret/internal/plots/**`, `pycaret/internal/patches/**`, `pycaret/internal/pycaret_experiment/tabular_experiment.py`, `pycaret/internal/plots/helper.py`

- [ ] **Step 1: Sweep for `cm.get_cmap`**

Run:

```bash
git grep -nE 'cm\.get_cmap\(|matplotlib\.cm\.get_cmap\(|plt\.cm\.get_cmap\(' pycaret/
```

Expected (one of):
- No matches → no fix needed; record in LOG.md and skip to Step 2.
- 1+ matches → matplotlib 3.9 deprecates `cm.get_cmap`; replacement is `mpl.colormaps[name]` or `mpl.colormaps.get_cmap(name)`. Edit each match, preferring `mpl.colormaps[name]`. Add `import matplotlib as mpl` only if not already present at the top of the file.

If matches are found, fix them now. Each fix is a one-line replacement.

- [ ] **Step 2: Sweep for `Figure.set_tight_layout`**

```bash
git grep -nE 'set_tight_layout\(|tight_layout=True' pycaret/
```

Expected:
- `set_tight_layout(True)` → replace with `set_layout_engine("tight")`.
- `Figure(tight_layout=True)` → leave (keyword still works in 3.8).
- No matches → skip.

- [ ] **Step 3: Sweep for `Axes.bar(..., tick_label=...)`**

```bash
git grep -nE '\.bar\([^)]*tick_label\s*=' pycaret/
```

Expected: typically no matches in pycaret's plot code. If matches exist, the deprecation guidance is to set tick labels via `ax.set_xticks(positions, labels)` after the bar call.

- [ ] **Step 4: Re-run smoke**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --tb=short
```

Expected: any matplotlib-deprecation failures from earlier baseline are now gone. New failures (if any) belong to W2 step 5 or W3/W4.

- [ ] **Step 5: Commit (only if Steps 1–3 produced edits)**

```bash
git add -p   # stage only the relevant matplotlib-3.8 edits
git commit -m "$(cat <<'EOF'
fix(plot): matplotlib 3.8 deprecations in pycaret/internal/plots

Sweep replacements:
  - cm.get_cmap(name) -> mpl.colormaps[name]
  - Figure.set_tight_layout(True) -> Figure.set_layout_engine("tight")
  - (other in-file matches as listed)

No behavior change; matplotlib >=3.8 emits DeprecationWarnings (and
3.9+ removes the old APIs) for the patterns swept here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no edits were needed, append a note to `docs/superpowers/agents/plotting-dev/LOG.md` ("W2 step 1–3: no matches; sweep clean") and skip the commit.

---

## Task 10: W2 sweep — Styler.applymap and numpy-2 residue

**Files:**
- Modify (potentially): `pycaret/internal/plots/**`, `pycaret/internal/patches/**`

- [ ] **Step 1: Confirm `Styler.applymap` is absent**

```bash
git grep -nE '\.applymap\(' pycaret/
```

Expected: no matches (Phase 2 already swept). If matches re-appeared, replace `applymap` with `map` for both `DataFrame.applymap` and `Styler.applymap` call sites.

- [ ] **Step 2: Confirm numpy-2 scalar/API residue is absent in plotting paths**

```bash
git grep -nE 'np\.NaN|np\.product|np\.in1d|np\.trapz|np\.bool8|np\.float_' pycaret/internal/plots/ pycaret/internal/patches/
```

Expected: no matches. If matches exist, apply the row-11 sweep (`NaN→nan`, `product→prod`, `in1d→isin`, `trapz→trapezoid`, `bool8→bool_`, `float_→float64`).

- [ ] **Step 3: Commit (only if edits)**

```bash
git add -p
git commit -m "$(cat <<'EOF'
fix(plot): residual pandas/numpy-2 sweep in plotting paths

Catches sites that escaped the Phase 2 sweep because they live under
pycaret/internal/plots or pycaret/internal/patches — paths Phase 2 did
not exhaustively visit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no edits, append a note to LOG.md ("W2 step 1–2: residual sweep clean") and skip the commit.

---

## Task 11: W3 — plotly-resampler floor lift

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current floor and the latest stable**

```bash
grep -n 'plotly-resampler' pyproject.toml
pip index versions plotly-resampler 2>/dev/null || pip install plotly-resampler --dry-run 2>&1 | head
```

Expected: current floor is `>=0.8.3.1`. Note the latest stable from PyPI.

- [ ] **Step 2: Lift the floor**

Edit `pyproject.toml`. Change:

```toml
"plotly-resampler>=0.8.3.1",
```

to:

```toml
"plotly-resampler>=0.10",  # adjust to actual latest minor
```

(Use the actual latest stable minor as the floor, not a specific patch.)

- [ ] **Step 3: Reinstall and verify import**

```bash
pip install -e ".[full]" --quiet --upgrade plotly-resampler 2>&1 | tail
python -c "import plotly_resampler; print(plotly_resampler.__version__)"
```

Expected: import succeeds; version matches the new floor.

- [ ] **Step 4: Re-run smoke**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --tb=short
```

Expected: no new failures introduced (smoke does not exercise plotly-resampler call sites — those are time-series only). If a non-time-series plot now fails because of a transitive plotly upgrade, file a row and either fix or roll back the floor.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
feat(plot): lift plotly-resampler floor to >=0.10 (or actual latest)

The previous floor (>=0.8.3.1) predates the modernized plotly stack.
plotly-resampler is exercised by time-series plot paths only; full
verification deferred to Phase 4. This commit confirms install resolves
and the import succeeds under the modernized base deps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: W4 — mljar-scikit-plot audit

**Files:**
- Modify (potentially): `pyproject.toml`

- [ ] **Step 1: Locate scikitplot call sites**

```bash
git grep -nE 'import scikitplot|from scikitplot' pycaret/
```

Expected: `pycaret/internal/pycaret_experiment/tabular_experiment.py:14` and `pycaret/internal/plots/helper.py:6` use `import scikitplot as skplt`.

- [ ] **Step 2: Find the `skplt.*` calls**

```bash
git grep -n 'skplt\.' pycaret/
```

Expected: a handful of calls (e.g., `skplt.metrics.plot_confusion_matrix`, `skplt.metrics.plot_lift_curve`, `skplt.metrics.plot_cumulative_gain`, etc.). These map to the `confusion_matrix`, `lift`, `gain`, `ks` plot keys.

- [ ] **Step 3: Confirm the smoke covers these and they pass**

The smoke from Task 3 already includes `confusion_matrix`, `lift`, `gain`, `ks`. If they pass under matplotlib ≥3.8 + the latest mljar-scikit-plot, no floor lift is needed.

```bash
pytest "tests/smoke/test_plotting.py::test_classification_plot[confusion_matrix]" "tests/smoke/test_plotting.py::test_classification_plot[lift]" "tests/smoke/test_plotting.py::test_classification_plot[gain]" "tests/smoke/test_plotting.py::test_classification_plot[ks]" -v
```

Expected: all pass.

- [ ] **Step 4: If they pass, no `pyproject.toml` edit; append LOG.md note**

Add to `docs/superpowers/agents/plotting-dev/LOG.md`:

```markdown
## 2026-05-06 — W4 mljar-scikit-plot audit
- Smoke entries `confusion_matrix`, `lift`, `gain`, `ks` pass under
  matplotlib >= 3.8 with current unpinned mljar-scikit-plot.
- No floor lift required.
```

Commit:

```bash
git add docs/superpowers/agents/plotting-dev/LOG.md
git commit -m "$(cat <<'EOF'
docs(plotting-dev): W4 mljar-scikit-plot audit — clean

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: If they fail, add a floor lift**

If any `skplt.*`-backed plot fails:
- Inspect `pip index versions mljar-scikit-plot` and pick the latest stable.
- Edit `pyproject.toml` to add `mljar-scikit-plot>=<latest>,<...>`.
- Re-run the four smoke entries.
- If still broken, the failure may be inside `scikitplot` itself with matplotlib 3.8; either upstream a fix or degrade the affected plot under W6.
- Commit with `feat(plot): lift mljar-scikit-plot floor to >=<version>`.

---

## Task 13: W6 — populate DEGRADED.md if any

**Files:**
- Modify (potentially): `docs/superpowers/agents/plotting-dev/DEGRADED.md`
- Modify (potentially): `pycaret/internal/pycaret_experiment/tabular_experiment.py` and/or `pycaret/clustering/oop.py`
- Modify (potentially): `tests/smoke/test_plotting.py`

- [ ] **Step 1: Identify any visualizer still failing after Tasks 7–12**

Run smoke; collect any test that still fails (not skip, not pass).

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --tb=short 2>&1 | tee /tmp/phase3-smoke-final.txt
```

If all pass: skip Task 13 entirely. Add to LOG.md: "DEGRADED.md remains empty".

- [ ] **Step 2: For each still-failing plot, add a DEGRADED.md row**

```markdown
| <plot_key> | classification | <one-line reason> | <github issue or taxonomy row id> | <what would unblock it> |
```

- [ ] **Step 3: Disable the visualizer in its dispatch site**

Find the `if plot == "<plot_key>":` block (or equivalent) in `tabular_experiment.py` (or `clustering/oop.py`) and replace its body's first executable line with:

```python
raise NotImplementedError(
    f"plot='<plot_key>' is temporarily disabled in pycaret-ng under "
    f"matplotlib>=3.8 / yellowbrick>=1.5; tracked in "
    f"docs/superpowers/agents/plotting-dev/DEGRADED.md"
)
```

- [ ] **Step 4: Skip the entry in the smoke harness**

Add `<plot_key>` to the corresponding `*_DEGRADED` set in `tests/smoke/test_plotting.py`.

- [ ] **Step 5: Re-run smoke**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v
```

Expected: no failures. Skips for the degraded entries are fine.

- [ ] **Step 6: Commit (one commit covering all degradations)**

```bash
git add docs/superpowers/agents/plotting-dev/DEGRADED.md \
        pycaret/internal/pycaret_experiment/tabular_experiment.py \
        pycaret/clustering/oop.py \
        tests/smoke/test_plotting.py
git commit -m "$(cat <<'EOF'
fix(plot): degrade <N> visualizer(s) under modernized deps

Each disabled visualizer raises NotImplementedError with a pointer to
DEGRADED.md. Smoke entries skip-marked. Restoration criteria captured
per-row in DEGRADED.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: W7 — close taxonomy row 18 + refresh backlog

**Files:**
- Modify: `docs/superpowers/FAILURE_TAXONOMY.md`
- Modify: `docs/superpowers/MIGRATION_BACKLOG.md`

- [ ] **Step 1: Close row 18 (and any sibling rows from Task 6) in the taxonomy**

Edit `docs/superpowers/FAILURE_TAXONOMY.md`. Change row 18's Status column from `open` to `closed` and append to its Notes column the closing SHA. The Notes addition should be a sentence like:

```
Closed by `<sha-of-task-7>` + `<sha-of-task-8>` (pipeline-aware is_classifier/is_regressor shims in pycaret/internal/patches/yellowbrick.py installed via mock.patch in tabular_experiment.py).
```

For each Task 6 row whose underlying issue is resolved by Tasks 7–12, do the same.

- [ ] **Step 2: Refresh the row counts in `MIGRATION_BACKLOG.md`**

Update the "Row counts by owner" table to reflect Phase 3's closures. The plotting-dev row count (currently 0 in the backlog as written) should now show the rows that passed through Phase 3.

If any Phase 3 rows landed in DEGRADED.md instead of being closed, mark them `degraded` in the taxonomy and call them out in the backlog.

- [ ] **Step 3: Verify smoke is still green**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v
```

Expected: all pass or skip-only.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md docs/superpowers/MIGRATION_BACKLOG.md
git commit -m "$(cat <<'EOF'
docs(taxonomy,backlog): close row 18 + Phase 3 plotting closure pass

Rows 18 (and 20+ as appended in Phase 3) closed by the yellowbrick
pipeline-unwrap shims and the matplotlib 3.8 sweep. MIGRATION_BACKLOG
row counts refreshed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Cherry-pick verification + PR

**Files:** none (process step)

- [ ] **Step 1: Identify cherry-pick-eligible commits**

The following Phase 3 commits must apply cleanly onto `upstream/master`:
- All `fix(plot): …` commits
- All `feat(plot): …` commits
- All `chore(plot): …` commits

Exempt (pycaret-ng-only infra):
- `test(smoke): …` commits
- `docs(plotting-dev): …`, `docs(taxonomy): …`, `docs(backlog): …`, `docs(specs): …`, `docs(plans): …` commits

- [ ] **Step 2: Verify cherry-pick cleanliness**

For each cherry-pick-eligible commit SHA, dry-run a cherry-pick onto `upstream/master`:

```bash
git fetch upstream master
git checkout -b cherry-pick-test upstream/master
git cherry-pick --no-commit <sha>
git status
git cherry-pick --abort
```

Repeat per SHA. Expected: each cherry-pick reports no conflicts.

If a commit conflicts, that's a Gate D failure — investigate. The most likely causes:
- The fix touched a file that has diverged structurally from upstream.
- The fix depends on a Phase 1 or Phase 2 change that hasn't been cherry-picked upstream yet (acceptable: Gate D is "applies cleanly with the prior cherry-picked commits already on master", and prior phases are landed in `modernize` not upstream — so the verification baseline for cherry-pick is `modernize` minus this branch, not `upstream/master`).

Adjust the verification baseline as needed: prefer `git checkout -b cherry-pick-test modernize` and confirm the commit applies cleanly there.

- [ ] **Step 3: Clean up the verification branch**

```bash
git checkout phase-3-plotting
git branch -D cherry-pick-test
```

- [ ] **Step 4: Push branch and open PR**

```bash
git push -u origin phase-3-plotting
```

Then open the PR `phase-3-plotting → modernize` (the `modernize` integration branch is the spec's merge target, not `master`).

PR body template:

```markdown
## Phase 3 — Plotting Stack Modernization

Closes FAILURE_TAXONOMY row 18 (yellowbrick pipeline-unwrap) + sibling rows
appended during Phase 3 smoke baseline.

### Workstreams
- W1: Yellowbrick pipeline-aware is_classifier/is_regressor shims.
- W2: matplotlib 3.8 / pandas-Styler / numpy-2 residue sweep.
- W3: plotly-resampler floor lift.
- W4: mljar-scikit-plot audit (no lift required / lift to vX.Y).
- W5: tests/smoke/test_plotting.py (pycaret-ng-only infra; not on the
  default `pytest tests/` discovery path).
- W6: DEGRADED.md (N rows / empty).
- W7: Taxonomy + backlog refresh.

### Verification
- Local smoke: `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` green / skips-only.
- CI matrix: see Actions tab.
- Parity gate B: waived for Phase 3 (visual outputs).
- Cherry-pick discipline: each `fix(plot)`/`feat(plot)`/`chore(plot)`
  commit applies cleanly on `modernize` HEAD pre-merge. `test(smoke)` and
  `docs(*)` commits are pycaret-ng-only.

### Spec / Plan
- `docs/superpowers/specs/2026-05-06-phase-3-plotting-design.md`
- `docs/superpowers/plans/2026-05-06-phase-3-plotting.md`
```

- [ ] **Step 5: Wait for CI; merge when green**

CI gates per master spec § 3.4:
- Gate A: `pytest tests/` matrix on `{3.11, 3.12, 3.13} × {linux, macos}` green.
- Gate D: cherry-pick cleanliness verified in Step 2.

Parity gate B is waived. Smoke gate C is local-only and was verified in Task 14 step 3.

When green, merge. Then:
- Update `docs/superpowers/agents/plotting-dev/LOG.md` with the merge SHA.
- The orchestrator (you) advances to Phase 4 spec authoring.

---

## Self-Review Checklist

After implementation, verify:
- [ ] `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` is green or skips-only on `phase-3-plotting` HEAD.
- [ ] FAILURE_TAXONOMY row 18 is `closed` with closure SHA in Notes.
- [ ] Any DEGRADED.md row points to a real `NotImplementedError` raise site and to a tracking link.
- [ ] No commit message uses `--no-verify`, no commit skips hooks.
- [ ] Each cherry-pick-eligible commit applies cleanly onto `modernize` HEAD without this branch.
- [ ] Total wall-clock for `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` is under 90 s on the dev laptop.
- [ ] No edits made to upstream-pure files outside the plotting stack (no sklearn / pandas / time-series collateral damage).

If any item fails, that's not Phase 3 done — fix before opening PR.
