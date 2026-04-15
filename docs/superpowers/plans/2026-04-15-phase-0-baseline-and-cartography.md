# Phase 0 — Baseline & Cartography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the foundational infrastructure for `pycaret-ng` modernization — branch topology, agent charters, CI matrix, parity harness vs. frozen PyCaret 3.4.0, and a categorized failure taxonomy that unblocks Phases 1-5.

**Architecture:** Use a long-lived `modernize` integration branch off `master` (which stays a pure upstream mirror). Every subsequent phase branches off `modernize` and merges back when its gates pass. The parity harness is a `pytest`-runnable test suite that diffs modernized outputs against a frozen 3.4.0 baseline built in an isolated env. The failure taxonomy is a single shared markdown file owned by the Cartographer agent.

**Tech Stack:** Python 3.11/3.12/3.13, pytest, GitHub Actions, uv (for isolated env baseline build), Claude Code subagents (Ecosystem Researcher, Dep Cartographer).

**Spec reference:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md`

---

## File Structure

**Created in Phase 0:**

| Path | Responsibility |
|------|----------------|
| `docs/superpowers/agents/researcher/CHARTER.md` | Ecosystem Researcher scope |
| `docs/superpowers/agents/cartographer/CHARTER.md` | Dep Cartographer scope |
| `docs/superpowers/agents/sklearn-dev/CHARTER.md` | sklearn Migration Dev scope |
| `docs/superpowers/agents/pandas-dev/CHARTER.md` | pandas/numpy Migration Dev scope |
| `docs/superpowers/agents/plotting-dev/CHARTER.md` | Plotting Migration Dev scope |
| `docs/superpowers/agents/ts-dev/CHARTER.md` | Time-series Migration Dev scope |
| `docs/superpowers/agents/qa/CHARTER.md` | Data Scientist / QA scope |
| `docs/superpowers/agents/release/CHARTER.md` | Release Engineer scope |
| `docs/superpowers/FAILURE_TAXONOMY.md` | Shared taxonomy (Cartographer-owned) |
| `docs/superpowers/agents/researcher/FINDINGS.md` | Researcher output (written by subagent) |
| `docs/superpowers/MIGRATION_BACKLOG.md` | Phase 1-5 ordering, synthesized from taxonomy |
| `.github/workflows/ci.yml` | Pinned-deps CI (3.11/3.12/3.13 × linux/macos) |
| `.github/workflows/ci-unpinned.yml` | Weekly cron, unpinned deps |
| `.github/workflows/release.yml` | Tag-triggered PyPI publish skeleton |
| `tests/parity/__init__.py` | Package marker |
| `tests/parity/conftest.py` | Fixtures: baseline loader, tolerance config |
| `tests/parity/datasets.py` | Reference dataset loaders |
| `tests/parity/baseline.py` | Baseline artifact schema + loader |
| `tests/parity/test_compare_models_parity.py` | compare_models leaderboard parity |
| `tests/parity/test_predict_parity.py` | Predict output parity |
| `tests/parity/baselines/3.4.0/<dataset>/leaderboard.json` | Frozen 3.4.0 leaderboards |
| `tests/parity/baselines/3.4.0/<dataset>/predictions.npz` | Frozen 3.4.0 predictions |
| `scripts/__init__.py` | Package marker |
| `scripts/build_parity_baseline.py` | Builds frozen 3.4.0 baseline artifacts |
| `docs/superpowers/FORK_TOPOLOGY.md` | Branch / remote conventions |

**Modified in Phase 0:** none (spec/plan only; source code untouched).

**Branches created:**
- `modernize` (long-lived integration branch off `master`)

---

## Task 1: Fork hygiene — verify remotes & create `modernize` branch

**Files:**
- Create: `docs/superpowers/FORK_TOPOLOGY.md`

**No tests** — this task is git configuration; validation is `git` commands returning expected output.

- [ ] **Step 1: Verify upstream remote exists and points at `pycaret/pycaret`**

Run: `git remote -v`
Expected output contains:
```
origin    https://github.com/olaTechie/pycaret.git (fetch)
origin    https://github.com/olaTechie/pycaret.git (push)
upstream  https://github.com/pycaret/pycaret.git (fetch)
upstream  https://github.com/pycaret/pycaret.git (push)
```

If `upstream` is missing: `git remote add upstream https://github.com/pycaret/pycaret.git`

- [ ] **Step 2: Verify `master` is clean and on current commit**

Run: `git status && git rev-parse HEAD`
Expected: `working tree clean` plus a commit SHA.

- [ ] **Step 3: Create the `modernize` branch from current `master`**

Run:
```bash
git checkout -b modernize
git push -u origin modernize
```
Expected: new remote branch created.

- [ ] **Step 4: Return to `master` for subsequent plan work**

Run: `git checkout master`

All Phase 0 plan commits happen on `master` (they are docs/plan artifacts, not modernization code).
Phases 1-5 will branch off `modernize`.

- [ ] **Step 5: Create the branch topology doc**

Create `docs/superpowers/FORK_TOPOLOGY.md` with this exact content:

````markdown
# pycaret-ng Branch & Remote Topology

## Remotes

- `origin` → `olaTechie/pycaret` — our fork, where all work lives.
- `upstream` → `pycaret/pycaret` — the canonical PyCaret repo. Read-only reference; we never push here.

## Branches

| Branch | Purpose | Lifecycle |
|--------|---------|-----------|
| `master` | Pure mirror of upstream + our own docs/plans. No modernization code. | Permanent. Rebase onto `upstream/master` at phase boundaries. |
| `modernize` | Integration branch for all Phase 1-5 dep modernization. | Permanent until v1.0.0 release; merged to `master` at release cut. |
| `phase-N-<topic>` | Feature branch per phase (e.g. `phase-1-sklearn`). | Short-lived; merged into `modernize` when gates pass, then deleted. |
| `phase-6-X-<topic>` | Feature branches for LLM phases (e.g. `phase-6-0-llm-infra`). | Short-lived; merged into `modernize` then to `master` per minor release. |

## Workflow

1. Branch phase work off `modernize`.
2. Each commit on a phase branch must satisfy gate D (cherry-pickable onto `upstream/master`) for Phases 0-5.
3. Open a PR into `modernize` when phase gates pass.
4. Rebase `modernize` onto new upstream tags at phase boundaries if upstream releases.
5. v1.0.0 release: fast-forward `master` to `modernize`, tag `v1.0.0`.

## Never do these things

- Do NOT force-push `master` or `modernize`.
- Do NOT rewrite commits on `modernize` once merged — cherry-pickability depends on commit stability.
- Do NOT push to `upstream`.
````

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/FORK_TOPOLOGY.md
git commit -m "docs(fork): define branch & remote topology for pycaret-ng modernization"
```

---

## Task 2: Write agent CHARTERs (Phase 0-5 agents)

**Files:**
- Create: `docs/superpowers/agents/researcher/CHARTER.md`
- Create: `docs/superpowers/agents/cartographer/CHARTER.md`
- Create: `docs/superpowers/agents/sklearn-dev/CHARTER.md`
- Create: `docs/superpowers/agents/pandas-dev/CHARTER.md`
- Create: `docs/superpowers/agents/plotting-dev/CHARTER.md`
- Create: `docs/superpowers/agents/ts-dev/CHARTER.md`
- Create: `docs/superpowers/agents/qa/CHARTER.md`
- Create: `docs/superpowers/agents/release/CHARTER.md`

Every charter uses this shared schema: **Role**, **Phase**, **Inputs**, **Outputs**, **Stop conditions**, **Out-of-scope**, **Handoff protocol**.

- [ ] **Step 1: Write `researcher/CHARTER.md`**

```markdown
# Ecosystem Researcher — Charter

**Role:** Survey peer ML projects and upstream PyCaret to inform modernization patterns.

**Phase:** 0 only (short-lived).

**Inputs:**
- `pyproject.toml` (current pinned deps)
- Upstream issues & PRs at `pycaret/pycaret` (via `gh issue list`, `gh pr list`)
- Peer projects to survey: keras (keras-team/keras), ludwig (ludwig-ai/ludwig), sktime (sktime/sktime), autogluon (autogluon/autogluon), flaml (microsoft/FLAML)

**Outputs:**
- `docs/superpowers/agents/researcher/FINDINGS.md`, structured as:
  1. Upstream PRs/issues relevant to our dep bumps (with cherry-pick candidates flagged)
  2. Per-peer migration pattern summary (how did they handle sklearn ≥ 1.5? pandas ≥ 2.2? numpy ≥ 2.0?)
  3. Recommended adoption patterns for pycaret-ng (concrete, not generic)

**Stop conditions:**
- FINDINGS.md committed with all three sections populated.
- At least 10 concrete upstream references (URLs) cited.

**Out-of-scope:**
- Does NOT modify any source code.
- Does NOT write CI workflows or tests.
- Does NOT populate FAILURE_TAXONOMY.md (that is the Cartographer's job).

**Handoff protocol:**
- On completion, notify Orchestrator; Orchestrator reads FINDINGS.md before dispatching migration devs.
```

- [ ] **Step 2: Write `cartographer/CHARTER.md`**

```markdown
# Dep Cartographer — Charter

**Role:** Run the test suite under unpinned modern deps, classify every failure by root-cause dep, and populate the shared failure taxonomy.

**Phase:** 0 initially; re-invoked briefly at each Phase 1-4 start to refresh rows.

**Inputs:**
- A scratch env with modern unpinned deps (scikit-learn ≥ 1.5, pandas ≥ 2.2, numpy ≥ 2.0, scipy ≥ 1.12, matplotlib ≥ 3.8, sktime latest, pmdarima latest, statsmodels latest).
- Raw pytest output from running `pytest tests/ -x --tb=short` in that env.

**Outputs:**
- Rows in `docs/superpowers/FAILURE_TAXONOMY.md` with fields:
  `| ID | Module | Failing test | Error signature | Root-cause dep | Owner agent | Notes |`
- One row per distinct failure signature (deduplicate by stack trace fingerprint, not per-test).

**Stop conditions:**
- Every observed failure has a row.
- Every row has an owner agent from {sklearn-dev, pandas-dev, plotting-dev, ts-dev, release} — no `TBD` owners.
- Row count sanity-checked by Orchestrator against test failure count.

**Out-of-scope:**
- Does NOT fix any failing test.
- Does NOT modify `pyproject.toml` or any source file.
- Does NOT open PRs.

**Handoff protocol:**
- Commits taxonomy updates to `master` (Phase 0) or the relevant phase branch (Phases 1-4 refresh).
- Orchestrator slices taxonomy by `Owner agent` when dispatching migration devs.
```

- [ ] **Step 3: Write `sklearn-dev/CHARTER.md`**

```markdown
# sklearn Migration Dev — Charter

**Role:** Fix all `sklearn`-tagged failures to make PyCaret work on scikit-learn ≥ 1.5 while satisfying gates A, B, C, D.

**Phase:** 1.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = sklearn-dev`.
- Branch: `phase-1-sklearn` (branch off `modernize`).
- Parity harness: `tests/parity/`.

**Outputs:**
- Commits on `phase-1-sklearn`, one logical change per commit.
- Each commit message: `fix(sklearn): <short description>` referencing the taxonomy row ID(s) closed.
- PR opened to `modernize` when all sklearn rows are closed.

**Stop conditions:**
- All `sklearn`-tagged rows marked closed in the taxonomy.
- `pytest tests/` green in the CI matrix.
- `pytest tests/parity/` within tolerance (metric Δ < 1e-4, rank-correlation > 0.999).
- Every commit passes `git cherry-pick` dry-run onto `upstream/master` (gate D).

**Out-of-scope:**
- Pandas/numpy dtype fixes (hand to pandas-dev).
- Plotting stack fixes (hand to plotting-dev).
- Time-series (hand to ts-dev).
- If an sklearn fix requires touching pandas/plotting/ts code, flag in the PR description and open a handoff issue.

**Handoff protocol:**
- When blocked on a cross-cutting dep, append a note to the taxonomy row with `HANDOFF: <agent>` and move on.
- Orchestrator merges `phase-1-sklearn` → `modernize` only after QA signs off.
```

- [ ] **Step 4: Write `pandas-dev/CHARTER.md`**

Identical shape to sklearn-dev charter; substitutions:
- Role: "Fix all `pandas`- and `numpy`-tagged failures for pandas ≥ 2.2 and numpy ≥ 2.0 compatibility."
- Phase: 2.
- Branch: `phase-2-pandas`.
- Commit prefix: `fix(pandas):` or `fix(numpy):`.
- Out-of-scope: sklearn, plotting, time-series.

Write the exact file:

```markdown
# pandas / numpy Migration Dev — Charter

**Role:** Fix all `pandas`- and `numpy`-tagged failures for pandas ≥ 2.2 and numpy ≥ 2.0 compatibility while satisfying gates A, B, C, D.

**Phase:** 2.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = pandas-dev`.
- Branch: `phase-2-pandas` (branch off `modernize` after Phase 1 merge).
- Parity harness: `tests/parity/`.

**Outputs:**
- Commits on `phase-2-pandas`, one logical change per commit.
- Each commit message: `fix(pandas): ...` or `fix(numpy): ...` referencing the taxonomy row ID(s) closed.
- PR opened to `modernize` when all pandas/numpy rows are closed.

**Stop conditions:**
- All `pandas`- and `numpy`-tagged rows closed.
- `pytest tests/` green in CI matrix.
- `pytest tests/parity/` within tolerance.
- Every commit passes `git cherry-pick` dry-run onto `upstream/master`.

**Out-of-scope:**
- sklearn fixes (hand to sklearn-dev).
- Plotting fixes (hand to plotting-dev).
- Time-series fixes (hand to ts-dev).

**Handoff protocol:** Same as sklearn-dev.
```

- [ ] **Step 5: Write `plotting-dev/CHARTER.md`**

```markdown
# Plotting Migration Dev — Charter

**Role:** Fix all plotting-stack failures for matplotlib ≥ 3.8, yellowbrick, schemdraw ≥ 0.16, plotly-resampler latest while satisfying gates A, C, D. Gate B (numerical parity) is WAIVED for this phase because visual output is not numerically comparable.

**Phase:** 3.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = plotting-dev`.
- Branch: `phase-3-plotting` (can run parallel to Phase 2 per spec).
- Smoke notebook suite: `tutorials/`.

**Outputs:**
- Commits on `phase-3-plotting`, one logical change per commit.
- Commit prefix: `fix(plot):`.
- PR opened to `modernize` when all plotting rows closed.

**Stop conditions:**
- All `matplotlib|yellowbrick|schemdraw|plotly|plotly-resampler|mljar-scikit-plot`-tagged rows closed.
- `pytest tests/` green (plot-rendering tests included).
- Tutorial smoke notebooks render without exceptions.
- Cherry-pick dry-run green.

**Out-of-scope:**
- sklearn/pandas/time-series fixes.

**Handoff protocol:** Same as sklearn-dev.
```

- [ ] **Step 6: Write `ts-dev/CHARTER.md`**

```markdown
# Time-series Migration Dev — Charter

**Role:** Fix all time-series-stack failures (sktime, pmdarima, statsmodels, tbats) while satisfying gates A, B, C, D. Highest risk of structural breakage; MAY ship with narrowed API documented in `DEGRADED.md`.

**Phase:** 4.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = ts-dev`.
- Branch: `phase-4-timeseries`.
- Parity harness (time-series subset): `tests/parity/test_*_parity.py` with `--ts` marker.

**Outputs:**
- Commits on `phase-4-timeseries`.
- Commit prefix: `fix(ts):` or `chore(ts):` for narrowing decisions.
- If narrowing required: `docs/superpowers/agents/ts-dev/DEGRADED.md` enumerating removed / modified APIs with rationale.
- PR opened to `modernize` when ts rows closed.

**Stop conditions:**
- All `sktime|pmdarima|statsmodels|tbats`-tagged rows closed OR explicitly marked DEGRADED with rationale.
- `pytest tests/` green (time-series subset, with DEGRADED tests skipped via marker).
- Parity within tolerance for retained API surface.
- Cherry-pick dry-run green.

**Out-of-scope:** sklearn/pandas/plotting.

**Narrowing protocol:**
- If sktime breaking changes are structural (e.g., forecaster API gone), propose narrowing via PR comment to Orchestrator BEFORE committing removal.
- Every removed API gets a `DEGRADED.md` entry with: old signature, reason removed, suggested user workaround.

**Handoff protocol:** Same as sklearn-dev.
```

- [ ] **Step 7: Write `qa/CHARTER.md`**

```markdown
# Data Scientist / QA — Charter

**Role:** Validate each phase's output against gates B (parity) and C (smoke). Authority to block merges.

**Phase:** Continuous across Phases 1-5; active whenever a migration dev opens a PR.

**Inputs:**
- PR head commit on any `phase-N-*` branch.
- Frozen baseline: `tests/parity/baselines/3.4.0/`.
- Tutorial notebooks: `tutorials/`.

**Outputs:**
- `docs/superpowers/agents/qa/phase-N-parity.md` per phase, containing:
  - Per-dataset parity report (metric deltas, rank-correlation).
  - Per-tutorial smoke result (pass/fail, stderr excerpt on fail).
  - Per-estimator tolerance-widening requests with rationale.
- Blocking review comments on the PR when gates fail.

**Stop conditions per phase:**
- Parity report committed.
- `LGTM` on the PR OR explicit block with reasons.

**Out-of-scope:**
- Does NOT modify source code.
- Does NOT modify `pyproject.toml`.

**Authority:**
- Can block PR merge to `modernize`.
- Can propose per-estimator tolerance widening; Orchestrator approves.

**Handoff protocol:**
- Runs automatically on every PR via CI; also invoked manually by Orchestrator for deep review.
```

- [ ] **Step 8: Write `release/CHARTER.md`**

```markdown
# Release Engineer — Charter

**Role:** Own CI scaffolding (Phase 0), PyPI publishing, rename to `pycaret-ng`, upstream PR authoring, and version tagging.

**Phases:** 0 (CI scaffold), 5 (v1.0.0 release), 6.X (v1.1-v1.4 releases).

**Inputs:**
- Passing CI on `modernize`.
- Phase completion signals from Orchestrator.
- PyPI trusted-publishing configuration on `olaTechie/pycaret`.

**Outputs:**
- `.github/workflows/ci.yml`, `ci-unpinned.yml`, `release.yml`.
- `pyproject.toml` rename (`pycaret` → `pycaret-ng`) at Phase 5.
- `docs/superpowers/MIGRATION.md` (upstream 3.4.0 → pycaret-ng 1.0.0 user-facing delta).
- One upstream PR to `pycaret/pycaret` per phase (0-5) with cherry-picked commits.
- Git tag + PyPI publish per release.

**Stop conditions per release:**
- PyPI package visible at `https://pypi.org/project/pycaret-ng/<version>/`.
- Upstream PRs opened (not necessarily merged).
- MIGRATION.md updated.

**Out-of-scope:**
- Does NOT fix failing tests (migration devs' job).
- Does NOT modify source code in `pycaret/`.

**Handoff protocol:**
- Invoked by Orchestrator when a phase's merge to `modernize` passes all gates.
```

- [ ] **Step 9: Commit all charters**

```bash
git add docs/superpowers/agents/
git commit -m "docs(agents): Phase 0-5 agent charters (researcher, cartographer, 4x migration devs, QA, release)"
```

---

## Task 3: CI skeleton — pinned-deps workflow

**Files:**
- Create: `.github/workflows/ci.yml`

No TDD: this is infra. Validation is GitHub Actions acceptance (YAML parses, workflow appears in Actions tab).

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: ci (pinned)

on:
  push:
    branches:
      - master
      - modernize
      - "phase-*"
  pull_request:
    branches:
      - master
      - modernize

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    name: test py${{ matrix.python-version }} on ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install uv
        run: python -m pip install --upgrade pip uv
      - name: Install pycaret with full extras
        run: python -m uv pip install --system -e ".[full,test]"
      - name: Run test suite
        run: pytest tests/ -x --tb=short -m "not time_series_slow"
      - name: Run parity harness (modernize branch & PRs only)
        if: github.ref == 'refs/heads/modernize' || github.base_ref == 'modernize'
        run: pytest tests/parity/ -v
```

- [ ] **Step 2: Verify YAML is syntactically valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: exit 0 with no output.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: pinned-deps matrix (3.11/3.12/3.13 x linux/macos) + parity gate on modernize"
```

---

## Task 4: CI skeleton — weekly unpinned workflow

**Files:**
- Create: `.github/workflows/ci-unpinned.yml`

- [ ] **Step 1: Write `.github/workflows/ci-unpinned.yml`**

```yaml
name: ci (unpinned, weekly)

on:
  schedule:
    - cron: "0 7 * * 1"  # Mondays 07:00 UTC
  workflow_dispatch: {}

jobs:
  test-unpinned:
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    runs-on: ubuntu-latest
    name: unpinned py${{ matrix.python-version }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: modernize
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install uv
        run: python -m pip install --upgrade pip uv
      - name: Install pycaret with latest deps (overriding pins)
        run: |
          python -m uv pip install --system -e ".[full,test]"
          python -m uv pip install --system --upgrade \
            "scikit-learn>=1.5" "pandas>=2.2" "numpy>=2.0" \
            "scipy>=1.12" "matplotlib>=3.8" "sktime" "pmdarima" "statsmodels"
      - name: Run test suite (allowed to fail; populates dashboard)
        run: pytest tests/ --tb=short --junitxml=junit.xml -m "not time_series_slow"
        continue-on-error: true
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: junit-${{ matrix.python-version }}
          path: junit.xml
```

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci-unpinned.yml'))"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci-unpinned.yml
git commit -m "ci: weekly unpinned-deps canary on modernize for regression catching"
```

---

## Task 5: Release workflow skeleton

**Files:**
- Create: `.github/workflows/release.yml`

This is a skeleton — it does NOT publish yet (Phase 5 wires PyPI trusted publishing). It validates on every tag that the build succeeds.

- [ ] **Step 1: Write `.github/workflows/release.yml`**

```yaml
name: release

on:
  push:
    tags:
      - "v*.*.*"
  workflow_dispatch: {}

jobs:
  build:
    runs-on: ubuntu-latest
    name: build sdist & wheel
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build
        run: python -m pip install --upgrade pip build
      - name: Build sdist & wheel
        run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
  # publish job wired in Phase 5 with PyPI trusted publishing
```

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: release workflow skeleton (build only; publish wired in Phase 5)"
```

---

## Task 6: Parity harness — datasets module (TDD)

**Files:**
- Create: `tests/parity/__init__.py`
- Create: `tests/parity/datasets.py`
- Test: `tests/parity/test_datasets.py`

`datasets.py` exposes a single function `load_reference(name: str) -> (X, y, task)` returning features, target, and one of `{"classification", "regression", "time_series"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/parity/__init__.py` as an empty file.

Create `tests/parity/test_datasets.py`:

```python
import numpy as np
import pandas as pd
import pytest

from tests.parity.datasets import REFERENCE_DATASETS, load_reference


@pytest.mark.parametrize("name", list(REFERENCE_DATASETS.keys()))
def test_load_reference_returns_expected_shape(name):
    X, y, task = load_reference(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0
    assert task in {"classification", "regression", "time_series"}


def test_reference_datasets_includes_all_five():
    assert set(REFERENCE_DATASETS.keys()) == {
        "iris",
        "diabetes",
        "california_housing",
        "credit",
        "airline_passengers",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/parity/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.parity.datasets'`.

- [ ] **Step 3: Implement `tests/parity/datasets.py`**

```python
"""Reference dataset loaders for the parity harness.

Each loader returns (X: DataFrame, y: Series, task: str) deterministically
so parity comparisons against frozen 3.4.0 baselines are reproducible.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris

RANDOM_STATE = 42


def _load_iris() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_iris(as_frame=True)
    return data.data, data.target, "classification"


def _load_diabetes() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_diabetes(as_frame=True)
    return data.data, data.target, "regression"


def _load_california_housing() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target, "regression"


def _load_credit() -> Tuple[pd.DataFrame, pd.Series, str]:
    # PyCaret ships a 'credit' dataset in pycaret.datasets
    from pycaret.datasets import get_data
    df = get_data("credit", verbose=False)
    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    return X, y, "classification"


def _load_airline_passengers() -> Tuple[pd.DataFrame, pd.Series, str]:
    from pycaret.datasets import get_data
    df = get_data("airline", verbose=False)
    # airline dataset is a single-column series indexed by date
    y = df.iloc[:, 0].astype(float)
    X = pd.DataFrame(index=df.index)  # no exog features
    return X, y, "time_series"


REFERENCE_DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.Series, str]]] = {
    "iris": _load_iris,
    "diabetes": _load_diabetes,
    "california_housing": _load_california_housing,
    "credit": _load_credit,
    "airline_passengers": _load_airline_passengers,
}


def load_reference(name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    if name not in REFERENCE_DATASETS:
        raise KeyError(
            f"Unknown reference dataset {name!r}. "
            f"Available: {sorted(REFERENCE_DATASETS)}"
        )
    return REFERENCE_DATASETS[name]()
```

- [ ] **Step 4: Run to verify passing**

Run: `pytest tests/parity/test_datasets.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/parity/__init__.py tests/parity/datasets.py tests/parity/test_datasets.py
git commit -m "test(parity): reference dataset loaders (iris, diabetes, california_housing, credit, airline)"
```

---

## Task 7: Parity harness — baseline artifact schema (TDD)

**Files:**
- Create: `tests/parity/baseline.py`
- Test: `tests/parity/test_baseline.py`

`baseline.py` defines `LeaderboardBaseline` and `PredictionBaseline` dataclasses plus load/save helpers. Artifacts live at `tests/parity/baselines/<version>/<dataset>/{leaderboard.json,predictions.npz}`.

- [ ] **Step 1: Write the failing test**

Create `tests/parity/test_baseline.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from tests.parity.baseline import (
    LeaderboardBaseline,
    PredictionBaseline,
    load_leaderboard,
    load_predictions,
    save_leaderboard,
    save_predictions,
)


def test_leaderboard_roundtrip(tmp_path):
    b = LeaderboardBaseline(
        dataset="iris",
        version="3.4.0",
        task="classification",
        rows=[
            {"Model": "lr", "Accuracy": 0.96, "AUC": 0.99},
            {"Model": "rf", "Accuracy": 0.95, "AUC": 0.98},
        ],
    )
    path = tmp_path / "leaderboard.json"
    save_leaderboard(b, path)
    loaded = load_leaderboard(path)
    assert loaded == b


def test_predictions_roundtrip(tmp_path):
    b = PredictionBaseline(
        dataset="iris",
        version="3.4.0",
        model="lr",
        predictions=np.array([0, 1, 2, 1]),
        probabilities=np.array([[0.9, 0.05, 0.05],
                                [0.1, 0.8, 0.1],
                                [0.05, 0.05, 0.9],
                                [0.1, 0.8, 0.1]]),
    )
    path = tmp_path / "predictions.npz"
    save_predictions(b, path)
    loaded = load_predictions(path)
    assert loaded.dataset == b.dataset
    assert loaded.model == b.model
    np.testing.assert_array_equal(loaded.predictions, b.predictions)
    np.testing.assert_array_almost_equal(loaded.probabilities, b.probabilities)


def test_load_leaderboard_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_leaderboard(tmp_path / "nonexistent.json")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/parity/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.parity.baseline'`.

- [ ] **Step 3: Implement `tests/parity/baseline.py`**

```python
"""Schema & I/O for parity baseline artifacts.

Artifacts live under tests/parity/baselines/<version>/<dataset>/:
  - leaderboard.json — sorted compare_models() output.
  - predictions.npz  — per-model predictions + probabilities on the holdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class LeaderboardBaseline:
    dataset: str
    version: str
    task: str  # "classification" | "regression" | "time_series"
    rows: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "task": self.task,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LeaderboardBaseline":
        return cls(
            dataset=d["dataset"],
            version=d["version"],
            task=d["task"],
            rows=d["rows"],
        )


@dataclass
class PredictionBaseline:
    dataset: str
    version: str
    model: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None  # None for regression/TS


def save_leaderboard(b: LeaderboardBaseline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(b.to_dict(), indent=2, sort_keys=True))


def load_leaderboard(path: Path) -> LeaderboardBaseline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return LeaderboardBaseline.from_dict(json.loads(path.read_text()))


def save_predictions(b: PredictionBaseline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"predictions": b.predictions}
    if b.probabilities is not None:
        arrays["probabilities"] = b.probabilities
    np.savez(
        path,
        dataset=np.array(b.dataset),
        version=np.array(b.version),
        model=np.array(b.model),
        **arrays,
    )


def load_predictions(path: Path) -> PredictionBaseline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=False)
    probs = npz["probabilities"] if "probabilities" in npz.files else None
    return PredictionBaseline(
        dataset=str(npz["dataset"]),
        version=str(npz["version"]),
        model=str(npz["model"]),
        predictions=npz["predictions"],
        probabilities=probs,
    )
```

- [ ] **Step 4: Run to verify passing**

Run: `pytest tests/parity/test_baseline.py -v`
Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/parity/baseline.py tests/parity/test_baseline.py
git commit -m "test(parity): baseline artifact schema (leaderboard JSON + predictions NPZ)"
```

---

## Task 8: Parity harness — conftest & tolerance config

**Files:**
- Create: `tests/parity/conftest.py`

- [ ] **Step 1: Write `tests/parity/conftest.py`**

```python
"""Parity harness fixtures and config.

Tolerances are the gate-B thresholds from the pycaret-ng spec:
  - Metric absolute delta < 1e-4
  - Prediction rank-correlation > 0.999

Per-estimator widening is allowed via PARITY_TOLERANCE_OVERRIDES and must
be justified in docs/superpowers/agents/qa/phase-N-parity.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

BASELINE_VERSION = "3.4.0"
BASELINE_ROOT = Path(__file__).parent / "baselines" / BASELINE_VERSION

METRIC_ABS_TOLERANCE = 1e-4
PREDICTION_RANK_CORR_MIN = 0.999

# Per-estimator overrides. Populated by QA as needed; empty in Phase 0.
PARITY_TOLERANCE_OVERRIDES: Dict[str, Dict[str, float]] = {}


@dataclass
class ParityConfig:
    baseline_root: Path
    metric_abs_tol: float
    rank_corr_min: float
    overrides: Dict[str, Dict[str, float]]


@pytest.fixture(scope="session")
def parity_config() -> ParityConfig:
    return ParityConfig(
        baseline_root=BASELINE_ROOT,
        metric_abs_tol=METRIC_ABS_TOLERANCE,
        rank_corr_min=PREDICTION_RANK_CORR_MIN,
        overrides=PARITY_TOLERANCE_OVERRIDES,
    )


def pytest_collection_modifyitems(config, items):
    """Skip parity tests when baseline artifacts are missing.

    This lets CI run on a fresh clone before Task 12 produces baselines.
    """
    if not BASELINE_ROOT.exists():
        skip = pytest.mark.skip(
            reason=f"Baseline not built yet at {BASELINE_ROOT}. "
            f"Run scripts/build_parity_baseline.py."
        )
        for item in items:
            if "parity" in str(item.fspath):
                item.add_marker(skip)
```

- [ ] **Step 2: Verify import succeeds (no test yet — this is fixture plumbing)**

Run: `python -c "from tests.parity.conftest import ParityConfig; print(ParityConfig)"`
Expected: `<class 'tests.parity.conftest.ParityConfig'>`.

- [ ] **Step 3: Commit**

```bash
git add tests/parity/conftest.py
git commit -m "test(parity): conftest with tolerance config (metric<1e-4, rank-corr>0.999)"
```

---

## Task 9: Parity harness — compare_models leaderboard test (TDD)

**Files:**
- Create: `tests/parity/test_compare_models_parity.py`

Test compares current-HEAD `compare_models()` leaderboard against frozen 3.4.0 leaderboard, per dataset, within tolerance.

- [ ] **Step 1: Write the test**

```python
"""Parity test: compare_models() leaderboard vs. frozen 3.4.0 baseline.

This test will SKIP cleanly until Task 12 builds the baseline artifacts.
Once artifacts exist, it asserts metric absolute deltas are within tolerance.
"""
from __future__ import annotations

import pytest

from tests.parity.baseline import load_leaderboard
from tests.parity.datasets import REFERENCE_DATASETS, load_reference


@pytest.mark.parametrize("dataset_name", ["iris", "diabetes", "california_housing", "credit"])
def test_compare_models_leaderboard_parity(dataset_name, parity_config):
    baseline_path = parity_config.baseline_root / dataset_name / "leaderboard.json"
    if not baseline_path.exists():
        pytest.skip(f"Baseline missing: {baseline_path}")

    baseline = load_leaderboard(baseline_path)
    X, y, task = load_reference(dataset_name)

    current = _run_compare_models(X, y, task)

    # Rows are keyed by Model id
    baseline_by_model = {r["Model"]: r for r in baseline.rows}
    current_by_model = {r["Model"]: r for r in current}

    missing = set(baseline_by_model) - set(current_by_model)
    assert not missing, (
        f"Models present in 3.4.0 but absent in current: {sorted(missing)}"
    )

    for model, base_row in baseline_by_model.items():
        cur_row = current_by_model[model]
        tol = parity_config.overrides.get(model, {})
        for metric, base_val in base_row.items():
            if metric == "Model" or not isinstance(base_val, (int, float)):
                continue
            cur_val = cur_row.get(metric)
            assert cur_val is not None, f"Metric {metric} missing for model {model}"
            metric_tol = tol.get(metric, parity_config.metric_abs_tol)
            assert abs(cur_val - base_val) <= metric_tol, (
                f"{dataset_name}/{model}/{metric}: "
                f"current={cur_val}, baseline={base_val}, delta={abs(cur_val - base_val)}, "
                f"tol={metric_tol}"
            )


def _run_compare_models(X, y, task):
    """Run compare_models on current HEAD and return leaderboard as list of dicts."""
    if task == "classification":
        from pycaret.classification import ClassificationExperiment
        exp = ClassificationExperiment()
        exp.setup(data=_to_frame(X, y), target=y.name or "target", session_id=42, verbose=False, html=False)
    elif task == "regression":
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment()
        exp.setup(data=_to_frame(X, y), target=y.name or "target", session_id=42, verbose=False, html=False)
    else:
        pytest.skip(f"Task {task} handled in test_predict_parity")

    _ = exp.compare_models(n_select=1, verbose=False)
    leaderboard_df = exp.pull()
    leaderboard_df = leaderboard_df.reset_index().rename(columns={"index": "Model"})
    # Determinism: sort by Model id
    leaderboard_df = leaderboard_df.sort_values("Model")
    return leaderboard_df.to_dict(orient="records")


def _to_frame(X, y):
    import pandas as pd
    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    return df
```

- [ ] **Step 2: Run to verify it SKIPs cleanly (no baseline yet)**

Run: `pytest tests/parity/test_compare_models_parity.py -v`
Expected: all tests marked SKIPPED with reason "Baseline missing:" or "Baseline not built yet".

- [ ] **Step 3: Commit**

```bash
git add tests/parity/test_compare_models_parity.py
git commit -m "test(parity): compare_models leaderboard parity test (skips until baseline built)"
```

---

## Task 10: Parity harness — predict parity test (TDD)

**Files:**
- Create: `tests/parity/test_predict_parity.py`

- [ ] **Step 1: Write the test**

```python
"""Parity test: per-model prediction arrays vs. frozen 3.4.0 baseline.

Uses Spearman rank-correlation for continuous outputs and exact-match-rate
for classification labels.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from tests.parity.baseline import load_predictions
from tests.parity.datasets import load_reference


@pytest.mark.parametrize("dataset_name", ["iris", "diabetes", "california_housing", "credit"])
def test_predict_parity(dataset_name, parity_config):
    baseline_dir = parity_config.baseline_root / dataset_name
    if not baseline_dir.exists():
        pytest.skip(f"Baseline dir missing: {baseline_dir}")

    pred_files = sorted(baseline_dir.glob("predictions_*.npz"))
    if not pred_files:
        pytest.skip(f"No per-model predictions in {baseline_dir}")

    X, y, task = load_reference(dataset_name)

    for pred_path in pred_files:
        baseline = load_predictions(pred_path)
        current_preds, current_probs = _predict_current(X, y, task, baseline.model)

        if task == "classification":
            match_rate = float(np.mean(current_preds == baseline.predictions))
            assert match_rate >= 0.99, (
                f"{dataset_name}/{baseline.model}: label match rate {match_rate:.4f} < 0.99"
            )
            if baseline.probabilities is not None and current_probs is not None:
                for col in range(baseline.probabilities.shape[1]):
                    rho, _ = spearmanr(current_probs[:, col], baseline.probabilities[:, col])
                    assert rho >= parity_config.rank_corr_min, (
                        f"{dataset_name}/{baseline.model}/proba[{col}]: "
                        f"rank-corr {rho:.4f} < {parity_config.rank_corr_min}"
                    )
        elif task == "regression":
            rho, _ = spearmanr(current_preds, baseline.predictions)
            assert rho >= parity_config.rank_corr_min, (
                f"{dataset_name}/{baseline.model}: "
                f"regression rank-corr {rho:.4f} < {parity_config.rank_corr_min}"
            )


def _predict_current(X, y, task, model_name):
    """Re-fit and predict with named model on current HEAD."""
    if task == "classification":
        from pycaret.classification import ClassificationExperiment
        exp = ClassificationExperiment()
    else:
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment()

    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    exp.setup(data=df, target=target, session_id=42, verbose=False, html=False)
    model = exp.create_model(model_name, verbose=False)
    preds_df = exp.predict_model(model, data=df, verbose=False)
    preds = preds_df["prediction_label"].values if "prediction_label" in preds_df.columns else preds_df["Label"].values
    probs = None
    if task == "classification":
        score_cols = [c for c in preds_df.columns if c.startswith("prediction_score_") or c.startswith("Score_")]
        if score_cols:
            probs = preds_df[sorted(score_cols)].values
    return preds, probs
```

- [ ] **Step 2: Run to verify it SKIPs cleanly**

Run: `pytest tests/parity/test_predict_parity.py -v`
Expected: all tests SKIPPED with "Baseline dir missing" or "No per-model predictions".

- [ ] **Step 3: Commit**

```bash
git add tests/parity/test_predict_parity.py
git commit -m "test(parity): predict-output parity test (label match + rank-corr)"
```

---

## Task 11: Baseline build script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/build_parity_baseline.py`

This script is RUN MANUALLY in an isolated env with `pycaret==3.4.0` installed on Python 3.11. It produces the frozen baselines the parity tests diff against. It does NOT run in CI.

- [ ] **Step 1: Create `scripts/__init__.py`**

Empty file:

```python
```

- [ ] **Step 2: Write `scripts/build_parity_baseline.py`**

```python
"""Build frozen PyCaret 3.4.0 baseline artifacts for parity testing.

USAGE:
    # In an ISOLATED env with pycaret==3.4.0 on Python 3.11:
    uv venv --python 3.11 /tmp/pycaret-baseline
    source /tmp/pycaret-baseline/bin/activate
    uv pip install "pycaret[full]==3.4.0"
    cd <repo-root>
    python scripts/build_parity_baseline.py

The script writes to tests/parity/baselines/3.4.0/<dataset>/ and must be
re-run only if the reference dataset set changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.parity.baseline import (  # noqa: E402
    LeaderboardBaseline,
    PredictionBaseline,
    save_leaderboard,
    save_predictions,
)
from tests.parity.datasets import REFERENCE_DATASETS, load_reference  # noqa: E402

BASELINE_VERSION = "3.4.0"
OUT_ROOT = REPO_ROOT / "tests" / "parity" / "baselines" / BASELINE_VERSION

TOP_N_MODELS = 3  # persist predictions for the top-N leaderboard rows


def build_classification_or_regression(dataset_name: str, X, y, task: str):
    from pycaret.classification import ClassificationExperiment
    from pycaret.regression import RegressionExperiment

    exp_cls = ClassificationExperiment if task == "classification" else RegressionExperiment
    exp = exp_cls()
    df = X.copy()
    target = y.name or "target"
    df[target] = y.values
    exp.setup(data=df, target=target, session_id=42, verbose=False, html=False)

    _ = exp.compare_models(n_select=TOP_N_MODELS, verbose=False)
    leaderboard_df = exp.pull().reset_index().rename(columns={"index": "Model"})
    leaderboard_df = leaderboard_df.sort_values("Model")

    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_leaderboard(
        LeaderboardBaseline(
            dataset=dataset_name,
            version=BASELINE_VERSION,
            task=task,
            rows=leaderboard_df.to_dict(orient="records"),
        ),
        out_dir / "leaderboard.json",
    )

    top_models = leaderboard_df["Model"].head(TOP_N_MODELS).tolist()
    for model_name in top_models:
        model = exp.create_model(model_name, verbose=False)
        preds_df = exp.predict_model(model, data=df, verbose=False)
        preds = (preds_df["prediction_label"].values
                 if "prediction_label" in preds_df.columns
                 else preds_df["Label"].values)
        probs = None
        if task == "classification":
            score_cols = sorted(
                [c for c in preds_df.columns
                 if c.startswith("prediction_score_") or c.startswith("Score_")]
            )
            if score_cols:
                probs = preds_df[score_cols].values
        save_predictions(
            PredictionBaseline(
                dataset=dataset_name,
                version=BASELINE_VERSION,
                model=model_name,
                predictions=np.asarray(preds),
                probabilities=probs,
            ),
            out_dir / f"predictions_{model_name}.npz",
        )

    print(f"[ok] {dataset_name}: leaderboard + {len(top_models)} prediction files")


def build_time_series(dataset_name: str, X, y):
    from pycaret.time_series import TSForecastingExperiment

    exp = TSForecastingExperiment()
    exp.setup(data=y, fh=12, session_id=42, verbose=False)
    _ = exp.compare_models(n_select=TOP_N_MODELS, verbose=False)
    leaderboard_df = exp.pull().reset_index().rename(columns={"index": "Model"})
    leaderboard_df = leaderboard_df.sort_values("Model")

    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_leaderboard(
        LeaderboardBaseline(
            dataset=dataset_name,
            version=BASELINE_VERSION,
            task="time_series",
            rows=leaderboard_df.to_dict(orient="records"),
        ),
        out_dir / "leaderboard.json",
    )

    top_models = leaderboard_df["Model"].head(TOP_N_MODELS).tolist()
    for model_name in top_models:
        model = exp.create_model(model_name, verbose=False)
        preds = exp.predict_model(model, fh=12, verbose=False)
        preds_arr = np.asarray(preds["y_pred"].values if "y_pred" in preds.columns else preds.iloc[:, 0].values, dtype=float)
        save_predictions(
            PredictionBaseline(
                dataset=dataset_name,
                version=BASELINE_VERSION,
                model=model_name,
                predictions=preds_arr,
                probabilities=None,
            ),
            out_dir / f"predictions_{model_name}.npz",
        )
    print(f"[ok] {dataset_name}: ts leaderboard + {len(top_models)} prediction files")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    import pycaret
    if not pycaret.__version__.startswith("3.4.0"):
        raise SystemExit(
            f"Expected pycaret==3.4.0 but got {pycaret.__version__}. "
            f"Activate the isolated baseline env before running this script."
        )
    for name in REFERENCE_DATASETS:
        X, y, task = load_reference(name)
        if task == "time_series":
            build_time_series(name, X, y)
        else:
            build_classification_or_regression(name, X, y, task)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Lint-check the script (syntax only; we don't execute it here)**

Run: `python -c "import ast; ast.parse(open('scripts/build_parity_baseline.py').read())"`
Expected: exit 0 with no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py scripts/build_parity_baseline.py
git commit -m "scripts: build_parity_baseline.py — freezes PyCaret 3.4.0 leaderboards + predictions"
```

---

## Task 12: Build the frozen 3.4.0 baseline artifacts

**Files:**
- Create: `tests/parity/baselines/3.4.0/<dataset>/leaderboard.json` (5 files)
- Create: `tests/parity/baselines/3.4.0/<dataset>/predictions_*.npz` (up to 15 files)

This task EXECUTES the build script in an isolated env and commits the resulting artifacts.

- [ ] **Step 1: Create the isolated baseline env**

Run:
```bash
python -m pip install --user uv
uv venv --python 3.11 /tmp/pycaret-baseline
source /tmp/pycaret-baseline/bin/activate
uv pip install "pycaret[full]==3.4.0"
```
Expected: venv active, `python -c "import pycaret; print(pycaret.__version__)"` prints `3.4.0`.

- [ ] **Step 2: Run the baseline build script**

Run (with venv still active, from repo root):
```bash
python scripts/build_parity_baseline.py
```
Expected: five `[ok] <dataset>: ...` lines, no tracebacks. Runtime ~5-15 minutes.

- [ ] **Step 3: Deactivate venv and return to project env**

Run: `deactivate`

- [ ] **Step 4: Verify artifacts exist**

Run: `find tests/parity/baselines/3.4.0 -type f | sort`
Expected: 5 `leaderboard.json` files + ≥10 `predictions_*.npz` files across 5 dataset directories.

- [ ] **Step 5: Run the parity test suite to confirm it now passes vs. itself**

Run: `pytest tests/parity/ -v`
Expected: all tests PASS (current HEAD is still PyCaret 3.4.0, so parity is trivial — this is the Phase 0 sanity check).

- [ ] **Step 6: Commit**

```bash
git add tests/parity/baselines/3.4.0/
git commit -m "test(parity): frozen PyCaret 3.4.0 baseline artifacts (leaderboards + predictions)"
```

---

## Task 13: FAILURE_TAXONOMY.md scaffold

**Files:**
- Create: `docs/superpowers/FAILURE_TAXONOMY.md`

- [ ] **Step 1: Write the scaffold**

```markdown
# FAILURE_TAXONOMY

Shared, Cartographer-owned. Every test failure observed under unpinned modern
deps gets a row. Owner agent is set from {sklearn-dev, pandas-dev, plotting-dev,
ts-dev, release}. No row may have `TBD` owner when Phase 0 completes.

## Schema

| Field | Meaning |
|-------|---------|
| ID | Monotonic integer, never reused. |
| Module | `pycaret.<module>` path most implicated. |
| Failing test | `tests/<path>::<test_name>` — first observed. |
| Error signature | Exception class + first 80 chars of message. Dedup key. |
| Root-cause dep | One of: sklearn, pandas, numpy, scipy, matplotlib, yellowbrick, schemdraw, plotly, plotly-resampler, sktime, pmdarima, statsmodels, tbats. |
| Owner agent | sklearn-dev \| pandas-dev \| plotting-dev \| ts-dev \| release |
| Status | open \| in-progress \| closed \| degraded |
| Notes | Fix SHA, handoff markers, rationale for tolerance widening, etc. |

## Environment used to generate this taxonomy

- Python version: <filled by Cartographer at dispatch>
- OS: <filled>
- Unpinned versions: <filled — `pip freeze` excerpt>

## Rows

| ID | Module | Failing test | Error signature | Root-cause dep | Owner agent | Status | Notes |
|----|--------|--------------|------------------|----------------|-------------|--------|-------|
| <!-- Cartographer appends rows here --> |

## Completion checklist (Phase 0)

- [ ] Every distinct error signature has a row.
- [ ] No row has `TBD` owner.
- [ ] Row count ≥ distinct-test-failure count from the raw pytest run.
- [ ] Cartographer LOG.md updated with the `pip freeze` used.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md
git commit -m "docs(taxonomy): FAILURE_TAXONOMY scaffold with schema and completion checklist"
```

---

## Task 14: Dispatch Ecosystem Researcher agent

**Files:**
- Create: `docs/superpowers/agents/researcher/FINDINGS.md` (written by the dispatched subagent)
- Create: `docs/superpowers/agents/researcher/LOG.md`

Uses the Claude Code `Agent` tool with `subagent_type=general-purpose`.

- [ ] **Step 1: Dispatch the Researcher subagent**

Call the `Agent` tool with:
- `subagent_type`: `general-purpose`
- `description`: `pycaret-ng ecosystem research`
- `prompt`:

````
You are the Ecosystem Researcher for the pycaret-ng fork. Your charter is at
docs/superpowers/agents/researcher/CHARTER.md — read it before starting.

Your deliverable is docs/superpowers/agents/researcher/FINDINGS.md with three
sections:

1. **Upstream cherry-pick candidates** — browse `pycaret/pycaret` issues and PRs
   (use the `gh` CLI or WebFetch on github.com). Identify:
   - Unmerged PRs that fix sklearn/pandas/numpy compat.
   - Open issues describing breakage under modern deps.
   - Closed-unmerged PRs with useful diffs we can adapt.
   Provide at least 10 concrete references with URLs and one-line summaries
   each. Tag each with the dep it addresses.

2. **Peer migration patterns** — for each of keras, ludwig, sktime, autogluon,
   flaml, summarize how they handled:
   - scikit-learn ≥ 1.5 (set_output, transformer tags, container API)
   - pandas ≥ 2.2 (applymap → map, CoW, dtype changes)
   - numpy ≥ 2.0 (bool8, scalar deprecations)
   Cite commit SHAs or PR numbers where possible, not just prose.

3. **Recommended adoption patterns for pycaret-ng** — three concrete
   recommendations for our modernization, each phrased as "When fixing X in
   PyCaret's Y module, apply pattern Z as seen in <peer project reference>."

Constraints:
- Do NOT modify any source code.
- Do NOT touch FAILURE_TAXONOMY.md (that's the Cartographer's file).
- Do NOT write CI workflows.
- Also write docs/superpowers/agents/researcher/LOG.md summarizing your
  session: start time, major queries run, rough duration.
- Commit your work with message: `docs(researcher): ecosystem findings for pycaret-ng modernization`.

Report back in under 200 words: how many upstream references you found, one
headline recommendation, and any surprises.
````

- [ ] **Step 2: Verify FINDINGS.md has all three required sections**

Run:
```bash
grep -c "^## " docs/superpowers/agents/researcher/FINDINGS.md
```
Expected: at least 3 (section headings).

Run:
```bash
grep -c "https://github.com/" docs/superpowers/agents/researcher/FINDINGS.md
```
Expected: at least 10 URLs.

- [ ] **Step 3: If either check fails, re-dispatch the Researcher agent with the specific gap called out, then re-verify.**

- [ ] **Step 4: Confirm commit exists**

Run: `git log --oneline -5`
Expected: top commit message matches `docs(researcher):`.

---

## Task 15: Dispatch Dep Cartographer agent

**Files:**
- Modify: `docs/superpowers/FAILURE_TAXONOMY.md` (rows appended by subagent)
- Create: `docs/superpowers/agents/cartographer/LOG.md`

Cartographer runs the test suite in an unpinned env and classifies failures.

- [ ] **Step 1: Dispatch the Cartographer subagent**

Call the `Agent` tool with:
- `subagent_type`: `general-purpose`
- `description`: `pycaret-ng failure cartography`
- `prompt`:

````
You are the Dep Cartographer for the pycaret-ng fork. Your charter is at
docs/superpowers/agents/cartographer/CHARTER.md — read it before starting.

Procedure:

1. Create an isolated env with modern unpinned deps:
   ```
   python -m pip install --user uv
   uv venv --python 3.11 /tmp/pycaret-unpinned
   source /tmp/pycaret-unpinned/bin/activate
   uv pip install -e ".[full,test]"
   uv pip install --upgrade \
     "scikit-learn>=1.5" "pandas>=2.2" "numpy>=2.0" \
     "scipy>=1.12" "matplotlib>=3.8" "sktime" "pmdarima" "statsmodels"
   pip freeze > /tmp/unpinned-freeze.txt
   ```

2. Run the test suite with short tracebacks:
   ```
   pytest tests/ -x=false --tb=short --no-header -rN 2>&1 | tee /tmp/pytest-unpinned.log
   ```
   Note: `--tb=short` gives you the error class + message you need for the
   signature field.

3. Deduplicate failures by (ExceptionClass, first ~80 chars of message).

4. For each distinct signature, append a row to
   docs/superpowers/FAILURE_TAXONOMY.md with:
   - Monotonic ID (check existing max + 1)
   - Module = the deepest `pycaret.*` frame in the traceback
   - Failing test = the first test observed with this signature
   - Error signature = `<ExceptionClass>: <first 80 chars>`
   - Root-cause dep = use the error message + deprecation warnings + the
     FINDINGS.md from the Researcher to classify
   - Owner agent = from {sklearn-dev, pandas-dev, plotting-dev, ts-dev, release}
   - Status = open
   - Notes = any cross-cutting flags, e.g., `also-touches: pandas`

5. Fill the "Environment used to generate this taxonomy" section with the
   Python version, OS, and a condensed `pip freeze` (just the major
   ML deps, ≤ 20 lines).

6. Write docs/superpowers/agents/cartographer/LOG.md summarizing: total
   failures, distinct signatures, runtime, any tests that hung.

7. Constraints:
   - Do NOT fix any test.
   - Do NOT modify pyproject.toml or any pycaret/ source file.
   - Do NOT run parity tests (Task 12 handled that; parity will fail in
     unpinned env for obvious reasons).
   - Commit with message: `docs(taxonomy): initial Phase 0 cartography from unpinned-deps run`.

Report back in under 200 words: total-failure count, distinct-signature count,
top 3 most-impacted `pycaret.*` submodules, and which owner agent gets the most
rows.
````

- [ ] **Step 2: Verify taxonomy has rows and zero `TBD` owners**

Run:
```bash
awk '/^\| *[0-9]+ *\|/' docs/superpowers/FAILURE_TAXONOMY.md | wc -l
```
Expected: > 0 (specific number depends on unpinned test run).

Run:
```bash
grep -c "TBD" docs/superpowers/FAILURE_TAXONOMY.md
```
Expected: 0.

- [ ] **Step 3: If `TBD` count > 0, re-dispatch the Cartographer to resolve owner assignments, then re-verify.**

- [ ] **Step 4: Confirm commit**

Run: `git log --oneline -5`
Expected: top commit matches `docs(taxonomy):`.

---

## Task 16: Synthesize migration backlog

**Files:**
- Create: `docs/superpowers/MIGRATION_BACKLOG.md`

Orchestrator-owned synthesis that turns the taxonomy into an ordered Phase 1-5 plan.

- [ ] **Step 1: Count rows per owner**

Run (adapt awk column index after verifying taxonomy layout):
```bash
awk -F'|' '/^\| *[0-9]+/ {gsub(/ /, "", $7); print $7}' docs/superpowers/FAILURE_TAXONOMY.md | sort | uniq -c | sort -rn
```
Expected: lines like `  43 sklearn-dev`, `  18 pandas-dev`, etc. Capture numbers.

- [ ] **Step 2: Write `MIGRATION_BACKLOG.md`**

Fill in real numbers from Step 1 where the document says `<N>`.

```markdown
# Migration Backlog (Phases 1-5)

Synthesized from `FAILURE_TAXONOMY.md` at end of Phase 0.

## Row counts by owner

| Owner agent | Rows | Share |
|-------------|------|-------|
| sklearn-dev | <N> | <pct>% |
| pandas-dev | <N> | <pct>% |
| plotting-dev | <N> | <pct>% |
| ts-dev | <N> | <pct>% |
| release | <N> | <pct>% |

Total rows: <N>

## Phase ordering (locked)

1. **Phase 1 — sklearn** (largest blast radius; no downstream dep on other phases).
2. **Phase 2 — pandas/numpy** (rebased on Phase 1).
3. **Phase 3 — plotting** (may run parallel to Phase 2 if independent).
4. **Phase 4 — time-series** (rebased on 1-3; highest risk).
5. **Phase 5 — release** (rename, PyPI publish, upstream PRs).

## Phase-start entry criteria

Every phase begins with a Cartographer refresh on the current `modernize` HEAD
(charter allows re-invocation). Row set may shrink as upstream deps evolve.

## Phase-close exit criteria (all four gates)

- Gate A: `pytest tests/` green on CI matrix.
- Gate B (phases 1-4 only): `pytest tests/parity/` within tolerance.
- Gate C: tutorial notebooks render end-to-end.
- Gate D (phases 0-5 only): every commit cherry-pick-clean onto
  `upstream/master`.

## Parallel-work decision

Phase 2 and Phase 3 MAY run in parallel if the Cartographer flags no
cross-cutting rows (i.e., no row tagged both `pandas` and `matplotlib`).
Orchestrator decides at Phase 2 kickoff.

## Next action

Invoke `writing-plans` for Phase 1 (sklearn migration) using
`docs/superpowers/agents/sklearn-dev/CHARTER.md` plus the
`sklearn-dev`-tagged slice of the taxonomy as input.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/MIGRATION_BACKLOG.md
git commit -m "docs(backlog): Phase 1-5 migration backlog synthesized from cartography"
```

---

## Task 17: Phase 0 gate — verify all artifacts

**Files:** none created; this is a verification task.

- [ ] **Step 1: Verify all Phase 0 artifacts exist**

Run:
```bash
for f in \
  docs/superpowers/FORK_TOPOLOGY.md \
  docs/superpowers/FAILURE_TAXONOMY.md \
  docs/superpowers/MIGRATION_BACKLOG.md \
  docs/superpowers/agents/researcher/CHARTER.md \
  docs/superpowers/agents/researcher/FINDINGS.md \
  docs/superpowers/agents/cartographer/CHARTER.md \
  docs/superpowers/agents/sklearn-dev/CHARTER.md \
  docs/superpowers/agents/pandas-dev/CHARTER.md \
  docs/superpowers/agents/plotting-dev/CHARTER.md \
  docs/superpowers/agents/ts-dev/CHARTER.md \
  docs/superpowers/agents/qa/CHARTER.md \
  docs/superpowers/agents/release/CHARTER.md \
  .github/workflows/ci.yml \
  .github/workflows/ci-unpinned.yml \
  .github/workflows/release.yml \
  tests/parity/__init__.py \
  tests/parity/conftest.py \
  tests/parity/datasets.py \
  tests/parity/baseline.py \
  tests/parity/test_compare_models_parity.py \
  tests/parity/test_predict_parity.py \
  scripts/build_parity_baseline.py; do
    [ -f "$f" ] && echo "ok   $f" || echo "MISS $f"
done
```
Expected: every line starts with `ok   `. Any `MISS` means go back to the relevant task.

- [ ] **Step 2: Verify baseline artifacts exist**

Run:
```bash
find tests/parity/baselines/3.4.0 -type f -name "leaderboard.json" | wc -l
```
Expected: `5` (one per reference dataset).

Run:
```bash
find tests/parity/baselines/3.4.0 -type f -name "predictions_*.npz" | wc -l
```
Expected: at least `10` (top-N predictions per non-TS dataset).

- [ ] **Step 3: Verify parity harness runs green against itself (current HEAD is still 3.4.0 code)**

Run: `pytest tests/parity/ -v`
Expected: 0 failures. (Skips allowed for time-series if not in baseline.)

- [ ] **Step 4: Verify `modernize` branch exists on remote**

Run: `git ls-remote --heads origin modernize`
Expected: a line with a SHA and `refs/heads/modernize`.

- [ ] **Step 5: Verify FAILURE_TAXONOMY is populated and clean**

Run: `grep -c "TBD" docs/superpowers/FAILURE_TAXONOMY.md`
Expected: `0`.

Run: `awk -F'|' '/^\| *[0-9]+/ {n++} END {print n}' docs/superpowers/FAILURE_TAXONOMY.md`
Expected: > 0.

- [ ] **Step 6: Tag the Phase 0 completion**

Run:
```bash
git tag phase-0-complete
git push origin phase-0-complete
```
Expected: tag pushed.

- [ ] **Step 7: Announce Phase 0 complete and hand off to writing-plans for Phase 1**

The next `writing-plans` invocation takes `docs/superpowers/agents/sklearn-dev/CHARTER.md` plus the `sklearn-dev`-tagged slice of the FAILURE_TAXONOMY as input and produces `docs/superpowers/plans/<date>-phase-1-sklearn.md`.
