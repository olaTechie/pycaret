# Phase 4 Time-series Stack Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize pycaret's time-series stack on `phase-4-timeseries` with a ship-degraded-by-default posture per the master spec, closing taxonomy rows 12 (tbats), 13 (sktime), and 16 (pmdarima `force_all_finite`), and adding a TS smoke harness.

**Architecture:** Probe sktime/pmdarima/tbats install in a fresh `.venv-phase4` first. Tier-1 (mechanical) fixes land regardless of Tier-2 outcome: pycaret's own `force_all_finite` → `ensure_all_finite` rename, pmdarima call-site shim, statsmodels floor lift. Tier-3 (graceful-disable) wraps the tbats import. Tier-2 (sktime) is probe-driven: lift if clean, narrow API + DEGRADED if structurally broken. Verification is a tiny `tests/smoke/test_time_series.py` (mirroring Phase 3 plotting smoke) plus the pre-existing Phase 3 plotting smoke for regression.

**Tech Stack:** Python 3.12, uv (venv + pip), sktime, pmdarima, statsmodels, tbats, pycaret 3.4.0 codebase.

**Spec:** `docs/superpowers/specs/2026-05-06-phase-4-timeseries-design.md`

**Branch:** `phase-4-timeseries` (off `phase-3-plotting` HEAD `0a23a615`; rebase onto `modernize` after PR #3 merges).

**Verification floor (local):** `.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v` green or skips-only. Per-test 30 s timeout, aggregate <120 s. Plus the pre-existing Phase 3 plotting smoke (`.venv-phase3`) must not regress after rebase.

**User constraint:** TS-stack installs are heavy. The plan uses `uv` for fast resolution; if any install hangs > 5 minutes, the worker reports BLOCKED rather than waiting indefinitely. The user has authorized "ship-degraded" — install failures are an outcome, not a blocker.

---

## File Structure

**Create:**
- `.venv-phase4/` — uv-managed Python 3.12 venv (gitignored via `.gitignore`)
- `pycaret/internal/patches/pmdarima.py` — pmdarima compatibility shim (only if Tier-1 surfaces a need)
- `tests/smoke/test_time_series.py` — TS smoke harness
- `docs/superpowers/agents/ts-dev/CHARTER.md` — agent charter (mirror of plotting-dev)
- `docs/superpowers/agents/ts-dev/LOG.md` — append-only progress log
- `docs/superpowers/agents/ts-dev/DEGRADED.md` — TS visualizer/forecaster registry

**Modify:**
- `pycaret/internal/preprocess/iterative_imputer.py:185-194` — `force_all_finite` → `ensure_all_finite`
- `pycaret/containers/models/time_series.py:1141-1240` — wrap BATS/TBATS import in graceful-disable guard
- `pyproject.toml` — sktime unpin, pmdarima floor (probe-decided), statsmodels floor lift, possibly tbats removal
- `.gitignore` — append `.venv-phase4/`
- `pycaret/utils/_sklearn_compat.py` — add helper if pmdarima needs a kwarg-translating `check_array` wrapper
- `docs/superpowers/FAILURE_TAXONOMY.md` — close rows 12, 13, 16 (or mark degraded)
- `docs/superpowers/MIGRATION_BACKLOG.md` — refresh row counts at Phase 4 close

**Reference (read, don't modify):**
- `pycaret/internal/patches/yellowbrick.py` — pattern for monkey-patch via `mock.patch` at experiment init
- `pycaret/internal/patches/sklearn.py` — pattern for prefix-monkey-patch
- `pycaret/utils/_sklearn_compat.py:get_base_scorer_class` — pattern for try-import + clear-error fallback
- `tests/smoke/test_plotting.py` — pattern for the new TS smoke harness
- `docs/superpowers/agents/plotting-dev/{CHARTER,LOG,DEGRADED}.md` — patterns for ts-dev docs

---

## Task 1: Fresh `.venv-phase4` + install probe

**Files:**
- Create: `.venv-phase4/` (gitignored)
- Modify: `.gitignore`

- [ ] **Step 1: Add `.venv-phase4/` to `.gitignore`**

```bash
echo '.venv-phase4/' >> .gitignore
```

- [ ] **Step 2: Create the venv with uv**

```bash
cd /Users/uthlekan/Library/CloudStorage/Dropbox/00_ToReview/10_PluginSkills/pycaret
uv venv .venv-phase4 --python 3.12
.venv-phase4/bin/python --version
```

Expected: `Python 3.12.13` (or whatever 3.12 minor uv resolves to). If a different minor lands, fine — only the major.minor matters.

- [ ] **Step 3: Install pycaret + TS extras into the new venv (probe)**

```bash
uv pip install --python .venv-phase4/bin/python -e ".[full]" 2>&1 | tee /tmp/phase4-install.log | tail -20
```

Expected: install completes in <5 minutes. If it hangs or errors, **report BLOCKED with the last 50 lines of `/tmp/phase4-install.log`**. The user has authorized the degrade fallback if install fails.

If the install succeeds, the log file is the input for sktime/pmdarima/tbats version detection in later tasks.

- [ ] **Step 4: Capture installed versions**

```bash
.venv-phase4/bin/python -c "
import importlib
for pkg in ['sktime', 'pmdarima', 'tbats', 'statsmodels', 'sklearn', 'pandas', 'numpy']:
    try:
        m = importlib.import_module(pkg)
        print(f'{pkg:13s} {m.__version__}')
    except Exception as e:
        print(f'{pkg:13s} INSTALL_FAIL: {type(e).__name__}: {e}')
" | tee /tmp/phase4-versions.txt
```

Expected: a clean line per package with the resolved version. Any `INSTALL_FAIL` entries inform Tier-3 (e.g., tbats may legitimately fail to import under numpy 2 — that's the expected outcome).

- [ ] **Step 5: Commit `.gitignore` only**

```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
chore(ci): ignore .venv-phase4 working venv

Mirrors the prior phase-N venv ignores.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: ts-dev agent docs scaffold

**Files:**
- Create: `docs/superpowers/agents/ts-dev/CHARTER.md`
- Create: `docs/superpowers/agents/ts-dev/LOG.md`
- Create: `docs/superpowers/agents/ts-dev/DEGRADED.md`

The agent docs already exist from Phase 0 (`docs/superpowers/agents/ts-dev/`). Inspect first; if a stale charter exists, rewrite it with current spec/plan paths just like Phase 3 did for `plotting-dev`.

- [ ] **Step 1: Inspect existing ts-dev docs**

```bash
ls docs/superpowers/agents/ts-dev/ 2>&1 || echo 'directory does not exist'
cat docs/superpowers/agents/ts-dev/CHARTER.md 2>&1 | head -30
```

If files exist, treat the next steps as overwrites (same pattern Phase 3 followed).

- [ ] **Step 2: Write `CHARTER.md`**

```markdown
# Time-series Migration Dev — Charter

**Phase:** 4 (Time-series Stack)
**Branch:** `phase-4-timeseries` (off `phase-3-plotting`)
**Spec:** `docs/superpowers/specs/2026-05-06-phase-4-timeseries-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-phase-4-timeseries.md`

## Inputs
- `FAILURE_TAXONOMY.md` rows tagged `sktime | pmdarima | statsmodels | tbats` (currently 12, 13, 16; possibly 17+ from Phase 4 install probe).
- Master spec § 4 Phase 4 paragraph; § 8 risk #1 (sktime structural drift).

## Outputs
- Cherry-pickable commits on `phase-4-timeseries` (one logical change per commit).
- New `tests/smoke/test_time_series.py` (pycaret-ng infra, exempt from Gate D).
- Updated `FAILURE_TAXONOMY.md` and `MIGRATION_BACKLOG.md`.
- `DEGRADED.md` rows for tbats (definite) and any sktime narrowings (probe-driven).

## Stop criteria
- All in-scope rows closed or degraded.
- Smoke harness green locally.
- PR open from `phase-4-timeseries → modernize` with CI green.

## Out-of-scope handoffs
- New time-series estimators → not Phase 4.
- Plotly-resampler bump deferred from Phase 3 → optional in Phase 4 (only if a smoke-reachable site surfaces); otherwise Phase 5.
- Joblib `Memory.bytes_limit` (row 14) → Phase 5.

## Authority
- May add taxonomy rows. May not edit closed rows owned by other agents.
- May edit `pyproject.toml` for TS-stack dep floors only.
- May narrow pycaret's TS API (raise NotImplementedError + DEGRADED row) without further authorization, per the master spec's pre-authorization.
```

- [ ] **Step 3: Write `LOG.md`**

```markdown
# Time-series Migration Dev — Log

Append-only progress log.

## 2026-05-06 — Phase 4 kickoff
- Plan committed: `docs/superpowers/plans/2026-05-06-phase-4-timeseries.md`.
- Branch: `phase-4-timeseries` off `phase-3-plotting` HEAD `0a23a615`.
```

- [ ] **Step 4: Write `DEGRADED.md`**

```markdown
# DEGRADED.md — Time-series Forecaster/Visualizer Registry

Forecasters / visualizers that pycaret-ng explicitly disables under
modernized deps. A disabled entry raises `NotImplementedError` from its
container's `class_def` accessor (or the relevant dispatch site) with a
pointer to this file. The corresponding smoke entry is skip-marked.

## Schema

| Entry | Kind | Disabled because | Tracking | Restoration criterion |
|-------|------|-------------------|----------|------------------------|

## Rows

(none yet — populated by Tier-3 tbats degrade and any Tier-2 sktime narrowings)
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/agents/ts-dev/
git commit -m "$(cat <<'EOF'
docs(ts-dev): refresh charter, log, DEGRADED registry for Phase 4

Aligns with the Phase 3 pattern. Charter cites the new Phase 4 spec
and plan; DEGRADED.md is empty (schema only) pending tier-2/tier-3
outcomes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Smoke harness package skeleton

**Files:**
- Create: `tests/smoke/test_time_series.py` (skeleton — fixture only, no test bodies)

`tests/smoke/__init__.py` and `tests/smoke/conftest.py` already exist from Phase 3 — reused verbatim. The `matplotlib.use("Agg")` line in conftest still applies (TS uses matplotlib backends).

- [ ] **Step 1: Write the skeleton**

```python
"""Phase 4 time-series smoke harness.

Mirror of tests/smoke/test_plotting.py — minimal local-only coverage,
parametrized over a small set of forecasters. Pass condition is "no
exception raised". No image diff, no parity check, no SHAP.

Per-test timeout 30 s, aggregate target <120 s.

Run via:

    .venv-phase4/bin/python -m pytest --confcutdir=tests/smoke \\
        tests/smoke/test_time_series.py -v
"""
from __future__ import annotations

import pytest

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


@pytest.fixture(scope="module")
def ts_setup():
    # Airline passengers — 144 monthly observations, classic TS smoke test.
    data = get_data("airline", verbose=False)
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=12,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
        session_id=42,
    )
    return exp


# Skip-list mirrors docs/superpowers/agents/ts-dev/DEGRADED.md.
TS_DEGRADED: set[str] = set()

# Forecaster IDs are the keys pycaret accepts in `compare_models(include=...)`
# / `create_model(estimator=...)`. One representative per family. We
# intentionally exclude `tbats` and `bats` because Tier-3 disables them.
TS_FORECASTERS = sorted(
    [
        "naive", "snaive", "polytrend", "arima", "auto_arima",
        "exp_smooth", "ets", "theta", "stlf",
        "lr_cds_dt",  # linear regression with conditional deseasonalizer
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("forecaster", TS_FORECASTERS)
def test_forecaster_create(ts_setup, forecaster):
    if forecaster in TS_DEGRADED:
        pytest.skip(f"forecaster='{forecaster}' is degraded — see DEGRADED.md")
    model = ts_setup.create_model(forecaster, verbose=False)
    # Sanity: a fitted model has a `predict` method (or sktime equivalent).
    assert hasattr(model, "predict") or hasattr(model, "_predict")
```

- [ ] **Step 2: Verify collect-only on the new venv**

```bash
.venv-phase4/bin/python -m pytest --collect-only --confcutdir=tests/smoke tests/smoke/test_time_series.py 2>&1 | tail -15
```

Expected: 10 tests collected. If pytest reports any import error from `pycaret.time_series` or `pycaret.datasets`, that's a Tier-2 sktime structural problem — capture the traceback for Task 8 (sktime probe).

- [ ] **Step 3: Commit (skeleton only — running tests deferred)**

```bash
git add tests/smoke/test_time_series.py
git commit -m "$(cat <<'EOF'
test(smoke): scaffold tests/smoke/test_time_series.py

Mirror of Phase 3's plotting smoke. Parametrizes over one forecaster
per family on the airline dataset with fh=12. Skip-list mirrors
DEGRADED.md. Test bodies will land progressively as Tier-1/2/3 fixes
unblock specific forecasters.

This commit may include failing tests on phase-4-timeseries HEAD
until subsequent tasks land. Pycaret-ng-only infra; exempt from Gate D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Tier-1 — pycaret's own `force_all_finite` rename

**Files:**
- Modify: `pycaret/internal/preprocess/iterative_imputer.py:184-195`

Sklearn ≥1.6 deprecated `force_all_finite=` in favor of `ensure_all_finite=`. Sklearn 1.8 (latest within `<2`) removes it entirely. Pycaret's iterative imputer wrapper uses the old kwarg.

- [ ] **Step 1: Read the current call site**

```bash
sed -n '180,200p' pycaret/internal/preprocess/iterative_imputer.py
```

Expected: a block setting `force_all_finite` to `"allow-nan"` or `True` and passing it to `self._validate_data(...)`.

- [ ] **Step 2: Rewrite the kwarg name**

```python
# Replace:
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )

# With:
        if is_scalar_nan(self.missing_values):
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            ensure_all_finite=ensure_all_finite,
        )
```

- [ ] **Step 3: Verify imports still work**

```bash
.venv-phase4/bin/python -c "from pycaret.internal.preprocess.iterative_imputer import IterativeImputerWithCustomLogger; print('ok')" 2>&1 | tail -3
```

Expected: `ok`. If any other site references the renamed local variable, fix the references too. (Grep the file for `force_all_finite`; expect zero matches after the edit.)

- [ ] **Step 4: Commit**

```bash
git add pycaret/internal/preprocess/iterative_imputer.py
git commit -m "$(cat <<'EOF'
fix(sklearn): force_all_finite -> ensure_all_finite in IterativeImputer

sklearn 1.6 deprecated force_all_finite= in favor of ensure_all_finite=
on _validate_data / check_array. sklearn 1.8 removed the old kwarg
entirely. Pycaret's iterative imputer wrapper at
pycaret/internal/preprocess/iterative_imputer.py was the only direct
call site. Renaming is safe across the >=1.6 floor.

Closes the pycaret-side half of FAILURE_TAXONOMY row 16. The pmdarima
side is addressed separately in Task 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Tier-1 — pmdarima `force_all_finite` shim

**Files:**
- Possibly Create: `pycaret/internal/patches/pmdarima.py`
- Possibly Modify: `pycaret/internal/patches/__init__.py`

Pmdarima 2.x still uses the deprecated kwarg internally when calling sklearn. We can't edit pmdarima's source, so we either (a) install a pmdarima version that's been updated, or (b) monkey-patch sklearn's `check_array` to translate the kwarg back-compat.

The plan picks (b): a focused monkey-patch in `pycaret/internal/patches/pmdarima.py`, installed at module-import time (mirrors `pycaret/internal/patches/yellowbrick.py`'s pattern at the patches/__init__.py level).

- [ ] **Step 1: Check whether pmdarima still has the issue under the freshly-installed version**

```bash
.venv-phase4/bin/python -c "
import warnings
warnings.simplefilter('error')  # turn DeprecationWarnings into errors temporarily
try:
    import pmdarima as pm
    from pmdarima import auto_arima
    import numpy as np
    pm.auto_arima(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), seasonal=False, max_p=1, max_q=1, max_d=0)
    print('pmdarima OK — no force_all_finite issue')
except Exception as e:
    print(f'pmdarima FAIL: {type(e).__name__}: {e}')
" 2>&1 | tail -5
```

Expected: either `pmdarima OK` (Tier-1 done; skip the shim, proceed to Step 5) or a `TypeError: check_array() got an unexpected keyword argument 'force_all_finite'`. The error path triggers Steps 2-4.

If the install probe captured a `pmdarima INSTALL_FAIL` line earlier, this whole task collapses to: add a degrade row for pmdarima in DEGRADED.md, skip-list `arima` and `auto_arima` in the smoke harness, and proceed to Task 6.

- [ ] **Step 2: Write the patch module (only if Step 1 surfaced the kwarg error)**

```python
"""Pycaret-side shim for pmdarima's stale sklearn API usage.

pmdarima 2.x calls sklearn.utils.validation.check_array with
``force_all_finite=`` — sklearn ≥1.6 deprecated that kwarg in favor of
``ensure_all_finite=``, and sklearn ≥1.8 removed it. Pycaret cannot
edit pmdarima's source, so we monkey-patch sklearn's check_array at
import time to translate the legacy kwarg name when callers still use
it. This is import-time global; safe because the new name is also
accepted, so legitimate callers using ``ensure_all_finite=`` are
unaffected.

Activated by importing this module from pycaret/internal/patches/__init__.py.
"""
from __future__ import annotations

from sklearn.utils import validation as _sk_validation

_original_check_array = _sk_validation.check_array


def _check_array_with_legacy_kwarg(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _original_check_array(*args, **kwargs)


_sk_validation.check_array = _check_array_with_legacy_kwarg
```

- [ ] **Step 3: Wire the shim into the patches package**

Edit `pycaret/internal/patches/__init__.py`. Current contents (per Phase 3 read):

```python
import pycaret.internal.patches.matplotlib_compat  # noqa: F401
```

Append:

```python
import pycaret.internal.patches.pmdarima  # noqa: F401
```

So the file now reads:

```python
import pycaret.internal.patches.matplotlib_compat  # noqa: F401
import pycaret.internal.patches.pmdarima  # noqa: F401
```

- [ ] **Step 4: Re-run the pmdarima probe**

```bash
.venv-phase4/bin/python -c "
import pycaret  # triggers patches import
import pmdarima as pm
import numpy as np
pm.auto_arima(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), seasonal=False, max_p=1, max_q=1, max_d=0)
print('pmdarima OK with shim')
"
```

Expected: `pmdarima OK with shim`. If it still fails, the error has shifted to a different sklearn API — capture and decide degrade vs further shim in a follow-up commit.

- [ ] **Step 5: Commit (only if Steps 2-4 ran)**

```bash
git add pycaret/internal/patches/pmdarima.py pycaret/internal/patches/__init__.py
git commit -m "$(cat <<'EOF'
fix(ts): pmdarima sklearn force_all_finite shim

pmdarima 2.x still passes force_all_finite= to sklearn.check_array,
which sklearn 1.8 removed (renamed to ensure_all_finite= since 1.6).
Until pmdarima releases a sklearn-1.6+-compatible version, monkey-patch
sklearn.utils.validation.check_array at pycaret import time to
translate the legacy kwarg.

Closes the pmdarima side of FAILURE_TAXONOMY row 16.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Tier-1 — statsmodels floor lift

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current floor and the installed version**

```bash
grep 'statsmodels' pyproject.toml
.venv-phase4/bin/python -c "import statsmodels; print(statsmodels.__version__)"
```

Expected: pyproject reads `statsmodels>=0.12.1`; installed version is whatever uv resolved (likely 0.14.x).

- [ ] **Step 2: Lift the floor to the resolved minor**

Pick the minor version from the installed value (e.g., if installed is `0.14.6`, lift to `>=0.14`). Write the change as a `<` cap on the next major (e.g., `>=0.14,<1`).

Edit `pyproject.toml`. Replace:

```toml
"statsmodels>=0.12.1",
```

With (substituting the actual minor):

```toml
"statsmodels>=0.14,<1",
```

- [ ] **Step 3: Re-resolve and verify import**

```bash
uv pip install --python .venv-phase4/bin/python -e ".[full]" --quiet 2>&1 | tail -5
.venv-phase4/bin/python -c "import statsmodels; print('ok', statsmodels.__version__)"
```

Expected: `ok 0.14.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
feat(ts): lift statsmodels floor to >=0.14,<1

Previous floor (>=0.12.1) predated pandas 2.x. The installed version
under the modernized base is 0.14.x; lift the floor to match. Cap on
the next major to allow patch/minor drift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Tier-3 — tbats graceful-disable

**Files:**
- Modify: `pycaret/containers/models/time_series.py:1141-1240` (BATSContainer + TBATSContainer)

The master spec § 8 risk #2 names tbats as numpy-1-only and unmaintained. The expected outcome is graceful-disable: the import is wrapped, and instantiation raises a clear `NotImplementedError`.

- [ ] **Step 1: Inspect both containers**

```bash
sed -n '1130,1260p' pycaret/containers/models/time_series.py
```

Expected: two classes, `BATSContainer` and `TBATSContainer`, each with an `__init__` that does `from sktime.forecasting.bats import BATS` (or tbats), creates a dummy, and registers a container.

- [ ] **Step 2: Wrap the imports with try/except + degrade marker**

For both containers, replace the top-of-`__init__` import with:

```python
        try:
            from sktime.forecasting.bats import BATS  # type: ignore
            BATS()  # raise early on numpy-2 incompat
            self._tbats_disabled = False
            class_def = BATS
        except (ImportError, AttributeError, TypeError) as _exc:
            self._tbats_disabled = True
            self._tbats_disable_reason = f"{type(_exc).__name__}: {_exc}"
            class_def = _DegradedBATSFamily
```

(For `TBATSContainer`, swap `BATS` → `TBATS` and `bats` → `tbats` in the import path.)

Add a stub class above both containers (around line 1140):

```python
class _DegradedBATSFamily:
    """Stub for tbats / BATS forecasters under modernized deps.

    tbats is unmaintained and numpy-1-only; under numpy ≥2 the import
    or instantiation fails. Per the master spec's Phase 4 fallback
    policy, instantiating a BATS / TBATS forecaster raises NotImplementedError
    pointing to DEGRADED.md.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "BATS / TBATS forecasters are temporarily disabled in pycaret-ng "
            "under numpy ≥2: tbats is unmaintained and numpy-1-only. Use "
            "auto_arima or exp_smooth instead. Tracked in "
            "docs/superpowers/agents/ts-dev/DEGRADED.md."
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "BATS / TBATS unavailable — see DEGRADED.md."
        )
```

Plus update each container's `class_def` registration to use the local variable picked above (not the literal `BATS` / `TBATS`).

- [ ] **Step 3: Verify the smoke harness's TS_FORECASTERS list does not include `tbats` / `bats`**

```bash
grep -E '"(tbats|bats)"' tests/smoke/test_time_series.py
```

Expected: no matches (the skeleton already excludes them per Task 3).

- [ ] **Step 4: Quick import-only verification**

```bash
.venv-phase4/bin/python -c "
import pycaret.containers.models.time_series as ts
# Force the container module to load (this is what setup() does)
print('module loaded ok')
" 2>&1 | tail -3
```

Expected: `module loaded ok`. The graceful-disable means the import succeeds even without tbats installed.

- [ ] **Step 5: Add the DEGRADED row**

Append to `docs/superpowers/agents/ts-dev/DEGRADED.md`:

```markdown
| `bats` | forecaster | tbats library is numpy-1-only and unmaintained; import fails under numpy ≥2 | FAILURE_TAXONOMY row 12 | tbats releases a numpy-2 compatible version OR pycaret vendors a successor (sktime's BATS shim) |
| `tbats` | forecaster | same root cause as bats | FAILURE_TAXONOMY row 12 | same as bats |
```

- [ ] **Step 6: Commit**

```bash
git add pycaret/containers/models/time_series.py docs/superpowers/agents/ts-dev/DEGRADED.md
git commit -m "$(cat <<'EOF'
fix(ts): graceful-disable BATS/TBATS forecasters under numpy >=2

tbats is unmaintained and numpy-1-only (per master spec § 8 risk #2).
Under numpy ≥2 the import / instantiation fails. Per the spec's
fallback policy, both BATSContainer and TBATSContainer now register a
_DegradedBATSFamily stub when the real import fails. Instantiation raises
NotImplementedError pointing to DEGRADED.md.

Closes FAILURE_TAXONOMY row 12.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Tier-2 — sktime probe + decision

**Files:**
- Modify: `pyproject.toml` (sktime pin)
- Possibly Modify: TS source files for narrowing
- Modify: `docs/superpowers/agents/ts-dev/DEGRADED.md`

This is the high-risk task. The plan picks a path based on what the probe surfaces.

- [ ] **Step 1: Read the current pin**

```bash
grep 'sktime' pyproject.toml
```

Expected: `"sktime>=0.31.0,<0.31.1",`

- [ ] **Step 2: Probe import + smoke under the existing pin**

```bash
.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v --tb=line --timeout=30 2>&1 | tee /tmp/phase4-ts-baseline.txt | tail -20
```

This is the **baseline pass** — it tells you what works under the *pinned* sktime. Capture failures.

If the smoke is fully green here: the pin is fine, sktime drift didn't break us. Proceed to Step 5 (decision = lift).

If failures exist: they're inputs to taxonomy and may already justify narrowing.

- [ ] **Step 3: Lift the sktime pin and re-probe**

Edit `pyproject.toml`:

```toml
"sktime>=0.31.0,<0.31.1",
```

Lift to (substituting current latest minor at session time):

```toml
"sktime>=0.31",
```

Re-resolve:

```bash
uv pip install --python .venv-phase4/bin/python -e ".[full]" --quiet 2>&1 | tail -5
.venv-phase4/bin/python -c "import sktime; print(sktime.__version__)"
```

Then re-run the smoke:

```bash
.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v --tb=line --timeout=30 2>&1 | tee /tmp/phase4-ts-unpinned.txt | tail -20
```

- [ ] **Step 4: Decision — lift, narrow, or stay pinned**

Compare `/tmp/phase4-ts-baseline.txt` vs `/tmp/phase4-ts-unpinned.txt`:

- **Lift wins:** unpinned smoke is no worse than pinned. Keep the lifted pin. Closes row 13.
- **Narrow wins:** unpinned smoke surfaces ≤ 3 new failures and they're isolatable (specific forecasters or specific entry points in pycaret's `TSForecastingExperiment`). For each failure: add to `TS_DEGRADED` set in the smoke, add a DEGRADED.md row, raise `NotImplementedError` at the corresponding pycaret entry point. Keep the lifted pin.
- **Roll back:** unpinned smoke surfaces ≥ 4 new failures or surfaces failures that span pycaret's TS API broadly (e.g., `setup()` itself fails). Revert the pin lift, keep the bound, and document in `LOG.md` that sktime structural drift exceeded our cheap-fix budget. Open row 13 with a "deferred to next session" note.

The user has authorized any of these three outcomes.

- [ ] **Step 5: For each forecaster narrowed, add a DEGRADED.md row + skip-list update**

Pattern (per failed forecaster):

```markdown
| `<forecaster_id>` | forecaster | sktime <new_version> changed <specific API>; pycaret's wrapper at <path:line> calls the old shape | FAILURE_TAXONOMY row 13 / sub-row | sktime stabilizes the API OR pycaret rewrites the wrapper |
```

Update `tests/smoke/test_time_series.py`:

```python
TS_DEGRADED: set[str] = {
    "<forecaster_id_1>",
    "<forecaster_id_2>",
}
```

Each NotImplementedError site (in `pycaret/containers/models/time_series.py` or the relevant container) follows the same pattern as Task 7's `_DegradedBATSFamily`.

- [ ] **Step 6: Verify smoke green again**

```bash
.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v
```

Expected: green or skips-only.

- [ ] **Step 7: Commit (whichever decision was taken)**

For lift:

```bash
git add pyproject.toml
git commit -m "feat(ts): unpin sktime to >=0.31 — no smoke regressions

Closes FAILURE_TAXONOMY row 13.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

For narrow:

```bash
git add pyproject.toml pycaret/containers/models/time_series.py docs/superpowers/agents/ts-dev/DEGRADED.md tests/smoke/test_time_series.py
git commit -m "fix(ts): narrow TS API for sktime drift — N forecasters degraded

sktime <new_version> introduced API changes that pycaret's wrappers
do not yet adapt to. Per spec policy (b), the affected forecasters
raise NotImplementedError pointing to DEGRADED.md. Smoke skip-listed.

Closes FAILURE_TAXONOMY row 13 (degraded).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

For rollback:

```bash
git checkout pyproject.toml  # revert
# Edit LOG.md to record the deferral
git add docs/superpowers/agents/ts-dev/LOG.md
git commit -m "docs(ts-dev): defer sktime unpin — drift exceeds Phase 4 budget

Phase 4 retains the existing >=0.31.0,<0.31.1 pin. Row 13 stays open
with reproduction notes captured in LOG.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: TS smoke harness — final verification

**Files:**
- Modify: `tests/smoke/test_time_series.py` (add an additional `predict` smoke)

After Tasks 5-8, the smoke should pass for the non-degraded forecasters. Add a second test that exercises the predict path so we don't accidentally ship a setup-only-green harness.

- [ ] **Step 1: Append a predict smoke**

```python
@pytest.mark.timeout(30)
@pytest.mark.parametrize("forecaster", TS_FORECASTERS)
def test_forecaster_predict(ts_setup, forecaster):
    if forecaster in TS_DEGRADED:
        pytest.skip(f"forecaster='{forecaster}' is degraded — see DEGRADED.md")
    model = ts_setup.create_model(forecaster, verbose=False)
    preds = ts_setup.predict_model(model, verbose=False)
    assert preds is not None
    assert len(preds) > 0
```

- [ ] **Step 2: Run the full smoke**

```bash
.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v --timeout=30 2>&1 | tee /tmp/phase4-smoke-final.txt | tail -15
```

Expected: 2 × N tests, of which the degraded entries skip. Aggregate <120 s.

- [ ] **Step 3: Verify the Phase 3 plotting smoke still passes (rebase regression check)**

```bash
.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v --timeout=30 2>&1 | tail -5
```

Expected: same 38 passed, 3 skipped, 0 failed result as Phase 3 close. Phase 4 is on top of Phase 3, not a refactor of it.

- [ ] **Step 4: Commit**

```bash
git add tests/smoke/test_time_series.py
git commit -m "test(smoke): add predict-path smoke for time-series forecasters

Setup-only smoke wasn't exercising the predict pipeline. Adds
test_forecaster_predict that runs predict_model on each non-degraded
forecaster. Aggregate budget unchanged (still <120 s).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Taxonomy + backlog refresh

**Files:**
- Modify: `docs/superpowers/FAILURE_TAXONOMY.md`
- Modify: `docs/superpowers/MIGRATION_BACKLOG.md`
- Modify: `docs/superpowers/agents/ts-dev/LOG.md`

- [ ] **Step 1: Update each row in FAILURE_TAXONOMY.md**

For each of rows 12, 13, 16:

- Change `Status` from `open` (or `degraded`) to either `closed` (if a real fix landed) or `degraded` (if the spec's fallback policy applies).
- Append the closing/degrading SHA to `Notes` in the form `Closed by <sha>` or `Degraded in <sha> — see DEGRADED.md`.

- [ ] **Step 2: Refresh `MIGRATION_BACKLOG.md` row counts**

Replace the ts-dev line in the row-counts table with the post-Phase-4 reality. Match the formatting Phase 3 used for plotting-dev.

- [ ] **Step 3: Append a Phase 4 close LOG entry**

```markdown
## 2026-05-06 — Phase 4 close
- pmdarima force_all_finite shim landed at `<sha-of-task-5>`.
- pycaret's own force_all_finite call site fixed at `<sha-of-task-4>`.
- statsmodels floor lifted at `<sha-of-task-6>`.
- BATS/TBATS graceful-disable at `<sha-of-task-7>`. Closes row 12.
- sktime decision: <lift | narrow | rollback> at `<sha-of-task-8>`.
- TS smoke result: <N passed, M skipped, 0 failed in T s>.
- Phase 3 plotting smoke regression check: 38 passed, 3 skipped, 0 failed (unchanged).
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md docs/superpowers/MIGRATION_BACKLOG.md docs/superpowers/agents/ts-dev/LOG.md
git commit -m "docs(taxonomy,backlog,ts-dev): close rows 12/13/16 + Phase 4 closure

Phase 4 closure pass:
  - rows 12 (tbats), 16 (pmdarima force_all_finite) marked closed/degraded
    with closing SHAs.
  - row 13 (sktime) status reflects the Tier-2 decision.
  - MIGRATION_BACKLOG ts-dev counts refreshed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Push + open PR

**Files:** none (process step)

- [ ] **Step 1: Cherry-pick scope sanity check**

For each `fix(ts)` / `feat(ts)` / `chore(ts)` commit on this branch since `phase-3-plotting`:

```bash
for sha in $(git log --reverse --pretty=format:'%H' phase-3-plotting..HEAD | xargs -I{} sh -c 'git log -1 --format="%H %s" {}' | grep -E '(fix|feat|chore)\(ts\)' | awk "{print \$1}"); do
  echo "-- $sha --"
  git show --name-only --pretty=format: $sha | grep -v '^$'
done
```

Verify each touches only `pycaret/`, `pyproject.toml`, or `.gitignore`. Files like `tests/smoke/`, `docs/superpowers/` should not appear in `fix(ts)` commits — they belong in `test(smoke)` and `docs(*)` respectively. If a commit mixes scopes, note in the PR body (same discipline note Phase 3 made).

- [ ] **Step 2: Push the branch**

```bash
git push -u origin phase-4-timeseries
```

- [ ] **Step 3: Open the PR via gh**

Write the body to a temporary file inside the repo's `.git/` (gitignored, won't be staged):

```bash
cat > .git/PR_BODY.md <<'EOF'
## Summary

Modernizes the time-series stack on top of Phases 1-3. Adopts a
ship-degraded-by-default posture per the master spec's pre-authorized
fallback policy. Closes rows 12 (tbats — degraded), 16 (pmdarima
force_all_finite shim), and either closes or degrades row 13 (sktime
unpin decision).

**Spec:** `docs/superpowers/specs/2026-05-06-phase-4-timeseries-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-phase-4-timeseries.md`

## Workstreams

- **Tier-1 (mechanical):** pycaret's own `force_all_finite` →
  `ensure_all_finite` (Task 4); pmdarima sklearn-kwarg shim (Task 5);
  statsmodels floor lift (Task 6).
- **Tier-2 (probe-driven):** sktime unpin decision (Task 8). Outcome:
  <lift | narrow | rollback>. <One-line summary of failures handled>.
- **Tier-3 (graceful-disable):** BATS / TBATS forecaster containers
  register a `_DegradedBATSFamily` stub when the real import fails (Task 7).
  Instantiation raises NotImplementedError pointing to DEGRADED.md.
- **Smoke:** new `tests/smoke/test_time_series.py` parametrizes one
  forecaster per family on the airline dataset. Setup + predict paths.
- **Docs:** ts-dev CHARTER / LOG / DEGRADED scaffolded; FAILURE_TAXONOMY
  rows 12, 13, 16 closed or degraded; MIGRATION_BACKLOG ts-dev counts
  refreshed.

## Verification

- TS smoke (Phase 4 venv): `<N passed, M skipped, 0 failed in T s>`.
- Plotting smoke (Phase 3 venv) regression check: 38 passed, 3 skipped,
  0 failed (unchanged).
- CI matrix: see Actions tab.

## Test plan

- [x] TS smoke green locally on `phase-4-timeseries` HEAD.
- [ ] CI matrix `{3.11, 3.12, 3.13} × {linux, macos}` green.
- [x] All `fix(ts)` / `feat(ts)` commits touch only upstream-relevant paths.
- [x] DEGRADED.md rows have restoration criteria.
- [x] Taxonomy rows 12, 13, 16 updated with closing SHAs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

Then open the PR with explicit `--repo`:

```bash
gh pr create --repo olaTechie/pycaret \
  --base modernize \
  --head olaTechie:phase-4-timeseries \
  --title "Phase 4 — Time-series Stack Modernization" \
  --body-file .git/PR_BODY.md
```

If the gh remote-default issue from Phase 3 recurs (it defaults to upstream), the `--repo olaTechie/pycaret` flag forces the right target.

- [ ] **Step 4: Cleanup**

```bash
rm .git/PR_BODY.md
```

- [ ] **Step 5: Verify PR opened**

```bash
gh pr view --repo olaTechie/pycaret <pr-number>
```

The PR number is printed by `gh pr create`. Capture it for the LOG.

- [ ] **Step 6: Final LOG entry**

Append to `docs/superpowers/agents/ts-dev/LOG.md`:

```markdown
## 2026-05-06 — Phase 4 PR open
- PR: https://github.com/olaTechie/pycaret/pull/<n>
- Awaiting CI matrix.
- Next: Phase 5 (release engineering) once CI green.
```

Commit:

```bash
git add docs/superpowers/agents/ts-dev/LOG.md
git commit -m "docs(ts-dev): record Phase 4 PR URL

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Then push the new commit:

```bash
git push
```

---

## Self-Review Checklist

After all tasks, verify:
- [ ] `.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v` is green or skips-only.
- [ ] `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` still 38/3/0 (Phase 3 regression check).
- [ ] FAILURE_TAXONOMY rows 12, 13, 16 have either `closed` or `degraded` status with closing SHAs in Notes.
- [ ] DEGRADED.md has at least the BATS/TBATS rows; Tier-2 narrowings if any.
- [ ] No `fix(ts)` commit touches `tests/smoke/`, `docs/superpowers/`, or other pycaret-ng-only paths.
- [ ] Aggregate TS smoke wall-clock <120 s.
- [ ] Plotting smoke wall-clock unchanged (~12 s).

If any item fails, that's not Phase 4 done — fix before merging.
