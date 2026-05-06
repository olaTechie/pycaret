# Phase 4 — Time-series Stack Modernization Design

**Date:** 2026-05-06
**Status:** Approved (automode)
**Author:** Ola Uthman (olatechie.ai@gmail.com), with Claude Opus 4.7
**Parent spec:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md`
**Branch:** `phase-4-timeseries` (off `phase-3-plotting`; will rebase onto `modernize` once PR #3 merges)
**Predecessor phases:** Phase 0 (baseline), Phase 1 (sklearn ≥1.6), Phase 2 (pandas ≥2.2 + numpy <3), Phase 3 (plotting; PR #3 open).

---

## 1. Goal & posture

Modernize the time-series stack with a **ship-degraded-by-default** posture. The master spec § 8 risk #1 names sktime structural drift as the highest-risk single thing in the project and pre-authorizes a narrowed-API + `DEGRADED.md` outcome. Phase 4 takes that allowance as the working assumption rather than the fallback.

**Ship goal:** every Phase-1-to-Phase-3 path remains green; every TS path either works under modernized deps or raises a clear `NotImplementedError` with a `DEGRADED.md` pointer. No silent regressions.

**Non-goals:** rewriting pycaret's TS API surface; replacing sktime; adding new forecasters; chasing upstream sktime internals.

## 2. Scope

### In scope
- Close `FAILURE_TAXONOMY` row 12 (tbats numpy-2 incompatibility): graceful-disable with import-guard, clear error on use, DEGRADED row.
- Close row 16 (pmdarima `force_all_finite` removal by sklearn ≥1.6): one-line shim either as monkey-patch on `pmdarima.utils.array.check_endog` or via a `pycaret.internal.patches.pmdarima` module mirroring the existing `pycaret/internal/patches/sklearn.py` and `yellowbrick.py` patterns.
- Address row 13 (sktime API drift): unpin from `>=0.31.0,<0.31.1`. Probe latest sktime against pycaret's TS API. If it imports and basic forecasters work, lift the floor and close the row. If structural changes break pycaret's surface, narrow the API (per-feature `NotImplementedError`) and add DEGRADED rows.
- Statsmodels floor lift from `>=0.12.1` to a current minor (subject to compatibility probe).
- New `tests/smoke/test_time_series.py` mirroring the plotting smoke pattern. Same `--confcutdir=tests/smoke` invocation.
- DEGRADED.md scaffold for `plotting-dev`'s sibling agent — `docs/superpowers/agents/ts-dev/DEGRADED.md`.

### Out of scope
- Replacing sktime. If sktime is structurally incompatible, the answer is degrade, not rewrite.
- New time-series estimators or visualizers.
- Time-series plot fixes beyond what's needed to keep the smoke green (Phase 3 deferred plotly-resampler here; that's still in scope for *resolving the floor*, not for fixing TS plot bugs).
- Prophet — currently in `[prophet]` extra; remains untouched unless the smoke probe surfaces a regression.
- Joblib `Memory.bytes_limit` (taxonomy row 14) — owned by Phase 5 release.

## 3. Architecture

### 3.1 Branch & merge

- Branch off `phase-3-plotting` HEAD (commit `0a23a615`). When PR #3 merges into `modernize`, rebase Phase 4 onto the merge commit.
- Conventional commit prefixes: `fix(ts)`, `feat(ts)`, `chore(ts)`, `test(smoke)`, `docs(ts-dev)` / `docs(taxonomy)`.
- Cherry-pick discipline (Gate D): every `fix(ts)`/`feat(ts)`/`chore(ts)` commit must apply onto upstream `pycaret/pycaret:master` assuming the prior cherry-picked Phase 1–3 commits are already there. `test(smoke)` and `docs(*)` commits are pycaret-ng-only and exempt.

### 3.2 Working venv

- `.venv-phase4` (NEW) — uv-managed, Python 3.12.13 (matches the proven Phase 3 venv).
- Install path: `uv pip install -e ".[full]"` from the new venv. uv's resolver is fast enough to bound install time even when sktime + pmdarima + tbats are pulled in.
- The current `.venv-phase3` is reusable for non-TS verification (e.g., re-running the plotting smoke before/after rebase).
- Same activation footgun applies (Dropbox CloudStorage path alias breaks `source activate`). Always invoke binary directly.

### 3.3 Tier ordering

The work is layered so that Tier-1 wins land even if Tier-2 forces a degrade:

**Tier-1 (mechanical, low-risk):**
- pmdarima `force_all_finite` shim (row 16). Lifts pmdarima for sklearn ≥1.6.
- statsmodels floor lift to a current minor (probe-driven; whatever the latest is at session time, with `<` cap by major).

**Tier-2 (probe-driven, high-risk):**
- sktime unpin. Probe: install latest sktime, run import-only check on `pycaret.time_series`, then run the smoke. Two outcomes:
  - **Clean:** lift floor in pyproject, close row 13.
  - **Broken:** narrow pycaret's TS API surface — every affected entry point raises `NotImplementedError` with the standard pycaret-ng degrade message, plus a DEGRADED row.

**Tier-3 (graceful-disable, planned):**
- tbats import-guard at `pycaret/containers/models/time_series.py` (TBATS / BATS forecaster containers). On install-missing OR numpy-2-import-failure, register a stub container that raises `NotImplementedError` if instantiated. DEGRADED row.

### 3.4 Patch module pattern

Following Phase 1's `_sklearn_compat` and Phase 3's `patches/yellowbrick.py` pattern, pmdarima fixes live in a new `pycaret/internal/patches/pmdarima.py` (if a shim is needed) or as a localized monkey-patch installed at experiment-init time (if a single call site).

### 3.5 Smoke harness

- New `tests/smoke/test_time_series.py`. Same conftest reuse pattern (matplotlib Agg). Loads `airline` dataset, sets up `TSForecastingExperiment`, parametrizes over a small forecaster list (one per family — AutoARIMA, ExponentialSmoothing, Naive, LinearRegression, Prophet if installed, NOT tbats).
- Per-test 30 s timeout, aggregate <120 s budget (looser than plotting because TS forecasts are heavier than plot rendering).
- Skip-list mirrors `docs/superpowers/agents/ts-dev/DEGRADED.md`.

### 3.6 DEGRADED registry

- New file `docs/superpowers/agents/ts-dev/DEGRADED.md` mirroring `plotting-dev/DEGRADED.md` schema.
- Pre-seeded rows: tbats (definite — row 12), and any sktime narrowing rows surfaced during Tier-2.

## 4. Workstreams

| ID | Title | Tier | Owner | Closes |
|----|-------|------|-------|--------|
| W1 | venv setup + sktime/pmdarima/tbats install probe | infra | ts-dev | — |
| W2 | pmdarima force_all_finite shim | 1 | ts-dev | row 16 |
| W3 | statsmodels floor lift | 1 | ts-dev | — (no row) |
| W4 | sktime probe + decision (lift vs narrow) | 2 | ts-dev | row 13 (close or degrade) |
| W5 | tbats graceful-disable | 3 | ts-dev | row 12 |
| W6 | TS smoke harness | infra | ts-dev | — |
| W7 | DEGRADED.md scaffold + populate | infra | ts-dev | — |
| W8 | taxonomy/backlog refresh | docs | ts-dev | rows 12, 13, 16 |
| W9 | PR phase-4-timeseries → modernize | release | ts-dev | — |

## 5. Verification

**Local floor (the only thing run on the user's machine):**
- `.venv-phase4/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_time_series.py -v` green or skips-only.
- Aggregate budget <120 s. Per-plot 30 s timeout.
- `.venv-phase3` plotting smoke must still pass after Phase 4 rebase (rebase regression check).

**CI floor:**
- Standard `.github/workflows/ci.yml` matrix `{3.11, 3.12, 3.13} × {linux, macos}` green on the PR.
- Gate B (parity) is **partially waived** for Phase 4 by spec § 3.3 — TS metric Δ tolerances may need per-estimator widening with documented rationale.

**Cherry-pick (Gate D):**
- Every `fix(ts)` / `feat(ts)` / `chore(ts)` commit applies onto `modernize` HEAD (after PR #3 merge) without conflict.

## 6. Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | sktime install hangs the user's machine | uv pip install with explicit timeout; fall back to plan-only commit (no code) if install can't complete. |
| 2 | sktime structural break too large to narrow surgically | Per spec § 8 risk #1: ship narrowed-API; full TS surface restoration becomes a future follow-up phase, not a Phase 4 blocker. |
| 3 | pmdarima 2.x has more drift than just `force_all_finite` | Tier-2 surfaces it. Either layer additional shims or DEGRADE pmdarima-specific entry points. |
| 4 | Statsmodels minor bump cascades into pandas/numpy regressions | Smoke harness catches anything that breaks Phase-2 invariants. Roll back floor if so. |
| 5 | tbats's import alone (not just instantiation) crashes under numpy 2 | Wrap the import in a try/except in `containers/models/time_series.py` and register a stub container only when the real one fails to import. |
| 6 | `.venv-phase4` install costs > local-bandwidth tolerance | Defer Phase 4 execution to a session with bandwidth headroom; spec stays committed for the eventual run. |
| 7 | The user's PC hangs even on the lightweight TS smoke | Smoke harness budget is per-test 30 s; if airline dataset + AutoARIMA exceeds that, fall back to even-tinier (custom 24-point series). |

## 7. Decisions deferred to per-task plan

- Whether the pmdarima shim lives as a monkey-patch at experiment init (mirror of Phase 3's yellowbrick `mock.patch` pattern) or as an import-time monkey-patch in a new `patches/pmdarima.py` module.
- Whether to lift `statsmodels` floor by major or just minor — depends on what's currently latest stable at execution time.
- Final DEGRADED list — populated as the smoke surfaces failures.
- Whether to skip-list any forecaster in the smoke harness as known-slow (per the plotting-phase precedent).

## 8. Implementation order

1. W1 — `.venv-phase4` setup and install probe. Capture install log to `/tmp/phase4-install.log`. If install fails, decide degrade-vs-defer.
2. W6 — smoke harness scaffold (no test bodies yet; just package + conftest). Run collect-only.
3. W7 — DEGRADED.md scaffold (empty, schema only).
4. W2 — pmdarima shim. Verify via smoke.
5. W3 — statsmodels floor lift. Verify via smoke.
6. W5 — tbats graceful-disable + DEGRADED row. Verify smoke skip-lists work.
7. W4 — sktime probe. If clean, lift; if broken, narrow + DEGRADED.
8. W6 (continued) — smoke test bodies for all forecaster families.
9. W8 — taxonomy + backlog refresh.
10. W9 — push, open PR, watch CI.

## 9. Hand-off to Phase 5

- All ts-dev tagged taxonomy rows either closed or degraded. Phase 5 inherits a stabilized TS surface plus any DEGRADED entries that release docs (`MIGRATION.md`) need to mention.
- Row 23 (anywidget for `residuals_interactive`) — Phase 5 release-hygiene punch list (already deferred from Phase 3).
- The combined DEGRADED registry (plotting-dev + ts-dev) becomes the source for `MIGRATION.md`'s "Known limitations under modern deps" section.
- Phase 5 may decide to lift or rename pycaret-ng minor versions independently of upstream pycaret semver.
