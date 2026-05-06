# Plotting Migration Dev — Log

Append-only progress log.

## 2026-05-06 — Phase 3 kickoff
- Plan committed: `docs/superpowers/plans/2026-05-06-phase-3-plotting.md`.

## 2026-05-06 — Venv invocation pattern locked
- `.venv-phase3` is Python 3.12.13 (uv-managed). `source .venv-phase3/bin/activate` is broken because the activate script bakes in an out-of-date Dropbox CloudStorage path alias (`00Todo/00_ToReview` vs `00_ToReview`), causing `which python` to fall back to anaconda's 3.13 — which trips pycaret's `__init__.py` version guard (taxonomy row 1, owned by Phase 5/release).
- Canonical local invocation: `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/...`. `--confcutdir` skips the root `tests/conftest.py` which eagerly imports pycaret.time_series → sktime (not yet installed; Phase 4 owns).
- Plan updated to use the canonical form throughout. CI uses its own venv setup and is unaffected.

## 2026-05-06 — Tasks 1–2 landed
- Task 1: agent docs scaffold (`66deab1e`).
- Task 2: `tests/smoke/` package + conftest (`5378c571`). Collect-only verified: `--confcutdir=tests/smoke` bypasses sktime-laden root conftest; pytest 9.0.3 sees 0 tests (correct) and exits cleanly.

## 2026-05-06 — Phase 3 W1 (yellowbrick) closed
- Smoke harness across clf/reg/clu (`10901cd9`).
- Empirical taxonomy rows 20–23 (`5eefefc0`).
- Yellowbrick role-detection delegated to sklearn (`7017c0a3`); patches extended to threshold + importances consumers (`98bcfa91`). Closes rows 18, 21, 22.
- Schemdraw 0.16+ `Drawing.draw(canvas=)` (`aa22f038`). Closes row 20.
- Two degraded visualizers (`58ae9a2b`): clf `error` (yellowbrick ClassPredictionError 3-tuple unpack against sklearn ≥1.6) and clu `distance` (yellowbrick InterclusterDistance using removed `np.percentile(interpolation=)`). Rows 24, 25.
- Row 23 (`residuals_interactive` needs `anywidget`) remains open — deferred to Phase 5 release-hygiene punch list.

## 2026-05-06 — Phase 3 verification floor
- `.venv-phase3/bin/python -m pytest --confcutdir=tests/smoke tests/smoke/test_plotting.py -v` → **38 passed, 3 skipped (degraded), 0 failed in 11.58s**. Well within the 90s budget.
- Out-of-scope deferrals: full `pytest tests/` matrix (Gate A) is CI-only; tutorial smoke (Gate C) deferred until pyproject `[full]` install is repeatable. Gate B is waived for Phase 3 per master spec.
