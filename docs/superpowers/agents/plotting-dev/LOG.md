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
