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
