# Time-series Migration Dev — Charter

**Phase:** 4 (Time-series Stack)
**Branch:** `phase-4-timeseries` (off `phase-3-plotting`)
**Spec:** `docs/superpowers/specs/2026-05-06-phase-4-timeseries-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-phase-4-timeseries.md`

## Inputs
- `FAILURE_TAXONOMY.md` rows tagged `sktime | pmdarima | statsmodels | tbats` (currently 12, 13, 16; possibly more from Phase 4 install probe).
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
- May narrow pycaret's TS API (raise NotImplementedError + DEGRADED row) without further authorization, per the master spec's pre-authorization for Phase 4.
