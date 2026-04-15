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
<!-- Cartographer appends rows here -->

## Completion checklist (Phase 0)

- [ ] Every distinct error signature has a row.
- [ ] No row has `TBD` owner.
- [ ] Row count ≥ distinct-test-failure count from the raw pytest run.
- [ ] Cartographer LOG.md updated with the `pip freeze` used.
