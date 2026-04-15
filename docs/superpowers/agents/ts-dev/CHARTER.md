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
