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
