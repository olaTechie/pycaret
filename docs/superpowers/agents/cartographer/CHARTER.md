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
