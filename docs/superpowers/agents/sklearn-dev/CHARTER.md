# sklearn Migration Dev — Charter

**Role:** Fix all `sklearn`-tagged failures to make PyCaret work on scikit-learn ≥ 1.5 while satisfying gates A, B, C, D.

**Phase:** 1.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = sklearn-dev`.
- Branch: `phase-1-sklearn` (branch off `modernize`).
- Parity harness: `tests/parity/`.

**Outputs:**
- Commits on `phase-1-sklearn`, one logical change per commit.
- Each commit message: `fix(sklearn): <short description>` referencing the taxonomy row ID(s) closed.
- PR opened to `modernize` when all sklearn rows are closed.

**Stop conditions:**
- All `sklearn`-tagged rows marked closed in the taxonomy.
- `pytest tests/` green in the CI matrix.
- `pytest tests/parity/` within tolerance (metric Δ < 1e-4, rank-correlation > 0.999).
- Every commit passes `git cherry-pick` dry-run onto `upstream/master` (gate D).

**Out-of-scope:**
- Pandas/numpy dtype fixes (hand to pandas-dev).
- Plotting stack fixes (hand to plotting-dev).
- Time-series (hand to ts-dev).
- If an sklearn fix requires touching pandas/plotting/ts code, flag in the PR description and open a handoff issue.

**Handoff protocol:**
- When blocked on a cross-cutting dep, append a note to the taxonomy row with `HANDOFF: <agent>` and move on.
- Orchestrator merges `phase-1-sklearn` → `modernize` only after QA signs off.
