# pandas / numpy Migration Dev — Charter

**Role:** Fix all `pandas`- and `numpy`-tagged failures for pandas ≥ 2.2 and numpy ≥ 2.0 compatibility while satisfying gates A, B, C, D.

**Phase:** 2.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = pandas-dev`.
- Branch: `phase-2-pandas` (branch off `modernize` after Phase 1 merge).
- Parity harness: `tests/parity/`.

**Outputs:**
- Commits on `phase-2-pandas`, one logical change per commit.
- Each commit message: `fix(pandas): ...` or `fix(numpy): ...` referencing the taxonomy row ID(s) closed.
- PR opened to `modernize` when all pandas/numpy rows are closed.

**Stop conditions:**
- All `pandas`- and `numpy`-tagged rows closed.
- `pytest tests/` green in CI matrix.
- `pytest tests/parity/` within tolerance.
- Every commit passes `git cherry-pick` dry-run onto `upstream/master`.

**Out-of-scope:**
- sklearn fixes (hand to sklearn-dev).
- Plotting fixes (hand to plotting-dev).
- Time-series fixes (hand to ts-dev).

**Handoff protocol:** Same as sklearn-dev.
