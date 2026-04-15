# Plotting Migration Dev — Charter

**Role:** Fix all plotting-stack failures for matplotlib ≥ 3.8, yellowbrick, schemdraw ≥ 0.16, plotly-resampler latest while satisfying gates A, C, D. Gate B (numerical parity) is WAIVED for this phase because visual output is not numerically comparable.

**Phase:** 3.

**Inputs:**
- FAILURE_TAXONOMY.md rows with `Owner agent = plotting-dev`.
- Branch: `phase-3-plotting` (can run parallel to Phase 2 per spec).
- Smoke notebook suite: `tutorials/`.

**Outputs:**
- Commits on `phase-3-plotting`, one logical change per commit.
- Commit prefix: `fix(plot):`.
- PR opened to `modernize` when all plotting rows closed.

**Stop conditions:**
- All `matplotlib|yellowbrick|schemdraw|plotly|plotly-resampler|mljar-scikit-plot`-tagged rows closed.
- `pytest tests/` green (plot-rendering tests included).
- Tutorial smoke notebooks render without exceptions.
- Cherry-pick dry-run green.

**Out-of-scope:**
- sklearn/pandas/time-series fixes.

**Handoff protocol:** Same as sklearn-dev.
