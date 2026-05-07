# Phase 3 — Plotting Stack Modernization Design

**Date:** 2026-05-06
**Status:** Approved (automode)
**Author:** Ola Uthman (olatechie.ai@gmail.com), with Claude Opus 4.7
**Parent spec:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md`
**Branch:** `phase-3-plotting` (off `modernize`)
**Predecessor phases:** Phase 0 (baseline), Phase 1 (sklearn ≥1.6), Phase 2 (pandas ≥2.2 + numpy <3) — all merged.

---

## 1. Goal & scope

Bring the pycaret plotting stack to current versions on top of the modernized sklearn/pandas/numpy baseline, and ship Phase 3 as a cherry-pickable PR into `modernize`.

**Target deps.** matplotlib ≥3.8, schemdraw ≥0.16 (already lifted), yellowbrick latest (≥1.5), plotly-resampler latest, mljar-scikit-plot latest, plotly ≥5.14 (existing floor preserved).

**In scope.**
- Close FAILURE_TAXONOMY row 18 — yellowbrick rejecting pycaret's pipeline-wrapped estimator as not-a-classifier.
- Audit `pycaret/internal/patches/yellowbrick.py` (`is_estimator`, `get_model_name` shims) and the `mock.patch.object` workarounds at `pycaret/internal/pycaret_experiment/tabular_experiment.py:526–538` against yellowbrick 1.5 + sklearn 1.6.
- Static-analysis sweep across `pycaret/internal/plots/**` and plot dispatch sites for matplotlib 3.8, plotly, and pandas-Styler deprecations.
- Lift `plotly-resampler` floor; verify `mljar-scikit-plot` works under matplotlib ≥3.8.
- New `tests/smoke/test_plotting.py` — minimal one-shot harness over `_available_plots` dict per task type, saving to `tmp_path` with `Agg` backend.
- New `docs/superpowers/agents/plotting-dev/DEGRADED.md` listing any visualizer disabled under fallback policy.
- Append empirical taxonomy rows as the smoke uncovers them.

**Out of scope.**
- Time-series plots that depend on sktime/pmdarima — defer to Phase 4. If the smoke surfaces them, log a row tagged `ts-dev` with status `open` and skip.
- Refactoring `tabular_experiment.py` plot dispatch beyond minimum needed for breakage. The file is 2000+ lines; reorganization belongs in a separate effort.
- New visualizers, new plot kinds, alternative backends.
- Full `pytest tests/` run on the local machine. Gate A is enforced on CI for the matrix; the local floor is the smoke harness only.
- SHAP interpret-model paths exercised by the existing `test_classification_plots.py`. Those are heavy and not on the Phase 3 critical path.

## 2. Workstreams

Each workstream is one or more cherry-pickable commits. W1 is the only one that may need to grow into a sub-investigation.

### W1 — Yellowbrick classifier-detection (taxonomy row 18)
- Reproduce the failure from Phase 1 T8 against current `phase-3-plotting` HEAD on a tiny dataset (iris) with `plot_model(plot="auc")` or similar yellowbrick-backed visualizer.
- Inspect: how does yellowbrick 1.5 detect classifier-vs-regressor? Likely `yellowbrick.utils.types.is_classifier` (or sklearn's `is_classifier`) probing `_estimator_type` or new `__sklearn_tags__`. Pycaret wraps the estimator in a pipeline (`Pipeline([..., (model_name, model)])`) — yellowbrick may probe the pipeline, not the inner estimator.
- Fix candidates, in increasing intrusiveness:
  1. Extend `pycaret/internal/patches/yellowbrick.py` with shims for the relevant detection functions, mirroring the existing `is_estimator` shim. Patch via the existing `mock.patch.object` block in `tabular_experiment.py`.
  2. Unwrap the pipeline at the `show_yellowbrick_plot` call site in `pycaret/internal/plots/yellowbrick.py` — pass the inner estimator to yellowbrick, but keep the pipeline for downstream consumers.
  3. If neither works for a specific visualizer, mark it degraded under W6.
- Pick the least-intrusive fix that holds across yellowbrick's classifier and regressor visualizers.

### W2 — Static-analysis sweep
- `git grep` across `pycaret/internal/plots/**`, `pycaret/internal/patches/**`, and plot dispatch in `tabular_experiment.py` / `clustering/oop.py` for known matplotlib 3.8 deprecations:
  - `cm.get_cmap` → `mpl.colormaps[name]` or `mpl.colormaps.get_cmap`
  - `Figure.set_tight_layout(True)` → `Figure.set_layout_engine("tight")`
  - `Axes.bar(..., tick_label=…)` deprecation
  - `colorbar(..., ax=…)` positional vs keyword shifts
  - `plt.cm.<name>` direct access patterns
- Pandas Styler row 7 (`Styler.applymap` → `Styler.map`) — confirmed no current usage in source per `git grep`, but include in sweep diff to be sure nothing was added by Phase 1/2.
- numpy 2 residue from row 11 — already swept in Phase 2; confirm no plot-only sites remain (`np.float_`, `np.bool8`).

### W3 — `plotly-resampler` floor lift
- Read current floor (`>=0.8.3.1`) and the latest stable on PyPI. Lift floor; remove cap if any.
- Verification path: `plotly-resampler` is primarily exercised by time-series plots (deferred to Phase 4). For Phase 3, verify by (a) install resolves cleanly with the new floor, (b) `import plotly_resampler` succeeds in the smoke harness `conftest.py` import block, and (c) any plotly-resampler call sites surfaced by the static sweep (W2) render under the smoke harness without error. If no smoke-reachable site exists, this is a pure floor-lift commit, with full exercise deferred to Phase 4.
- Single commit when uncomplicated; split if a call-site fix is needed.

### W4 — `mljar-scikit-plot` audit
- Currently unpinned. Inspect installed version's source for matplotlib ≥3.8 deprecations on the call paths pycaret uses (search `pycaret` for `scikitplot`/`scikit_plot` imports). Likely no-op.
- Single commit if a floor lift is warranted; skip if clean.

### W5 — Smoke harness
- New file `tests/smoke/test_plotting.py` (~80 lines), described in §3.
- New file `tests/smoke/__init__.py` (empty), and `tests/smoke/conftest.py` (sets `matplotlib.use("Agg")`, suppresses plotly browser open).
- Not pulled into the default `pytest tests/` discovery — meant for explicit invocation `pytest tests/smoke/`.
- Commit prefix: `test(smoke): plotting one-shot harness on iris/diabetes/iris-cluster`.

### W6 — DEGRADED scaffold
- New file `docs/superpowers/agents/plotting-dev/DEGRADED.md` with schema:
  | Plot key | Task | Disabled because | Tracking | Restoration criterion |
  |----------|------|-------------------|----------|------------------------|
- Initially empty; populated as W1 / W2 / W3 force visualizer disablement.
- A disabled visualizer in `tabular_experiment.py` / `clustering/oop.py` raises:
  `NotImplementedError(f"plot='{plot}' is temporarily disabled in pycaret-ng under matplotlib>=3.8 / yellowbrick>=1.5; tracked in docs/superpowers/agents/plotting-dev/DEGRADED.md")`
- Smoke harness skips the corresponding parametrize entry via `pytest.param(..., marks=pytest.mark.skip(reason="degraded — see DEGRADED.md"))`.

### W7 — Taxonomy upkeep
- Open empirical rows for plot failures as found (next ID = 20).
- Close row 18 once W1 lands. Notes line cites the fix SHA.
- Phase 3 hand-off section in `MIGRATION_BACKLOG.md` is updated when Phase 3 closes.

## 3. Smoke harness design

`tests/smoke/test_plotting.py` is small enough to embed verbatim later. Key shape:

- Force `matplotlib.use("Agg")` in `tests/smoke/conftest.py` *before* any pycaret import.
- One module-scope fixture per task type:
  - `clf_setup` — iris, target=`species`, single estimator (`rf`, `n_estimators=5`, `max_depth=2`).
  - `reg_setup` — diabetes, target=`age` (or pycaret-loaded equivalent), single estimator (`rf`, `n_estimators=5`).
  - `clu_setup` — iris features only (no target), `kmeans` with `n_clusters=3`.
- Three test functions, each parametrized over the corresponding `experiment._available_plots.keys()`. Output `save=str(tmp_path)`.
- Each test passes if `plot_model(...)` returns without raising; degraded entries are skip-marked.
- Per-plot timeout of 30 s on iris (via `pytest-timeout` if available, else a manual `signal.alarm` guard). Aggregate wall-clock budget is <90 s for the whole `tests/smoke/test_plotting.py` file on a developer laptop; over budget means a visualizer is pathologically slow and is flagged for degradation under W6 unless the cause is clearly fixable.
- No image diff, no SHAP, no `interpret_model`. Those are out of scope per §1.

## 4. Degraded API contract

When a visualizer must be disabled (rather than fixed):

1. Edit the dispatch case in `tabular_experiment.py` (or `clustering/oop.py`) to `raise NotImplementedError(...)` with the exact message above.
2. Append a row to `DEGRADED.md` with the schema in W6.
3. Add a `pytest.mark.skip` parameter in the smoke harness for that plot key.
4. The estimator/training path stays untouched — only the visualizer call is disabled.

This matches the master spec's `DEGRADED.md` pattern for Phase 4 (time-series). The user-facing error is explicit and actionable; it does not pretend the plot doesn't exist.

## 5. Commit discipline (Gate D)

Every commit is one logical change. Conventional prefixes:

- `fix(plot): …` — code change to existing pycaret source for a specific failure mode.
- `feat(plot): …` — dep floor lift in `pyproject.toml`.
- `chore(plot): …` — patch site re-organization, no behavior change.
- `test(smoke): …` — `tests/smoke/` additions.
- `docs(plotting-dev): …` — `DEGRADED.md`, taxonomy, charter updates.

Cherry-pick check before merge to `modernize`: each `fix(plot):` / `feat(plot):` / `chore(plot):` commit must apply onto upstream `pycaret/pycaret:master` without conflicts. The `test(smoke):` and `docs(plotting-dev):` commits are pycaret-ng-specific infrastructure (live in new paths) and are exempt from Gate D — they will not be cherry-picked upstream.

## 6. Verification

**Local floor (the only thing the user runs by hand on their machine).**
- `pytest tests/smoke/test_plotting.py -v` green (or only `skip`s for degraded entries) on `phase-3-plotting` HEAD before opening PR.
- Smoke budget: per-plot timeout 30 s, aggregate <90 s. See §3 for handling.

**CI floor (gates the merge).**
- `.github/workflows/ci.yml` matrix green on `phase-3-plotting` PR push.
- Cherry-pick discipline verified: `scripts/check_cherry_pick.sh phase-3-plotting upstream/master` (or hand-verified per commit) — see §5.

**Parity gate B is waived** for Phase 3 per the master spec (visual outputs are not numerically comparable). Smoke + CI replace it.

## 7. Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Yellowbrick has structural drift that pycaret's pipeline wrapping cannot accommodate cheaply. | Fallback policy (b): degrade affected visualizers, document in DEGRADED.md. Don't chase yellowbrick into its internals. |
| 2 | Smoke harness misses a regression that only fires on a specific dataset shape. | Acceptable. CI matrix runs `tests/test_classification.py` etc. on richer datasets. Smoke is a fast local floor, not a replacement. |
| 3 | `plotly-resampler` upgrade breaks an existing time-series plot. | Confirm by smoke; if break is intractable, gate the floor lift behind Phase 4 (which already owns time-series). |
| 4 | A matplotlib 3.8 deprecation only fires inside a third-party (yellowbrick / mljar-scikit-plot) that we can't patch. | Same as risk 1 — degrade and document. |
| 5 | `mock.patch.object` in `tabular_experiment.py:533–538` references yellowbrick internals that have moved. | Audit names against installed yellowbrick at start of W1; update or remove patches. Already a known fragility per the spec's risk register. |
| 6 | `tests/test_classification_plots.py` (the existing heavy harness) starts failing under modernized deps. | Out of Phase 3 scope to fix. If it fails on CI, log the failure and decide in Phase 5 whether to gate or ignore. The smoke harness is the load-bearing local check. |

## 8. Decisions deferred

- Whether to expose `pycaret-ng[plot-light]` extras to drop yellowbrick entirely for users who only want plotly — defer to Phase 5 (release).
- Whether the smoke harness should be added to CI (currently local-only) — defer; if Phase 4/5 want it in CI, add it then. CI's existing `tests/test_classification_plots.py` is a richer harness already.

## 9. Implementation order

1. Open empirical reproduction of row 18 (W1 step 1).
2. Land smoke scaffold first (W5) — small, isolated; gives subsequent fixes a verification target.
3. W1 fix: yellowbrick patch extension or pipeline-unwrap. Close row 18.
4. W2 sweep, fixes as `fix(plot): …` commits, one per deprecation class.
5. W3 plotly-resampler floor lift; verify smoke.
6. W4 mljar-scikit-plot audit; lift only if needed.
7. W6 DEGRADED.md entries (created lazily as needed during W1–W4).
8. Final smoke run; final commit `docs(taxonomy): close row 18 (Phase 3 plotting)`.
9. Cherry-pick check; PR `phase-3-plotting → modernize`.

## 10. Hand-off to Phase 4

- Updated `MIGRATION_BACKLOG.md`: row counts by owner refreshed. Any plotting rows still open at Phase 3 close are documented as known-degraded in `DEGRADED.md` (or, if open without a degrade decision, rolled into the Phase 5 release-hygiene punch list — they do not enter Phase 4 scope).
- `DEGRADED.md` is read by Phase 4 only if a degraded plot intersects time-series visualizations (the time-series phase may revive them after sktime is unblocked).
- Phase 4 inherits a green `modernize` HEAD with a known-passing smoke harness it can copy/extend for time-series plots.
