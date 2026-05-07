# Migration Backlog (Phases 1-5)

Synthesized from `FAILURE_TAXONOMY.md` at end of Phase 0.

**Note:** the current taxonomy is a **seed set** (14 rows, static-analysis + local reproduction). An empirical Cartographer dispatch remains deferred. This backlog will be refreshed when that run completes; phase ordering below is based on the seed set + the spec's locked phase sequence.

## Row counts by owner (post-Phase-3)

| Owner agent | Rows | Status |
|-------------|------|--------|
| sklearn-dev | 4 (IDs 3, 4, 5, 6) | all closed in Phase 1 |
| pandas-dev | 4 (IDs 7, 8, 9, 11) | closed in Phase 2 |
| plotting-dev | 6 (IDs 18, 20, 21, 22, 23, 24, 25) | 4 closed, 2 degraded, 1 open (23 anywidget) |
| ts-dev | 3 (IDs 12, 13, 16) | open — Phase 4 |
| release | 5 (IDs 1, 2, 10, 14, 19) | open — Phase 5 hygiene |

Total: 14 seed rows + 11 empirical (rows 15–25). Phase 3 contributed rows 20–25 from the smoke-harness baseline.

## Phase ordering (locked per design spec)

1. **Phase 1 — sklearn** — largest blast radius per spec; seed rows 3, 4, 5, 6 cover the category-encoders `Tags` import, `_PredictScorer` removal, tag API migration, and sklearn 1.6 symbol-move imports.
2. **Phase 2 — pandas/numpy** — rebases on Phase 1. Seed rows 7, 8, 9 (pandas 2.2 applymap + CoW) and 11 (numpy 2 scalar/API sweep) cover the core migration.
3. **Phase 3 — plotting** — closed (PR pending). Empirical smoke baseline (commit `10901cd9`) added rows 20–25; rows 18, 20, 21, 22 closed; rows 24, 25 degraded (yellowbrick-internal bugs); row 23 (anywidget) remains open with decision deferred to Phase 5.
4. **Phase 4 — time-series** — rebases on 1-3. Seed rows 12 (tbats graceful-disable), 13 (sktime API drift) flag the two known risks. `DEGRADED.md` likely needed.
5. **Phase 5 — release** — rename distribution to `pycaret-ng`, PyPI publish, upstream PRs. Seed rows 1, 2, 10, 14 cover py-version guard removal, joblib pins, `distutils.LooseVersion` removal, joblib 1.5 `Memory.bytes_limit` migration.

## Phase-start entry criteria

Every phase begins with a Cartographer refresh on the current `modernize` HEAD (charter allows re-invocation). Row set may shrink as upstream deps evolve. For Phase 1, the refresh is **required** before migration work begins — the seed set is known-incomplete.

## Phase-close exit criteria (all four gates)

- Gate A: `pytest tests/` green on CI matrix (3.11/3.12/3.13 × linux/macos).
- Gate B (phases 1-4 only): `pytest tests/parity/` within tolerance (metric Δ < 1e-4, rank-corr > 0.999). **Requires Task 12 baseline artifacts to be built first** — currently deferred.
- Gate C: tutorial notebooks render end-to-end.
- Gate D (phases 0-5 only): every commit cherry-pick-clean onto `upstream/master`.

## Parallel-work decision

Phase 2 and Phase 3 MAY run in parallel if the refreshed taxonomy flags no cross-cutting rows (no row tagged both `pandas` and `matplotlib`). Orchestrator decides at Phase 2 kickoff.

## Deferred Phase 0 deliverables

- **Task 12** — empirical 3.4.0 baseline artifacts. Required before Gate B can be enforced. Deferred due to session-restart volatility wiping `/tmp` venvs mid-build. Resume in a dedicated session with a persistent venv location.
- **Task 15** — empirical Cartographer run. Required to densify the taxonomy beyond the 14 seed rows. Same deferral rationale.

Neither blocks Phase 1 *planning* (which this backlog enables). Both block Phase 1 *gate enforcement*: QA cannot report parity until baselines exist, and migration devs will hit signatures not yet in the taxonomy — they must append rows as they encounter them (charter permits this).

## Next action

Invoke `writing-plans` for Phase 1 (sklearn migration) using `docs/superpowers/agents/sklearn-dev/CHARTER.md` plus the `sklearn-dev`-tagged slice of the taxonomy (rows 3, 4, 5, 6) + Researcher's R1 recommendation (sktime dual-API tag shim pattern) as input. Per the charter, Phase 1 work lives on `phase-1-sklearn` branched from `modernize`.
