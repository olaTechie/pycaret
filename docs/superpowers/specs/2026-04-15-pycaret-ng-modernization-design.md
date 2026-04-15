# pycaret-ng — Modernization + LLM Integration Design

**Date:** 2026-04-15
**Status:** Approved (awaiting written-spec review)
**Author:** Ola Uthman (olatechie.ai@gmail.com), with Claude Opus 4.6
**Fork:** `olaTechie/pycaret` (upstream: `pycaret/pycaret`)
**Target distribution:** `pycaret-ng` on PyPI
**Approach:** Phase-gated sweep (coordinated modernization → progressive feature releases)

---

## 1. Goal

Produce `pycaret-ng`, a pip-installable soft-fork of PyCaret 3.4.0 that:

- Works cleanly on Python 3.11, 3.12, 3.13 with modern unpinned dependencies (scikit-learn ≥ 1.5, pandas ≥ 2.2, numpy ≥ 2.0, scipy ≥ 1.12, matplotlib ≥ 3.8, updated time-series stack).
- Preserves upstream-quality standards: every dep-modernization commit is cherry-pickable to `pycaret/pycaret`.
- Adds first-class LLM features (conversational SDK, EDA advisor, auto reports, LLM estimators, MCP server) backed by Anthropic (cloud) and Ollama/llama.cpp (local-first).
- Ships in progressive minor versions so users can opt into the feature tier they need.

## 2. Scope & success gates

### In scope

- Dep modernization of all task modules (classification, regression, clustering, anomaly, time-series) and internals.
- PyPI publishing as `pycaret-ng`; public import path stays `import pycaret` for drop-in compat.
- CI matrix on Python 3.11 / 3.12 / 3.13 × linux / macos.
- Parity harness vs. frozen PyCaret 3.4.0 on reference datasets.
- LLM features A–G from the brainstorming session, split across v1.1–v1.4.

### Out of scope

- New ML estimators beyond LLM/HF additions in Phase 6.3.
- Architectural refactors beyond what dep changes force.
- GPU/MPS acceleration of classical pipelines (HF path will use torch, but sklearn path stays CPU).
- Hard coupling to `mltoolkit-plugin`. Shared code goes into a small standalone package.

### Success gates

| Gate | Name | Applies to | Criterion |
|------|------|------------|-----------|
| A | Test suite green | Phases 1-5 | `pytest` passes on all CI matrix cells with unpinned modern deps. |
| B | Numerical parity | Phases 1-4 | Metric Δ < 1e-4 vs. 3.4.0 baseline on reference datasets; prediction rank-correlation > 0.999. |
| C | Core smoke | Phases 1-5 | `classify/regress/cluster/anomaly/time-series` tutorials run end-to-end. Time-series may ship "known-degraded" with `DEGRADED.md`. |
| D | Cherry-pickable | Phases 0-5 | Each commit applies cleanly onto upstream `pycaret/pycaret:master`. Phase 6 exempt (features live in new paths). |
| E | LLM eval stability | Phase 6 | Golden-Q&A regression suite passes at temperature=0; rank-correlation tolerance at temperature>0. |

## 3. Architecture

### 3.1 Repository & branch topology

- Stay in `olaTechie/pycaret` fork. `master` mirrors upstream 3.4.0 (pristine, enables gate D).
- Long-lived `modernize` branch off `master` is the integration line.
- Each phase gets a feature branch (`phase-0-baseline`, `phase-1-sklearn`, `phase-2-pandas`, `phase-3-plotting`, `phase-4-timeseries`, `phase-5-release`, `phase-6-0-llm-infra`, `phase-6-1-conversational`, `phase-6-2-artifacts`, `phase-6-3-llm-estimators`, `phase-6-4-mcp`). Merges to `modernize` only when its gates pass.

### 3.2 Distribution

- PyPI name: `pycaret-ng`. Internal import path unchanged (`import pycaret`), so users migrate by swapping the install line.
- Base install stays lightweight. LLM deps live in extras:
  - `pip install pycaret-ng[llm]` → Anthropic + pydantic
  - `pip install pycaret-ng[llm-local]` → Ollama + llama-cpp-python
  - `pip install pycaret-ng[hf]` → transformers + torch
  - `pip install pycaret-ng[mcp]` → MCP server deps
  - `pip install pycaret-ng[full]` → everything (retains upstream semantics)

### 3.3 Parity harness

- New `tests/parity/` directory, runnable as `pytest tests/parity/ --baseline=3.4.0`.
- Frozen `pycaret==3.4.0` baseline built once via `scripts/build_parity_baseline.py` in an isolated conda env with Python 3.11 and last-working pinned deps. Outputs committed to `tests/parity/baselines/3.4.0/<dataset>/{leaderboard.json,predictions.npz}`.
- Reference datasets: iris, diabetes, california_housing, credit (UCI default-of-credit), airline-passengers (for time-series). *Note: Boston housing is removed from sklearn ≥ 1.2 for ethical reasons — we use California housing instead.*
- Tolerances: metric Δ < 1e-4, prediction rank-correlation > 0.999. Per-estimator widening allowed when upstream change is documented and benign.
- `random_state=42` everywhere for determinism.

### 3.4 CI matrix

- `.github/workflows/ci.yml` — `{3.11, 3.12, 3.13} × {linux, macos}` on every push/PR, pinned modern dep set.
- `.github/workflows/ci-unpinned.yml` — weekly cron against latest deps, catches upstream regressions.
- `.github/workflows/release.yml` — tag-triggered PyPI trusted publishing.

### 3.5 Filesystem layout (additions)

```
pycaret/                             # existing source, modified in-place
  llm/                               # NEW (Phase 6.0+)
    __init__.py                      # public API: ask, recommend, explain, advise_preprocessing, generate_report
    providers/
      __init__.py                    # Provider protocol + registry
      anthropic.py                   # AnthropicProvider (+ prompt caching)
      ollama.py                      # OllamaProvider
      llamacpp.py                    # LlamaCppProvider
      cache.py                       # shared disk cache
      cost.py                        # cost tracker + budget guard
    prompts/
      registry.py                    # versioned prompt templates
      templates/                     # .md files, one per feature
    eval/
      golden/                        # reference Q&A / artifact pairs
      run_eval.py                    # pytest-invoked eval harness
    features/
      conversation.py                # A + C + F
      advisor.py                     # B
      reporting.py                   # D (shares with mltoolkit-plugin via pycaret-report)
      interpretability.py            # F
  containers/models/
    classification/
      llm_classifier.py              # NEW Phase 6.3
      hf_transformer_classifier.py   # NEW Phase 6.3
    regression/
      llm_embedding_regressor.py     # NEW
pycaret_mcp/                         # NEW (Phase 6.4)
  __init__.py
  server.py                          # MCP server entrypoint
  tools.py                           # @mcp.tool() wrappers
  session.py                         # per-connection experiment state
tests/
  parity/                            # NEW (Phase 0)
    conftest.py
    datasets.py
    test_compare_models_parity.py
    test_predict_parity.py
    baselines/3.4.0/<dataset>/{leaderboard.json,predictions.npz}
docs/
  superpowers/
    specs/2026-04-15-pycaret-ng-modernization-design.md    # this file
    specs/<date>-phase-N-<topic>-design.md                  # per-phase sub-specs
    plans/<date>-phase-N-<topic>-plan.md                    # writing-plans outputs
    agents/
      researcher/{CHARTER.md,LOG.md,FINDINGS.md}
      cartographer/{CHARTER.md,LOG.md}
      sklearn-dev/{CHARTER.md,LOG.md}
      pandas-dev/{CHARTER.md,LOG.md}
      plotting-dev/{CHARTER.md,LOG.md}
      ts-dev/{CHARTER.md,LOG.md,DEGRADED.md}
      qa/{CHARTER.md,LOG.md,phase-N-parity.md}
      release/{CHARTER.md,LOG.md}
      llm-infra-dev/{CHARTER.md,LOG.md}
      llm-feature-dev/{CHARTER.md,LOG.md}
      llm-estimator-dev/{CHARTER.md,LOG.md}
      prompt-eng-eval/{CHARTER.md,LOG.md}
    FAILURE_TAXONOMY.md              # shared, cartographer-owned
    MIGRATION.md                     # release-engineer-owned, user-facing
.github/workflows/
  ci.yml
  ci-unpinned.yml
  release.yml
```

## 4. Phase breakdown & release cadence

Each phase is its own spec → plan → implementation cycle.

### v1.0.0 — Modernization (Phases 0-5)

**Phase 0 — Baseline & Cartography** *(Ecosystem Researcher + Dep Cartographer + Release Engineer)*
Fork hygiene, branch topology, CI matrix skeleton, parity harness scaffolding, reference-dataset baseline build, upstream-issue triage (mine `pycaret/pycaret` issues & PRs for unmerged fixes), survey peer projects (Keras/Ludwig/sktime/autogluon/flaml). Output: `FAILURE_TAXONOMY.md`, parity-harness green against frozen 3.4.0, prioritized migration backlog.

**Phase 1 — scikit-learn ≥ 1.5** *(sklearn Dev + QA)*
Biggest blast radius. `pycaret/containers/models/*` estimator wrappers, `set_output` API adoption, transformer API (`_more_tags`, `__sklearn_tags__`), removed private imports. Hot spots: preprocess pipeline, `compare_models`, `tune_model`, cross-val wrappers.

**Phase 2 — pandas ≥ 2.2 + numpy ≥ 2.0** *(pandas Dev + QA)*
`applymap` → `map`, copy-on-write semantics, deprecated dtype behaviors, `pd.Index.astype` changes, `numpy.bool8` removals. Hot spots: `pycaret/internal/preprocess/`, `pycaret/utils/time_series/`.

**Phase 3 — Plotting stack** *(Plotting Dev + QA)*
matplotlib ≥ 3.8 (unlocks `schemdraw` ≥ 0.16), yellowbrick compat, plotly-resampler upgrade, `mljar-scikit-plot` review. Parity gate B waived (visual output not numerically comparable); gates A+C+D apply.

**Phase 4 — Time-series stack** *(TS Dev + QA)*
sktime unpin from `0.31.0,<0.31.1`, pmdarima/statsmodels upgrade, tbats compat. Highest risk of "known-degraded" outcome — if sktime's breaking changes are structural, ship with narrowed API + `DEGRADED.md` documenting the gap.

**Phase 5 — Release** *(Release Engineer)*
Rename distribution to `pycaret-ng` in `pyproject.toml`, set up PyPI trusted publishing, write `MIGRATION.md` (upstream 3.4.0 → pycaret-ng 1.0.0 delta), author upstream PRs (one per phase's cherry-picked commit set), cut v1.0.0.

Phases 1-4 are mostly sequential (each rebases on prior). Phase 3 (plotting) can run parallel to Phase 2 (pandas) since the plotting stack is largely independent of pandas internals.

### v1.1.0 — Conversational SDK (Phase 6.0 + 6.1)

**Phase 6.0 — LLM Infra** *(LLM Infra Dev)*
Foundational, blocks 6.1-6.4. `pycaret/llm/providers/` with `Provider` protocol (`complete`, `stream`, `embed`). Implementations: `AnthropicProvider` (anthropic SDK, prompt caching), `OllamaProvider`, `LlamaCppProvider`. Cross-cutting: disk-backed response cache keyed by prompt hash, cost tracker with budget guard, retry/backoff, Pydantic structured-output parser, prompt-template registry, async batching.

**Phase 6.1 — Conversational SDK** *(LLM Feature Dev + Prompt Eng/Eval)*
Features A, C, F.
- A: `pycaret.ask(query, data=df)` — NL → API call translator with tool-use.
- C: `pycaret.recommend(problem_description, data=df)` — structured estimator / metric / CV recommendation.
- F: `pycaret.explain(model, row)` — SHAP + narrative.

### v1.2.0 — Artifact generation (Phase 6.2)

Features B and D.
- B: `pycaret.advise_preprocessing(df)` — reads profile, returns ranked suggestions with rationale.
- D: `pycaret.generate_report(run_id)` — methods / results / model-card prose.

**Vendoring plan for D:** extract `mltoolkit-plugin/_shared/{methods_md,model_card}` scaffolds into a standalone `pycaret-report` sub-package so both projects depend on it. Avoids hard coupling between pycaret-ng and mltoolkit-plugin.

### v1.3.0 — LLMs as zoo estimators (Phase 6.3)

**Phase 6.3** *(LLM Estimator Dev + Prompt Eng/Eval)*
Feature E. New estimator containers: `LLMClassifier`, `LLMZeroShotClassifier`, `LLMEmbeddingRegressor` (embed → downstream sklearn head), `HFTransformerClassifier`. Integration: `compare_models(include=['llm_claude', 'llm_ollama', 'hf_distilbert'])`. Requires batching, per-model cost cap, timeout handling, prediction caching, tokenizer management. Biggest single work item of the whole project.

### v1.4.0 — MCP server (Phase 6.4)

**Phase 6.4** *(LLM Feature Dev)*
Feature G. `pycaret-mcp` console script exposes `setup`, `compare_models`, `tune_model`, `finalize_model`, `predict_model`, `save_model` as MCP tools over stdio. Per-connection session state holds the active experiment. No auth on stdio; v1.4.1 adds optional HTTP transport with bearer token.

## 5. Team composition

13-agent team, dispatched via Claude Code's built-in `Agent` tool with `subagent_type=general-purpose` (or `feature-dev:code-explorer/code-architect` where fitting). Each agent has `docs/superpowers/agents/<agent>/CHARTER.md` (scope + contracts), `LOG.md` (append-only progress), and contributes to shared artifacts.

### 5.1 Modernization agents (Phases 0-5)

1. **Ecosystem Researcher** *(short-lived, Phase 0)*
   In: `pyproject.toml`, upstream issues/PRs, peer-project list. Out: `researcher/FINDINGS.md` — migration patterns + upstream fixes to absorb. Stop: findings committed. Does not touch source.

2. **Dep Cartographer** *(short-lived, Phase 0; re-invoked at each phase start)*
   In: unpinned-deps test run output. Out: `FAILURE_TAXONOMY.md` rows (dep × module × root-cause class). Stop: every failure categorized with owner agent. Does not fix code.

3. **sklearn Migration Dev** *(Phase 1)*
   In: taxonomy rows tagged `sklearn`. Out: commits on `phase-1-sklearn`, one logical change per commit. Stop: all sklearn rows closed, parity + smoke green, PR opened to `modernize`. Out-of-scope: pandas/plotting/time-series fixes (flag and hand off).

4. **pandas/numpy Migration Dev** *(Phase 2)* — same contract, `pandas|numpy` tag.

5. **Plotting Migration Dev** *(Phase 3)* — same contract, `matplotlib|yellowbrick|schemdraw|plotly` tag; parity gate B waived.

6. **Time-series Migration Dev** *(Phase 4)* — same contract, `sktime|pmdarima|statsmodels|tbats` tag; may propose API narrowing if upstream breakage is structural (document in `DEGRADED.md`).

7. **Data Scientist / QA** *(continuous, every phase)*
   In: modernized branch HEAD, frozen 3.4.0 baseline. Out: `qa/phase-N-parity.md`, blocking comments on migration PRs. Authority: can block a phase merge.

8. **Release Engineer** *(Phase 0 CI scaffolding, Phase 5 publish, later for each minor release)*
   In: merged phase branches with passing CI. Out: `pyproject.toml` updates, PyPI publishes, upstream PRs, `MIGRATION.md`, release tags.

9. **Orchestrator** *(you + me)*
   Dispatches agents, arbitrates conflicts, owns phase-merge gates, maintains top-level plan.

### 5.2 LLM-feature agents (Phase 6)

10. **LLM Infra Dev** — owns Phase 6.0. Provider abstraction, caching, cost tracking, prompt registry.

11. **LLM Feature Dev** — owns 6.1, 6.2, 6.4. Rotates across features (shared infra).

12. **LLM Estimator Dev** — dedicated to Phase 6.3 (deepest integration).

13. **Prompt Engineer / Eval** — maintains prompt templates and eval harness (golden Q&A set per feature, regression-tested on Anthropic + Ollama). Critical because LLM outputs are non-deterministic; eval stability (gate E) depends on this role.

## 6. Data flow per phase

```
Cartographer writes row in FAILURE_TAXONOMY.md
        ↓
Orchestrator dispatches migration-dev Agent with {charter, taxonomy-slice, branch}
        ↓
Migration Dev commits fixes on phase branch (one logical change per commit)
        ↓
QA runs parity harness on phase branch HEAD → phase-N-parity.md
        ↓
Orchestrator reviews: all applicable gates green?
   ├─ yes → merge phase branch to `modernize`, proceed to next phase
   └─ no  → re-dispatch migration dev with QA blocking comments
```

Phase 6 substitutes the Eval agent for QA; dispatch flow otherwise identical.

## 7. Commit discipline (gate D)

Every commit on a Phases 0-5 branch must be:

1. **Isolated** — scoped to one root-cause class from the taxonomy.
2. **Self-contained** — test + fix + docstring in the same commit.
3. **Conventional message** — `fix(sklearn): ...`, `fix(pandas): ...`, `chore(ci): ...`, etc.
4. **Cherry-pick clean** — applies onto upstream `pycaret/pycaret:master` without conflicts (verified before merge to `modernize`).

Phase 6 commits are exempt from gate D and may use the path prefixes `feat(llm): ...`, `feat(mcp): ...`.

## 8. Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | sktime structural break in Phase 4 | Cartographer sizes impact in Phase 0; fallback to vendored sktime-compat shim or narrowed API + `DEGRADED.md`. |
| 2 | Parity tolerance calibration | Gate B uses metric Δ < 1e-4 and rank-correlation > 0.999, not bit-exactness. QA widens per-estimator with documented rationale. |
| 3 | Cherry-pick rot if upstream releases 3.5 mid-project | Keep `master` pure mirror; rebase `modernize` on new upstream tags at phase boundaries. |
| 4 | Upstream PR rejection | Gate D is cherry-pickability, not merge acceptance. Ship `pycaret-ng` regardless. |
| 5 | Agent context drift across subagent dispatches | Charters are self-contained; shared state on disk (`FAILURE_TAXONOMY.md`, `LOG.md`s); orchestrator hand-summarizes context into each dispatch prompt. |
| 6 | CI cost blowup | Parity harness runs only on `modernize` + PRs to it, not every phase-branch push. Unpinned matrix runs weekly. |
| 7 | LLM non-determinism breaking parity | Gate B does not apply to Phase 6; gate E (eval stability) substitutes: golden-Q&A at temperature=0 plus rank-correlation at temperature>0. |
| 8 | API cost during CI for Phase 6 | VCR-style record/replay fixtures for Anthropic; Ollama tests use a tiny model in a pinned Docker image; full-API smoke weekly only. |
| 9 | Prompt injection via `ask(data=df)` | Structured output parsing + tool-use schema. `MIGRATION.md` documents that `ask()` is unsafe with untrusted data. |
| 10 | Provider SDK drift (anthropic SDK pre-1.0) | Pin SDK minor versions; Provider protocol isolates us from drift. |
| 11 | mltoolkit-plugin coupling | Extract shared report-writing code into standalone `pycaret-report` package; both projects depend on it. No direct coupling. |

## 9. Decisions deferred to per-phase specs

- Exact sklearn floor (1.5? 1.6? latest?) — Phase 1 spec decides on Cartographer findings.
- `set_output(transform='pandas')` project-wide vs. keep numpy flow — Phase 1.
- pandas CoW adoption strategy — Phase 2.
- sktime narrowing scope if needed — Phase 4.
- `pycaret-ng` semver alignment with upstream 3.4.0 or independent versioning — Phase 5.
- Local-model default for CI (e.g., `llama3.2:1b`) — Phase 6.0.
- Whether `pycaret.ask()` supports streaming to the caller — Phase 6.1.
- MCP HTTP transport auth design — Phase 6.4.1.

## 10. Implementation order

1. Write Phase 0 implementation plan (via `writing-plans` skill).
2. Execute Phase 0 (baseline + cartography + CI scaffold + parity harness).
3. At Phase 0 completion, write Phase 1 spec + plan; repeat per phase.
4. Release v1.0.0 after Phase 5.
5. Phase 6 minor-release cadence (6.0+6.1 → v1.1, 6.2 → v1.2, 6.3 → v1.3, 6.4 → v1.4).

## 11. Versioning

- `pycaret-ng` starts at `1.0.0` at end of Phase 5 (modernization complete).
- Semver: breaking API changes bump major, new features bump minor, fixes bump patch.
- `pycaret-ng` does not promise semver alignment with upstream PyCaret; `MIGRATION.md` documents the delta.
