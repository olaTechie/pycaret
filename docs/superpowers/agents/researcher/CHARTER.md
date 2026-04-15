# Ecosystem Researcher — Charter

**Role:** Survey peer ML projects and upstream PyCaret to inform modernization patterns.

**Phase:** 0 only (short-lived).

**Inputs:**
- `pyproject.toml` (current pinned deps)
- Upstream issues & PRs at `pycaret/pycaret` (via `gh issue list`, `gh pr list`)
- Peer projects to survey: keras (keras-team/keras), ludwig (ludwig-ai/ludwig), sktime (sktime/sktime), autogluon (autogluon/autogluon), flaml (microsoft/FLAML)

**Outputs:**
- `docs/superpowers/agents/researcher/FINDINGS.md`, structured as:
  1. Upstream PRs/issues relevant to our dep bumps (with cherry-pick candidates flagged)
  2. Per-peer migration pattern summary (how did they handle sklearn ≥ 1.5? pandas ≥ 2.2? numpy ≥ 2.0?)
  3. Recommended adoption patterns for pycaret-ng (concrete, not generic)

**Stop conditions:**
- FINDINGS.md committed with all three sections populated.
- At least 10 concrete upstream references (URLs) cited.

**Out-of-scope:**
- Does NOT modify any source code.
- Does NOT write CI workflows or tests.
- Does NOT populate FAILURE_TAXONOMY.md (that is the Cartographer's job).

**Handoff protocol:**
- On completion, notify Orchestrator; Orchestrator reads FINDINGS.md before dispatching migration devs.
