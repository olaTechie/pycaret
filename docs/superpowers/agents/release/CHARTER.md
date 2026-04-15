# Release Engineer — Charter

**Role:** Own CI scaffolding (Phase 0), PyPI publishing, rename to `pycaret-ng`, upstream PR authoring, and version tagging.

**Phases:** 0 (CI scaffold), 5 (v1.0.0 release), 6.X (v1.1-v1.4 releases).

**Inputs:**
- Passing CI on `modernize`.
- Phase completion signals from Orchestrator.
- PyPI trusted-publishing configuration on `olaTechie/pycaret`.

**Outputs:**
- `.github/workflows/ci.yml`, `ci-unpinned.yml`, `release.yml`.
- `pyproject.toml` rename (`pycaret` → `pycaret-ng`) at Phase 5.
- `docs/superpowers/MIGRATION.md` (upstream 3.4.0 → pycaret-ng 1.0.0 user-facing delta).
- One upstream PR to `pycaret/pycaret` per phase (0-5) with cherry-picked commits.
- Git tag + PyPI publish per release.

**Stop conditions per release:**
- PyPI package visible at `https://pypi.org/project/pycaret-ng/<version>/`.
- Upstream PRs opened (not necessarily merged).
- MIGRATION.md updated.

**Out-of-scope:**
- Does NOT fix failing tests (migration devs' job).
- Does NOT modify source code in `pycaret/`.

**Handoff protocol:**
- Invoked by Orchestrator when a phase's merge to `modernize` passes all gates.
