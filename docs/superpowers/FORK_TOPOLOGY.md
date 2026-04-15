# pycaret-ng Branch & Remote Topology

## Remotes

- `origin` → `olaTechie/pycaret` — our fork, where all work lives.
- `upstream` → `pycaret/pycaret` — the canonical PyCaret repo. Read-only reference; we never push here.

## Branches

| Branch | Purpose | Lifecycle |
|--------|---------|-----------|
| `master` | Pure mirror of upstream + our own docs/plans. No modernization code. | Permanent. Rebase onto `upstream/master` at phase boundaries. |
| `modernize` | Integration branch for all Phase 1-5 dep modernization. | Permanent until v1.0.0 release; merged to `master` at release cut. |
| `phase-N-<topic>` | Feature branch per phase (e.g. `phase-1-sklearn`). | Short-lived; merged into `modernize` when gates pass, then deleted. |
| `phase-6-X-<topic>` | Feature branches for LLM phases (e.g. `phase-6-0-llm-infra`). | Short-lived; merged into `modernize` then to `master` per minor release. |

## Workflow

1. Branch phase work off `modernize`.
2. Each commit on a phase branch must satisfy gate D (cherry-pickable onto `upstream/master`) for Phases 0-5.
3. Open a PR into `modernize` when phase gates pass.
4. Rebase `modernize` onto new upstream tags at phase boundaries if upstream releases.
5. v1.0.0 release: fast-forward `master` to `modernize`, tag `v1.0.0`.

## Never do these things

- Do NOT force-push `master` or `modernize`.
- Do NOT rewrite commits on `modernize` once merged — cherry-pickability depends on commit stability.
- Do NOT push to `upstream`.
