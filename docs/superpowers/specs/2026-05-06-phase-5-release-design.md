# Phase 5 — Release Engineering Design

**Date:** 2026-05-06
**Status:** Approved (automode)
**Author:** Ola Uthman (olatechie.ai@gmail.com), with Claude Opus 4.7
**Parent spec:** `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md`
**Branch:** `phase-5-release` (off `phase-4-timeseries`)
**Predecessor phases:** Phase 0 (baseline), Phase 1 (sklearn ≥1.6), Phase 2 (pandas ≥2.2 + numpy <3), Phase 3 (plotting; PR #3 open), Phase 4 (time-series; PR #4 open).

---

## 1. Goal & posture

Cut **pycaret-ng v1.0.0**: the modernization-complete distribution, ready for the user to publish to PyPI as their action. Phase 5 lands the *code-side* work that makes a release possible — distribution rename, version bump, Python-guard widening, user-facing migration notes, PyPI publish workflow YAML — and stops short of any external-account work that only the user can do.

**Non-goals.**
- Publishing the actual v1.0.0 wheel and sdist to PyPI. That requires registering `pycaret-ng` on pypi.org and configuring trusted publishing on the `olaTechie/pycaret` repository. The Phase 5 PR adds the `release.yml` publish job that fires on tag push; the user pushes the tag (or manually triggers the workflow) when their PyPI side is ready.
- Authoring upstream PRs against `pycaret/pycaret`. Each of Phases 1–4 maintained cherry-pick discipline (Gate D); Phase 5 documents which commits are upstream-ready in `MIGRATION.md` and leaves the actual PR opening to a follow-up user-driven activity.
- Closing the latent / external-blocker punch-list items (rows 14, 23, Phase 4 pmdarima shim). Those become **v1.0.x** patch-release work as their dependencies unblock.

## 2. Scope

### In scope
- `pyproject.toml`: `name = "pycaret"` → `"pycaret-ng"`; `version = "3.4.0"` → `"1.0.0"`. Update the project description, URLs, and classifiers as needed (Python versions supported, license, status).
- `pycaret/__init__.py`: widen the Python version guard to admit 3.13 (and 3.14 if available). Closes `FAILURE_TAXONOMY` row 1.
- `pycaret/utils/_show_versions.py`: update any hard-coded distribution name strings.
- `MIGRATION.md` at repo root: user-facing delta from upstream `pycaret==3.4.0` to `pycaret-ng==1.0.0` covering install path swap (`pip install pycaret` → `pip install pycaret-ng`), modernized dep floors, the DEGRADED registry summary (plotting-dev + ts-dev), and known limitations.
- `.github/workflows/release.yml`: append the PyPI publish job (uses `pypa/gh-action-pypi-publish@release/v1` with trusted publishing). Job runs on tag push after the build job.
- Row 19 punch (soft test deps): gate `tests/test_classification_parallel.py`, `tests/test_clustering_engines.py`, `tests/test_persistence.py` behind `pytest.importorskip("fugue" / "daal4py" / "moto")` so collection no longer errors when the soft deps are absent.
- README.md: minimal update — install line and a "this is pycaret-ng" header. Full README rewrite is out of scope.
- CHANGELOG entry for v1.0.0 if a CHANGELOG file exists; otherwise skip.

### Out of scope
- Closing rows 14, 23 (anywidget), and the Phase 4 latent pmdarima shim. Tracked as v1.0.x candidates.
- Replacing the README wholesale or rewriting docs.
- Adding new features (LLM phases 6.0+ remain post-v1.0.0).
- Changing semver — pycaret-ng starts at 1.0.0 per master spec § 11.
- Touching the parity harness or CI infrastructure beyond the publish job.
- Running the actual `git tag v1.0.0 && git push --tags` — user action.

## 3. Architecture

### 3.1 Branch & merge

- Branch off `phase-4-timeseries` HEAD. Phase 5 commits stack on top of Phase 4.
- Conventional commit prefixes: `feat(release):`, `fix(release):`, `docs(release):`, `chore(release):`, `test(release):`.
- Cherry-pick discipline (Gate D): the Python-guard widening (`pycaret/__init__.py`) and the row 19 test gating are upstream-ready. The distribution rename (`pyproject.toml`), version bump, MIGRATION.md, and `release.yml` publish job are pycaret-ng-only and exempt — they wouldn't apply upstream by definition.

### 3.2 Distribution rename mechanics

The rename touches three places:
1. **`pyproject.toml`** — `[project] name = "pycaret-ng"`. Internal import path stays `pycaret` (so `import pycaret as pc` continues to work, matching master spec § 3.2).
2. **`pycaret/utils/_show_versions.py`** — any string identifying the package by name.
3. **`MIGRATION.md`** — explains the rename and the install-line swap.

The internal import path NOT being renamed is deliberate: it keeps user code drop-in compatible. Only the `pip install` line changes.

### 3.3 Python guard widening

Replace the current dual `RuntimeError` (rejects 3.13+) with a single soft-floor check: error only on `< 3.9`. Drop the upper bound entirely — modern dep floors in `pyproject.toml` already gate the Python ceiling implicitly (some deps will refuse to install on too-new Python).

### 3.4 MIGRATION.md structure

Single user-facing markdown at repo root. Sections:
1. **TL;DR** — one paragraph: install change + modernized deps.
2. **Why pycaret-ng** — short context (upstream pycaret 3.4.0 modernization stalled; this fork resumes).
3. **Install** — pip swap, optional extras.
4. **Modernized dependency floors** — table of before/after for sklearn, pandas, numpy, scipy, matplotlib, sktime, etc.
5. **Known limitations under modern deps** — combined DEGRADED registry summary (BATS/TBATS forecasters, classification `error` plot, clustering `distance` plot, `auto_arima` smoke skip, latent rows).
6. **Cherry-pick provenance** — per-phase summary of what's upstream-ready (Gate D commits) and what's pycaret-ng-only.
7. **Roadmap** — pointer to master spec for v1.1.0+ (LLM phases).

Length target: 300–500 lines. Concrete and actionable, not marketing.

### 3.5 PyPI publish job

Add to `.github/workflows/release.yml`:

```yaml
  publish:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    permissions:
      id-token: write
    environment:
      name: pypi
      url: https://pypi.org/p/pycaret-ng
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

This wires *the workflow*. Trusted publishing requires a separate one-time configuration on pypi.org (register `pycaret-ng` package + add the GitHub Actions trusted publisher pointing at `olaTechie/pycaret` `phase-5-release` workflow). That configuration is the user's action; the workflow is dormant until then.

### 3.6 Row 19 punch (soft test deps)

The three failing collection tests in row 19 import optional packages without a guard:

- `tests/test_classification_parallel.py` — `fugue` (Spark/Dask backend)
- `tests/test_clustering_engines.py` — `daal4py` (Intel acceleration)
- `tests/test_persistence.py` — `moto` (AWS mock)

Add `pytest.importorskip("<pkg>")` at the top of each. Standard pytest pattern; behaves as a clean skip when the dep is absent. Out-of-scope: actually installing the soft deps in CI — that decision belongs to the test extras (`[test]`) which is a wider question.

## 4. Workstreams

| ID | Title | Closes |
|----|-------|--------|
| W1 | Distribution rename in pyproject.toml + _show_versions.py | — |
| W2 | Version bump 3.4.0 → 1.0.0 | — |
| W3 | Python guard widening in pycaret/__init__.py | row 1 |
| W4 | MIGRATION.md draft | — |
| W5 | release.yml publish job | — |
| W6 | Row 19 soft-dep test gating | row 19 |
| W7 | README minimal update | — |
| W8 | Taxonomy + backlog refresh | — |
| W9 | PR `phase-5-release` → `phase-4-timeseries` | — |

## 5. Verification

**Local floor:**
- `python -m build` from `.venv-phase4` (or any clean venv) builds the sdist + wheel without errors. The built wheel's metadata reads `Name: pycaret-ng`, `Version: 1.0.0`, `Provides: pycaret`.
- `pip install dist/pycaret_ng-1.0.0-*.whl` into a throwaway venv, then `python -c "import pycaret; print(pycaret.__version__)"` reports `1.0.0` and the import path is still `pycaret`.
- Phase 3 + Phase 4 smokes still pass on their respective venvs (regression check).

**CI floor:**
- `.github/workflows/ci.yml` matrix green on the PR (3.11 / 3.12 / 3.13 × linux/macos). Python 3.13 is the new addition this phase admits.
- `release.yml` build job green on tag push (publish job dormant until pypi side is configured).

**No formal Gate D / Gate B / smoke harness for Phase 5** beyond inheriting the prior ones — Phase 5 doesn't fix bugs, it ships the modernization.

## 6. Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Internal `import pycaret` still references the old distribution name in some metadata path | Smoke import test post-build verifies `__version__ == "1.0.0"`; `pip show pycaret-ng` reports correct metadata. |
| 2 | Python 3.13 actually has a runtime issue we missed in prior phases | Phase 5 widening is necessary to reach 3.13 in CI; if CI surfaces 3.13-only failures, treat as discovered taxonomy rows and fix in v1.0.1. |
| 3 | MIGRATION.md becomes a marketing document instead of a useful diff | Keep it concrete: dep-floor tables, install-line swaps, DEGRADED entries. Hard cap at 500 lines. |
| 4 | PyPI trusted-publishing setup misconfiguration causes the first tag push to silently fail | Workflow is dormant by design until the user explicitly configures the pypi side. The `environment: pypi` block prevents accidental publish from a misconfigured fork. |
| 5 | Row 19 importorskip pattern hides legitimate test failures | Clean idiom; pytest emits a visible skip with the missing-package reason. The opposite (collection error) was strictly worse. |
| 6 | Cherry-pick commits pile up if upstream `pycaret/pycaret` keeps drifting before the user opens upstream PRs | Out of Phase 5 scope. Each phase already certified its commits as cherry-pickable per Gate D; eventual upstream PRs will rebase as needed. |

## 7. Decisions deferred to per-task plan

- Whether the `release.yml` publish job uses an `environment:` gate (recommended for trusted publishing best practice) or just `if: tag-push`.
- Exact dep classifiers in `pyproject.toml` `[project.classifiers]` — current upstream list may need additions (`Programming Language :: Python :: 3.13`).
- Whether to drop `RuntimeError` entirely on Python `< 3.9` or replace with a `warnings.warn` (we'll keep the hard floor to prevent silent failures).
- README rewrite scope — current minimum is install line + ng header.

## 8. Implementation order

1. W3 (Python guard widening) — small, isolated.
2. W6 (row 19 soft-dep gating) — small, isolated, upstream-cherry-pickable.
3. W1 (distribution rename in `pyproject.toml`).
4. W2 (version bump).
5. W4 (MIGRATION.md draft).
6. W5 (release.yml publish job).
7. W7 (README minimal update).
8. W8 (taxonomy + backlog refresh).
9. W9 (push + PR).

## 9. Hand-off

- After PR #5 merges (whenever PRs #3, #4, #5 land in `modernize` and that merges to a `master` of the user's choice), the user can:
  1. Configure pypi.org trusted publisher for `pycaret-ng` pointing at `olaTechie/pycaret` and the `release.yml` workflow.
  2. Tag `v1.0.0` and push the tag — `release.yml` builds and (with pypi side configured) publishes.
- Post-1.0.0 cleanup tracking:
  - Row 23 (anywidget) → v1.0.1 candidate.
  - Row 14 (joblib `Memory.bytes_limit`) → activates when joblib `<1.5` cap can be lifted; v1.0.x.
  - Phase 4 pmdarima `force_all_finite` shim → activates when sklearn 1.8 becomes installable; v1.0.x.
- Master spec § 4 v1.1.0+ (LLM features) becomes the new active spec for further work.
