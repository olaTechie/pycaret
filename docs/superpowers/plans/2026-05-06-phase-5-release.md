# Phase 5 Release Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut **pycaret-ng v1.0.0** by renaming the distribution, bumping the version, widening the Python guard, authoring `MIGRATION.md`, wiring the PyPI publish workflow, and gating soft-dep tests behind importable checks.

**Architecture:** Surgical edits to `pyproject.toml`, `pycaret/__init__.py`, and `pycaret/utils/_show_versions.py` carry the rename + version bump. New `MIGRATION.md` at repo root carries the user-facing delta. New publish job in `.github/workflows/release.yml` wires PyPI trusted publishing (workflow only — pypi.org configuration is user-action). `pytest.importorskip` at the head of three test files closes taxonomy row 19. No new modules, no architectural changes.

**Tech Stack:** Python 3.12, pyproject.toml (PEP 621), GitHub Actions, PyPI trusted publishing.

**Spec:** `docs/superpowers/specs/2026-05-06-phase-5-release-design.md`

**Branch:** `phase-5-release` (off `phase-4-timeseries`).

**Verification floor (local):** `python -m build` from any clean Python 3.12 venv produces sdist + wheel; `pip install dist/pycaret_ng-1.0.0-*.whl` into a throwaway venv; `python -c "import pycaret; print(pycaret.__version__)"` reports `1.0.0`. The Phase 3 plotting smoke (`.venv-phase3`) and Phase 4 TS smoke (`.venv-phase4`) must remain green after rebase.

---

## File Structure

**Create:**
- `MIGRATION.md` — user-facing migration guide at repo root (~300–500 lines)

**Modify:**
- `pyproject.toml` — `name = "pycaret"` → `"pycaret-ng"`; `version = "3.4.0"` → `"1.0.0"`; add `Programming Language :: Python :: 3.13` classifier
- `pycaret/__init__.py` — replace dual RuntimeError with single `< 3.9` floor; bump `version_` literal to `"1.0.0"`
- `pycaret/utils/_show_versions.py` — update any hard-coded distribution name strings (probe first; the file may not have any)
- `.github/workflows/release.yml` — append publish job that fires on tag push
- `tests/test_classification_parallel.py:1` — prepend `pytest.importorskip("fugue")`
- `tests/test_clustering_engines.py:1` — prepend `pytest.importorskip("daal4py")`
- `tests/test_persistence.py:1` — prepend `pytest.importorskip("moto")`
- `README.md` — replace install-line section with `pycaret-ng` install + a short "this is pycaret-ng" header
- `docs/superpowers/FAILURE_TAXONOMY.md` — close rows 1, 19
- `docs/superpowers/MIGRATION_BACKLOG.md` — refresh release row counts at Phase 5 close
- `docs/superpowers/agents/release/LOG.md` — append Phase 5 closure entry (file may not exist; create if so)

**Reference (read, don't modify):**
- `docs/superpowers/agents/plotting-dev/DEGRADED.md` — for MIGRATION.md "Known limitations" section
- `docs/superpowers/agents/ts-dev/DEGRADED.md` — same purpose
- All prior phase LOGs — for MIGRATION.md "Cherry-pick provenance" section

---

## Task 1: Widen Python guard (W3)

**Files:**
- Modify: `pycaret/__init__.py:11-26`

Closes `FAILURE_TAXONOMY` row 1.

- [ ] **Step 1: Read the current guard**

```bash
sed -n '1,30p' pycaret/__init__.py
```

Expected: dual-RuntimeError block — one for `< 3.9`, one for `>= 3.13`. Both error messages still say `"Pycaret only supports python 3.9, 3.10, 3.11, 3.12"`.

- [ ] **Step 2: Replace the guard**

Replace the entire block (lines 1-26 of the current file) with:

```python
import sys

from pycaret.utils._show_versions import show_versions

version_ = "1.0.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# pycaret-ng targets Python 3.9+. The dep floors in pyproject.toml
# implicitly gate the upper end (some deps will refuse to install on
# too-new Python). We hard-error only on the lower bound so that an
# accidental Python 3.8 invocation gives a clear diagnostic.
if sys.version_info < (3, 9):
    raise RuntimeError(
        "pycaret-ng requires Python >= 3.9. Your actual Python version: ",
        sys.version_info,
        "Please upgrade your Python.",
    )
```

(Note: the version bump from `3.4.0` to `1.0.0` is included here — Task 4 will not re-edit this file.)

- [ ] **Step 3: Verify import on .venv-phase3 (Python 3.12)**

```bash
.venv-phase3/bin/python -c "import pycaret; print('version:', pycaret.__version__)"
```

Expected: `version: 1.0.0`.

- [ ] **Step 4: Commit**

```bash
git add pycaret/__init__.py
git commit -m "$(cat <<'EOF'
fix(release): widen Python guard + bump version to 1.0.0

pycaret-ng targets Python 3.9+. The previous guard hard-errored on
3.13, blocking the modernized stack from running on the latest CPython.
Replace with a single < 3.9 floor; the upper end is implicitly gated
by dep floors in pyproject.toml.

Bumps version_ to 1.0.0 in the same edit to keep pycaret/__init__.py
self-consistent (pyproject.toml version bump lands in a separate
commit).

Closes FAILURE_TAXONOMY row 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Soft-dep test gating (W6)

**Files:**
- Modify: `tests/test_classification_parallel.py:1` (prepend)
- Modify: `tests/test_clustering_engines.py:1` (prepend)
- Modify: `tests/test_persistence.py:1` (prepend)

Closes `FAILURE_TAXONOMY` row 19.

- [ ] **Step 1: Prepend importorskip to test_classification_parallel.py**

Read the current first line:

```bash
head -1 tests/test_classification_parallel.py
```

Expected: `import pycaret.classification as pc`.

Replace the file's content so the first lines become:

```python
import pytest

pytest.importorskip("fugue")

import pycaret.classification as pc
```

(Insert the two new lines at the very top; everything else stays.)

- [ ] **Step 2: Prepend importorskip to test_clustering_engines.py**

Read the current first line:

```bash
head -1 tests/test_clustering_engines.py
```

Expected: `import daal4py`.

Replace so the first lines become:

```python
import pytest

pytest.importorskip("daal4py")

import daal4py
```

- [ ] **Step 3: Prepend importorskip to test_persistence.py**

Read the current first lines:

```bash
head -5 tests/test_persistence.py
```

Expected: imports including `from moto import mock_s3` on line 5.

Replace so the very top becomes:

```python
import pytest

pytest.importorskip("moto")
```

(Keep the existing imports; the importorskip just needs to run before any `from moto import ...`. Insert above the existing first import.)

- [ ] **Step 4: Verify pytest collects without errors**

```bash
.venv-phase3/bin/python -m pytest --collect-only tests/test_classification_parallel.py tests/test_clustering_engines.py tests/test_persistence.py 2>&1 | tail -10
```

Expected: pytest reports each file as either collected or skipped (depending on whether the soft dep happens to be installed). No collection errors.

If `pycaret.parallel` import inside `test_classification_parallel.py` triggers a fugue import deeper in pycaret's source even when fugue is missing, the importorskip won't fire fast enough. In that case, also add `pytest.importorskip("fugue")` before the `from pycaret.parallel import FugueBackend` line.

- [ ] **Step 5: Commit**

```bash
git add tests/test_classification_parallel.py tests/test_clustering_engines.py tests/test_persistence.py
git commit -m "$(cat <<'EOF'
fix(release): gate soft-dep tests behind pytest.importorskip

tests/test_classification_parallel.py, tests/test_clustering_engines.py,
and tests/test_persistence.py imported fugue / daal4py / moto at module
level. When the soft deps were absent (the [test] extras don't include
them by default), pytest erred at collection rather than skipping.

Standard pytest pattern via importorskip: the test now skips cleanly
when the dep is missing.

Closes FAILURE_TAXONOMY row 19.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Distribution rename in pyproject.toml (W1)

**Files:**
- Modify: `pyproject.toml` — `[project]` table

- [ ] **Step 1: Inspect the current `[project]` block**

```bash
sed -n '1,45p' pyproject.toml
```

Expected: `name = "pycaret"`, `version = "3.4.0"`, classifiers list ending at Python 3.12.

- [ ] **Step 2: Rename + version bump + classifier add**

Make three edits in `pyproject.toml`:

(a) Replace `name = "pycaret"` with `name = "pycaret-ng"`.

(b) Replace `version = "3.4.0"` with `version = "1.0.0"`.

(c) In the `classifiers = [...]` list, add `"Programming Language :: Python :: 3.13",` directly below the existing `"Programming Language :: Python :: 3.12",` entry.

Optional polish (do at engineer's judgment):
- Update `description` field to mention pycaret-ng if it doesn't already.
- Update `urls` if the repo URL has changed.

- [ ] **Step 3: Verify the edits parsed**

```bash
.venv-phase3/bin/python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    d = tomllib.load(f)
p = d['project']
print('name:', p['name'])
print('version:', p['version'])
print('python-3.13 classifier:', any('3.13' in c for c in p['classifiers']))
"
```

Expected output:

```
name: pycaret-ng
version: 1.0.0
python-3.13 classifier: True
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
feat(release): rename distribution to pycaret-ng + bump to 1.0.0

Per master spec § 3.2 / § 11: the PyPI distribution becomes pycaret-ng,
internal import path stays `pycaret` for drop-in compatibility (users
swap their pip install line, not their import statements).

Adds Programming Language :: Python :: 3.13 classifier (Phase 5
widened the runtime guard to admit 3.13).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update _show_versions.py (W1 cont)

**Files:**
- Possibly Modify: `pycaret/utils/_show_versions.py`

Probe-driven. The file may already abstract the distribution name correctly.

- [ ] **Step 1: Inspect the file for hard-coded distribution name**

```bash
grep -nE '"pycaret"|"3\.4\.0"' pycaret/utils/_show_versions.py 2>&1
```

Expected (one of):
- No matches → no edit needed; skip to Step 4.
- 1+ matches → review and decide whether each occurrence is the *distribution* name (rename to `pycaret-ng`) or the *import* name (leave as `pycaret`).

- [ ] **Step 2: If matches exist, edit each one**

For occurrences that are PyPI distribution-name references (e.g., `"pycaret" in metadata strings`), update to `"pycaret-ng"`.

For occurrences that are Python import path references (e.g., `import pycaret` strings), leave as is.

If the file uses `importlib.metadata.version("pycaret")`, change the argument to `"pycaret-ng"`.

- [ ] **Step 3: Verify the import path still resolves**

```bash
.venv-phase3/bin/python -c "from pycaret.utils._show_versions import show_versions; show_versions()" 2>&1 | head -10
```

Expected: prints the version table without ImportError. The pycaret-ng version line should read `1.0.0` (or the value of `pycaret.__version__`).

- [ ] **Step 4: Commit (only if Steps 1-2 produced edits)**

```bash
git add pycaret/utils/_show_versions.py
git commit -m "$(cat <<'EOF'
chore(release): update _show_versions distribution-name strings

Aligns hard-coded distribution-name references with the pycaret-ng
rename. Import-path references (which stay 'pycaret') are unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If no edits were needed, skip the commit and append a note to the LOG.

---

## Task 5: Author MIGRATION.md (W4)

**Files:**
- Create: `MIGRATION.md` at repo root

- [ ] **Step 1: Gather inputs**

Collect these into scratch notes (the engineer needs them all to fill the migration doc):

```bash
# Modernized dep floors — read pyproject.toml current state
grep -E 'numpy|pandas|scikit-learn|scipy|matplotlib|sktime|statsmodels|pmdarima|tbats|joblib|category-encoders|imbalanced-learn|yellowbrick|schemdraw|plotly-resampler|mljar-scikit-plot' pyproject.toml | head -25

# DEGRADED registry summaries
cat docs/superpowers/agents/plotting-dev/DEGRADED.md
cat docs/superpowers/agents/ts-dev/DEGRADED.md

# Cherry-pick provenance
git log --oneline modernize..phase-5-release | grep -E '^[a-f0-9]+ (fix|feat|chore)\(' | head -50
```

- [ ] **Step 2: Write `MIGRATION.md`**

Use this exact skeleton; fill in concrete values from Step 1 where indicated:

```markdown
# Migrating to pycaret-ng 1.0.0

`pycaret-ng` is a soft-fork of [PyCaret 3.4.0](https://github.com/pycaret/pycaret) that resumes modernization on Python 3.9–3.13, modern scikit-learn / pandas / numpy, and the current sktime / pmdarima / matplotlib / yellowbrick. The internal import path stays `pycaret` so existing user code continues to run unchanged.

## TL;DR

```bash
pip uninstall pycaret
pip install pycaret-ng
```

```python
import pycaret  # unchanged
```

Same API. Modernized dependency floors. A short list of known-degraded plot/forecaster entries (see § 5 below).

## Why pycaret-ng

Upstream `pycaret==3.4.0` capped scikit-learn at `<1.5`, pandas at `<2.2`, and numpy at `<1.27`. Modernization across those caps stalled in upstream. pycaret-ng resumes that work in five phases — sklearn, pandas+numpy, plotting, time-series, release — with cherry-pickable commits where possible (Gate D), so upstream can adopt phases as their own roadmap permits.

## Install

| What you want | Command |
|---------------|---------|
| Base | `pip install pycaret-ng` |
| All extras | `pip install pycaret-ng[full]` |
| Time-series only | `pip install pycaret-ng[time_series]` |
| MLOps extras | `pip install pycaret-ng[mlops]` |

(Optional extras list mirrors upstream pycaret 3.4.0 — same names, same contents minus the modernization-blocked deps.)

## Modernized dependency floors

| Package | Before (pycaret 3.4.0) | After (pycaret-ng 1.0.0) | Phase that lifted it |
|---------|------------------------|--------------------------|----------------------|
| scikit-learn | `<1.5` | `>=1.6,<2` | Phase 1 |
| pandas | `<2.2` | `>=2.2,<3` | Phase 2 |
| numpy | `>=1.21,<1.27` | `>=1.26,<3` | Phase 2 |
| matplotlib | `<3.8` | `>=3.8` | Phase 3 |
| schemdraw | `==0.15` | `>=0.16` | Phase 3 |
| sktime | `>=0.31.0,<0.31.1` | `>=0.31` | Phase 4 |
| statsmodels | `>=0.12.1` | `>=0.14,<1` | Phase 4 |
| imbalanced-learn | `>=0.12,<0.14` | `>=0.14,<0.15` | Phase 1 |
| category-encoders | `>=2.4` | `>=2.7,<3` | Phase 1 |

Python: drops the `< 3.13` upper bound — pycaret-ng runs on 3.9, 3.10, 3.11, 3.12, 3.13.

## Known limitations under modern deps

These are documented in the per-phase DEGRADED registries (`docs/superpowers/agents/plotting-dev/DEGRADED.md` and `docs/superpowers/agents/ts-dev/DEGRADED.md`).

### Plot model

- `plot_model(plot="error")` (classification): yellowbrick's `ClassPredictionError` unpacks a 3-tuple from a sklearn helper whose return shape changed under sklearn ≥1.6. Internal to yellowbrick. Use `plot="confusion_matrix"` or `plot="class_report"` instead until yellowbrick releases a sklearn-1.6-aware version.
- `plot_model(plot="distance")` (clustering): yellowbrick's `InterclusterDistance` calls `np.percentile(..., interpolation=...)` which numpy ≥2 removed. Use a different cluster diagnostic (`plot="silhouette"` or `plot="elbow"`) until yellowbrick fixes upstream.
- `plot_model(plot="residuals_interactive")` (regression): requires the optional `anywidget` package. Either `pip install anywidget` or use `plot="residuals"` (static) instead.

### Time-series forecasters

- `bats`, `tbats`: graceful-disabled when `tbats` (numpy-1-only and unmaintained) cannot be imported. The container deactivates with a logger warning. Use `auto_arima`, `exp_smooth`, or `theta` instead.
- `auto_arima`: works fine on real workloads. The Phase 4 smoke skip-listed it because the default search space exceeds a 30-second budget — that's a smoke artifact, not a runtime concern.

### Latent (no-op today)

- pmdarima still calls sklearn's `force_all_finite=` kwarg internally. sklearn 1.6+ deprecated it; sklearn 1.8 removed it. sktime's current sklearn cap (<1.8) keeps this latent. The defensive shim is sketched but not wired; pycaret-ng v1.0.x will revive it when sklearn 1.8 becomes installable.
- joblib's `Memory.bytes_limit` — pinned `<1.5` for now; revives when the joblib 1.5 migration lands.

## Cherry-pick provenance

For users / forks tracking modernization commits independently:

| Phase | Commit prefix | Cherry-pick clean? |
|-------|---------------|---------------------|
| Phase 1 (sklearn) | `fix(sklearn)`, `feat(sklearn)` | yes |
| Phase 2 (pandas+numpy) | `fix(pandas)`, `feat(pandas)`, `fix(numpy)`, `feat(numpy)` | yes |
| Phase 3 (plotting) | `fix(plot)`, `feat(plot)` | yes (one mixed-scope commit noted in PR #3) |
| Phase 4 (time-series) | `fix(ts)`, `feat(ts)` | yes |
| Phase 5 (release) | `fix(release)` (Python guard, soft-dep gating only) | partial (`feat(release)` rename + version are pycaret-ng-only) |

The full per-phase commit list lives in each phase's spec at `docs/superpowers/specs/`. Each `fix(*)` / `feat(*)` commit on phases 1–4 was certified to apply onto upstream `pycaret/pycaret:master` (Gate D). Upstream PRs against `pycaret/pycaret` are not part of pycaret-ng's release process; opening them is a separate user-driven activity if desired.

## Roadmap

pycaret-ng v1.0.0 is the modernization-complete release. The master spec at `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` describes v1.1.0+ feature work (LLM phases 6.0–6.4): conversational SDK, EDA advisor, auto reports, LLM zoo estimators, MCP server. None of those land in v1.0.x.

## Compatibility commitment

- Existing user code that does `from pycaret.classification import setup, compare_models, ...` continues to work unchanged.
- Public API surface is unchanged from upstream 3.4.0 except for the documented degraded entries above.
- Semver: pycaret-ng v1.0.x for patch + small additive features; v1.1.0 introduces the LLM optional extras.
- pycaret-ng does *not* promise semver alignment with upstream pycaret — once upstream resumes releases, the version trees may diverge.
```

(Adjust the dep-floor table to match the *actual* current `pyproject.toml` floors; the values above are based on Phase 1–4 commits as captured at plan-write time.)

- [ ] **Step 3: Spot-check rendering**

```bash
head -50 MIGRATION.md
```

Expected: clean markdown, no broken table headers, no template `[fill in]` markers.

- [ ] **Step 4: Commit**

```bash
git add MIGRATION.md
git commit -m "$(cat <<'EOF'
docs(release): MIGRATION.md — upstream 3.4.0 -> pycaret-ng 1.0.0

User-facing migration guide at repo root. Sections:
  - TL;DR: install swap.
  - Why pycaret-ng: motivation.
  - Install: extras matrix.
  - Modernized dep floors: before/after table.
  - Known limitations: combined plotting + ts DEGRADED summary
    with user-actionable workarounds.
  - Cherry-pick provenance: per-phase Gate D status.
  - Roadmap: pointer to master spec for v1.1.0+ LLM phases.
  - Compatibility commitment: import path unchanged; semver.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PyPI publish workflow (W5)

**Files:**
- Modify: `.github/workflows/release.yml`

Adds the publish job. Workflow YAML only — pypi.org trusted-publisher configuration is user-action.

- [ ] **Step 1: Inspect the current release.yml**

```bash
cat .github/workflows/release.yml
```

Expected: a single `build` job that produces sdist + wheel and uploads as an artifact named `dist`.

- [ ] **Step 2: Append the publish job**

Add the following directly below the existing `build:` job (preserve the build job verbatim; the new publish job goes underneath it, replacing the `# publish job wired in Phase 5 with PyPI trusted publishing` comment):

```yaml
  publish:
    needs: build
    runs-on: ubuntu-latest
    name: publish to PyPI
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

- [ ] **Step 3: Validate YAML syntax**

```bash
.venv-phase3/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))" && echo 'YAML OK'
```

Expected: `YAML OK`. If yaml.YAMLError is raised, fix and re-run.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "$(cat <<'EOF'
feat(release): wire PyPI trusted publishing on tag push

Adds a publish job to release.yml that downloads the build artifact
and publishes via pypa/gh-action-pypi-publish@release/v1. Fires only
on tag pushes matching v*.*.* (matches the existing trigger).

Trusted publishing requires a one-time configuration on pypi.org —
register the pycaret-ng project and add the GitHub Actions trusted
publisher pointing at olaTechie/pycaret with this workflow file. That
configuration is user-action; the workflow is dormant until then.

The environment: pypi block prevents accidental publish from
forks-without-the-environment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Minimal README update (W7)

**Files:**
- Modify: `README.md`

Add a "this is pycaret-ng" header and update the install line. Full README rewrite is out of scope.

- [ ] **Step 1: Inspect the current install section**

```bash
grep -n 'pip install\|## Install\|^# ' README.md | head -20
```

Note the line numbers of the title (likely line 1) and the first `pip install pycaret` line.

- [ ] **Step 2: Add a fork-fact note above the project title**

Add a fenced callout at the very top of `README.md`, *above* the existing top-level `#` heading:

```markdown
> **Note:** This is `pycaret-ng`, a soft-fork of upstream [pycaret](https://github.com/pycaret/pycaret) modernized for Python 3.9–3.13, scikit-learn ≥ 1.6, pandas ≥ 2.2, numpy ≥ 2 (with backwards compat to 1.26 for tbats), and current sktime / matplotlib / yellowbrick. Install via `pip install pycaret-ng`. Migration guide: [MIGRATION.md](MIGRATION.md).

```

- [ ] **Step 3: Update the install line**

Find every `pip install pycaret` (without `-ng`) in `README.md`. For each, decide:
- If it's a CURRENT install instruction → replace with `pip install pycaret-ng`.
- If it's a HISTORICAL reference (e.g., a screenshot caption from old docs) → leave.

For most upstream READMEs, there will be 1–3 install lines to update.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(release): README — pycaret-ng install line + fork-fact callout

Minimal touch: callout at top of README pointing at MIGRATION.md and
naming the fork; install line updated from `pip install pycaret` to
`pip install pycaret-ng`. Full README rewrite is out of v1.0.0 scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Local build verification

**Files:** none (process step)

- [ ] **Step 1: Build sdist + wheel**

```bash
.venv-phase3/bin/pip install --quiet --upgrade build
.venv-phase3/bin/python -m build 2>&1 | tail -15
```

Expected: `Successfully built pycaret_ng-1.0.0.tar.gz and pycaret_ng-1.0.0-py3-none-any.whl` (or similar). `dist/` contains both files.

- [ ] **Step 2: Inspect wheel metadata**

```bash
.venv-phase3/bin/python -c "
import zipfile, glob
whl = glob.glob('dist/pycaret_ng-1.0.0-*.whl')[0]
with zipfile.ZipFile(whl) as z:
    meta = next(n for n in z.namelist() if n.endswith('METADATA'))
    print(z.read(meta).decode()[:600])
"
```

Expected: METADATA includes `Name: pycaret-ng`, `Version: 1.0.0`, classifiers including `Python :: 3.13`.

- [ ] **Step 3: Throwaway-venv install + import test**

```bash
uv venv .venv-throwaway --python 3.12
uv pip install --python .venv-throwaway/bin/python "$(ls dist/pycaret_ng-1.0.0-*.whl)"
.venv-throwaway/bin/python -c "import pycaret; print('version:', pycaret.__version__); print('package:', __import__('importlib.metadata').metadata.version('pycaret-ng'))"
rm -rf .venv-throwaway dist
```

Expected:

```
version: 1.0.0
package: 1.0.0
```

If the throwaway install fails (missing transitive deps because we're skipping `[full]`), that's expected — we're only verifying the package metadata + import path, not the full extras.

- [ ] **Step 4: No commit (this is a verification task only)**

If anything in steps 1-3 errored, fix the underlying issue (likely a typo in pyproject.toml) and re-run before proceeding.

---

## Task 9: Taxonomy + backlog refresh + final LOG (W8)

**Files:**
- Modify: `docs/superpowers/FAILURE_TAXONOMY.md`
- Modify: `docs/superpowers/MIGRATION_BACKLOG.md`
- Possibly Create / Modify: `docs/superpowers/agents/release/LOG.md`

- [ ] **Step 1: Close row 1 in the taxonomy**

Edit `docs/superpowers/FAILURE_TAXONOMY.md`. Find row 1 (the `pycaret.__init__` Python version guard). Change `Status` column from `open` to `closed`. Append closing note to `Notes`:

```
Closed by `<sha-of-task-1>` — replaces the dual `<3.9` / `>=3.13` RuntimeError with a single `<3.9` floor; modern dep floors implicitly gate the upper end.
```

- [ ] **Step 2: Close row 19 in the taxonomy**

Find row 19 (`fugue | daal4py | moto` collection errors). Change `Status` from `open` to `closed`. Append:

```
Closed by `<sha-of-task-2>` — three test files now use `pytest.importorskip` at module head; pytest emits a clean skip when the soft dep is absent.
```

- [ ] **Step 3: Refresh `MIGRATION_BACKLOG.md`**

Find the row-counts table. Update the `release` line to reflect post-Phase-5 reality:

```markdown
| release | 5 (IDs 1, 2, 10, 14, 19) | rows 1, 10, 19 closed in Phase 5; rows 2 (joblib FastMemorizedFunc — pin already in place from Phase 0) and 14 (joblib Memory.bytes_limit — latent until joblib >=1.5 reachable) remain |
```

(Row 10 was already closed by `a712ad17` in an earlier phase.)

- [ ] **Step 4: Append closure entry to the release agent log**

Check whether `docs/superpowers/agents/release/LOG.md` exists. If yes, append; if no, create.

```markdown
# Release Engineer — Log

Append-only progress log.

## 2026-05-06 — Phase 5 close
- Spec: `docs/superpowers/specs/2026-05-06-phase-5-release-design.md` (`3204a51e`).
- Plan: `docs/superpowers/plans/2026-05-06-phase-5-release.md`.
- Distribution rename + version bump (`<sha-of-task-3>`, `<sha-of-task-4>`): pycaret-ng 1.0.0.
- Python guard widened (`<sha-of-task-1>`): admits 3.13. Closes row 1.
- Soft-dep test gating (`<sha-of-task-2>`). Closes row 19.
- MIGRATION.md authored (`<sha-of-task-5>`).
- release.yml publish job wired (`<sha-of-task-6>`). Workflow dormant until pypi.org trusted-publisher configuration lands (user-action).
- README minimal touch (`<sha-of-task-7>`).
- Local build verification: `pycaret_ng-1.0.0-py3-none-any.whl` builds cleanly, METADATA correct, throwaway install resolves `import pycaret` to version 1.0.0.

## Outstanding (post-1.0.0 punch list)
- Row 2 (joblib FastMemorizedFunc): pin already in place, holds.
- Row 14 (joblib Memory.bytes_limit): activates when joblib < 1.5 cap can be lifted.
- Row 23 (anywidget for residuals_interactive): v1.0.x candidate.
- Phase 4 latent: pmdarima force_all_finite shim — activates when sklearn 1.8 reachable.
- Upstream PRs against pycaret/pycaret: separate user-driven activity.
- PyPI publish: register pycaret-ng on pypi.org + configure GitHub Actions trusted publisher; then push v1.0.0 tag.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/FAILURE_TAXONOMY.md docs/superpowers/MIGRATION_BACKLOG.md docs/superpowers/agents/release/LOG.md
git commit -m "$(cat <<'EOF'
docs(taxonomy,backlog,release): close rows 1/19 + Phase 5 closure

Phase 5 closure pass:
  - Row 1 (Python version guard): closed by <sha-of-task-1>.
  - Row 19 (soft-dep collection errors): closed by <sha-of-task-2>.
  - MIGRATION_BACKLOG release-agent line refreshed; rows 2 + 14 remain
    open as latent / blocked by external version constraints.
  - Release LOG records spec + plan SHAs, the closing SHAs for each
    workstream, and the post-1.0.0 punch list.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `<sha-of-task-N>` with actual SHAs when committing — collected from the prior tasks.)

---

## Task 10: Push + open PR (W9)

**Files:** none (process step)

- [ ] **Step 1: Cherry-pick scope sanity check**

```bash
for sha in $(git log --reverse --pretty=format:'%H' phase-4-timeseries..HEAD | xargs -I{} sh -c 'git log -1 --format="%H %s" {}' | grep -E '(fix|feat|chore)\(release\)' | awk "{print \$1}"); do
  echo "-- $sha --"
  git show --name-only --pretty=format: $sha | grep -v '^$'
done
```

Verify each `fix(release)` commit (Tasks 1, 2) touches only `pycaret/`, `tests/`, or `pyproject.toml`. The `feat(release)` rename + publish-job commits and `docs(release)` MIGRATION.md / README touch pycaret-ng-only paths and are exempt from Gate D.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin phase-5-release
```

- [ ] **Step 3: Write the PR body**

```bash
cat > .git/PR_BODY.md <<'EOF'
## Summary

Cuts **pycaret-ng v1.0.0**. Stacks on PRs #3 (Phase 3 plotting) and #4 (Phase 4 time-series); base will auto-retarget as those merge.

**Spec:** `docs/superpowers/specs/2026-05-06-phase-5-release-design.md`
**Plan:** `docs/superpowers/plans/2026-05-06-phase-5-release.md`

## What changes

- **Distribution rename:** `pyproject.toml` `name = "pycaret"` → `"pycaret-ng"`. Internal import path stays `pycaret` for drop-in compat.
- **Version bump:** 3.4.0 → 1.0.0.
- **Python guard widened** (closes FAILURE_TAXONOMY row 1): `pycaret/__init__.py` admits 3.13. Modern dep floors implicitly gate the upper end.
- **MIGRATION.md authored:** user-facing delta from upstream 3.4.0 to pycaret-ng 1.0.0 — install swap, dep-floor table, DEGRADED summary, cherry-pick provenance, compatibility commitment.
- **PyPI publish workflow** wired in `release.yml` (workflow YAML only; pypi.org trusted-publishing configuration is user-action; workflow is dormant until then).
- **Soft-dep test gating** (closes row 19): `pytest.importorskip` at module head of `test_classification_parallel.py`, `test_clustering_engines.py`, `test_persistence.py`.
- **README minimal touch:** fork-fact callout + `pip install pycaret-ng` line.

## What does NOT change

- Internal import path (`import pycaret as pc` works unchanged).
- Public API surface (modulo the documented degraded entries from Phases 3 + 4).
- Existing user code.

## Verification

- Local build: `python -m build` produces `pycaret_ng-1.0.0-*.whl` cleanly.
- Wheel metadata: `Name: pycaret-ng`, `Version: 1.0.0`, includes Python 3.13 classifier.
- Throwaway-venv install: `pip install dist/pycaret_ng-1.0.0-*.whl` then `import pycaret` reports version 1.0.0.
- Phase 3 plotting smoke (.venv-phase3): unchanged (38 passed, 3 skipped).
- Phase 4 TS smoke (.venv-phase4): unchanged (18 passed, 2 skipped).
- CI matrix on this PR: Python 3.13 row is the new addition.

## Outstanding (post-1.0.0)

- Register `pycaret-ng` on pypi.org and add the GitHub Actions trusted publisher (user-action). Then push the v1.0.0 tag.
- v1.0.x patch list: row 14 (joblib Memory.bytes_limit), row 23 (anywidget for residuals_interactive), Phase 4 pmdarima shim — all latent under current dep caps.

## Test plan

- [x] Local build green; wheel metadata correct.
- [x] Phase 3 + Phase 4 smokes regression-checked.
- [ ] CI matrix `{3.11, 3.12, 3.13} × {linux, macos}` green on this PR.
- [x] Taxonomy rows 1, 19 closed.
- [x] MIGRATION.md authored with concrete dep-floor table + DEGRADED summary.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

- [ ] **Step 4: Open the PR**

Per Phase 3 / Phase 4 lessons — `gh` defaults to upstream `pycaret/pycaret` repo, must override:

```bash
gh pr create --repo olaTechie/pycaret \
  --base phase-4-timeseries \
  --head olaTechie:phase-5-release \
  --title "Phase 5 — Release Engineering (pycaret-ng v1.0.0)" \
  --body-file .git/PR_BODY.md
```

The PR base targets `phase-4-timeseries` because PR #4 is open against that. When PR #4 merges, this PR's base auto-retargets up the stack.

- [ ] **Step 5: Cleanup + capture PR number**

```bash
rm .git/PR_BODY.md
```

Capture the PR number from the gh output (`https://github.com/olaTechie/pycaret/pull/<n>`) and append to release/LOG.md:

```markdown
## 2026-05-06 — Phase 5 PR open
- PR: https://github.com/olaTechie/pycaret/pull/<n>
- Awaiting CI matrix.
- Next action (user): pypi.org trusted-publisher configuration; tag v1.0.0; push tag.
```

Commit:

```bash
git add docs/superpowers/agents/release/LOG.md
git commit -m "docs(release): record Phase 5 PR URL

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push
```

---

## Self-Review Checklist

After all tasks, verify:
- [ ] `python -m build` produces `pycaret_ng-1.0.0-*.whl` cleanly with correct metadata.
- [ ] `pip install dist/pycaret_ng-1.0.0-*.whl` into throwaway venv, then `import pycaret` reports version `1.0.0`.
- [ ] Phase 3 plotting smoke still 38 passed, 3 skipped.
- [ ] Phase 4 TS smoke still 18 passed, 2 skipped.
- [ ] FAILURE_TAXONOMY rows 1, 19 are `closed` with closing SHAs in Notes.
- [ ] MIGRATION.md exists at repo root, ≥ 200 lines, includes the concrete dep-floor table.
- [ ] `.github/workflows/release.yml` has both `build` and `publish` jobs; `publish` runs `pypa/gh-action-pypi-publish@release/v1`.
- [ ] `pyproject.toml` has `name = "pycaret-ng"`, `version = "1.0.0"`, Python 3.13 classifier.
- [ ] PR opened against `phase-4-timeseries`.
- [ ] No commit message uses `--no-verify`, no commit skips hooks.

If any item fails, that's not Phase 5 done — fix before merging.
