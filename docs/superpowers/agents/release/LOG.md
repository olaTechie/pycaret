# Release Engineer — Log

Append-only progress log.

## 2026-05-06 — Phase 5 close

**Spec:** `docs/superpowers/specs/2026-05-06-phase-5-release-design.md` (`3204a51e`).
**Plan:** `docs/superpowers/plans/2026-05-06-phase-5-release.md` (`ff6ff43a`).

### Closed
- **Row 1** (Python version guard): closed by `a848daa7` — `pycaret/__init__.py` replaces dual `<3.9` / `>=3.13` RuntimeError with single `<3.9` floor. `pyproject.toml` `requires-python` lifted from `>=3.9,<3.13` to `>=3.9` in `d6ce570f`.
- **Row 19** (soft-dep collection errors): closed by `2ab98966` — `pytest.importorskip("fugue" / "daal4py" / "moto")` at module head of `test_classification_parallel.py`, `test_clustering_engines.py`, `test_persistence.py`. Verified on `.venv-phase4`: clean skip when dep absent.

### Distribution work
- **Distribution rename + version bump** (`d6ce570f`): `pyproject.toml` name `"pycaret"` → `"pycaret-ng"`, version `"3.4.0"` → `"1.0.0"`, description updated, `Python :: 3.13` classifier added. Internal import path stays `pycaret`.
- **`pycaret/__init__.py` `version_`** also bumped to `"1.0.0"` (combined with row 1 fix in `a848daa7`).
- **`_show_versions.py` probe**: no edit required. The `"pycaret"` string in `required_deps` is the *import path* (unchanged), not the distribution name.
- **MIGRATION.md** (`45cb8976`): user-facing 116-line guide at repo root — install swap, dep-floor table, combined plotting + ts DEGRADED summary, cherry-pick provenance, compatibility commitment.
- **`release.yml` publish job** (`47655242`): added `publish` job that fires on tag push, downloads the build artifact, and publishes via `pypa/gh-action-pypi-publish@release/v1`. Workflow dormant until pypi.org trusted-publisher configuration (user-action).
- **README.md** (`14b39e15`): top-of-file fork-fact callout, header tagline updated, all `pip install pycaret[<extra>]` lines updated to `pycaret-ng`, supported Python list extended to 3.13.

### Local build verification
Sequence run on `.venv-phase4`:
```
.venv-phase4/bin/python -m build
# → Successfully built pycaret_ng-1.0.0.tar.gz and pycaret_ng-1.0.0-py3-none-any.whl
```

Wheel METADATA:
- `Name: pycaret-ng`
- `Version: 1.0.0`
- `Summary: pycaret-ng — modernized soft-fork of PyCaret (low-code ML for Python 3.9–3.13, ...)`
- `Programming Language :: Python :: 3.13` ✓

Throwaway-venv install + import (run from `/tmp` to avoid source-tree shadowing):
```
import pycaret  # version: 1.0.0
importlib.metadata.metadata("pycaret-ng")["Name"]  # pycaret-ng
importlib.metadata.version("pycaret-ng")  # 1.0.0
```

### Phase 3 + Phase 4 regression checks
- Phase 3 plotting smoke (`.venv-phase3`): unchanged (38 passed, 3 skipped).
- Phase 4 TS smoke (`.venv-phase4`): unchanged (18 passed, 2 skipped).

### Outstanding (post-1.0.0 punch list)
- **Row 2** (joblib `FastMemorizedFunc.func_id`): pin already in place from Phase 0; holds.
- **Row 14** (joblib `Memory.bytes_limit`): activates when joblib `<1.5` cap can be lifted.
- **Row 23** (anywidget for `residuals_interactive`): v1.0.x candidate — needs decision on whether to add as a dep, lazy-import + degrade, or document as user opt-in.
- **Phase 4 latent** (pmdarima `force_all_finite` shim): activates when sklearn 1.8 becomes installable. Currently sktime 0.40.1 caps sklearn `<1.8`; sklearn 1.7.x still accepts the deprecated kwarg with a FutureWarning.
- **Upstream PRs against `pycaret/pycaret`**: separate user-driven activity. Each Phase 1–4 `fix(*)`/`feat(*)` commit was certified cherry-pickable per Gate D; opening PRs upstream is not part of pycaret-ng's release process.
- **PyPI publish**: user-action — register `pycaret-ng` on pypi.org, configure GitHub Actions trusted publisher pointing at `olaTechie/pycaret` `release.yml`, then `git tag v1.0.0 && git push --tags`.

### Master spec § 4 v1.1.0+
LLM phases (6.0 conversational SDK, 6.1 EDA advisor, 6.2 auto reports, 6.3 LLM zoo estimators, 6.4 MCP server) become the active roadmap once v1.0.0 is published.

## 2026-05-06 — Phase 5 PR open
- PR: https://github.com/olaTechie/pycaret/pull/5 (base `phase-4-timeseries`; auto-retargets up the stack as PRs #3, #4 merge).
- Branch state: 9 commits on `phase-5-release` since `phase-4-timeseries`.
- Awaiting CI matrix.
- **Next user actions:**
  1. Watch CI on PRs #3, #4, #5 (in stack order).
  2. Merge PRs as they go green.
  3. Once `modernize` includes Phase 5, configure pypi.org trusted publisher for `pycaret-ng` pointing at `olaTechie/pycaret` `release.yml`.
  4. Tag `v1.0.0` and push the tag — `release.yml` builds and (with pypi side configured) publishes.

## 2026-05-06 — Row 23 (anywidget) closed
- Closed by `4cab8593` — lazy-import + NotImplementedError at `tabular_experiment.py:residuals_interactive` with a clear `pip install anywidget` / use `plot='residuals'` hint.
- Matches the master spec's fix-or-disable fallback policy (b); fail-loud, not silent.
- DEGRADED.md (plotting-dev) row added; MIGRATION.md "Known limitations" updated; FAILURE_TAXONOMY row 23 marked closed.
- Smokes unchanged after the fix: plotting 38 passed / 3 skipped, TS 18 passed / 2 skipped.
- Pushed to PR #5; the fix is part of the v1.0.0 release.

## Outstanding (revised post-anywidget-close)
- **Row 2** (joblib `FastMemorizedFunc`): pin already in place, holds.
- **Row 14** (joblib `Memory.bytes_limit`): activates when joblib `<1.5` cap can be lifted — v1.0.x candidate.
- **Phase 4 latent** (pmdarima `force_all_finite` shim): activates when sklearn 1.8 reachable — v1.0.x candidate.
- All other rows from FAILURE_TAXONOMY are now closed or degraded.

## 2026-05-06 — CI unblock pass

First CI run on PRs #3, #4, #5 came back fully red. Two distinct failure modes:

1. **Python 3.11 / 3.12 (linux + macos):** `TypeError: Descriptors cannot be created directly` — mlflow's transitive `opentelemetry _pb2.py` modules clash with protobuf ≥4 under the C++ implementation. Fix landed as `cfe7708c` on `phase-5-release`: top-of-`tests/conftest.py` sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before any pycaret/mlflow import. Mirrors what `tests/smoke/conftest.py` already does.

2. **Python 3.13 (linux + macos):** `htmlmin@0.1.12` build fails — Python 3.13 removed the `cgi` module; `htmlmin` 0.1.12 is unmaintained. uv resolved an older `ydata-profiling` on 3.13 that still required `htmlmin`. Fix in same commit: bump `ydata-profiling` floor in both `full` and `analysis` extras from `>=4.3.1` to `>=4.18` (4.18.4 verified to have no `htmlmin` requirement).

Cherry-picked `cfe7708c` to `phase-3-plotting` (`a0cdc5fa`) and `phase-4-timeseries` (`f053b022`) so all three open PRs pick up the fix on their next CI run. Both auto-merged on the pyproject.toml hunk.

Local smokes unchanged: plotting 38/3, ts 18/2 (combined 56/5 in 24.4s).

## 2026-05-06 — CI unblock pass round 2

Round 1 fixed install-time errors. Round 2 addresses follow-on issues that surfaced once tests actually ran:

1. **Python 3.11/3.12 (linux+macos): parity collection error.** `tests/parity/test_baseline.py` and three siblings did `from tests.parity.baseline import ...`. `tests/__init__.py` doesn't exist, so `tests` isn't a discoverable package. Fixed by switching all four files to relative imports (`from .baseline import ...`). Could've added `tests/__init__.py` instead, but relative imports are scoped surgically and don't risk surprising pytest's collection elsewhere. Local `pytest --collect-only tests/parity/` collects 17 tests cleanly.

2. **Python 3.13 (linux+macos): pmdarima + scipy build failures.** Two distinct sub-issues:
   - `pmdarima 2.0.4` build fails under uv's strict PEP 517 isolation — its `setup.py` imports numpy without declaring it as a build dep. Fixed by adding `[tool.uv.extra-build-dependencies]` for pmdarima with `numpy`, `setuptools`, `cython`.
   - `scipy<=1.11.4` (the "to fix later" upstream cap) has no Python 3.13 wheel. Fixed by lifting to `>=1.11,<2` — closes that comment.

Verified locally with a fresh `.venv-py313`: full install resolves cleanly. Resulting versions on Python 3.13: pmdarima 2.1.1 (jumped from 2.0.4 because uv could now build the latest), numpy 2.3.5 (numpy 2!), scipy 1.16.3, tbats 1.1.3 (installs but graceful-disable activates under numpy 2). htmlmin not pulled in.

Cherry-picked `422472de` to `phase-3-plotting` (`729ff43a`) and `phase-4-timeseries` (`3d636100`). Smokes unchanged.

The Python 3.13 path now exercises the Phase 4 graceful-disable code path for tbats — the dormancy from Phase 4 (numpy 1.26 picked locally) flips to active when 3.13 forces numpy 2.
