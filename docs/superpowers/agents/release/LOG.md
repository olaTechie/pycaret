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
