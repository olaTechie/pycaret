# Local development & testing — pycaret-ng v1.0.0

This guide is for working with the modernized `pycaret-ng` source tree locally. For end-user install / migration from upstream `pycaret==3.4.0`, see [MIGRATION.md](MIGRATION.md). For the master modernization roadmap, see [`docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md`](docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md).

## Repository layout

| Path | Purpose |
|------|---------|
| `pycaret/` | Source. Internal import path is unchanged from upstream. |
| `tests/` | Full pytest suite (hundreds of tests; ≥30 min on CI). |
| `tests/smoke/` | **Fast verification harness** added by pycaret-ng (≈25 s, 56 tests). |
| `docs/superpowers/` | Per-phase specs, plans, agent logs, taxonomy, DEGRADED registries. |
| `MIGRATION.md` | Upstream 3.4.0 → pycaret-ng 1.0.0 user-facing delta. |
| `pyproject.toml` | Modernized dep floors. `name = "pycaret-ng"`, `version = "1.0.0"`. |

## Development venvs

Two pre-built venvs are committed-ignored:

- `.venv-phase3` — Python 3.12.13, **without** sktime. Use for plotting smoke and any non-time-series work.
- `.venv-phase4` — Python 3.12.13, **with** the full `[full]` extras (sktime, pmdarima, tbats, statsmodels, ydata-profiling, mlflow, fugue, ray, etc.).

> ⚠️ **Do not use `source .venv-phase*/bin/activate`.** The activate script's `PATH` was baked in with a Dropbox CloudStorage path alias that no longer resolves; after `source activate`, `which python` falls back to anaconda's 3.13, which trips pycaret's runtime guard. Always invoke the binary directly: `.venv-phase4/bin/python ...`.

## Workflow 1 — fast smoke (≈25 s)

The smoke harness is the load-bearing local verification. It exercises every entry in `_available_plots` for classification / regression / clustering, and every forecaster family in time-series. Pass condition is "no exception raised".

```bash
cd /Users/uthlekan/Library/CloudStorage/Dropbox/00_ToReview/10_PluginSkills/pycaret

# Plotting smoke (Phase 3 venv — no sktime needed)
.venv-phase3/bin/python -m pytest \
    --confcutdir=tests/smoke \
    tests/smoke/test_plotting.py -v

# Time-series smoke (Phase 4 venv — needs sktime, pmdarima, tbats)
.venv-phase4/bin/python -m pytest \
    --confcutdir=tests/smoke \
    tests/smoke/test_time_series.py -v

# Both at once on Phase 4
.venv-phase4/bin/python -m pytest \
    --confcutdir=tests/smoke \
    tests/smoke/ -v
```

**Expected:** 56 passed, 5 skipped, ≈25 s. The 5 skips are documented degradations:
- classification `error` (yellowbrick `ClassPredictionError` 3-tuple unpack vs sklearn ≥1.6)
- clustering `distance` (yellowbrick `np.percentile(interpolation=)` vs numpy ≥2)
- regression `residuals_interactive` (anywidget not in base deps)
- TS `auto_arima` × 2 (sktime 0.40.1 wider grid search exceeds 30 s budget)

All five are tracked in `docs/superpowers/agents/{plotting-dev,ts-dev}/DEGRADED.md` with restoration criteria.

The `--confcutdir=tests/smoke` flag is **required** — it bypasses the root `tests/conftest.py`, which eagerly imports `pycaret.time_series` → `sktime`. Without it, plotting smoke would also need sktime installed.

## Workflow 2 — install pycaret-ng into a fresh venv

```bash
cd /Users/uthlekan/Library/CloudStorage/Dropbox/00_ToReview/10_PluginSkills/pycaret

# Build sdist + wheel
.venv-phase4/bin/python -m build
ls dist/
# pycaret_ng-1.0.0-py3-none-any.whl
# pycaret_ng-1.0.0.tar.gz

# Throwaway venv install
uv venv .venv-test --python 3.12
uv pip install --python .venv-test/bin/python "$(ls dist/pycaret_ng-1.0.0-*.whl)"

# Sanity-check import + version (run from /tmp to avoid the source-tree shadowing the import)
cd /tmp
$OLDPWD/.venv-test/bin/python -c "
import pycaret
import importlib.metadata as m
print('import path version:', pycaret.__version__)
print('PyPI metadata name:', m.metadata('pycaret-ng')['Name'])
print('PyPI metadata version:', m.version('pycaret-ng'))
"
# Expected:
#   import path version: 1.0.0
#   PyPI metadata name: pycaret-ng
#   PyPI metadata version: 1.0.0
```

To run a real classification flow against the wheel:

```bash
$OLDPWD/.venv-test/bin/python -c "
from pycaret.classification import setup, compare_models
from pycaret.datasets import get_data
data = get_data('juice', verbose=False)
exp = setup(data, target='Purchase', html=False, verbose=False, n_jobs=1, fold=2)
best = compare_models(include=['rf', 'lr', 'dt'], n_select=1, verbose=False)
print('best model:', type(best).__name__)
"
```

Cleanup: `rm -rf .venv-test dist/`.

## Workflow 3 — targeted subsets of the full pytest suite

The full `pytest tests/` runs hundreds of tests across all task families and exceeds 60 min on free-tier GitHub runners. Pin specific subsets with `-k` and `--timeout`.

```bash
# Classification core (~5–10 min)
.venv-phase4/bin/python -m pytest tests/test_classification.py -v --timeout=300

# Regression core (~5–10 min)
.venv-phase4/bin/python -m pytest tests/test_regression.py -v --timeout=300

# Clustering core (~3–5 min)
.venv-phase4/bin/python -m pytest tests/test_clustering.py -v --timeout=300

# TS setup + create_model (~10 min)
.venv-phase4/bin/python -m pytest tests/test_time_series_setup.py tests/test_time_series_models.py -v --timeout=300

# Persistence + sklearn-compat shims (fast, ~30 s)
.venv-phase4/bin/python -m pytest tests/test_persistence.py tests/test_sklearn_compat.py -v
```

`--timeout=300` (5 min per test) prevents hung individual tests from blocking the whole subset. Tests requiring optional deps (`fugue`, `daal4py`, `moto`) skip cleanly via `pytest.importorskip` at module head.

## Workflow 4 — quick sanity (≈30 s, no fixtures)

If you only want to confirm the code imports and one classifier trains:

```bash
.venv-phase4/bin/python -c "
import pycaret
from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data
exp = ClassificationExperiment()
exp.setup(get_data('iris', verbose=False), target='species',
          html=False, verbose=False, n_jobs=1, fold=2)
m = exp.create_model('rf', n_estimators=5, max_depth=2, verbose=False)
print('OK — pycaret-ng', pycaret.__version__, 'works on iris with', type(m).__name__)
"
```

## Common gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: pycaret-ng requires Python >= 3.10` | Activated venv with stale Dropbox path → falls back to anaconda 3.13 (which then trips the import path's own guard) | Invoke `.venv-phase4/bin/python` directly. Never `source activate`. |
| `ModuleNotFoundError: No module named 'sktime'` when running plotting smoke | Root `tests/conftest.py` eagerly imports pycaret.time_series | Add `--confcutdir=tests/smoke` to the pytest invocation. |
| `TypeError: Descriptors cannot be created directly` (protobuf) | mlflow's transitive `_pb2.py` modules incompatible with protobuf ≥4 | Already handled in `tests/smoke/conftest.py` and root `tests/conftest.py` by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. If you bypass the conftest, set the env var manually before importing pycaret. |
| `NotImplementedError: plot='residuals_interactive' requires the optional anywidget` | Plot raise from `tabular_experiment.py` for missing optional dep | `pip install anywidget` or use `plot='residuals'`. |
| Smoke test `auto_arima` skipped | Wider sktime 0.40.1 search space exceeds 30 s budget | Real workloads are unaffected — this is a smoke-budget skip only. |
| `ValueError: tbats / BATS forecaster disabled` (logger warning) | tbats is unmaintained and numpy-1-only; under numpy ≥2 the import or instantiation fails | Use `auto_arima`, `exp_smooth`, or `theta` instead. |

## Cherry-pick discipline (Gate D)

If you're upstreaming fixes to `pycaret/pycaret`, use only these commit prefixes:

- `fix(sklearn): …`, `fix(pandas): …`, `fix(numpy): …`, `fix(plot): …`, `fix(ts): …` — apply onto upstream master cleanly assuming earlier-phase fixes are in.
- `feat(*): …` — same.

Skip these (pycaret-ng-only):

- `test(smoke): …` — new test infrastructure.
- `docs(*): …` — agent logs, taxonomy, this guide.
- `feat(release): rename distribution to pycaret-ng + bump to 1.0.0` (`d6ce570f`) and the publish workflow — fork-specific.
- The Python-guard widening + `requires-python` lift — Phase 5-only.

The full per-phase commit list lives in each phase's spec at `docs/superpowers/specs/`.

## CI status (as of v1.0.0 ship)

Local smokes are green. The full pytest matrix on CI was reaching 60+ minutes during the v1.0.0 prep and got cancelled — a test-suite scoping issue, not a modernization regression. v1.0.x patch releases will tackle this with `pytest -m "fast"` / `pytest -m "slow"` markers and a nightly cron for the slow tier.

See `docs/superpowers/agents/release/LOG.md` for the full round-by-round CI iteration history.

## Phase work outside this guide

The five phases (sklearn, pandas+numpy, plotting, time-series, release) each have a spec + plan + agent log. The roadmap beyond v1.0.0 is the LLM phases (6.0–6.4) in the master spec — not yet started.
