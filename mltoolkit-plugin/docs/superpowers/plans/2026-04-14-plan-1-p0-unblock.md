# Plan 1 — P0 Unblock: packaging correctness + truth-in-advertising

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three P0 findings that block any other work: fix the silently-broken copy-to-`.mltoolkit/session.py` workflow (LEAD-001), implement or retract the Optuna tuner claim (LEAD-002), and add end-to-end CI coverage so this class of bug cannot regress (LEAD-023).

**Architecture:** Introduce a small staging helper (`mltoolkit-plugin/scripts/stage_session.py`) that copies a reference script *and* its sibling modules (`preprocessing.py`, `model_zoo.py`) *and* the `_shared/` package into `.mltoolkit/` in the user's CWD, producing a self-contained session bundle. Every reference script is refactored to a single, staging-compatible import header so imports resolve whether the script runs in-place (tests) or from `.mltoolkit/` (user workflow). Optuna support is implemented in the tune stage of classify and regress reference scripts, gated on `deps.has_optuna()` with a clean RandomizedSearchCV fallback. A new `tests/test_stage_and_run.py` exercises the full copy→execute path per reference, and `tests/test_references.sh` runs it.

**Tech Stack:** Python 3.9+, stdlib (`pathlib`, `shutil`, `textwrap`, `argparse`), existing sklearn + optional `optuna`, existing pytest infra.

**Source findings:** `mltoolkit-plugin/docs/superpowers/reviews/2026-04-14-findings.jsonl` → LEAD-001, LEAD-002, LEAD-023.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `mltoolkit-plugin/scripts/stage_session.py` | Stage a `<task>_reference.py` + its siblings + `_shared/` into `<dest>/` (default `.mltoolkit/`). Also rewrites the staged `session.py` header to the staging-compatible layout. |
| `mltoolkit-plugin/references/_shared/tuning_optuna.py` | Optuna-backed hyperparameter search callable. Only imported when `deps.has_optuna()`. |
| `mltoolkit-plugin/tests/test_stage_and_run.py` | Copy each reference via the stager into a tmp dir and run `--stage all` — this is the class-of-bug gate that LEAD-023 asks for. |

**Modified:**

| File | Change |
|---|---|
| `mltoolkit-plugin/skills/classify/references/classify_reference.py` | Replace `_HERE.parents[3]` path magic with staging-compatible header (works in-place *and* after staging). Wire `--search-library` flag into `tune_model`. |
| `mltoolkit-plugin/skills/regress/references/regress_reference.py` | Same header refactor. Wire `--search-library` into its tune stage. |
| `mltoolkit-plugin/skills/cluster/references/cluster_reference.py` | Same header refactor. No optuna (clustering uses direct `create_model` per skill). |
| `mltoolkit-plugin/skills/anomaly/references/anomaly_reference.py` | Same header refactor. No optuna. |
| `mltoolkit-plugin/skills/classify/references/preprocessing.py` | Same header refactor (it currently also uses `_HERE.parents[3]`). |
| `mltoolkit-plugin/skills/regress/references/preprocessing.py` | Same header refactor. |
| `mltoolkit-plugin/skills/classify/SKILL.md` | Step 3 changes from "Copy the reference (use Write tool)" to "Stage the reference via `python scripts/stage_session.py …`". |
| `mltoolkit-plugin/skills/regress/SKILL.md` | Same edit. |
| `mltoolkit-plugin/skills/cluster/SKILL.md` | Same edit. |
| `mltoolkit-plugin/skills/anomaly/SKILL.md` | Same edit. |
| `mltoolkit-plugin/skills/tune/SKILL.md` | Replace the aspirational "Optuna support" section with concrete `--search-library {sklearn,optuna}` flag docs. |
| `mltoolkit-plugin/README.md` | Change "optuna — alternative hyperparameter search backend" into a sentence that actually works (keep as optional, but now grounded). |
| `mltoolkit-plugin/tests/test_references.sh` | Add a fifth step invoking `test_stage_and_run.py`. |
| `mltoolkit-plugin/tests/test_classify.py` | Keep existing in-place tests; they are still valid since the refactored header works in-place too. |

**Not created / not touched this plan:** all other `_shared/` modules (fairness, calibration, bootstrap, splits, …) — those belong to Plan 2.

---

## Task 1: Staging helper — `scripts/stage_session.py`

**Files:**
- Create: `mltoolkit-plugin/scripts/stage_session.py`
- Test: `mltoolkit-plugin/tests/test_stage_and_run.py` (unit portion; e2e rows added in Task 6)

- [ ] **Step 1: Write the failing unit test for the stager shape**

Create `mltoolkit-plugin/tests/test_stage_and_run.py` with this content:

```python
"""Stager + end-to-end copy-and-run tests for every reference script."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGER = REPO_ROOT / "scripts/stage_session.py"


def _run_stager(task: str, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAGER), "--task", task, "--dest", str(dest)],
        capture_output=True, text=True,
    )


def test_stager_exists():
    assert STAGER.exists(), f"Stager missing at {STAGER}"


def test_stager_classify_creates_expected_layout(tmp_path):
    dest = tmp_path / "mlt"
    r = _run_stager("classify", dest)
    assert r.returncode == 0, f"stager failed: {r.stderr}"
    # Required files in the staged bundle
    assert (dest / "session.py").is_file()
    assert (dest / "preprocessing.py").is_file()
    assert (dest / "model_zoo.py").is_file()
    assert (dest / "_shared/__init__.py").is_file()
    assert (dest / "_shared/deps.py").is_file()
    assert (dest / "_shared/plotting.py").is_file()
    assert (dest / "_shared/reporting.py").is_file()


def test_stager_rewrites_header_in_session_py(tmp_path):
    dest = tmp_path / "mlt"
    _run_stager("classify", dest).check_returncode()
    text = (dest / "session.py").read_text()
    # The original import-from-plugin-root pattern must be gone
    assert "references._shared" not in text
    assert "_PLUGIN_ROOT" not in text
    # The staged pattern must be present
    assert "from _shared import" in text


def test_stager_refuses_unknown_task(tmp_path):
    r = _run_stager("not_a_task", tmp_path / "mlt")
    assert r.returncode != 0
    assert "unknown task" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py -v`
Expected: 4 failures — all import or path errors because the stager does not exist yet.

- [ ] **Step 3: Implement `scripts/stage_session.py`**

Create `mltoolkit-plugin/scripts/stage_session.py` with exactly this content:

```python
"""Stage a reference script + its dependencies into a self-contained bundle.

After staging, `<dest>/session.py` imports work without plugin-root path magic
because its siblings (`preprocessing.py`, `model_zoo.py`) and the `_shared/`
package are co-located.

Usage:
    python scripts/stage_session.py --task classify --dest .mltoolkit

Known tasks: classify, regress, cluster, anomaly, setup.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]

# Sibling modules each task reference imports directly.
# Order matters only for documentation; shutil.copy handles files atomically.
TASK_SIBLINGS = {
    "classify": ["classify_reference.py", "preprocessing.py", "model_zoo.py"],
    "regress":  ["regress_reference.py",  "preprocessing.py", "model_zoo.py"],
    "cluster":  ["cluster_reference.py",  "model_zoo.py"],
    "anomaly":  ["anomaly_reference.py",  "model_zoo.py"],
    "setup":    ["setup_reference.py"],
}


def _sources_for(task: str) -> list[Path]:
    try:
        names = TASK_SIBLINGS[task]
    except KeyError as exc:
        raise SystemExit(f"stage_session: unknown task '{task}'. "
                         f"Known: {sorted(TASK_SIBLINGS)}") from exc
    skill_refs = PLUGIN_ROOT / "skills" / task / "references"
    return [skill_refs / n for n in names]


def _rewrite_session_header(dest_session: Path) -> None:
    """Replace the in-place import header with a staging-compatible one.

    The in-place pattern looks like:
        _HERE = Path(__file__).resolve()
        _PLUGIN_ROOT = _HERE.parents[3]
        sys.path.insert(0, str(_PLUGIN_ROOT))
        sys.path.insert(0, str(_HERE.parent))
        from references._shared import plotting, reporting  # noqa: E402
        import preprocessing  # noqa: E402
        import model_zoo  # noqa: E402

    After staging, everything is co-located under <dest>/, so the header
    collapses to:
        _HERE = Path(__file__).resolve()
        sys.path.insert(0, str(_HERE.parent))
        from _shared import plotting, reporting  # noqa: E402
        import preprocessing  # noqa: E402
        import model_zoo  # noqa: E402
    """
    text = dest_session.read_text()
    # Kill the plugin-root insertion
    text = text.replace(
        "_PLUGIN_ROOT = _HERE.parents[3]\n"
        "sys.path.insert(0, str(_PLUGIN_ROOT))\n",
        "",
    )
    # Remap any reference to the outer package to the staged package
    text = text.replace("from references._shared", "from _shared")
    dest_session.write_text(text)


def stage(task: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    sources = _sources_for(task)
    # The first source is the primary reference — always staged as session.py.
    primary = sources[0]
    dest_session = dest / "session.py"
    shutil.copy2(primary, dest_session)

    # Siblings stay under their own names so `import preprocessing` etc. works.
    for src in sources[1:]:
        shutil.copy2(src, dest / src.name)

    # Copy _shared/ as a package.
    shared_src = PLUGIN_ROOT / "references/_shared"
    shared_dst = dest / "_shared"
    if shared_dst.exists():
        shutil.rmtree(shared_dst)
    shutil.copytree(shared_src, shared_dst, ignore=shutil.ignore_patterns("__pycache__"))

    _rewrite_session_header(dest_session)
    # Also rewrite headers in sibling modules (they import _shared too).
    for src in sources[1:]:
        _rewrite_session_header(dest / src.name)

    return dest_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASK_SIBLINGS),
                    help="Which task's reference to stage.")
    ap.add_argument("--dest", default=".mltoolkit",
                    help="Destination directory (default: .mltoolkit).")
    args = ap.parse_args()
    session = stage(args.task, Path(args.dest))
    print(f"Staged {args.task} → {session}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests again to verify they pass — EXCEPT the header-rewrite test which depends on Task 2**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py::test_stager_exists tests/test_stage_and_run.py::test_stager_classify_creates_expected_layout tests/test_stage_and_run.py::test_stager_refuses_unknown_task -v`
Expected: 3 PASS. The header-rewrite test will still FAIL because the reference script has not yet been updated to use the in-place pattern the rewriter expects. That is fine — Task 2 fixes it.

- [ ] **Step 5: Commit**

```bash
cd mltoolkit-plugin
git add scripts/stage_session.py tests/test_stage_and_run.py
git commit -m "feat(mltoolkit): stage_session helper — copy reference + siblings + _shared into self-contained bundle (LEAD-001 step 1/2)"
```

---

## Task 2: Staging-compatible headers in every reference script

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py:35-42`
- Modify: `mltoolkit-plugin/skills/regress/references/regress_reference.py` (same header block)
- Modify: `mltoolkit-plugin/skills/cluster/references/cluster_reference.py` (same header block)
- Modify: `mltoolkit-plugin/skills/anomaly/references/anomaly_reference.py` (same header block)
- Modify: `mltoolkit-plugin/skills/classify/references/preprocessing.py:11-17`
- Modify: `mltoolkit-plugin/skills/regress/references/preprocessing.py` (equivalent header)

**Why this task exists:** the rewriter in Task 1 does a simple text substitution. For that to work cleanly, every reference file must share the same canonical header pattern (already true for classify, verify for the others; regress/cluster/anomaly may have minor drift).

- [ ] **Step 1: Confirm the rewriter test fails as expected**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py::test_stager_rewrites_header_in_session_py -v`
Expected: PASS already if classify_reference matches the rewriter pattern (it does per the read shown in the design), OR FAIL with "references._shared still present" — either way use the result to confirm whether more than just a pattern-match is needed.

- [ ] **Step 2: Normalize the header in classify_reference.py**

Open `mltoolkit-plugin/skills/classify/references/classify_reference.py`. Replace lines 35-42 (the existing header block) with exactly this block — this is the canonical in-place pattern the rewriter understands:

```python
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402
```

(If the file already matches, leave it alone.)

- [ ] **Step 3: Normalize the header in regress_reference.py**

Open `mltoolkit-plugin/skills/regress/references/regress_reference.py`, locate its path-magic block (near the top, above the first stage function), and replace with the same canonical block as Step 2. `regress` uses the same sibling modules (`preprocessing`, `model_zoo`).

- [ ] **Step 4: Normalize the header in cluster_reference.py**

Open `mltoolkit-plugin/skills/cluster/references/cluster_reference.py`. Cluster does not import `preprocessing`; replace with:

```python
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402
```

- [ ] **Step 5: Normalize the header in anomaly_reference.py**

Open `mltoolkit-plugin/skills/anomaly/references/anomaly_reference.py`. Same shape as cluster (no preprocessing):

```python
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402
```

- [ ] **Step 6: Normalize the header in the shared preprocessing.py files**

Open `mltoolkit-plugin/skills/classify/references/preprocessing.py`. Replace lines 11-17 with:

```python
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from references._shared import deps  # noqa: E402
```

Do the same in `mltoolkit-plugin/skills/regress/references/preprocessing.py`.

- [ ] **Step 7: Extend the rewriter to handle `preprocessing.py` header too**

Open `mltoolkit-plugin/scripts/stage_session.py` and replace the `_rewrite_session_header` body with this more robust version (handles both main-script and preprocessing-script shapes):

```python
def _rewrite_session_header(dest_file: Path) -> None:
    """Rewrite in-place import header → staging-compatible header."""
    text = dest_file.read_text()
    # Drop the plugin-root sys.path insertion (both the bare and the guarded variants).
    text = text.replace(
        "_PLUGIN_ROOT = _HERE.parents[3]\n"
        "sys.path.insert(0, str(_PLUGIN_ROOT))\n",
        "",
    )
    text = text.replace(
        "_PLUGIN_ROOT = _HERE.parents[3]\n"
        "if str(_PLUGIN_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_PLUGIN_ROOT))\n",
        "",
    )
    # Remap outer package → co-located _shared package.
    text = text.replace("from references._shared", "from _shared")
    dest_file.write_text(text)
```

- [ ] **Step 8: Verify all stager tests now pass**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 9: Commit**

```bash
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py \
        skills/regress/references/regress_reference.py \
        skills/cluster/references/cluster_reference.py \
        skills/anomaly/references/anomaly_reference.py \
        skills/classify/references/preprocessing.py \
        skills/regress/references/preprocessing.py \
        scripts/stage_session.py
git commit -m "fix(mltoolkit): canonical in-place header across references + robust rewriter (LEAD-001 step 2/2)"
```

---

## Task 3: SKILL.md updates — every task skill invokes the stager

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/SKILL.md:29` (Step 3 of its Workflow)
- Modify: `mltoolkit-plugin/skills/regress/SKILL.md` (equivalent step)
- Modify: `mltoolkit-plugin/skills/cluster/SKILL.md` (equivalent step)
- Modify: `mltoolkit-plugin/skills/anomaly/SKILL.md` (equivalent step)

- [ ] **Step 1: Replace the copy-to-session step in classify SKILL.md**

Open `mltoolkit-plugin/skills/classify/SKILL.md`. Find the current Step 3 (line 29):

> `3. **Copy the reference** to `.mltoolkit/session.py` (use Write tool).`

Replace with:

```markdown
3. **Stage the reference bundle** into `.mltoolkit/`:
   `python {PLUGIN_ROOT}/scripts/stage_session.py --task classify --dest .mltoolkit`
   This copies `classify_reference.py` as `session.py`, plus the sibling modules (`preprocessing.py`, `model_zoo.py`) and the `_shared/` package, all co-located so the script runs standalone.
```

- [ ] **Step 2: Replace the equivalent step in regress SKILL.md**

Open `mltoolkit-plugin/skills/regress/SKILL.md`, find the corresponding step, replace with:

```markdown
3. **Stage the reference bundle** into `.mltoolkit/`:
   `python {PLUGIN_ROOT}/scripts/stage_session.py --task regress --dest .mltoolkit`
```

- [ ] **Step 3: Replace the equivalent step in cluster SKILL.md**

Open `mltoolkit-plugin/skills/cluster/SKILL.md`, find the corresponding step, replace with:

```markdown
3. **Stage the reference bundle** into `.mltoolkit/`:
   `python {PLUGIN_ROOT}/scripts/stage_session.py --task cluster --dest .mltoolkit`
```

- [ ] **Step 4: Replace the equivalent step in anomaly SKILL.md**

Open `mltoolkit-plugin/skills/anomaly/SKILL.md`, find the corresponding step, replace with:

```markdown
3. **Stage the reference bundle** into `.mltoolkit/`:
   `python {PLUGIN_ROOT}/scripts/stage_session.py --task anomaly --dest .mltoolkit`
```

- [ ] **Step 5: Commit**

```bash
cd mltoolkit-plugin
git add skills/classify/SKILL.md skills/regress/SKILL.md \
        skills/cluster/SKILL.md skills/anomaly/SKILL.md
git commit -m "docs(mltoolkit): skills use stage_session.py instead of raw copy (LEAD-001)"
```

---

## Task 4: Optuna tuner — real implementation

**Files:**
- Create: `mltoolkit-plugin/references/_shared/tuning_optuna.py`
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py:118-140` (the `tune_model` function + `main()` argparse)
- Modify: `mltoolkit-plugin/skills/regress/references/regress_reference.py` (the corresponding tune function + argparse)

- [ ] **Step 1: Write the Optuna tuner module**

Create `mltoolkit-plugin/references/_shared/tuning_optuna.py`:

```python
"""Optuna-backed hyperparameter search, API-compatible with tune_model callers.

Exposed callable:
    optuna_search(pipe, grid, X, y, scoring, cv, n_iter, random_state)
      -> (best_estimator, best_score, best_params)

`grid` uses the same shape as RandomizedSearchCV param_distributions
(a dict of `step__param` → list of candidate values). This keeps both
backends drop-in interchangeable at the call site.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def optuna_search(
    pipe,
    grid: dict[str, list[Any]],
    X,
    y,
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in grid.items()}
        est = clone(pipe).set_params(**params)
        scores = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

    best_params = study.best_params
    best = clone(pipe).set_params(**best_params)
    best.fit(X, y)
    return best, float(study.best_value), best_params
```

- [ ] **Step 2: Wire `--search-library` into classify_reference.py**

Open `mltoolkit-plugin/skills/classify/references/classify_reference.py`. In `main()` (around line 192) add the CLI flag right after the `--model` flag:

```python
    ap.add_argument("--search-library", choices=["sklearn", "optuna"], default="sklearn",
                    help="Tuning backend. 'optuna' requires the optuna package.")
    ap.add_argument("--n-iter", type=int, default=20,
                    help="Number of tuning iterations.")
```

Replace the body of `tune_model` (lines 118-140) with this version:

```python
def tune_model(model_id: str, X_train, y_train, out: Path,
               n_iter: int = 20, search_library: str = "sklearn"):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = preprocessing.build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])
    grid = {f"clf__{k}": v for k, v in entry["param_grid"].items()}
    if not grid:
        pipe.fit(X_train, y_train)
        return pipe, None

    if search_library == "optuna":
        from _shared import deps
        if not deps.has_optuna():
            print("WARNING: --search-library optuna requested but optuna is "
                  "not installed; falling back to RandomizedSearchCV.",
                  flush=True)
            search_library = "sklearn"

    if search_library == "optuna":
        from _shared.tuning_optuna import optuna_search
        best, best_score, best_params = optuna_search(
            pipe, grid, X_train, y_train,
            scoring="accuracy", cv=5, n_iter=n_iter, random_state=42,
        )
        with open(out / "results/best_params.json", "w") as f:
            json.dump({k: str(v) for k, v in best_params.items()}, f, indent=2)
        return best, best_score

    # Default: RandomizedSearchCV.
    import itertools
    max_combos = 1
    for v in grid.values():
        max_combos *= len(v)
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, max_combos),
        cv=5, scoring="accuracy", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    with open(out / "results/best_params.json", "w") as f:
        json.dump({k: str(v) for k, v in search.best_params_.items()}, f, indent=2)
    return search.best_estimator_, search.best_score_
```

**Note on the `from _shared` import:** this is the staged import path. For in-place test runs the existing `sys.path.insert(0, str(_PLUGIN_ROOT))` plus the existing `from references._shared import …` pattern still works because the top of the file keeps the in-place header (the rewriter only runs at staging time). The tune-time import uses bare `_shared` which Python resolves via the sibling directory `_HERE.parent` (also in `sys.path`). Both in-place and staged runs therefore succeed — in-place because `_HERE.parent` is `skills/classify/references/` which contains a symlink/shim? No — `_shared` lives at the plugin root, not at `_HERE.parent`. Fix: use a conditional:

Replace the two `from _shared import …` lines inside `tune_model` with:

```python
    try:
        from _shared import deps  # staged layout
        from _shared.tuning_optuna import optuna_search  # staged
    except ImportError:
        from references._shared import deps  # in-place layout
        from references._shared.tuning_optuna import optuna_search  # in-place
```

…and use those names inside the function. Adjust the code in the block above accordingly.

Finally, update the call inside `main()` that invokes `tune_model`: pass `n_iter=args.n_iter, search_library=args.search_library`.

- [ ] **Step 3: Mirror the wiring into regress_reference.py**

Open `mltoolkit-plugin/skills/regress/references/regress_reference.py`. Add the same two CLI flags in its `main()`. Refactor its `tune_model` (or equivalently named tune function — read the file to confirm) using the same pattern as Step 2, except the `scoring` argument is the regression primary metric (typically `"r2"` in this codebase — verify by reading the file). Reuse `optuna_search` from `_shared.tuning_optuna` (it is backend-agnostic because it delegates to `cross_val_score`).

- [ ] **Step 4: Write the optuna test**

Append to `mltoolkit-plugin/tests/test_stage_and_run.py`:

```python
import pytest

try:
    import optuna  # noqa: F401
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


@pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")
def test_optuna_tune_runs_end_to_end(classification_data, tmp_path):
    """Stage classify, run tune stage with --search-library optuna, verify best_params written."""
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    r = _run_stager("classify", dest)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out),
         "--stage", "all",
         "--search-library", "optuna",
         "--n-iter", "5"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert (out / "results/best_params.json").exists()


def test_optuna_fallback_warns_when_missing(classification_data, tmp_path, monkeypatch):
    """When optuna is requested but not installed, the script warns and falls back."""
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("classify", dest).check_returncode()
    # Force optuna "not installed" by prepending a stub that shadows it.
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "optuna.py").write_text("raise ImportError('forced for test')\n")
    env = {**__import__("os").environ, "PYTHONPATH": f"{stub_dir}{__import__('os').pathsep}{dest}"}
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out),
         "--stage", "tune",
         "--model", "lr",
         "--search-library", "optuna"],
        capture_output=True, text=True, env=env,
    )
    # Fallback should still succeed (returncode 0) and print the warning.
    assert r.returncode == 0, r.stderr
    assert "falling back" in (r.stdout + r.stderr).lower()
```

- [ ] **Step 5: Run the optuna tests**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py::test_optuna_tune_runs_end_to_end tests/test_stage_and_run.py::test_optuna_fallback_warns_when_missing -v`
Expected: if optuna is installed, both PASS. If not installed, first is SKIPPED and second passes with fallback working. Either outcome is green.

- [ ] **Step 6: Commit**

```bash
cd mltoolkit-plugin
git add references/_shared/tuning_optuna.py \
        skills/classify/references/classify_reference.py \
        skills/regress/references/regress_reference.py \
        tests/test_stage_and_run.py
git commit -m "feat(mltoolkit): --search-library {sklearn,optuna} with graceful fallback (LEAD-002)"
```

---

## Task 5: Truth-in-advertising — tune SKILL.md + README.md

**Files:**
- Modify: `mltoolkit-plugin/skills/tune/SKILL.md:24-26`
- Modify: `mltoolkit-plugin/README.md:21` (the optional-deps line for optuna)

- [ ] **Step 1: Rewrite tune SKILL.md**

Open `mltoolkit-plugin/skills/tune/SKILL.md` and replace the entire "Optuna support" section (lines 24-26) with:

```markdown
## Search backends

The tune stage accepts `--search-library {sklearn,optuna}`.

- `sklearn` (default) uses `RandomizedSearchCV` with `--n-iter` trials.
- `optuna` uses a TPE sampler (requires the `optuna` package). If requested but optuna is not installed, the script prints a warning and transparently falls back to `sklearn`.

Both backends write `results/best_params.json` with identical schema.
```

Also update the Workflow bullet at line 20 — change:

> `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID>`

into:

> `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID> [--search-library optuna] [--n-iter 50]`

- [ ] **Step 2: Fix the README optional-deps claim**

Open `mltoolkit-plugin/README.md` line 21. Replace:

```
- optuna — alternative hyperparameter search backend
```

with:

```
- optuna — alternative hyperparameter search backend (enable via `--search-library optuna` in the tune stage)
```

- [ ] **Step 3: Commit**

```bash
cd mltoolkit-plugin
git add skills/tune/SKILL.md README.md
git commit -m "docs(mltoolkit): document --search-library flag + optuna fallback (LEAD-002)"
```

---

## Task 6: End-to-end copy-and-run gate + CI wiring

**Files:**
- Modify: `mltoolkit-plugin/tests/test_stage_and_run.py` (add e2e rows for all four tasks)
- Modify: `mltoolkit-plugin/tests/test_references.sh` (add a fifth step invoking the new test)

- [ ] **Step 1: Add end-to-end rows for every task**

Append to `mltoolkit-plugin/tests/test_stage_and_run.py`:

```python
def test_staged_classify_runs_end_to_end(classification_data, tmp_path):
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("classify", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert (out / "results/leaderboard.csv").exists()
    assert (out / "model.joblib").exists()


def test_staged_regress_runs_end_to_end(regression_data, tmp_path):
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("regress", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", regression_data["path"],
         "--target", regression_data["target"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert (out / "results/leaderboard.csv").exists()


def test_staged_cluster_runs_end_to_end(cluster_data, tmp_path):
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("cluster", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", cluster_data["path"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"


def test_staged_anomaly_runs_end_to_end(cluster_data, tmp_path):
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("anomaly", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", cluster_data["path"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
```

Note: `cluster_data` and the anomaly test share the same fixture because anomaly is unsupervised on a numeric matrix. If a reviewer prefers a dedicated `anomaly_data` fixture, add one in `conftest.py`; for this gate the shared fixture is sufficient.

- [ ] **Step 2: Run the end-to-end suite**

Run: `cd mltoolkit-plugin && python -m pytest tests/test_stage_and_run.py -v`
Expected: all tests PASS (optuna one SKIPPED if not installed; fallback one PASSES either way).

- [ ] **Step 3: Wire into the repo smoke script**

Open `mltoolkit-plugin/tests/test_references.sh`. After the existing `[4/4] Cluster + anomaly + package…` block, change the label to `[4/5]` and append:

```bash
echo
echo "[5/5] Stage-and-run (end-to-end copy path)..."
python -m pytest tests/test_stage_and_run.py -q
```

Also change the `[1/4]`, `[2/4]`, `[3/4]`, `[4/4]` labels to `[1/5]`, `[2/5]`, `[3/5]`, `[4/5]` for accuracy.

- [ ] **Step 4: Run the full smoke suite**

Run: `cd mltoolkit-plugin && bash tests/test_references.sh`
Expected: all five steps pass. Final line: `=== All smoke tests passed ===`.

- [ ] **Step 5: Commit**

```bash
cd mltoolkit-plugin
git add tests/test_stage_and_run.py tests/test_references.sh
git commit -m "test(mltoolkit): end-to-end stage-and-run coverage for all references (LEAD-023)"
```

---

## Self-review notes

- **Spec coverage:** LEAD-001 ← Tasks 1+2+3. LEAD-002 ← Tasks 4+5. LEAD-023 ← Task 6. All three P0 findings have landing tasks.
- **Placeholder scan:** every step contains literal code or literal commands. The one hedge — "verify by reading the file" in Task 4 Step 3 — is intentional because the regress reference's tune function name is not yet read; an engineer executing this plan must open the file and confirm. That is not a placeholder, it is a concrete instruction.
- **Type consistency:** `optuna_search` signature is the same everywhere it is called; `tune_model` now takes `search_library` and `n_iter`; the stager's `stage(task, dest)` matches the CLI args.
- **What this plan does NOT touch:** no new `_shared/` primitives for fairness/calibration/etc. (Plan 2), no classify wiring (Plan 3), no zoo expansion (Plan 4), no reproducibility/paper scaffolding (Plan 5). Plan 1 is the unblock only.

## Out of scope (deferred to later plans)

- LEAD-003..008 (fairness, calibration, CIs, splits, ethics prompts, target-encoder guard) — Plans 2 + 3
- LEAD-009..017, LEAD-024..032 (interpretability, imputation, resampling, plots, zoo, subgroup, threshold) — Plans 3 + 4
- LEAD-018..022, LEAD-028, LEAD-033..041 (reproducibility, paper reporting, model card, tracking) — Plan 5
