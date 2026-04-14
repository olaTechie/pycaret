# Plan 5 — Reproducibility + paper-ready Tier B/C

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.

**Goal:** Complete the PyCaret-to-paper journey. Emit `run_manifest.json` at every stage, add `--finalize`, pin requirements, add experiment tracking, generate methods.md, model_card.md, and a STARD/CONSORT-AI checklist as Tier B/C artifacts. Wire `summary_report.html` as the default evaluate capstone. Make Tier-C pluck the chosen estimator into `src/train.py` so inference parity is code-enforced.

**Architecture:** New `_shared/` primitives + wiring into classify/regress references + Tier B/C transform extensions. Package transforms (`tier_b_transform.py`, `tier_c_transform.py`) gain an "artifacts" harvest that carries paper-reporting files from `.mltoolkit/results/` into the deliverable.

**Findings covered:** LEAD-018, -019, -022, -028, -033, -034, -038, -040, -041.

---

## Task 1: `_shared/run_manifest.py` + wire into all tasks

**Create:** `references/_shared/run_manifest.py`, `tests/test_shared_run_manifest.py`.

```python
# run_manifest.py
"""Reproducibility manifest — versions, seed, split, CV, hyperparams."""
from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _pkg(name: str):
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def build_manifest(*, stage: str, args_dict: dict, extra: dict | None = None) -> dict:
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            p: _pkg(p)
            for p in ("scikit-learn", "pandas", "numpy", "scipy", "matplotlib",
                      "seaborn", "joblib", "xgboost", "lightgbm", "catboost",
                      "imbalanced-learn", "shap", "optuna", "mlflow", "category-encoders")
            if _pkg(p) is not None
        },
        "args": {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                 for k, v in args_dict.items()},
        "extra": extra or {},
    }


def write_manifest(out: Path, manifest: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "run_manifest.json"
    # Append-mode: keep a list of stage manifests if file exists.
    existing = []
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(manifest)
    p.write_text(json.dumps(existing, indent=2))
    return p
```

Tests: `build_manifest` has required fields, `write_manifest` appends across calls.

**Wire into classify/regress:** at the end of `main()`, call:
```python
from references._shared.run_manifest import build_manifest, write_manifest
mods_write = write_manifest(out / "results", build_manifest(
    stage=args.stage, args_dict=vars(args),
    extra={"best_model_id": best_id}))
```

Same pattern for cluster/anomaly references (skip the `best_id`).

Commit: `feat(mltoolkit): run_manifest.json at every stage (LEAD-018, -034)`.

---

## Task 2: `--finalize` stage (LEAD-019)

In classify + regress `main()`, after evaluate, add:

```python
if args.finalize and best_model is not None:
    final = best_model  # already the tuned/calibrated model
    final.fit(X, y)  # fit on all data (X_train ∪ X_test)
    joblib.dump(final, out / "model_final.joblib")
    (out / "results/finalize_note.json").write_text(json.dumps({
        "note": "Fit on full dataset (train+holdout). Do NOT re-evaluate on holdout.",
        "n_rows": len(X), "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
```

Add `ap.add_argument("--finalize", action="store_true")`. Test that `--finalize` produces `model_final.joblib`.

Commit: `feat(mltoolkit): --finalize stage refits on full dataset (LEAD-019)`.

---

## Task 3: `_shared/methods_md.py` — TRIPOD+AI methods scaffold

New module generating a methods-section markdown from a run_manifest + leaderboard + best_params. Structure:
1. Data source (from datasheet.md if present)
2. Sample size & EPV (from epv_audit.json)
3. Preprocessing (imputation, encoders, resampling)
4. Split strategy (group/time/stratified)
5. Model search space (zoo members)
6. Tuning procedure (search_library + n_iter + scoring)
7. Performance metrics (leaderboard.csv + holdout_metrics_ci.json)
8. Fairness analysis (subgroup_metrics.csv if present)
9. Calibration (calibration.json if present)
10. Software versions (run_manifest packages)

Test: given a dict of inputs, the output markdown contains headings 1-10.

Tier B and C package transforms call this with the harvested artifacts to emit `reports/methods.md`.

Commit: `feat(mltoolkit): methods.md scaffold with TRIPOD+AI alignment (LEAD-028)`.

---

## Task 4: `_shared/model_card.py` + Tier-B/C integration (LEAD-040)

New module generating `reports/model_card.md` with sections: Intended use, Out-of-scope use, Training data, Evaluation data, Performance, Fairness, Ethical considerations, Caveats, Contact. Populated from datasheet.md + run_manifest + fairness_disparities.json + calibration.json.

Commit: `feat(mltoolkit): model_card.md generator (LEAD-040)`.

---

## Task 5: `_shared/checklists.py` — STARD + CONSORT-AI + TRIPOD+AI (LEAD-041)

New module emitting compliance checklists as markdown. Each checklist is a table with columns: item, description, reported? (with a `?` default — user fills in), location (file ref from run).

Tier B/C package transforms emit:
- `reports/tripod_ai_checklist.md` always
- `reports/stard_checklist.md` if `--diagnostic` passed at package time
- `reports/consort_ai_checklist.md` if `--interventional` passed

Commit: `feat(mltoolkit): TRIPOD+AI / STARD / CONSORT-AI checklists (LEAD-041)`.

---

## Task 6: `--track {none,mlflow}` (LEAD-033)

In classify + regress: add `--track` flag. When `mlflow` and the package is importable, wrap the whole `main()` in:

```python
if args.track == "mlflow":
    try:
        import mlflow
        mlflow.set_experiment("mltoolkit")
        mlflow.autolog()  # captures sklearn fits automatically
    except ImportError:
        print("WARNING: --track mlflow requested but mlflow is not installed.",
              flush=True)
```

Commit: `feat(mltoolkit): --track mlflow autolog (LEAD-033)`.

---

## Task 7: importlib.metadata-based requirements pinning (LEAD-034)

In `skills/package/references/tier_b_transform.py` and `tier_c_transform.py`, after detecting imports, resolve each to a pinned version via `importlib.metadata.version(pkg)` and write a `requirements.txt` with exact pins. Test: call transform on a fixture session.py, grep `requirements.txt` for `scikit-learn==X.Y.Z`.

Commit: `feat(mltoolkit): package transforms pin requirements.txt via importlib.metadata (LEAD-034)`.

---

## Task 8: summary_report.html (LEAD-038)

Wire `reporting.summary_report(title, tables, figures, output)` into classify/regress `evaluate()` as the last step. Collect key tables (leaderboard, holdout metrics, calibration) + figure paths and produce `results/summary_report.html`.

Commit: `feat(mltoolkit): summary_report.html capstone at end of evaluate (LEAD-038)`.

---

## Task 9: Tier-C extracts chosen estimator into src/train.py (LEAD-022)

In `skills/package/references/tier_c_transform.py`, parse the staged `session.py`'s `best_model_id` (from `best_params.json` or a marker line), then emit:
- `src/train.py` — contains a `train(data_path, target_col) -> model` function that hard-codes the chosen model class + tuned hyperparams (from `best_params.json`).
- `src/inference.py` — `predict(model, df) -> y`.

Test: run Tier-C on a staged session, assert `src/train.py` imports the right estimator class and calls `.fit(...)`.

Commit: `feat(mltoolkit): Tier-C emits src/train.py + src/inference.py (LEAD-022)`.

---

## Task 10: End-to-end paper-bundle test

Extend `tests/test_stage_and_run.py` with `test_tier_b_paper_bundle` that:
1. Stages classify
2. Runs `--stage all --calibrate sigmoid --bootstrap 100 --group-col group --decision-curve --finalize --ensemble voting`
3. Invokes Tier-B transform
4. Asserts the tier_b deliverable has: `reports/methods.md`, `reports/model_card.md`, `reports/tripod_ai_checklist.md`, `requirements.txt` (pinned), `run_manifest.json`, `model_final.joblib`, `summary_report.html`.

Commit: `test(mltoolkit): end-to-end Tier-B paper-bundle assertion`.

---

## Self-review

- LEAD-018 → T1; -019 → T2; -022 → T9; -028 → T3; -033 → T6; -034 → T7; -038 → T8; -040 → T4; -041 → T5.
- All 41 original findings now land in plans 1-5.
