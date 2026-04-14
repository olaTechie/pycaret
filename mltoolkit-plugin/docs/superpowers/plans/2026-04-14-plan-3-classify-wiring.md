# Plan 3 — Classify skill wires up the backbone

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `classify_reference.py` from a generic ML demo into a paper-grade clinical-prediction toolkit by composing the Plan 2 primitives. Delivers group/time splits, sensitive-attribute handling, calibration, bootstrap CIs, imputation routing, resampling, threshold optimization, SHAP, subgroup + fairness metrics, decision curves, and ethics-gated setup.

**Architecture:** Additive — the existing stages (eda/compare/tune/evaluate/all) keep their names and defaults stay backward-compatible (no new flag changes existing behavior). New CLI flags gate new behavior. The setup skill gains an ethics prompt that produces `.mltoolkit/datasheet.md` — consumed by classify at run time for the sensitive-attribute list.

**Tech Stack:** sklearn (`LearningCurveDisplay`, `CalibratedClassifierCV`, `PartialDependenceDisplay`, impute family), imblearn (optional, for SMOTE/ADASYN), shap (optional), plus every `_shared/` module from Plan 2.

**Findings covered:** LEAD-003, -004, -005, -006, -007, -008, -010, -011, -012, -017, -024, -025, -026, -027, -029, -031.

---

## File Structure

**Modified:**

| File | Change |
|---|---|
| `mltoolkit-plugin/skills/setup/references/setup_reference.py` | Emit `.mltoolkit/datasheet.md` with ethics / consent / provenance prompts. |
| `mltoolkit-plugin/skills/setup/SKILL.md` | Add the ethics prompts to the workflow + require datasheet.md. |
| `mltoolkit-plugin/skills/classify/references/classify_reference.py` | Add CLI flags; wire every `_shared/*` module into the right stage. Big file — changes are surgical per-task below. |
| `mltoolkit-plugin/skills/classify/references/preprocessing.py` | Route high-cardinality categoricals through `safe_high_cardinality_encoder`. New `--imputation` routing. |
| `mltoolkit-plugin/skills/classify/SKILL.md` | Document every new CLI flag. |
| `mltoolkit-plugin/tests/test_classify.py` | Add flag-behavior tests (group-col, imputation, calibration output). |
| `mltoolkit-plugin/tests/test_stage_and_run.py` | Add an e2e "paper-mode" run exercising calibrate + group-col + decision-curve together. |

**Not touched:** regress/cluster/anomaly reference scripts (Plan 4). Tier-C packaging (Plan 5).

---

## CLI surface (final shape after Plan 3)

```
python session.py --data ... --target ... --output-dir ... --stage <stage>
    # existing (no behavior change when absent)
    --cv 5 --model <id> --search-library {sklearn,optuna} --n-iter 20
    # new this plan (all default to "off" / "none" → current behavior)
    --group-col <colname>
    --time-col <colname>
    --sensitive-features <col1,col2,...>
    --allow-target-encode-on-sensitive
    --imputation {simple,iterative,knn,drop}
    --resample {none,smote,adasyn}
    --calibrate {none,sigmoid,isotonic}
    --optimize-threshold {none,youden,f1,mcc,fixed-recall}
    --fixed-recall 0.80
    --decision-curve
    --bootstrap 1000
```

---

## Task 1: setup_reference.py emits datasheet.md

**Files:**
- Modify: `mltoolkit-plugin/skills/setup/references/setup_reference.py`
- Modify: `mltoolkit-plugin/skills/setup/SKILL.md`
- Test: `mltoolkit-plugin/tests/test_stage_and_run.py` (add a row that stages setup and verifies datasheet.md)

- [ ] **Step 1: Read the current setup_reference.py to find the natural insertion point**

`Read` the file. Locate `main()`. After any existing run_eda / schema-writing call, insert a `write_datasheet(out)` call.

- [ ] **Step 2: Implement `write_datasheet(out: Path)`**

Add this function near the top of `setup_reference.py` (after imports, before `main`):

```python
def write_datasheet(out: Path) -> Path:
    """Emit a datasheet.md scaffold. Claude (the calling agent) fills it in
    by asking the user the prompts inline; this file is the prompt template."""
    p = out / "datasheet.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "# Datasheet — mltoolkit session\n\n"
        "Fill this file in before the classify/regress skill trains any model.\n\n"
        "## Provenance\n"
        "- **Source**: <where did this data come from?>\n"
        "- **Collection date(s)**: <range>\n"
        "- **Sampling**: <convenience / random / census / other>\n\n"
        "## Consent & ethics\n"
        "- **IRB / ethics approval**: <IRB number or 'exempt' or 'not applicable'>\n"
        "- **Consent basis**: <broad / specific / waiver / public>\n"
        "- **PHI / PII**: <present / none / de-identified>\n\n"
        "## Protected attributes\n"
        "List every column that encodes or proxies for a protected attribute "
        "(race, ethnicity, sex, gender, age band, zip code, religion, "
        "disability, national origin, pregnancy, sexual orientation):\n\n"
        "- `<col_name>` — <reason protected>\n\n"
        "Pass these as `--sensitive-features col1,col2,...` to classify / regress\n"
        "so the plugin refuses to target-encode them and emits per-group metrics.\n\n"
        "## Known limitations\n"
        "- <dataset bias, coverage gaps, measurement issues>\n"
    )
    return p
```

Also call it in `main()`:

```python
write_datasheet(out)
print(f"Datasheet scaffold written to {out / 'datasheet.md'} — fill it in before training.")
```

- [ ] **Step 3: Update setup SKILL.md**

Open `mltoolkit-plugin/skills/setup/SKILL.md`. Between step 5 and step 6, insert:

```markdown
6. **Fill out `.mltoolkit/datasheet.md`** with the user. Ask them in order:
   a. **Data provenance** (source, collection dates, sampling)
   b. **Consent & ethics** (IRB status, consent basis, PHI/PII presence)
   c. **Protected attributes** (race, ethnicity, sex, gender, age, zip, religion, disability, national origin, pregnancy, sexual orientation). Record each column name the user identifies as sensitive.
   d. **Known limitations** (bias, coverage gaps, measurement issues)
   Save the filled datasheet. The list of protected columns is passed to classify/regress as `--sensitive-features`.
```

Renumber subsequent steps.

- [ ] **Step 4: Test**

Append to `tests/test_stage_and_run.py`:

```python
def test_setup_emits_datasheet(classification_data, tmp_path):
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("setup", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    ds = out / "datasheet.md"
    assert ds.exists()
    text = ds.read_text()
    for heading in ("Provenance", "Consent", "Protected attributes"):
        assert heading in text
```

Run: `python -m pytest tests/test_stage_and_run.py::test_setup_emits_datasheet -v` — expect PASS.

- [ ] **Step 5: Commit**

```bash
cd mltoolkit-plugin
git add skills/setup/references/setup_reference.py skills/setup/SKILL.md tests/test_stage_and_run.py
git commit -m "feat(mltoolkit): setup emits datasheet.md ethics scaffold (LEAD-007)"
```

---

## Task 2: Classify CLI flags + argparse wiring

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py:208-220` (the `main()` argparse block)

This task only adds flags and parses them — behavior wiring is in later tasks. Default values preserve current behavior.

- [ ] **Step 1: Extend the argparse block**

In `main()`, after the existing `--n-iter` argument, add:

```python
    ap.add_argument("--group-col", default=None,
                    help="Column to use for Group/StratifiedGroup K-fold. Routed through _shared.splits.make_splitter.")
    ap.add_argument("--time-col", default=None,
                    help="Column to use for TimeSeriesSplit. Must be parseable by pd.to_datetime.")
    ap.add_argument("--sensitive-features", default="",
                    help="Comma-separated column names treated as protected attributes.")
    ap.add_argument("--allow-target-encode-on-sensitive", action="store_true",
                    help="Override the refusal to target-encode sensitive columns.")
    ap.add_argument("--imputation", choices=["simple", "iterative", "knn", "drop"],
                    default="simple")
    ap.add_argument("--resample", choices=["none", "smote", "adasyn"], default="none",
                    help="Imbalanced-class resampling (requires imblearn).")
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none",
                    help="Wrap the final model in CalibratedClassifierCV.")
    ap.add_argument("--optimize-threshold",
                    choices=["none", "youden", "f1", "mcc", "fixed-recall"], default="none")
    ap.add_argument("--fixed-recall", type=float, default=0.80,
                    help="Target recall for --optimize-threshold=fixed-recall.")
    ap.add_argument("--decision-curve", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap iterations for holdout CIs. 0 disables.")
```

Also, right after `args = ap.parse_args()`, normalize `sensitive`:

```python
    sensitive = [c.strip() for c in args.sensitive_features.split(",") if c.strip()]
```

- [ ] **Step 2: Verify in-place smoke still passes with default flags**

Run: `python -m pytest tests/test_classify.py -v` — expect existing 4 tests PASS unchanged.

- [ ] **Step 3: Commit**

```bash
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py
git commit -m "feat(mltoolkit): classify CLI — add group/time/sensitive/imputation/resample/calibrate/threshold flags"
```

---

## Task 3: EDA stage — Table 1 + EPV + missingness

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py` (`run_eda` function around line 56)

- [ ] **Step 1: Extend `run_eda`**

At the end of `run_eda(df, target, out)` (right after the correlation heatmap save), add:

```python
    # --- Sample-size audit -------------------------------------------------
    try:
        from _shared.epv import audit_epv
    except ImportError:
        from references._shared.epv import audit_epv
    epv = audit_epv(df.drop(columns=[target]), df[target])
    pd.Series(epv).to_json(out / "results/epv_audit.json", indent=2)
    if epv["low_epv_warning"]:
        print(f"WARNING: low events-per-variable (EPV={epv['epv']:.2f}). "
              "Consider reducing features or regularization.", flush=True)
    if epv["rare_outcome_warning"]:
        print(f"WARNING: rare outcome (prevalence={epv['minority_prevalence']:.3f}).",
              flush=True)

    # --- Table 1 -----------------------------------------------------------
    try:
        from _shared.table1 import table1
    except ImportError:
        from references._shared.table1 import table1
    t1 = table1(df.drop(columns=[target]), group=df[target] if df[target].nunique() <= 10 else None)
    t1.to_csv(out / "results/table1.csv", index=False)
    (out / "results/table1.md").write_text(reporting.df_to_markdown(t1))

    # --- Missingness figure -----------------------------------------------
    miss = df.isna().mean().sort_values(ascending=False)
    if (miss > 0).any():
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * (miss > 0).sum())))
        miss[miss > 0].plot(kind="barh", ax=ax)
        ax.set_xlabel("Missing fraction")
        ax.set_title("Missingness by column")
        plotting.save_fig(fig, out / "artifacts/missingness"); plt.close(fig)
```

- [ ] **Step 2: Test**

Extend `tests/test_classify.py` with:

```python
def test_eda_emits_epv_and_table1(classification_data, tmp_path):
    import subprocess, sys
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out),
         "--stage", "eda"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/epv_audit.json").exists()
    assert (out / "results/table1.csv").exists()
```

Run: `python -m pytest tests/test_classify.py::test_eda_emits_epv_and_table1 -v` — expect PASS.

- [ ] **Step 3: Commit**

```bash
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): classify EDA — EPV audit + Table 1 + missingness figure (LEAD-012, -025, -027)"
```

---

## Task 4: Preprocessing — imputation routing + sensitive-attr guard

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/preprocessing.py`

- [ ] **Step 1: Extend `build_preprocessor` signature**

Change the signature to accept `imputation: str = "simple"` and `sensitive: Sequence[str] = ()` and `allow_te_on_sensitive: bool = False`. Route imputation choices to sklearn imputers:

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def _make_imputer(kind: str, *, strategy_num="median", strategy_cat="most_frequent"):
    if kind == "simple":
        return SimpleImputer(strategy=strategy_num), SimpleImputer(strategy=strategy_cat)
    if kind == "iterative":
        return IterativeImputer(random_state=42), SimpleImputer(strategy=strategy_cat)
    if kind == "knn":
        return KNNImputer(), SimpleImputer(strategy=strategy_cat)
    if kind == "drop":
        # 'drop' is a caller-level contract (callers drop rows before build_preprocessor),
        # but to keep preprocessing a single call-path we just use Simple here.
        return SimpleImputer(strategy=strategy_num), SimpleImputer(strategy=strategy_cat)
    raise ValueError(f"unknown imputation kind: {kind}")
```

In `build_preprocessor`, replace the two `SimpleImputer(...)` calls with `imp_num, imp_cat = _make_imputer(imputation)` and use them. Replace the high-cardinality TargetEncoder block with:

```python
if high_card:
    try:
        from _shared.encoders_safe import is_sensitive_column, safe_high_cardinality_encoder
    except ImportError:
        from references._shared.encoders_safe import is_sensitive_column, safe_high_cardinality_encoder

    # Per-column: if sensitive and not overridden, refuse.
    for col in high_card:
        if is_sensitive_column(col, sensitive) and not allow_te_on_sensitive:
            # Caller must drop or explicitly opt-in; refusal surface is loud.
            safe_high_cardinality_encoder(col, sensitive,
                                          allow_target_encode_on_sensitive=False)
    # Use TE for non-sensitive high-card if category_encoders present; else ordinal.
    if deps.has_category_encoders():
        import category_encoders as ce
        enc = ce.TargetEncoder()
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    transformers.append((
        "cat_high",
        Pipeline([("imp", imp_cat), ("enc", enc)]),
        high_card,
    ))
```

- [ ] **Step 2: Update callers**

In `classify_reference.py`, every call to `preprocessing.build_preprocessor(X)` becomes:

```python
preprocessing.build_preprocessor(
    X, imputation=args.imputation,
    sensitive=sensitive,
    allow_te_on_sensitive=args.allow_target_encode_on_sensitive,
)
```

(The `sensitive` and `args` variables are captured from `main`'s scope — refactor `compare_models`, `tune_model`, and `evaluate` to accept these as parameters, or move them into a closure.)

Simpler refactor: keep functions signature-compatible by wrapping. Add a module-level helper:

```python
def _preprocessor(X, args, sensitive):
    return preprocessing.build_preprocessor(
        X, imputation=args.imputation,
        sensitive=sensitive,
        allow_te_on_sensitive=args.allow_target_encode_on_sensitive,
    )
```

Then pass `args` and `sensitive` into each stage function.

- [ ] **Step 3: Tests**

Append to `tests/test_classify.py`:

```python
def test_preprocessor_refuses_sensitive_target_encode():
    import pandas as pd, pytest
    df = pd.DataFrame({
        "num_a": range(100),
        "race": [f"group_{i%15}" for i in range(100)],  # high-cardinality + sensitive
        "target": [0, 1] * 50,
    })
    with pytest.raises(ValueError, match="sensitive"):
        preprocessing.build_preprocessor(
            df.drop(columns=["target"]),
            sensitive=["race"],
            allow_te_on_sensitive=False,
        )


def test_preprocessor_allows_sensitive_when_overridden():
    import pandas as pd
    df = pd.DataFrame({
        "num_a": range(100),
        "race": [f"g_{i%15}" for i in range(100)],
        "target": [0, 1] * 50,
    })
    pre = preprocessing.build_preprocessor(
        df.drop(columns=["target"]),
        sensitive=["race"],
        allow_te_on_sensitive=True,
    )
    assert pre is not None


def test_imputation_iterative_builds():
    import pandas as pd
    df = pd.DataFrame({"a": [1.0, None, 3.0] * 20, "b": range(60)})
    pre = preprocessing.build_preprocessor(df, imputation="iterative")
    assert pre is not None
```

- [ ] **Step 4: Run and commit**

```bash
python -m pytest tests/test_classify.py -v
# expect all PASS
cd mltoolkit-plugin
git add skills/classify/references/preprocessing.py skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): preprocessor routes imputation + refuses TE on sensitive cols (LEAD-008, -025)"
```

---

## Task 5: Compare + Tune stages — split routing + clinical scorers

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py` (`compare_models`, `tune_model`)

- [ ] **Step 1: Route splitter via `make_splitter`**

Replace the `cv=args.cv` integer in both `cross_val_score` (compare) and `RandomizedSearchCV` (tune) with a splitter instance. Inside each stage function, build:

```python
try:
    from _shared.splits import make_splitter
except ImportError:
    from references._shared.splits import make_splitter

cv_splitter = make_splitter(
    y_train,
    n_splits=cv,
    groups=(df[args.group_col].loc[X_train.index] if args.group_col else None),
    time_order=(pd.to_datetime(df[args.time_col]).loc[X_train.index] if args.time_col else None),
)
```

(`df` must be accessible inside stage functions — pass it in, or compute the groups/times in `main()` and pass them as function args. Recommended: pre-compute `groups` and `time_order` once in `main()`.)

Use `cv=cv_splitter` everywhere previously `cv=5` was passed. For `cross_val_score` with `GroupKFold`, also pass `groups=groups`.

- [ ] **Step 2: Clinical-utility scorers**

In `compare_models`, extend the scorers dict for binary tasks:

```python
from sklearn.metrics import make_scorer

def _ppv_at_recall(y_true, y_proba, target_recall=0.80):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ok = rec >= target_recall
    return float(prec[ok].max() if ok.any() else 0.0)

def _spec_at_recall(y_true, y_proba, target_recall=0.80):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ok = tpr >= target_recall
    return float((1 - fpr[ok]).max() if ok.any() else 0.0)

if n_classes == 2:
    scorers["ppv_at_recall_80"] = make_scorer(_ppv_at_recall, needs_proba=True)
    scorers["spec_at_recall_80"] = make_scorer(_spec_at_recall, needs_proba=True)
```

- [ ] **Step 3: Test**

Append to `tests/test_classify.py`:

```python
def test_compare_stage_uses_group_col(classification_data, tmp_path):
    """When --group-col is passed, compare runs without crashing and leaderboard emits."""
    import subprocess, sys, pandas as pd
    df = pd.read_csv(classification_data["path"])
    df["group"] = [i % 10 for i in range(len(df))]
    data_path = tmp_path / "with_groups.csv"
    df.to_csv(data_path, index=False)
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", str(data_path), "--target", "target",
         "--output-dir", str(out), "--stage", "compare",
         "--group-col", "group", "--cv", "3"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/leaderboard.csv").exists()
```

Run + commit:

```bash
python -m pytest tests/test_classify.py::test_compare_stage_uses_group_col -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): compare/tune route through make_splitter + add clinical scorers (LEAD-006, -031)"
```

---

## Task 6: Resampling — `--resample` via imblearn.pipeline.Pipeline

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py`

- [ ] **Step 1: Build a resampling pipeline helper**

Insert near the top of the file:

```python
def _pipeline(pre, estimator, *, resample: str, sensitive):
    if resample == "none":
        return Pipeline([("pre", pre), ("clf", estimator)])
    try:
        from _shared import deps
    except ImportError:
        from references._shared import deps
    if not deps.has_imblearn():
        print(f"WARNING: --resample {resample} requested but imblearn is not installed; "
              "running without resampling.", flush=True)
        return Pipeline([("pre", pre), ("clf", estimator)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    if resample == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
    else:  # adasyn
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN(random_state=42)
    return ImbPipeline([("pre", pre), ("resample", sampler), ("clf", estimator)])
```

- [ ] **Step 2: Wire into compare + tune**

In `compare_models` and `tune_model`, replace `pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])` with `pipe = _pipeline(pre, entry["estimator"], resample=args.resample, sensitive=sensitive)`.

- [ ] **Step 3: Test**

Append to `tests/test_classify.py`:

```python
def test_resample_smote_runs_when_imblearn_present(classification_data, tmp_path):
    from references._shared import deps
    if not deps.has_imblearn():
        import pytest
        pytest.skip("imblearn not installed")
    import subprocess, sys
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"], "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "compare", "--cv", "3",
         "--resample", "smote"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
```

Run + commit:

```bash
python -m pytest tests/test_classify.py::test_resample_smote_runs_when_imblearn_present -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): --resample {smote,adasyn} via imblearn.pipeline (LEAD-011)"
```

---

## Task 7: Calibrate stage

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/references/classify_reference.py`

- [ ] **Step 1: Wrap final model when `--calibrate != none`**

In `main()`, after the tune stage sets `best_model`, add:

```python
    if args.calibrate != "none" and best_model is not None:
        from sklearn.calibration import CalibratedClassifierCV
        best_model = CalibratedClassifierCV(best_model, method=args.calibrate, cv="prefit")
        best_model.fit(X_train, y_train)  # cv="prefit" needs a held-out fit — simple path
```

Note: `cv="prefit"` requires the inner estimator to already be fitted on a disjoint set. For paper-grade calibration we should refit on a partition. Simple first cut: use `CalibratedClassifierCV(base, method=..., cv=5)` which does stratified CV internally:

```python
    if args.calibrate != "none" and best_model is not None:
        from sklearn.calibration import CalibratedClassifierCV
        # base = best_model's *unfitted* skeleton. Rebuild via zoo to avoid leak.
        zoo_entry = model_zoo.get_zoo()[best_model_id]
        pre = _preprocessor(X_train, args, sensitive)
        base = _pipeline(pre, zoo_entry["estimator"], resample=args.resample, sensitive=sensitive)
        best_model = CalibratedClassifierCV(base, method=args.calibrate, cv=5)
        best_model.fit(X_train, y_train)
```

- [ ] **Step 2: Evaluate — emit calibration summary + reliability diagram**

Inside `evaluate(...)`, right after computing `y_proba = model.predict_proba(X_test)[:, 1]` (or its binary equivalent), add:

```python
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            from _shared.calibration import calibration_summary, reliability_diagram
        except ImportError:
            from references._shared.calibration import calibration_summary, reliability_diagram
        cal = calibration_summary(y_test, y_proba, n_bins=10)
        (out / "results/calibration.json").write_text(json.dumps(cal, indent=2))
        fig = reliability_diagram(y_test, y_proba, n_bins=10)
        plotting.save_fig(fig, out / "artifacts/reliability"); plt.close(fig)
```

- [ ] **Step 3: Test**

```python
def test_evaluate_emits_calibration_for_binary(classification_data, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"], "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/calibration.json").exists()
    assert list((out / "artifacts").glob("reliability*"))
```

Run + commit:

```bash
python -m pytest tests/test_classify.py::test_evaluate_emits_calibration_for_binary -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): --calibrate + reliability diagram + calibration.json (LEAD-004)"
```

---

## Task 8: Bootstrap CIs + per-fold scores on holdout

**Files:**
- Modify: `classify_reference.py` `evaluate(...)`

- [ ] **Step 1: Add bootstrap CI emission**

Inside `evaluate`, after computing `y_pred` and `y_proba`, add:

```python
    if args.bootstrap and len(np.unique(y_test)) == 2:
        try:
            from _shared.bootstrap import bootstrap_ci
        except ImportError:
            from references._shared.bootstrap import bootstrap_ci
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        ci = {
            "accuracy": bootstrap_ci(accuracy_score, y_test, y_pred,
                                     n_boot=args.bootstrap, random_state=42),
            "f1":       bootstrap_ci(f1_score, y_test, y_pred,
                                     n_boot=args.bootstrap, random_state=42),
            "roc_auc":  bootstrap_ci(roc_auc_score, y_test, y_proba,
                                     n_boot=args.bootstrap, random_state=42),
        }
        (out / "results/holdout_metrics_ci.json").write_text(json.dumps(ci, indent=2))
```

- [ ] **Step 2: Per-fold scores in compare**

In `compare_models`, after `cross_val_score`, also write the fold scores. Switch from `cross_val_score` to `cross_validate` so we get per-fold arrays:

```python
from sklearn.model_selection import cross_validate

res = cross_validate(pipe, X_train, y_train,
                     cv=cv_splitter, scoring=scorers,
                     return_train_score=False, n_jobs=-1)
for name in scorers:
    row[name] = float(res[f"test_{name}"].mean())
    row[f"{name}_std"] = float(res[f"test_{name}"].std())
```

Also emit `results/leaderboard_folds.csv` with a row per (model, fold) pair using `res`.

- [ ] **Step 3: Test + commit**

```python
def test_bootstrap_cis_emitted(classification_data, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"], "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5", "--bootstrap", "200"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/holdout_metrics_ci.json").exists()
```

```bash
python -m pytest tests/test_classify.py::test_bootstrap_cis_emitted -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): bootstrap CIs on holdout + per-fold scores (LEAD-005)"
```

---

## Task 9: Threshold optimization

**Files:**
- Modify: `classify_reference.py` `evaluate(...)`

- [ ] **Step 1: Implement threshold picker**

Add module-level function:

```python
def _pick_threshold(y_true, y_proba, method: str, *, fixed_recall=0.80):
    import numpy as np
    from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve
    thresholds = np.linspace(0.01, 0.99, 99)
    if method == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return {"threshold": float(thr[idx]), "criterion": "youden",
                "score": float(j[idx])}
    if method == "f1":
        scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(scores))
        return {"threshold": float(thresholds[idx]), "criterion": "f1",
                "score": float(scores[idx])}
    if method == "mcc":
        scores = [matthews_corrcoef(y_true, (y_proba >= t).astype(int)) for t in thresholds]
        idx = int(np.argmax(scores))
        return {"threshold": float(thresholds[idx]), "criterion": "mcc",
                "score": float(scores[idx])}
    if method == "fixed-recall":
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        ok = rec >= fixed_recall
        if not ok.any():
            return {"threshold": 0.5, "criterion": "fixed-recall", "score": float("nan")}
        idx = int(np.argmax(prec * ok))
        t = thr[idx] if idx < len(thr) else 0.5
        return {"threshold": float(t), "criterion": "fixed-recall",
                "score": float(prec[idx])}
    raise ValueError(f"unknown threshold method {method}")
```

- [ ] **Step 2: Call from evaluate**

Inside `evaluate`, if `args.optimize_threshold != "none"` and binary:

```python
    if args.optimize_threshold != "none" and len(np.unique(y_test)) == 2:
        picked = _pick_threshold(y_test, y_proba, args.optimize_threshold,
                                 fixed_recall=args.fixed_recall)
        (out / "results/threshold.json").write_text(json.dumps(picked, indent=2))
```

- [ ] **Step 3: Test + commit**

```python
def test_optimize_threshold_youden(classification_data, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"], "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5", "--optimize-threshold", "youden"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    import json
    th = json.loads((out / "results/threshold.json").read_text())
    assert 0 < th["threshold"] < 1
    assert th["criterion"] == "youden"
```

```bash
python -m pytest tests/test_classify.py::test_optimize_threshold_youden -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): --optimize-threshold {youden,f1,mcc,fixed-recall} (LEAD-024)"
```

---

## Task 10: Subgroup + fairness + decision curve in evaluate

**Files:**
- Modify: `classify_reference.py` `evaluate(...)`

- [ ] **Step 1: Emit subgroup metrics + fairness + decision curve**

Inside `evaluate`, after threshold optimization, add:

```python
    # Subgroup + fairness metrics when --group-col is available.
    if args.group_col and len(np.unique(y_test)) == 2:
        groups_test = df[args.group_col].loc[X_test.index]
        try:
            from _shared.fairness import group_metrics, disparity_ratios
        except ImportError:
            from references._shared.fairness import group_metrics, disparity_ratios
        per = group_metrics(y_test, y_pred, y_proba, groups_test)
        per.to_csv(out / "results/subgroup_metrics.csv", index=False)
        ratios = disparity_ratios(per)
        (out / "results/fairness_disparities.json").write_text(json.dumps(ratios, indent=2))

    # Decision-curve analysis.
    if args.decision_curve and len(np.unique(y_test)) == 2:
        try:
            from _shared.decision_curve import decision_curve_figure
        except ImportError:
            from references._shared.decision_curve import decision_curve_figure
        fig = decision_curve_figure(y_test, y_proba)
        plotting.save_fig(fig, out / "artifacts/decision_curve"); plt.close(fig)
```

`df` must be in scope — pass it through or close over it from `main`. Simplest: change `evaluate(model, X_test, y_test, out)` to `evaluate(model, X_test, y_test, df, out, args)`.

- [ ] **Step 2: Test + commit**

```python
def test_subgroup_metrics_and_decision_curve(classification_data, tmp_path):
    import subprocess, sys, pandas as pd
    df = pd.read_csv(classification_data["path"])
    df["group"] = [i % 3 for i in range(len(df))]
    data_path = tmp_path / "with_group.csv"
    df.to_csv(data_path, index=False)
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", str(data_path), "--target", "target",
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5",
         "--group-col", "group", "--decision-curve"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/subgroup_metrics.csv").exists()
    assert (out / "results/fairness_disparities.json").exists()
    assert list((out / "artifacts").glob("decision_curve*"))
```

```bash
python -m pytest tests/test_classify.py::test_subgroup_metrics_and_decision_curve -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): subgroup metrics + fairness disparities + decision curve (LEAD-003, -026, -029)"
```

---

## Task 11: Expanded evaluate plots — learning curve, class report heatmap, PDP, SHAP

**Files:**
- Modify: `classify_reference.py` `evaluate(...)`

- [ ] **Step 1: Learning curve + class-report heatmap**

At the end of `evaluate`, add:

```python
    from sklearn.model_selection import learning_curve
    try:
        train_sizes, tr, te = learning_curve(
            model, X_test, y_test, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), random_state=42,
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, tr.mean(axis=1), "o-", label="Train")
        ax.plot(train_sizes, te.mean(axis=1), "o-", label="CV")
        ax.set_xlabel("Training examples"); ax.set_ylabel("Score")
        ax.set_title("Learning curve"); ax.legend()
        plotting.save_fig(fig, out / "artifacts/learning_curve"); plt.close(fig)
    except Exception:
        pass

    # Class-report heatmap (already computed as `report`).
    cr = pd.DataFrame(report).T.drop(columns="support", errors="ignore")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(cr))))
    sns.heatmap(cr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    plotting.save_fig(fig, out / "artifacts/classification_report_heatmap"); plt.close(fig)
```

- [ ] **Step 2: SHAP (gated)**

```python
    try:
        from _shared import deps
    except ImportError:
        from references._shared import deps
    if deps._check("shap"):
        try:
            import shap
            explainer = shap.Explainer(model.predict, X_test.iloc[:200])
            sv = explainer(X_test.iloc[:200])
            fig = plt.figure(figsize=(7, 6))
            shap.plots.beeswarm(sv, show=False)
            plotting.save_fig(fig, out / "artifacts/shap_beeswarm"); plt.close(fig)
        except Exception as e:
            print(f"SHAP failed (continuing): {e}", flush=True)
```

- [ ] **Step 3: Test + commit**

```python
def test_evaluate_emits_learning_curve(classification_data, tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"], "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all", "--cv", "3", "--n-iter", "5"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert list((out / "artifacts").glob("learning_curve*"))
    assert list((out / "artifacts").glob("classification_report_heatmap*"))
```

```bash
python -m pytest tests/test_classify.py::test_evaluate_emits_learning_curve -v
cd mltoolkit-plugin
git add skills/classify/references/classify_reference.py tests/test_classify.py
git commit -m "feat(mltoolkit): evaluate emits learning curve + class report heatmap + SHAP (LEAD-010, -017)"
```

---

## Task 12: classify SKILL.md — document every new flag

**Files:**
- Modify: `mltoolkit-plugin/skills/classify/SKILL.md`

- [ ] **Step 1: Add a "Paper-mode flags" section**

Insert after the existing "Adaptation rules" section:

```markdown
## Paper-mode flags

These flags turn the plugin into a TRIPOD+AI-grade reporting toolkit. Defaults preserve original behavior.

| Flag | Effect | Output |
|---|---|---|
| `--group-col <col>` | Route CV through GroupKFold / StratifiedGroupKFold. | per-fold + subgroup_metrics.csv |
| `--time-col <col>` | TimeSeriesSplit. | per-fold |
| `--sensitive-features a,b,c` | Plugin refuses to target-encode these. Required for `--group-col` based subgroup metrics when the group is a protected attribute. | |
| `--allow-target-encode-on-sensitive` | Override the refusal (record rationale in datasheet.md). | |
| `--imputation {simple,iterative,knn,drop}` | Imputer class. | missingness.png |
| `--resample {smote,adasyn}` | imblearn resampler before fit. | |
| `--calibrate {sigmoid,isotonic}` | Wrap final model in CalibratedClassifierCV. | calibration.json, reliability.png |
| `--optimize-threshold {youden,f1,mcc,fixed-recall} --fixed-recall 0.80` | Pick operating point. | threshold.json |
| `--decision-curve` | Vickers 2006 net-benefit plot. | decision_curve.png |
| `--bootstrap N` | N-sample percentile CI on holdout metrics. | holdout_metrics_ci.json |

Fill out `.mltoolkit/datasheet.md` (from setup) with protected-attribute column names and pass them via `--sensitive-features`.
```

- [ ] **Step 2: Commit**

```bash
cd mltoolkit-plugin
git add skills/classify/SKILL.md
git commit -m "docs(mltoolkit): classify SKILL.md — paper-mode flags reference"
```

---

## Task 13: End-to-end "paper-mode" run

**Files:**
- Modify: `mltoolkit-plugin/tests/test_stage_and_run.py` (append)

- [ ] **Step 1: Add the e2e test**

```python
def test_staged_classify_paper_mode_end_to_end(classification_data, tmp_path):
    """Stage classify + run with every paper-mode flag; verify every new artifact."""
    import pandas as pd
    df = pd.read_csv(classification_data["path"])
    df["group"] = [i % 4 for i in range(len(df))]
    data_path = tmp_path / "paper.csv"
    df.to_csv(data_path, index=False)

    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("classify", dest).check_returncode()
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", str(data_path), "--target", "target",
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5",
         "--group-col", "group",
         "--calibrate", "sigmoid",
         "--optimize-threshold", "youden",
         "--decision-curve",
         "--bootstrap", "100"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    for artifact in [
        "results/leaderboard.csv",
        "results/epv_audit.json",
        "results/table1.csv",
        "results/calibration.json",
        "results/threshold.json",
        "results/subgroup_metrics.csv",
        "results/fairness_disparities.json",
        "results/holdout_metrics_ci.json",
    ]:
        assert (out / artifact).exists(), f"missing {artifact}"
    assert list((out / "artifacts").glob("reliability*"))
    assert list((out / "artifacts").glob("decision_curve*"))
    assert list((out / "artifacts").glob("learning_curve*"))
```

- [ ] **Step 2: Run + commit**

```bash
python -m pytest tests/test_stage_and_run.py::test_staged_classify_paper_mode_end_to_end -v
cd mltoolkit-plugin
git add tests/test_stage_and_run.py
git commit -m "test(mltoolkit): e2e paper-mode run exercises every new flag"
```

---

## Self-review notes

- **Coverage:** LEAD-007→T1; -010→T11; -011→T6; -017→T11; -024→T9; -025→T3+T4; -026→T10; -031→T5; -003→T10; -004→T7; -005→T8; -006→T5; -008→T4; -012→T3; -027→T3; -029→T10.
- **Type consistency:** `evaluate` now takes `(model, X_test, y_test, df, out, args)` after T10; every caller passes these. `_preprocessor(X, args, sensitive)` is the only `build_preprocessor` wrapper. `_pipeline(pre, est, resample, sensitive)` is the only Pipeline/ImbPipeline wrapper.
- **Deferred to Plan 5:** run_manifest.json, --finalize stage, Tier-C code emission for these flags, methods.md, model card, STARD/CONSORT, MLflow tracking, experiment-tracking CLI.

## Out of scope

- Regress/cluster/anomaly equivalents (Plan 4).
- Packaging-side parity for new artifacts (Plan 5).
- Zoo expansion (Plan 4).
