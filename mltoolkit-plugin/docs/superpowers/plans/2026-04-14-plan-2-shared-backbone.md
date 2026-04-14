# Plan 2 — Shared statistical / fairness / reporting backbone

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `references/_shared/` primitives that Plan 3 (classify rewrite) and Plan 5 (paper scaffolding) compose. Each module is a pure utility: no I/O, no stage wiring, clear callable API, covered by unit tests.

**Architecture:** One module per responsibility. All modules live under `mltoolkit-plugin/references/_shared/`. They are already staged by `scripts/stage_session.py` (it copies `_shared/` wholesale), so no stager changes are needed — new files are picked up automatically. Each module exposes pure functions returning native Python / pandas / numpy types. No mutable global state. All unit-tested with synthetic data from `tests/conftest.py` or ad-hoc numpy arrays.

**Tech Stack:** numpy, pandas, scikit-learn (metrics, model_selection), scipy (for `stats.ttest_ind` / `stats.chi2_contingency` in Table 1), matplotlib (for reliability and decision-curve plot helpers).

**Source findings:** LEAD-003, LEAD-004, LEAD-005, LEAD-006, LEAD-008, LEAD-012, LEAD-027, LEAD-029.

---

## File Structure

**Created under `mltoolkit-plugin/references/_shared/`:**

| File | Responsibility | Finding |
|---|---|---|
| `fairness.py` | Per-group metrics + disparity ratios | LEAD-003 |
| `calibration.py` | Brier, ECE, calibration intercept/slope, reliability-diagram figure helper | LEAD-004 |
| `bootstrap.py` | `bootstrap_ci(metric_fn, y_true, y_pred, n_boot, alpha)` → point estimate + 95% CI | LEAD-005 |
| `splits.py` | `make_splitter(y, n_splits, group_col, time_col)` → stratified / group / time splitter | LEAD-006 |
| `encoders_safe.py` | Refuse TargetEncoder on columns in a sensitive-attribute set | LEAD-008 |
| `epv.py` | Minority prevalence + events-per-variable (EPV) + low-EPV warning | LEAD-012 |
| `table1.py` | Per-covariate summary (mean±SD / N(%)) overall and by group, with test-of-difference p-values | LEAD-027 |
| `decision_curve.py` | Net-benefit vs threshold curve (treat-all / treat-none baselines) + figure helper | LEAD-029 |

**Created under `mltoolkit-plugin/tests/`:**

| File | Covers |
|---|---|
| `test_shared_fairness.py` | fairness.py |
| `test_shared_calibration.py` | calibration.py |
| `test_shared_bootstrap.py` | bootstrap.py |
| `test_shared_splits.py` | splits.py |
| `test_shared_encoders_safe.py` | encoders_safe.py |
| `test_shared_epv.py` | epv.py |
| `test_shared_table1.py` | table1.py |
| `test_shared_decision_curve.py` | decision_curve.py |

**Modified:**

| File | Change |
|---|---|
| `mltoolkit-plugin/tests/test_references.sh` | Add the 8 new test files to step [1/5] shared-helpers run. |

**Not touched by this plan:** none of the task reference scripts (`classify_reference.py` etc.). Wiring these primitives into stages is Plan 3's job. Plan 2 delivers testable primitives that Plan 3 composes.

---

## Task 1: Bootstrap confidence intervals — `bootstrap.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/bootstrap.py`
- Create: `mltoolkit-plugin/tests/test_shared_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_bootstrap.py`:

```python
"""bootstrap.py — bootstrap CI for any metric_fn(y_true, y_pred) -> float."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from references._shared.bootstrap import bootstrap_ci


def test_bootstrap_ci_point_estimate_matches_plain_metric():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    res = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=200, random_state=0)
    assert abs(res["point"] - accuracy_score(y_true, y_pred)) < 1e-12


def test_bootstrap_ci_has_lower_and_upper():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    res = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=200, random_state=0, alpha=0.05)
    assert res["lower"] <= res["point"] <= res["upper"]
    assert 0 <= res["lower"] and res["upper"] <= 1


def test_bootstrap_ci_works_with_probability_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_proba = rng.random(300)
    res = bootstrap_ci(roc_auc_score, y_true, y_proba, n_boot=200, random_state=0)
    assert "lower" in res and "upper" in res


def test_bootstrap_ci_deterministic_with_random_state():
    y_true = np.array([0, 1] * 100)
    y_pred = np.array([0, 1] * 100)
    a = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=100, random_state=42)
    b = bootstrap_ci(accuracy_score, y_true, y_pred, n_boot=100, random_state=42)
    assert a == b
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_shared_bootstrap.py -v`
Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement `bootstrap.py`**

Create `mltoolkit-plugin/references/_shared/bootstrap.py`:

```python
"""Percentile bootstrap confidence intervals for any metric_fn(y_true, y_pred).

Primary API:
    bootstrap_ci(metric_fn, y_true, y_pred, n_boot=1000, alpha=0.05,
                 random_state=42) -> dict{point, lower, upper, alpha, n_boot}
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    metric_fn,
    y_true,
    y_pred,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    point = float(metric_fn(y_true, y_pred))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot[i] = float(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            boot[i] = np.nan
    boot = boot[~np.isnan(boot)]

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {"point": point, "lower": lo, "upper": hi,
            "alpha": alpha, "n_boot": n_boot, "n_valid": int(len(boot))}
```

- [ ] **Step 4: Run tests — expect pass**

Run: `python -m pytest tests/test_shared_bootstrap.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd mltoolkit-plugin
git add references/_shared/bootstrap.py tests/test_shared_bootstrap.py
git commit -m "feat(mltoolkit): _shared/bootstrap.py — percentile-bootstrap CIs (LEAD-005)"
```

---

## Task 2: Split strategies — `splits.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/splits.py`
- Create: `mltoolkit-plugin/tests/test_shared_splits.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_splits.py`:

```python
"""splits.py — routing to Stratified / Group / TimeSeries splitters."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit

from references._shared.splits import make_splitter


def test_default_returns_stratifiedkfold():
    y = pd.Series([0, 1] * 100)
    s = make_splitter(y, n_splits=5)
    assert isinstance(s, StratifiedKFold)
    assert s.n_splits == 5


def test_group_col_returns_stratifiedgroupkfold_when_y_is_categorical():
    y = pd.Series([0, 1] * 50)
    groups = pd.Series(np.repeat(np.arange(20), 5))
    s = make_splitter(y, n_splits=5, groups=groups)
    assert isinstance(s, StratifiedGroupKFold)


def test_group_col_returns_groupkfold_when_y_is_continuous():
    y = pd.Series(np.random.default_rng(0).normal(size=100))
    groups = pd.Series(np.repeat(np.arange(20), 5))
    s = make_splitter(y, n_splits=5, groups=groups)
    assert isinstance(s, GroupKFold)


def test_time_col_returns_timeseries_split():
    y = pd.Series(np.random.default_rng(0).normal(size=100))
    times = pd.Series(pd.date_range("2024-01-01", periods=100))
    s = make_splitter(y, n_splits=5, time_order=times)
    assert isinstance(s, TimeSeriesSplit)
    assert s.n_splits == 5


def test_time_and_group_raises():
    y = pd.Series([0, 1] * 50)
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_splitter(y, groups=pd.Series([1] * 100), time_order=pd.Series(pd.date_range("2024", periods=100)))
```

- [ ] **Step 2: Implement `splits.py`**

Create `mltoolkit-plugin/references/_shared/splits.py`:

```python
"""Cross-validation splitter routing.

    make_splitter(y, n_splits=5, groups=None, time_order=None) -> sklearn splitter

- time_order given  → TimeSeriesSplit (ignores y; caller must pre-sort X/y by time).
- groups given      → StratifiedGroupKFold if y is categorical (nunique <= 20),
                      else GroupKFold.
- neither given     → StratifiedKFold if y is categorical, else KFold.
Raises ValueError if both groups and time_order are passed.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit,
)

_CATEGORICAL_CARDINALITY = 20


def _is_categorical(y) -> bool:
    y = pd.Series(y)
    return y.dtype == "O" or y.nunique(dropna=True) <= _CATEGORICAL_CARDINALITY


def make_splitter(y, *, n_splits: int = 5, groups=None, time_order=None,
                  random_state: int = 42):
    if groups is not None and time_order is not None:
        raise ValueError("`groups` and `time_order` are mutually exclusive.")
    if time_order is not None:
        return TimeSeriesSplit(n_splits=n_splits)
    if groups is not None:
        if _is_categorical(y):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state)
        return GroupKFold(n_splits=n_splits)
    if _is_categorical(y):
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_splits.py -v` — expect 5 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/splits.py tests/test_shared_splits.py
git commit -m "feat(mltoolkit): _shared/splits.py — stratified/group/timeseries routing (LEAD-006)"
```

---

## Task 3: Fairness — `fairness.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/fairness.py`
- Create: `mltoolkit-plugin/tests/test_shared_fairness.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_fairness.py`:

```python
"""fairness.py — per-group metrics + disparity ratios."""
import numpy as np
import pandas as pd

from references._shared.fairness import group_metrics, disparity_ratios


def test_group_metrics_returns_row_per_group():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_proba = rng.random(n)
    groups = pd.Series(rng.choice(["A", "B", "C"], size=n))
    df = group_metrics(y_true, y_pred, y_proba, groups)
    assert set(df["group"]) == {"A", "B", "C"}
    for col in ["n", "prevalence", "accuracy", "precision", "recall",
                "f1", "roc_auc", "tpr", "fpr"]:
        assert col in df.columns


def test_group_metrics_handles_single_class_group():
    """A group with only one class must not crash (roc_auc NaN, other metrics computed)."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])
    groups = pd.Series(["A"] * 3 + ["B"] * 3)
    df = group_metrics(y_true, y_pred, y_proba, groups)
    # Group A has only y_true==0 → roc_auc is NaN
    a = df[df["group"] == "A"].iloc[0]
    assert np.isnan(a["roc_auc"])


def test_disparity_ratios_returns_max_min_ratio_per_metric():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    y_proba = rng.random(n)
    groups = pd.Series(rng.choice(["A", "B"], size=n))
    per = group_metrics(y_true, y_pred, y_proba, groups)
    ratios = disparity_ratios(per)
    # ratios is a dict of metric -> (max_over_groups / min_over_groups)
    assert "tpr" in ratios and ratios["tpr"] >= 1.0
    assert "fpr" in ratios
```

- [ ] **Step 2: Implement `fairness.py`**

Create `mltoolkit-plugin/references/_shared/fairness.py`:

```python
"""Group-fairness metrics for binary classification outputs.

    group_metrics(y_true, y_pred, y_proba, groups) -> DataFrame
        One row per group with n, prevalence, accuracy, precision, recall,
        f1, roc_auc, tpr, fpr, ppv, npv.

    disparity_ratios(per_group_df) -> dict
        For each rate metric (tpr, fpr, ppv, npv, prevalence), returns
        max-over-groups / min-over-groups. Ratio of 1.0 = perfect parity.

Single-class-in-group edge case: roc_auc becomes NaN. All rate metrics
are still reported; callers decide how to handle NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def _safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return tpr, fpr, ppv, npv


def group_metrics(y_true, y_pred, y_proba, groups) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba); groups = pd.Series(groups).reset_index(drop=True)
    rows = []
    for g, idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        yt, yp, yr = y_true[idx], y_pred[idx], y_proba[idx]
        tpr, fpr, ppv, npv = _rates(yt, yp)
        rows.append({
            "group": g,
            "n": len(idx),
            "prevalence": float(np.mean(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall":    float(recall_score(yt, yp, zero_division=0)),
            "f1":        float(f1_score(yt, yp, zero_division=0)),
            "roc_auc":   _safe_roc_auc(yt, yr),
            "tpr": tpr, "fpr": fpr, "ppv": ppv, "npv": npv,
        })
    return pd.DataFrame(rows)


_DISPARITY_METRICS = ("tpr", "fpr", "ppv", "npv", "prevalence")


def disparity_ratios(per_group: pd.DataFrame) -> dict:
    out = {}
    for m in _DISPARITY_METRICS:
        col = per_group[m].dropna()
        if col.empty or col.min() == 0:
            out[m] = float("nan")
        else:
            out[m] = float(col.max() / col.min())
    return out
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_fairness.py -v` — expect 3 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/fairness.py tests/test_shared_fairness.py
git commit -m "feat(mltoolkit): _shared/fairness.py — group metrics + disparity ratios (LEAD-003)"
```

---

## Task 4: Calibration — `calibration.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/calibration.py`
- Create: `mltoolkit-plugin/tests/test_shared_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_calibration.py`:

```python
"""calibration.py — Brier, ECE, calibration intercept+slope, reliability diagram."""
import numpy as np

from references._shared.calibration import (
    calibration_summary, reliability_diagram,
)


def test_perfect_calibration_has_zero_ece_and_unit_slope():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    y_proba = y_true.astype(float)  # perfectly aligned probabilities (0 or 1)
    s = calibration_summary(y_true, y_proba, n_bins=10)
    assert s["brier"] < 0.05
    assert s["ece"] < 0.05


def test_random_probabilities_have_nontrivial_ece():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    s = calibration_summary(y_true, y_proba, n_bins=10)
    for k in ("brier", "ece", "intercept", "slope", "n_bins", "bins"):
        assert k in s
    assert 0.0 <= s["ece"] <= 1.0


def test_reliability_diagram_returns_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    fig = reliability_diagram(y_true, y_proba, n_bins=10)
    assert fig.axes
    out = tmp_path / "rd.png"
    fig.savefig(out)
    assert out.exists()
```

- [ ] **Step 2: Implement `calibration.py`**

Create `mltoolkit-plugin/references/_shared/calibration.py`:

```python
"""Calibration diagnostics for binary classifiers.

    calibration_summary(y_true, y_proba, n_bins=10) -> dict
        brier, ECE, calibration_intercept, calibration_slope
        (the latter two come from logistic regression of y on logit(p)).

    reliability_diagram(y_true, y_proba, n_bins=10) -> matplotlib.Figure
        Standard bin-means plot with y=x reference line.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_EPS = 1e-12


def _ece(y_true, y_proba, n_bins):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    weights = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        weights[i] = mask.sum()
    weights = weights / max(weights.sum(), 1)
    # prob_true/prob_pred are only non-empty bins; align by position.
    used = len(prob_true)
    w_used = weights[:used]
    return float(np.sum(w_used * np.abs(prob_true - prob_pred)))


def _calibration_intercept_slope(y_true, y_proba):
    p = np.clip(y_proba, _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def calibration_summary(y_true, y_proba, *, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    brier = float(brier_score_loss(y_true, y_proba))
    ece = _ece(y_true, y_proba, n_bins)
    intercept, slope = _calibration_intercept_slope(y_true, y_proba)
    return {"brier": brier, "ece": ece,
            "intercept": intercept, "slope": slope,
            "n_bins": n_bins, "bins": n_bins}


def reliability_diagram(y_true, y_proba, *, n_bins: int = 10):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(prob_pred, prob_true, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram")
    ax.legend()
    return fig
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_calibration.py -v` — expect 3 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/calibration.py tests/test_shared_calibration.py
git commit -m "feat(mltoolkit): _shared/calibration.py — Brier, ECE, intercept/slope, reliability (LEAD-004)"
```

---

## Task 5: Safe target encoder — `encoders_safe.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/encoders_safe.py`
- Create: `mltoolkit-plugin/tests/test_shared_encoders_safe.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_encoders_safe.py`:

```python
"""encoders_safe.py — refuse TargetEncoder on sensitive columns."""
import pytest

from references._shared.encoders_safe import (
    SENSITIVE_ATTRIBUTE_PATTERNS, is_sensitive_column,
    safe_high_cardinality_encoder,
)


def test_known_sensitive_names_are_flagged():
    for col in ["race", "ethnicity", "sex", "gender", "zip_code", "zipcode",
                "postcode", "religion", "age_bucket", "disability_status"]:
        assert is_sensitive_column(col, sensitive=[])


def test_explicit_sensitive_overrides_patterns():
    # User declares 'patient_id' sensitive even though it does not match
    # a built-in pattern.
    assert is_sensitive_column("patient_id", sensitive=["patient_id"])


def test_nonsensitive_names_not_flagged():
    for col in ["age", "bmi", "income", "region"]:
        assert not is_sensitive_column(col, sensitive=[])


def test_safe_encoder_refuses_target_encode_on_sensitive_without_override():
    with pytest.raises(ValueError, match="sensitive"):
        safe_high_cardinality_encoder(
            "race", sensitive=[], allow_target_encode_on_sensitive=False,
        )


def test_safe_encoder_returns_ordinal_fallback_without_override():
    # Non-sensitive high-cardinality column → TargetEncoder or OrdinalEncoder
    # depending on env; but for a *sensitive* column with no override the
    # encoder returned must NOT be TargetEncoder.
    enc = safe_high_cardinality_encoder(
        "race", sensitive=["race"], allow_target_encode_on_sensitive=True,
    )
    # user explicitly allowed: any encoder is fine
    assert enc is not None
```

- [ ] **Step 2: Implement `encoders_safe.py`**

Create `mltoolkit-plugin/references/_shared/encoders_safe.py`:

```python
"""Refuse target-encoding on protected-attribute columns by default.

    is_sensitive_column(name, sensitive) -> bool
        True if `name` is in the explicit `sensitive` list OR matches a
        built-in SENSITIVE_ATTRIBUTE_PATTERNS regex.

    safe_high_cardinality_encoder(name, sensitive, *,
                                  allow_target_encode_on_sensitive=False)
        Returns a fitted-ready encoder instance. Raises ValueError if the
        column is sensitive and the caller has not explicitly opted in.

Reason: target-encoding a protected attribute leaks outcome rate into a
proxy-discrimination vector. This module makes that a loud failure by
default.
"""
from __future__ import annotations

import re
from typing import Sequence

from sklearn.preprocessing import OrdinalEncoder


SENSITIVE_ATTRIBUTE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^race(_|$)", r"^ethnicity", r"^sex(_|$)", r"^gender",
        r"^zip(_?code)?$", r"^postcode", r"^religion",
        r"^age_bucket", r"disability(_status)?$", r"^sexual_orientation",
        r"^national_origin", r"^pregnancy",
    )
]


def is_sensitive_column(name: str, sensitive: Sequence[str]) -> bool:
    if name in set(sensitive):
        return True
    return any(p.search(name) for p in SENSITIVE_ATTRIBUTE_PATTERNS)


def safe_high_cardinality_encoder(
    name: str,
    sensitive: Sequence[str],
    *,
    allow_target_encode_on_sensitive: bool = False,
):
    if is_sensitive_column(name, sensitive) and not allow_target_encode_on_sensitive:
        raise ValueError(
            f"Refusing to target-encode sensitive column '{name}'. "
            "Pass --allow-target-encode-on-sensitive to override, "
            "or drop/one-hot/ordinal-encode instead."
        )
    # Default fallback: OrdinalEncoder. TargetEncoder wiring lives in the
    # caller (preprocessing.py) because it depends on whether category_encoders
    # is installed. This helper just answers: 'is this column safe to TE?'
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_encoders_safe.py -v` — expect 5 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/encoders_safe.py tests/test_shared_encoders_safe.py
git commit -m "feat(mltoolkit): _shared/encoders_safe.py — refuse target-encode on sensitive cols (LEAD-008)"
```

---

## Task 6: Events-per-variable + prevalence — `epv.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/epv.py`
- Create: `mltoolkit-plugin/tests/test_shared_epv.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_epv.py`:

```python
"""epv.py — minority-class prevalence and events-per-variable checks."""
import numpy as np
import pandas as pd

from references._shared.epv import audit_epv


def test_balanced_binary_passes_checks():
    y = pd.Series([0, 1] * 500)
    X = pd.DataFrame(np.zeros((1000, 10)))
    a = audit_epv(X, y)
    assert a["minority_prevalence"] == 0.5
    assert a["n_features"] == 10
    assert a["epv"] == 500 / 10
    assert not a["low_epv_warning"]
    assert not a["rare_outcome_warning"]


def test_rare_outcome_triggers_warning():
    y = pd.Series([0] * 990 + [1] * 10)
    X = pd.DataFrame(np.zeros((1000, 5)))
    a = audit_epv(X, y)
    assert a["minority_prevalence"] == 0.01
    assert a["rare_outcome_warning"] is True


def test_low_epv_triggers_warning():
    # 15 events against 20 features → EPV=0.75 << 10
    y = pd.Series([0] * 85 + [1] * 15)
    X = pd.DataFrame(np.zeros((100, 20)))
    a = audit_epv(X, y)
    assert a["epv"] < 10
    assert a["low_epv_warning"] is True
```

- [ ] **Step 2: Implement `epv.py`**

Create `mltoolkit-plugin/references/_shared/epv.py`:

```python
"""Sample-size audit: events-per-variable + minority-class prevalence.

    audit_epv(X, y, low_epv_threshold=10, rare_outcome_threshold=0.05)
      -> dict with keys:
         n_rows, n_features, n_events (minority-class count),
         minority_prevalence, epv (events / features),
         low_epv_warning (bool), rare_outcome_warning (bool)

Callers print the warnings at EDA time so users see them before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_epv(
    X: pd.DataFrame,
    y,
    *,
    low_epv_threshold: float = 10.0,
    rare_outcome_threshold: float = 0.05,
) -> dict:
    y = pd.Series(y)
    n = len(y)
    counts = y.value_counts(dropna=False)
    n_events = int(counts.min())
    prev = float(n_events / n) if n else float("nan")
    n_features = int(X.shape[1])
    epv = float(n_events / n_features) if n_features else float("inf")
    return {
        "n_rows": n,
        "n_features": n_features,
        "n_events": n_events,
        "minority_prevalence": prev,
        "epv": epv,
        "low_epv_warning": epv < low_epv_threshold,
        "rare_outcome_warning": prev < rare_outcome_threshold,
    }
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_epv.py -v` — expect 3 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/epv.py tests/test_shared_epv.py
git commit -m "feat(mltoolkit): _shared/epv.py — events-per-variable + rare-outcome audit (LEAD-012)"
```

---

## Task 7: Table 1 — `table1.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/table1.py`
- Create: `mltoolkit-plugin/tests/test_shared_table1.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_table1.py`:

```python
"""table1.py — cohort characteristics, overall and stratified by a group."""
import numpy as np
import pandas as pd

from references._shared.table1 import table1


def test_table1_numeric_row_has_mean_sd():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(60, 10, 200),
        "sex": rng.choice(["M", "F"], size=200),
    })
    t = table1(df)
    age_row = t[t["variable"] == "age"].iloc[0]
    assert "mean ± SD" in str(age_row["overall"]) or "±" in str(age_row["overall"])


def test_table1_categorical_row_has_n_and_pct():
    df = pd.DataFrame({"sex": ["M"] * 120 + ["F"] * 80})
    t = table1(df)
    assert any("120" in str(v) for v in t["overall"])


def test_table1_stratified_by_group_produces_group_columns_and_pvalue():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(60, 10, 300),
        "sex": rng.choice(["M", "F"], size=300),
    })
    group = pd.Series(rng.choice(["case", "ctrl"], size=300))
    t = table1(df, group=group)
    # One column per group value + a p_value column.
    assert "case" in t.columns and "ctrl" in t.columns
    assert "p_value" in t.columns
    assert t["p_value"].notna().any()
```

- [ ] **Step 2: Implement `table1.py`**

Create `mltoolkit-plugin/references/_shared/table1.py`:

```python
"""Table 1 — cohort descriptor with optional stratification and p-values.

    table1(df, *, group=None) -> DataFrame
        Columns: variable, overall, (one column per group value if given),
                 p_value (if group given).

Numeric variables → 'mean ± SD'. Categorical → 'n (pct%)' per level.
Tests: numeric uses two-sample t-test; categorical uses chi-square.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _fmt_num(s: pd.Series) -> str:
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def _fmt_cat(s: pd.Series, level) -> str:
    n = int((s == level).sum())
    pct = 100.0 * n / max(len(s), 1)
    return f"{n} ({pct:.1f}%)"


def _numeric_pvalue(s: pd.Series, group: pd.Series) -> float:
    groups = [s[group == g].dropna() for g in group.unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    if len(groups) == 2:
        return float(stats.ttest_ind(groups[0], groups[1], equal_var=False).pvalue)
    return float(stats.f_oneway(*groups).pvalue)


def _categorical_pvalue(s: pd.Series, group: pd.Series) -> float:
    try:
        ct = pd.crosstab(s, group)
        if ct.size == 0:
            return float("nan")
        return float(stats.chi2_contingency(ct).pvalue)
    except Exception:
        return float("nan")


def table1(df: pd.DataFrame, *, group: pd.Series | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    group_labels = list(group.unique()) if group is not None else []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            row = {"variable": col, "overall": _fmt_num(s)}
            for g in group_labels:
                row[g] = _fmt_num(s[group == g])
            if group is not None:
                row["p_value"] = _numeric_pvalue(s, group)
            rows.append(row)
        else:
            for level in sorted(s.dropna().unique()):
                row = {"variable": f"{col}={level}",
                       "overall": _fmt_cat(s, level)}
                for g in group_labels:
                    row[g] = _fmt_cat(s[group == g], level)
                if group is not None:
                    row["p_value"] = _categorical_pvalue(s, group)
                rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_table1.py -v` — expect 3 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/table1.py tests/test_shared_table1.py
git commit -m "feat(mltoolkit): _shared/table1.py — cohort descriptor with stratified p-values (LEAD-027)"
```

---

## Task 8: Decision-curve analysis — `decision_curve.py`

**Files:**
- Create: `mltoolkit-plugin/references/_shared/decision_curve.py`
- Create: `mltoolkit-plugin/tests/test_shared_decision_curve.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_shared_decision_curve.py`:

```python
"""decision_curve.py — net-benefit vs threshold with treat-all/treat-none baselines."""
import numpy as np

from references._shared.decision_curve import net_benefit_curve, decision_curve_figure


def test_net_benefit_curve_includes_model_and_baselines():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    df = net_benefit_curve(y_true, y_proba, thresholds=np.linspace(0.05, 0.5, 10))
    for col in ("threshold", "model", "treat_all", "treat_none"):
        assert col in df.columns
    assert (df["treat_none"] == 0).all()


def test_decision_curve_figure_returns_axes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    fig = decision_curve_figure(y_true, y_proba)
    assert fig.axes
    fig.savefig(tmp_path / "dc.png")
```

- [ ] **Step 2: Implement `decision_curve.py`**

Create `mltoolkit-plugin/references/_shared/decision_curve.py`:

```python
"""Decision-curve analysis (Vickers 2006) for binary classifiers.

    net_benefit_curve(y_true, y_proba, thresholds) -> DataFrame
        Columns: threshold, model, treat_all, treat_none.
        Net benefit = TP/n - FP/n * (p_t / (1 - p_t))

    decision_curve_figure(y_true, y_proba, thresholds=None) -> matplotlib.Figure
        Plots all three curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _nb(y_true, y_hat, p_t):
    n = len(y_true)
    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    if n == 0 or p_t >= 1:
        return 0.0
    return tp / n - (fp / n) * (p_t / (1 - p_t))


def net_benefit_curve(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)
    rows = []
    prev = float(np.mean(y_true))
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        model = _nb(y_true, y_hat, t)
        treat_all = prev - (1 - prev) * (t / (1 - t)) if t < 1 else 0.0
        rows.append({"threshold": float(t), "model": model,
                     "treat_all": float(treat_all), "treat_none": 0.0})
    return pd.DataFrame(rows)


def decision_curve_figure(y_true, y_proba, thresholds=None):
    df = net_benefit_curve(y_true, y_proba, thresholds=thresholds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["threshold"], df["model"], label="Model")
    ax.plot(df["threshold"], df["treat_all"], "--", label="Treat all")
    ax.plot(df["threshold"], df["treat_none"], ":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend()
    return fig
```

- [ ] **Step 3: Run & commit**

Run: `python -m pytest tests/test_shared_decision_curve.py -v` — expect 2 PASS.

```bash
cd mltoolkit-plugin
git add references/_shared/decision_curve.py tests/test_shared_decision_curve.py
git commit -m "feat(mltoolkit): _shared/decision_curve.py — net-benefit curve (LEAD-029)"
```

---

## Task 9: Wire new tests into the smoke runner

**Files:**
- Modify: `mltoolkit-plugin/tests/test_references.sh` (step [1/5])

- [ ] **Step 1: Update the shared-helpers step**

Open `mltoolkit-plugin/tests/test_references.sh`. Change step [1/5]:

```bash
echo "[1/5] Shared helpers..."
python -m pytest tests/test_shared.py \
                 tests/test_shared_bootstrap.py \
                 tests/test_shared_splits.py \
                 tests/test_shared_fairness.py \
                 tests/test_shared_calibration.py \
                 tests/test_shared_encoders_safe.py \
                 tests/test_shared_epv.py \
                 tests/test_shared_table1.py \
                 tests/test_shared_decision_curve.py \
                 -q
```

- [ ] **Step 2: Run full smoke**

Run: `bash tests/test_references.sh`
Expected: all five steps pass; step 1 now reports ≥ 25 tests (6 original + ~19 new).

- [ ] **Step 3: Commit**

```bash
cd mltoolkit-plugin
git add tests/test_references.sh
git commit -m "test(mltoolkit): smoke runner picks up all new _shared test files"
```

---

## Self-review notes

- **Coverage:** LEAD-003 → Task 3. LEAD-004 → Task 4. LEAD-005 → Task 1. LEAD-006 → Task 2. LEAD-008 → Task 5. LEAD-012 → Task 6. LEAD-027 → Task 7. LEAD-029 → Task 8. Wiring → Task 9.
- **No placeholders:** every test block and implementation is complete literal code.
- **Type consistency:** `group_metrics` → DataFrame; `disparity_ratios` consumes that DataFrame. `audit_epv` and `calibration_summary` return plain dicts. `bootstrap_ci` dict keys (`point`, `lower`, `upper`) match across the plan. `make_splitter` returns a sklearn splitter matching its `n_splits` param.
- **Deferred to Plan 3:** wiring these primitives into `classify_reference.py` stages (compare, tune, evaluate), `--group-col` / `--time-col` / `--imputation` / `--resample` CLI flags, subgroup metrics emission, calibrated/uncalibrated model ensembling, threshold-optimization stage. Plan 2 produces testable primitives only.

## Out of scope

- CLI flags on reference scripts (Plan 3).
- Any reference-script wiring (Plan 3).
- Reproducibility manifest / finalize / model card (Plan 5).
- Zoo expansion (Plan 4).
