# mltoolkit Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Claude Code plugin that generates native-Python ML code (sklearn + optional boosters) which runs inline during the session and packages into a tiered deliverable on demand.

**Architecture:** Reference-script + skill-playbook architecture. Each skill owns a complete, tested Python reference. Claude copies a reference into `.mltoolkit/session.py` in the user's CWD, adapts variables, runs it staged, and shows results. A `package` skill transforms the session scratchpad into a Tier A/B/C deliverable.

**Tech Stack:** Python ≥3.9, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib. Optional: xgboost, lightgbm, catboost, imblearn, category_encoders, optuna, plotly. Plugin ships only `SKILL.md` files, reference `.py` scripts, shell scripts, and pytest tests — no runtime Python library.

**Spec:** `docs/superpowers/specs/2026-04-14-mltoolkit-plugin-design.md`

---

## File Structure

All paths relative to repo root. Plugin lives at `mltoolkit-plugin/`:

```
mltoolkit-plugin/
├── .claude-plugin/
│   └── plugin.json                          # Plugin manifest
├── scripts/
│   └── check-env.sh                         # Reports Python/package versions
├── references/
│   └── _shared/
│       ├── __init__.py
│       ├── deps.py                          # has_xgboost(), etc.
│       ├── plotting.py                      # set_style(), save_fig()
│       └── reporting.py                     # df_to_markdown, summary_report
├── skills/
│   ├── setup/
│   │   ├── SKILL.md
│   │   └── references/setup_reference.py    # Data load + EDA
│   ├── classify/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── classify_reference.py        # End-to-end classification
│   │       ├── preprocessing.py             # ColumnTransformer builder
│   │       └── model_zoo.py                 # id -> {estimator, param_grid}
│   ├── regress/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── regress_reference.py
│   │       ├── preprocessing.py
│   │       └── model_zoo.py
│   ├── cluster/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── cluster_reference.py
│   │       └── model_zoo.py
│   ├── anomaly/
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── anomaly_reference.py
│   │       └── model_zoo.py
│   ├── compare/SKILL.md
│   ├── tune/SKILL.md
│   ├── eda/SKILL.md
│   └── package/
│       ├── SKILL.md
│       └── references/
│           ├── tier_a_template.py           # Flatten script
│           ├── tier_b_readme_template.md
│           └── tier_c_scaffold/
│               ├── src/{preprocess,train,predict}.py
│               ├── tests/test_pipeline.py
│               ├── api.py
│               ├── Dockerfile
│               └── requirements.txt
├── agents/ml-pipeline.md
└── tests/
    ├── conftest.py                          # Synthetic data fixtures
    ├── test_shared.py                       # Unit tests for _shared helpers
    ├── test_classify.py                     # Smoke tests
    ├── test_regress.py
    ├── test_cluster.py
    ├── test_anomaly.py
    ├── test_package.py
    └── test_references.sh                   # End-to-end runner
```

**Boundaries:** Each reference script is independently runnable and self-testable. Shared helpers have no dependencies on skill-specific code. The package skill transforms scripts but does not import them.

---

## Task 1: Plugin scaffold & test infrastructure

**Files:**
- Create: `mltoolkit-plugin/.claude-plugin/plugin.json`
- Create: `mltoolkit-plugin/scripts/check-env.sh`
- Create: `mltoolkit-plugin/references/_shared/__init__.py`
- Create: `mltoolkit-plugin/tests/conftest.py`
- Create: `mltoolkit-plugin/.gitignore`

- [ ] **Step 1: Write the plugin manifest**

Create `mltoolkit-plugin/.claude-plugin/plugin.json`:
```json
{
  "name": "mltoolkit",
  "version": "0.1.0",
  "description": "Standalone ML plugin — generates native Python code (sklearn + optional boosters) for classification, regression, clustering, and anomaly detection. No PyCaret dependency.",
  "skills": [
    "skills/setup",
    "skills/classify",
    "skills/regress",
    "skills/cluster",
    "skills/anomaly",
    "skills/compare",
    "skills/tune",
    "skills/eda",
    "skills/package"
  ],
  "agents": ["agents/ml-pipeline.md"]
}
```

- [ ] **Step 2: Write the environment check script**

Create `mltoolkit-plugin/scripts/check-env.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== mltoolkit environment check ==="
echo "Python: $(python3 --version 2>&1)"
python3 - <<'PY'
import sys, importlib
required = ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "joblib"]
optional = ["xgboost", "lightgbm", "catboost", "imblearn", "category_encoders", "optuna", "plotly"]
missing_req = []
print("\nRequired:")
for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {m}: NOT INSTALLED")
        missing_req.append(m)
print("\nOptional:")
for m in optional:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {m}: not installed")
if missing_req:
    sys.exit(f"\nMissing required: {missing_req}")
print("\nAll required packages present.")
PY
```

Make it executable: `chmod +x mltoolkit-plugin/scripts/check-env.sh`

- [ ] **Step 3: Write the pytest fixtures**

Create `mltoolkit-plugin/tests/conftest.py`:
```python
"""Shared pytest fixtures: synthetic datasets for reference-script smoke tests."""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs


@pytest.fixture
def classification_data(tmp_path):
    """Binary classification dataset with mixed numeric/categorical features."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    # Add a categorical column
    df["cat_a"] = np.random.RandomState(42).choice(["a", "b", "c"], size=500)
    df["target"] = y
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "target": "target", "df": df}


@pytest.fixture
def regression_data(tmp_path):
    """Regression dataset."""
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    df["cat_a"] = np.random.RandomState(42).choice(["a", "b", "c"], size=500)
    df["target"] = y
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "target": "target", "df": df}


@pytest.fixture
def cluster_data(tmp_path):
    """Unsupervised clustering dataset (no target)."""
    X, _ = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    path = tmp_path / "cluster.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "df": df}


@pytest.fixture
def anomaly_data(tmp_path):
    """Dataset with injected outliers."""
    rng = np.random.RandomState(42)
    normal = rng.normal(0, 1, size=(480, 5))
    outliers = rng.normal(8, 1, size=(20, 5))
    X = np.vstack([normal, outliers])
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    path = tmp_path / "anomaly.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "df": df}
```

- [ ] **Step 4: Write the .gitignore**

Create `mltoolkit-plugin/.gitignore`:
```
__pycache__/
*.pyc
.pytest_cache/
.mltoolkit/
```

- [ ] **Step 5: Create empty __init__.py and verify scaffold**

Create `mltoolkit-plugin/references/_shared/__init__.py` (empty file).

Run: `bash mltoolkit-plugin/scripts/check-env.sh`
Expected: lists Python + package versions, exits 0 if sklearn/pandas/matplotlib installed.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/.claude-plugin mltoolkit-plugin/scripts mltoolkit-plugin/references mltoolkit-plugin/tests/conftest.py mltoolkit-plugin/.gitignore
git commit -m "feat(mltoolkit): scaffold plugin manifest, env check, test fixtures"
```

---

## Task 2: Shared helper — deps.py

**Files:**
- Create: `mltoolkit-plugin/references/_shared/deps.py`
- Create: `mltoolkit-plugin/tests/test_shared.py`

- [ ] **Step 1: Write the failing test**

Create `mltoolkit-plugin/tests/test_shared.py`:
```python
"""Unit tests for references/_shared helpers."""
import sys
from pathlib import Path

# Make the references package importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from references._shared import deps


def test_deps_returns_booleans():
    """All has_* functions return bool without raising."""
    for fn_name in [
        "has_xgboost", "has_lightgbm", "has_catboost",
        "has_imblearn", "has_category_encoders", "has_optuna", "has_plotly",
    ]:
        fn = getattr(deps, fn_name)
        result = fn()
        assert isinstance(result, bool), f"{fn_name} returned {type(result)}"


def test_deps_sklearn_always_available():
    """sklearn is required — has_sklearn must return True in test env."""
    assert deps.has_sklearn() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py -v`
Expected: FAIL — ModuleNotFoundError on `references._shared.deps`.

- [ ] **Step 3: Write deps.py**

Create `mltoolkit-plugin/references/_shared/deps.py`:
```python
"""Optional-dependency detection helpers.

Each ``has_*`` function returns a bool without raising. Reference scripts
use these to gracefully skip models/features when their packages are missing.
"""
import importlib


def _check(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def has_sklearn() -> bool:
    return _check("sklearn")


def has_xgboost() -> bool:
    return _check("xgboost")


def has_lightgbm() -> bool:
    return _check("lightgbm")


def has_catboost() -> bool:
    return _check("catboost")


def has_imblearn() -> bool:
    return _check("imblearn")


def has_category_encoders() -> bool:
    return _check("category_encoders")


def has_optuna() -> bool:
    return _check("optuna")


def has_plotly() -> bool:
    return _check("plotly")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mltoolkit-plugin/references/_shared/deps.py mltoolkit-plugin/tests/test_shared.py
git commit -m "feat(mltoolkit): add deps.py for optional-package detection"
```

---

## Task 2a: Shared helper — plotting.py

**Files:**
- Create: `mltoolkit-plugin/references/_shared/plotting.py`
- Modify: `mltoolkit-plugin/tests/test_shared.py` (add plotting tests)

- [ ] **Step 1: Write the failing tests**

Append to `mltoolkit-plugin/tests/test_shared.py`:
```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt
from references._shared import plotting


def test_set_style_is_idempotent():
    """set_style() can be called repeatedly without error."""
    plotting.set_style()
    plotting.set_style()


def test_save_fig_writes_png_and_pdf(tmp_path):
    """save_fig writes both PNG and PDF at 300 DPI."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = plotting.save_fig(fig, tmp_path / "test_plot")
    assert (tmp_path / "test_plot.png").exists()
    assert (tmp_path / "test_plot.pdf").exists()
    assert paths == {
        "png": tmp_path / "test_plot.png",
        "pdf": tmp_path / "test_plot.pdf",
    }
    plt.close(fig)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py::test_save_fig_writes_png_and_pdf -v`
Expected: FAIL — ImportError on `references._shared.plotting`.

- [ ] **Step 3: Write plotting.py**

Create `mltoolkit-plugin/references/_shared/plotting.py`:
```python
"""Consistent figure styling + save helpers for publication-quality output."""
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Apply the mltoolkit default publication style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })


def save_fig(fig, path: Union[str, Path]) -> dict:
    """Save a figure as both PNG and PDF at 300 DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Base path without extension. Both .png and .pdf are written.

    Returns
    -------
    dict with keys 'png' and 'pdf' mapping to the actual Path objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}
```

- [ ] **Step 4: Run tests**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add mltoolkit-plugin/references/_shared/plotting.py mltoolkit-plugin/tests/test_shared.py
git commit -m "feat(mltoolkit): add plotting.py with consistent style + save_fig"
```

---

## Task 2b: Shared helper — reporting.py

**Files:**
- Create: `mltoolkit-plugin/references/_shared/reporting.py`
- Modify: `mltoolkit-plugin/tests/test_shared.py`

- [ ] **Step 1: Write the failing tests**

Append to `mltoolkit-plugin/tests/test_shared.py`:
```python
import pandas as pd
from references._shared import reporting


def test_df_to_markdown_contains_headers():
    df = pd.DataFrame({"model": ["rf", "lr"], "accuracy": [0.9, 0.8]})
    md = reporting.df_to_markdown(df)
    assert "model" in md and "accuracy" in md
    assert "rf" in md and "lr" in md


def test_summary_report_writes_html(tmp_path):
    df = pd.DataFrame({"model": ["rf"], "accuracy": [0.9]})
    out = tmp_path / "report.html"
    result = reporting.summary_report(
        title="Test Run",
        tables={"Leaderboard": df},
        figures=[],
        output=out,
    )
    assert out.exists()
    content = out.read_text()
    assert "Test Run" in content
    assert "Leaderboard" in content
    assert result == out
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py -v -k "markdown or summary_report"`
Expected: FAIL — ImportError on `references._shared.reporting`.

- [ ] **Step 3: Write reporting.py**

Create `mltoolkit-plugin/references/_shared/reporting.py`:
```python
"""Markdown / LaTeX / HTML report generation for ML results."""
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    return df.to_markdown(index=False, floatfmt=floatfmt)


def df_to_latex(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a LaTeX table (booktabs style)."""
    return df.to_latex(index=False, float_format=lambda x: f"%{floatfmt}" % x)


def summary_report(
    title: str,
    tables: Dict[str, pd.DataFrame],
    figures: List[Union[str, Path]],
    output: Union[str, Path],
) -> Path:
    """Emit a one-page HTML summary with tables + embedded figures.

    Parameters
    ----------
    title : str
    tables : dict of {section_name: DataFrame}
    figures : list of image paths (PNG)
    output : destination HTML path
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }",
        "h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }",
        "h2 { color: #333; margin-top: 1.5em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f5f5f5; }",
        "img { max-width: 100%; margin: 1em 0; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for name, df in tables.items():
        parts.append(f"<h2>{name}</h2>")
        parts.append(df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if figures:
        parts.append("<h2>Figures</h2>")
        for fig_path in figures:
            parts.append(f'<img src="{Path(fig_path).as_posix()}" alt="figure">')
    parts.append("</body></html>")
    output.write_text("\n".join(parts))
    return output
```

- [ ] **Step 4: Run tests**

Run: `cd mltoolkit-plugin && pytest tests/test_shared.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add mltoolkit-plugin/references/_shared/reporting.py mltoolkit-plugin/tests/test_shared.py
git commit -m "feat(mltoolkit): add reporting.py with markdown/latex/html helpers"
```

---

## Task 3: Classify — preprocessing + model_zoo

**Files:**
- Create: `mltoolkit-plugin/skills/classify/references/preprocessing.py`
- Create: `mltoolkit-plugin/skills/classify/references/model_zoo.py`
- Create: `mltoolkit-plugin/tests/test_classify.py`

- [ ] **Step 1: Write the failing tests**

Create `mltoolkit-plugin/tests/test_classify.py`:
```python
"""Smoke tests for classify reference components."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "skills/classify/references"))

import preprocessing
import model_zoo


def test_build_preprocessor_on_mixed_df(classification_data):
    df = classification_data["df"].drop(columns=["target"])
    pre = preprocessing.build_preprocessor(df)
    X = pre.fit_transform(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] >= df.shape[1]  # OHE expands


def test_model_zoo_has_core_entries():
    zoo = model_zoo.get_zoo()
    for mid in ["lr", "ridge", "knn", "dt", "rf", "et", "gbc", "ada", "svc", "nb", "mlp"]:
        assert mid in zoo, f"Missing core model: {mid}"
        entry = zoo[mid]
        assert "estimator" in entry
        assert "param_grid" in entry


def test_model_zoo_optional_models_respect_deps():
    zoo = model_zoo.get_zoo()
    # If xgboost is installed, 'xgb' should be present; otherwise absent
    from references._shared import deps
    if deps.has_xgboost():
        assert "xgb" in zoo
    else:
        assert "xgb" not in zoo
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mltoolkit-plugin && pytest tests/test_classify.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write preprocessing.py**

Create `mltoolkit-plugin/skills/classify/references/preprocessing.py`:
```python
"""Preprocessing pipeline builder for classification tasks."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Ensure _shared is importable regardless of CWD
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]  # mltoolkit-plugin/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from references._shared import deps  # noqa: E402


CARDINALITY_THRESHOLD = 10  # categoricals with > this go to target-encoding path


def _split_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in categorical if df[c].nunique(dropna=True) <= CARDINALITY_THRESHOLD]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > CARDINALITY_THRESHOLD]
    return numeric, low_card, high_card


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer tuned for the given DataFrame's schema."""
    numeric, low_card, high_card = _split_columns(df)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scl", StandardScaler())]),
            numeric,
        ))
    if low_card:
        transformers.append((
            "cat_low",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            low_card,
        ))
    if high_card:
        if deps.has_category_encoders():
            import category_encoders as ce
            enc = ce.TargetEncoder()
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append((
            "cat_high",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", enc)]),
            high_card,
        ))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
```

- [ ] **Step 4: Write model_zoo.py**

Create `mltoolkit-plugin/skills/classify/references/model_zoo.py`:
```python
"""Classification model zoo — id -> {estimator, param_grid, requires}."""
import sys
from pathlib import Path

from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


def get_zoo() -> dict:
    zoo = {
        "lr": {
            "estimator": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
            "param_grid": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "requires": None,
        },
        "ridge": {
            "estimator": RidgeClassifier(random_state=42),
            "param_grid": {"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "requires": None,
        },
        "knn": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
            "requires": None,
        },
        "dt": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
            "requires": None,
        },
        "rf": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
            "requires": None,
        },
        "et": {
            "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
            "param_grid": {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20]},
            "requires": None,
        },
        "gbc": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            "requires": None,
        },
        "ada": {
            "estimator": AdaBoostClassifier(random_state=42),
            "param_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
            "requires": None,
        },
        "svc": {
            "estimator": SVC(probability=True, random_state=42),
            "param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "requires": None,
        },
        "nb": {
            "estimator": GaussianNB(),
            "param_grid": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
            "requires": None,
        },
        "mlp": {
            "estimator": MLPClassifier(max_iter=500, random_state=42),
            "param_grid": {"hidden_layer_sizes": [(50,), (100,), (100, 50)],
                           "alpha": [0.0001, 0.001]},
            "requires": None,
        },
    }
    if deps.has_xgboost():
        import xgboost as xgb
        zoo["xgb"] = {
            "estimator": xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
            "param_grid": {"n_estimators": [100, 300], "max_depth": [3, 6, 9],
                           "learning_rate": [0.05, 0.1]},
            "requires": "xgboost",
        }
    if deps.has_lightgbm():
        import lightgbm as lgb
        zoo["lgbm"] = {
            "estimator": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            "param_grid": {"n_estimators": [100, 300], "num_leaves": [31, 63],
                           "learning_rate": [0.05, 0.1]},
            "requires": "lightgbm",
        }
    if deps.has_catboost():
        import catboost as cb
        zoo["cat"] = {
            "estimator": cb.CatBoostClassifier(random_state=42, verbose=False),
            "param_grid": {"iterations": [100, 300], "depth": [4, 6, 8],
                           "learning_rate": [0.05, 0.1]},
            "requires": "catboost",
        }
    return zoo
```

- [ ] **Step 5: Run tests**

Run: `cd mltoolkit-plugin && pytest tests/test_classify.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/classify/references/preprocessing.py mltoolkit-plugin/skills/classify/references/model_zoo.py mltoolkit-plugin/tests/test_classify.py
git commit -m "feat(mltoolkit): classify preprocessing + model zoo with optional booster support"
```

---

## Task 4: Classify — reference script (end-to-end)

**Files:**
- Create: `mltoolkit-plugin/skills/classify/references/classify_reference.py`
- Modify: `mltoolkit-plugin/tests/test_classify.py` (add end-to-end smoke test)

- [ ] **Step 1: Write the failing smoke test**

Append to `mltoolkit-plugin/tests/test_classify.py`:
```python
import subprocess


def test_classify_reference_runs_end_to_end(classification_data, tmp_path):
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    out_dir = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out_dir),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "results/leaderboard.csv").exists()
    assert (out_dir / "artifacts").exists()
    # At least one figure produced
    figures = list((out_dir / "artifacts").glob("*.png"))
    assert len(figures) >= 1
    # Saved model
    assert (out_dir / "model.joblib").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mltoolkit-plugin && pytest tests/test_classify.py::test_classify_reference_runs_end_to_end -v`
Expected: FAIL — script not found.

- [ ] **Step 3: Write classify_reference.py**

Create `mltoolkit-plugin/skills/classify/references/classify_reference.py`:
```python
"""Publication-quality classification pipeline — standalone, no PyCaret.

Stages (via --stage):
    eda       Data overview + exploratory figures
    compare   Cross-validated comparison of all available models
    tune      RandomizedSearchCV on the best model from compare
    evaluate  Holdout evaluation with confusion matrix, ROC, PR curves
    all       Runs eda -> compare -> tune -> evaluate

Usage:
    python classify_reference.py --data data.csv --target y --output-dir out/ --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


# ----- Stages ---------------------------------------------------------------

def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in columns: {list(df.columns)}")
    if df[target].nunique() < 2:
        raise SystemExit(f"Target '{target}' has only one unique value.")
    return df


def run_eda(df: pd.DataFrame, target: str, out: Path):
    plotting.set_style()
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "nunique": df.nunique(),
    })
    summary.to_csv(out / "results/schema.csv")

    # Class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Class distribution — {target}")
    ax.set_ylabel("count")
    plotting.save_fig(fig, out / "artifacts/class_distribution")
    plt.close(fig)

    # Correlation heatmap for numeric features
    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap")
        plt.close(fig)


def compare_models(X_train, y_train, out: Path, cv: int = 5) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = preprocessing.build_preprocessor(X_train)

    n_classes = len(np.unique(y_train))
    scorers = {
        "accuracy": "accuracy",
        "f1": "f1_macro" if n_classes > 2 else "f1",
    }
    if n_classes == 2:
        scorers["roc_auc"] = "roc_auc"

    rows = []
    for mid, entry in zoo.items():
        pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])
        row = {"model": mid}
        try:
            for name, scoring in scorers.items():
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                row[name] = float(scores.mean())
                row[f"{name}_std"] = float(scores.std())
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(
        by="roc_auc" if "roc_auc" in scorers else "f1",
        ascending=False, na_position="last",
    ).reset_index(drop=True)
    leaderboard.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(leaderboard))
    return leaderboard


def tune_model(model_id: str, X_train, y_train, out: Path, n_iter: int = 20):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = preprocessing.build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("clf", entry["estimator"])])
    grid = {f"clf__{k}": v for k, v in entry["param_grid"].items()}

    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, sum(len(v) for v in grid.values())),
        cv=5, scoring="accuracy", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    with open(out / "results/best_params.json", "w") as f:
        json.dump({k: str(v) for k, v in search.best_params_.items()}, f, indent=2)
    return search.best_estimator_, search.best_score_


def evaluate(model, X_test, y_test, out: Path):
    plotting.set_style()
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(out / "results/classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plotting.save_fig(fig, out / "artifacts/confusion_matrix"); plt.close(fig)

    # ROC + PR for binary
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("ROC Curve")
        plotting.save_fig(fig, out / "artifacts/roc_curve"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Precision-Recall Curve")
        plotting.save_fig(fig, out / "artifacts/pr_curve"); plt.close(fig)

    # Permutation importance
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "results/permutation_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(imp))))
        top = imp.head(20)[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("Feature importance (permutation)")
        plotting.save_fig(fig, out / "artifacts/feature_importance"); plt.close(fig)
    except Exception:
        pass


# ----- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "tune", "evaluate", "all"], default="all")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--model", default=None, help="Model id for --stage tune (defaults to top of leaderboard)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.target)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y if y.nunique() < 20 else None, random_state=42,
    )

    if args.stage in ("eda", "all"):
        run_eda(df, args.target, out)

    best_model = None
    if args.stage in ("compare", "all"):
        lb = compare_models(X_train, y_train, out, cv=args.cv)
        top_id = lb.iloc[0]["model"]
        best_model_id = top_id
    else:
        best_model_id = args.model

    if args.stage in ("tune", "all") and best_model_id:
        best_model, score = tune_model(best_model_id, X_train, y_train, out)
        print(f"Tuned {best_model_id}: CV score = {score:.4f}")

    if args.stage in ("evaluate", "all") and best_model is not None:
        best_model.fit(X_train, y_train)
        evaluate(best_model, X_test, y_test, out)
        joblib.dump(best_model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test**

Run: `cd mltoolkit-plugin && pytest tests/test_classify.py::test_classify_reference_runs_end_to_end -v -s`
Expected: PASS. Script runs on synthetic data and produces leaderboard + figures + model.

- [ ] **Step 5: Commit**

```bash
git add mltoolkit-plugin/skills/classify/references/classify_reference.py mltoolkit-plugin/tests/test_classify.py
git commit -m "feat(mltoolkit): end-to-end classify reference script with staged CLI"
```

---

## Task 5: Classify — SKILL.md playbook

**Files:**
- Create: `mltoolkit-plugin/skills/classify/SKILL.md`

- [ ] **Step 1: Write the SKILL.md**

Create `mltoolkit-plugin/skills/classify/SKILL.md`:
```markdown
---
name: classify
description: Build a classification pipeline with native sklearn code. Runs inline, shows leaderboard + figures, then packages into a deliverable.
triggers:
  - classify
  - classification
  - predict category
  - binary classification
  - multiclass
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Classification Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py` (in user's CWD)
- Reference: `{SKILL_DIR}/references/classify_reference.py`
- Plugin-wide helpers: `{PLUGIN_ROOT}/references/_shared/`
- Confirm the user's data path and target column before running anything.

## Workflow

1. **Read the reference script** at `{SKILL_DIR}/references/classify_reference.py` so you know its stages.
2. **Create `.mltoolkit/` if missing** in the user's CWD and add to `.gitignore`.
3. **Copy the reference** to `.mltoolkit/session.py` (use Write tool).
4. **Run EDA stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`
5. **Read and present** `.mltoolkit/results/schema.csv` and the generated figures in `.mltoolkit/artifacts/`.
6. **Run compare stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
7. **Read and present** `.mltoolkit/results/leaderboard.csv` as a markdown table.
8. **Ask the user** which model(s) to tune. If unsure, propose the top 1.
9. **Run tune stage**:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID>`
10. **Run evaluate stage**:
    `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage evaluate`
11. **Present holdout metrics** and figures (confusion matrix, ROC, PR, feature importance).
12. **Ask the user** if they want to package the pipeline (invoke `mltoolkit:package`).

## Adaptation rules (invariants)

- **Never import pycaret** in any generated code.
- **Never hardcode the target column** — always route it through `--target`.
- If rows > 100k: pass `--cv 3` to reduce fold count; warn the user.
- If the target has >20 classes, skip stratified splitting (script handles this automatically).
- If a class imbalance >4:1 is detected in EDA and `imblearn` is installed, mention SMOTE as an option for the user.
- All artifacts land under `.mltoolkit/artifacts/` and `.mltoolkit/results/`. Do not write elsewhere without asking.

## Iteration prompts

When user asks to:
- **"add a model"** → edit `.mltoolkit/session.py`, extend `get_zoo()` import with the new model, re-run compare.
- **"remove a model"** → add `--exclude` filter to the session copy and re-run.
- **"tune more"** → re-run tune with higher `--n-iter` (may require editing the script).
- **"handle imbalance"** → edit `session.py` to wrap estimator in `imblearn.pipeline.Pipeline` with SMOTE.

## When to hand off to package

Prompt the user to package once:
- They've picked a winning model
- Holdout metrics look acceptable
- Iteration has stabilized

Then invoke `mltoolkit:package` and pass through the task type (`classification`).
```

- [ ] **Step 2: Verify the file is valid**

Run: `head -30 mltoolkit-plugin/skills/classify/SKILL.md`
Expected: valid YAML frontmatter + markdown body.

- [ ] **Step 3: Commit**

```bash
git add mltoolkit-plugin/skills/classify/SKILL.md
git commit -m "feat(mltoolkit): classify SKILL.md playbook"
```

---

## Task 6: Regress — preprocessing + model_zoo + reference

**Files:**
- Create: `mltoolkit-plugin/skills/regress/references/preprocessing.py`
- Create: `mltoolkit-plugin/skills/regress/references/model_zoo.py`
- Create: `mltoolkit-plugin/skills/regress/references/regress_reference.py`
- Create: `mltoolkit-plugin/tests/test_regress.py`

- [ ] **Step 1: Write the smoke test**

Create `mltoolkit-plugin/tests/test_regress.py`:
```python
"""Smoke test for regress reference."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_regress_reference_runs_end_to_end(regression_data, tmp_path):
    script = REPO_ROOT / "skills/regress/references/regress_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", regression_data["path"],
         "--target", regression_data["target"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/leaderboard.csv").exists()
    assert (out / "model.joblib").exists()
    figures = list((out / "artifacts").glob("*.png"))
    assert len(figures) >= 1
```

- [ ] **Step 2: Write regress preprocessing.py**

Create `mltoolkit-plugin/skills/regress/references/preprocessing.py` — identical to the classify version (same logic applies to regression targets):
```python
"""Preprocessing pipeline for regression tasks (identical to classify — targets don't change feature prep)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


CARDINALITY_THRESHOLD = 10


def _split_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in categorical if df[c].nunique(dropna=True) <= CARDINALITY_THRESHOLD]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > CARDINALITY_THRESHOLD]
    return numeric, low_card, high_card


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric, low_card, high_card = _split_columns(df)
    transformers = []
    if numeric:
        transformers.append(("num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]),
            numeric))
    if low_card:
        transformers.append(("cat_low",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            low_card))
    if high_card:
        if deps.has_category_encoders():
            import category_encoders as ce
            enc = ce.TargetEncoder()
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat_high",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]),
            high_card))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
```

- [ ] **Step 3: Write regress model_zoo.py**

Create `mltoolkit-plugin/skills/regress/references/model_zoo.py`:
```python
"""Regression model zoo."""
import sys
from pathlib import Path

from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


def get_zoo() -> dict:
    zoo = {
        "lr":    {"estimator": LinearRegression(n_jobs=-1), "param_grid": {}, "requires": None},
        "ridge": {"estimator": Ridge(random_state=42),
                  "param_grid": {"alpha": [0.1, 1.0, 10.0]}, "requires": None},
        "lasso": {"estimator": Lasso(random_state=42, max_iter=5000),
                  "param_grid": {"alpha": [0.01, 0.1, 1.0]}, "requires": None},
        "en":    {"estimator": ElasticNet(random_state=42, max_iter=5000),
                  "param_grid": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
                  "requires": None},
        "knn":   {"estimator": KNeighborsRegressor(n_jobs=-1),
                  "param_grid": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
                  "requires": None},
        "dt":    {"estimator": DecisionTreeRegressor(random_state=42),
                  "param_grid": {"max_depth": [None, 5, 10, 20]}, "requires": None},
        "rf":    {"estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
                  "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
                  "requires": None},
        "et":    {"estimator": ExtraTreesRegressor(random_state=42, n_jobs=-1),
                  "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
                  "requires": None},
        "gbr":   {"estimator": GradientBoostingRegressor(random_state=42),
                  "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                                 "max_depth": [3, 5]}, "requires": None},
        "ada":   {"estimator": AdaBoostRegressor(random_state=42),
                  "param_grid": {"n_estimators": [50, 100]}, "requires": None},
        "svr":   {"estimator": SVR(),
                  "param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
                  "requires": None},
        "mlp":   {"estimator": MLPRegressor(max_iter=500, random_state=42),
                  "param_grid": {"hidden_layer_sizes": [(50,), (100,), (100, 50)]},
                  "requires": None},
    }
    if deps.has_xgboost():
        import xgboost as xgb
        zoo["xgb"] = {"estimator": xgb.XGBRegressor(random_state=42, n_jobs=-1),
                      "param_grid": {"n_estimators": [100, 300], "max_depth": [3, 6, 9],
                                     "learning_rate": [0.05, 0.1]}, "requires": "xgboost"}
    if deps.has_lightgbm():
        import lightgbm as lgb
        zoo["lgbm"] = {"estimator": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                       "param_grid": {"n_estimators": [100, 300], "num_leaves": [31, 63]},
                       "requires": "lightgbm"}
    if deps.has_catboost():
        import catboost as cb
        zoo["cat"] = {"estimator": cb.CatBoostRegressor(random_state=42, verbose=False),
                      "param_grid": {"iterations": [100, 300], "depth": [4, 6, 8]},
                      "requires": "catboost"}
    return zoo
```

- [ ] **Step 4: Write regress_reference.py**

Create `mltoolkit-plugin/skills/regress/references/regress_reference.py`:
```python
"""Publication-quality regression pipeline — standalone."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import preprocessing  # noqa: E402
import model_zoo  # noqa: E402


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in columns: {list(df.columns)}")
    return df


def run_eda(df: pd.DataFrame, target: str, out: Path):
    plotting.set_style()
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "nunique": df.nunique(),
    }).to_csv(out / "results/schema.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[target], kde=True, ax=ax)
    ax.set_title(f"Target distribution — {target}")
    plotting.save_fig(fig, out / "artifacts/target_distribution"); plt.close(fig)

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap"); plt.close(fig)


def compare_models(X_train, y_train, out: Path, cv: int = 5) -> pd.DataFrame:
    zoo = model_zoo.get_zoo()
    pre = preprocessing.build_preprocessor(X_train)

    rows = []
    for mid, entry in zoo.items():
        pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
        row = {"model": mid}
        try:
            r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            neg_rmse = cross_val_score(pipe, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error", n_jobs=-1)
            neg_mae = cross_val_score(pipe, X_train, y_train, cv=cv,
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
            row["r2"] = float(r2.mean()); row["r2_std"] = float(r2.std())
            row["rmse"] = float(-neg_rmse.mean()); row["mae"] = float(-neg_mae.mean())
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)

    lb = pd.DataFrame(rows).sort_values(by="r2", ascending=False, na_position="last").reset_index(drop=True)
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    return lb


def tune_model(model_id: str, X_train, y_train, out: Path, n_iter: int = 20):
    zoo = model_zoo.get_zoo()
    entry = zoo[model_id]
    pre = preprocessing.build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("reg", entry["estimator"])])
    grid = {f"reg__{k}": v for k, v in entry["param_grid"].items()}
    if not grid:  # model has no tunable params (e.g. plain LinearRegression)
        pipe.fit(X_train, y_train); return pipe, None
    search = RandomizedSearchCV(
        pipe, grid, n_iter=min(n_iter, sum(len(v) for v in grid.values())),
        cv=5, scoring="r2", n_jobs=-1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    with open(out / "results/best_params.json", "w") as f:
        json.dump({k: str(v) for k, v in search.best_params_.items()}, f, indent=2)
    return search.best_estimator_, search.best_score_


def evaluate(model, X_test, y_test, out: Path):
    plotting.set_style()
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    with open(out / "results/holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    residuals = y_test - y_pred

    # Residuals vs fitted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted"); ax.set_ylabel("Residual"); ax.set_title("Residuals vs Fitted")
    plotting.save_fig(fig, out / "artifacts/residuals_vs_fitted"); plt.close(fig)

    # Prediction vs actual
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Prediction vs Actual")
    plotting.save_fig(fig, out / "artifacts/prediction_vs_actual"); plt.close(fig)

    # Q-Q plot
    from scipy import stats as scipy_stats
    fig, ax = plt.subplots(figsize=(6, 5))
    scipy_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of residuals")
    plotting.save_fig(fig, out / "artifacts/qq_plot"); plt.close(fig)

    # Permutation importance
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({"feature": X_test.columns,
                            "importance_mean": r.importances_mean,
                            "importance_std": r.importances_std}
                          ).sort_values("importance_mean", ascending=False)
        imp.to_csv(out / "results/permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(imp))))
        top = imp.head(20)[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_xlabel("Permutation importance"); ax.set_title("Feature importance")
        plotting.save_fig(fig, out / "artifacts/feature_importance"); plt.close(fig)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "tune", "evaluate", "all"], default="all")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.target)
    X = df.drop(columns=[args.target]); y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.stage in ("eda", "all"):
        run_eda(df, args.target, out)

    best_id = None
    if args.stage in ("compare", "all"):
        lb = compare_models(X_train, y_train, out, cv=args.cv)
        best_id = lb.iloc[0]["model"]
    else:
        best_id = args.model

    best_model = None
    if args.stage in ("tune", "all") and best_id:
        best_model, score = tune_model(best_id, X_train, y_train, out)
        print(f"Tuned {best_id}: CV R² = {score}")

    if args.stage in ("evaluate", "all") and best_model is not None:
        best_model.fit(X_train, y_train)
        evaluate(best_model, X_test, y_test, out)
        joblib.dump(best_model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run smoke test**

Run: `cd mltoolkit-plugin && pytest tests/test_regress.py -v -s`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/regress mltoolkit-plugin/tests/test_regress.py
git commit -m "feat(mltoolkit): regress reference + preprocessing + model zoo"
```

---

## Task 7: Regress — SKILL.md

**Files:**
- Create: `mltoolkit-plugin/skills/regress/SKILL.md`

- [ ] **Step 1: Write the playbook**

Create `mltoolkit-plugin/skills/regress/SKILL.md` (structure mirrors classify; adjust metrics/plots):
```markdown
---
name: regress
description: Build a regression pipeline with native sklearn code. Runs inline, shows leaderboard + residual/Q-Q/prediction plots, packages on demand.
triggers:
  - regress
  - regression
  - predict number
  - continuous prediction
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Regression Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py` (user's CWD)
- Reference: `{SKILL_DIR}/references/regress_reference.py`
- Shared helpers: `{PLUGIN_ROOT}/references/_shared/`

## Workflow

1. Read `{SKILL_DIR}/references/regress_reference.py`.
2. Ensure `.mltoolkit/` exists in user's CWD (gitignored).
3. Copy the reference to `.mltoolkit/session.py`.
4. Run EDA:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`
5. Show target distribution + correlation heatmap from `.mltoolkit/artifacts/`.
6. Run compare:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
7. Present `leaderboard.csv` — R², RMSE, MAE across models.
8. Ask which model to tune; default to top of R².
9. Run tune + evaluate stages.
10. Present residuals, Q-Q plot, prediction-vs-actual, permutation importance.
11. Offer to invoke `mltoolkit:package`.

## Adaptation rules

- **No pycaret imports, ever.**
- **Target always parameterized.**
- If rows > 100k, use `--cv 3`.
- If the target distribution is highly skewed (suggest via EDA), propose log-transform: edit `session.py` to wrap `y` with `np.log1p` / `np.expm1` via `TransformedTargetRegressor`.
- If residuals show strong heteroskedasticity, suggest quantile regression or a different model family.

## Iteration prompts

- **"transform the target"** → wrap estimator in `sklearn.compose.TransformedTargetRegressor`.
- **"remove outliers"** → add `IsolationForest` filter before splitting.
- **"try robust regression"** → add `HuberRegressor` or `RANSACRegressor` to the zoo import in `session.py`.

## Hand-off

Same as classify — invoke `mltoolkit:package` with task type `regression`.
```

- [ ] **Step 2: Commit**

```bash
git add mltoolkit-plugin/skills/regress/SKILL.md
git commit -m "feat(mltoolkit): regress SKILL.md playbook"
```

---

## Task 8: Cluster — reference + model_zoo + SKILL.md

**Files:**
- Create: `mltoolkit-plugin/skills/cluster/references/model_zoo.py`
- Create: `mltoolkit-plugin/skills/cluster/references/cluster_reference.py`
- Create: `mltoolkit-plugin/skills/cluster/SKILL.md`
- Create: `mltoolkit-plugin/tests/test_cluster.py`

- [ ] **Step 1: Write smoke test**

Create `mltoolkit-plugin/tests/test_cluster.py`:
```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_cluster_reference_runs_end_to_end(cluster_data, tmp_path):
    script = REPO_ROOT / "skills/cluster/references/cluster_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", cluster_data["path"],
         "--output-dir", str(out),
         "--stage", "all",
         "--n-clusters", "4"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/leaderboard.csv").exists()
    assert (out / "artifacts/pca_scatter.png").exists()
    assert (out / "results/assigned.csv").exists()
```

- [ ] **Step 2: Write cluster model_zoo.py**

Create `mltoolkit-plugin/skills/cluster/references/model_zoo.py`:
```python
"""Clustering model zoo."""
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture


def get_zoo(n_clusters: int = 4) -> dict:
    return {
        "kmeans":    {"estimator": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
                      "requires_n_clusters": True},
        "dbscan":    {"estimator": DBSCAN(),
                      "requires_n_clusters": False},
        "agglom":    {"estimator": AgglomerativeClustering(n_clusters=n_clusters),
                      "requires_n_clusters": True},
        "gmm":       {"estimator": GaussianMixture(n_components=n_clusters, random_state=42),
                      "requires_n_clusters": True},
    }
```

- [ ] **Step 3: Write cluster_reference.py**

Create `mltoolkit-plugin/skills/cluster/references/cluster_reference.py`:
```python
"""Publication-quality clustering pipeline — standalone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402


def _prep(df: pd.DataFrame) -> np.ndarray:
    num = df.select_dtypes(include="number")
    if num.empty:
        raise SystemExit("Clustering requires at least one numeric feature.")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return pipe.fit_transform(num), num.columns.tolist()


def elbow_plot(X: np.ndarray, out: Path, max_k: int = 10):
    from sklearn.cluster import KMeans
    inertias = []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), inertias, "o-")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow plot (KMeans)")
    plotting.save_fig(fig, out / "artifacts/elbow"); plt.close(fig)


def compare(X, out: Path, n_clusters: int):
    zoo = model_zoo.get_zoo(n_clusters=n_clusters)
    rows = []
    labelings = {}
    for mid, entry in zoo.items():
        try:
            model = entry["estimator"]
            if isinstance(model, type(zoo["gmm"]["estimator"])):  # GMM uses .predict
                labels = model.fit(X).predict(X)
            else:
                labels = model.fit_predict(X)
            labelings[mid] = labels
            if len(set(labels)) > 1 and -1 not in set(labels) or (len(set(labels) - {-1}) > 1):
                sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
            else:
                sil = float("nan")
            rows.append({"model": mid, "n_clusters": len(set(labels) - {-1}),
                         "silhouette": sil, "noise_points": int((labels == -1).sum())})
        except Exception as e:
            rows.append({"model": mid, "error": str(e)})
    lb = pd.DataFrame(rows).sort_values(by="silhouette", ascending=False, na_position="last")
    lb.to_csv(out / "results/leaderboard.csv", index=False)
    (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))
    return lb, labelings


def plot_pca(X, labels, out: Path, name: str):
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap="tab10", s=18, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"PCA projection — {name}")
    plotting.save_fig(fig, out / f"artifacts/pca_scatter"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "assign", "all"], default="all")
    ap.add_argument("--n-clusters", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True); (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    X, feat_names = _prep(df)

    if args.stage in ("eda", "all"):
        elbow_plot(X, out)

    if args.stage in ("compare", "all"):
        lb, labelings = compare(X, out, args.n_clusters)
        best = lb.iloc[0]["model"]
        if args.stage == "all":
            plot_pca(X, labelings[best], out, best)
            df_out = df.copy(); df_out["cluster"] = labelings[best]
            df_out.to_csv(out / "results/assigned.csv", index=False)
            joblib.dump(model_zoo.get_zoo(args.n_clusters)[best]["estimator"], out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write cluster SKILL.md**

Create `mltoolkit-plugin/skills/cluster/SKILL.md`:
```markdown
---
name: cluster
description: Build a clustering pipeline — KMeans, DBSCAN, Agglomerative, GMM. Runs inline with elbow + silhouette + PCA visualization.
triggers:
  - cluster
  - clustering
  - segment
  - group similar
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Clustering Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py`
- Reference: `{SKILL_DIR}/references/cluster_reference.py`

## Workflow

1. Read reference script.
2. Copy to `.mltoolkit/session.py`.
3. Run EDA — elbow plot for choosing k:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage eda`
4. Read elbow plot from `.mltoolkit/artifacts/elbow.png` — suggest a k to the user.
5. Run compare with chosen k:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage compare --n-clusters <K>`
6. Present leaderboard (silhouette, noise points).
7. Show PCA scatter colored by cluster labels.
8. Offer `mltoolkit:package`.

## Adaptation rules

- **No pycaret.**
- All features must be numeric after prep — if categorical columns exist, prompt user whether to one-hot them or drop.
- For DBSCAN, `n_clusters` is ignored (density-based).
- Suggest scaling is always on (already baked into the pipeline).
```

- [ ] **Step 5: Run smoke test**

Run: `cd mltoolkit-plugin && pytest tests/test_cluster.py -v -s`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/cluster mltoolkit-plugin/tests/test_cluster.py
git commit -m "feat(mltoolkit): cluster reference + SKILL.md"
```

---

## Task 9: Anomaly — reference + model_zoo + SKILL.md

**Files:**
- Create: `mltoolkit-plugin/skills/anomaly/references/model_zoo.py`
- Create: `mltoolkit-plugin/skills/anomaly/references/anomaly_reference.py`
- Create: `mltoolkit-plugin/skills/anomaly/SKILL.md`
- Create: `mltoolkit-plugin/tests/test_anomaly.py`

- [ ] **Step 1: Write smoke test**

Create `mltoolkit-plugin/tests/test_anomaly.py`:
```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_anomaly_reference_runs_end_to_end(anomaly_data, tmp_path):
    script = REPO_ROOT / "skills/anomaly/references/anomaly_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", anomaly_data["path"],
         "--output-dir", str(out),
         "--stage", "all",
         "--contamination", "0.05"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/scores.csv").exists()
    figures = list((out / "artifacts").glob("*.png"))
    assert len(figures) >= 1
```

- [ ] **Step 2: Write anomaly model_zoo.py**

Create `mltoolkit-plugin/skills/anomaly/references/model_zoo.py`:
```python
"""Anomaly detection model zoo."""
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def get_zoo(contamination: float = 0.05) -> dict:
    return {
        "iforest":   {"estimator": IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)},
        "lof":       {"estimator": LocalOutlierFactor(contamination=contamination, novelty=False)},
        "elliptic":  {"estimator": EllipticEnvelope(contamination=contamination, random_state=42)},
        "ocsvm":     {"estimator": OneClassSVM(nu=contamination)},
    }
```

- [ ] **Step 3: Write anomaly_reference.py**

Create `mltoolkit-plugin/skills/anomaly/references/anomaly_reference.py`:
```python
"""Publication-quality anomaly detection pipeline — standalone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
sys.path.insert(0, str(_HERE.parent))

from references._shared import plotting, reporting  # noqa: E402
import model_zoo  # noqa: E402


def _prep(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.empty:
        raise SystemExit("Anomaly detection requires numeric features.")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])
    return pipe.fit_transform(num), num.columns.tolist()


def score(X, model_id: str, contamination: float):
    entry = model_zoo.get_zoo(contamination)[model_id]
    est = entry["estimator"]
    if isinstance(est, LocalOutlierFactor):
        labels = est.fit_predict(X)
        # LOF: score_samples is negative_outlier_factor_
        scores = est.negative_outlier_factor_
    else:
        est.fit(X)
        labels = est.predict(X)
        scores = est.score_samples(X) if hasattr(est, "score_samples") else est.decision_function(X)
    # labels: 1 = inlier, -1 = outlier → convert to 0/1 anomaly flag
    is_anomaly = (labels == -1).astype(int)
    return est, scores, is_anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", choices=["eda", "compare", "assign", "all"], default="all")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--model", default="iforest")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True); (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    X, feat_names = _prep(df)

    if args.stage in ("compare", "all"):
        rows = []
        for mid in model_zoo.get_zoo(args.contamination):
            try:
                _, _, is_anom = score(X, mid, args.contamination)
                rows.append({"model": mid, "n_anomalies": int(is_anom.sum()),
                             "anomaly_rate": float(is_anom.mean())})
            except Exception as e:
                rows.append({"model": mid, "error": str(e)})
        lb = pd.DataFrame(rows)
        lb.to_csv(out / "results/leaderboard.csv", index=False)
        (out / "results/leaderboard.md").write_text(reporting.df_to_markdown(lb))

    model, scores, is_anom = score(X, args.model, args.contamination)

    # Score histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=50, alpha=0.7)
    ax.set_xlabel("Anomaly score"); ax.set_title(f"Score distribution — {args.model}")
    plotting.save_fig(fig, out / f"artifacts/score_histogram"); plt.close(fig)

    # PCA scatter
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = np.where(is_anom == 1, "red", "steelblue")
    ax.scatter(Xp[:, 0], Xp[:, 1], c=colors, s=18, alpha=0.7)
    ax.set_title(f"Anomalies (red) vs normal — {args.model}")
    plotting.save_fig(fig, out / "artifacts/pca_anomaly_scatter"); plt.close(fig)

    # Save scores + save model
    out_df = df.copy(); out_df["anomaly_score"] = scores; out_df["is_anomaly"] = is_anom
    out_df.to_csv(out / "results/scores.csv", index=False)
    # Top anomalies
    top = out_df.sort_values("anomaly_score").head(20)
    top.to_csv(out / "results/top_anomalies.csv", index=False)

    if not isinstance(model, LocalOutlierFactor):
        joblib.dump(model, out / "model.joblib")

    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write anomaly SKILL.md**

Create `mltoolkit-plugin/skills/anomaly/SKILL.md`:
```markdown
---
name: anomaly
description: Build an anomaly detection pipeline — IsolationForest, LOF, EllipticEnvelope, OneClassSVM. Ranks anomalies and visualizes on PCA.
triggers:
  - anomaly
  - outlier detection
  - find outliers
  - anomaly detection
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Anomaly Detection Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py`
- Reference: `{SKILL_DIR}/references/anomaly_reference.py`

## Workflow

1. Read reference.
2. Copy to `.mltoolkit/session.py`.
3. Ask user for expected contamination rate (default 5%).
4. Run compare with contamination:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage compare --contamination 0.05`
5. Present leaderboard (anomaly counts per model).
6. Run assign with chosen model (default iforest):
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage all --model iforest --contamination 0.05`
7. Present score histogram, PCA scatter, top-20 anomaly table.
8. Offer `mltoolkit:package`.

## Adaptation rules

- **No pycaret.**
- Contamination rate is critical — ask the user if unclear.
- Numeric features only after prep; prompt before dropping categoricals.
- LOF doesn't serialize (no `.joblib` saved) — note this to the user.
```

- [ ] **Step 5: Run smoke test**

Run: `cd mltoolkit-plugin && pytest tests/test_anomaly.py -v -s`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/anomaly mltoolkit-plugin/tests/test_anomaly.py
git commit -m "feat(mltoolkit): anomaly reference + SKILL.md"
```

---

## Task 10: Setup skill + setup_reference.py

**Files:**
- Create: `mltoolkit-plugin/skills/setup/references/setup_reference.py`
- Create: `mltoolkit-plugin/skills/setup/SKILL.md`

- [ ] **Step 1: Write setup_reference.py**

Create `mltoolkit-plugin/skills/setup/references/setup_reference.py`:
```python
"""Generic data-loading + EDA helper. Task-agnostic — run first in a session."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PLUGIN_ROOT))
from references._shared import plotting  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--output-dir", default=".mltoolkit")
    args = ap.parse_args()

    out = Path(args.output_dir); (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    plotting.set_style()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns from {args.data}")
    print(df.head())

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    summary.to_csv(out / "results/schema.csv")
    print("\nSchema:\n", summary)

    num_df = df.select_dtypes(include="number")
    if len(num_df.columns) >= 2:
        fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_df.columns) + 4),) * 2)
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        plotting.save_fig(fig, out / "artifacts/correlation_heatmap"); plt.close(fig)

    if args.target and args.target in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        if df[args.target].dtype.kind in "iufc" and df[args.target].nunique() > 20:
            sns.histplot(df[args.target], kde=True, ax=ax)
            ax.set_title(f"Target distribution — {args.target} (continuous)")
        else:
            df[args.target].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Target distribution — {args.target} (categorical)")
        plotting.save_fig(fig, out / "artifacts/target_distribution"); plt.close(fig)

    print(f"\nDone. Artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write setup SKILL.md**

Create `mltoolkit-plugin/skills/setup/SKILL.md`:
```markdown
---
name: setup
description: Prepare the mltoolkit session — load data, run EDA, auto-detect task type, create .mltoolkit scratchpad.
triggers:
  - load data
  - start ml
  - setup experiment
  - initialize
  - prepare data
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# mltoolkit Setup Playbook

## Workflow

1. **Check environment**: run `bash {PLUGIN_ROOT}/scripts/check-env.sh`. Report any missing required packages.
2. **Locate data**: confirm the data path with the user.
3. **Create `.mltoolkit/`** in user's CWD if missing. Add `.mltoolkit/` to `.gitignore` (create `.gitignore` if absent).
4. **Run setup_reference**:
   `python {SKILL_DIR}/references/setup_reference.py --data <DATA> [--target <TARGET>] --output-dir .mltoolkit`
5. **Read `schema.csv`** and the generated figures. Present to user.
6. **Infer task type**:
   - User specified target + target has ≤20 unique values or dtype object → classification
   - User specified target + target continuous → regression
   - No target → clustering or anomaly detection (ask user)
7. **Suggest next skill**: `mltoolkit:classify`, `mltoolkit:regress`, `mltoolkit:cluster`, or `mltoolkit:anomaly`.

## Adaptation rules

- Always create `.mltoolkit/` before any downstream skill runs.
- Never modify the user's data file.
- If rows > 1M, warn user and suggest subsampling.
```

- [ ] **Step 3: Commit**

```bash
git add mltoolkit-plugin/skills/setup
git commit -m "feat(mltoolkit): setup skill + reference"
```

---

## Task 11: Cross-cutting skills — compare, tune, eda

**Files:**
- Create: `mltoolkit-plugin/skills/compare/SKILL.md`
- Create: `mltoolkit-plugin/skills/tune/SKILL.md`
- Create: `mltoolkit-plugin/skills/eda/SKILL.md`

These skills don't own references — they instruct Claude to invoke the appropriate task-specific reference script with the right `--stage` flag.

- [ ] **Step 1: Write compare SKILL.md**

Create `mltoolkit-plugin/skills/compare/SKILL.md`:
```markdown
---
name: compare
description: Run cross-validated model comparison for the current task. Produces a leaderboard CSV + markdown.
triggers:
  - compare models
  - leaderboard
  - which model is best
  - model comparison
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Model Comparison Playbook

## Prerequisite

An active session scratchpad at `.mltoolkit/session.py`. If missing, invoke `mltoolkit:setup` first, then the task-specific skill (`classify`/`regress`/`cluster`/`anomaly`).

## Workflow

1. **Detect task type** by reading `.mltoolkit/session.py` (look for imports of classify/regress/cluster/anomaly modules).
2. **Run compare stage**: `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
   - Pass `--target` only if task is supervised.
3. **Read `.mltoolkit/results/leaderboard.csv`** and present as markdown.
4. **Identify top model(s)** for next steps.
```

- [ ] **Step 2: Write tune SKILL.md**

Create `mltoolkit-plugin/skills/tune/SKILL.md`:
```markdown
---
name: tune
description: Hyperparameter tuning for the current model — uses RandomizedSearchCV (or optuna if installed).
triggers:
  - tune
  - hyperparameter tuning
  - optimize hyperparameters
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Tuning Playbook

## Workflow

1. **Confirm model id** with user (from leaderboard).
2. **Run tune stage**: `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID>`
3. **Present `best_params.json`** and the new CV score.
4. If the tuned score is not better than the untuned CV score, mention that and offer to try another model or expand the grid (edit `session.py` accordingly).

## Optuna support

If `optuna` is installed, the generated session can be edited to use `optuna` instead of `RandomizedSearchCV`. Propose this when user asks for more aggressive tuning.
```

- [ ] **Step 3: Write eda SKILL.md**

Create `mltoolkit-plugin/skills/eda/SKILL.md`:
```markdown
---
name: eda
description: Generate exploratory figures for the current dataset (correlation heatmap, target distribution, feature distributions).
triggers:
  - eda
  - explore data
  - exploratory analysis
  - visualize data
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# EDA Playbook

## Workflow

1. **If no session exists**: run the generic EDA via `python {PLUGIN_ROOT}/skills/setup/references/setup_reference.py --data <DATA> --target <TARGET> --output-dir .mltoolkit`.
2. **If session exists**: run task-specific EDA — `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`.
3. **Read all figures** under `.mltoolkit/artifacts/` and present them to the user.
4. **Flag issues**: high missing rate, severe class imbalance (>4:1), highly skewed numeric targets, highly correlated features (>0.95).
```

- [ ] **Step 4: Commit**

```bash
git add mltoolkit-plugin/skills/compare mltoolkit-plugin/skills/tune mltoolkit-plugin/skills/eda
git commit -m "feat(mltoolkit): compare/tune/eda cross-cutting skills"
```

---

## Task 12: Package skill — Tier A (single file)

**Files:**
- Create: `mltoolkit-plugin/skills/package/SKILL.md`
- Create: `mltoolkit-plugin/skills/package/references/tier_a_transform.py`
- Create: `mltoolkit-plugin/tests/test_package.py`

- [ ] **Step 1: Write the failing test**

Create `mltoolkit-plugin/tests/test_package.py`:
```python
"""Package skill transformation tests."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tier_a_produces_flat_script(tmp_path):
    # Simulate a session.py
    session = tmp_path / "session.py"
    session.write_text("""import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--target")
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    print("running all stages")
if __name__ == "__main__":
    main()
""")
    transform = REPO_ROOT / "skills/package/references/tier_a_transform.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(transform), "--session", str(session),
         "--output-dir", str(out), "--name", "classification_pipeline"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    deliverable = out / "classification_pipeline.py"
    assert deliverable.exists()
    content = deliverable.read_text()
    assert "pycaret" not in content.lower()
    assert "import argparse" in content
```

- [ ] **Step 2: Run to verify failure**

Run: `cd mltoolkit-plugin && pytest tests/test_package.py::test_tier_a_produces_flat_script -v`
Expected: FAIL — file not found.

- [ ] **Step 3: Write tier_a_transform.py**

Create `mltoolkit-plugin/skills/package/references/tier_a_transform.py`:
```python
"""Tier A packaging: copy session.py into a clean named deliverable.

The session script is already end-to-end; Tier A strips its `.mltoolkit` output
default to a user-supplied directory and writes it to the chosen filename.
"""
import argparse
from pathlib import Path


def transform(session_path: Path, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = session_path.read_text()
    # Safety: if any accidental pycaret import slipped in, fail loudly.
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret references — refusing to package.")
    # Swap default output dir to a neutral name
    content = content.replace('default=".mltoolkit"', 'default="output"')
    deliverable = output_dir / f"{name}.py"
    deliverable.write_text(content)
    return deliverable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    p = transform(Path(args.session), Path(args.output_dir), args.name)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write package SKILL.md (Tier A portion)**

Create `mltoolkit-plugin/skills/package/SKILL.md`:
```markdown
---
name: package
description: Package the in-session ML work into a deliverable — Tier A (single file), Tier B (mini project), or Tier C (full scaffold).
triggers:
  - package
  - package this
  - create script
  - save project
  - finalize
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Package Playbook

## Prerequisite

An existing session at `.mltoolkit/session.py` (produced by `classify`/`regress`/`cluster`/`anomaly` skills).

## Workflow

1. **Ask user which tier**:
   - **A** — single self-contained `.py` script
   - **B** — script + `requirements.txt` + `README.md` (default, recommended)
   - **C** — full scaffold: `src/` (preprocess/train/predict), `tests/`, optional `api.py` + `Dockerfile`
2. **Ask output directory** (default: `./<task>_pipeline/`).
3. **Ask the name** of the pipeline (default: `<task>_pipeline`).
4. **Execute the tier transformation** (see below).
5. **Verify output** by running the emitted script against a small sample and confirming no pycaret imports leaked in.
6. **Show the user** the directory tree of what was produced.

## Tier A

Run: `python {SKILL_DIR}/references/tier_a_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME>`

## Tier B

(See Task 13 — adds README + requirements.txt generation)

## Tier C

(See Task 14 — splits into src/, adds tests + optional api/Dockerfile)

## Verification

After any tier:
1. Run `python <OUT>/<NAME>.py --data <USER_DATA> --target <TARGET> --output-dir <TEMP>`
2. Confirm it completes and produces the same artifacts the user saw during iteration.
3. Grep output for `pycaret` — must be empty.
```

- [ ] **Step 5: Run test**

Run: `cd mltoolkit-plugin && pytest tests/test_package.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/package/SKILL.md mltoolkit-plugin/skills/package/references/tier_a_transform.py mltoolkit-plugin/tests/test_package.py
git commit -m "feat(mltoolkit): package skill — Tier A (single file)"
```

---

## Task 13: Package skill — Tier B (mini project)

**Files:**
- Create: `mltoolkit-plugin/skills/package/references/tier_b_transform.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_b_readme_template.md`
- Modify: `mltoolkit-plugin/tests/test_package.py` (add Tier B test)
- Modify: `mltoolkit-plugin/skills/package/SKILL.md` (fill in Tier B section)

- [ ] **Step 1: Add failing test**

Append to `mltoolkit-plugin/tests/test_package.py`:
```python
def test_tier_b_produces_script_readme_requirements(tmp_path):
    session = tmp_path / "session.py"
    session.write_text("""import argparse
import pandas as pd
import sklearn.ensemble
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
if __name__ == "__main__":
    main()
""")
    transform = REPO_ROOT / "skills/package/references/tier_b_transform.py"
    out = tmp_path / "out"
    r = subprocess.run(
        ["python", str(transform), "--session", str(session),
         "--output-dir", str(out), "--name", "my_pipeline",
         "--task", "classification"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "my_pipeline.py").exists()
    assert (out / "requirements.txt").exists()
    assert (out / "README.md").exists()
    reqs = (out / "requirements.txt").read_text()
    assert "pandas" in reqs
    assert "scikit-learn" in reqs
    assert "pycaret" not in reqs.lower()
```

- [ ] **Step 2: Write readme template**

Create `mltoolkit-plugin/skills/package/references/tier_b_readme_template.md`:
```markdown
# {name}

A standalone {task} pipeline generated by mltoolkit.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python {name}.py --data path/to/your.csv --target {target} --output-dir output/ --stage all
```

### Stages

- `eda` — exploratory figures
- `compare` — cross-validated model comparison
- `tune` — hyperparameter tuning of top model
- `evaluate` — holdout evaluation + diagnostics
- `all` — runs everything

## Outputs

- `output/results/leaderboard.csv` — model comparison metrics
- `output/results/holdout_metrics.json` — final holdout performance
- `output/artifacts/*.png` — diagnostic figures
- `output/model.joblib` — trained pipeline (preprocessing + model)

## Load the trained model

```python
import joblib, pandas as pd
model = joblib.load("output/model.joblib")
new_data = pd.read_csv("new.csv")
predictions = model.predict(new_data)
```
```

- [ ] **Step 3: Write tier_b_transform.py**

Create `mltoolkit-plugin/skills/package/references/tier_b_transform.py`:
```python
"""Tier B packaging: single script + requirements.txt + README.md."""
import argparse
import ast
import re
import sys
from pathlib import Path

# Map Python import names to PyPI package names
_PYPI_MAP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
}


def detect_imports(source: str) -> set:
    """Parse the script's imports; return PyPI package names (top-level only)."""
    tree = ast.parse(source)
    top_level = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                top_level.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                top_level.add(node.module.split(".")[0])
    # Filter stdlib-ish common names
    stdlib = {"argparse", "json", "pathlib", "sys", "os", "subprocess", "re",
              "ast", "dataclasses", "typing", "collections", "itertools",
              "functools", "warnings", "datetime", "math", "random", "__future__"}
    external = {pkg for pkg in top_level if pkg not in stdlib}
    return {_PYPI_MAP.get(p, p) for p in external}


def transform(session_path: Path, output_dir: Path, name: str, task: str) -> dict:
    content = session_path.read_text()
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret — refusing to package.")
    output_dir.mkdir(parents=True, exist_ok=True)

    content = content.replace('default=".mltoolkit"', 'default="output"')

    # Detect the target arg default (if the session substituted a specific target)
    target_match = re.search(r'--target.*?default=["\'](\w+)["\']', content)
    target = target_match.group(1) if target_match else "target"

    script = output_dir / f"{name}.py"
    script.write_text(content)

    reqs = detect_imports(content)
    (output_dir / "requirements.txt").write_text("\n".join(sorted(reqs)) + "\n")

    template_path = Path(__file__).parent / "tier_b_readme_template.md"
    readme = template_path.read_text().format(name=name, task=task, target=target)
    (output_dir / "README.md").write_text(readme)

    return {"script": script,
            "requirements": output_dir / "requirements.txt",
            "readme": output_dir / "README.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True,
                    choices=["classification", "regression", "clustering", "anomaly"])
    args = ap.parse_args()
    out = transform(Path(args.session), Path(args.output_dir), args.name, args.task)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update package SKILL.md**

In `mltoolkit-plugin/skills/package/SKILL.md`, replace the Tier B placeholder section with:
```markdown
## Tier B (default)

Run: `python {SKILL_DIR}/references/tier_b_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME> --task <TASK>`

Where `<TASK>` is one of `classification`, `regression`, `clustering`, `anomaly`.

Produces:
- `<OUT>/<NAME>.py`
- `<OUT>/requirements.txt` (derived from imports actually used)
- `<OUT>/README.md`
```

- [ ] **Step 5: Run test**

Run: `cd mltoolkit-plugin && pytest tests/test_package.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/package/references/tier_b_transform.py mltoolkit-plugin/skills/package/references/tier_b_readme_template.md mltoolkit-plugin/tests/test_package.py mltoolkit-plugin/skills/package/SKILL.md
git commit -m "feat(mltoolkit): package skill — Tier B (script + requirements + README)"
```

---

## Task 14: Package skill — Tier C (full scaffold)

**Files:**
- Create: `mltoolkit-plugin/skills/package/references/tier_c_transform.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/preprocess.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/train.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/predict.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/tests/test_pipeline.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/api.py`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/Dockerfile`
- Create: `mltoolkit-plugin/skills/package/references/tier_c_scaffold/requirements.txt`
- Modify: `mltoolkit-plugin/tests/test_package.py`
- Modify: `mltoolkit-plugin/skills/package/SKILL.md`

- [ ] **Step 1: Add failing test**

Append to `mltoolkit-plugin/tests/test_package.py`:
```python
def test_tier_c_produces_scaffold(tmp_path):
    session = tmp_path / "session.py"
    session.write_text("""import argparse, pandas, sklearn.ensemble
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
if __name__ == "__main__":
    main()
""")
    transform = REPO_ROOT / "skills/package/references/tier_c_transform.py"
    out = tmp_path / "out"
    r = subprocess.run(
        ["python", str(transform), "--session", str(session),
         "--output-dir", str(out), "--name", "my_pipeline",
         "--task", "classification", "--with-api", "--with-docker"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    for rel in ["src/preprocess.py", "src/train.py", "src/predict.py",
                "tests/test_pipeline.py", "requirements.txt", "README.md",
                "api.py", "Dockerfile"]:
        assert (out / rel).exists(), f"missing {rel}"
```

- [ ] **Step 2: Write scaffold template files**

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/preprocess.py`:
```python
"""Preprocessing pipeline — extracted from training script."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]),
            numeric))
    if categorical:
        transformers.append(("cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            categorical))
    return ColumnTransformer(transformers, remainder="drop")
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/train.py`:
```python
"""Train a model and save the fitted pipeline."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor


def train(data_path: str, target: str, model, output_path: str):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target]); y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    print(f"Test score: {score:.4f}")
    joblib.dump(pipe, output_path)
    return pipe, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", default="model.joblib")
    args = ap.parse_args()

    # Swap in your chosen model here — default is RandomForest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    train(args.data, args.target, model, args.output)


if __name__ == "__main__":
    main()
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/src/predict.py`:
```python
"""Load a saved pipeline and produce predictions on new data."""
import argparse

import joblib
import pandas as pd


def predict(model_path: str, data_path: str, output_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    preds = model.predict(df)
    out = df.copy(); out["prediction"] = preds
    out.to_csv(output_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()
    predict(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/tests/test_pipeline.py`:
```python
"""Smoke test: the pipeline trains and predicts on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from train import train
from predict import predict


def test_end_to_end(tmp_path):
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    model_path = tmp_path / "model.joblib"
    train(str(data_path), "target", RandomForestClassifier(random_state=42), str(model_path))
    assert model_path.exists()

    new = df.drop(columns=["target"])
    new_path = tmp_path / "new.csv"
    new.to_csv(new_path, index=False)
    out_path = tmp_path / "preds.csv"
    preds = predict(str(model_path), str(new_path), str(out_path))
    assert "prediction" in preds.columns
    assert len(preds) == len(new)
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/api.py`:
```python
"""FastAPI endpoint serving a saved pipeline."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(__file__).parent / "model.joblib"
app = FastAPI(title="mltoolkit prediction API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    records: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    df = pd.DataFrame(req.records)
    preds = model.predict(df).tolist()
    return {"predictions": preds}
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `mltoolkit-plugin/skills/package/references/tier_c_scaffold/requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
fastapi
uvicorn
pydantic
```

- [ ] **Step 3: Write tier_c_transform.py**

Create `mltoolkit-plugin/skills/package/references/tier_c_transform.py`:
```python
"""Tier C packaging: full project scaffold with src/, tests/, optional api + Dockerfile."""
import argparse
import shutil
from pathlib import Path

from tier_b_transform import detect_imports


SCAFFOLD_DIR = Path(__file__).parent / "tier_c_scaffold"


def transform(session_path: Path, output_dir: Path, name: str, task: str,
              with_api: bool, with_docker: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    content = session_path.read_text()
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret — refusing to package.")

    # Copy scaffold files
    (output_dir / "src").mkdir(exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)

    shutil.copy2(SCAFFOLD_DIR / "src/preprocess.py", output_dir / "src/preprocess.py")
    shutil.copy2(SCAFFOLD_DIR / "src/train.py", output_dir / "src/train.py")
    shutil.copy2(SCAFFOLD_DIR / "src/predict.py", output_dir / "src/predict.py")
    shutil.copy2(SCAFFOLD_DIR / "tests/test_pipeline.py", output_dir / "tests/test_pipeline.py")

    # Combine template requirements with detected ones
    base_reqs = set((SCAFFOLD_DIR / "requirements.txt").read_text().splitlines())
    base_reqs.discard("")
    detected = detect_imports(content)
    if not with_api:
        base_reqs -= {"fastapi", "uvicorn", "pydantic"}
    all_reqs = sorted(base_reqs | detected)
    (output_dir / "requirements.txt").write_text("\n".join(all_reqs) + "\n")

    # README
    readme = f"""# {name}

A standalone {task} pipeline scaffolded by mltoolkit.

## Layout

- `src/preprocess.py` — ColumnTransformer builder
- `src/train.py` — training entry point (`python src/train.py --data X --target Y`)
- `src/predict.py` — inference entry point (`python src/predict.py --model M --data X`)
- `tests/test_pipeline.py` — smoke test (`pytest tests/`)
"""
    if with_api:
        readme += "- `api.py` — FastAPI server (`uvicorn api:app --reload`)\n"
    if with_docker:
        readme += "- `Dockerfile` — container build (`docker build -t {name} .`)\n".format(name=name)
    (output_dir / "README.md").write_text(readme)

    produced = {"src": output_dir / "src", "tests": output_dir / "tests",
                "requirements": output_dir / "requirements.txt", "readme": output_dir / "README.md"}
    if with_api:
        shutil.copy2(SCAFFOLD_DIR / "api.py", output_dir / "api.py")
        produced["api"] = output_dir / "api.py"
    if with_docker:
        shutil.copy2(SCAFFOLD_DIR / "Dockerfile", output_dir / "Dockerfile")
        produced["dockerfile"] = output_dir / "Dockerfile"
    return produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True,
                    choices=["classification", "regression", "clustering", "anomaly"])
    ap.add_argument("--with-api", action="store_true")
    ap.add_argument("--with-docker", action="store_true")
    args = ap.parse_args()
    produced = transform(Path(args.session), Path(args.output_dir), args.name,
                         args.task, args.with_api, args.with_docker)
    for k, v in produced.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update package SKILL.md with Tier C**

In `mltoolkit-plugin/skills/package/SKILL.md`, replace the Tier C placeholder section with:
```markdown
## Tier C (full scaffold)

Ask the user whether to include:
- `api.py` + FastAPI (y/n)
- `Dockerfile` (y/n)

Then run:
`python {SKILL_DIR}/references/tier_c_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME> --task <TASK> [--with-api] [--with-docker]`

Produces:
- `<OUT>/src/{preprocess,train,predict}.py`
- `<OUT>/tests/test_pipeline.py`
- `<OUT>/requirements.txt` (combined template + detected)
- `<OUT>/README.md`
- `<OUT>/api.py` (if `--with-api`)
- `<OUT>/Dockerfile` (if `--with-docker`)

The `src/train.py` is a skeleton — the user will need to swap in their chosen model from the `session.py`. Advise them of this.
```

- [ ] **Step 5: Run tests**

Run: `cd mltoolkit-plugin && pytest tests/test_package.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add mltoolkit-plugin/skills/package
git commit -m "feat(mltoolkit): package skill — Tier C (full scaffold with api/docker)"
```

---

## Task 15: ML pipeline agent

**Files:**
- Create: `mltoolkit-plugin/agents/ml-pipeline.md`

- [ ] **Step 1: Write the agent definition**

Create `mltoolkit-plugin/agents/ml-pipeline.md`:
```markdown
---
name: ml-pipeline
description: End-to-end ML pipeline orchestrator — uses mltoolkit skills to run setup → task-specific workflow → compare → tune → package.
model: sonnet
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
  - Skill
---

# mltoolkit ML Pipeline Agent

You orchestrate end-to-end ML workflows using mltoolkit skills. You do not write ML code directly — you delegate to skills.

## Workflow

1. **Clarify intent**: data path, task type (if ambiguous), target column, time budget.
2. **Invoke `mltoolkit:setup`** — load data, run EDA, identify task type.
3. **Invoke the task skill** (`classify`, `regress`, `cluster`, or `anomaly`) — runs compare, shows leaderboard, suggests next steps.
4. **Invoke `mltoolkit:tune`** on the user-selected model.
5. **Present holdout evaluation**. Offer to invoke `mltoolkit:package`.
6. **Invoke `mltoolkit:package`** — ask user for tier (A/B/C) and options.

## Principles

- **Defer to skills for how**. You decide what skill to use and in what order; the skill's SKILL.md decides how.
- **Never bypass a skill**. Don't write ML code directly in the session.
- **Always check `.mltoolkit/session.py` exists** before invoking compare/tune/package.
- **Summarize progress at each step** — let the user interrupt if they want to branch.
```

- [ ] **Step 2: Commit**

```bash
git add mltoolkit-plugin/agents/ml-pipeline.md
git commit -m "feat(mltoolkit): ml-pipeline agent definition"
```

---

## Task 16: End-to-end smoke test runner

**Files:**
- Create: `mltoolkit-plugin/tests/test_references.sh`

- [ ] **Step 1: Write the shell runner**

Create `mltoolkit-plugin/tests/test_references.sh`:
```bash
#!/usr/bin/env bash
# Run every reference script end-to-end against synthetic data.
# Exit non-zero on any failure. CI-friendly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PLUGIN_ROOT"

echo "=== mltoolkit reference smoke tests ==="
echo

echo "[1/4] Shared helpers..."
pytest tests/test_shared.py -q

echo
echo "[2/4] Classify reference..."
pytest tests/test_classify.py -q

echo
echo "[3/4] Regress reference..."
pytest tests/test_regress.py -q

echo
echo "[4/4] Cluster + anomaly + package..."
pytest tests/test_cluster.py tests/test_anomaly.py tests/test_package.py -q

echo
echo "=== All smoke tests passed ==="
```

Make executable: `chmod +x mltoolkit-plugin/tests/test_references.sh`

- [ ] **Step 2: Run the full suite**

Run: `bash mltoolkit-plugin/tests/test_references.sh`
Expected: all four groups PASS; final message "All smoke tests passed".

- [ ] **Step 3: Commit**

```bash
git add mltoolkit-plugin/tests/test_references.sh
git commit -m "test(mltoolkit): end-to-end reference smoke-test runner"
```

---

## Task 17: README for the plugin

**Files:**
- Create: `mltoolkit-plugin/README.md`

- [ ] **Step 1: Write plugin README**

Create `mltoolkit-plugin/README.md`:
```markdown
# mltoolkit — Claude Code ML Plugin

Standalone machine learning plugin for Claude Code. Generates native Python code (scikit-learn + optional XGBoost/LightGBM/CatBoost) for classification, regression, clustering, and anomaly detection. **No PyCaret dependency.**

## Install

1. Clone the repo.
2. Load the plugin: `claude --plugin-dir ./mltoolkit-plugin`
3. Verify: `bash mltoolkit-plugin/scripts/check-env.sh`

## Required

- Python ≥ 3.9
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

## Optional

- xgboost, lightgbm, catboost — adds more models to the zoo
- imblearn — SMOTE for class imbalance
- category_encoders — TargetEncoder for high-cardinality features
- optuna — alternative hyperparameter search backend
- plotly — interactive exploration plots

## Skills

| Skill | Purpose |
|-------|---------|
| `mltoolkit:setup` | Load data, run EDA, identify task type |
| `mltoolkit:classify` | Binary/multiclass classification |
| `mltoolkit:regress` | Continuous-value regression |
| `mltoolkit:cluster` | Unsupervised clustering |
| `mltoolkit:anomaly` | Outlier / anomaly detection |
| `mltoolkit:compare` | Re-run model comparison |
| `mltoolkit:tune` | Hyperparameter tuning |
| `mltoolkit:eda` | Regenerate EDA figures |
| `mltoolkit:package` | Package session into A/B/C tier deliverable |

## How it works

1. You ask Claude to classify/regress/cluster/etc. your data.
2. Claude copies a pre-tested reference script into `.mltoolkit/session.py` in your CWD.
3. Claude runs stages (`eda`, `compare`, `tune`, `evaluate`) and shows you results inline.
4. You iterate — try different models, tweak preprocessing, tune further.
5. When you're happy, ask Claude to "package this." Claude emits a single file, mini project, or full scaffold depending on your choice.

The packaged output is pure scikit-learn (+ optional boosters) — you can drop it anywhere.

## Tests

```bash
bash mltoolkit-plugin/tests/test_references.sh
```

## Architecture

See `docs/superpowers/specs/2026-04-14-mltoolkit-plugin-design.md` for the full design.
```

- [ ] **Step 2: Commit**

```bash
git add mltoolkit-plugin/README.md
git commit -m "docs(mltoolkit): plugin README"
```

---

## Final verification

- [ ] Run: `bash mltoolkit-plugin/scripts/check-env.sh`
  Expected: reports environment, exit 0.

- [ ] Run: `bash mltoolkit-plugin/tests/test_references.sh`
  Expected: all tests pass.

- [ ] Grep for pycaret leaks:
  ```bash
  grep -r -i pycaret mltoolkit-plugin/ --exclude-dir=.git && echo "LEAK" || echo "clean"
  ```
  Expected: prints "clean" (no pycaret references in the plugin).

---

## Self-Review

**Spec coverage check:**
- §2 Goals → Task 1 (scaffold), Tasks 2-14 (zero pycaret, inline execution, tiered delivery, graceful fallback) ✅
- §3 Non-goals → explicitly excluded; no tasks touch them ✅
- §4 User decisions → captured in tasks (ML-engineer flow, inline+package, classify/regress/cluster/anomaly only, matplotlib+seaborn, sklearn+boosters optional, new dir, tiered output, approach 1) ✅
- §5 Architecture → Tasks 1-17 implement the directory layout ✅
- §6 Workflow → skill playbooks in Tasks 5, 7, 8, 9, 10, 11 ✅
- §7 Reference scripts → Tasks 3, 4 (classify), 6 (regress), 8 (cluster), 9 (anomaly) ✅
- §7.6 Shared helpers → Tasks 2, 2a, 2b ✅
- §8 Skill spec → all SKILL.md files produced in appropriate tasks ✅
- §9 Session scratchpad protocol → baked into each skill's playbook ✅
- §10 Error handling → inline in the reference scripts (Tasks 4, 6, 8, 9) ✅
- §11 Testing strategy → Tasks 1 (fixtures), 4/6/8/9 (smoke tests), 16 (runner) ✅
- §12 Dependencies → Task 1 check-env.sh + Task 17 README ✅
- §13 Migration → new directory `mltoolkit-plugin/` (not overwriting) ✅
- §15 Success criteria → Final verification section ✅

**Placeholder scan:** No "TBD", "TODO", or vague instructions. All code blocks contain complete code. ✅

**Type/name consistency:**
- Model IDs in classify zoo (lr/ridge/knn/dt/rf/et/gbc/ada/svc/nb/mlp + xgb/lgbm/cat) consistent across model_zoo.py, tests, and SKILL.md ✅
- Regress IDs (lr/ridge/lasso/en/knn/dt/rf/et/gbr/ada/svr/mlp + boosters) consistent ✅
- Cluster IDs (kmeans/dbscan/agglom/gmm) consistent ✅
- Anomaly IDs (iforest/lof/elliptic/ocsvm) consistent ✅
- Stage names (eda/compare/tune/evaluate/all) consistent across references and SKILL.md ✅
- File paths: `{SKILL_DIR}/references/<name>_reference.py` consistent ✅
- Output directories: `.mltoolkit/artifacts/`, `.mltoolkit/results/` consistent ✅
