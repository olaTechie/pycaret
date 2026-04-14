"""Unit tests for references/_shared helpers."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from references._shared import deps, plotting, reporting


# ---------- deps ----------

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
    assert deps.has_sklearn() is True


# ---------- plotting ----------

def test_set_style_is_idempotent():
    plotting.set_style()
    plotting.set_style()


def test_save_fig_writes_png_and_pdf(tmp_path):
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


# ---------- reporting ----------

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
