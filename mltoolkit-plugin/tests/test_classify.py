"""Smoke tests for classify reference components."""
import subprocess
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
    assert X.shape[1] >= df.shape[1] - 1  # OHE expands (cat_a with 3 vals -> 3 cols)


def test_model_zoo_has_core_entries():
    zoo = model_zoo.get_zoo()
    for mid in ["lr", "ridge", "knn", "dt", "rf", "et", "gbc", "ada", "svc", "nb", "mlp"]:
        assert mid in zoo, f"Missing core model: {mid}"
        entry = zoo[mid]
        assert "estimator" in entry
        assert "param_grid" in entry


def test_model_zoo_optional_models_respect_deps():
    zoo = model_zoo.get_zoo()
    from references._shared import deps
    if deps.has_xgboost():
        assert "xgb" in zoo
    else:
        assert "xgb" not in zoo


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
    figures = list((out_dir / "artifacts").glob("*.png"))
    assert len(figures) >= 1
    assert (out_dir / "model.joblib").exists()
