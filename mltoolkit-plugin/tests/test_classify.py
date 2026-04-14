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
    for mid in ["lr", "ridge", "knn", "dt", "rf", "et", "gbc", "ada", "svc", "nb",
                "mlp", "qda", "lda", "dummy", "par", "bnb"]:
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


# ----- Plan 3: paper-mode feature tests ------------------------------------

def test_preprocessor_refuses_sensitive_target_encode():
    import pandas as pd
    import pytest
    df = pd.DataFrame({
        "num_a": range(100),
        "race": [f"group_{i%15}" for i in range(100)],
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


def test_eda_emits_epv_and_table1(classification_data, tmp_path):
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


def test_compare_stage_uses_group_col(classification_data, tmp_path):
    import pandas as pd
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
    assert (out / "results/leaderboard_folds.csv").exists()


def test_bootstrap_and_threshold_emitted(classification_data, tmp_path):
    import json
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5",
         "--bootstrap", "100",
         "--optimize-threshold", "youden"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/holdout_metrics_ci.json").exists()
    th = json.loads((out / "results/threshold.json").read_text())
    assert 0 < th["threshold"] < 1
    assert th["criterion"] == "youden"


def test_subgroup_metrics_and_decision_curve(classification_data, tmp_path):
    import pandas as pd
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


def test_calibration_emitted_for_binary(classification_data, tmp_path):
    script = REPO_ROOT / "skills/classify/references/classify_reference.py"
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(script),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out), "--stage", "all",
         "--cv", "3", "--n-iter", "5"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "results/calibration.json").exists()
    assert list((out / "artifacts").glob("reliability*"))
    assert list((out / "artifacts").glob("learning_curve*"))
    assert list((out / "artifacts").glob("classification_report_heatmap*"))
