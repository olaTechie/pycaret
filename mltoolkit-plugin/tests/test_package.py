"""Package skill transformation tests."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


_SAMPLE_SESSION = """import argparse
import pandas as pd
import sklearn.ensemble
import scipy.stats
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--target", default="target")
    ap.add_argument("--output-dir", default=".mltoolkit")
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()
    print("running stages")
if __name__ == "__main__":
    main()
"""


def test_tier_a_produces_flat_script(tmp_path):
    session = tmp_path / "session.py"
    session.write_text(_SAMPLE_SESSION)
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
    # output-dir default was swapped from .mltoolkit to output
    assert 'default="output"' in content


def test_tier_b_produces_script_readme_requirements(tmp_path):
    session = tmp_path / "session.py"
    session.write_text(_SAMPLE_SESSION)
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


def test_tier_b_requirements_are_pinned(tmp_path):
    """LEAD-034: requirements.txt entries must include version pins for installed pkgs."""
    session = tmp_path / "session.py"
    session.write_text(_SAMPLE_SESSION)
    transform = REPO_ROOT / "skills/package/references/tier_b_transform.py"
    out = tmp_path / "out"
    r = subprocess.run(
        ["python", str(transform), "--session", str(session),
         "--output-dir", str(out), "--name", "p", "--task", "classification"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    reqs = (out / "requirements.txt").read_text()
    # At least one pinned entry (scikit-learn or pandas with ==).
    assert any("==" in line for line in reqs.splitlines()), reqs


def test_tier_c_substitutes_chosen_estimator(tmp_path):
    """LEAD-022: tier_c patches src/train.py with --best-model-id."""
    session = tmp_path / "session.py"
    session.write_text(_SAMPLE_SESSION)
    best_params = tmp_path / "best_params.json"
    best_params.write_text('{"clf__C": "1.0", "clf__max_iter": "500"}')
    transform = REPO_ROOT / "skills/package/references/tier_c_transform.py"
    out = tmp_path / "out"
    r = subprocess.run(
        ["python", str(transform), "--session", str(session),
         "--output-dir", str(out), "--name", "p", "--task", "classification",
         "--best-model-id", "lr", "--best-params", str(best_params)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    train_py = (out / "src/train.py").read_text()
    assert "from sklearn.linear_model import LogisticRegression" in train_py
    assert "LogisticRegression(" in train_py
    assert "RandomForestClassifier" not in train_py
    # Params flowed through.
    assert "C=1.0" in train_py


def test_tier_c_produces_scaffold(tmp_path):
    session = tmp_path / "session.py"
    session.write_text(_SAMPLE_SESSION)
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
