"""Stager + end-to-end copy-and-run tests for every reference script."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGER = REPO_ROOT / "scripts/stage_session.py"

try:
    import optuna  # noqa: F401
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


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
    assert "references._shared" not in text
    assert "_PLUGIN_ROOT" not in text
    assert "from _shared import" in text


def test_stager_refuses_unknown_task(tmp_path):
    r = _run_stager("not_a_task", tmp_path / "mlt")
    assert r.returncode != 0
    msg = (r.stderr + r.stdout).lower()
    assert "unknown task" in msg or "invalid choice" in msg


@pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")
def test_optuna_tune_runs_end_to_end(classification_data, tmp_path):
    """Stage classify, run tune stage with --search-library optuna, verify best_params written."""
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("classify", dest).check_returncode()
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


def test_optuna_fallback_warns_when_missing(classification_data, tmp_path):
    """When optuna is not importable, requesting it warns and falls back to sklearn."""
    dest = tmp_path / "mlt"
    out = tmp_path / "out"
    _run_stager("classify", dest).check_returncode()

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "optuna.py").write_text("raise ImportError('forced for test')\n")
    env = {**os.environ, "PYTHONPATH": f"{stub_dir}{os.pathsep}{dest}"}

    # Seed best_model_id via a --stage all compare first? Simpler: use a model the zoo has.
    r = subprocess.run(
        [sys.executable, str(dest / "session.py"),
         "--data", classification_data["path"],
         "--target", classification_data["target"],
         "--output-dir", str(out),
         "--stage", "all",
         "--search-library", "optuna",
         "--n-iter", "3"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "falling back" in (r.stdout + r.stderr).lower()
