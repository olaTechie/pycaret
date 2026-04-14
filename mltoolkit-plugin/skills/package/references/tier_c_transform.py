"""Tier C packaging: full project scaffold with src/, tests/, optional api + Dockerfile."""
import argparse
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_b_transform import detect_imports  # noqa: E402


def _pin(pkg: str) -> str:
    """Return `pkg==X.Y.Z` if installed, else bare `pkg` (LEAD-034).

    Passes through lines that already have a version specifier (e.g. base_reqs
    entries like `fastapi>=0.100`).
    """
    if any(c in pkg for c in "=<>!"):
        return pkg
    try:
        return f"{pkg}=={_pkg_version(pkg)}"
    except PackageNotFoundError:
        return pkg


SCAFFOLD_DIR = Path(__file__).parent / "tier_c_scaffold"


def transform(session_path: Path, output_dir: Path, name: str, task: str,
              with_api: bool, with_docker: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    content = session_path.read_text()
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret — refusing to package.")

    (output_dir / "src").mkdir(exist_ok=True)
    (output_dir / "tests").mkdir(exist_ok=True)

    shutil.copy2(SCAFFOLD_DIR / "src/preprocess.py", output_dir / "src/preprocess.py")
    shutil.copy2(SCAFFOLD_DIR / "src/train.py", output_dir / "src/train.py")
    shutil.copy2(SCAFFOLD_DIR / "src/predict.py", output_dir / "src/predict.py")
    shutil.copy2(SCAFFOLD_DIR / "tests/test_pipeline.py", output_dir / "tests/test_pipeline.py")

    base_reqs = set(r for r in (SCAFFOLD_DIR / "requirements.txt").read_text().splitlines() if r.strip())
    detected = detect_imports(content)
    if not with_api:
        base_reqs -= {"fastapi", "uvicorn", "pydantic"}
    all_reqs = sorted(base_reqs | detected)
    pinned = sorted({_pin(r) for r in all_reqs})
    (output_dir / "requirements.txt").write_text("\n".join(pinned) + "\n")

    readme_lines = [
        f"# {name}",
        "",
        f"A standalone {task} pipeline scaffolded by mltoolkit.",
        "",
        "## Layout",
        "",
        "- `src/preprocess.py` — ColumnTransformer builder",
        "- `src/train.py` — training entry point (`python src/train.py --data X --target Y`)",
        "- `src/predict.py` — inference entry point (`python src/predict.py --model M --data X`)",
        "- `tests/test_pipeline.py` — smoke test (`pytest tests/`)",
    ]
    if with_api:
        readme_lines.append("- `api.py` — FastAPI server (`uvicorn api:app --reload`)")
    if with_docker:
        readme_lines.append(f"- `Dockerfile` — container build (`docker build -t {name} .`)")
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

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
