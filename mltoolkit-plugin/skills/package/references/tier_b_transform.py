"""Tier B packaging: single script + requirements.txt + README.md."""
import argparse
import ast
import re
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _pin(pkg: str) -> str:
    """Return `pkg==X.Y.Z` if installed, else bare `pkg` (LEAD-034)."""
    try:
        return f"{pkg}=={_pkg_version(pkg)}"
    except PackageNotFoundError:
        return pkg

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
    stdlib = {"argparse", "json", "pathlib", "sys", "os", "subprocess", "re",
              "ast", "dataclasses", "typing", "collections", "itertools",
              "functools", "warnings", "datetime", "math", "random", "__future__",
              "scipy"}  # scipy is a direct import in some references
    external = {pkg for pkg in top_level if pkg not in stdlib}
    # Skip internal references package — it's not a PyPI dep
    external.discard("references")
    external.discard("preprocessing")
    external.discard("model_zoo")
    return {_PYPI_MAP.get(p, p) for p in external}


def transform(session_path: Path, output_dir: Path, name: str, task: str) -> dict:
    content = session_path.read_text()
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret — refusing to package.")
    output_dir.mkdir(parents=True, exist_ok=True)

    content = content.replace('default=".mltoolkit"', 'default="output"')

    target_match = re.search(r'--target.*?default=["\'](\w+)["\']', content)
    target = target_match.group(1) if target_match else "target"

    script = output_dir / f"{name}.py"
    script.write_text(content)

    reqs = detect_imports(content)
    if "scipy" in content:
        reqs.add("scipy")
    pinned = sorted(_pin(r) for r in reqs)
    (output_dir / "requirements.txt").write_text("\n".join(pinned) + "\n")

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
