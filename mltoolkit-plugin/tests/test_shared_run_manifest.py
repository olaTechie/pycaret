"""run_manifest.py — versions, args, append semantics."""
import json

from references._shared.run_manifest import build_manifest, write_manifest


def test_build_manifest_has_required_fields():
    m = build_manifest(stage="compare",
                       args_dict={"cv": 5, "group_col": None, "bootstrap": 0})
    assert m["stage"] == "compare"
    assert "timestamp_utc" in m
    assert "python_version" in m
    assert "platform" in m
    assert "packages" in m
    assert "scikit-learn" in m["packages"]
    assert "args" in m and m["args"]["cv"] == 5


def test_build_manifest_extra_is_carried_through():
    m = build_manifest(stage="evaluate", args_dict={}, extra={"best": "lr"})
    assert m["extra"]["best"] == "lr"


def test_write_manifest_appends(tmp_path):
    out = tmp_path / "res"
    write_manifest(out, build_manifest(stage="eda", args_dict={}))
    write_manifest(out, build_manifest(stage="compare", args_dict={}))
    write_manifest(out, build_manifest(stage="tune", args_dict={}))
    data = json.loads((out / "run_manifest.json").read_text())
    assert isinstance(data, list) and len(data) == 3
    assert [m["stage"] for m in data] == ["eda", "compare", "tune"]
