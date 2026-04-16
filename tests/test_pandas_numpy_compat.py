"""Unit tests for pycaret-ng Phase 2 pandas/numpy compatibility migrations."""
from __future__ import annotations

import numpy as np
import pytest


def test_dependencies_module_imports_on_python312plus():
    """pycaret/utils/_dependencies.py must not import distutils."""
    import pycaret.utils._dependencies as deps
    import inspect
    src = inspect.getsource(deps)
    assert "distutils" not in src, (
        "pycaret/utils/_dependencies.py still references distutils; "
        "Python 3.12 removed it. Use packaging.version.Version."
    )
    assert "from packaging.version import Version" in src, (
        "Expected 'from packaging.version import Version' in pycaret/utils/_dependencies.py"
    )


def test_get_module_version_returns_packaging_version():
    """get_module_version must return packaging.version.Version instances."""
    from packaging.version import Version
    from pycaret.utils._dependencies import get_module_version
    result = get_module_version("numpy")
    assert isinstance(result, Version), (
        f"Expected packaging.version.Version, got {type(result).__name__}"
    )


def test_get_installed_modules_values_are_packaging_versions():
    """get_installed_modules must return {name: Version} mapping."""
    from packaging.version import Version
    from pycaret.utils._dependencies import get_installed_modules
    modules = get_installed_modules()
    assert isinstance(modules, dict)
    assert "numpy" in modules
    for name, ver in modules.items():
        if ver is None:
            continue
        assert isinstance(ver, Version), (
            f"Module {name}: expected Version, got {type(ver).__name__}"
        )
