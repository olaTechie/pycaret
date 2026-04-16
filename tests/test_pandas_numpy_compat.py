"""Unit tests for pycaret-ng Phase 2 pandas/numpy compatibility migrations."""
from __future__ import annotations

import numpy as np
import pytest


def test_numpy_prod_available_in_sklearn_patch():
    """pycaret/internal/patches/sklearn.py must use np.prod (numpy 2.0-safe)."""
    import pycaret.internal.patches.sklearn as sk_patch
    import inspect
    src = inspect.getsource(sk_patch)
    assert "np.product" not in src, (
        "pycaret/internal/patches/sklearn.py still contains np.product; "
        "numpy 2.0 removed it. Replace with np.prod."
    )
    assert "np.prod(" in src, (
        "Expected np.prod(...) call in pycaret/internal/patches/sklearn.py"
    )


def test_numpy_prod_matches_legacy_product():
    """np.prod must return the same value np.product returned on numpy<2.0."""
    sizes = [2, 3, 4]
    assert np.prod(sizes, dtype=np.uint64) == 24


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
