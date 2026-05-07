import sys

from pycaret.utils._show_versions import show_versions

version_ = "1.0.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# pycaret-ng targets Python 3.10+. The dep floors in pyproject.toml
# implicitly gate the upper end. The 3.10 lower bound is forced by
# imbalanced-learn>=0.14 (Phase 1's sklearn 1.6 floor required this
# bump; older imbalanced-learn requires sklearn <1.5).
if sys.version_info < (3, 10):
    raise RuntimeError(
        "pycaret-ng requires Python >= 3.10. Your actual Python version: ",
        sys.version_info,
        "Please upgrade your Python.",
    )
