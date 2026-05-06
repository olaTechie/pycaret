import sys

from pycaret.utils._show_versions import show_versions

version_ = "1.0.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# pycaret-ng targets Python 3.9+. The dep floors in pyproject.toml
# implicitly gate the upper end (some deps will refuse to install on
# too-new Python). We hard-error only on the lower bound so that an
# accidental Python 3.8 invocation gives a clear diagnostic.
if sys.version_info < (3, 9):
    raise RuntimeError(
        "pycaret-ng requires Python >= 3.9. Your actual Python version: ",
        sys.version_info,
        "Please upgrade your Python.",
    )
