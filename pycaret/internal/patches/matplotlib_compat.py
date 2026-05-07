"""Matplotlib compatibility patches for pycaret-ng.

yellowbrick 1.5 (latest release, effectively unmaintained) calls
matplotlib's Axes.stem() with use_line_collection=True, which matplotlib
3.8 removed. This module monkey-patches Axes.stem to silently drop the
kwarg, keeping yellowbrick's CooksDistance and other stem-based plots
functional on modern matplotlib.

Loaded early via pycaret/internal/patches/__init__.py.
"""
from __future__ import annotations

import matplotlib.axes

_original_stem = matplotlib.axes.Axes.stem


def _patched_stem(self, *args, **kwargs):
    kwargs.pop("use_line_collection", None)
    return _original_stem(self, *args, **kwargs)


matplotlib.axes.Axes.stem = _patched_stem
