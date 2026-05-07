"""Unit tests for pycaret-ng Phase 3 plotting compatibility patches."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_stem_accepts_use_line_collection_without_error():
    """After the patch, ax.stem() must silently accept use_line_collection.

    yellowbrick 1.5 passes this kwarg; matplotlib 3.8+ removed it.
    The patch wraps Axes.stem to drop it silently.
    """
    import pycaret.internal.patches.matplotlib_compat  # applies the patch

    fig, ax = plt.subplots()
    ax.stem([1, 2, 3], [4, 5, 6], use_line_collection=True)
    plt.close(fig)


def test_stem_still_works_without_use_line_collection():
    """Normal stem() calls must not be affected by the patch."""
    import pycaret.internal.patches.matplotlib_compat  # applies the patch

    fig, ax = plt.subplots()
    ax.stem([1, 2, 3], [4, 5, 6])
    plt.close(fig)


def test_schemdraw_imports_on_modern_matplotlib():
    """schemdraw >=0.16 must import cleanly on matplotlib >=3.8."""
    from schemdraw import Drawing
    from schemdraw.flow import Arrow, Data, RoundBox, Subroutine

    assert Drawing is not None
