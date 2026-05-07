"""Shared fixtures and backend setup for the Phase 3 plotting smoke harness.

This conftest forces a non-interactive matplotlib backend BEFORE any
pycaret import in the test module, so headless CI/local runs don't try
to open a display.
"""
import os

# Headless matplotlib — must run before pycaret pulls matplotlib in.
import matplotlib

matplotlib.use("Agg")

# Suppress plotly's browser-open on `fig.show()`.
os.environ.setdefault("PLOTLY_RENDERER", "json")

import pytest
