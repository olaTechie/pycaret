"""Shared fixtures and backend setup for the pycaret-ng smoke harnesses.

This conftest forces a non-interactive matplotlib backend BEFORE any
pycaret import in the test modules, so headless CI/local runs don't
try to open a display. It also picks the pure-Python protobuf
implementation so packages shipping older _pb2.py files (e.g.,
mlflow's tracking client transitives) don't error on import under
protobuf >= 4.

Used by tests/smoke/test_plotting.py (Phase 3) and
tests/smoke/test_time_series.py (Phase 4).
"""
import os

# Pure-Python protobuf — must run before any package that imports
# generated _pb2.py modules built against older protoc. Phase 4
# surfaced this with mlflow / google.protobuf transitives.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Headless matplotlib — must run before pycaret pulls matplotlib in.
import matplotlib

matplotlib.use("Agg")

# Suppress plotly's browser-open on `fig.show()`.
os.environ.setdefault("PLOTLY_RENDERER", "json")

import pytest
