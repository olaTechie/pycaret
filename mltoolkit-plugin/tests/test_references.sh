#!/usr/bin/env bash
# Run every reference script end-to-end against synthetic data.
# Exit non-zero on any failure. CI-friendly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PLUGIN_ROOT"

echo "=== mltoolkit reference smoke tests ==="
echo

echo "[1/5] Shared helpers..."
python -m pytest tests/test_shared.py -q

echo
echo "[2/5] Classify reference..."
python -m pytest tests/test_classify.py -q

echo
echo "[3/5] Regress reference..."
python -m pytest tests/test_regress.py -q

echo
echo "[4/5] Cluster + anomaly + package..."
python -m pytest tests/test_cluster.py tests/test_anomaly.py tests/test_package.py -q

echo
echo "[5/5] Stage-and-run (end-to-end copy path)..."
python -m pytest tests/test_stage_and_run.py -q

echo
echo "=== All smoke tests passed ==="
