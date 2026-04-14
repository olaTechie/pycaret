#!/usr/bin/env bash
set -euo pipefail

echo "=== mltoolkit environment check ==="
echo "Python: $(python3 --version 2>&1)"
python3 - <<'PY'
import sys, importlib
required = ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "joblib"]
optional = ["xgboost", "lightgbm", "catboost", "imblearn", "category_encoders", "optuna", "plotly"]
missing_req = []
print("\nRequired:")
for m in required:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {m}: NOT INSTALLED")
        missing_req.append(m)
print("\nOptional:")
for m in optional:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {m}: not installed")
if missing_req:
    sys.exit(f"\nMissing required: {missing_req}")
print("\nAll required packages present.")
PY
