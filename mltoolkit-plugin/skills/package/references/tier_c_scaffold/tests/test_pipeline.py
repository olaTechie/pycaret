"""Smoke test: the pipeline trains and predicts on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from train import train
from predict import predict


def test_end_to_end(tmp_path):
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    model_path = tmp_path / "model.joblib"
    train(str(data_path), "target", RandomForestClassifier(random_state=42), str(model_path))
    assert model_path.exists()

    new = df.drop(columns=["target"])
    new_path = tmp_path / "new.csv"
    new.to_csv(new_path, index=False)
    out_path = tmp_path / "preds.csv"
    preds = predict(str(model_path), str(new_path), str(out_path))
    assert "prediction" in preds.columns
    assert len(preds) == len(new)
