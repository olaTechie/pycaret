"""FastAPI endpoint serving a saved pipeline."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(__file__).parent / "model.joblib"
app = FastAPI(title="mltoolkit prediction API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    records: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    df = pd.DataFrame(req.records)
    preds = model.predict(df).tolist()
    return {"predictions": preds}
