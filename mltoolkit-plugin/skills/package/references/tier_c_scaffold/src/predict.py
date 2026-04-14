"""Load a saved pipeline and produce predictions on new data."""
import argparse

import joblib
import pandas as pd


def predict(model_path: str, data_path: str, output_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    preds = model.predict(df)
    out = df.copy(); out["prediction"] = preds
    out.to_csv(output_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args()
    predict(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
