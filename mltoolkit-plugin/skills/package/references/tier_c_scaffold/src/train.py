"""Train a model and save the fitted pipeline."""
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor


def train(data_path: str, target: str, model, output_path: str):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target]); y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    print(f"Test score: {score:.4f}")
    joblib.dump(pipe, output_path)
    return pipe, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", default="model.joblib")
    args = ap.parse_args()

    # Swap in your chosen model here — default is RandomForest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    train(args.data, args.target, model, args.output)


if __name__ == "__main__":
    main()
