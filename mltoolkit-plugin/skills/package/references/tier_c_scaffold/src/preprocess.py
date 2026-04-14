"""Preprocessing pipeline — extracted from training script."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]),
            numeric))
    if categorical:
        transformers.append(("cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            categorical))
    return ColumnTransformer(transformers, remainder="drop")
