"""Regression model zoo."""
import sys
from pathlib import Path

from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


def get_zoo() -> dict:
    zoo = {
        "lr":    {"estimator": LinearRegression(n_jobs=-1), "param_grid": {}, "requires": None},
        "ridge": {"estimator": Ridge(random_state=42),
                  "param_grid": {"alpha": [0.1, 1.0, 10.0]}, "requires": None},
        "lasso": {"estimator": Lasso(random_state=42, max_iter=5000),
                  "param_grid": {"alpha": [0.01, 0.1, 1.0]}, "requires": None},
        "en":    {"estimator": ElasticNet(random_state=42, max_iter=5000),
                  "param_grid": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
                  "requires": None},
        "knn":   {"estimator": KNeighborsRegressor(n_jobs=-1),
                  "param_grid": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
                  "requires": None},
        "dt":    {"estimator": DecisionTreeRegressor(random_state=42),
                  "param_grid": {"max_depth": [None, 5, 10, 20]}, "requires": None},
        "rf":    {"estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
                  "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
                  "requires": None},
        "et":    {"estimator": ExtraTreesRegressor(random_state=42, n_jobs=-1),
                  "param_grid": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
                  "requires": None},
        "gbr":   {"estimator": GradientBoostingRegressor(random_state=42),
                  "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                                 "max_depth": [3, 5]}, "requires": None},
        "ada":   {"estimator": AdaBoostRegressor(random_state=42),
                  "param_grid": {"n_estimators": [50, 100]}, "requires": None},
        "svr":   {"estimator": SVR(),
                  "param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
                  "requires": None},
        "mlp":   {"estimator": MLPRegressor(max_iter=500, random_state=42),
                  "param_grid": {"hidden_layer_sizes": [(50,), (100,), (100, 50)]},
                  "requires": None},
    }
    if deps.has_xgboost():
        import xgboost as xgb
        zoo["xgb"] = {"estimator": xgb.XGBRegressor(random_state=42, n_jobs=-1),
                      "param_grid": {"n_estimators": [100, 300], "max_depth": [3, 6, 9],
                                     "learning_rate": [0.05, 0.1]}, "requires": "xgboost"}
    if deps.has_lightgbm():
        import lightgbm as lgb
        zoo["lgbm"] = {"estimator": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                       "param_grid": {"n_estimators": [100, 300], "num_leaves": [31, 63]},
                       "requires": "lightgbm"}
    if deps.has_catboost():
        import catboost as cb
        zoo["cat"] = {"estimator": cb.CatBoostRegressor(random_state=42, verbose=False),
                      "param_grid": {"iterations": [100, 300], "depth": [4, 6, 8]},
                      "requires": "catboost"}
    return zoo
