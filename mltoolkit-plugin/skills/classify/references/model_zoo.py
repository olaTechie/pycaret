"""Classification model zoo — id -> {estimator, param_grid, requires}."""
import sys
from pathlib import Path

from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


def get_zoo() -> dict:
    zoo = {
        "lr": {
            "estimator": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
            "param_grid": {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "requires": None,
        },
        "ridge": {
            "estimator": RidgeClassifier(random_state=42),
            "param_grid": {"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            "requires": None,
        },
        "knn": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
            "requires": None,
        },
        "dt": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
            "requires": None,
        },
        "rf": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
            "requires": None,
        },
        "et": {
            "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
            "param_grid": {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20]},
            "requires": None,
        },
        "gbc": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            "requires": None,
        },
        "ada": {
            "estimator": AdaBoostClassifier(random_state=42),
            "param_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 1.0, 1.5]},
            "requires": None,
        },
        "svc": {
            "estimator": SVC(probability=True, random_state=42),
            "param_grid": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
            "requires": None,
        },
        "nb": {
            "estimator": GaussianNB(),
            "param_grid": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
            "requires": None,
        },
        "mlp": {
            "estimator": MLPClassifier(max_iter=500, random_state=42),
            "param_grid": {"hidden_layer_sizes": [(50,), (100,), (100, 50)],
                           "alpha": [0.0001, 0.001]},
            "requires": None,
        },
    }
    if deps.has_xgboost():
        import xgboost as xgb
        zoo["xgb"] = {
            "estimator": xgb.XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
            "param_grid": {"n_estimators": [100, 300], "max_depth": [3, 6, 9],
                           "learning_rate": [0.05, 0.1]},
            "requires": "xgboost",
        }
    if deps.has_lightgbm():
        import lightgbm as lgb
        zoo["lgbm"] = {
            "estimator": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            "param_grid": {"n_estimators": [100, 300], "num_leaves": [31, 63],
                           "learning_rate": [0.05, 0.1]},
            "requires": "lightgbm",
        }
    if deps.has_catboost():
        import catboost as cb
        zoo["cat"] = {
            "estimator": cb.CatBoostClassifier(random_state=42, verbose=False),
            "param_grid": {"iterations": [100, 300], "depth": [4, 6, 8],
                           "learning_rate": [0.05, 0.1]},
            "requires": "catboost",
        }
    return zoo
