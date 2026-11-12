"""
models.py  --  THE FOUR ORDINARY MACHINE-LEARNING MODELS
========================================================

These are the same four model families HSBC used (their Section 3.3).  All
four are "classical": they run on a normal computer.  Each one receives a row
of inputs (one of the feature sets A-F) and returns p(fill), a number between
0 and 1.

    LR   logistic regression   - draws one straight dividing line.  The only
                                 "linear" model; it cannot combine features
                                 unless the inputs already contain combinations.
    RF   random forest         - averages many decision trees.
    XGB  gradient boosting     - builds trees one after another, each fixing
                                 the mistakes of the previous ones (XGBoost).
    NN   neural network        - a small feed-forward network.

"Hyper-parameters" are a model's settings (how deep the trees are, how strong
the regularisation is ...).  Like HSBC we choose them by cross-validation
inside the training data only: split the training events into 3 random parts,
train on 2, score on the third, and keep the setting with the best score.
Our grids are smaller than HSBC's so the experiment runs in about an hour on
three cores of a laptop; the differences are listed in PREREGISTRATION.md.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C

MODEL_NAMES = ("LR", "RF", "XGB", "NN")


def make_model(name: str):
    """Return (estimator, grid of settings to try)."""
    seed = C.MODEL_SEED
    if name == "LR":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", LogisticRegression(C=1.0, max_iter=3000))])
        grid = {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    elif name == "RF":
        est = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                     n_jobs=1, random_state=seed)
        grid = {"max_depth": [4, 8]}
    elif name == "XGB":
        est = XGBClassifier(n_estimators=100, tree_method="hist", n_jobs=1,
                            random_state=seed, verbosity=0)
        grid = {"max_depth": [3, 5], "learning_rate": [0.1, 0.01]}
    elif name == "NN":
        est = Pipeline([("scale", StandardScaler()),
                        ("clf", MLPClassifier(max_iter=300, early_stopping=True,
                                              n_iter_no_change=10, random_state=seed))])
        grid = {"clf__hidden_layer_sizes": [(3,), (100,), (164, 65)],
                "clf__activation": ["relu", "tanh"]}
    else:
        raise ValueError(f"unknown model {name}")
    return est, grid
