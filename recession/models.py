"""Model pipelines. Scaling and L1 feature selection live *inside* each
pipeline, so they are re-fit on the training data of every fold and refit."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 8
INNER_SPLITS = 3


def _l1_selector(c: float = 0.1) -> SelectFromModel:
    return SelectFromModel(
        LogisticRegression(l1_ratio=1, solver="liblinear", C=c, max_iter=10_000, random_state=SEED)
    )


def _search(pipeline: Pipeline, grid: dict) -> GridSearchCV:
    """Hyperparameters chosen by time-ordered CV inside the training window."""
    return GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=INNER_SPLITS),
        error_score=np.nan,
    )


def make_model(name: str):
    if name == "yield_curve_logit":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=10_000))])
    if name == "l1_logit":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", _l1_selector()),
                ("clf", LogisticRegression(max_iter=10_000)),
            ]
        )
        return _search(
            pipeline,
            {"select__estimator__C": [0.05, 0.1, 0.5], "clf__C": [0.01, 0.1, 1.0]},
        )
    if name == "xgboost":
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("select", _l1_selector()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=SEED,
                        n_jobs=4,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )
        return _search(pipeline, {"select__estimator__C": [0.1, 0.5], "clf__max_depth": [2, 3]})
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ("yield_curve_logit", "l1_logit", "xgboost")
