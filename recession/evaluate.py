"""Evaluation protocols: the original (leaky) one, reproduced for comparison,
and leak-free single-split and walk-forward evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from recession.data import build_features, yield_curve_features
from recession.models import SEED, make_model

SPLIT = pd.Timestamp("2002-01-01")


@dataclass(frozen=True)
class Scores:
    roc_auc: float
    avg_precision: float
    brier: float
    n_months: int
    n_recession_months: int

    @classmethod
    def compute(cls, y: pd.Series, p: np.ndarray) -> Scores:
        return cls(
            roc_auc=float(roc_auc_score(y, p)),
            avg_precision=float(average_precision_score(y, p)),
            brier=float(brier_score_loss(y, p)),
            n_months=len(y),
            n_recession_months=int(y.sum()),
        )


def _model_inputs(name: str, raw: pd.DataFrame, fit_end: pd.Timestamp):
    if name == "yield_curve_logit":
        X = yield_curve_features(raw)
        y = raw["recession"].shift(-1).rename("recession_next_month")
        frame = X.join(y).dropna()
        return frame.drop(columns=y.name), frame[y.name].astype(int)
    return build_features(raw, fit_end)


def leak_free_single_split(name: str, raw: pd.DataFrame) -> tuple[Scores, pd.Series]:
    """Fit once on months before SPLIT, score every month from SPLIT on."""
    X, y = _model_inputs(name, raw, fit_end=SPLIT - pd.offsets.MonthBegin(1))
    train, test = X.index < SPLIT, X.index >= SPLIT
    model = make_model(name).fit(X[train], y[train])
    p = model.predict_proba(X[test])[:, 1]
    return Scores.compute(y[test], p), pd.Series(p, index=X.index[test])


def leak_free_walk_forward(name: str, raw: pd.DataFrame) -> tuple[Scores, pd.Series]:
    """Refit every January on all months before it, then forecast that year.

    Differencing decisions, scaling, feature selection and hyperparameters are
    all re-learned from the training window at each refit.
    """
    years = sorted({d.year for d in raw.index if d >= SPLIT})
    preds = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        X, y = _model_inputs(name, raw, fit_end=start - pd.offsets.MonthBegin(1))
        train = X.index < start
        test = (X.index >= start) & (X.index < start + pd.DateOffset(years=1))
        if not test.any():
            continue
        model = make_model(name).fit(X[train], y[train])
        preds.append(pd.Series(model.predict_proba(X[test])[:, 1], index=X.index[test]))
    p = pd.concat(preds)
    _, y_all = _model_inputs(name, raw, fit_end=SPLIT)
    y_test = y_all.loc[p.index]
    return Scores.compute(y_test, p.values), p


def original_leaky(raw_cleaned: pd.DataFrame) -> dict[str, Scores]:
    """Reproduce the original notebook: ADF + scaling were fit on all months
    (data/Dataset_Cleaned.csv) and L1 selection is fit on all months too."""
    df = raw_cleaned.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    y = (df["Regime"] == "Recession").astype(int).values
    X = df.drop(columns=["Date", "Regime"])
    selector = SelectFromModel(
        LogisticRegression(l1_ratio=1, solver="liblinear", C=0.1, max_iter=10_000)
    ).fit(X, y)  # <- uses validation-period labels
    X = X.loc[:, selector.get_support()]
    train = (df["Date"] < SPLIT).values
    results = {}
    for name, model in {
        "l1_logit": LogisticRegression(max_iter=10_000),
        "xgboost": XGBClassifier(random_state=SEED, eval_metric="logloss"),
    }.items():
        fitted = clone(model).fit(X[train], y[train])
        p = fitted.predict_proba(X[~train])[:, 1]
        results[name] = Scores.compute(pd.Series(y[~train]), p)
    return results
