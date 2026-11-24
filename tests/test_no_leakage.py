"""Guards against look-ahead leakage in feature construction and evaluation."""

import numpy as np
import pandas as pd
import pytest

from recession.data import build_features, decide_differencing, load_raw
from recession.models import make_model

FIT_END = pd.Timestamp("2001-12-01")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


def test_differencing_ignores_data_after_fit_end(raw):
    before = decide_differencing(raw.drop(columns="recession"), FIT_END)
    scrambled = raw.astype(float)
    future = scrambled.index > FIT_END
    rng = np.random.default_rng(0)
    cols = scrambled.columns.drop("recession")
    scrambled.loc[future, cols] = rng.normal(size=(future.sum(), len(cols)))
    after = decide_differencing(scrambled.drop(columns="recession"), FIT_END)
    assert before == after


def test_features_at_t_do_not_use_later_rows(raw):
    X, _ = build_features(raw, FIT_END)
    t = pd.Timestamp("2005-06-01")
    perturbed = raw.astype(float)
    cols = perturbed.columns.drop("recession")
    later = perturbed.index > t
    perturbed.loc[later, cols] = perturbed.loc[later, cols] * 3 + 7
    X2, _ = build_features(perturbed, FIT_END)
    pd.testing.assert_series_equal(X.loc[t], X2.loc[t])


def test_target_is_next_month_regime(raw):
    _, y = build_features(raw, FIT_END)
    t = y.index[100]
    assert y.loc[t] == raw["recession"].shift(-1).loc[t]


def test_preprocessing_lives_inside_the_model():
    """Scaling and L1 selection must be pipeline steps so CV and refits re-fit them."""
    for name in ("l1_logit", "xgboost"):
        steps = dict(make_model(name).estimator.steps)
        assert {"scale", "select", "clf"} <= steps.keys()
