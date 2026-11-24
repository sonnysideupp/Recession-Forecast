"""Load FRED-MD data and build leak-free features.

Every transformation that is *learned* from data (which series to difference)
is decided on the training window only, so features for month t never depend
on information after t or on the evaluation period.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LAGS = (3, 6, 9, 12, 18)
ADF_THRESHOLD = 0.01


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly macro series plus the NBER regime, indexed by month."""
    macro = pd.read_csv(data_dir / "Macroeconomic_Variables.csv")
    macro = macro.rename(columns={"sasdate": "Date"})
    regimes = pd.read_csv(data_dir / "Recession_Periods.csv")
    if len(regimes) != len(macro):
        raise ValueError("macro and regime files must cover the same months")
    macro = macro.copy()
    macro["Date"] = pd.to_datetime(macro["Date"], format="%m/%d/%Y")
    macro["recession"] = (regimes["Regime"].values == "Recession").astype(int)
    return macro.set_index("Date")


def _needs_difference(series: pd.Series) -> bool:
    """True if an ADF test cannot reject a unit root at ADF_THRESHOLD."""
    clean = series.dropna()
    if clean.nunique() <= 1:
        return False
    return adfuller(clean, result_object=False)[1] > ADF_THRESHOLD


def decide_differencing(raw: pd.DataFrame, fit_end: pd.Timestamp) -> dict[str, int]:
    """Number of differences (0-2) per series, decided on data up to fit_end only.

    Mirrors the original notebook (ADF test, difference, test again) but never
    looks past fit_end.
    """
    window = raw.loc[:fit_end]
    orders: dict[str, int] = {}
    for col in window.columns:
        series, order = window[col], 0
        while order < 2 and _needs_difference(series):
            series, order = series.diff(), order + 1
        orders[col] = order
    return orders


def recession_target(raw: pd.DataFrame, horizon: int) -> pd.Series:
    """y[t] = 1 if month t+horizon is an NBER recession month."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 month")
    return raw["recession"].shift(-horizon).rename(f"recession_in_{horizon}m")


def build_features(
    raw: pd.DataFrame, fit_end: pd.Timestamp, horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """Features X[t] use data up to month t; target y[t] = recession in month t+horizon.

    Series with any missing value are dropped (as in the original notebook);
    this uses only data availability, never the labels.
    """
    predictors = raw.drop(columns="recession")
    predictors = predictors.loc[:, predictors.notna().all()]
    orders = decide_differencing(predictors, fit_end)

    transformed = predictors.copy()
    for col, order in orders.items():
        for _ in range(order):
            transformed[col] = transformed[col].diff()

    lagged = [transformed]
    for lag in LAGS:
        lagged.append(transformed.shift(lag).add_suffix(f" {lag}M lag"))
    features = pd.concat(lagged, axis=1)

    return _with_target(features, recession_target(raw, horizon))


def yield_curve_inputs(raw: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Classic benchmark: 10y minus 3m Treasury spread and its lags (levels)."""
    spread = (raw["GS10"] - raw["TB3MS"]).rename("term_spread")
    cols = [spread] + [spread.shift(lag).rename(f"term_spread {lag}M lag") for lag in LAGS]
    return _with_target(pd.concat(cols, axis=1), recession_target(raw, horizon))


def _with_target(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    frame = features.join(target).dropna()
    return frame.drop(columns=target.name), frame[target.name].astype(int)
