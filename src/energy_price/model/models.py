"""Historical reference and linear quantile regression for P10, P50 and P90.

Reference: for each local clock time (slot), the empirical quantiles of the prices of that slot
over the last ``window_days`` days known at the issue time (D-1-window to D-2). It needs no model
and no features, so it is the value a model has to beat.

Linear quantile regression: one linear equation per quantile, fitted with the pinball loss
(``sklearn.linear_model.QuantileRegressor``, L1 penalty ``alpha``). Each quantile has its own
coefficients, so radiation may lower P10 strongly while leaving P90 almost unchanged: the price
has two clouds, short (median 18.9 ct/kWh) and long (median 5.6 ct/kWh, down to -688). Ordinary
least squares would fit one mean between them.

Feature sets:

- ``B1`` calendar only: hour dummies, a linear trend within each hour (position x hour), weekend,
  PV share of cantons with a holiday.
- ``B2`` B1 plus radiation (per 100 W/m²) and radiation change x position, because rising sun
  tips the system long towards :45 and falling sun short (EDA, H8).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from energy_price.availability import known_at_issue
from energy_price.model.dataset import TARGET
from energy_price.model.scores import COLUMNS, QUANTILES

HOURS = range(24)
FEATURE_SETS = ("B1", "B2")


def features(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Design matrix of ``feature_set`` for the rows of ``table`` (without intercept)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected one of {FEATURE_SETS}")
    hour, position = table["hour"], table["position"]
    columns = {f"hour_{h:02d}": (hour == h).astype(float) for h in HOURS if h != 0}
    columns |= {f"trend_{h:02d}": position * (hour == h) for h in HOURS}
    columns["weekend"] = table["weekend"]
    columns["holiday_pv_share"] = table["holiday_pv_share"]
    if feature_set == "B2":
        columns["radiation_100"] = table["radiation"] / 100
        columns["radiation_change_100_x_position"] = table["radiation_change"] / 100 * position
    return pd.DataFrame(columns, index=table.index)


class LinearQuantileModel:
    """One ``QuantileRegressor`` per quantile on standardised features of ``feature_set``.

    Mean and standard deviation for the standardisation come from the training rows only. Rows
    with a missing feature get no forecast (NaN). Crossing quantiles are sorted per row.
    The HiGHS solver is deterministic: the same data give the same coefficients, no seed needed.
    """

    def __init__(self, feature_set: str = "B2", alpha: float = 0.0, quantiles: tuple[float, ...] = QUANTILES):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}, expected one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.alpha = alpha
        self.quantiles = quantiles
        self.models: dict[float, QuantileRegressor] = {}

    def fit(self, table: pd.DataFrame) -> LinearQuantileModel:
        x = features(table, self.feature_set)
        rows = x.notna().all(axis=1) & table[TARGET].notna()
        if not rows.any():
            raise ValueError("no complete training rows")
        x, y = x[rows], table.loc[rows, TARGET]
        self.mean_ = x.mean()
        self.scale_ = x.std(ddof=0).replace(0.0, 1.0)  # a constant column (e.g. no holiday in the window) stays 0
        z = (x - self.mean_) / self.scale_
        self.models = {
            tau: QuantileRegressor(quantile=tau, alpha=self.alpha, solver="highs").fit(z.to_numpy(), y.to_numpy())
            for tau in self.quantiles
        }
        self.n_train_ = int(rows.sum())
        return self

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("fit the model first")
        x = features(table, self.feature_set)
        rows = x.notna().all(axis=1)
        out = pd.DataFrame(np.nan, index=table.index, columns=[COLUMNS[tau] for tau in self.quantiles])
        if rows.any():
            z = ((x[rows] - self.mean_) / self.scale_).to_numpy()
            pred = np.column_stack([self.models[tau].predict(z) for tau in self.quantiles])
            out.loc[rows] = np.sort(pred, axis=1)
        return out

    def coefficients(self) -> pd.DataFrame:
        """Coefficients in the original units (ct/kWh per unit of the feature), one column per quantile."""
        if not self.models:
            raise RuntimeError("fit the model first")
        out = {}
        for tau, model in self.models.items():
            coef = pd.Series(model.coef_, index=self.mean_.index) / self.scale_
            intercept = model.intercept_ - float((coef * self.mean_).sum())
            out[COLUMNS[tau]] = pd.concat([pd.Series({"intercept": intercept}), coef])
        return pd.DataFrame(out)


def slot(stamps_local: pd.Series) -> pd.Series:
    """Local clock time as minutes after midnight. Both passes of the repeated hour share a slot."""
    return stamps_local.dt.hour * 60 + stamps_local.dt.minute


def reference_forecast(prices: pd.DataFrame, target_rows: pd.DataFrame, day: dt.date,
                       window_days: int = 56, quantiles: tuple[float, ...] = QUANTILES) -> pd.DataFrame:
    """Quantiles per slot of the prices known for ``day`` (D-1-window_days to D-2), for ``target_rows``.

    ``prices`` needs ``timestamp_local`` and the target; ``target_rows`` are the quarter hours of
    delivery day ``day``. Slots without history (e.g. 02:00 to 02:45 after a short day) stay NaN.
    """
    known = known_at_issue(prices, day)
    start = pd.Timestamp(day - dt.timedelta(days=window_days + 1)).tz_localize(known["timestamp_local"].dt.tz)
    window = known[known["timestamp_local"] >= start]
    table = window.groupby(slot(window["timestamp_local"]))[TARGET].quantile(list(quantiles)).unstack()
    table.columns = [COLUMNS[tau] for tau in table.columns]
    out = table.reindex(slot(target_rows["timestamp_local"]).to_numpy())
    out.index = target_rows.index
    return out
