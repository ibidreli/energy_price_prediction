"""Building blocks shared by all models: P10, P50 and P90 for every quarter hour of a delivery day.

``QuantileModel`` is the interface the backtest relies on: ``fit(table)`` and ``predict(table)``
returning ``q10``, ``q50`` and ``q90``. ``PerQuantileModel`` implements it for the usual case of
one estimator per quantile on a named feature set (``features.design``). A new model then only
says which estimator to use, e.g.::

    class GradientBoostingQuantileModel(PerQuantileModel):
        def make_estimator(self, tau):
            return HistGradientBoostingRegressor(loss="quantile", quantile=tau)

``PerQuantileModel`` takes care of what every model must get right: features only from the rows
given to ``fit`` (the backtest passes only prices known at the issue time), the standardisation
fitted on those rows, no forecast for rows with a missing feature, and sorted quantiles.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from energy_price.features.dataset import TARGET
from energy_price.features.design import FEATURE_SETS, design_matrix

QUANTILES = (0.1, 0.5, 0.9)
COLUMNS = {0.1: "q10", 0.5: "q50", 0.9: "q90"}


@runtime_checkable
class QuantileModel(Protocol):
    """A model that learns from past quarter hours and forecasts P10, P50 and P90.

    ``predict`` returns one row per input row with the columns ``q10``, ``q50`` and ``q90`` (NaN
    where the model cannot forecast) and ``q10 <= q50 <= q90``.
    """

    def fit(self, table: pd.DataFrame) -> QuantileModel: ...

    def predict(self, table: pd.DataFrame) -> pd.DataFrame: ...


class PerQuantileModel:
    """One estimator per quantile on the columns of ``feature_set``. Subclasses define ``make_estimator``.

    ``standardise=True`` scales every feature with mean and standard deviation of the training rows
    (needed for penalised linear models, harmless for trees). A constant column stays 0.
    """

    standardise = True

    def __init__(self, feature_set: str = "B2", quantiles: tuple[float, ...] = QUANTILES):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}, expected one of {tuple(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.quantiles = quantiles
        self.models: dict[float, Any] = {}

    def make_estimator(self, tau: float) -> Any:
        """A new, unfitted scikit-learn style estimator for quantile ``tau``."""
        raise NotImplementedError

    def features(self, table: pd.DataFrame) -> pd.DataFrame:
        """Design matrix of the model. Override to add own columns."""
        return design_matrix(table, self.feature_set)

    def fit(self, table: pd.DataFrame) -> PerQuantileModel:
        x = self.features(table)
        rows = x.notna().all(axis=1) & table[TARGET].notna()
        if not rows.any():
            raise ValueError("no complete training rows")
        x, y = x[rows], table.loc[rows, TARGET]
        self.mean_ = x.mean() if self.standardise else pd.Series(0.0, index=x.columns)
        self.scale_ = x.std(ddof=0).replace(0.0, 1.0) if self.standardise else pd.Series(1.0, index=x.columns)
        z = ((x - self.mean_) / self.scale_).to_numpy()
        self.models = {tau: self.make_estimator(tau).fit(z, y.to_numpy()) for tau in self.quantiles}
        self.n_train_ = int(rows.sum())
        return self

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("fit the model first")
        x = self.features(table)
        rows = x.notna().all(axis=1)
        out = pd.DataFrame(np.nan, index=table.index, columns=[COLUMNS[tau] for tau in self.quantiles])
        if rows.any():
            z = ((x[rows] - self.mean_) / self.scale_).to_numpy()
            pred = np.column_stack([self.models[tau].predict(z) for tau in self.quantiles])
            out.loc[rows] = np.sort(pred, axis=1)
        return out
