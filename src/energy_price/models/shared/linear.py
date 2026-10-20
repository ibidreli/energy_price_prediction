"""Linear quantile regression for P10, P50 and P90, usable by every model (first used by the baseline).

One linear equation per quantile, fitted with the pinball loss (``sklearn.linear_model.QuantileRegressor``,
L1 penalty ``alpha``). Each quantile has its own coefficients, so radiation may lower P10 strongly
while leaving P90 almost unchanged: the price has two clouds, short (median 18.9 ct/kWh) and long
(median 5.6 ct/kWh, down to -688). Ordinary least squares would fit one mean between them.
The HiGHS solver is deterministic: the same data give the same coefficients, no seed needed.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import QuantileRegressor

from energy_price.models.shared.base import COLUMNS, QUANTILES, PerQuantileModel


class LinearQuantileModel(PerQuantileModel):
    """``QuantileRegressor`` per quantile on standardised features of ``feature_set``."""

    def __init__(self, feature_set: str = "B2", alpha: float = 0.0, quantiles: tuple[float, ...] = QUANTILES):
        super().__init__(feature_set, quantiles)
        self.alpha = alpha

    def make_estimator(self, tau: float) -> QuantileRegressor:
        return QuantileRegressor(quantile=tau, alpha=self.alpha, solver="highs")

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
