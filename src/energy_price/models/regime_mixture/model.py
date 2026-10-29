"""Regime model short/long (issue #16): P(short) and price quantiles per direction, mixed to P10, P50, P90.

The price has two clouds: short (price A, median 18.9 ct/kWh, never negative in 2026) and long
(price B, median 5.6 ct/kWh, down to -688). A single quantile model lands between them, where
hardly any real price lies. This model splits the forecast in two parts:

1. ``p`` = P(short) per quarter hour from a logistic regression. The direction follows the clock
   time and the position in the hour (EDA H8: at 17:00 the system is short in 30 % of the cases,
   at 17:45 in 79 %), so the calendar features of B1/B2 carry the signal.
2. Price quantiles separately for short and long, fitted with the linear quantile regression of
   the baseline on the rows of each direction only.

The forecast distribution is the mixture F(x) = p * F_short(x) + (1 - p) * F_long(x); P10, P50 and
P90 are the values where F reaches 0.1, 0.5 and 0.9. F_short and F_long are piecewise linear
between the fitted quantiles ``GRID_QUANTILES`` (0 below the lowest, 1 above the highest).

The direction label ``short`` (TSI < 0) is a training target like the price, never a feature, and
the backtest gives the model only rows up to D-2. ``predict`` also returns ``p_short``, the
forecast probability that the control area is short (who pays whom).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor

from energy_price.features.dataset import DIRECTION, TARGET
from energy_price.features.design import FEATURE_SETS, design_matrix
from energy_price.models.shared.base import COLUMNS, QUANTILES

GRID_QUANTILES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
REGIMES = {"short": 1.0, "long": 0.0}


def mixture_quantiles(p: np.ndarray, q_short: np.ndarray, q_long: np.ndarray,
                      taus: tuple[float, ...] = GRID_QUANTILES,
                      levels: tuple[float, ...] = QUANTILES) -> np.ndarray:
    """Quantiles ``levels`` of the mixture p * F_short + (1 - p) * F_long, one row per forecast.

    ``q_short`` and ``q_long`` hold the quantiles ``taus`` of each direction (rows x len(taus)).
    Each CDF is piecewise linear between its quantiles, 0 below the first and 1 above the last.
    """
    grid = np.asarray(taus, dtype=float)
    out = np.empty((len(p), len(levels)))
    for i in range(len(p)):
        qs, ql = np.sort(q_short[i]), np.sort(q_long[i])
        x = np.unique(np.concatenate([qs, ql]))
        cdf = (p[i] * np.interp(x, qs, grid, left=0.0, right=1.0)
               + (1 - p[i]) * np.interp(x, ql, grid, left=0.0, right=1.0))
        cdf = np.maximum.accumulate(cdf)
        # interp needs increasing support points: keep the first x of every distinct CDF value
        keep = np.concatenate([[True], np.diff(cdf) > 0])
        out[i] = np.interp(levels, cdf[keep], x[keep])
    return out


class RegimeMixtureModel:
    """Logistic regression for P(short) and linear quantile regressions per direction on ``feature_set``."""

    def __init__(self, feature_set: str = "B2", C: float = 1.0, alpha: float = 0.0):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r}, expected one of {tuple(FEATURE_SETS)}")
        self.feature_set, self.C, self.alpha = feature_set, C, alpha
        self.classifier = None
        self.quantile_models: dict[str, dict[float, QuantileRegressor]] = {}

    def _standardised(self, table: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        x = design_matrix(table, self.feature_set)
        rows = x.notna().all(axis=1)
        return ((x[rows] - self.mean_) / self.scale_).to_numpy(), rows

    def fit(self, table: pd.DataFrame) -> RegimeMixtureModel:
        x = design_matrix(table, self.feature_set)
        rows = x.notna().all(axis=1) & table[TARGET].notna() & table[DIRECTION].notna()
        if not rows.any():
            raise ValueError("no complete training rows (price, direction and features)")
        x, y, short = x[rows], table.loc[rows, TARGET], table.loc[rows, DIRECTION]
        if short.nunique() < 2:
            raise ValueError("training rows contain only one direction")
        self.mean_ = x.mean()
        self.scale_ = x.std(ddof=0).replace(0.0, 1.0)
        z = ((x - self.mean_) / self.scale_).to_numpy()

        self.classifier = LogisticRegression(C=self.C, max_iter=2000).fit(z, short.to_numpy())
        self.quantile_models = {}
        for name, value in REGIMES.items():
            mask = (short == value).to_numpy()
            self.quantile_models[name] = {
                tau: QuantileRegressor(quantile=tau, alpha=self.alpha, solver="highs").fit(z[mask], y.to_numpy()[mask])
                for tau in GRID_QUANTILES
            }
        self.n_train_ = int(rows.sum())
        self.share_short_ = float(short.mean())
        return self

    def _regime_quantiles(self, z: np.ndarray, name: str) -> np.ndarray:
        return np.column_stack([self.quantile_models[name][tau].predict(z) for tau in GRID_QUANTILES])

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.classifier is None:
            raise RuntimeError("fit the model first")
        columns = [COLUMNS[tau] for tau in QUANTILES] + ["p_short"]
        out = pd.DataFrame(np.nan, index=table.index, columns=columns)
        z, rows = self._standardised(table)
        if rows.any():
            p = self.classifier.predict_proba(z)[:, list(self.classifier.classes_).index(1.0)]
            mixed = mixture_quantiles(p, self._regime_quantiles(z, "short"), self._regime_quantiles(z, "long"))
            out.loc[rows, columns[:3]] = np.sort(mixed, axis=1)
            out.loc[rows, "p_short"] = p
        return out


MODEL = RegimeMixtureModel
ISSUE = 16

# Tuning on April and May: feature set x training window (logistic C = 1, no L1 penalty, as the
# baseline chose alpha = 0 on the same period).
GRID = [
    {"feature_set": fs, "window": w}
    for fs in ("B1", "B2")
    for w in ("expanding", "rolling_56d")
]

# Chosen on the tuning period (`make backtest MODEL=regime_mixture STEP=tune`): B2 with the expanding
# window had the lowest pinball loss (3.323 against 3.449 of the reference); fixed before the test.
CHOSEN = {"regime_mixture": {"feature_set": "B2", "window": "expanding"}}

RUN_INFO = {"deterministic": "LogisticRegression (lbfgs) and QuantileRegressor (HiGHS) have no random component"}
