"""Regime model short/long (issue #16): P(short) and price quantiles per direction, mixed to P10, P50, P90.

The price has two clouds, short (price A, never negative in 2026) and long (price B, down to -688 ct/kWh).
A single quantile model lands between them; modelling the direction first should fit the jump better.
Status: placeholder from the template, the real model follows (steps 3 to 5 in issue #16).

Copy this folder to start a new model (docs/models/README.md). Only this file and README.md
change. Shared building blocks to reuse:

- ``energy_price.features.design``        feature sets B1 and B2, new feature blocks go there
- ``energy_price.models.shared.base``     ``PerQuantileModel``: one estimator per quantile,
                                          standardisation, missing features and sorted quantiles
- ``energy_price.models.shared.linear``   ``LinearQuantileModel`` of the baseline

The placeholder below runs end to end (quantiles of the training prices, no features), so the
backtest of a fresh copy works before the real model is written:
``make backtest MODEL=<name> STEP=tune``, then ``STEP=test``.
"""

from sklearn.dummy import DummyRegressor

from energy_price.models.shared.base import PerQuantileModel


class Model(PerQuantileModel):
    """TODO: replace the estimator, e.g. HistGradientBoostingRegressor(loss="quantile", quantile=tau)."""

    def __init__(self, feature_set: str = "B1"):
        super().__init__(feature_set)

    def make_estimator(self, tau: float):
        return DummyRegressor(strategy="quantile", quantile=tau)


MODEL = Model

# TODO: settings to compare on April and May; keyword arguments of Model plus "window".
GRID = [{"feature_set": "B1", "window": w} for w in ("expanding", "rolling_56d")]

# TODO: the best entry of reports/<name>/tune_scores.csv, fixed before the test period is run.
CHOSEN = {"placeholder": {"feature_set": "B1", "window": "expanding"}}
