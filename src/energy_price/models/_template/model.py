"""TODO: name of the model (issue #<n>). One sentence on the approach and why it should beat the baseline.

Copy this folder to start a new model (docs/models/README.md). Only this file and README.md
change. Shared building blocks to reuse:

- ``energy_price.features.design``        feature sets B1 and B2, new feature blocks go there
- ``energy_price.models.shared.base``     ``PerQuantileModel``: one estimator per quantile,
                                          standardisation, missing features and sorted quantiles
- ``energy_price.models.shared.linear``   ``LinearQuantileModel`` of the baseline

The placeholder below runs end to end (quantiles of the training prices, no features), so the
backtest of a fresh copy works before the real model is written: ``make select MODEL=<name>``, or a
quick smoke run with ``python -m energy_price.evaluation.experiment <name> select --days 2026-04-01 2026-04-03``.
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

# TODO: settings compared by `select`: keyword arguments of Model, WITHOUT "window". Every setting runs
# with every window of WINDOWS (optional here, default DESIGN.windows) and their window mean.
GRID = [{"feature_set": "B1"}]

# TODO: the selected candidate of reports/<name>/select_selection.json (settings incl. "window"), copied by hand.
CHOSEN = {"template": {"feature_set": "B1", "window": "expanding"}}
