"""Histogram gradient boosting for P10, P50 and P90, reusable by other models."""

from sklearn.ensemble import HistGradientBoostingRegressor

from energy_price.models.shared.base import PerQuantileModel


class GradientBoostingQuantileModel(PerQuantileModel):
    """One regularised tree ensemble per quantile on the shared feature sets.

    Early stopping is deliberately disabled: the estimator's default validation split is
    random. The number of iterations is fixed before the time-ordered backtest instead.
    Folds, not the three estimators, are parallelised by the experiment runner.
    """

    standardise = False

    def __init__(
        self,
        feature_set: str = "B2",
        learning_rate: float = 0.05,
        max_iter: int = 150,
        max_leaf_nodes: int = 7,
        min_samples_leaf: int = 50,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ):
        super().__init__(feature_set)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def make_estimator(self, tau: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=tau,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=False,
            random_state=self.random_state,
        )
