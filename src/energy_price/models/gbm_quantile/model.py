"""Issue #17: quantile boosting on B2, compared using the shared weekly backtest."""

from energy_price.models.shared.boosting import GradientBoostingQuantileModel

MODEL = GradientBoostingQuantileModel

# Four conservative configurations, each with both training windows. The first is
# the anchor; the others change learning rate, tree size or leaf support individually.
GRID = [
    {"feature_set": "B2", "window": window, "learning_rate": rate, "max_iter": 150,
     "max_leaf_nodes": leaves, "min_samples_leaf": support, "l2_regularization": 1.0,
     "random_state": 0}
    for window in ("expanding", "rolling_56d")
    for rate, leaves, support in ((0.05, 7, 50), (0.1, 7, 50), (0.05, 15, 50), (0.05, 7, 100))
]

# Lowest mean pinball on April/May (3.3086848336904033), fixed before any test run.
# Full selection record: reports/gbm_quantile/tune_selection.json.
CHOSEN = {
    "gbm_quantile": {
        "feature_set": "B2", "window": "expanding", "learning_rate": 0.05, "max_iter": 150,
        "max_leaf_nodes": 7, "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 0,
    },
}

# Shuffle raw inputs jointly by source, then recompute derived B2 features.
IMPORTANCE_GROUPS = {
    "calendar": ["hour", "position", "weekend"],
    "holidays": ["holiday_pv_share"],
    "radiation": ["radiation", "radiation_change"],
}
IMPORTANCE_REPEATS = 5

RUN_INFO = {
    "availability": "D-2 cutoff; daily Swissgrid publication not yet confirmed",
    "early_stopping": False,
    "random_state": 0,
    "feature_comparison": "same B2 features as the linear baseline; no extra data sources",
}
