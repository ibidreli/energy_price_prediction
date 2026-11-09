"""Issue #25: frozen experts, configurable pool and weekly full-information routing."""

from energy_price.models.shared.routing import TimeRouterModel

MODEL = TimeRouterModel
ISSUE = 25
EXPERIMENT_RUNNER = "energy_price.evaluation.time_router"

# All settings were selected in their own April/May experiments. No expert is
# retuned for the combination. Additional admitted experts only need an entry.
POOL = {
    "reference": {"kind": "reference", "window_days": 56},
    "B2": {"module": "baseline_model", "settings": {"feature_set": "B2", "window": "expanding", "alpha": 0.0}},
    "gbm_quantile": {"module": "gbm_quantile", "settings": {
        "feature_set": "B2", "window": "expanding", "learning_rate": 0.05, "max_iter": 150,
        "max_leaf_nodes": 7, "min_samples_leaf": 50, "l2_regularization": 1.0, "random_state": 0}},
    "regime_mixture": {"module": "regime_mixture", "settings": {
        "feature_set": "B2", "window": "expanding", "C": 1.0, "alpha": 0.0}},
}

LAYOUTS = {"global": [0, 24], "domain": [0, 8, 16, 24], "6h": [0, 6, 12, 18, 24], "hourly": list(range(25))}
REVISION = "hourly_v2"
# Time-of-day resolution and refit frequency are separate: 24 clock-hour
# decisions, rebuilt once per week. Low power is reported, never a silent
# downgrade to one expert/weight vector for the whole day.
MAIN_LAYOUT = "hourly"
SELECTION_MODE = "significant"
PRIMARY_STRATEGY = "selection"
GRID = [{"strategy": name} for name in ("mean", "selection", "weights")]
# The pool runner reads design.json and nomination.json instead of CHOSEN.
CHOSEN = {}
SEED = 0
BOOTSTRAP_REPS = 5000
BLOCK_DAYS = 7
MIN_HISTORY_DAYS = 28
POOL_FREEZE_DATE = "2026-11-27"
CALIBRATION = {"calibration_days": 56, "grouping": "6h"}
