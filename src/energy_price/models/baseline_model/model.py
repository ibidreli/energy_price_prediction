"""Baseline (issue #6): linear quantile regression on calendar (B1) and calendar plus radiation (B2).

The model and the feature sets are shared building blocks (``models/shared/linear.py``,
``features/design.py``); this file only fixes which variants are compared and which settings were
chosen. Description and results: README.md in this folder and docs/models/baseline_model.md.
"""

from energy_price.models.shared.linear import LinearQuantileModel

MODEL = LinearQuantileModel

# Tuning on April and May: feature set x training window x L1 penalty alpha.
GRID = [
    {"feature_set": fs, "window": w, "alpha": a}
    for fs in ("B1", "B2")
    for w in ("expanding", "rolling_56d")
    for a in (0.0, 0.01, 0.1)
]

# Chosen on the tuning period (`make baseline-tune`): for both feature sets the expanding window
# without penalty had the lowest mean pinball loss, see docs/models/baseline_model.md.
CHOSEN = {
    "B1": {"feature_set": "B1", "window": "expanding", "alpha": 0.0},
    "B2": {"feature_set": "B2", "window": "expanding", "alpha": 0.0},
}

COEFFICIENTS_OF = "B2"
COEFFICIENTS_SHOWN = {
    "radiation_100": "Radiation, per 100 W/m²",
    "radiation_change_100_x_position": "Radiation change × position in hour",
    "weekend": "Weekend",
    "holiday_pv_share": "Holiday, full PV share",
}
RUN_INFO = {"deterministic": "QuantileRegressor with the HiGHS solver has no random component; no seed needed"}
