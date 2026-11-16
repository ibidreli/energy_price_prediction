"""Issue #19: predeclared calibration grid and selection for each frozen base."""

from energy_price.models.shared.calibration import CalibratedQuantileModel

MODEL = CalibratedQuantileModel
ISSUE = 19
BASES = {"B2": "baseline_model:B2", "gbm_quantile": "gbm_quantile:gbm_quantile"}
GRID = [
    {"base": base, "calibration_days": days, "grouping": grouping}
    for base in BASES.values()
    for days in (28, 56)
    for grouping in ("global", "6h", "hour")
] + [{"base": base, "calibrated": False} for base in BASES.values()]

# Frozen on April/May before the test run. Baseline: no candidate met both
# constraints, so lowest pinball fallback. GBM: successful constrained selection.
CHOSEN = {'calibrated_B2': {'base': 'baseline_model:B2',
                   'calibration_days': 56,
                   'grouping': 'global'},
 'raw_B2': {'base': 'baseline_model:B2', 'calibrated': False},
 'calibrated_gbm_quantile': {'base': 'gbm_quantile:gbm_quantile',
                             'calibration_days': 56,
                             'grouping': '6h'},
 'raw_gbm_quantile': {'base': 'gbm_quantile:gbm_quantile', 'calibrated': False}}


def select_candidates(scores, settings):
    """Lowest pinball among successful candidates, otherwise lowest pinball.

    Success on tuning: 78-82% coverage and <=1% loss increase against the
    matching uncalibrated base, all scored on identical delivery days.
    A failed constraint is reported, never hidden by selecting on test data.
    """
    chosen = {}
    for name, base in BASES.items():
        raw = next(key for key, value in settings.items() if value == {"base": base, "calibrated": False})
        candidates = [key for key, value in settings.items() if value.get("base") == base
                      and value.get("calibrated", True)]
        successful = [key for key in candidates if 0.78 <= scores.loc[key, "coverage_80"] <= 0.82
                      and scores.loc[key, "pinball"] <= 1.01 * scores.loc[raw, "pinball"]]
        best = scores.loc[successful or candidates, "pinball"].idxmin()
        chosen[f"calibrated_{name}"] = settings[best]
        chosen[f"raw_{name}"] = settings[raw]
    return chosen


RUN_INFO = {
    "calibration": "separate 10% tails; finite-sample rank; weekly OOS residuals with D-2 cutoff",
    "selection": "per base: lowest pinball satisfying 78-82% coverage and <=1% loss increase; otherwise lowest pinball",
    "warmup_days": 28,
    "holdout": "locked",
}
