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
# The calibration itself keeps all rows; the base runs with its own window and refit frequency.
WINDOWS = ("expanding",)

# provisional until #51: settings selected with the old pipeline, rerun `make select MODEL=calibrated_bands`.
CHOSEN = {
    "calibrated_B2": {"base": "baseline_model:B2", "calibration_days": 56, "grouping": "global",
                      "window": "expanding"},  # provisional until #51
    "raw_B2": {"base": "baseline_model:B2", "calibrated": False, "window": "expanding"},  # provisional until #51
    "calibrated_gbm_quantile": {"base": "gbm_quantile:gbm_quantile", "calibration_days": 56, "grouping": "6h",
                                "window": "expanding"},  # provisional until #51
    "raw_gbm_quantile": {"base": "gbm_quantile:gbm_quantile", "calibrated": False,
                         "window": "expanding"},  # provisional until #51
}


def select_candidates(scores, settings):
    """Per base: lowest mean daily pinball among successful candidates, otherwise lowest mean daily pinball.

    Success on the validation days: 78-82% coverage and <=1% loss increase against the matching
    uncalibrated base, all scored on identical quarter hours. A failed constraint is reported in
    ``select_selection.json``, never hidden. Settings are matched on ``base`` and ``calibrated`` only,
    so the training ``window`` added by ``select`` does not matter.
    """
    chosen = {}
    for name, base in BASES.items():
        raw = next(key for key, value in settings.items()
                   if value.get("base") == base and value.get("calibrated", True) is False)
        candidates = [key for key, value in settings.items() if value.get("base") == base
                      and value.get("calibrated", True)]
        successful = [key for key in candidates if 0.78 <= scores.loc[key, "coverage_80"] <= 0.82
                      and scores.loc[key, "mean_daily_pinball"] <= 1.01 * scores.loc[raw, "mean_daily_pinball"]]
        best = scores.loc[successful or candidates, "mean_daily_pinball"].idxmin()
        chosen[f"calibrated_{name}"] = settings[best]
        chosen[f"raw_{name}"] = settings[raw]
    return chosen


RUN_INFO = {
    "calibration": "separate 10% tails; finite-sample rank; out-of-sample residuals of the base with its own "
                   "window and refit folds, training rows up to D-2",
    "selection": "per base: lowest mean daily pinball satisfying 78-82% coverage and <=1% loss increase on the "
                 "validation days; otherwise lowest mean daily pinball",
    "warmup_days": 28,
    "test": "locked: select sees only validation days, the test period is scored once by final-test (#53)",
}
