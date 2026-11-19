"""Quantile LASSO (issue #27): do prices and system balance until D-2 improve the forecast?

One linear quantile regression per quantile with an L1 penalty, which keeps only the features that
carry weight. The model is the shared one of the baseline (``models/shared/linear.py``) and runs over
all quarter hours together: with about 150 training days a separate model per quarter hour, as in
classic LEAR, would have only about 150 rows per quarter hour. Description and results: README.md in
this folder. Comparison with other models and surviving features: ``diagnostics.py``.
"""

from energy_price.models.shared.linear import LinearQuantileModel

MODEL = LinearQuantileModel
ISSUE = 27

# Compared by `select`: feature set x L1 penalty, each with every window of DESIGN.windows. Finer than
# the baseline (0, 0.01, 0.1): at 0.01 only part of the coefficients survives, so a coarser grid would
# mainly test strong penalties.
GRID = [
    {"feature_set": fs, "alpha": a}
    for fs in ("B2", "B3", "B4")
    for a in (0.001, 0.003, 0.01, 0.03)
]

# provisional until #51. Two variants: the best setting overall and, because issue #27 asks whether
# data until D-2 help, the best setting with such data (B3 or B4).
CHOSEN = {
    "quantile_lasso": {"feature_set": "B2", "window": "expanding", "alpha": 0.001},
    "quantile_lasso_d2": {"feature_set": "B4", "window": "expanding", "alpha": 0.03},
}

RUN_INFO = {
    "deterministic": "QuantileRegressor with the HiGHS solver has no random component; no seed needed",
    "availability": "D-2 cutoff; daily availability at D-1 11:00 confirmed by user on 2026-10-22",
    "known_limit": "past prices are final here but were provisional at D-1 11:00 (docs/data.md)",
}
