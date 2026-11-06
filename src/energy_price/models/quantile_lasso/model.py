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

# Tuning on April and May: feature set x L1 penalty. Finer than the baseline (0, 0.01, 0.1), which
# chose 0: at 0.01 only part of the coefficients survives, so the old grid mainly tested strong penalties.
GRID = [
    {"feature_set": fs, "window": "expanding", "alpha": a}
    for fs in ("B2", "B3", "B4")
    for a in (0.001, 0.003, 0.01, 0.03)
]

# Empty until the tuning has run: the test step refuses to start without it.
CHOSEN: dict[str, dict] = {}

RUN_INFO = {
    "deterministic": "QuantileRegressor with the HiGHS solver has no random component; no seed needed",
    "availability": "D-2 cutoff; daily availability at D-1 11:00 confirmed by user on 2026-10-22",
    "known_limit": "past prices are final here but were provisional at D-1 11:00 (docs/data.md)",
}
