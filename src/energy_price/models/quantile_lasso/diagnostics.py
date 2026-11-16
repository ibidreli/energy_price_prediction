"""Do data until D-2 help? Comparison of the quantile LASSO with other models and its surviving features.

``make backtest`` compares a model with the reference and the baseline B2 only. This script adds what
issue #27 asks for, on the same quarter hours for all models:

- ``gbm_quantile`` as a further comparison, fitted here with its unchanged ``CHOSEN``,
- the daily pinball difference of every pair with a 95 % block bootstrap interval,
- the share of weekly fits in which a feature keeps a coefficient, per quantile. This describes what
  the model uses, not what causes the price,
- B2 trained only from the first day with B3 and B4 features. B3 and B4 have fewer training rows than
  B2 (minimum history, docs/data.md), so a difference to B2 also measures the data. This variant
  separates the two. It is only reported and never selected.

Nothing is selected here. On ``tune`` the best setting per feature set is analysed, on ``test`` the
settings of ``CHOSEN``.

Usage (repository root, after ``make backtest MODEL=quantile_lasso STEP=<period>``)::

    python -m energy_price.models.quantile_lasso.diagnostics --period tune
    python -m energy_price.models.quantile_lasso.diagnostics --period test

Writes ``<period>_diagnostic_scores.csv``, ``<period>_comparison.csv`` and
``<period>_surviving_features.csv`` to ``reports/quantile_lasso/``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import REFERENCE, evaluate, period_days
from energy_price.evaluation.experiment import BASELINE_VARIANT, label, load, run_variant
from energy_price.evaluation.scores import COLUMNS, block_bootstrap
from energy_price.features.dataset import build_table, load_inputs
from energy_price.features.design import BLOCKS, FEATURE_SETS, design_matrix
from energy_price.paths import REPORTS_DIR

NAME = "quantile_lasso"
BOOSTING = "gbm_quantile"
BASELINE = f"baseline_{BASELINE_VARIANT}"
NONZERO = 1e-9  # the L1 penalty sets coefficients to zero up to the precision of the solver
QUANTILE_COLUMNS = list(COLUMNS.values())


def best_per_feature_set(scores: pd.DataFrame, grid: list[dict]) -> dict[str, dict]:
    """The setting with the lowest pinball loss of each feature set, by its name in the tuning results."""
    best = {}
    for feature_set in dict.fromkeys(settings["feature_set"] for settings in grid):
        named = {label(settings): settings for settings in grid if settings["feature_set"] == feature_set}
        name = scores.loc[list(named), "pinball"].idxmin()
        best[name] = named[name]
    return best


def first_complete_day(table: pd.DataFrame, feature_set: str) -> pd.Timestamp:
    """First delivery day with a value in every feature of ``feature_set``."""
    return table.loc[design_matrix(table, feature_set).notna().all(axis=1), "delivery_day"].min()


def comparison(daily: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Daily pinball difference ``model`` minus ``against`` with its 95 % interval; below 0 ``model`` is better."""
    rows = []
    for model, against in pairs:
        diff = (daily[model] - daily[against]).to_numpy()
        low, high = block_bootstrap(diff)
        rows.append({"model": model, "against": against, "pinball_model": daily[model].mean(),
                     "pinball_against": daily[against].mean(), "mean_difference": diff.mean(),
                     "low": low, "high": high, "entirely_below_zero": high < 0, "n_days": len(diff)})
    return pd.DataFrame(rows)


def surviving_features(coefficients: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per model, feature and quantile: share of weekly fits with a coefficient, and the one of the last fit."""
    blocks = {column: block for block in FEATURE_SETS["B4"] for column in BLOCKS[block](table)}
    rows = coefficients[coefficients["feature"] != "intercept"]
    long = rows.melt(id_vars=["model", "feature", "fit_day"], value_vars=QUANTILE_COLUMNS,
                     var_name="quantile", value_name="coefficient").sort_values("fit_day")
    out = long.groupby(["model", "feature", "quantile"], sort=False).agg(
        share_of_fits=("coefficient", lambda values: (values.abs() > NONZERO).mean()),
        last_coefficient=("coefficient", "last"),
    ).reset_index()
    return out.assign(block=out["feature"].map(blocks))


def _check_matches_saved(fresh: pd.DataFrame, saved: pd.DataFrame, name: str) -> None:
    """A fresh fit must give the forecast of the backtest, otherwise code or data have changed in between."""
    old = saved[saved["model"] == name].set_index("timestamp_utc")[QUANTILE_COLUMNS]
    new = fresh.set_index("timestamp_utc")[QUANTILE_COLUMNS].reindex(old.index)
    if old.empty or not np.allclose(old.to_numpy(), new.to_numpy(), atol=1e-9, equal_nan=True):
        raise ValueError(f"the fresh fit of {name} differs from the saved forecast; rerun the backtest")


def analyse(table: pd.DataFrame, days: list[pd.Timestamp], saved: pd.DataFrame, candidates: dict[str, dict],
            coefficients: pd.DataFrame | None = None,
            jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scores, pairwise comparison and surviving features of ``candidates``, on the common quarter hours.

    ``saved`` holds the forecasts of the backtest. Without ``coefficients`` the candidates are fitted
    again, because the tuning step of the shared pipeline does not keep them.
    """
    if BASELINE not in set(saved["model"]):
        raise ValueError(f"forecasts of the baseline ({BASELINE}) are missing; a comparison without it would be "
                         "incomplete, run the baseline on this period first")
    module = load(NAME)
    if coefficients is None:
        fits = []
        for name, settings in candidates.items():
            fresh, coef = run_variant(module, table, days, name, settings, jobs)
            _check_matches_saved(fresh, saved, name)
            fits.append(coef.assign(model=name))
        coefficients = pd.concat(fits, ignore_index=True)

    frames = [saved[saved["model"].isin([REFERENCE, BASELINE, *candidates])]]
    boosting = load(BOOSTING)
    frames.append(run_variant(boosting, table, days, BOOSTING, boosting.CHOSEN[BOOSTING], jobs)[0])

    first = first_complete_day(table, "B4")
    b2_names = [name for name, settings in candidates.items() if settings["feature_set"] == "B2"]
    same_start = {}
    for name in b2_names:
        same_start[name] = f"{name}_from_{first:%m-%d}"
        later = table[table["delivery_day"] >= first]
        frames.append(run_variant(module, later, days, same_start[name], candidates[name], jobs)[0])

    scores, daily = evaluate(pd.concat(frames, ignore_index=True))
    pairs = [(name, other) for name in candidates for other in (REFERENCE, BASELINE, BOOSTING)]
    for name in candidates:
        if name not in b2_names:
            pairs += [(name, b2) for b2 in b2_names] + [(name, same_start[b2]) for b2 in b2_names]
    return scores, comparison(daily, pairs), surviving_features(coefficients, table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", choices=("tune", "test"), default="test")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    module = load(NAME)
    out = REPORTS_DIR / NAME
    if args.period == "tune":
        candidates = best_per_feature_set(pd.read_csv(out / "tune_scores.csv", index_col=0), module.GRID)
        coefficients = None
    else:
        if not module.CHOSEN:
            raise ValueError("CHOSEN is empty: run tuning and fix the selected settings before testing")
        candidates = module.CHOSEN
        coefficients = pd.read_csv(out / "test_coefficients.csv", parse_dates=["fit_day"])

    saved = pd.read_parquet(out / f"{args.period}_forecasts.parquet")
    scores, compared, features = analyse(build_table(load_inputs()), period_days(args.period), saved, candidates,
                                         coefficients, args.jobs)
    scores.to_csv(out / f"{args.period}_diagnostic_scores.csv")
    compared.to_csv(out / f"{args.period}_comparison.csv", index=False)
    features.to_csv(out / f"{args.period}_surviving_features.csv", index=False)

    pd.set_option("display.width", 180, "display.max_columns", None, "display.max_rows", None)
    columns = ["pinball", "mae_q50", "coverage_80", "width_80", "skill_vs_reference", "n_days"]
    print(scores[[c for c in columns if c in scores.columns]].sort_values("pinball").round(4))
    print(compared.round(4))
    d2 = features[features["block"].isin(["price_level", "system_balance", "cloud_cover"])]
    print(d2.pivot_table(index=["model", "feature"], columns="quantile", values="share_of_fits").round(2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
