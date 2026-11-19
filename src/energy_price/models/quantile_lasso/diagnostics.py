"""Do data until D-2 help? Comparison of the quantile LASSO with other models and its surviving features.

``select`` compares the candidates with the reference and the baseline B2 only. This script adds what
issue #27 asks for, on the same quarter hours for all models and on the days of the last ``select`` run
(validation days only):

- ``gbm_quantile`` as a further comparison, fitted here with its unchanged ``CHOSEN``,
- the daily pinball difference of every pair with a 95 % block bootstrap interval,
- the share of fits in which a feature keeps a coefficient, per quantile. This describes what the
  model uses, not what causes the price,
- B2 trained only from the first day with B3 and B4 features. B3 and B4 have fewer training rows than
  B2 (minimum history, docs/data.md), so a difference to B2 also measures the data. This variant
  separates the two. It is only reported and never selected.

Nothing is selected here. ``--candidates best`` (default) analyses the select candidate with the lowest
mean daily pinball loss of each feature set (``select_scores.csv``, block ``all``), ``--candidates chosen``
the settings of ``CHOSEN``. Forecasts and coefficients come from ``select_forecasts.parquet`` and
``select_coefficients.csv``; the further models are fitted with the refit frequency of that run.

Usage (repository root, after ``make select MODEL=quantile_lasso``)::

    python -m energy_price.models.quantile_lasso.diagnostics
    python -m energy_price.models.quantile_lasso.diagnostics --candidates chosen

Writes ``diagnostics_<candidates>_scores.csv``, ``_comparison.csv`` and ``_surviving_features.csv`` to
``reports/quantile_lasso/``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import BASELINE, REFERENCE, evaluate
from energy_price.evaluation.design import WINDOW_MEAN
from energy_price.evaluation.experiment import check_days, development_table, label, load, run_variant, windows_of
from energy_price.evaluation.scores import BOOTSTRAP_BLOCK_DAYS, COLUMNS, block_bootstrap
from energy_price.features.design import BLOCKS, FEATURE_SETS, design_matrix
from energy_price.paths import REPORTS_DIR

NAME = "quantile_lasso"
BOOSTING = "gbm_quantile"
NONZERO = 1e-9  # the L1 penalty sets coefficients to zero up to the precision of the solver
QUANTILE_COLUMNS = list(COLUMNS.values())


def candidates_of(grid: list[dict], windows: tuple[str, ...]) -> dict[str, dict]:
    """The candidates of ``select`` by their name: every GRID setting with every window, and the window mean."""
    out = {}
    for settings in grid:
        for window in windows:
            out[label({**settings, "window": window})] = {**settings, "window": window}
        if len(windows) > 1:
            out[label({**settings, "window": WINDOW_MEAN})] = {**settings, "window": WINDOW_MEAN}
    return out


def best_per_feature_set(scores: pd.DataFrame, candidates: dict[str, dict]) -> dict[str, dict]:
    """The candidate with the lowest mean daily pinball loss of each feature set, by its name in ``scores``.

    ``scores`` holds one row per model over all validation days (``select_scores.csv``, block ``all``).
    """
    best = {}
    for feature_set in dict.fromkeys(settings["feature_set"] for settings in candidates.values()):
        named = [name for name, settings in candidates.items() if settings["feature_set"] == feature_set]
        name = scores.loc[named, "mean_daily_pinball"].idxmin()
        best[name] = candidates[name]
    return best


def chosen_candidates(chosen: dict[str, dict], candidates: dict[str, dict]) -> dict[str, dict]:
    """The select candidate of every ``CHOSEN`` setting, by its name in the select results."""
    out = {}
    for variant, settings in chosen.items():
        names = [name for name, candidate in candidates.items() if candidate == settings]
        if not names:
            raise ValueError(f"CHOSEN {variant} {settings} is not a candidate of the select run; rerun select")
        out[names[0]] = settings
    return out


def candidate_coefficients(coefficients: pd.DataFrame, candidates: dict[str, dict],
                           windows: tuple[str, ...]) -> pd.DataFrame:
    """Coefficients of every fit of ``candidates`` from ``select_coefficients.csv``, ``model`` = candidate name.

    A window mean has no fit of its own: it gets the fits of each of its ``windows``.
    """
    frames = []
    for name, settings in candidates.items():
        params = {k: v for k, v in settings.items() if k != "window"}
        fitted = windows if settings["window"] == WINDOW_MEAN else (settings["window"],)
        parts = [coefficients[coefficients["model"] == label({**params, "window": w})] for w in fitted]
        if any(part.empty for part in parts):
            raise ValueError(f"coefficients of {name} are missing in the select results; rerun select")
        frames.append(pd.concat(parts).assign(model=name))
    return pd.concat(frames, ignore_index=True)


def first_complete_day(table: pd.DataFrame, feature_set: str) -> pd.Timestamp:
    """First delivery day with a value in every feature of ``feature_set``."""
    return table.loc[design_matrix(table, feature_set).notna().all(axis=1), "delivery_day"].min()


def comparison(daily: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Daily pinball difference ``model`` minus ``against`` with its 95 % interval; below 0 ``model`` is better.

    The interval needs at least one bootstrap block of 7 days; with fewer days (a smoke run) it is NaN.
    """
    rows = []
    for model, against in pairs:
        diff = (daily[model] - daily[against]).to_numpy()
        low, high = block_bootstrap(diff) if len(diff) >= BOOTSTRAP_BLOCK_DAYS else (np.nan, np.nan)
        rows.append({"model": model, "against": against, "pinball_model": daily[model].mean(),
                     "pinball_against": daily[against].mean(), "mean_difference": diff.mean(),
                     "low": low, "high": high, "entirely_below_zero": high < 0, "n_days": len(diff)})
    return pd.DataFrame(rows)


def surviving_features(coefficients: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per model, feature and quantile: share of fits with a coefficient, and the one of the last fit."""
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
    """A fresh fit must give the forecast of select, otherwise code or data have changed in between."""
    old = saved[saved["model"] == name].set_index("timestamp_utc")[QUANTILE_COLUMNS]
    new = fresh.set_index("timestamp_utc")[QUANTILE_COLUMNS].reindex(old.index)
    if old.empty or not np.allclose(old.to_numpy(), new.to_numpy(), atol=1e-9, equal_nan=True):
        raise ValueError(f"the fresh fit of {name} differs from the saved forecast; rerun select")


def analyse(table: pd.DataFrame, days: list[pd.Timestamp], saved: pd.DataFrame, candidates: dict[str, dict],
            coefficients: pd.DataFrame | None = None, jobs: int = -1,
            refit_every: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scores, pairwise comparison and surviving features of ``candidates``, on the common quarter hours.

    ``saved`` holds the forecasts of select. Without ``coefficients`` the candidates are fitted again
    and must reproduce ``saved``. Every fit here refits every ``refit_every`` days (None: the model's own).
    """
    if BASELINE not in set(saved["model"]):
        raise ValueError(f"forecasts of the baseline ({BASELINE}) are missing; a comparison without it would be "
                         "incomplete, run the baseline on this period first")
    module = load(NAME)
    if coefficients is None:
        fits = []
        for name, settings in candidates.items():
            fresh, coef = run_variant(module, table, days, name, settings, jobs, refit_every)
            _check_matches_saved(fresh, saved, name)
            fits.append(coef.assign(model=name))
        coefficients = pd.concat(fits, ignore_index=True)

    frames = [saved[saved["model"].isin([REFERENCE, BASELINE, *candidates])]]
    boosting = load(BOOSTING)
    frames.append(run_variant(boosting, table, days, BOOSTING, boosting.CHOSEN[BOOSTING], jobs, refit_every)[0])

    first = first_complete_day(table, "B4")
    b2_names = [name for name, settings in candidates.items() if settings["feature_set"] == "B2"]
    same_start = {}
    for name in b2_names:
        same_start[name] = f"{name}_from_{first:%m-%d}"
        later = table[table["delivery_day"] >= first]
        frames.append(run_variant(module, later, days, same_start[name], candidates[name], jobs, refit_every)[0])

    scores, daily = evaluate(pd.concat(frames, ignore_index=True))
    pairs = [(name, other) for name in candidates for other in (REFERENCE, BASELINE, BOOSTING)]
    for name in candidates:
        if name not in b2_names:
            pairs += [(name, b2) for b2 in b2_names] + [(name, same_start[b2]) for b2 in b2_names]
    return scores, comparison(daily, pairs), surviving_features(coefficients, table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", choices=("best", "chosen"), default="best",
                        help="best: lowest mean daily pinball per feature set in select; chosen: CHOSEN")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    module = load(NAME)
    out = REPORTS_DIR / NAME
    selection_path = out / "select_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} is missing: run make select MODEL={NAME} first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    windows = windows_of(module)
    candidates = candidates_of(module.GRID, windows)
    if args.candidates == "best":
        scores = pd.read_csv(out / "select_scores.csv")
        analysed = best_per_feature_set(scores[scores["block"] == "all"].set_index("model"), candidates)
    else:
        analysed = chosen_candidates(module.CHOSEN, candidates)
    coefficients = candidate_coefficients(pd.read_csv(out / "select_coefficients.csv", parse_dates=["fit_day"]),
                                          analysed, windows)

    days = check_days(tuple(selection["period"]))
    saved = pd.read_parquet(out / "select_forecasts.parquet")
    scores, compared, features = analyse(development_table(), days, saved, analysed, coefficients, args.jobs,
                                         selection["refit_every"])
    prefix = f"diagnostics_{args.candidates}"
    scores.to_csv(out / f"{prefix}_scores.csv")
    compared.to_csv(out / f"{prefix}_comparison.csv", index=False)
    features.to_csv(out / f"{prefix}_surviving_features.csv", index=False)

    pd.set_option("display.width", 180, "display.max_columns", None, "display.max_rows", None)
    columns = ["mean_daily_pinball", "pinball", "mae_q50", "coverage_80", "width_80", "skill_vs_reference", "n_days"]
    print(scores[[c for c in columns if c in scores.columns]].sort_values("mean_daily_pinball").round(4))
    print(compared.round(4))
    d2 = features[features["block"].isin(["price_level", "system_balance", "cloud_cover"])]
    print(d2.pivot_table(index=["model", "feature"], columns="quantile", values="share_of_fits").round(2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
