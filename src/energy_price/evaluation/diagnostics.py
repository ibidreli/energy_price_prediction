"""Out-of-sample importance and score breakdowns, shared by all model experiments."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from energy_price.evaluation.backtest import common_rows, folds, training_rows
from energy_price.evaluation.scores import score
from energy_price.models.shared.base import QuantileModel
from energy_price.models.shared.importance import grouped_permutation_importance


def _importance_fold(
    table: pd.DataFrame,
    block: list[pd.Timestamp],
    make_model: Callable[[], QuantileModel],
    groups: Mapping[str, Sequence[str]],
    window_days: int | None,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    model = make_model().fit(training_rows(table, block[0], window_days))
    validation = table[table["delivery_day"].isin(block)]
    return grouped_permutation_importance(model, validation, groups, n_repeats, random_state).assign(
        fit_day=block[0], validation_end=block[-1], n_train=getattr(model, "n_train_", np.nan))


def backtest_importance(
    table: pd.DataFrame,
    days: list[pd.Timestamp],
    make_model: Callable[[], QuantileModel],
    groups: Mapping[str, Sequence[str]],
    window_days: int | None = None,
    n_repeats: int = 5,
    random_state: int = 0,
    jobs: int = -1,
) -> pd.DataFrame:
    """Fit on history up to D-2, explain only the following validation block.

    Each weekly model is explained separately; validation targets are never passed to fit.
    """
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs)(
            delayed(_importance_fold)(table, block, make_model, groups, window_days, n_repeats, random_state)
            for block in folds(days)
        )
    return pd.concat(results, ignore_index=True)


def importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    """Row-weighted mean importance; spread across folds/repeats is descriptive, not a CI."""
    records = []
    for group, rows in importance.groupby("group", sort=False):
        weights = rows["n_quarter_hours"].to_numpy()
        mean = np.average(rows["importance"], weights=weights)
        std = np.sqrt(np.average((rows["importance"] - mean) ** 2, weights=weights))
        records.append({"group": group, "importance_mean": mean, "importance_std": std,
                        "n_measurements": len(rows)})
    return pd.DataFrame(records).sort_values("importance_mean", ascending=False, ignore_index=True)


def breakdown_scores(forecasts: pd.DataFrame, balance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scores by hour, month, day type and actual direction, on the same rows for every model.

    The actual TSI is joined after prediction, exclusively for evaluation. It never enters
    the model input table or the training/importance feature groups.
    """
    rows = common_rows(forecasts).copy()
    rows["month"] = rows["delivery_day"].dt.month
    rows["day_type"] = np.where(rows["holiday_pv_share"] > 0, "holiday",
                                np.where(rows["weekend"] > 0, "weekend", "weekday"))
    dimensions = ["hour", "month", "day_type"]
    if balance is not None:
        rows = rows.merge(balance[["timestamp_utc", "tsi_mw"]], on="timestamp_utc", how="left",
                          validate="many_to_one")
        rows["direction"] = pd.Series(pd.NA, index=rows.index, dtype="string")
        rows.loc[rows["tsi_mw"].notna(), "direction"] = np.where(
            rows.loc[rows["tsi_mw"].notna(), "tsi_mw"] < 0, "short", "long")
        dimensions.append("direction")
    records = []
    for dimension in dimensions:
        for (model, value), subset in rows.groupby(["model", dimension]):
            records.append({"model": model, "dimension": dimension, "value": str(value),
                            **score(subset), "n_days": subset["delivery_day"].nunique()})
    return pd.DataFrame(records)
