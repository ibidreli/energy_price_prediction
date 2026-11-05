"""Grouped permutation importance for any model with the shared quantile interface."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from energy_price.evaluation.scores import score
from energy_price.features.dataset import TARGET
from energy_price.models.shared.base import QuantileModel


def grouped_permutation_importance(
    model: QuantileModel,
    table: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Increase in validation pinball loss when a source's raw inputs are shuffled together.

    The model is already fitted on earlier data; this function never fits or changes it.
    Derived interactions are recomputed by ``predict``. All groups use the same complete
    validation rows and permutations. Results measure model reliance, not causality;
    correlated sources and interactions mean importances need not add up.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    required = {column for columns in groups.values() for column in columns}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"importance columns missing: {sorted(missing)}")
    if not groups or any(not columns for columns in groups.values()):
        raise ValueError("importance groups must contain at least one column each")

    prediction = model.predict(table)
    complete = prediction.notna().all(axis=1) & table[TARGET].notna()
    if "delivery_day" in table:
        # Match the backtest: an incomplete weather day is excluded for every model.
        complete &= complete.groupby(table["delivery_day"]).transform("all")
    validation = table.loc[complete].copy()
    if validation.empty:
        raise ValueError("no complete validation rows for permutation importance")
    baseline = score(pd.concat([validation, prediction.loc[complete]], axis=1))["pinball"]
    rng = np.random.default_rng(random_state)
    permutations = [rng.permutation(len(validation)) for _ in range(n_repeats)]
    records = []
    for group, columns in groups.items():
        columns = list(columns)
        if TARGET in columns:
            raise ValueError("the target cannot be an importance feature")
        for repeat, permutation in enumerate(permutations):
            shuffled = validation.copy()
            shuffled[columns] = validation.iloc[permutation][columns].to_numpy()
            forecast = model.predict(shuffled)
            loss = score(pd.concat([validation, forecast], axis=1))["pinball"]
            records.append({"group": group, "repeat": repeat, "baseline_pinball": baseline,
                            "permuted_pinball": loss, "importance": loss - baseline,
                            "n_quarter_hours": len(validation)})
    return pd.DataFrame(records)
