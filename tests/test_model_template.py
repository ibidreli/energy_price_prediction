"""Tests of the model template. Optional: copy to tests/test_<name>.py for a new model.

After copying: set NAME to the folder name and add tests for new features, above all that they
are known at D-1, 11:00 (tests/test_features_dataset.py shows how).
"""

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import evaluate, run_reference
from energy_price.features.dataset import calendar_features, quarter_hours
from energy_price.evaluation.experiment import load, run_variant
from energy_price.models.shared.base import QuantileModel

NAME = "_template"


def synthetic_table(first: str = "2026-04-01", days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range(first, periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    table["aep_ct_kwh"] = 10 + table["hour"] + rng.normal(0, 2, len(table))
    return table


def test_model_fulfils_the_interface_of_all_models():
    table = synthetic_table()

    module = load(NAME)
    variant, settings = next(iter(module.CHOSEN.items()))
    params = {k: v for k, v in settings.items() if k != "window"}
    model = module.MODEL(**params).fit(table)
    out = model.predict(table)

    assert isinstance(model, QuantileModel)
    assert list(out.columns) == ["q10", "q50", "q90"]
    assert out.index.equals(table.index)
    assert ((out["q10"] <= out["q50"]) & (out["q50"] <= out["q90"])).all()


def test_backtest_runs_end_to_end_against_the_reference():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-01", "2026-05-08", freq="D"))

    module = load(NAME)
    variant, settings = next(iter(module.CHOSEN.items()))

    forecasts, _ = run_variant(module, table, days, variant, settings, jobs=1)
    scores, _ = evaluate(pd.concat([run_reference(table, days), forecasts], ignore_index=True))

    assert set(scores.index) == {"reference", variant}
    assert scores.loc[variant, "n_days"] == len(days)
