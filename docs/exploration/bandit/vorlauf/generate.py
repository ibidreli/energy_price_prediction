"""Forecasts of existing models for tune and test (never the holdout), saved for the aggregation check."""
import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import pandas as pd

import energy_price
from energy_price.evaluation.backtest import run_model, run_reference, period_days
from energy_price.features.dataset import TARGET, build_table, load_inputs

out = Path(sys.argv[1])
jobs = int(sys.argv[2])
wanted = sys.argv[3:]
print("package:", energy_price.__file__, flush=True)
table = build_table(load_inputs())
print(list(table.columns), flush=True)
WINDOWS = {"expanding": None, "rolling_56d": 56}
SPEC = {
    "B1": ("baseline_model", "B1"),
    "B2": ("baseline_model", "B2"),
    "gbm": ("gbm_quantile", "gbm_quantile"),
    "regime": ("regime_mixture", "regime_mixture"),
}
keep = [c for c in ("timestamp_local", "timestamp_utc", "delivery_day", TARGET, "q10", "q50", "q90", "model")]
for period in ("tune", "test"):
    days = period_days(period)
    for name in wanted:
        path = out / f"{period}_{name}.parquet"
        if name == "reference":
            frame = run_reference(table, days)
        else:
            folder, variant = SPEC[name]
            module = import_module(f"energy_price.models.{folder}.model")
            params = dict(module.CHOSEN[variant])
            window = WINDOWS[params.pop("window", "expanding")]
            frame, _ = run_model(table, days, partial(module.MODEL, **params), name, window, jobs)
        frame = frame.assign(model=name)[[c for c in keep if c in frame.columns or c == "model"]]
        frame.to_parquet(path, index=False)
        print("wrote", path, len(frame), flush=True)
