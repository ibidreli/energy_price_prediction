"""Configurable, content-addressed out-of-sample expert forecasts.

The cache keys contain the actual fit/predict inputs, frozen settings, numerical
versions and expert/feature source code. No report from a different experiment
is trusted just because its model name matches.
"""

from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json

from joblib import Memory, Parallel, delayed, parallel_config
import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import folds, training_rows
from energy_price.features.dataset import TARGET, DIRECTION, quarter_hours
from energy_price.models.shared.reference import reference_forecast
from energy_price.paths import REPORTS_DIR, ROOT

Q = ["q10", "q50", "q90"]
MEMORY = Memory(REPORTS_DIR / "forecast_pool" / "cache", verbose=0)


def pool_fingerprint(pool):
    paths = set((ROOT / "src/energy_price/features").glob("*.py"))
    paths.update((ROOT / "src/energy_price/models/shared").glob("*.py"))
    # Routing and calibration do not change the raw experts.
    paths = {p for p in paths if p.stem not in {"forecast_pool", "routing", "calibration"}}
    paths.update(ROOT / "src/energy_price" / p for p in ("availability.py", "evaluation/backtest.py"))
    for spec in pool.values():
        if spec.get("module"):
            paths.update((ROOT / "src/energy_price/models" / spec["module"]).glob("*.py"))
    payload = json.dumps(pool, sort_keys=True).encode() + b"".join(p.read_bytes() for p in sorted(paths))
    payload += repr([version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")]).encode()
    return sha256(payload).hexdigest()


@MEMORY.cache
def _predict(module_name, settings, train, target, source_version):
    module = import_module(f"energy_price.models.{module_name}.model")
    return module.MODEL(**settings).fit(train).predict(target)[Q]


def _fold(table, block, name, spec, fingerprint):
    settings = dict(spec.get("settings", {}))
    window = settings.pop("window", "expanding")
    if window not in ("expanding", "rolling_56d"):
        raise ValueError(f"unsupported training window {window}")
    target = table[table.delivery_day.isin(block)].copy()
    train = training_rows(table, block[0], None if window == "expanding" else 56)
    # Targets and actual direction are never passed to predict, including wrappers.
    prediction = _predict(spec["module"], settings, train,
                          target.drop(columns=[TARGET, DIRECTION], errors="ignore"), fingerprint)
    target[Q] = prediction
    return target.assign(model=name, fit_day=block[0], train_last_day=train.delivery_day.max())


def generate_pool(table, days, pool, jobs=2):
    """Run unchanged experts in the shared weekly folds; reference stays daily.

    Call separately for warm-up, tune and test, preserving their established
    weekly anchors. The returned long table retains missing predictions.
    """
    if not pool or any(not name or ":" in name for name in pool):
        raise ValueError("pool needs unique nonempty names without ':'")
    frames, tasks = [], []
    for name, spec in pool.items():
        if spec.get("kind") == "reference":
            for day in days:
                target = table[table.delivery_day == day].copy()
                train = training_rows(table, day, spec.get("window_days", 56))
                target[Q] = reference_forecast(train, target, day.date(), spec.get("window_days", 56))
                frames.append(target.assign(model=name, fit_day=day, train_last_day=train.delivery_day.max()))
        else:
            # An added expert must not force refitting unchanged members.
            fingerprint = pool_fingerprint({name: spec})
            tasks.extend(delayed(_fold)(table, block, name, spec, fingerprint) for block in folds(days))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        frames.extend(Parallel(n_jobs=jobs)(tasks))
    out = pd.concat(frames, ignore_index=True)
    validate_pool(out, list(pool))
    return out


def validate_pool(frame, names):
    """Fail on missing experts/quarters, contradictory targets, leakage or crossing.

    Explicit NaN forecasts are allowed (missing weather), but absent rows are not.
    UTC keys retain the repeated autumn DST hour.
    """
    if set(frame.model.unique()) != set(names):
        raise ValueError("pool membership differs from configuration")
    if frame.duplicated(["model", "timestamp_utc"]).any():
        raise ValueError("duplicate expert quarter hour")
    if frame[["fit_day", "train_last_day", "delivery_day"]].isna().any().any():
        raise ValueError("missing forecast provenance")
    if (frame.train_last_day > frame.fit_day - pd.Timedelta(days=2)).any():
        raise ValueError("D-2 violation in expert training")
    if (frame.fit_day > frame.delivery_day).any():
        raise ValueError("forecast fitted after delivery")
    for name in names:
        member = frame[frame.model == name]
        expected = quarter_hours(sorted(frame.delivery_day.unique()))
        if set(member.timestamp_utc) != set(expected.timestamp_utc):
            raise ValueError(f"missing quarter hours in {name}")
    if (frame.groupby("timestamp_utc")[TARGET].nunique(dropna=False) > 1).any():
        raise ValueError("experts have different targets")
    complete = frame[Q].notna().all(axis=1)
    values = frame.loc[complete, Q].to_numpy()
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError("invalid expert quantiles")


def wide_pool(frame, names):
    validate_pool(frame, names)
    meta = [c for c in ("timestamp_utc", "timestamp_local", "delivery_day", "hour", "weekend", TARGET, DIRECTION)
            if c in frame]
    out = frame[frame.model == names[0]][meta].set_index("timestamp_utc").sort_index()
    for name in names:
        member = frame[frame.model == name].set_index("timestamp_utc")
        for q in Q:
            out[f"{name}:{q}"] = member[q]
    cols = [f"{name}:{q}" for name in names for q in Q]
    # Exclude an incomplete day for everyone, including training/calibration.
    valid = np.isfinite(out[cols + [TARGET]]).all(axis=1)
    good = valid.groupby(out.delivery_day).all()
    return out[out.delivery_day.isin(good.index[good])].reset_index()
