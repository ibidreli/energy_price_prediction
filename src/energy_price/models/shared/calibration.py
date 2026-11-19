"""Asymmetric rolling calibration from strictly out-of-sample forecasts of a frozen base model.

The base is a CHOSEN variant of another model folder (``"baseline_model:B2"``) and runs exactly as in
its own walk-forward backtest: its training window (``window`` in CHOSEN, ``window_mean`` = mean of the
per-window forecasts over its ``WINDOWS``) and its refit frequency (``REFIT_EVERY``, else the design
default). The outer backtest supplies only rows with a price known at its issue time (up to D-2).
Inside ``fit`` the base forecasts the calibration days fold by fold (``folds(days, refit_every)``);
each fold's base sees only ``training_rows(table, fit_day, window)``, so no calibration target is used
to fit its forecaster. ``predict`` refits the base on the rows given to ``fit`` with fit day = last
training day + 2. Corrections stay fixed until the next outer refit. Time dependence means this is
empirical calibration, not an IID coverage guarantee.
"""

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import math

from joblib import Memory
import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import KEYS, QUANTILE_COLUMNS, folds, training_rows, window_mean
from energy_price.evaluation.design import DESIGN, WINDOW_MEAN, window_days
from energy_price.evaluation.experiment import refit_every_of, windows_of
from energy_price.features.dataset import TARGET
from energy_price.paths import REPORTS_DIR, ROOT

# Days of data before the first historical base forecast (fit day >= first training day + 29).
WARMUP_DAYS = 28

# Joblib hashes the cached function, but not the implementation of a model class
# supplied as an argument. Namespace by all package sources and numerical versions
# so changes in a base model, its features or its solver cannot reuse old results.
_sources = sorted((ROOT / "src" / "energy_price").rglob("*.py"))
_digest = sha256(b"".join(path.read_bytes() for path in _sources)
                 + repr(tuple(version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib"))).encode())
_memory = Memory(REPORTS_DIR / "calibrated_bands" / "cache" / _digest.hexdigest(), verbose=0)


@dataclass(frozen=True)
class Base:
    """A frozen base variant: model class, settings without window, its window(s) and refit frequency."""

    model: type
    settings: dict
    window: str                  # CHOSEN window, may be ``window_mean``
    windows: tuple[str, ...]     # the single window, or the windows averaged by ``window_mean``
    refit_every: int


def _base(name: str) -> Base:
    """The CHOSEN variant ``"<model folder>:<variant>"`` with its window and refit frequency from model.py."""
    module_name, variant = name.split(":")
    module = import_module(f"energy_price.models.{module_name}.model")
    settings = dict(module.CHOSEN[variant])
    window = settings.pop("window", None)
    if window is None:
        raise ValueError(f"CHOSEN {name} has no 'window'")
    if window == WINDOW_MEAN:
        windows = windows_of(module)
    else:
        window_days(window)  # rejects unknown names
        windows = (window,)
    return Base(module.MODEL, settings, window, tuple(windows), refit_every_of(module))


def _fit_predict(model, settings, window, refit_every, train, target):
    """Fit one base model and forecast ``target``. ``window`` and ``refit_every`` are part of the cache key only."""
    del window, refit_every
    return model(**settings).fit(train).predict(target)


# Content-addressed: data, base configuration, window and refit frequency.
_cached_forecast = _memory.cache(_fit_predict)


def _forecast(base: Base, window: str, train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    return _cached_forecast(base.model, base.settings, window, base.refit_every, train, target)


def _mean_over_windows(target: pd.DataFrame, forecasts: list[pd.DataFrame]) -> pd.DataFrame:
    """``backtest.window_mean`` of per-window base forecasts, back on the index of ``target``."""
    keys = target[KEYS]
    frames = [keys.join(forecast[QUANTILE_COLUMNS]).assign(model=str(i)) for i, forecast in enumerate(forecasts)]
    mean = window_mean(frames).set_index(KEYS)[QUANTILE_COLUMNS]
    values = mean.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    return pd.DataFrame(values, index=target.index, columns=QUANTILE_COLUMNS)


def _base_forecast(base: Base, table: pd.DataFrame, fit_day: pd.Timestamp,
                   target: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Base forecast of ``target`` fitted at ``fit_day`` on its window(s), and the last training day."""
    forecasts = []
    for window in base.windows:
        train = training_rows(table, fit_day, window_days(window))
        forecasts.append(_forecast(base, window, train, target))
    out = forecasts[0] if len(forecasts) == 1 else _mean_over_windows(target, forecasts)
    return out, train["delivery_day"].max()  # the window limits the start only, the end is D-2 for all


def upper_quantile(values: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample upper order statistic at ceil((n+1)*(1-alpha))."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if not len(values) or rank > len(values):
        raise ValueError("insufficient calibration residuals for a finite correction")
    return float(np.partition(values, rank - 1)[rank - 1])


class CalibratedQuantileModel:
    """Wrap a frozen named base model; independently correct each 10% tail.

    ``base`` identifies a model folder and its CHOSEN variant. Historical base forecasts cover the last
    ``calibration_days`` training days (after ``WARMUP_DAYS`` of data) with the base's own window and
    refit folds. Sparse hour groups fall back to pooled residuals. P50 is never changed.
    """

    def __init__(self, base="baseline_model:B2", calibration_days=28, grouping="global", calibrated=True):
        if calibration_days not in (28, 56):
            raise ValueError("calibration_days must be 28 or 56")
        if grouping not in ("global", "6h", "hour"):
            raise ValueError("grouping must be global, 6h or hour")
        self.base = base
        self.calibration_days = calibration_days
        self.grouping = grouping
        self.calibrated = calibrated

    def _groups(self, table):
        if self.grouping == "global":
            return pd.Series(0, index=table.index)
        return table["hour"] // (6 if self.grouping == "6h" else 1)

    def fit(self, table):
        if table.empty:
            raise ValueError("no training rows")
        self.train_ = table.copy()
        self.corrections_ = {}
        if not self.calibrated:
            return self
        base = _base(self.base)
        last = table["delivery_day"].max()
        first = last - pd.Timedelta(days=self.calibration_days - 1)
        start = max(first, DESIGN.first_train_day + pd.Timedelta(days=WARMUP_DAYS + 1))
        frames = []
        for block in folds(list(pd.date_range(start, last, freq="D")), base.refit_every):
            target = table[table["delivery_day"].isin(block)]
            if target.empty:
                continue
            pred, train_last = _base_forecast(base, table, block[0], target.drop(columns=[TARGET]))
            rows = target[["delivery_day", "hour", TARGET]].join(pred)
            rows["fit_day"] = block[0]
            rows["train_last_day"] = train_last
            frames.append(rows)
        if not frames:
            raise ValueError("no out-of-sample calibration history after warm-up")
        history = pd.concat(frames).dropna(subset=[TARGET, "q10", "q90"])
        history = history[np.isfinite(history[[TARGET, "q10", "q90"]]).all(axis=1)]
        self.calibration_ = history.copy()
        lower = history["q10"] - history[TARGET]
        upper = history[TARGET] - history["q90"]
        self.pooled_ = (upper_quantile(lower), upper_quantile(upper))
        groups = self._groups(history)
        for group in groups.unique():
            mask = groups == group
            if mask.sum() >= 20:
                self.corrections_[group] = (upper_quantile(lower[mask]), upper_quantile(upper[mask]))
        return self

    def predict(self, table):
        if not hasattr(self, "train_"):
            raise RuntimeError("fit the model first")
        if table.empty:
            return pd.DataFrame(index=table.index, columns=["q10", "q50", "q90"], dtype=float)
        fit_day = self.train_["delivery_day"].max() + pd.Timedelta(days=2)
        out, _ = _base_forecast(_base(self.base), self.train_, fit_day,
                                table.drop(columns=[TARGET], errors="ignore"))
        out = out.copy()
        if self.calibrated:
            corrections = [self.corrections_.get(g, self.pooled_) for g in self._groups(table)]
            correction = np.asarray(corrections).reshape(-1, 2)
            out["q10"] = np.minimum(out["q10"] - correction[:, 0], out["q50"])
            out["q90"] = np.maximum(out["q90"] + correction[:, 1], out["q50"])
        return out
