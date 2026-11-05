"""Asymmetric rolling calibration from strictly out-of-sample weekly forecasts.

The outer backtest supplies only observed prices up to its weekly D-2 cutoff.
Inner weekly forecasts have their own D-2 cutoff; no calibration target is
used to fit its forecaster. Corrections stay fixed until the next outer refit.
Time dependence means this is empirical calibration, not an IID coverage guarantee.
"""

from importlib import import_module
from importlib.metadata import version
from hashlib import sha256
import math

from joblib import Memory
import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import training_rows
from energy_price.features.dataset import FIRST_DAY, TARGET
from energy_price.paths import REPORTS_DIR, ROOT

# Joblib hashes the cached function, but not the implementation of a model class
# supplied as an argument. Namespace by all package sources and numerical versions
# so changes in a base model, its features or its solver cannot reuse old results.
_sources = sorted((ROOT / "src" / "energy_price").rglob("*.py"))
_digest = sha256(b"".join(path.read_bytes() for path in _sources)
                 + repr(tuple(version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib"))).encode())
_memory = Memory(REPORTS_DIR / "calibrated_bands" / "cache" / _digest.hexdigest(), verbose=0)


def _forecast(base: str, train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Content-addressed cache, including data and the frozen base configuration."""
    module_name, variant = base.split(":")
    module = import_module(f"energy_price.models.{module_name}.model")
    settings = dict(module.CHOSEN[variant])
    window = settings.pop("window", "expanding")
    if window != "expanding":
        raise ValueError("calibration currently requires an expanding base model")
    return _cached_forecast(module.MODEL, settings, train, target)


@_memory.cache
def _cached_forecast(model, settings, train, target):
    return model(**settings).fit(train).predict(target)


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

    ``base`` identifies a model folder and its CHOSEN variant. Historical
    forecasters use at least 28 days of warm-up and weekly expanding fits.
    Sparse hour groups fall back to pooled residuals. P50 is never changed.
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
        last = table["delivery_day"].max()
        first = last - pd.Timedelta(days=self.calibration_days - 1)
        frames = []
        for day in pd.date_range(FIRST_DAY + pd.Timedelta(days=29), last, freq="7D"):
            end = day + pd.Timedelta(days=6)
            if end < first:
                continue
            target = table[table["delivery_day"].between(day, end)]
            if target.empty:
                continue
            train = training_rows(table, day, None)
            pred = _forecast(self.base, train, target.drop(columns=[TARGET]))
            rows = target[["delivery_day", "hour", TARGET]].join(pred)
            rows["fit_day"] = day
            rows["train_last_day"] = train["delivery_day"].max()
            frames.append(rows)
        if not frames:
            raise ValueError("no out-of-sample calibration history after warm-up")
        history = pd.concat(frames)
        history = history[history["delivery_day"] >= first].dropna(subset=[TARGET, "q10", "q90"])
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
        out = _forecast(self.base, self.train_, table.drop(columns=[TARGET], errors="ignore")).copy()
        if self.calibrated:
            corrections = [self.corrections_.get(g, self.pooled_) for g in self._groups(table)]
            correction = np.asarray(corrections).reshape(-1, 2)
            out["q10"] = np.minimum(out["q10"] - correction[:, 0], out["q50"])
            out["q90"] = np.maximum(out["q90"] + correction[:, 1], out["q50"])
        return out
