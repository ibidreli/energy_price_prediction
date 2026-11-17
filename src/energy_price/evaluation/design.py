"""Backtest design shared by all models: training start, validation blocks, locked test period, windows.

Every delivery day D is an origin: the forecast is issued at D-1, 11:00 and a model is trained on the
rows known then (prices and system balance up to the end of D-2, ``availability.known_at_issue``).

- January to March are training only.
- Validation: 2026-04-01 to 2026-08-31, reported per calendar month (``block_of``). The settings of a
  model are selected on the mean daily pinball loss over all validation days.
- Test period: 2026-09-01 to 2026-10-31, locked (``test_days`` raises ``TestLockedError`` unless it is
  released). It is evaluated exactly once after the freeze. It is a "Testperiode mit Vorbehalt": no
  price or forecast of it was ever scored, but the EDA looked at its system balance and weather.
- Refit: every ``refit_every`` days (default 1, daily). A model may set ``REFIT_EVERY`` in ``model.py``.
- Training windows ``WINDOW_DAYS``: ``expanding`` (all rows since ``first_train_day``) or the last w days
  D-1-w to D-2. ``Design.windows`` is the default set of a model; ``rolling_28d`` is only for sparse
  linear models and set by them via ``WINDOWS`` in ``model.py`` (Lago et al. 2021: at least 8 weeks).
- ``WINDOW_MEAN``: the per-row mean of the quantiles over the windows of one setting, an extra candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from energy_price.features.dataset import FIRST_DAY

WINDOW_DAYS: dict[str, int | None] = {"expanding": None, "rolling_28d": 28, "rolling_56d": 56, "rolling_90d": 90}
WINDOW_MEAN = "window_mean"


@dataclass(frozen=True)
class Design:
    """Periods, refit frequency and default training windows of the walk-forward backtest."""

    first_train_day: pd.Timestamp
    validation: tuple[str, str]
    test: tuple[str, str]
    refit_every: int = 1
    windows: tuple[str, ...] = ("expanding", "rolling_56d", "rolling_90d")

    def __post_init__(self) -> None:
        unknown = [w for w in self.windows if w not in WINDOW_DAYS]
        if unknown or not self.windows:
            raise ValueError(f"unknown or no windows {unknown}, expected some of {tuple(WINDOW_DAYS)}")
        if self.refit_every < 1:
            raise ValueError("refit_every must be at least 1")
        first, (v_start, v_end), (t_start, t_end) = self.first_train_day, *map(_days, (self.validation, self.test))
        if not first < v_start <= v_end < t_start <= t_end:
            raise ValueError("expected first_train_day < validation start <= validation end < test start <= test end")

    @property
    def test_start(self) -> pd.Timestamp:
        """First delivery day of the test period; no raw data from this day on enters a development run."""
        return pd.Timestamp(self.test[0])

    def as_config(self) -> dict:
        """Plain values for run info and W&B."""
        out = asdict(self)
        out["first_train_day"] = str(self.first_train_day.date())
        out["validation"], out["test"], out["windows"] = list(self.validation), list(self.test), list(self.windows)
        return out


def _days(period: tuple[str, str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(period[0]), pd.Timestamp(period[1])


DESIGN = Design(first_train_day=FIRST_DAY, validation=("2026-04-01", "2026-08-31"), test=("2026-09-01", "2026-10-31"))


class TestLockedError(RuntimeError):
    """The test period was requested without releasing it."""

    __test__ = False  # not a pytest test class


def validation_days(design: Design = DESIGN) -> list[pd.Timestamp]:
    """All delivery days of the validation period, in order."""
    return list(pd.date_range(*design.validation, freq="D"))


def test_days(design: Design = DESIGN, unlock: bool = False) -> list[pd.Timestamp]:
    """All delivery days of the test period; only with ``unlock=True`` (``final-test --unlock-test``)."""
    if not unlock:
        raise TestLockedError("the test period is locked; it is evaluated once with final-test --unlock-test")
    return list(pd.date_range(*design.test, freq="D"))


test_days.__test__ = False  # not a pytest test function


def block_of(day) -> str:
    """Validation block of a delivery day: its calendar month, e.g. ``"2026-04"``."""
    return pd.Timestamp(day).strftime("%Y-%m")


def window_days(window: str) -> int | None:
    """Length of a training window in days, None for ``expanding``."""
    if window not in WINDOW_DAYS:
        raise ValueError(f"unknown window {window!r}, expected one of {tuple(WINDOW_DAYS)}")
    return WINDOW_DAYS[window]
