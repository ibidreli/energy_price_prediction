"""Design of the walk-forward backtest: periods, test lock, blocks and windows."""

import pandas as pd
import pytest

from energy_price.evaluation import design as d
from energy_price.features.dataset import FIRST_DAY


def test_development_and_test_periods_follow_the_decision():
    assert d.DESIGN.first_train_day == FIRST_DAY == pd.Timestamp("2026-01-01")
    assert d.DESIGN.validation == ("2026-04-01", "2026-08-31")
    assert d.DESIGN.test == ("2026-09-01", "2026-10-31")
    assert d.DESIGN.refit_every == 1
    assert d.DESIGN.windows == ("expanding", "rolling_56d", "rolling_90d")
    assert d.WINDOW_DAYS == {"expanding": None, "rolling_28d": 28, "rolling_56d": 56, "rolling_90d": 90}


def test_validation_days_cover_april_to_august_in_five_blocks():
    days = d.validation_days()

    assert days[0] == pd.Timestamp("2026-04-01") and days[-1] == pd.Timestamp("2026-08-31")
    assert len(days) == 153
    assert sorted({d.block_of(day) for day in days}) == ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
    assert max(days) < d.DESIGN.test_start


def test_test_period_is_locked_unless_released():
    with pytest.raises(d.TestLockedError):
        d.test_days()

    days = d.test_days(unlock=True)

    assert days[0] == pd.Timestamp("2026-09-01") and days[-1] == pd.Timestamp("2026-10-31")


@pytest.mark.parametrize("changes", [
    {"windows": ("rolling_7d",)},
    {"windows": ()},
    {"refit_every": 0},
    {"validation": ("2026-04-01", "2026-09-15")},  # overlaps the test period
])
def test_inconsistent_designs_are_rejected(changes):
    settings = {"first_train_day": FIRST_DAY, "validation": d.DESIGN.validation, "test": d.DESIGN.test, **changes}

    with pytest.raises(ValueError):
        d.Design(**settings)


def test_window_days_rejects_unknown_windows():
    assert d.window_days("rolling_90d") == 90
    assert d.window_days("expanding") is None
    with pytest.raises(ValueError, match="unknown window"):
        d.window_days(d.WINDOW_MEAN)


def test_design_config_is_plain_json():
    import json

    config = d.DESIGN.as_config()

    assert json.loads(json.dumps(config)) == config
    assert config["first_train_day"] == "2026-01-01" and config["refit_every"] == 1
