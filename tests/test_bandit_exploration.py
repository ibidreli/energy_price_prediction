"""Bandit and online aggregation: no feedback from the future, and the bandit sees only what it chose."""

import numpy as np
import pandas as pd
import pytest

from energy_price.exploration.bandit import run
from energy_price.exploration.bandit.run import block_bootstrap, main, pinball_losses, to_arrays, tune_selection
from energy_price.exploration.bandit.strategies import GRID, hedge, run_strategy, thompson

DAYS, SLOTS, N_ARMS = 40, 8, 3
SETTINGS = {"eta": 0.3, "half_life": 14.0}


@pytest.fixture
def data():
    """Three arms with sorted quantiles; arm 1 is unbiased, the other two lie clearly above the price."""
    rng = np.random.default_rng(1)
    prices = rng.normal(10, 2, (DAYS, SLOTS))
    centre = np.stack([prices + 4, prices + rng.normal(0, 0.2, prices.shape), prices + 8], axis=2)
    forecasts = np.stack([centre - 2, centre, centre + 2], axis=-1)
    return forecasts, prices, np.ones(DAYS, dtype=bool)


@pytest.mark.parametrize("name", list(GRID))
def test_decision_for_a_day_ignores_prices_of_the_two_days_before(name, data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    day = 20
    changed = losses.copy()
    changed[day - 1:] += 1000.0
    first = run_strategy(name, forecasts, losses, valid, SETTINGS)
    second = run_strategy(name, forecasts, changed, valid, SETTINGS)
    np.testing.assert_array_equal(first[:day + 1], second[:day + 1])


def test_decision_uses_the_prices_of_d_minus_2(data):
    """Positive control for the test above: losses of D-2 do change the forecast of D."""
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    day = 20
    changed = losses.copy()
    changed[day - 2, :, 1] += 50.0
    groups = np.zeros(SLOTS, dtype=int)
    first = hedge(forecasts, losses, valid, groups, **SETTINGS)
    second = hedge(forecasts, changed, valid, groups, **SETTINGS)
    np.testing.assert_array_equal(first[:day], second[:day])
    assert not np.array_equal(first[day], second[day])


def _hide_unchosen(losses: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Losses where every arm the strategy did not choose is changed."""
    changed = losses + 100.0
    days, slots = np.ogrid[:DAYS, :SLOTS]
    changed[days, slots, choices] = losses[days, slots, choices]
    return changed


def test_bandit_learns_only_from_the_arm_it_chose(data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    _, choices = thompson(forecasts, losses, valid, 14.0, False, 0.05, seed=3)
    _, again = thompson(forecasts, _hide_unchosen(losses, choices), valid, 14.0, False, 0.05, seed=3)
    np.testing.assert_array_equal(choices, again)


def test_full_information_reacts_to_arms_it_did_not_choose(data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    _, choices = thompson(forecasts, losses, valid, 14.0, True, 0.05, seed=3)
    _, again = thompson(forecasts, _hide_unchosen(losses, choices), valid, 14.0, True, 0.05, seed=3)
    assert not np.array_equal(choices, again)


def test_every_arm_keeps_the_minimum_share_and_the_best_arm_wins(data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    shares = np.mean([np.bincount(thompson(forecasts, losses, valid, np.inf, False, 0.05, seed)[1].ravel(),
                                  minlength=N_ARMS) / (DAYS * SLOTS) for seed in range(50)], axis=0)
    assert (shares >= 0.05).all()
    assert shares.argmax() == 1


def test_thompson_is_reproducible_with_a_seed(data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    first = thompson(forecasts, losses, valid, 14.0, False, 0.05, seed=7)[1]
    np.testing.assert_array_equal(first, thompson(forecasts, losses, valid, 14.0, False, 0.05, seed=7)[1])


def test_hedge_moves_to_the_best_arm_and_keeps_quantiles_sorted(data):
    forecasts, prices, valid = data
    losses = pinball_losses(forecasts, prices)
    out = hedge(forecasts, losses, valid, np.zeros(SLOTS, dtype=int), eta=1.0, half_life=np.inf)
    assert (np.diff(out, axis=-1) >= 0).all()
    error_late = np.abs(out[-5:, :, 1] - prices[-5:]).mean()
    error_start = np.abs(out[:2, :, 1] - prices[:2]).mean()
    assert error_late < 0.1 * error_start


@pytest.mark.parametrize("name", list(GRID))
def test_a_day_without_prices_leaves_no_gap_in_later_forecasts(name, data):
    forecasts, prices, valid = data
    prices = prices.copy()
    prices[10] = np.nan
    valid = valid.copy()
    valid[10] = False
    losses = pinball_losses(forecasts, prices)
    assert np.isfinite(run_strategy(name, forecasts, losses, valid, SETTINGS)).all()


def test_tuning_marks_one_setting_per_strategy(data):
    forecasts, prices, valid = data
    selection = tune_selection(forecasts[:12], prices[:12], valid[:12])
    assert selection.groupby("strategy")["chosen"].sum().eq(1).all()
    assert set(selection["strategy"]) == set(GRID)


def test_arrays_drop_a_day_if_one_arm_lacks_a_forecast():
    days = [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")]
    stamps = pd.date_range("2026-06-01", periods=192, freq="15min", tz="Europe/Zurich")
    arms = [pd.DataFrame({"model": arm, "timestamp_local": stamps, "delivery_day": stamps.tz_localize(None).normalize(),
                          "aep_ct_kwh": 5.0, "q10": 1.0, "q50": 2.0, "q90": 3.0})
            for arm in ("reference", "B1", "B2", "gbm_quantile")]
    frame = pd.concat(arms, ignore_index=True)
    frame.loc[(frame["model"] == "B2") & (frame["delivery_day"] == days[1]), "q50"] = np.nan
    forecasts, prices, valid = to_arrays(frame, days)
    assert forecasts.shape == (2, 96, 4, 3)
    assert prices.shape == (2, 96)
    assert valid.tolist() == [True, False]


def test_interval_of_a_constant_difference_is_that_constant():
    low, high = block_bootstrap(np.full(30, 0.25), reps=50)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_tune_and_test_keep_the_weekly_refit_days_of_the_official_backtest(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "member_forecasts", lambda table, days, jobs: calls.append(days) or pd.DataFrame({"a": [1]}))
    run.tune_and_test_forecasts(pd.DataFrame(), jobs=1)
    assert [days[0] for days in calls] == [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-01")]
    assert calls[-1][-1] == pd.Timestamp("2026-08-31")


def test_the_holdout_cannot_be_requested():
    with pytest.raises(SystemExit):
        main(["holdout"])
