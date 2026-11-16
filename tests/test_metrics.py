import numpy as np
import pandas as pd
import pytest

from energy_price.metrics import (
    BASELINE_LAGS_DAYS,
    QUANTILES,
    baseline_quantiles,
    coverage,
    mae,
    mean_pinball_loss,
    mean_width,
    pinball_loss,
    score,
)


def test_pinball_loss_at_the_median_is_half_the_mae():
    actual = np.array([1.0, 4.0, -2.0, 10.0])
    predicted = np.array([2.0, 2.0, 0.0, 5.0])

    assert pinball_loss(actual, predicted, 0.5) == pytest.approx(mae(actual, predicted) / 2)


def test_pinball_loss_weights_a_too_low_forecast_with_tau_and_a_too_high_one_with_one_minus_tau():
    assert pinball_loss([10.0], [0.0], 0.9) == pytest.approx(9.0)
    assert pinball_loss([10.0], [20.0], 0.9) == pytest.approx(1.0)
    assert pinball_loss([10.0], [0.0], 0.1) == pytest.approx(1.0)
    assert pinball_loss([10.0], [20.0], 0.1) == pytest.approx(9.0)


def test_pinball_loss_is_smallest_at_the_empirical_quantile():
    actual = np.arange(1.0, 101.0)
    grid = np.arange(0.0, 101.0)
    for tau in QUANTILES:
        losses = [pinball_loss(actual, np.full(actual.shape, q), tau) for q in grid]
        best = grid[int(np.argmin(losses))]

        assert abs(best - np.quantile(actual, tau)) <= 1.0


def test_pinball_loss_rejects_a_quantile_outside_zero_and_one():
    with pytest.raises(ValueError, match="between 0 and 1"):
        pinball_loss([1.0], [1.0], 1.0)


def test_mean_pinball_loss_averages_over_the_quantiles():
    actual = np.array([5.0, 7.0])
    forecasts = {0.1: np.array([1.0, 2.0]), 0.5: np.array([5.0, 6.0]), 0.9: np.array([9.0, 9.0])}
    expected = np.mean([pinball_loss(actual, q, tau) for tau, q in forecasts.items()])

    assert mean_pinball_loss(actual, forecasts) == pytest.approx(expected)


def test_coverage_counts_values_on_the_bounds_as_inside():
    assert coverage([0.0, 5.0, 10.0, 11.0], [0.0] * 4, [10.0] * 4) == pytest.approx(0.75)


def test_coverage_and_width_reject_crossing_quantiles():
    with pytest.raises(ValueError, match="crossing quantiles"):
        coverage([1.0], [5.0], [2.0])
    with pytest.raises(ValueError, match="crossing quantiles"):
        mean_width([5.0], [2.0])


def test_inputs_with_missing_values_or_different_shapes_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        mae([1.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="differ in shape"):
        mae([1.0, 2.0], [1.0])


def test_score_returns_the_main_metric_followed_by_the_accompanying_scores():
    actual = np.array([1.0, 3.0, 8.0])
    result = score(actual, p10=[0.0, 0.0, 0.0], p50=[2.0, 2.0, 2.0], p90=[4.0, 4.0, 4.0])

    assert list(result) == ["mean_pinball", "mae_p50", "coverage_p10_p90", "width_p10_p90"]
    assert result["mae_p50"] == pytest.approx(mae(actual, [2.0, 2.0, 2.0]))
    assert result["coverage_p10_p90"] == pytest.approx(2 / 3)
    assert result["width_p10_p90"] == pytest.approx(4.0)


def test_baseline_quantiles_use_the_same_quarter_hour_on_d2_to_d8_and_leave_incomplete_windows_empty():
    stamps = pd.date_range("2026-01-01 12:00", periods=12, freq="D", tz="Europe/Zurich")
    prices = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": np.arange(12, dtype=float)})

    base = baseline_quantiles(prices)
    first_complete = max(BASELINE_LAGS_DAYS)

    assert base.iloc[:first_complete].isna().all().all()
    # day 10 (value 10) uses the values of days 2 to 8 before it, i.e. 2 to 8, median 5
    assert base.loc[10, 0.5] == pytest.approx(5.0)
    assert base.loc[10, 0.1] <= base.loc[10, 0.5] <= base.loc[10, 0.9]
