import numpy as np
import pytest

from energy_price.metrics import QUANTILES, TARGET_COVERAGE, coverage, mae, mean_pinball_loss, mean_width, pinball_loss, score


def test_pinball_weights_under_and_over_forecasts_by_the_quantile():
    # actual 10: forecast 6 is 4 too low, forecast 14 is 4 too high
    assert pinball_loss([10.0], [6.0], 0.9) == pytest.approx(0.9 * 4)
    assert pinball_loss([10.0], [14.0], 0.9) == pytest.approx(0.1 * 4)


def test_pinball_of_the_median_is_half_the_mae():
    actual, predicted = np.array([1.0, -5.0, 20.0]), np.array([3.0, 0.0, 10.0])

    assert pinball_loss(actual, predicted, 0.5) == pytest.approx(mae(actual, predicted) / 2)


def test_pinball_is_zero_for_a_perfect_forecast():
    assert pinball_loss([-688.48, 13.1], [-688.48, 13.1], 0.1) == 0.0


def test_mean_pinball_averages_over_the_quantiles():
    actual = [10.0]
    forecasts = {0.1: [6.0], 0.5: [10.0], 0.9: [14.0]}

    assert mean_pinball_loss(actual, forecasts) == pytest.approx((0.1 * 4 + 0 + 0.1 * 4) / 3)


def test_coverage_includes_both_bounds_and_width_is_the_mean():
    actual = [0.0, 5.0, 10.0, 11.0]
    lower, upper = [0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]

    assert coverage(actual, lower, upper) == 0.75
    assert mean_width(lower, upper) == 10.0


def test_crossing_quantiles_are_refused():
    with pytest.raises(ValueError, match="crossing"):
        coverage([1.0], [5.0], [2.0])


def test_missing_values_are_refused():
    with pytest.raises(ValueError, match="NaN"):
        mae([1.0, np.nan], [1.0, 2.0])


def test_quantile_outside_zero_and_one_is_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        pinball_loss([1.0], [1.0], 1.0)


def test_score_reports_the_agreed_metrics_in_order():
    result = score([10.0, 20.0], p10=[5.0, 5.0], p50=[10.0, 10.0], p90=[15.0, 15.0])

    assert list(result) == ["mean_pinball", "mae_p50", "coverage_p10_p90", "width_p10_p90"]
    assert result["coverage_p10_p90"] == 0.5
    assert result["width_p10_p90"] == 10.0
    assert result["mae_p50"] == 5.0


def test_the_agreed_quantiles_give_an_80_percent_interval():
    assert QUANTILES == (0.1, 0.5, 0.9)
    assert TARGET_COVERAGE == pytest.approx(0.8)
