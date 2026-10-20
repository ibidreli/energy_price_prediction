import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation.scores import daily_pinball, pinball, score, skill_score


def frame(y, q10, q50, q90, day="2026-07-01"):
    return pd.DataFrame({"aep_ct_kwh": y, "q10": q10, "q50": q50, "q90": q90, "delivery_day": pd.Timestamp(day)})


def test_pinball_matches_the_worked_example():
    # Sunny noon, actual price -30 ct/kWh: P10 = -21, P50 = 7, P90 = 24 (docs/models/baseline_model.md)
    assert pinball([-30], [-21], 0.1)[0] == pytest.approx(8.1)
    assert pinball([-30], [7], 0.5)[0] == pytest.approx(18.5)
    assert pinball([-30], [24], 0.9)[0] == pytest.approx(5.4)
    assert score(frame([-30.0], [-21.0], [7.0], [24.0]))["pinball"] == pytest.approx(32.0 / 3)


def test_pinball_weights_the_two_sides_by_tau():
    assert pinball([10], [0], 0.9)[0] == pytest.approx(9.0)  # forecast too low costs tau
    assert pinball([0], [10], 0.9)[0] == pytest.approx(1.0)  # forecast too high costs 1 - tau


def test_pinball_at_the_median_is_half_the_absolute_error():
    y, q = np.array([3.0, -5.0, 12.0]), np.array([1.0, 0.0, 20.0])
    assert pinball(y, q, 0.5).mean() == pytest.approx(np.abs(y - q).mean() / 2)


def test_score_reports_coverage_width_and_mae():
    result = score(frame([5.0, 50.0, -40.0, 10.0], [0.0] * 4, [10.0] * 4, [20.0] * 4))

    assert result["coverage_80"] == pytest.approx(0.5)
    assert result["width_80"] == pytest.approx(20.0)
    assert result["mae_q50"] == pytest.approx((5 + 40 + 50 + 0) / 4)
    assert result["n_quarter_hours"] == 4


def test_score_rejects_crossing_quantiles_and_missing_values():
    with pytest.raises(ValueError, match="cross"):
        score(frame([5.0], [10.0], [5.0], [20.0]))
    with pytest.raises(ValueError, match="missing"):
        score(frame([np.nan], [0.0], [5.0], [20.0]))


def test_skill_score_is_relative_improvement():
    assert skill_score(9.0, 10.0) == pytest.approx(0.1)
    assert skill_score(11.0, 10.0) == pytest.approx(-0.1)


def test_daily_pinball_averages_per_delivery_day():
    data = pd.concat([frame([0.0], [0.0], [0.0], [0.0], "2026-07-01"), frame([10.0], [0.0], [10.0], [10.0], "2026-07-02")])

    daily = daily_pinball(data)

    assert daily.loc[pd.Timestamp("2026-07-01")] == pytest.approx(0.0)
    assert daily.loc[pd.Timestamp("2026-07-02")] == pytest.approx(0.1 * 10 / 3)
