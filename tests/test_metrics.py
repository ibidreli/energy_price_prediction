import pandas as pd
import pytest

from energy_price.metrics import baseline_quantiles, coverage, mae, mean_pinball, pinball

TZ = "Europe/Zurich"


def test_mae():
    y = pd.Series([0.0, 10.0, -20.0])
    assert mae(y, pd.Series([1.0, 8.0, -20.0])) == pytest.approx(1.0)


def test_pinball_at_median_is_half_the_mae():
    y, q = pd.Series([3.0, -5.0, 12.0]), pd.Series([1.0, 0.0, 10.0])
    assert pinball(y, q, 0.5) == pytest.approx(mae(y, q) / 2)


def test_pinball_penalises_the_wrong_side_more():
    y = pd.Series([10.0])
    # A 0.9 quantile below the observation costs 0.9 per unit, above it only 0.1.
    assert pinball(y, pd.Series([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball(y, pd.Series([12.0]), 0.9) == pytest.approx(0.2)
    assert mean_pinball(y, {0.9: pd.Series([8.0]), 0.1: pd.Series([12.0])}) == pytest.approx(1.8)


def test_coverage_counts_the_bounds_as_inside():
    y = pd.Series([1.0, 5.0, 9.0, 11.0])
    assert coverage(y, pd.Series([1.0] * 4), pd.Series([9.0] * 4)) == pytest.approx(0.75)


def test_baseline_uses_the_same_quarter_hour_of_d_minus_2_to_d_minus_8():
    start = pd.Timestamp("2026-07-01", tz=TZ)
    stamps = pd.date_range(start, pd.Timestamp("2026-07-11", tz=TZ), freq="15min", inclusive="left")
    day = (stamps.normalize() - start).days
    prices = pd.DataFrame({"timestamp_local": stamps, "aep_ct_kwh": day * 100 + stamps.hour * 4 + stamps.minute // 15})

    base = baseline_quantiles(prices)

    assert base[day < 8].isna().all().all()  # D-8 lies before the data
    noon = (day == 9) & (stamps.hour == 12) & (stamps.minute == 0)
    # D-2 .. D-8 of day 9 are days 1 .. 7, all at quarter hour 48; the median is day 4.
    assert base.loc[noon, 0.5].item() == pytest.approx(4 * 100 + 48)
    assert base.loc[noon, 0.9].item() < 7 * 100 + 48
