import datetime as dt

import pandas as pd
import pytest

from energy_price.availability import issue_time, known_at_issue, swissgrid_cutoff

TZ = "Europe/Zurich"


def quarter_hours(start: str, end: str) -> pd.DataFrame:
    stamps = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="15min", inclusive="left")
    return pd.DataFrame({"timestamp_local": stamps, "value": range(len(stamps))})


@pytest.mark.parametrize(
    ("day", "cutoff"),
    [
        (dt.date(2026, 7, 16), "2026-07-15 00:00+02:00"),
        (dt.date(2026, 3, 30), "2026-03-29 00:00+01:00"),  # cutoff lies before the switch to summer time
        (dt.date(2026, 10, 26), "2026-10-25 00:00+02:00"),
    ],
)
def test_swissgrid_cutoff_is_the_start_of_d_minus_1(day, cutoff):
    assert swissgrid_cutoff(day) == pd.Timestamp(cutoff)


def test_cutoff_lies_before_the_issue_time_on_every_day():
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(365)]

    assert all(swissgrid_cutoff(d) < issue_time(d) for d in days)


def test_known_at_issue_keeps_d_minus_2_and_drops_everything_later():
    frame = quarter_hours("2026-07-13 00:00", "2026-07-17 00:00")

    known = known_at_issue(frame, dt.date(2026, 7, 16))

    assert known["timestamp_local"].max() == pd.Timestamp("2026-07-14 23:45", tz=TZ)
    assert (known["timestamp_local"] < issue_time(dt.date(2026, 7, 16))).all()


def test_allowed_distance_depends_on_the_target_quarter_hour():
    frame = quarter_hours("2026-07-13 00:00", "2026-07-17 00:00")
    last_known = known_at_issue(frame, dt.date(2026, 7, 16))["timestamp_local"].max()

    assert pd.Timestamp("2026-07-16 00:00", tz=TZ) - last_known == pd.Timedelta(hours=24, minutes=15)
    assert pd.Timestamp("2026-07-16 23:45", tz=TZ) - last_known == pd.Timedelta(hours=48)


def test_a_fixed_35_hour_lag_would_leak_for_late_quarter_hours():
    # Documents why known_at_issue exists: shifting by a fixed lag reaches past the issue time.
    target = pd.Timestamp("2026-07-16 23:45", tz=TZ)

    assert target - pd.Timedelta(hours=35) > issue_time(dt.date(2026, 7, 16))


def test_known_at_issue_rejects_naive_timestamps():
    frame = pd.DataFrame({"timestamp_local": pd.date_range("2026-07-14", periods=2, freq="15min")})

    with pytest.raises(TypeError, match="timezone-aware"):
        known_at_issue(frame, dt.date(2026, 7, 16))
