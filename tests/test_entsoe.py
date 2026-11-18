from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from energy_price.data import entsoe
from energy_price.data.entsoe import (
    availability,
    availability_summary,
    known_at_issue_entsoe,
    load_entsoe,
)
from energy_price.data.entsoe_xml import TZ, local_day_bounds
from energy_price.data.fetch_entsoe import SERIES, Request, run
from energy_price.paths import ROOT
from entsoe_fixtures import gl_document, period, price_document, price_series, time_series

D = date(2026, 11, 25)
LOG = ROOT / "data" / "entsoe" / "snapshots" / "fetch_log.csv"


def local(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=TZ)


def hourly(first: date, last: date | None = None, value=lambda hour: 1.0, price: bool = False) -> bytes:
    """Hourly values over the local days ``first`` to ``last``; ``value`` gets the hour index."""
    start, _ = local_day_bounds(first)
    _, end = local_day_bounds(last or first)
    values = [value(i) for i in range(int((end - start) / timedelta(hours=1)))]
    if price:
        return price_document([price_series([period(start, end, "PT60M", values, "price.amount")], "A01")])
    return gl_document([time_series([period(start, end, "PT60M", values)])])


def store(root, name, zone, fetched, content, day=None, period_days=None, source="live"):
    """Store one response as the fetch job does: live per day, backfill over ``period_days``."""
    series = next(s for s in SERIES if s.name == name and s.zone == zone)
    if source == "live":
        request = Request(series, *local_day_bounds(day), delivery_day=day)
    else:
        first, last = period_days
        request = Request(series, local_day_bounds(first)[0], local_day_bounds(last)[1])
    folder = root / ("snapshots" if source == "live" else "backfill")
    run([request], folder, source, "token", get=lambda params, token: (200, content), now=lambda: fetched,
        sleep=lambda s: None)


def values(frame, name, zone="CH"):
    return frame.loc[(frame["series"] == name) & (frame["zone"] == zone), "value"]


def test_snapshots_become_quarter_hours_with_their_version(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D - timedelta(days=1), 10, 17), hourly(D, value=float), day=D)
    frame = load_entsoe(tmp_path)
    assert len(frame) == 96
    assert frame["value"].tolist() == [float(h) for h in range(24) for _ in range(4)]
    assert (frame["version_day"] == pd.Timestamp(D)).all() and (frame["source"] == "live").all()
    assert frame["fetched_at_utc"].iloc[0] == pd.Timestamp("2026-11-24T09:17:00Z")
    assert frame["timestamp_local"].iloc[0] == pd.Timestamp("2026-11-25 00:00", tz=TZ)


def test_the_finer_resolution_wins_within_a_response(tmp_path):
    start, end = local_day_bounds(D)
    doc = gl_document([time_series([period(start, end, "PT60M", [1.0] * 24)]),
                       time_series([period(start, end, "PT15M", [2.0] * 96)])])
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), doc, day=D)
    assert values(load_entsoe(tmp_path), "load_day_ahead").tolist() == [2.0] * 96


def test_a_changed_snapshot_file_is_rejected(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), hourly(D), day=D)
    path = next((tmp_path / "snapshots").glob("*.xml"))
    path.write_bytes(hourly(D, value=lambda h: 5.0))
    with pytest.raises(ValueError, match="differ from manifest"):
        load_entsoe(tmp_path)


def test_no_snapshots_give_an_empty_table_with_all_columns(tmp_path):
    frame = load_entsoe(tmp_path)
    assert frame.empty and list(frame.columns) == entsoe.COLUMNS
    assert isinstance(frame["fetched_at_utc"].dtype, pd.DatetimeTZDtype)


@pytest.mark.parametrize(("second", "used"), [(local(D - timedelta(days=1), 10, 59, 59), True),
                                               (local(D - timedelta(days=1), 11, 0, 0), True),
                                               (local(D - timedelta(days=1), 11, 0, 1), False)])
def test_live_values_count_only_if_fetched_by_d_minus_1_11_00(tmp_path, second, used):
    store(tmp_path, "price_day_ahead", "CH", local(D - timedelta(days=3), 9, 17), hourly(D - timedelta(days=2), price=True),
          day=D - timedelta(days=2))
    store(tmp_path, "load_day_ahead", "CH", second, hourly(D), day=D)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert (len(values(known, "load_day_ahead")) == 96) is used


def test_the_newest_live_version_before_the_issue_time_wins(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D - timedelta(days=1), 9, 17), hourly(D, value=lambda h: 1.0), day=D)
    store(tmp_path, "load_day_ahead", "CH", local(D - timedelta(days=1), 10, 37), hourly(D, value=lambda h: 2.0), day=D)
    store(tmp_path, "load_day_ahead", "CH", local(D - timedelta(days=1), 15, 17), hourly(D, value=lambda h: 3.0), day=D)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert values(known, "load_day_ahead").unique().tolist() == [2.0]


def test_a_value_outside_the_requested_day_gets_one_version(tmp_path):
    week = hourly(D - timedelta(days=1), D, value=lambda h: 1.0)
    store(tmp_path, "load_week_ahead", "CH", local(D - timedelta(days=2), 10, 17), week, day=D - timedelta(days=1))
    store(tmp_path, "load_week_ahead", "CH", local(D - timedelta(days=1), 10, 17), hourly(D, value=lambda h: 2.0), day=D)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    on_d = known[(known["series"] == "load_week_ahead") & (known["local_day"] == pd.Timestamp(D))]
    assert len(on_d) == 96 and on_d["value"].unique().tolist() == [2.0]


def test_the_price_of_the_delivery_day_never_enters_even_if_stored_early(tmp_path):
    early = local(D - timedelta(days=1), 9, 17)
    store(tmp_path, "price_day_ahead", "CH", early, hourly(D - timedelta(days=1), price=True), day=D - timedelta(days=1))
    store(tmp_path, "price_day_ahead", "CH", early, hourly(D, value=lambda h: 999.0, price=True), day=D)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert known["local_day"].max() == pd.Timestamp(D - timedelta(days=1))
    assert 999.0 not in values(known, "price_day_ahead").tolist()


def test_actual_load_ends_with_d_minus_2(tmp_path):
    fetched = local(D - timedelta(days=1), 10, 17)
    for offset in (2, 1):
        day = D - timedelta(days=offset)
        store(tmp_path, "load_actual", "CH", fetched, hourly(day, value=lambda h, o=offset: float(o)), day=day)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert values(known, "load_actual").unique().tolist() == [2.0]


def test_the_backfill_never_fills_a_gap_in_the_live_period(tmp_path):
    store(tmp_path, "price_day_ahead", "CH", local(D - timedelta(days=5), 10, 17), hourly(D - timedelta(days=5), price=True),
          day=D - timedelta(days=5))
    store(tmp_path, "price_day_ahead", "CH", local(D + timedelta(days=5), 12, 0),
          hourly(D - timedelta(days=30), D + timedelta(days=1), price=True),
          period_days=(D - timedelta(days=30), D + timedelta(days=1)), source="backfill")
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert (known["source"] == "live").all()
    assert known["local_day"].max() == pd.Timestamp(D - timedelta(days=5))


def test_history_uses_only_the_series_the_rules_allow(tmp_path, monkeypatch):
    first, last = date(2026, 1, 1), date(2026, 1, 31)
    fetched = local(date(2026, 11, 19), 12, 0)
    store(tmp_path, "load_day_ahead", "CH", fetched, hourly(first, last), period_days=(first, last), source="backfill")
    store(tmp_path, "price_day_ahead", "CH", fetched, hourly(first, last, price=True), period_days=(first, last),
          source="backfill")
    frame = load_entsoe(tmp_path)
    known = known_at_issue_entsoe(frame, date(2026, 1, 15))
    assert set(known["series"]) == {"price_day_ahead"}
    assert known["local_day"].max() == pd.Timestamp("2026-01-14")

    monkeypatch.setitem(entsoe.HISTORY_RULES, "load_day_ahead", True)
    known = known_at_issue_entsoe(frame, date(2026, 1, 15))
    assert values(known, "load_day_ahead").size == 15 * 96  # 1 to 15 January, never a later day


def test_history_takes_the_earliest_download_of_a_value(tmp_path):
    first, last = date(2026, 1, 1), date(2026, 1, 31)
    for hour, value in [(12, 1.0), (13, 2.0)]:
        store(tmp_path, "price_day_ahead", "CH", local(date(2026, 11, 19), hour), hourly(first, last, value=lambda h, v=value: v,
              price=True), period_days=(first, last), source="backfill")
    known = known_at_issue_entsoe(load_entsoe(tmp_path), date(2026, 1, 15))
    assert values(known, "price_day_ahead").unique().tolist() == [1.0]


def test_selection_rejects_naive_times_and_unknown_series(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), hourly(D), day=D)
    frame = load_entsoe(tmp_path)
    with pytest.raises(TypeError):
        known_at_issue_entsoe(frame.assign(fetched_at_utc=frame["fetched_at_utc"].dt.tz_localize(None)), D)
    with pytest.raises(ValueError, match="no availability rule"):
        known_at_issue_entsoe(frame.assign(series="load_month_ahead"), D)


def log_row(fetched: datetime, series: str, needed: date, status: str = "ok", complete: bool = True, zone: str = "CH"):
    return {"fetched_at_utc": pd.Timestamp(fetched).tz_convert("UTC").isoformat(), "source": "live", "series": series,
            "zone": zone, "delivery_day": needed.isoformat(), "status": status, "complete": str(complete)}


def week_of_runs(series: str = "load_day_ahead", days: int = 7, bad: dict | None = None):
    """Runs on the fetch days T for D = T+1: 10:17 and 10:37, then 15:17. ``bad`` maps day index to a run."""
    rows = []
    for i in range(days):
        fetch_day = date(2026, 11, 19) + timedelta(days=i)
        needed = fetch_day + timedelta(days=1 + entsoe.LATEST_DAY[series])
        morning = (bad or {}).get(i, [("ok", True)])
        for (status, complete), minute in zip(morning, (17, 37)):
            rows.append(log_row(local(fetch_day, 10, minute), series, needed, status, complete))
        rows.append(log_row(local(fetch_day, 15, 17), series, needed))
    return pd.DataFrame(rows)


def test_a_day_complete_before_11_counts():
    report = availability(week_of_runs())
    assert report["delivery_day"].min() == pd.Timestamp("2026-11-20")
    assert report["checked"].all() and report["complete_before_issue"].all()
    assert report["first_complete_local"].iloc[0] == pd.Timestamp("2026-11-19 10:17", tz=TZ)


def test_a_value_that_appears_only_after_11_does_not_count():
    log = week_of_runs(bad={2: [("no_data", False), ("no_data", False)]})
    log.loc[(log["fetched_at_utc"] == pd.Timestamp(local(date(2026, 11, 21), 15, 17)).tz_convert("UTC").isoformat()),
            "complete"] = "True"
    row = availability(log).iloc[2]
    assert row["checked"] and not row["complete_before_issue"]
    assert row["statuses_before_issue"] == "no_data"
    assert row["first_complete_local"] == pd.Timestamp("2026-11-21 15:17", tz=TZ)


def test_a_day_without_answer_before_11_is_unchecked():
    report = availability(week_of_runs(bad={3: [("network_error", False), ("network_error", False)]}))
    assert not report.iloc[3]["checked"] and report.iloc[3]["runs_before_issue"] == 2


def test_actual_load_and_prices_are_matched_to_the_delivery_day_they_serve():
    for series, needed in [("load_actual", "2026-11-18"), ("price_day_ahead", "2026-11-19")]:
        report = availability(week_of_runs(series))
        assert report.iloc[0]["delivery_day"] == pd.Timestamp("2026-11-20")
        assert report.iloc[0]["needed_day"] == pd.Timestamp(needed)


@pytest.mark.parametrize(("days", "bad", "meets"), [
    (7, None, True),
    (6, None, False),
    (8, {5: [("ok", False), ("ok", False)]}, False),
    (8, {5: [("network_error", False), ("network_error", False)]}, True),
])
def test_the_criterion_needs_seven_checked_days_all_complete(days, bad, meets):
    summary = availability_summary(availability(week_of_runs(days=days, bad=bad)))
    assert bool(summary["meets_criterion"].iloc[0]) is meets


def test_the_summary_lists_unchecked_days():
    summary = availability_summary(availability(week_of_runs(days=8, bad={5: [("network_error", False)] * 2})))
    assert summary["unchecked_days"].iloc[0] == "11-25"


@pytest.mark.realdata
@pytest.mark.skipif(not LOG.exists(), reason="no live ENTSO-E log yet, see docs/data.md, section 7")
def test_the_live_job_answered_before_11_on_every_day():
    log = pd.read_csv(LOG, dtype=str, keep_default_na=False)
    unchecked = availability(log).query("not checked")
    assert unchecked.empty, f"no answer before 11:00 on:\n{unchecked[['series', 'zone', 'delivery_day']]}"
