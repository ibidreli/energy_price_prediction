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
from energy_price.paths import ROOT
from entsoe_fixtures import gl_document, period, time_series
from entsoe_store import local, store

D = date(2026, 11, 25)
LOGS = list((ROOT / "data" / "entsoe").glob("*/snapshots/fetch_log.csv"))


def days(n: int) -> timedelta:
    return timedelta(days=n)


def values(frame, name, zone="CH", provider="energy_charts"):
    rows = frame[(frame["series"] == name) & (frame["zone"] == zone) & (frame["provider"] == provider)]
    return rows.sort_values("timestamp_utc")["value"]


@pytest.mark.parametrize("provider", ["energy_charts", "entsoe_api"])
def test_snapshots_of_both_sources_become_quarter_hours(tmp_path, provider):
    store(tmp_path, "load_day_ahead", "CH", local(D - days(1), 10, 17), D, value=float, provider=provider)
    frame = load_entsoe(tmp_path)
    assert len(frame) == 96 and (frame["provider"] == provider).all() and (frame["source"] == "live").all()
    assert values(frame, "load_day_ahead", provider=provider).tolist() == [float(h) for h in range(24) for _ in range(4)]
    assert frame["fetched_at_utc"].iloc[0] == pd.Timestamp("2026-11-24T09:17:00Z")
    assert frame["timestamp_local"].min() == pd.Timestamp("2026-11-25 00:00", tz=TZ)


def test_a_live_window_of_three_days_is_one_snapshot(tmp_path):
    store(tmp_path, "price_day_ahead", "CH", local(D - days(1), 10, 17), D - days(2), D)
    frame = load_entsoe(tmp_path)
    assert frame["snapshot"].nunique() == 1 and frame["local_day"].nunique() == 3


def test_entsoe_solar_and_wind_become_two_series(tmp_path):
    start, end = local_day_bounds(D)
    doc = gl_document([
        time_series([period(start, end, "PT60M", [300.0] * 24)], business_type="A94", psr_type="B16"),
        time_series([period(start, end, "PT60M", [20.0] * 24)], business_type="A94", psr_type="B19"),
        time_series([period(start, end, "PT60M", [5.0] * 24)], business_type="A94", psr_type="B18"),
    ])
    store(tmp_path, "wind_solar_day_ahead", "CH", local(D - days(1), 10, 17), D, provider="entsoe_api", content=doc)
    frame = load_entsoe(tmp_path)
    assert set(frame["series"]) == {"solar_day_ahead", "wind_day_ahead"}
    assert frame.groupby("series")["group"].nunique().to_dict() == {"solar_day_ahead": 1, "wind_day_ahead": 2}


def test_the_finer_resolution_wins_within_a_response(tmp_path):
    start, end = local_day_bounds(D)
    doc = gl_document([time_series([period(start, end, "PT60M", [1.0] * 24)]),
                       time_series([period(start, end, "PT15M", [2.0] * 96)])])
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), D, provider="entsoe_api", content=doc)
    assert values(load_entsoe(tmp_path), "load_day_ahead", provider="entsoe_api").tolist() == [2.0] * 96


def test_a_changed_snapshot_file_is_rejected(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), D)
    path = next((tmp_path / "energy_charts" / "snapshots").glob("*.json"))
    path.write_bytes(path.read_bytes().replace(b'"load": 1.0', b'"load": 5.0'))
    with pytest.raises(ValueError, match="differ from manifest"):
        load_entsoe(tmp_path)


def test_no_snapshots_give_an_empty_table_with_all_columns(tmp_path):
    frame = load_entsoe(tmp_path)
    assert frame.empty and list(frame.columns) == entsoe.COLUMNS
    assert isinstance(frame["fetched_at_utc"].dtype, pd.DatetimeTZDtype)
    assert known_at_issue_entsoe(frame, D).empty


@pytest.mark.parametrize(("fetched", "used"), [(local(D - days(1), 10, 59, 59), True),
                                               (local(D - days(1), 11, 0, 0), True),
                                               (local(D - days(1), 11, 0, 1), False)])
def test_live_values_count_only_if_fetched_by_d_minus_1_11_00(tmp_path, fetched, used):
    store(tmp_path, "price_day_ahead", "CH", local(D - days(3), 9, 17), D - days(2))
    store(tmp_path, "load_day_ahead", "CH", fetched, D)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert (len(values(known, "load_day_ahead")) == 96) is used


def test_the_newest_live_version_before_the_issue_time_wins(tmp_path):
    for hour, minute, value in [(9, 17, 1.0), (10, 37, 2.0), (15, 17, 3.0)]:
        store(tmp_path, "load_day_ahead", "CH", local(D - days(1), hour, minute), D, value=lambda h, v=value: v)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert values(known, "load_day_ahead").unique().tolist() == [2.0]


def test_a_value_in_two_live_windows_gets_one_version(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D - days(2), 10, 17), D - days(3), D - days(1), value=lambda h: 1.0)
    store(tmp_path, "load_day_ahead", "CH", local(D - days(1), 10, 17), D - days(2), D, value=lambda h: 2.0)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    on_d = known[known["local_day"] == pd.Timestamp(D)]
    assert len(on_d) == 96 and on_d["value"].unique().tolist() == [2.0]


def test_the_price_of_the_delivery_day_never_enters_even_if_stored_early(tmp_path):
    store(tmp_path, "price_day_ahead", "CH", local(D - days(1), 9, 17), D - days(1), D,
          value=lambda h: 999.0 if h >= 24 else 80.0)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert known["local_day"].max() == pd.Timestamp(D - days(1))
    assert 999.0 not in values(known, "price_day_ahead").tolist()


def test_actual_load_ends_with_d_minus_2(tmp_path):
    store(tmp_path, "load_actual", "CH", local(D - days(1), 10, 17), D - days(2), D - days(1),
          value=lambda h: 2.0 if h < 24 else 1.0)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert values(known, "load_actual").unique().tolist() == [2.0]


def test_the_backfill_never_fills_a_gap_in_the_live_period(tmp_path):
    store(tmp_path, "price_day_ahead", "CH", local(D - days(5), 10, 17), D - days(5))
    store(tmp_path, "price_day_ahead", "CH", local(D + days(5), 12, 0), D - days(30), D + days(1), source="backfill")
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    assert (known["source"] == "live").all()
    assert known["local_day"].max() == pd.Timestamp(D - days(5))


def test_each_source_has_its_own_live_period(tmp_path):
    # Energy-Charts runs live, the ENTSO-E API has only its history: its history still counts.
    store(tmp_path, "price_day_ahead", "CH", local(D - days(5), 10, 17), D - days(5))
    store(tmp_path, "price_day_ahead", "CH", local(D + days(5), 12, 0), D - days(30), D + days(1), source="backfill",
          provider="entsoe_api")
    known = known_at_issue_entsoe(load_entsoe(tmp_path), D)
    api = known[known["provider"] == "entsoe_api"]
    assert (api["source"] == "backfill").all() and api["local_day"].max() == pd.Timestamp(D - days(1))


def test_history_uses_only_the_series_the_rules_allow(tmp_path, monkeypatch):
    first, last, fetched = date(2026, 1, 1), date(2026, 1, 31), local(date(2026, 11, 19), 12, 0)
    store(tmp_path, "load_day_ahead", "CH", fetched, first, last, source="backfill")
    store(tmp_path, "price_day_ahead", "CH", fetched, first, last, source="backfill")
    frame = load_entsoe(tmp_path)
    known = known_at_issue_entsoe(frame, date(2026, 1, 15))
    assert set(known["series"]) == {"price_day_ahead"}
    assert known["local_day"].max() == pd.Timestamp("2026-01-14")

    monkeypatch.setitem(entsoe.HISTORY_RULES["energy_charts"], "load_day_ahead", True)
    known = known_at_issue_entsoe(frame, date(2026, 1, 15))
    assert values(known, "load_day_ahead").size == 15 * 96  # 1 to 15 January, never a later day


def test_history_rules_are_per_source(tmp_path, monkeypatch):
    first, last, fetched = date(2026, 1, 1), date(2026, 1, 31), local(date(2026, 11, 19), 12, 0)
    for provider in ["energy_charts", "entsoe_api"]:
        store(tmp_path, "load_day_ahead", "CH", fetched, first, last, source="backfill", provider=provider)
    monkeypatch.setitem(entsoe.HISTORY_RULES["entsoe_api"], "load_day_ahead", True)
    known = known_at_issue_entsoe(load_entsoe(tmp_path), date(2026, 1, 15))
    assert set(known.loc[known["series"] == "load_day_ahead", "provider"]) == {"entsoe_api"}


def test_history_takes_the_earliest_download_of_a_value(tmp_path):
    first, last = date(2026, 1, 1), date(2026, 1, 31)
    for hour, value in [(12, 1.0), (13, 2.0)]:
        store(tmp_path, "price_day_ahead", "CH", local(date(2026, 11, 19), hour), first, last,
              value=lambda h, v=value: v, source="backfill")
    known = known_at_issue_entsoe(load_entsoe(tmp_path), date(2026, 1, 15))
    assert values(known, "price_day_ahead").unique().tolist() == [1.0]


def test_selection_rejects_naive_times_unknown_series_and_unknown_sources(tmp_path):
    store(tmp_path, "load_day_ahead", "CH", local(D, 9, 17), D)
    frame = load_entsoe(tmp_path)
    with pytest.raises(TypeError):
        known_at_issue_entsoe(frame.assign(fetched_at_utc=frame["fetched_at_utc"].dt.tz_localize(None)), D)
    with pytest.raises(ValueError, match="no availability rule"):
        known_at_issue_entsoe(frame.assign(series="load_month_ahead"), D)
    with pytest.raises(ValueError, match="no history rules"):
        known_at_issue_entsoe(frame.assign(provider="smard"), D)


def log_row(fetched: datetime, series: str, needed: date, status: str = "ok", complete: bool = True,
            zone: str = "CH", provider: str = "energy_charts"):
    return {"fetched_at_utc": pd.Timestamp(fetched).tz_convert("UTC").isoformat(), "provider": provider,
            "source": "live", "series": series, "zone": zone, "delivery_day": needed.isoformat(), "status": status,
            "complete": str(complete)}


def week_of_runs(series: str = "load_day_ahead", n: int = 7, bad: dict | None = None, provider: str = "energy_charts"):
    """Runs on the fetch days T for D = T+1: 10:17 and 10:37, then 15:17. ``bad`` maps day index to a run."""
    rows = []
    for i in range(n):
        fetch_day = date(2026, 11, 26) + days(i)
        needed = fetch_day + days(1 + entsoe.LATEST_DAY[series])
        morning = (bad or {}).get(i, [("ok", True)])
        for (status, complete), minute in zip(morning, (17, 37)):
            rows.append(log_row(local(fetch_day, 10, minute), series, needed, status, complete, provider=provider))
        rows.append(log_row(local(fetch_day, 15, 17), series, needed, provider=provider))
    return pd.DataFrame(rows)


def test_a_day_complete_before_11_counts():
    report = availability(week_of_runs())
    assert report["delivery_day"].min() == pd.Timestamp("2026-11-27")
    assert report["checked"].all() and report["complete_before_issue"].all()
    assert report["first_complete_local"].iloc[0] == pd.Timestamp("2026-11-26 10:17", tz=TZ)


def test_a_value_that_appears_only_after_11_does_not_count():
    log = week_of_runs(bad={2: [("no_data", False), ("no_data", False)]})
    late = pd.Timestamp(local(date(2026, 11, 28), 15, 17)).tz_convert("UTC").isoformat()
    log.loc[log["fetched_at_utc"] == late, "complete"] = "True"
    row = availability(log).iloc[2]
    assert row["checked"] and not row["complete_before_issue"]
    assert row["statuses_before_issue"] == "no_data"
    assert row["first_complete_local"] == pd.Timestamp("2026-11-28 15:17", tz=TZ)


def test_a_day_without_answer_before_11_is_unchecked():
    report = availability(week_of_runs(bad={3: [("network_error", False), ("network_error", False)]}))
    assert not report.iloc[3]["checked"] and report.iloc[3]["runs_before_issue"] == 2


def test_actual_load_and_prices_are_matched_to_the_delivery_day_they_serve():
    for series, needed in [("load_actual", "2026-11-25"), ("price_day_ahead", "2026-11-26")]:
        report = availability(week_of_runs(series))
        assert report.iloc[0]["delivery_day"] == pd.Timestamp("2026-11-27")
        assert report.iloc[0]["needed_day"] == pd.Timestamp(needed)


def test_sources_are_evaluated_separately():
    log = pd.concat([week_of_runs(), week_of_runs(n=3, provider="entsoe_api")])
    summary = availability_summary(availability(log)).set_index("provider")
    assert summary.loc["energy_charts", "meets_criterion"] and not summary.loc["entsoe_api", "meets_criterion"]
    assert summary.loc["entsoe_api", "days"] == 3


@pytest.mark.parametrize(("n", "bad", "meets"), [
    (7, None, True),
    (6, None, False),
    (8, {5: [("ok", False), ("ok", False)]}, False),
    (8, {5: [("network_error", False), ("network_error", False)]}, True),
])
def test_the_criterion_needs_seven_checked_days_all_complete(n, bad, meets):
    summary = availability_summary(availability(week_of_runs(n=n, bad=bad)))
    assert bool(summary["meets_criterion"].iloc[0]) is meets


def test_the_summary_lists_unchecked_days():
    summary = availability_summary(availability(week_of_runs(n=8, bad={5: [("network_error", False)] * 2})))
    assert summary["unchecked_days"].iloc[0] == "12-02"


def test_a_single_run_after_11_gives_no_day_to_evaluate(tmp_path, capsys):
    store(tmp_path, "load_day_ahead", "CH", local(D - days(1), 13, 56), D - days(2), D)
    log_path = tmp_path / "energy_charts" / "snapshots" / "fetch_log.csv"
    assert availability(pd.read_csv(log_path, dtype=str, keep_default_na=False)).empty
    entsoe.main(["--log", str(log_path), "--out", str(tmp_path / "availability.csv")])
    assert "no delivery day to evaluate yet" in capsys.readouterr().out
    assert not (tmp_path / "availability.csv").exists()


def test_the_availability_report_reads_the_real_fetch_log_format(tmp_path):
    for i in range(2):
        store(tmp_path, "load_day_ahead", "CH", local(D + days(i), 10, 17), D + days(i - 1), D + days(i + 1))
    log = pd.read_csv(tmp_path / "energy_charts" / "snapshots" / "fetch_log.csv", dtype=str, keep_default_na=False)
    report = availability(log)
    assert report["delivery_day"].tolist() == [pd.Timestamp(D + days(1))]
    assert report["complete_before_issue"].all()


@pytest.mark.realdata
@pytest.mark.skipif(not LOGS, reason="no live fetch log yet, see docs/data.md, section 7")
def test_the_live_job_answered_before_11_on_every_day():
    log = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in LOGS])
    unchecked = availability(log).query("not checked")
    assert unchecked.empty, f"no answer before 11:00 on:\n{unchecked[['provider', 'series', 'zone', 'delivery_day']]}"
