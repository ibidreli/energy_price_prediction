from datetime import date, datetime, timedelta, timezone

import pytest

from energy_price.data.entsoe_xml import (
    EntsoeResponseError,
    data_checksum,
    is_complete,
    local_day_bounds,
    parse,
)
from entsoe_fixtures import acknowledgement, gl_document, period, price_document, price_series, time_series

DAY = date(2026, 11, 19)
START, END = local_day_bounds(DAY)


def test_local_day_bounds_follow_the_clock_changes():
    assert local_day_bounds(date(2026, 3, 29)) == (datetime(2026, 3, 28, 23, tzinfo=timezone.utc),
                                                   datetime(2026, 3, 29, 22, tzinfo=timezone.utc))
    start, end = local_day_bounds(date(2026, 10, 25))
    assert end - start == timedelta(hours=25)


def test_quarter_hourly_load_gives_one_point_per_step():
    response = parse(gl_document([time_series([period(START, END, "PT15M", range(96))])]))
    assert len(response.points) == 96
    assert response.points[0].start_utc == START and response.points[0].value == 0.0
    assert response.points[-1].end_utc == END and response.points[-1].value == 95.0
    assert response.groups == ["A04"]
    assert is_complete(response.points, DAY)


def test_a03_repeats_a_value_until_the_next_listed_position():
    end = START + timedelta(hours=2)
    values = [10.0, None, None, None, 20.0, None, None, None]
    response = parse(price_document([price_series([period(START, end, "PT15M", values, "price.amount")])]))
    assert [p.value for p in response.points] == [10.0] * 4 + [20.0] * 4


def test_a01_keeps_a_missing_position_as_gap():
    values = [1.0] * 96
    values[40] = None
    response = parse(gl_document([time_series([period(START, END, "PT15M", values)])]))
    assert len(response.points) == 95
    assert not is_complete(response.points, DAY)


@pytest.mark.parametrize(("day", "hours"), [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)])
def test_hourly_values_cover_the_quarter_hours_of_switch_days(day, hours):
    start, end = local_day_bounds(day)
    response = parse(gl_document([time_series([period(start, end, "PT60M", range(hours))])]))
    assert len(response.points) == hours
    assert is_complete(response.points, day)


def test_daily_resolution_follows_the_local_calendar():
    start, _ = local_day_bounds(date(2026, 3, 28))
    _, end = local_day_bounds(date(2026, 3, 30))
    series = [time_series([period(start, end, "P1D", [7000, 6500, 7100])], business_type=t) for t in ("A60", "A61")]
    response = parse(gl_document(series))
    short_day = [p for p in response.points if p.group == "A60"][1]
    assert short_day.end_utc - short_day.start_utc == timedelta(hours=23)
    assert response.groups == ["A60", "A61"]
    assert is_complete(response.points, date(2026, 3, 29))


def test_every_group_must_cover_the_day():
    solar = time_series([period(START, END, "PT60M", [1.0] * 24)], business_type="A94", psr_type="B16")
    wind = time_series([period(START, START + timedelta(hours=12), "PT60M", [2.0] * 12)], business_type="A94",
                       psr_type="B19", domain="inBiddingZone_Domain.mRID")
    response = parse(gl_document([solar, wind]))
    assert response.groups == ["A94/B16", "A94/B19"]
    assert not is_complete(response.points, DAY)


def test_no_matching_data_is_no_data_and_not_an_error():
    response = parse(acknowledgement())
    assert response.no_data and response.points == ()
    assert "999" in response.reason


def test_other_acknowledgements_are_errors():
    with pytest.raises(EntsoeResponseError, match="acknowledgement"):
        parse(acknowledgement(code="401", message="Unauthorized"))


@pytest.mark.parametrize("content", [b"Unauthorized", b"<html><body>gateway</body></html>"])
def test_unreadable_responses_are_errors(content):
    with pytest.raises(EntsoeResponseError):
        parse(content)


def test_the_namespace_version_does_not_matter():
    series = [time_series([period(START, END, "PT60M", range(24))])]
    assert parse(gl_document(series, namespace="")).points == parse(gl_document(series)).points


def test_checksum_ignores_creation_time_and_document_id():
    series = [time_series([period(START, END, "PT60M", range(24))])]
    first = parse(gl_document(series, created="2026-11-18T08:00:00Z"))
    second = parse(gl_document(series, created="2026-11-18T09:37:00Z"))
    changed = parse(gl_document([time_series([period(START, END, "PT60M", [*range(23), 99])])]))
    assert data_checksum(first.points) == data_checksum(second.points)
    assert data_checksum(first.points) != data_checksum(changed.points)


def test_two_values_for_the_same_step_are_rejected():
    first = time_series([period(START, END, "PT60M", [1.0] * 24)])
    second = time_series([period(START, END, "PT60M", [2.0] * 24)])
    with pytest.raises(EntsoeResponseError, match="two values"):
        parse(gl_document([first, second]))


@pytest.mark.parametrize(("resolution", "values", "match"), [
    ("PT15M", [1.0] * 97, "outside"),
    ("PT7M", [1.0], "multiple"),
    ("P1Y", [1.0], "resolution"),
])
def test_malformed_periods_are_rejected(resolution, values, match):
    with pytest.raises(EntsoeResponseError, match=match):
        parse(gl_document([time_series([period(START, END, resolution, values)])]))


def test_unknown_curve_types_are_rejected():
    with pytest.raises(EntsoeResponseError, match="curve type"):
        parse(gl_document([time_series([period(START, END, "PT60M", [1.0] * 24)], curve_type="A02")]))
