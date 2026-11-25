"""Issue #37 data for the models: every stored version, and the version known at the issue time.

The data (load and generation forecasts, actual load, day-ahead prices) comes from two sources, each in
its own folder below ``data/entsoe``: ``energy_charts`` (public, no key) and ``entsoe_api`` (ENTSO-E
REST API, token). ``load_entsoe`` reads the live snapshots (``<source>/snapshots``) and the one-time
history (``<source>/backfill``) of both into one long table: one row per quarter hour, source
(``provider``), series, zone, group and version. A value of a coarser resolution (hourly values, the
daily minimum and maximum of the week-ahead forecast) is repeated for each of its quarter hours. Where
a response holds the same group in two resolutions, the finer one is kept. The ENTSO-E document A69
is split into ``solar_day_ahead`` and ``wind_day_ahead``, so both sources have the same series.
Units: MW for load and generation, EUR/MWh for prices.

``known_at_issue_entsoe`` selects what was known when the forecast for delivery day D is issued, for
each source on its own:

- Never a value of a later local day than the series allows (``LATEST_DAY``): forecasts up to D, the
  actual load up to D-2, day-ahead prices up to D-1. The Swiss price of D appears at D-1 11:10.
- Live period of the source (a live run of it happened by the issue time): only its live snapshots
  fetched by D-1 11:00, of each value the newest. A day without such a snapshot has no value. The
  backfill never fills the gap, because it was downloaded later.
- Before: backfill values of the series that ``HISTORY_RULES[source]`` allows, of each value the
  earliest download. This transfers the result of the live check to the history. It is an
  assumption, like the D-2 limit for Swissgrid (docs/data.md, sections 1 and 7).

``availability`` evaluates the fetch logs against the criterion of docs/data.md, section 7.

Usage: ``python -m energy_price.data.entsoe --log data/entsoe/*/snapshots/fetch_log.csv --out data/entsoe/availability.csv``
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from energy_price.availability import issue_time
from energy_price.data import energy_charts_json, entsoe_xml
from energy_price.data.fetch_energy_charts import series_by_key, series_id

TZ = "Europe/Zurich"
QUARTER_HOUR = pd.Timedelta(minutes=15)
PROVIDERS = ("energy_charts", "entsoe_api")
SOURCES = {"live": "snapshots", "backfill": "backfill"}

# Latest local day relative to the delivery day D that a series may contribute. ``wind_solar_day_ahead``
# is the request of the ENTSO-E API; it appears only in the fetch log, the table splits it.
LATEST_DAY = {
    "load_day_ahead": 0,
    "load_week_ahead": 0,
    "solar_day_ahead": 0,
    "wind_day_ahead": 0,
    "wind_solar_day_ahead": 0,
    "load_actual": -2,
    "price_day_ahead": -1,
}
# May the backfill be used before the live period? Decided per source from the live check
# (docs/data.md, section 7). Until then only the day-ahead prices, whose auction for D-1 closes on D-2.
PENDING = {"load_day_ahead": False, "load_week_ahead": False, "solar_day_ahead": False, "wind_day_ahead": False,
           "load_actual": False, "price_day_ahead": True}
HISTORY_RULES = {"energy_charts": dict(PENDING), "entsoe_api": dict(PENDING)}
# ENTSO-E production types of the A69 document: solar, wind offshore, wind onshore.
ENTSOE_GENERATION = {"B16": "solar_day_ahead", "B18": "wind_day_ahead", "B19": "wind_day_ahead"}
COLUMNS = ["provider", "source", "series", "zone", "fetched_at_utc", "group", "resolution",
           "timestamp_utc", "timestamp_local", "local_day", "value", "snapshot"]
VALUE_KEY = ["provider", "series", "zone", "group", "timestamp_utc"]
ANSWERED = {"ok", "no_data"}


def empty() -> pd.DataFrame:
    utc, local = "datetime64[us, UTC]", f"datetime64[us, {TZ}]"
    dtypes = {"fetched_at_utc": utc, "timestamp_utc": utc, "timestamp_local": local, "local_day": "datetime64[us]",
              "value": float}
    return pd.DataFrame({c: pd.Series([], dtype=dtypes.get(c, object)) for c in COLUMNS})


def parse_snapshot(path: Path, row: Mapping[str, str]) -> entsoe_xml.Response:
    if row["provider"] == "energy_charts":
        return energy_charts_json.parse(path.read_bytes(), series_id(series_by_key()[(row["series"], row["zone"])]))
    if row["provider"] == "entsoe_api":
        return entsoe_xml.parse(path.read_bytes())
    raise ValueError(f"{path.name}: unknown provider {row['provider']!r}")


def quarter_hours(points: tuple[entsoe_xml.Point, ...]) -> pd.DataFrame:
    """One row per quarter hour and group; the finer resolution wins where a group comes in two."""
    frame = pd.DataFrame({
        "group": [p.group for p in points],
        "resolution": [p.resolution for p in points],
        "start": pd.to_datetime([p.start_utc for p in points], utc=True),
        "end": pd.to_datetime([p.end_utc for p in points], utc=True),
        "value": [p.value for p in points],
    })
    steps = ((frame["end"] - frame["start"]) / QUARTER_HOUR).astype(int)
    rows = frame.loc[frame.index.repeat(steps)].reset_index(drop=True)
    rows["steps"] = steps.repeat(steps).to_numpy()
    rows["timestamp_utc"] = rows["start"] + rows.groupby(frame.index.repeat(steps)).cumcount().to_numpy() * QUARTER_HOUR
    rows = rows.sort_values(["steps", "timestamp_utc"], kind="stable").drop_duplicates(["group", "timestamp_utc"])
    return rows.sort_values(["group", "timestamp_utc"], ignore_index=True)


def read_snapshot(path: Path, row: Mapping[str, str]) -> pd.DataFrame:
    """Quarter hours of one stored response; its values must match the checksum in the manifest."""
    if row["source"] not in SOURCES:
        raise ValueError(f"{path.name}: unknown source {row['source']!r}")
    response = parse_snapshot(path, row)
    if entsoe_xml.data_checksum(response.points) != row["data_sha256"]:
        raise ValueError(f"{path.name}: values differ from manifest.csv")
    if not response.points:
        return empty()
    rows = quarter_hours(response.points)
    series = pd.Series(row["series"], index=rows.index)
    if row["series"] == "wind_solar_day_ahead":
        series = rows["group"].map(lambda group: ENTSOE_GENERATION.get(group.partition("/")[2]))
    out = pd.DataFrame({
        "provider": row["provider"], "source": row["source"], "series": series, "zone": row["zone"],
        "fetched_at_utc": pd.Timestamp(row["fetched_at_utc"]).tz_convert("UTC"),
        "group": rows["group"], "resolution": rows["resolution"], "timestamp_utc": rows["timestamp_utc"],
        "value": rows["value"], "snapshot": path.name,
    })[lambda frame: frame["series"].notna()]
    out["timestamp_local"] = out["timestamp_utc"].dt.tz_convert(TZ)
    out["local_day"] = out["timestamp_local"].dt.tz_localize(None).dt.normalize()
    return out[COLUMNS].astype(empty().dtypes.to_dict())


def load_entsoe(root: str | Path) -> pd.DataFrame:
    """All versions of both sources below ``root``; empty if nothing was fetched yet."""
    frames = []
    for provider in PROVIDERS:
        for folder in SOURCES.values():
            manifest = Path(root) / provider / folder / "manifest.csv"
            if not manifest.exists():
                continue
            meta = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            frames += [read_snapshot(manifest.parent / row["file"], row) for row in meta.to_dict("records")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty()
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["provider", "source", "series", "zone", "fetched_at_utc", "group", "timestamp_utc"],
                             ignore_index=True)


def live_starts(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Per source, the fetch time of its first live snapshot (sources without live snapshots are missing)."""
    live = frame[frame["source"] == "live"]
    return live.groupby("provider")["fetched_at_utc"].min().to_dict()


def known_at_issue_entsoe(frame: pd.DataFrame, delivery_day: dt.date,
                          live_from: Mapping[str, pd.Timestamp] | str = "auto") -> pd.DataFrame:
    """Rows of ``load_entsoe`` known when the forecast for ``delivery_day`` is issued (module docstring).

    ``live_from`` holds the start of the live period per source, by default ``live_starts(frame)``.
    Pass it when ``frame`` is only a window of the full table, so the window cannot move the start.
    """
    for column in ["fetched_at_utc", "timestamp_utc", "timestamp_local"]:
        if not isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            raise TypeError(f"{column} must be timezone-aware to compare it with the issue time")
    unknown = set(frame["series"]) - set(LATEST_DAY)
    if unknown:
        raise ValueError(f"no availability rule for series {sorted(unknown)}")
    unknown = set(frame["provider"]) - set(HISTORY_RULES)
    if unknown:
        raise ValueError(f"no history rules for sources {sorted(unknown)}")
    starts = live_starts(frame) if isinstance(live_from, str) else live_from
    cutoff = issue_time(delivery_day)
    latest = pd.Timestamp(delivery_day) + pd.to_timedelta(frame["series"].map(LATEST_DAY), unit="D")
    parts = []
    for provider, rows in frame[frame["local_day"] <= latest].groupby("provider"):
        start = starts.get(provider)
        if start is not None and start <= cutoff:
            rows = rows[(rows["source"] == "live") & (rows["fetched_at_utc"] <= cutoff)]
            # Per value, not per snapshot: a response may hold days next to the requested one.
            keep = rows.groupby(VALUE_KEY)["fetched_at_utc"].transform("max")
        else:
            rows = rows[(rows["source"] == "backfill") & rows["series"].map(HISTORY_RULES[provider]).astype(bool)]
            keep = rows.groupby(VALUE_KEY)["fetched_at_utc"].transform("min")
        parts.append(rows[rows["fetched_at_utc"] == keep])
    return pd.concat(parts) if parts else frame.iloc[:0]


def availability(log: pd.DataFrame) -> pd.DataFrame:
    """Per source, series, zone and delivery day D: was the day D needs complete in a run before D-1 11:00?

    The needed day is D for forecasts, D-2 for the actual load and D-1 for prices (``LATEST_DAY``).
    Only delivery days whose issue time lies between the first and the last logged live request of the
    source count. A day is ``checked`` if a request before the issue time got an answer (ok or no_data).
    ``first_complete_local`` is the first run with the whole needed day, also after the issue time.
    """
    live = log[log["source"] == "live"].copy()
    columns = ["provider", "series", "zone", "delivery_day", "needed_day", "checked", "complete_before_issue",
               "runs_before_issue", "statuses_before_issue", "first_complete_local"]
    if live.empty:
        return pd.DataFrame(columns=columns)
    live["fetched_at_utc"] = pd.to_datetime(live["fetched_at_utc"], utc=True, format="ISO8601")
    live["needed_day"] = pd.to_datetime(live["delivery_day"])
    live["delivery_day"] = live["needed_day"] - pd.to_timedelta(live["series"].map(LATEST_DAY), unit="D")
    issue = (live["delivery_day"] - pd.Timedelta(hours=13)).dt.tz_localize(TZ)  # D-1 11:00 local
    live["before"] = live["fetched_at_utc"] <= issue
    live["complete"] = live["complete"].astype(str).str.lower() == "true"
    live["answered"] = live["before"] & live["status"].isin(ANSWERED)
    live["good"] = live["before"] & live["complete"]

    frames = []
    for provider, source_rows in live.groupby("provider"):
        first, last = source_rows["fetched_at_utc"].min(), source_rows["fetched_at_utc"].max()
        days = pd.date_range(pd.Timestamp(first.tz_convert(TZ).date()) + pd.Timedelta(days=1),
                             pd.Timestamp(last.tz_convert(TZ).date()) + pd.Timedelta(days=1), freq="D")
        days = pd.DatetimeIndex([d for d in days if first <= (d - pd.Timedelta(hours=13)).tz_localize(TZ) <= last])
        if days.empty:  # no issue time between the first and the last run yet
            continue
        for (series, zone), rows in source_rows.groupby(["series", "zone"]):
            before = rows[rows["before"]]
            by_day = rows.groupby("delivery_day")
            statuses = before.groupby("delivery_day")["status"].agg(lambda s: "+".join(sorted(set(s))))
            complete = rows[rows["complete"]].groupby("delivery_day")["fetched_at_utc"].min().dt.tz_convert(TZ)
            frames.append(pd.DataFrame({
                "provider": provider, "series": series, "zone": zone, "delivery_day": days,
                "needed_day": days + pd.Timedelta(days=LATEST_DAY[series]),
                "checked": by_day["answered"].any().reindex(days, fill_value=False).to_numpy(),
                "complete_before_issue": by_day["good"].any().reindex(days, fill_value=False).to_numpy(),
                "runs_before_issue": before.groupby("delivery_day").size().reindex(days, fill_value=0).to_numpy(),
                "statuses_before_issue": statuses.reindex(days, fill_value="").to_numpy(),
                "first_complete_local": complete.reindex(days).array,
            }, columns=columns))
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["provider", "series", "zone", "delivery_day"], ignore_index=True)


def availability_summary(report: pd.DataFrame, min_days: int = 7) -> pd.DataFrame:
    """Per source, series and zone: checked days, of these complete before 11:00, and the criterion of docs/data.md."""
    keys = ["provider", "series", "zone"]
    grouped = report.groupby(keys)
    summary = pd.DataFrame({
        "days": grouped.size(),
        "checked": grouped["checked"].sum(),
        "complete_before_issue": grouped.apply(lambda r: (r["checked"] & r["complete_before_issue"]).sum(),
                                               include_groups=False),
        "unchecked_days": grouped.apply(lambda r: ", ".join(d.strftime("%m-%d") for d in r.loc[~r["checked"], "delivery_day"]),
                                        include_groups=False),
    })
    checked = report[report["checked"]].groupby(keys)["complete_before_issue"].all()
    summary["meets_criterion"] = (summary["checked"] >= min_days) & checked.reindex(summary.index).fillna(False)
    return summary.reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the availability of the issue #37 data at D-1 11:00.")
    parser.add_argument("--log", type=Path, nargs="*", default=[], help="fetch_log.csv of the live job, per source")
    parser.add_argument("--out", type=Path, required=True, help="CSV with one row per source, series, zone and day")
    args = parser.parse_args(argv)
    logs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in args.log if path.exists()]
    if not logs:
        print("no fetch log yet: the scheduled job has not run (docs/data.md, section 7)")
        return
    report = availability(pd.concat(logs, ignore_index=True))
    if report.empty:
        print("no delivery day to evaluate yet: it needs a run before and a run after an issue time D-1 11:00")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(availability_summary(report).to_string(index=False))
    print(f"wrote {args.out}: {len(report)} rows")


if __name__ == "__main__":
    main()
