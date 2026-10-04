"""What is known when the forecast for delivery day D is issued (D-1, 11:00 Swiss local time).

Swissgrid data (balance energy prices and the control area balance) is assumed to be known up to
the end of D-2. ASSUMPTION, not yet verified: the file is updated daily. It was observed once
(Sunday 2026-10-04, 03:21 Swiss time, data until 2026-10-03 23:45); Swissgrid's web page speaks
of weekly publication. ``data/control_area_balance/snapshots/fetch_log.csv`` collects the
evidence (see docs/data.md). If the file turns out to lag more, move ``swissgrid_cutoff`` back.

Use ``known_at_issue`` to select the history for a delivery day. Do not shift by a fixed lag:
the allowed distance depends on the target quarter hour (24 h 15 min for D 00:00, 48 h for
D 23:45). A fixed 35 h lag, for example, takes D-1 12:45 for the target D 23:45, which lies
after the issue time.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

TZ = "Europe/Zurich"
ISSUE_TIME_LOCAL = dt.time(11, 0)


def issue_time(day: dt.date) -> pd.Timestamp:
    """Time the forecast for delivery day ``day`` is issued: D-1, 11:00 Swiss local time."""
    return pd.Timestamp(dt.datetime.combine(day - dt.timedelta(days=1), ISSUE_TIME_LOCAL)).tz_localize(TZ)


def swissgrid_cutoff(day: dt.date) -> pd.Timestamp:
    """First quarter hour that is NOT yet known for delivery day ``day``: D-1, 00:00 Swiss local time."""
    return pd.Timestamp(day - dt.timedelta(days=1)).tz_localize(TZ)


def known_at_issue(frame: pd.DataFrame, day: dt.date, time_column: str = "timestamp_local") -> pd.DataFrame:
    """Rows of a Swissgrid table that were published when the forecast for ``day`` is issued.

    The values are the latest version Swissgrid published, which for past months are final
    settlement values; at the issue time only provisional values existed (see docs/data.md).
    """
    stamps = frame[time_column]
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise TypeError(f"{time_column} must be timezone-aware to compare it with the cutoff")
    return frame[stamps < swissgrid_cutoff(day)]
