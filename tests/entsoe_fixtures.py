"""Synthetic responses in the structure of the ENTSO-E REST API and the Energy-Charts API v2."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ZURICH = ZoneInfo("Europe/Zurich")

GL_NAMESPACE = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
PUBLICATION_NAMESPACE = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
ACK_NAMESPACE = "urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"


def stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def period(start: datetime, end: datetime, resolution: str, values: Sequence[float | None],
           value_tag: str = "quantity") -> str:
    """A ``Period``; ``None`` leaves a position out (a gap in A01, a repeat in A03)."""
    points = "".join(
        f"<Point><position>{i}</position><{value_tag}>{v}</{value_tag}></Point>"
        for i, v in enumerate(values, start=1) if v is not None
    )
    return (f"<Period><timeInterval><start>{stamp(start)}</start><end>{stamp(end)}</end></timeInterval>"
            f"<resolution>{resolution}</resolution>{points}</Period>")


def time_series(periods: Sequence[str], business_type: str = "A04", psr_type: str = "",
                curve_type: str = "A01", domain: str = "outBiddingZone_Domain.mRID") -> str:
    psr = f"<MktPSRType><psrType>{psr_type}</psrType></MktPSRType>" if psr_type else ""
    return (f"<TimeSeries><mRID>1</mRID><businessType>{business_type}</businessType>"
            f'<{domain} codingScheme="A01">10YCH-SWISSGRIDZ</{domain}>'
            f"<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name><curveType>{curve_type}</curveType>"
            f"{psr}{''.join(periods)}</TimeSeries>")


def gl_document(series: Sequence[str], created: str = "2026-11-18T09:00:00Z", namespace: str = GL_NAMESPACE) -> bytes:
    """A ``GL_MarketDocument`` (load and generation) with the given ``TimeSeries``."""
    xmlns = f' xmlns="{namespace}"' if namespace else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?><GL_MarketDocument{xmlns}><mRID>{created}</mRID>'
            f"<revisionNumber>1</revisionNumber><type>A65</type><createdDateTime>{created}</createdDateTime>"
            f"{''.join(series)}</GL_MarketDocument>").encode("utf-8")


def price_document(series: Sequence[str], created: str = "2026-11-18T09:00:00Z") -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?><Publication_MarketDocument xmlns="{PUBLICATION_NAMESPACE}">'
            f"<mRID>{created}</mRID><type>A44</type><createdDateTime>{created}</createdDateTime>"
            f"{''.join(series)}</Publication_MarketDocument>").encode("utf-8")


def price_series(periods: Sequence[str], curve_type: str = "A03") -> str:
    return (f"<TimeSeries><mRID>1</mRID><auction.type>A01</auction.type><businessType>A62</businessType>"
            f"<currency_Unit.name>EUR</currency_Unit.name><price_Measure_Unit.name>MWH</price_Measure_Unit.name>"
            f"<curveType>{curve_type}</curveType>{''.join(periods)}</TimeSeries>")


def acknowledgement(code: str = "999", message: str = "No matching data found") -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?><Acknowledgement_MarketDocument xmlns="{ACK_NAMESPACE}">'
            f"<mRID>x</mRID><createdDateTime>2026-11-18T09:00:00Z</createdDateTime>"
            f"<Reason><code>{code}</code><text>{message}</text></Reason>"
            f"</Acknowledgement_MarketDocument>").encode("utf-8")


def ec_document(start: datetime, end: datetime, step: timedelta, value: Callable[[int], float | None] = lambda i: 1.0,
                series_id: str = "load", resolution: str | None = None,
                generated: str = "2026-11-24T10:00:00+01:00") -> bytes:
    """An Energy-Charts v2 response with one row per ``step`` from ``start`` to ``end`` (local offsets).

    ``value`` gets the row index; ``None`` writes a null value. ``resolution`` overrides the header.
    """
    rows, stamp, i = [], start, 0
    while stamp < end:
        rows.append({"timestamp": stamp.astimezone(ZURICH).isoformat(), "values": {series_id: value(i)}})
        stamp, i = stamp + step, i + 1
    minutes = int(step.total_seconds() // 60)
    header = resolution or f"PT{minutes}M"
    return json.dumps({
        "schema_version": "2.0", "endpoint": "public_power_forecast", "country": "ch", "timezone": "Europe/Zurich",
        "resolution": header, "interval_minutes": minutes, "unit": "MW", "generated_at": generated,
        "series": [{"id": series_id, "name": series_id}], "data": rows,
    }).encode("utf-8")
