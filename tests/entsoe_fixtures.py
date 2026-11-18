"""Synthetic ENTSO-E responses in the structure of the Transparency Platform REST API."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

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
