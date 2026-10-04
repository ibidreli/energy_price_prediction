"""Helpers that write synthetic files in the structure of the Swissgrid price XML."""

from pathlib import Path

import pandas as pd

from energy_price.swissgrid import TZ


def write_xml(path: Path, series: dict[str, list[tuple[str, str]]], unit: str = "ct/kWh") -> Path:
    """Write a minimal file in the structure of the Swissgrid price XML (ISO-8859-1, like the originals)."""
    parts = [
        '<?xml version="1.0" encoding="ISO-8859-1" ?><template><timeSeriesGroup>',
        "<Report_Titel_de>Preise für Ausgleichsenergie</Report_Titel_de><timeSeriesGroupLink>",
    ]
    for name, rows in series.items():
        parts.append(f"<timeSeries><Report_TS_Titel_de>{name}</Report_TS_Titel_de><unit>{unit}</unit><timeSeriesData>")
        parts += [f"<DATAVALUE><TIME>{t}</TIME><VALUE>{v}</VALUE></DATAVALUE>" for t, v in rows]
        parts.append("</timeSeriesData></timeSeries>")
    parts.append("</timeSeriesGroupLink></timeSeriesGroup></template>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(parts).encode("iso-8859-1"))
    return path


def month_rows(year: int, month: int, value: str = "1") -> list[tuple[str, str]]:
    """All quarter hours of one local calendar month, with offsets as Swissgrid writes them."""
    start = pd.Timestamp(year=year, month=month, day=1, tz=TZ)
    stamps = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="15min", inclusive="left")
    return [(t.isoformat(), value) for t in stamps]
