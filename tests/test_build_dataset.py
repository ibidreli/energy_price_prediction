import pandas as pd
import pytest

from energy_price import build_dataset, checksums
from energy_price.swissgrid import OUTPUT_COLUMNS, SwissgridFormatError, load_prices
from xml_fixtures import month_rows, write_xml


def test_main_writes_dataset_and_summary(tmp_path, capsys):
    raw = tmp_path / "raw"
    write_xml(raw / "2601.xml", {"BG-AEP": month_rows(2026, 1, "5.0")})
    out = tmp_path / "processed" / "prices.parquet"

    build_dataset.main(["--raw", str(raw), "--out", str(out)])

    written = pd.read_parquet(out)
    assert list(written.columns) == OUTPUT_COLUMNS
    pd.testing.assert_frame_equal(written, load_prices(raw))
    assert not out.with_suffix(".parquet.tmp").exists()
    assert "2976 quarter hours" in capsys.readouterr().out


def test_main_keeps_previous_dataset_when_raw_data_is_invalid(tmp_path):
    raw = tmp_path / "raw"
    write_xml(raw / "2601.xml", {"BG-AEP": month_rows(2026, 1)})
    out = tmp_path / "prices.parquet"
    build_dataset.main(["--raw", str(raw), "--out", str(out)])
    before = out.read_bytes()

    write_xml(raw / "copy.xml", {"BG-AEP": month_rows(2026, 1)})
    with pytest.raises(SwissgridFormatError, match="duplicate"):
        build_dataset.main(["--raw", str(raw), "--out", str(out)])

    assert out.read_bytes() == before


def test_checksums_change_with_content_not_with_modification_time(tmp_path):
    f = write_xml(tmp_path / "2601.xml", {"BG-AEP": month_rows(2026, 1)})
    before = checksums.checksums(tmp_path)

    f.touch()
    assert checksums.checksums(tmp_path) == before

    write_xml(f, {"BG-AEP": month_rows(2026, 1, "2")})
    assert checksums.checksums(tmp_path) != before


def test_checksums_of_empty_directory_is_empty(tmp_path):
    assert checksums.checksums(tmp_path) == []


def test_checksums_only_cover_files_matching_the_pattern(tmp_path):
    (tmp_path / "run_2026-01-14T18.json").write_text("{}")
    (tmp_path / "manifest.csv").write_text("x")

    lines = checksums.checksums(tmp_path, pattern="run_*.json")

    assert len(lines) == 1 and lines[0].endswith("run_2026-01-14T18.json")
