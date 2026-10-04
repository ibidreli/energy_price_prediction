import datetime as dt

import numpy as np
import pandas as pd
import pytest

from energy_price.holidays_ch import CANTONS, cantonal_holidays
from energy_price.pv_sites import lv95_to_wgs84, snap_to_municipality, weighted_kmeans


def test_every_canton_has_the_national_day_and_new_year():
    table = cantonal_holidays([2026])

    for day in (dt.date(2026, 8, 1), dt.date(2026, 1, 1)):
        assert sorted(table.loc[table["date"] == day, "canton"]) == sorted(CANTONS)


def test_cantonal_differences_are_kept():
    table = cantonal_holidays([2026])
    zh = set(table.loc[table["canton"] == "ZH", "date"])
    ti = set(table.loc[table["canton"] == "TI", "date"])

    assert dt.date(2026, 4, 3) in zh  # Good Friday
    assert dt.date(2026, 6, 29) in ti and dt.date(2026, 6, 29) not in zh  # Saints Peter and Paul
    assert len(zh) == 9


def test_lv95_reference_point_is_bern():
    lat, lon = lv95_to_wgs84(2_600_000, 1_200_000)

    assert lat == pytest.approx(46.9511, abs=1e-3)
    assert lon == pytest.approx(7.4386, abs=1e-3)


def test_weighted_kmeans_finds_two_separated_groups():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(100, 1, (200, 2))])
    weights = np.ones(len(points))

    centres = weighted_kmeans(points, weights, k=2, seed=0)

    assert sorted(np.round(centres[:, 0])) == [0, 100]


def test_snap_moves_centres_onto_municipalities_and_shares_sum_to_one():
    pv = pd.DataFrame(
        {
            "municipality": ["Nord", "Nord", "Sued"],
            "canton": ["ZH", "ZH", "TI"],
            "kw": [10.0, 30.0, 20.0],
            "x_km": [2600.0, 2602.0, 2700.0],
            "y_km": [1250.0, 1250.0, 1100.0],
        }
    )
    sites = snap_to_municipality(np.array([[2601.0, 1249.0], [2690.0, 1110.0]]), pv)

    assert sites["site"].tolist() == ["Nord", "Sued"]
    # pv_share is rounded to 4 decimals on purpose
    assert sites["pv_share"].sum() == pytest.approx(1.0, abs=1e-4)
    assert sites.loc[0, "pv_share"] == pytest.approx(40 / 60, abs=1e-4)
