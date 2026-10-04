"""Choose the weather forecast sites from the location of the installed photovoltaic capacity.

Source: BFE register of all Swiss electricity production plants (opendata, LV95 coordinates).
The sites are the centres of a capacity-weighted k-means over all PV plants, each moved to the
nearest municipality with PV plants so that no site ends up in a lake or on a glacier.

Usage: ``python -m energy_price.pv_sites --register data/external/bfe_anlagen.zip --out data/meta``
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REGISTER_URL = "https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip"
PV_SUBCATEGORY = "subcat_2"
N_SITES = 6
MAX_SITES_EVALUATED = 12
SEEDS = range(5)


def lv95_to_wgs84(east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate swisstopo formula, accurate to about 1 m, sufficient for weather grid cells."""
    y = (np.asarray(east) - 2_600_000) / 1e6
    x = (np.asarray(north) - 1_200_000) / 1e6
    lon = 2.6779094 + 4.728982 * y + 0.791484 * y * x + 0.1306 * y * x**2 - 0.0436 * y**3
    lat = 16.9023892 + 3.238272 * x - 0.270978 * y**2 - 0.002528 * x**2 - 0.0447 * y**2 * x - 0.0140 * x**3
    return lat * 100 / 36, lon * 100 / 36


def read_pv_plants(register_zip: str | Path) -> pd.DataFrame:
    """PV plants with capacity in kW and LV95 coordinates in km (NaN if the register has none)."""
    with zipfile.ZipFile(register_zip) as z:
        plants = pd.read_csv(z.open("ElectricityProductionPlant.csv"), encoding="utf-8-sig")
    pv = plants[plants["SubCategory"] == PV_SUBCATEGORY]
    return pd.DataFrame(
        {
            "canton": pv["Canton"],
            "municipality": pv["Municipality"],
            "kw": pv["TotalPower"].astype(float),
            "x_km": pv["_x"] / 1000,
            "y_km": pv["_y"] / 1000,
        }
    ).reset_index(drop=True)


def weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, seed: int, iters: int = 50) -> np.ndarray:
    """Capacity-weighted k-means with k-means++ initialisation. Returns the k centres."""
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), size=1, p=weights / weights.sum())]
    for _ in range(1, k):
        d2 = _sq_distances(points, centres).min(axis=1)
        p = weights * d2
        centres = np.vstack([centres, points[rng.choice(len(points), p=p / p.sum())]])
    for _ in range(iters):
        labels = _sq_distances(points, centres).argmin(axis=1)
        moved = np.array([np.average(points[labels == j], axis=0, weights=weights[labels == j]) for j in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    return centres


def mean_distance_km(points: np.ndarray, weights: np.ndarray, centres: np.ndarray) -> float:
    """Capacity-weighted mean distance from a PV plant to its nearest site."""
    return float(np.average(np.sqrt(_sq_distances(points, centres).min(axis=1)), weights=weights))


def best_centres(points: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    return min((weighted_kmeans(points, weights, k, s) for s in SEEDS), key=lambda c: mean_distance_km(points, weights, c))


def snap_to_municipality(centres: np.ndarray, pv: pd.DataFrame) -> pd.DataFrame:
    """Move each centre to the capacity-weighted centre of the nearest municipality with PV plants."""
    towns = (
        pv.assign(wx=pv["x_km"] * pv["kw"], wy=pv["y_km"] * pv["kw"])
        .groupby(["municipality", "canton"], as_index=False)[["wx", "wy", "kw"]]
        .sum()
    )
    towns["x_km"] = towns["wx"] / towns["kw"]
    towns["y_km"] = towns["wy"] / towns["kw"]
    town_xy = towns[["x_km", "y_km"]].to_numpy()

    labels = _sq_distances(pv[["x_km", "y_km"]].to_numpy(), centres).argmin(axis=1)
    rows = []
    for j, centre in enumerate(centres):
        nearest = int(_sq_distances(town_xy, centre[None, :])[:, 0].argmin())
        town = towns.iloc[nearest]
        lat, lon = lv95_to_wgs84(town["x_km"] * 1000, town["y_km"] * 1000)
        c_lat, c_lon = lv95_to_wgs84(centre[0] * 1000, centre[1] * 1000)
        rows.append(
            {
                "site": town["municipality"],
                "canton": town["canton"],
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "pv_share": round(float(pv.loc[labels == j, "kw"].sum() / pv["kw"].sum()), 4),
                "centre_lat": round(float(c_lat), 4),
                "centre_lon": round(float(c_lon), 4),
                "snap_distance_km": round(float(np.sqrt(((town_xy[nearest] - centre) ** 2).sum())), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("pv_share", ascending=False, ignore_index=True)


def _sq_distances(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--register", type=Path, required=True, help="BFE register zip")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)

    pv = read_pv_plants(args.register)
    located = pv.dropna(subset=["x_km", "y_km"]).reset_index(drop=True)
    points, weights = located[["x_km", "y_km"]].to_numpy(), located["kw"].to_numpy()

    curve = []
    for k in range(1, MAX_SITES_EVALUATED + 1):
        centres = best_centres(points, weights, k)
        nearest = np.sqrt(_sq_distances(points, centres).min(axis=1))
        curve.append(
            {
                "n_sites": k,
                "mean_distance_km": round(mean_distance_km(points, weights, centres), 2),
                "share_within_25km": round(float(weights[nearest <= 25].sum() / weights.sum()), 4),
            }
        )
        if k == N_SITES:
            sites = snap_to_municipality(centres, located)

    by_canton = (pv.groupby("canton")["kw"].sum() / 1000).sort_values(ascending=False).rename("mw").reset_index()
    by_canton["share"] = (by_canton["mw"] / by_canton["mw"].sum()).round(4)

    args.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.out / "weather_sites.csv", index=False)
    pd.DataFrame(curve).to_csv(args.out / "pv_site_count.csv", index=False)
    by_canton.round({"mw": 1}).to_csv(args.out / "pv_capacity_by_canton.csv", index=False)
    (args.out / "pv_register_summary.json").write_text(
        json.dumps(
            {
                "source": REGISTER_URL,
                "pv_plants": len(pv),
                "pv_mw": round(pv["kw"].sum() / 1000, 1),
                "share_of_capacity_without_coordinates": round(1 - weights.sum() / pv["kw"].sum(), 4),
            },
            indent=2,
        )
        + "\n"
    )
    print(sites.to_string(index=False))


if __name__ == "__main__":
    main()
