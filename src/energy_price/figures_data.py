"""Render the figures of docs/data.md from the processed data.

Usage: ``python -m energy_price.figures_data --out docs/figures``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_price.pv_sites import lv95_to_wgs84, read_pv_plants  # noqa: E402

PROCESSED = Path("data/processed")
META = Path("data/meta")
REGISTER = Path("data/external/bfe_anlagen.zip")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
BLUE = "#2a78d6"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Categorical slots 1-6 of the reference palette, validated for CVD separation. One colour per site everywhere.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


GERMAN_MONTHS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def german_month_axis(axis: matplotlib.axis.Axis) -> None:
    """Month ticks with German abbreviations, independent of the system locale."""
    axis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=range(1, 13, 2)))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: GERMAN_MONTHS[matplotlib.dates.num2date(x).month - 1]))


def site_colours(sites: pd.DataFrame) -> dict[str, str]:
    return dict(zip(sites["site"], CATEGORICAL, strict=True))


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pv_map(out: Path, sites: pd.DataFrame) -> None:
    pv = read_pv_plants(REGISTER).dropna(subset=["x_km", "y_km"])
    lat, lon = lv95_to_wgs84(pv["x_km"].to_numpy() * 1000, pv["y_km"].to_numpy() * 1000)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    hb = ax.hexbin(lon, lat, C=pv["kw"] / 1000, reduce_C_function=np.sum, gridsize=70, bins="log",
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQ_BLUES), mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("installierte PV-Leistung pro Zelle [MW, log]", color=INK_2)
    colours = site_colours(sites)
    for _, s in sites.iterrows():
        ax.scatter(s["lon"], s["lat"], s=110, color=colours[s["site"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        # Sites in the north lie close together: put the label above or below depending on the neighbours.
        neighbours = sites[(sites["site"] != s["site"]) & ((sites["lon"] - s["lon"]).abs() < 1.0) & ((sites["lat"] - s["lat"]).abs() < 0.3)]
        below = len(neighbours) > 0 and s["lat"] < neighbours["lat"].max()
        offset, va = ((0, -14), "top") if below else ((0, 12), "bottom")
        ax.annotate(f"{s['site']} ({s['pv_share']:.0%})", (s["lon"], s["lat"]), xytext=offset, textcoords="offset points",
                    ha="center", va=va, fontsize=9, color=INK, fontweight="bold", zorder=4,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})
    ax.set(title="Wo die Photovoltaik steht und welche 6 Standorte sie vertreten",
           xlabel="Längengrad", ylabel="Breitengrad")
    ax.grid(False)
    ax.set_aspect(1 / np.cos(np.deg2rad(46.8)))
    save(fig, out, "pv_map_sites.png")


def fig_site_count(out: Path) -> None:
    curve = pd.read_csv(META / "pv_site_count.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, label, fmt in (
        (axes[0], "mean_distance_km", "mittlere Distanz zur nächsten Station [km]", "{:.0f} km"),
        (axes[1], "share_within_25km", "Anteil PV-Leistung innerhalb 25 km", "{:.0%}"),
    ):
        ax.plot(curve["n_sites"], curve[col], color=BLUE, marker="o", markersize=5)
        chosen = curve[curve["n_sites"] == 6].iloc[0]
        ax.scatter([6], [chosen[col]], s=120, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"6 Standorte: {fmt.format(chosen[col])}", (6, chosen[col]), xytext=(10, 10),
                    textcoords="offset points", color=INK, fontweight="bold")
        ax.set(xlabel="Anzahl Standorte", ylabel=label, xticks=curve["n_sites"])
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("Abnehmender Nutzen jedes weiteren Standorts")
    save(fig, out, "site_count.png")


def fig_site_correlation(out: Path, weather: pd.DataFrame, sites: pd.DataFrame) -> None:
    order = sites["site"].tolist()
    daily = weather.groupby(["delivery_day", "site"])["shortwave_radiation"].mean().unstack()[order]
    corr = daily.corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, vmin=0.8, vmax=1.0, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQ_BLUES))
    for i in range(len(order)):
        for j in range(len(order)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=SURFACE if v >= 0.92 else INK)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Korrelation der Tagesmittel", color=INK_2)
    ax.set_title("Strahlungsprognosen: Mittelland ähnlich, Tessin eigenständig")
    save(fig, out, "site_correlation.png")


def fig_tsi_price(out: Path, cab: pd.DataFrame, prices: pd.DataFrame) -> None:
    m = cab.merge(prices[["timestamp_utc", "aep_ct_kwh"]], on="timestamp_utc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    shown = m[m["aep_ct_kwh"].between(-60, 60)]
    hb = ax.hexbin(shown["tsi_mw"], shown["aep_ct_kwh"], gridsize=55, bins="log", mincnt=1, linewidths=0,
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", SEQ_BLUES))
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.text(0.02, 0.96, "short: Strom fehlt", transform=ax.transAxes, color=INK_2, va="top")
    ax.text(0.98, 0.96, "long: Strom zu viel", transform=ax.transAxes, color=INK_2, va="top", ha="right")
    ax.set(xlabel="Systembilanz TSI [MW]", ylabel="Ausgleichsenergiepreis [ct/kWh]", title="Der Preis folgt der Systembilanz")
    ax.text(0.02, 0.03, f"{(~m['aep_ct_kwh'].between(-60, 60)).mean():.1%} der Viertelstunden ausserhalb ±60 ct/kWh nicht gezeigt",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    fig.colorbar(hb, ax=ax, pad=0.01).set_label("Viertelstunden [log]", color=INK_2)

    bins = [-1100, -400, -200, -100, 0, 100, 200, 400, 800]
    labels = ["< −400", "−400…−200", "−200…−100", "−100…0", "0…100", "100…200", "200…400", "> 400"]
    share = m.groupby(pd.cut(m["tsi_mw"], bins, labels=labels), observed=True)["aep_ct_kwh"].agg(lambda s: (s < 0).mean())
    ax = axes[1]
    bars = ax.bar(range(len(share)), share.to_numpy(), color=BLUE, width=0.75)
    for rect, v in zip(bars, share.to_numpy()):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(share)), labels, rotation=35, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(ylim=(0, 1), xlabel="Systembilanz TSI [MW]", ylabel="Anteil negativer Preise", title="Negative Preise nur bei long")
    ax.grid(axis="x", visible=False)
    save(fig, out, "tsi_vs_price.png")


def fig_radiation_negative(out: Path, weather: pd.DataFrame, prices: pd.DataFrame, sites: pd.DataFrame) -> None:
    w = weather.dropna(subset=["shortwave_radiation"]).assign(wt=lambda d: d["site"].map(sites.set_index("site")["pv_share"]))
    w["x"] = w["shortwave_radiation"] * w["wt"]
    agg = w.groupby("delivery_day")[["x", "wt"]].sum()
    rad = (agg["x"] / agg["wt"]).rename("rad")
    p = prices.assign(day=prices["timestamp_local"].dt.tz_localize(None).dt.normalize())
    neg = p.groupby("day")["aep_ct_kwh"].agg(lambda s: (s < 0).mean()).rename("neg")
    j = pd.concat([rad, neg], axis=1, join="inner")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.scatter(j["rad"], j["neg"], s=22, color=BLUE, alpha=0.55, edgecolor="none")
    q = j.groupby(pd.qcut(j["rad"], 4), observed=True).agg(rad=("rad", "mean"), neg=("neg", "mean"))
    ax.plot(q["rad"], q["neg"], color=INK, marker="o", markersize=7, linewidth=2)
    ax.annotate("Mittel je Viertel der Tage", (q["rad"].iloc[-1], q["neg"].iloc[-1]), xytext=(-150, 40),
                textcoords="offset points", color=INK, arrowprops={"arrowstyle": "-", "color": INK_2})
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    rank = j["rad"].rank().corr(j["neg"].rank())
    ax.set(xlabel="prognostizierte Globalstrahlung am Liefertag, PV-gewichtet [W/m²]",
           ylabel="Anteil negativer Preise am Tag",
           title=f"Sonnige Tage: mehr negative Preise, aber nur mässig (Rangkorrelation {rank:.2f}, {len(j)} Tage)")
    save(fig, out, "radiation_vs_negative.png")


def fig_holidays(out: Path) -> None:
    h = pd.read_csv(META / "holidays_ch.csv", parse_dates=["date"])
    h = h[h["date"].dt.year == 2026].groupby("date").agg(name=("name", "first"), cantons=("canton", "nunique")).reset_index()
    labels = [f"{d:%d.%m.} {n}" for d, n in zip(h["date"], h["name"])]
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(range(len(h)), h["cantons"], color=BLUE, height=0.7)
    for rect, n in zip(bars, h["cantons"]):
        ax.text(rect.get_width() + 0.3, rect.get_y() + rect.get_height() / 2, str(n), va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(h)), labels)
    ax.invert_yaxis()
    ax.set(xlim=(0, 28), xticks=[0, 13, 26], xlabel="Anzahl Kantone mit gesetzlichem Feiertag",
           title=f"Feiertage 2026: {len(h)} Tage, nur {(h['cantons'] == 26).sum()} in allen 26 Kantonen")
    ax.grid(axis="y", visible=False)
    save(fig, out, "holidays_2026.png")


def fig_coverage(out: Path, prices: pd.DataFrame, cab: pd.DataFrame, weather: pd.DataFrame) -> None:
    # (name, first day, last day) as local calendar dates, both inclusive
    rows = [
        ("Ausgleichsenergiepreis (final)", prices["timestamp_local"].min(), prices["timestamp_local"].max()),
        ("Swissgrid Regelzonenbilanz", cab["timestamp_local"].min(), cab["timestamp_local"].max()),
        ("Wetterprognosen (6 Standorte)", weather["delivery_day"].min(), weather["delivery_day"].max()),
        ("Feiertage", pd.Timestamp("2026-01-01"), pd.Timestamp("2027-12-31")),
    ]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for i, (name, start, last) in enumerate(rows):
        s = pd.Timestamp(start).tz_localize(None).normalize()
        e = min(pd.Timestamp(last).tz_localize(None).normalize() + pd.Timedelta(days=1), pd.Timestamp("2027-01-01"))
        ax.barh(i, (e - s).days, left=s, height=0.55, color=BLUE)
        ax.text(e + pd.Timedelta(days=3), i, f"bis {pd.Timestamp(last).strftime('%d.%m.%Y')}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-31"))
    german_month_axis(ax.xaxis)
    ax.grid(axis="y", visible=False)
    ax.set_title("Abdeckung der Daten ab 2026 (Stand der Erhebung)")
    save(fig, out, "coverage.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    style()

    sites = pd.read_csv(META / "weather_sites.csv")
    prices = pd.read_parquet(PROCESSED / "balance_prices.parquet")
    prices = prices[prices["regime"] == "single_price"]
    cab = pd.read_parquet(PROCESSED / "control_area_balance.parquet")
    weather = pd.read_parquet(PROCESSED / "weather_forecasts.parquet")

    fig_pv_map(args.out, sites)
    fig_site_count(args.out)
    fig_site_correlation(args.out, weather, sites)
    fig_tsi_price(args.out, cab, prices)
    fig_radiation_negative(args.out, weather, prices, sites)
    fig_holidays(args.out)
    fig_coverage(args.out, prices, cab, weather)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
