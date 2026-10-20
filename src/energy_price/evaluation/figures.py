"""Figures of a backtest, for every model: pinball loss per hour, coverage per month, coefficients.

They read ``reports/<model>/<period>_forecasts.parquet`` and ``_coefficients.csv`` (written by
``evaluation.backtest``) and compare all models of the run with the reference on the same quarter hours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import REFERENCE, common_rows
from energy_price.evaluation.scores import pinball_per_row
from energy_price.features.dataset import TARGET

# Colours and axis style of the team's model figures. Categorical slots of the reference palette
# (validated all-pairs for colour-blind readers): the reference always takes the first slot, the
# models of a run follow in their order. The quantiles use an ordinal blue ramp.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
QUANTILE_SHADES = {"q10": "#86b6ef", "q50": "#2a78d6", "q90": "#104281"}
INK, INK_SECONDARY, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July",
          8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
RC = {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "figure.facecolor": SURFACE}


def series_colours(models: list[str]) -> dict[str, str]:
    """One colour per model in the given order (reference first), at most six models per figure."""
    if len(models) > len(CATEGORICAL):
        raise ValueError(f"at most {len(CATEGORICAL)} models per figure, got {len(models)}")
    return dict(zip(models, CATEGORICAL))


def style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def scored_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Common rows of all models with the pinball loss and whether the price lies inside P10 to P90."""
    rows = common_rows(forecasts)
    return rows.assign(pinball=pinball_per_row(rows), inside=(rows[TARGET] >= rows["q10"]) & (rows[TARGET] <= rows["q90"]))


def model_order(rows: pd.DataFrame, order: list[str] | None = None) -> list[str]:
    """Models of the run, reference first, then ``order`` or the order of appearance."""
    present = list(dict.fromkeys(rows["model"]))
    order = order or [REFERENCE] + [m for m in present if m != REFERENCE]
    return [m for m in order if m in present]


def pinball_by_hour(rows: pd.DataFrame, models: list[str], period: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    colours = series_colours(models)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    by_hour = rows.groupby(["model", "hour"])["pinball"].mean().unstack(0)
    for m in models:
        ax.plot(by_hour.index, by_hour[m], color=colours[m], linewidth=2, marker="o", markersize=3.5, label=m)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour of the delivery day (local time)", color=INK_SECONDARY, fontsize=9)
    style_axes(ax, f"Mean pinball loss per hour, {period} period (lower is better)", "ct/kWh")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def coverage_by_month(rows: pd.DataFrame, models: list[str], period: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    colours = series_colours(models)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    by_month = rows.groupby(["model", rows["delivery_day"].dt.month])["inside"].mean().unstack(0) * 100
    width = 0.26
    for i, m in enumerate(models):
        x = np.arange(len(by_month)) + (i - (len(models) - 1) / 2) * (width + 0.02)
        bars = ax.bar(x, by_month[m], width=width, color=colours[m], label=m)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK_SECONDARY, padding=2,
                     bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
    ax.axhline(80, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=0.5)
    ax.annotate("target 80 %", (len(by_month) - 0.5, 80), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(by_month)), [MONTHS.get(k, str(k)) for k in by_month.index])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    style_axes(ax, f"Share of prices between P10 and P90, {period} period", "%")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def backtest_figures(out_dir: Path, period: str, fig_dir: Path, order: list[str] | None = None) -> list[Path]:
    """Pinball loss per hour and coverage per month of ``period`` as PNG in ``fig_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = scored_rows(pd.read_parquet(out_dir / f"{period}_forecasts.parquet"))
    models = model_order(rows, order)
    fig_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        return [
            pinball_by_hour(rows, models, period, fig_dir / f"{period}_pinball_by_hour.png"),
            coverage_by_month(rows, models, period, fig_dir / f"{period}_coverage_by_month.png"),
        ]


def coefficients_figure(out_dir: Path, period: str, fig_dir: Path, variant: str, shown: dict[str, str]) -> Path:
    """Coefficients of ``variant`` per quantile of the last weekly fit, in ct/kWh per unit of the feature.

    ``shown`` maps feature names of the coefficients to the labels in the figure (in this order).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coefficients = pd.read_csv(out_dir / f"{period}_coefficients.csv", parse_dates=["fit_day"])
    last = coefficients[(coefficients["model"] == variant) & (coefficients["fit_day"] == coefficients["fit_day"].max())]
    last = last.set_index("feature").loc[list(shown)]
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{period}_coefficients.png"
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        height = 0.24
        for i, q in enumerate(QUANTILE_SHADES):
            y = np.arange(len(last)) + (i - 1) * (height + 0.02)
            bars = ax.barh(y, last[q], height=height, color=QUANTILE_SHADES[q], label=q.upper().replace("Q", "P"))
            ax.bar_label(bars, fmt="%.1f", fontsize=7, color=INK_SECONDARY, padding=3)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(last)), [shown[f] for f in last.index])
        ax.invert_yaxis()
        fit_day = last["fit_day"].iloc[0].date()
        style_axes(ax, f"{variant} coefficients per quantile, fit for {fit_day} (ct/kWh per unit)", "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path
