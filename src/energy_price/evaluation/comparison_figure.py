"""Plot paired daily loss intervals and coverage from existing score reports."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.evaluation.figures import AXIS, CATEGORICAL, GRID, INK_SECONDARY, RC, style_axes


def comparison_figure(out: Path, period: str, path: Path) -> Path:
    """Main calibrated comparison only; this function never fits or selects."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pd.read_csv(out / f"{period}_comparisons.csv")
    pairs = pairs[pairs.model.str.startswith("calibrated_")].reset_index(drop=True)
    scores = pd.read_csv(out / f"{period}_scores.csv", index_col=0)
    if pairs.empty or pairs.comparator.nunique() != 1:
        raise ValueError("expected calibrated pairs with one common comparator")
    comparator = pairs.comparator.iloc[0]
    labels = [name.removeprefix("calibrated_") for name in pairs.model]
    y = np.arange(len(pairs))
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig, (loss, coverage) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
        for i, row in pairs.iterrows():
            loss.plot([row.ci95_low, row.ci95_high], [i, i], color=CATEGORICAL[0], linewidth=3)
            loss.scatter(row.mean_difference, i, color=CATEGORICAL[0], s=40, zorder=3)
            cov = 100 * scores.loc[row.model, "coverage_80"]
            coverage.scatter(cov, i, color=CATEGORICAL[0], s=40, zorder=3)
            coverage.annotate(f"{cov:.2f}%", (cov, i), xytext=(6, 5), textcoords="offset points", fontsize=8)
        loss.axvline(0, color=AXIS, linewidth=1)
        base = 100 * scores.loc[comparator, "coverage_80"]
        coverage.axvline(base, color=CATEGORICAL[1], linestyle="--", label=f"Standard: {base:.2f}%")
        coverage.axvline(80, color=AXIS, linestyle=":", label="Nominal: 80%")
        for ax in (loss, coverage):
            style_axes(ax, "", "")
            ax.set_yticks(y, labels if ax is loss else [""] * len(y))
            ax.set_ylim(len(y) - .5, -.5)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=.8)
        loss.set_title("Gepaarte tägliche Pinball-Differenz mit 95%-Intervall", loc="left", fontsize=10)
        loss.set_xlabel("ct/kWh · links von null ist besser", fontsize=9)
        coverage.set_title("Coverage des P10–P90-Bands", loc="left", fontsize=10)
        coverage.set_xlabel("Anteil abgedeckter Viertelstunden (%)", fontsize=9)
        coverage.set_xlim(min(base, 100 * scores.loc[pairs.model, "coverage_80"].min()) - .5, max(80, base) + 1)
        coverage.legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(f"{period}: kalibrierte Kandidaten gegen kalibriertes Standardmodell", x=.02, ha="left", fontsize=12)
        note = ("Test ist Entwicklungsvergleich, kein blinder Nachweis" if period == "test"
                else "Einmaliger blinder Holdout; keine nachträgliche Anpassung")
        fig.text(.02, .015, "7-Kalendertage-Bootstrap · 5000 Wiederholungen · " + note,
                 fontsize=8, color=INK_SECONDARY)
        fig.tight_layout(rect=(0, .07, 1, .92))
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return path


def routing_schedule_figure(out: Path, period: str, path: Path) -> Path:
    """Show 24 independent clock-hour decisions for each weekly refit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    table = pd.read_csv(out / f"{period}_routing_schedule.csv", parse_dates=["fit_day"])
    table = table[(table.strategy == "selection") & (table.end_hour - table.start_hour == 1)]
    if table.empty:
        raise ValueError("no hourly selection schedule available")
    names = [c.removeprefix("history_pinball:") for c in table.columns if c.startswith("history_pinball:")]
    codes = {name: i for i, name in enumerate(names)}
    colours = (CATEGORICAL[:len(names)] if len(names) <= len(CATEGORICAL)
               else [plt.get_cmap("tab20")(i / len(names)) for i in range(len(names))])
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(len(names) + 1) - .5, len(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for ax, column, title in zip(axes, ("empirical_best", "chosen_expert"),
                                     ("Historisch bestes Modell je Uhrzeit", "Tatsächlich gewähltes Modell je Uhrzeit")):
            grid = table.pivot(index="fit_day", columns="start_hour", values=column).sort_index().reindex(columns=range(24))
            values = grid.apply(lambda series: series.map(codes)).to_numpy(float)
            ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(grid)), grid.index.strftime("%d.%m."), fontsize=8)
            ax.set_ylabel("Wochenbeginn", fontsize=9)
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=8)
            ax.tick_params(length=0)
        axes[-1].set_xlabel("Schweizer Uhrzeit · eine Auswahl für alle vier Viertelstunden", fontsize=9)
        fig.suptitle("Tageszeit-Routing: 24 Entscheidungen, wöchentlich neu bestimmt", x=.04, ha="left", fontsize=13)
        fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colours, names)],
                   loc="lower center", bbox_to_anchor=(.5, .045), ncol=min(4, len(names)), frameon=False)
        fig.text(.04, .012, "Gewinner aus damaligen Fehlern bis Wochenbeginn−2; keine nachträgliche Auswahl aus den Fehlern der Prognosewoche.",
                 color=INK_SECONDARY, fontsize=8)
        fig.tight_layout(rect=(.02, .11, 1, .94))
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("path", type=Path)
    parser.add_argument("--period", default="test")
    parser.add_argument("--schedule", action="store_true", help="plot the weekly 24-hour expert map")
    args = parser.parse_args()
    render = routing_schedule_figure if args.schedule else comparison_figure
    print(render(args.out, args.period, args.path))


if __name__ == "__main__":
    main()
