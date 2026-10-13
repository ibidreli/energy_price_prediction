"""Walk-forward backtest of the baseline: every delivery day is forecast as at D-1, 11:00.

For delivery day D a model sees only prices up to the end of D-2 (``known_at_issue``). Models are
refitted once a week: the fit for the first day of a week also serves the following six days, so
those days use even older data, never newer. Settings are chosen on the tuning period and then
applied unchanged to the test period. The holdout (September and October) stays locked until it
is explicitly released with ``--unlock-holdout``; it was not part of the EDA.

Usage:
``python -m energy_price.model.backtest tune``    grid over feature set, window and alpha on April and May
``python -m energy_price.model.backtest test``    reference, B1 and B2 on June to August, figures in docs/figures
``python -m energy_price.model.backtest holdout --unlock-holdout``

``--track`` logs every model as a W&B run in the group ``issue-6`` (docs/experiment_tracking.md).
Results are written to ``reports/baseline/`` (not versioned).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from energy_price.availability import known_at_issue
from energy_price.model.dataset import FIRST_DAY, ROOT, TARGET, build_table, load_inputs
from energy_price.model.models import LinearQuantileModel, reference_forecast
from energy_price.model.scores import COLUMNS, daily_pinball, pinball_per_row, score, skill_score

PERIODS = {
    "tune": ("2026-04-01", "2026-05-31"),
    "test": ("2026-06-01", "2026-08-31"),
    "holdout": ("2026-09-01", "2026-10-31"),
}
REFIT_DAYS = 7
REFERENCE_WINDOW_DAYS = 56
WINDOWS = {"expanding": None, "rolling_56d": 56}
ALPHAS = (0.0, 0.01, 0.1)
ISSUE = 6
OUT_DIR = ROOT / "reports" / "baseline"
FIG_DIR = ROOT / "docs" / "figures"


@dataclass(frozen=True)
class Config:
    feature_set: str
    window: str
    alpha: float

    @property
    def name(self) -> str:
        return self.feature_set


# Chosen on the tuning period (April and May, `make baseline-tune`): for both feature sets the
# expanding window without penalty had the lowest mean pinball loss, see docs/baseline.md.
CHOSEN = {
    "B1": Config("B1", "expanding", 0.0),
    "B2": Config("B2", "expanding", 0.0),
}


class HoldoutLockedError(RuntimeError):
    """The holdout period was requested without ``unlock_holdout``."""


def period_days(period: str, unlock_holdout: bool = False) -> list[pd.Timestamp]:
    if period == "holdout" and not unlock_holdout:
        raise HoldoutLockedError("the holdout is locked; release it once with --unlock-holdout")
    start, end = PERIODS[period]
    return list(pd.date_range(start, end, freq="D"))


def folds(days: list[pd.Timestamp], refit_days: int = REFIT_DAYS) -> list[list[pd.Timestamp]]:
    """Consecutive blocks of ``refit_days`` delivery days; the first day of a block is the fit day."""
    return [days[i:i + refit_days] for i in range(0, len(days), refit_days)]


def training_rows(table: pd.DataFrame, fit_day: pd.Timestamp, window_days: int | None) -> pd.DataFrame:
    """Rows with a price known at the issue time for ``fit_day``: up to D-2, optionally the last ``window_days``."""
    known = known_at_issue(table, fit_day.date())
    known = known[known[TARGET].notna() & (known["delivery_day"] >= FIRST_DAY)]
    if window_days is not None:
        known = known[known["delivery_day"] >= fit_day - pd.Timedelta(days=window_days + 1)]
    return known


def _fit_and_predict(table: pd.DataFrame, block: list[pd.Timestamp], config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = training_rows(table, block[0], WINDOWS[config.window])
    model = LinearQuantileModel(config.feature_set, config.alpha).fit(train)
    target = table[table["delivery_day"].isin(block)]
    forecast = pd.concat([target, model.predict(target)], axis=1)
    coef = model.coefficients().assign(fit_day=block[0], n_train=model.n_train_)
    return forecast, coef


def run_linear(table: pd.DataFrame, days: list[pd.Timestamp], config: Config, jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for ``days`` and the coefficients of every weekly fit. Folds run in parallel."""
    results = Parallel(n_jobs=jobs)(delayed(_fit_and_predict)(table, block, config) for block in folds(days))
    forecasts = pd.concat([r[0] for r in results], ignore_index=True).assign(model=config.name)
    coefficients = pd.concat([r[1] for r in results]).rename_axis("feature").reset_index()
    return forecasts, coefficients


def run_reference(table: pd.DataFrame, days: list[pd.Timestamp]) -> pd.DataFrame:
    prices = table[table[TARGET].notna()]
    frames = []
    for day in days:
        target = table[table["delivery_day"] == day]
        frames.append(pd.concat([target, reference_forecast(prices, target, day.date(), REFERENCE_WINDOW_DAYS)], axis=1))
    return pd.concat(frames, ignore_index=True).assign(model="reference")


def common_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Rows on delivery days where every model has a complete forecast and every price is known.

    All models are scored on the same quarter hours. A day with missing weather (2026-06-24) or a
    missing price drops out for all models alike.
    """
    q = list(COLUMNS.values())
    ok = forecasts[q + [TARGET]].notna().all(axis=1)
    per_day_model = ok.groupby([forecasts["delivery_day"], forecasts["model"]]).all().unstack()
    good_days = per_day_model.index[per_day_model.all(axis=1)]
    return forecasts[forecasts["delivery_day"].isin(good_days)]


def evaluate(forecasts: pd.DataFrame, reference: str = "reference") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per model and daily pinball loss per model, on the common rows."""
    rows = common_rows(forecasts)
    scores = pd.DataFrame({m: score(g) for m, g in rows.groupby("model")}).T
    scores["n_days"] = rows.groupby("model")["delivery_day"].nunique()
    if reference in scores.index:
        ref = scores.loc[reference, "pinball"]
        scores["skill_vs_reference"] = [skill_score(v, ref) for v in scores["pinball"]]
    daily = pd.DataFrame({m: daily_pinball(g) for m, g in rows.groupby("model")})
    if reference in daily.columns:
        better = daily.drop(columns=reference).lt(daily[reference], axis=0).mean()
        scores["share_days_better"] = better.reindex(scores.index)
    return scores, daily


def run_info(extra: dict | None = None) -> dict:
    """Code, data and package versions of a run, for ``reports/baseline/run_info.json``."""
    import joblib
    import sklearn

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        from energy_price.tracking import data_version
        version = data_version()
    except ImportError:  # tracking extra not installed
        version = None
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "data_version": version,
        "python": platform.python_version(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "joblib": joblib.__version__},
        "deterministic": "QuantileRegressor with the HiGHS solver has no random component; no seed needed",
        **(extra or {}),
    }


def track(split: str, name: str, config: dict, metrics: pd.Series, n_folds: int, period: tuple[str, str]) -> None:
    """Log one model as a W&B run (job type eval, group issue-6)."""
    from energy_price.tracking import start_run

    scheme = "rolling_window" if config.get("window", "").startswith("rolling") else "expanding_window"
    cv = {"scheme": scheme, "n_folds": n_folds, "test_start": period[0], "test_end": period[1],
          "refit_days": REFIT_DAYS}
    with start_run("eval", issue=ISSUE, name=name, config={**config, "cv": cv, "split": split}) as run:
        for key, value in metrics.items():
            if pd.notna(value):
                run.summary[f"{split}/{key}"] = float(value)


def tune(table: pd.DataFrame, jobs: int, out_dir: Path) -> pd.DataFrame:
    """Every combination of feature set, window and alpha on the tuning period, against the reference."""
    days = period_days("tune")
    configs = [Config(fs, w, a) for fs in ("B1", "B2") for w in WINDOWS for a in ALPHAS]
    frames = [run_reference(table, days)]
    for config in configs:
        forecast, _ = run_linear(table, days, config, jobs)
        frames.append(forecast.assign(model=f"{config.feature_set}|{config.window}|{config.alpha}"))
    scores, _ = evaluate(pd.concat(frames, ignore_index=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "tune_scores.csv")
    return scores


def evaluate_period(table: pd.DataFrame, period: str, jobs: int, out_dir: Path, unlock_holdout: bool = False,
                    do_track: bool = False) -> pd.DataFrame:
    """Reference, B1 and B2 with the chosen settings on ``period``; writes forecasts, scores and coefficients."""
    days = period_days(period, unlock_holdout)
    frames, coefficients = [run_reference(table, days)], []
    for config in CHOSEN.values():
        forecast, coef = run_linear(table, days, config, jobs)
        frames.append(forecast)
        coefficients.append(coef.assign(model=config.name))
    forecasts = pd.concat(frames, ignore_index=True)
    scores, daily = evaluate(forecasts)

    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / f"{period}_forecasts.parquet", index=False)
    scores.to_csv(out_dir / f"{period}_scores.csv")
    daily.to_csv(out_dir / f"{period}_daily_pinball.csv")
    pd.concat(coefficients, ignore_index=True).to_csv(out_dir / f"{period}_coefficients.csv", index=False)
    info = run_info({"period": period, "days": [str(d.date()) for d in (days[0], days[-1])],
                     "chosen": {k: asdict(v) for k, v in CHOSEN.items()},
                     "reference_window_days": REFERENCE_WINDOW_DAYS, "refit_days": REFIT_DAYS})
    (out_dir / f"{period}_run_info.json").write_text(json.dumps(info, indent=2))

    if do_track:
        n_folds = len(folds(days))
        track(period, "reference", {"model": "reference", "window": f"rolling_{REFERENCE_WINDOW_DAYS}d"},
              scores.loc["reference"], n_folds, PERIODS[period])
        for config in CHOSEN.values():
            track(period, config.name, {"model": "linear_quantile", **asdict(config)},
                  scores.loc[config.name], n_folds, PERIODS[period])
    return scores


# Chart colours: reference palette of the team's figures, categorical slots 1 to 3 (validated all-pairs
# for colour-blind readers) and an ordinal blue ramp for the three quantiles.
SERIES = {"reference": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"}
QUANTILE_SHADES = {"q10": "#86b6ef", "q50": "#2a78d6", "q90": "#104281"}
INK, INK_SECONDARY, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MONTHS = {6: "June", 7: "July", 8: "August", 9: "September", 10: "October"}
COEFFICIENTS_SHOWN = {
    "radiation_100": "Radiation, per 100 W/m²",
    "radiation_change_100_x_position": "Radiation change × position in hour",
    "weekend": "Weekend",
    "holiday_pv_share": "Holiday, full PV share",
}


def _axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def figures(period: str, out_dir: Path, fig_dir: Path) -> list[Path]:
    """Pinball loss per hour, coverage per month and B2 coefficients per quantile, as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forecasts = pd.read_parquet(out_dir / f"{period}_forecasts.parquet")
    coefficients = pd.read_csv(out_dir / f"{period}_coefficients.csv", parse_dates=["fit_day"])
    rows = common_rows(forecasts)
    rows = rows.assign(pinball=pinball_per_row(rows), inside=(rows[TARGET] >= rows["q10"]) & (rows[TARGET] <= rows["q90"]))
    models = [m for m in SERIES if m in set(rows["model"])]
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    style = {"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "figure.facecolor": SURFACE}

    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        by_hour = rows.groupby(["model", "hour"])["pinball"].mean().unstack(0)
        for m in models:
            ax.plot(by_hour.index, by_hour[m], color=SERIES[m], linewidth=2, marker="o", markersize=3.5, label=m)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Hour of the delivery day (local time)", color=INK_SECONDARY, fontsize=9)
        _axes(ax, f"Mean pinball loss per hour, {period} period (lower is better)", "ct/kWh")
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
        paths.append(fig_dir / f"baseline_{period}_pinball_by_hour.png")
        fig.tight_layout()
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.4))
        by_month = rows.groupby(["model", rows["delivery_day"].dt.month])["inside"].mean().unstack(0) * 100
        width = 0.26
        for i, m in enumerate(models):
            x = np.arange(len(by_month)) + (i - (len(models) - 1) / 2) * (width + 0.02)
            bars = ax.bar(x, by_month[m], width=width, color=SERIES[m], label=m)
            ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK_SECONDARY, padding=2,
                         bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
        ax.axhline(80, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=0.5)
        ax.annotate("target 80 %", (len(by_month) - 0.5, 80), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(range(len(by_month)), [MONTHS.get(k, str(k)) for k in by_month.index])
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        _axes(ax, f"Share of prices between P10 and P90, {period} period", "%")
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
        paths.append(fig_dir / f"baseline_{period}_coverage_by_month.png")
        fig.tight_layout()
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)

        last = coefficients[(coefficients["model"] == "B2") & (coefficients["fit_day"] == coefficients["fit_day"].max())]
        last = last.set_index("feature").loc[list(COEFFICIENTS_SHOWN)]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        height = 0.24
        for i, q in enumerate(QUANTILE_SHADES):
            y = np.arange(len(last)) + (i - 1) * (height + 0.02)
            bars = ax.barh(y, last[q], height=height, color=QUANTILE_SHADES[q], label=q.upper().replace("Q", "P"))
            ax.bar_label(bars, fmt="%.1f", fontsize=7, color=INK_SECONDARY, padding=3)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(range(len(last)), [COEFFICIENTS_SHOWN[f] for f in last.index])
        ax.invert_yaxis()
        fit_day = last["fit_day"].iloc[0].date()
        _axes(ax, f"B2 coefficients per quantile, fit for {fit_day} (ct/kWh per unit)", "")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        paths.append(fig_dir / f"baseline_{period}_coefficients.png")
        fig.tight_layout()
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["tune", "test", "holdout"])
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds, -1 uses all cores")
    parser.add_argument("--track", action="store_true", help="log the runs to W&B")
    parser.add_argument("--unlock-holdout", action="store_true", help="release the holdout period once")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    table = build_table(load_inputs())
    pd.set_option("display.width", 160)
    if args.step == "tune":
        scores = tune(table, args.jobs, args.out)
    else:
        scores = evaluate_period(table, args.step, args.jobs, args.out, args.unlock_holdout, args.track)
        for path in figures(args.step, args.out, FIG_DIR):
            print(f"figure {path.relative_to(ROOT)}")
    print(scores.round(3).to_string())
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
