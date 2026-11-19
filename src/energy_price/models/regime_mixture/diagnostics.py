"""How much is the direction worth? The regime model with four values of P(short) on the same price parts.

The regime model only beats a single quantile model if it knows the direction (short or long) well
enough. This script keeps the fits of the price quantiles per direction exactly as in the last
``select`` run (same days, same refit frequency, checked against ``select_forecasts.parquet``) and
only swaps p in the mixture F(x) = p * F_short(x) + (1 - p) * F_long(x):

- ``constant``: share of short quarter hours in the training rows (one value per fit),
- ``slot``: share of short per quarter hour of the day (hour x position) in the training rows,
- ``model``: the logistic regression of the regime model (the forecast of ``CHOSEN``),
- ``oracle``: the actual direction of the delivery quarter hour (0 or 1). Not available at D-1,
  11:00; it is an upper bound for the value of a perfect direction forecast.

The settings are those of ``CHOSEN`` with a single training window (a window mean has no single
fit to take apart). Nothing is selected here and only validation days are used. Every p is scored
with the pinball loss of the mixture, and the three forecast values of p also with Brier score, log
loss and AUC.

Usage (repository root, after ``make select MODEL=regime_mixture``)::

    python -m energy_price.models.regime_mixture.diagnostics

Writes ``reports/regime_mixture/validation_direction_forecasts.parquet``, ``_direction_scores.csv`` and
``_direction_reliability.csv`` and the figure ``docs/figures/regime_mixture/validation_direction_reliability.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from energy_price.evaluation.backtest import evaluate, folds, run_reference, training_rows
from energy_price.evaluation.design import DESIGN, WINDOW_MEAN, window_days
from energy_price.evaluation.experiment import check_days, development_table, label, load, windows_of
from energy_price.features.dataset import DIRECTION
from energy_price.models.regime_mixture.model import RegimeMixtureModel, mixture_quantiles
from energy_price.models.shared.base import COLUMNS, QUANTILES
from energy_price.paths import FIGURES_DIR, REPORTS_DIR

NAME = "regime_mixture"
VARIANTS = ("constant", "slot", "model", "oracle")
FORECAST_VARIANTS = ("constant", "slot", "model")
BINS = np.linspace(0, 1, 11)


def slot_share(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Share of short per quarter hour of the day (hour x position) in ``train``, for the rows of ``target``.

    A slot without training rows falls back to the overall share.
    """
    known = train[train[DIRECTION].notna()]
    share = known.groupby(["hour", "position"])[DIRECTION].mean()
    keys = pd.MultiIndex.from_frame(target[["hour", "position"]])
    return share.reindex(keys).fillna(known[DIRECTION].mean()).to_numpy()


def p_variants(train: pd.DataFrame, target: pd.DataFrame, p_model: np.ndarray) -> dict[str, np.ndarray]:
    """The four values of P(short) for the rows of ``target`` (oracle: NaN where the direction is unknown)."""
    known = train[DIRECTION].dropna()
    return {
        "constant": np.full(len(target), known.mean()),
        "slot": slot_share(train, target),
        "model": p_model,
        "oracle": target[DIRECTION].to_numpy(dtype=float),
    }


def _fold(table: pd.DataFrame, block: list[pd.Timestamp], settings: dict) -> pd.DataFrame:
    """One fit: forecasts of every p variant for the days of ``block``."""
    params = {k: v for k, v in settings.items() if k != "window"}
    window = window_days(settings["window"])
    train = training_rows(table, block[0], window)
    model = RegimeMixtureModel(**params).fit(train)
    target = table[table["delivery_day"].isin(block)]
    rows, p_model, q_short, q_long = model.components(target)
    # Rows without features (e.g. the missing weather of 2026-06-24) stay in with NaN forecasts, so
    # that ``evaluate`` drops those days for every variant and for the reference alike.
    complete = target[rows]
    columns = [COLUMNS[tau] for tau in QUANTILES]
    frames = []
    for variant, p in p_variants(train, complete, p_model).items():
        out = target.copy()
        out[columns + ["p_short"]] = np.nan
        ok = ~np.isnan(p)
        if ok.any():
            mixed = np.sort(mixture_quantiles(p[ok], q_short[ok], q_long[ok]), axis=1)
            out.loc[complete.index[ok], columns] = mixed
        out.loc[complete.index, "p_short"] = p
        frames.append(out.assign(model=f"p_{variant}"))
    return pd.concat(frames, ignore_index=True)


def run(table: pd.DataFrame, days: list[pd.Timestamp], settings: dict, jobs: int = -1,
        refit_every: int = DESIGN.refit_every) -> pd.DataFrame:
    """Forecasts of all p variants and of the reference for ``days``, with the folds of the backtest."""
    if settings["window"] == WINDOW_MEAN:
        raise ValueError("the direction diagnostics need a single training window, not the window mean")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=jobs)(delayed(_fold)(table, block, settings) for block in folds(days, refit_every))
    return pd.concat([run_reference(table, days), *parts], ignore_index=True)


def select_name(settings: dict, grid: list[dict], windows: tuple[str, ...]) -> str:
    """Name of ``settings`` (incl. ``window``) among the select candidates: a GRID setting with a window."""
    for params in grid:
        for window in windows:
            if {**params, "window": window} == settings:
                return label({**params, "window": window})
    raise ValueError(f"{settings} is not a candidate of select; rerun select")


def check_matches_select(forecasts: pd.DataFrame, saved: pd.DataFrame, name: str) -> None:
    """``p_model`` must reproduce the select forecast ``name``, otherwise code, data or settings have changed."""
    old = saved[saved["model"] == name].set_index("timestamp_utc")[[COLUMNS[tau] for tau in QUANTILES]]
    new = forecasts[forecasts["model"] == "p_model"].set_index("timestamp_utc")[old.columns].reindex(old.index)
    if old.empty or not np.allclose(old.to_numpy(), new.to_numpy(), atol=1e-9, equal_nan=True):
        raise ValueError(f"the mixture with the model's p differs from the select forecast {name}; rerun select")


def direction_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Brier score, log loss and AUC of the forecast values of p, on rows with a known direction."""
    out = {}
    for variant in FORECAST_VARIANTS:
        rows = forecasts[(forecasts["model"] == f"p_{variant}") & forecasts[DIRECTION].notna()
                         & forecasts["p_short"].notna()]
        y, p = rows[DIRECTION].to_numpy(), rows["p_short"].clip(1e-6, 1 - 1e-6).to_numpy()
        out[f"p_{variant}"] = {
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0.0, 1.0]),
            # A constant p does not rank the quarter hours of a fit: its AUC is 0.5 by definition.
            "auc": 0.5 if variant == "constant" else roc_auc_score(y, p),
            "mean_p": p.mean(),
            "share_short": y.mean(),
        }
    return pd.DataFrame(out).T


def reliability(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Observed share of short per bin of forecast p (width 0.1), for ``slot`` and ``model``."""
    frames = []
    for variant in ("slot", "model"):
        rows = forecasts[(forecasts["model"] == f"p_{variant}") & forecasts[DIRECTION].notna()
                         & forecasts["p_short"].notna()]
        bins = pd.cut(rows["p_short"], BINS, include_lowest=True)
        grouped = rows.groupby(bins, observed=True)
        frames.append(pd.DataFrame({
            "variant": variant,
            "p_mean": grouped["p_short"].mean(),
            "observed": grouped[DIRECTION].mean(),
            "n": grouped.size(),
        }).reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def reliability_figure(table: pd.DataFrame, path: Path) -> Path:
    """Forecast p against the observed share of short; the diagonal is a perfectly calibrated p."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from energy_price.evaluation.figures import AXIS, CATEGORICAL, INK_SECONDARY, RC, style_axes

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="perfectly calibrated")
        labels = {"slot": "share per quarter hour of the day", "model": "logistic regression (regime model)"}
        for colour, (variant, rows) in zip(CATEGORICAL[1:], table.groupby("variant", sort=False)):
            ax.plot(rows["p_mean"], rows["observed"], marker="o", color=colour, label=labels[variant])
        style_axes(ax, "P(short) forecast at D-1 11:00 against the observed direction", "observed share short")
        ax.set_xlabel("forecast P(short), bins of 0.1", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    module = load(NAME)
    out = REPORTS_DIR / NAME
    selection_path = out / "select_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} is missing: run make select MODEL={NAME} first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    settings = module.CHOSEN[NAME]
    name = select_name(settings, module.GRID, windows_of(module))
    forecasts = run(development_table(), check_days(tuple(selection["period"])), settings, args.jobs,
                    selection["refit_every"])
    check_matches_select(forecasts, pd.read_parquet(out / "select_forecasts.parquet"), name)
    scores, daily = evaluate(forecasts)
    scores = scores.join(direction_scores(forecasts))

    forecasts.to_parquet(out / "validation_direction_forecasts.parquet", index=False)
    scores.to_csv(out / "validation_direction_scores.csv")
    calibration = reliability(forecasts)
    calibration.to_csv(out / "validation_direction_reliability.csv", index=False)
    figure = reliability_figure(calibration, FIGURES_DIR / NAME / "validation_direction_reliability.png")

    pd.set_option("display.width", 160)
    columns = ["mean_daily_pinball", "pinball", "mae_q50", "coverage_80", "width_80", "skill_vs_reference",
               "share_days_better", "brier", "log_loss", "auc", "n_days"]
    print(scores[[c for c in columns if c in scores.columns]].round(4))
    print(calibration.round(3))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
