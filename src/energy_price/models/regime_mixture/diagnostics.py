"""How much is the direction worth? The regime model with four values of P(short) on the same price parts.

The regime model only beats a single quantile model if it knows the direction (short or long) well
enough. This script keeps the weekly fits of the price quantiles per direction exactly as in the
backtest and only swaps p in the mixture F(x) = p * F_short(x) + (1 - p) * F_long(x):

- ``constant``: share of short quarter hours in the training rows (one value per fit),
- ``slot``: share of short per quarter hour of the day (hour x position) in the training rows,
- ``model``: the logistic regression of the regime model (the forecast of ``CHOSEN``),
- ``oracle``: the actual direction of the delivery quarter hour (0 or 1). Not available at D-1,
  11:00; it is an upper bound for the value of a perfect direction forecast.

Nothing is selected here, so the test period may be used. Every p is scored with the pinball loss
of the mixture, and the three forecast values of p also with Brier score, log loss and AUC.

Usage (repository root)::

    python -m energy_price.models.regime_mixture.diagnostics            # test, June to August
    python -m energy_price.models.regime_mixture.diagnostics --period tune

Writes ``reports/regime_mixture/<period>_direction_forecasts.parquet``, ``_direction_scores.csv`` and
``_direction_reliability.csv``
and the figure ``docs/figures/regime_mixture/<period>_direction_reliability.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from energy_price.evaluation.backtest import evaluate, folds, period_days, run_reference, training_rows
from energy_price.features.dataset import DIRECTION, build_table, load_inputs
from energy_price.models.regime_mixture.model import CHOSEN, RegimeMixtureModel, mixture_quantiles
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
    """One weekly fit: forecasts of every p variant for the days of ``block``."""
    params = {k: v for k, v in settings.items() if k != "window"}
    window = {"expanding": None, "rolling_56d": 56}[settings.get("window", "expanding")]
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


def run(table: pd.DataFrame, days: list[pd.Timestamp], settings: dict, jobs: int = -1) -> pd.DataFrame:
    """Forecasts of all p variants and of the reference for ``days``, with the same weekly folds as the backtest."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=jobs)(delayed(_fold)(table, block, settings) for block in folds(days))
    return pd.concat([run_reference(table, days), *parts], ignore_index=True)


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
            # A constant p does not rank quarter hours within a week: its AUC is 0.5 by definition.
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
    parser.add_argument("--period", choices=("tune", "test"), default="test")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    settings = CHOSEN[NAME]
    forecasts = run(build_table(load_inputs()), period_days(args.period), settings, args.jobs)
    scores, daily = evaluate(forecasts)
    scores = scores.join(direction_scores(forecasts))

    out = REPORTS_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out / f"{args.period}_direction_forecasts.parquet", index=False)
    scores.to_csv(out / f"{args.period}_direction_scores.csv")
    calibration = reliability(forecasts)
    calibration.to_csv(out / f"{args.period}_direction_reliability.csv", index=False)
    figure = reliability_figure(calibration, FIGURES_DIR / NAME / f"{args.period}_direction_reliability.png")

    pd.set_option("display.width", 160)
    columns = ["pinball", "mae_q50", "coverage_80", "width_80", "skill_vs_reference", "share_days_better",
               "brier", "log_loss", "auc", "n_days"]
    print(scores[[c for c in columns if c in scores.columns]].round(4))
    print(calibration.round(3))
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
