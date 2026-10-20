"""Scores for probabilistic forecasts and a historical reference baseline.

Forecasts: P10, P50 and P90 of the price per quarter hour.
Main metric: mean pinball loss; additionally report median MAE,
P10–P90 coverage (target 80 %) and mean interval width.

See docs/evaluation.md and docs/metric.md.
Usage: python -m energy_price.metrics
"""

from __future__ import annotations

import argparse

import numpy as np
import numpy.typing as npt
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = QUANTILES[-1] - QUANTILES[0]

# D-2 is the last full day known at D-1 11:00 (see availability.py).
BASELINE_LAGS_DAYS = range(2, 9)

ArrayLike = npt.ArrayLike


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=float) for v in values]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"inputs differ in shape: {[a.shape for a in arrays]}")
    if any(np.isnan(a).any() for a in arrays):
        raise ValueError(
            "inputs contain NaN; drop or explain missing quarter hours before scoring"
        )
    return arrays


def pinball_loss(
    actual: ArrayLike, predicted: ArrayLike, quantile: float
) -> float:
    """Mean pinball loss in ct/kWh. At quantile 0.5 it is half the MAE."""
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must lie between 0 and 1, got {quantile}")
    y, q = _arrays(actual, predicted)
    diff = y - q
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def mean_pinball_loss(
    actual: ArrayLike, forecasts: dict[float, ArrayLike]
) -> float:
    """Pinball loss averaged over the quantiles in forecasts."""
    return float(
        np.mean(
            [
                pinball_loss(actual, predicted, q)
                for q, predicted in forecasts.items()
            ]
        )
    )


def pinball(y: ArrayLike, q: ArrayLike, tau: float) -> float:
    """Compatibility wrapper for callers using the original main API."""
    return pinball_loss(y, q, tau)


def mean_pinball(y: ArrayLike, quantiles: dict[float, ArrayLike]) -> float:
    """Compatibility wrapper for callers using the original main API."""
    return mean_pinball_loss(y, quantiles)


def coverage(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> float:
    """Share of actual values inside [lower, upper], both bounds included."""
    y, lo, hi = _arrays(actual, lower, upper)
    if (lo > hi).any():
        raise ValueError(
            f"lower bound above upper bound in {int((lo > hi).sum())} rows "
            "(crossing quantiles)"
        )
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean width of the interval [lower, upper] in ct/kWh."""
    lo, hi = _arrays(lower, upper)
    if (lo > hi).any():
        raise ValueError(
            f"lower bound above upper bound in {int((lo > hi).sum())} rows "
            "(crossing quantiles)"
        )
    return float(np.mean(hi - lo))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error in ct/kWh."""
    y, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(y - p)))


def score(
    actual: ArrayLike,
    p10: ArrayLike,
    p50: ArrayLike,
    p90: ArrayLike,
) -> dict[str, float]:
    """Main metric followed by the accompanying scores."""
    return {
        "mean_pinball": mean_pinball_loss(
            actual, {0.1: p10, 0.5: p50, 0.9: p90}
        ),
        "mae_p50": mae(actual, p50),
        "coverage_p10_p90": coverage(actual, p10, p90),
        "width_p10_p90": mean_width(p10, p90),
    }


def baseline_quantiles(
    prices: pd.DataFrame, value: str = "aep_ct_kwh"
) -> pd.DataFrame:
    """Quantiles of the same local quarter hour on D-2 .. D-8.

    prices needs timestamp_local and value. Rows whose window is not
    fully covered are NaN.
    """
    local = prices["timestamp_local"]
    keys = pd.DataFrame(
        {"date": local.dt.date, "tod": local.dt.strftime("%H:%M")},
        index=prices.index,
    )
    # The repeated 02:xx hour at the end of summer time maps to its
    # first occurrence.
    lookup = (
        prices[value]
        .set_axis(pd.MultiIndex.from_frame(keys))
        .groupby(level=[0, 1])
        .first()
    )
    lagged = pd.concat(
        [
            pd.Series(
                lookup.reindex(
                    list(
                        zip(
                            keys["date"] - pd.Timedelta(days=d),
                            keys["tod"],
                        )
                    )
                ).to_numpy(),
                index=prices.index,
            )
            for d in BASELINE_LAGS_DAYS
        ],
        axis=1,
    )
    complete = lagged.notna().all(axis=1)
    return pd.DataFrame(
        {
            tau: lagged.quantile(tau, axis=1).where(complete)
            for tau in QUANTILES
        }
    )


def main() -> None:
    from energy_price.swissgrid import load_prices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/ausgleichpreis")
    args = parser.parse_args()

    prices = load_prices(args.raw)
    prices = prices[prices["regime"] == "single_price"].reset_index(drop=True)
    base = baseline_quantiles(prices)
    rows = base.notna().all(axis=1)
    y = prices.loc[rows, "aep_ct_kwh"]
    base = base.loc[rows]
    stamps = prices.loc[rows, "timestamp_local"]
    negative = y < 0

    print(
        f"quarter hours        {rows.sum()} "
        f"({stamps.min():%Y-%m-%d} to {stamps.max():%Y-%m-%d})"
    )
    print(f"MAE                  {mae(y, base[0.5]):.2f} ct/kWh")
    print(
        f"MAE negative / rest  {mae(y[negative], base.loc[negative, 0.5]):.2f}"
        f" / {mae(y[~negative], base.loc[~negative, 0.5]):.2f} ct/kWh"
    )
    print(
        f"RMSE                 "
        f"{np.sqrt(((y - base[0.5]) ** 2).mean()):.2f} ct/kWh"
    )
    print(
        f"mean pinball         "
        f"{mean_pinball(y, {tau: base[tau] for tau in QUANTILES}):.2f} ct/kWh"
    )
    print(f"coverage 80 %        {coverage(y, base[0.1], base[0.9]):.1%}")
    print(f"interval width       {mean_width(base[0.1], base[0.9]):.2f} ct/kWh")


if __name__ == "__main__":
    main()