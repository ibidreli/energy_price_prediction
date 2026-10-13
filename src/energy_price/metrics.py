"""Evaluation metrics and reference baseline as defined in docs/metric.md.

Usage: ``python -m energy_price.metrics`` prints the baseline reference values of docs/metric.md.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
# Same quarter hour on D-2 .. D-8: D-2 is the last full day known at D-1 11:00 (see availability.py).
BASELINE_LAGS_DAYS = range(2, 9)


def mae(y: pd.Series, y_hat: pd.Series) -> float:
    return float((y - y_hat).abs().mean())


def pinball(y: pd.Series, q: pd.Series, tau: float) -> float:
    diff = y - q
    return float(np.maximum(tau * diff, (tau - 1) * diff).mean())


def mean_pinball(y: pd.Series, quantiles: dict[float, pd.Series]) -> float:
    """Pinball loss averaged over all quantile levels in ``quantiles``."""
    return float(np.mean([pinball(y, q, tau) for tau, q in quantiles.items()]))


def coverage(y: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    return float(((y >= lower) & (y <= upper)).mean())


def baseline_quantiles(prices: pd.DataFrame, value: str = "aep_ct_kwh") -> pd.DataFrame:
    """Quantiles of the price in the same local quarter hour on D-2 .. D-8, one row per input row.

    ``prices`` needs ``timestamp_local`` and ``value``. Rows whose window is not fully covered are NaN.
    """
    local = prices["timestamp_local"]
    keys = pd.DataFrame({"date": local.dt.date, "tod": local.dt.strftime("%H:%M")}, index=prices.index)
    # The repeated 02:xx hour at the end of summer time maps to its first occurrence.
    lookup = prices[value].set_axis(pd.MultiIndex.from_frame(keys)).groupby(level=[0, 1]).first()
    lagged = pd.concat(
        [
            pd.Series(
                lookup.reindex(list(zip(keys["date"] - pd.Timedelta(days=d), keys["tod"]))).to_numpy(),
                index=prices.index,
            )
            for d in BASELINE_LAGS_DAYS
        ],
        axis=1,
    )
    complete = lagged.notna().all(axis=1)
    return pd.DataFrame({tau: lagged.quantile(tau, axis=1).where(complete) for tau in QUANTILES})


def main() -> None:
    from energy_price.swissgrid import load_prices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/ausgleichpreis")
    args = parser.parse_args()

    prices = load_prices(args.raw)
    prices = prices[prices["regime"] == "single_price"].reset_index(drop=True)
    base = baseline_quantiles(prices)
    rows = base.notna().all(axis=1)
    y, base, stamps = prices.loc[rows, "aep_ct_kwh"], base[rows], prices.loc[rows, "timestamp_local"]
    negative = y < 0

    print(f"quarter hours        {rows.sum()} ({stamps.min():%Y-%m-%d} to {stamps.max():%Y-%m-%d})")
    print(f"MAE                  {mae(y, base[0.5]):.2f} ct/kWh")
    print(
        f"MAE negative / rest  {mae(y[negative], base.loc[negative, 0.5]):.2f}"
        f" / {mae(y[~negative], base.loc[~negative, 0.5]):.2f} ct/kWh"
    )
    print(f"RMSE                 {np.sqrt(((y - base[0.5]) ** 2).mean()):.2f} ct/kWh")
    print(f"mean pinball         {mean_pinball(y, {tau: base[tau] for tau in QUANTILES}):.2f} ct/kWh")
    print(f"coverage 80 %        {coverage(y, base[0.1], base[0.9]):.1%}")
    print(f"interval width       {(base[0.9] - base[0.1]).mean():.2f} ct/kWh")


if __name__ == "__main__":
    main()
