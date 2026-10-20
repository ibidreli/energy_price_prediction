"""Historical reference: the forecast without a model that every model has to beat.

For each local clock time (slot) the empirical quantiles of the prices of that slot over the last
``window_days`` days known at the issue time (D-1-window to D-2). It needs no model and no
features. The window of 56 days had the lowest error in the EDA (H11) and holds every weekday
equally often.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_price.availability import known_at_issue
from energy_price.features.dataset import TARGET
from energy_price.models.shared.base import COLUMNS, QUANTILES

WINDOW_DAYS = 56


def slot(stamps_local: pd.Series) -> pd.Series:
    """Local clock time as minutes after midnight. Both passes of the repeated hour share a slot."""
    return stamps_local.dt.hour * 60 + stamps_local.dt.minute


def reference_forecast(prices: pd.DataFrame, target_rows: pd.DataFrame, day: dt.date,
                       window_days: int = WINDOW_DAYS, quantiles: tuple[float, ...] = QUANTILES) -> pd.DataFrame:
    """Quantiles per slot of the prices known for ``day`` (D-1-window_days to D-2), for ``target_rows``.

    ``prices`` needs ``timestamp_local`` and the target; ``target_rows`` are the quarter hours of
    delivery day ``day``. Slots without history (e.g. 02:00 to 02:45 after a short day) stay NaN.
    """
    known = known_at_issue(prices, day)
    start = pd.Timestamp(day - dt.timedelta(days=window_days + 1)).tz_localize(known["timestamp_local"].dt.tz)
    window = known[known["timestamp_local"] >= start]
    table = window.groupby(slot(window["timestamp_local"]))[TARGET].quantile(list(quantiles)).unstack()
    table.columns = [COLUMNS[tau] for tau in table.columns]
    out = table.reindex(slot(target_rows["timestamp_local"]).to_numpy())
    out.index = target_rows.index
    return out
