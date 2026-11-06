"""Tests of the regime model short/long (issue #16)."""

import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation.backtest import evaluate, run_reference
from energy_price.evaluation.experiment import load, run_variant
from energy_price.features.dataset import calendar_features, quarter_hours
from energy_price.models.regime_mixture.model import GRID_QUANTILES, RegimeMixtureModel, mixture_quantiles
from energy_price.models.shared.base import QuantileModel

NAME = "regime_mixture"


def synthetic_table(first: str = "2026-04-01", days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Short in the second half of every hour (price around 18), long in the first half (around 5)."""
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range(first, periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    short = (table["position"] > 0) ^ (rng.uniform(size=len(table)) < 0.2)
    table["short"] = short.astype(float)
    table["aep_ct_kwh"] = np.where(short, 18.0, 5.0) + rng.normal(0, 2, len(table))
    return table


def uniform_quantiles(low: float, high: float, rows: int = 1) -> np.ndarray:
    return np.tile(low + (high - low) * np.asarray(GRID_QUANTILES), (rows, 1))


@pytest.mark.parametrize(("p", "regime"), [(1.0, "short"), (0.0, "long")])
def test_mixture_with_a_certain_direction_returns_the_quantiles_of_that_direction(p, regime):
    q_short, q_long = uniform_quantiles(10, 20), uniform_quantiles(-10, 0)
    expected = {"short": [11, 15, 19], "long": [-9, -5, -1]}[regime]

    out = mixture_quantiles(np.array([p]), q_short, q_long)

    assert out[0] == pytest.approx(expected)


def test_mixture_of_two_separate_directions_puts_the_median_between_them():
    out = mixture_quantiles(np.array([0.5]), uniform_quantiles(10, 20), uniform_quantiles(-10, 0))

    q10, q50, q90 = out[0]
    assert q10 == pytest.approx(-8)   # 20 % of the long distribution
    assert 0 <= q50 <= 10             # no mass between the two directions
    assert q90 == pytest.approx(18)   # 80 % of the short distribution


def test_model_fulfils_the_interface_of_all_models():
    table = synthetic_table()

    model = RegimeMixtureModel("B1").fit(table)
    out = model.predict(table)

    assert isinstance(model, QuantileModel)
    assert list(out.columns) == ["q10", "q50", "q90", "p_short"]
    assert out.index.equals(table.index)
    assert ((out["q10"] <= out["q50"]) & (out["q50"] <= out["q90"])).all()
    assert out["p_short"].between(0, 1).all()


def test_classifier_learns_the_direction_from_the_position_in_the_hour():
    table = synthetic_table()

    out = RegimeMixtureModel("B1").fit(table).predict(table)

    second_half = table["position"] > 0
    assert out.loc[second_half, "p_short"].mean() > 0.7
    assert out.loc[~second_half, "p_short"].mean() < 0.3


def test_forecast_does_not_use_the_direction_of_the_forecast_day():
    table = synthetic_table()
    model = RegimeMixtureModel("B1").fit(table)

    flipped = table.assign(short=1.0 - table["short"])

    pd.testing.assert_frame_equal(model.predict(table), model.predict(flipped))


def test_backtest_runs_end_to_end_against_the_reference():
    table = synthetic_table(days=40)
    days = list(pd.date_range("2026-05-01", "2026-05-03", freq="D"))

    module = load(NAME)
    settings = {"feature_set": "B1", "window": "expanding"}

    forecasts, _ = run_variant(module, table, days, "R1", settings, jobs=1)
    scores, _ = evaluate(pd.concat([run_reference(table, days), forecasts], ignore_index=True))

    assert set(scores.index) == {"reference", "R1"}
    assert scores.loc["R1", "n_days"] == len(days)


def test_slot_share_is_the_share_of_short_per_quarter_hour_of_the_day():
    from energy_price.models.regime_mixture.diagnostics import slot_share

    train = synthetic_table(days=20)
    target = synthetic_table(first="2026-05-01", days=1)

    share = slot_share(train, target)

    expected = train.groupby(["hour", "position"])["short"].mean()
    keys = list(zip(target["hour"], target["position"]))
    assert share == pytest.approx(expected.loc[keys].to_numpy())


def test_oracle_uses_the_actual_direction_and_the_forecasts_do_not():
    from energy_price.models.regime_mixture.diagnostics import p_variants

    train = synthetic_table(days=20)
    target = synthetic_table(first="2026-05-01", days=1, seed=1)
    p_model = np.full(len(target), 0.3)

    flipped = target.assign(short=1.0 - target["short"])
    p, p_flipped = p_variants(train, target, p_model), p_variants(train, flipped, p_model)

    assert p["oracle"] == pytest.approx(target["short"].to_numpy())
    for variant in ("constant", "slot", "model"):
        assert p[variant] == pytest.approx(p_flipped[variant])


def test_diagnostics_rank_the_oracle_first_on_a_clear_direction_pattern():
    from energy_price.models.regime_mixture.diagnostics import direction_scores, run

    table = synthetic_table(days=40)
    days = list(pd.date_range("2026-05-01", "2026-05-03", freq="D"))

    forecasts = run(table, days, {"feature_set": "B1", "window": "expanding"}, jobs=1)
    scores, _ = evaluate(forecasts)

    assert set(scores.index) == {"reference", "p_constant", "p_slot", "p_model", "p_oracle"}
    assert scores.loc["p_oracle", "pinball"] < scores.loc["p_model", "pinball"] < scores.loc["p_constant", "pinball"]
    assert direction_scores(forecasts).loc["p_model", "auc"] > 0.7


def test_diagnostics_score_the_reference_on_the_same_days_when_features_are_missing():
    from energy_price.models.regime_mixture.diagnostics import run

    table = synthetic_table(days=40)
    days = list(pd.date_range("2026-05-01", "2026-05-03", freq="D"))
    table.loc[table["delivery_day"] == days[1], "radiation"] = np.nan

    scores, _ = evaluate(run(table, days, {"feature_set": "B2", "window": "expanding"}, jobs=1))

    assert (scores["n_days"] == len(days) - 1).all()
