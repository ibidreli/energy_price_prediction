"""Temporal isolation, asymmetric correction and tuning selection for issue #19."""

import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation.backtest import run_model
from energy_price.features.dataset import TARGET, quarter_hours
from energy_price.models.shared import calibration
from energy_price.models.calibrated_bands.model import BASES, GRID, select_candidates


def table():
    frame = quarter_hours(list(pd.date_range("2026-01-01", "2026-04-12")))
    frame["hour"] = frame["timestamp_local"].dt.hour
    frame[TARGET] = np.where(frame["hour"] < 12, -4., 12.)
    return frame


def fake_forecast(base, train, target):
    assert TARGET not in target
    assert train["delivery_day"].max() <= target["delivery_day"].min() - pd.Timedelta(days=2)
    return pd.DataFrame({"q10": 0., "q50": 5., "q90": 8.}, index=target.index)


def test_outer_future_prices_and_inner_calibration_targets_never_enter_training(monkeypatch):
    monkeypatch.setattr(calibration, "_forecast", fake_forecast)
    source = table()
    day = pd.Timestamp("2026-04-04")
    factory = lambda: calibration.CalibratedQuantileModel(grouping="hour")
    before, _ = run_model(source, [day], factory, "calibrated", jobs=1)
    poisoned = source.copy()
    poisoned.loc[poisoned["delivery_day"] > day - pd.Timedelta(days=2), TARGET] = 1e9
    after, _ = run_model(poisoned, [day], factory, "calibrated", jobs=1)
    pd.testing.assert_frame_equal(before[["q10", "q50", "q90"]], after[["q10", "q50", "q90"]])
    known = source[source["delivery_day"] <= day - pd.Timedelta(days=2)]
    model = factory().fit(known)
    history = model.calibration_
    assert history["delivery_day"].min() == known["delivery_day"].max() - pd.Timedelta(days=27)
    assert (history["train_last_day"] <= history["fit_day"] - pd.Timedelta(days=2)).all()
    assert (history["fit_day"] <= history["delivery_day"]).all()


def test_separate_tails_preserve_median_order_missing_rows_and_index(monkeypatch):
    monkeypatch.setattr(calibration, "_forecast", fake_forecast)
    source = table()
    train = source[source["delivery_day"] < "2026-04-01"]
    model = calibration.CalibratedQuantileModel(grouping="hour").fit(train)
    target = source[source["delivery_day"] == "2026-04-03"].copy()
    prediction = model.predict(target)
    assert prediction.index.equals(target.index)
    assert (prediction.q50 == 5).all()
    assert (prediction.q10 <= prediction.q50).all()
    assert (prediction.q90 >= prediction.q50).all()
    assert (prediction.loc[target.hour < 12, "q10"] == -4).all()
    assert (prediction.loc[target.hour >= 12, "q90"] == 12).all()
    assert model.predict(target.iloc[:0]).empty
    def missing(base, train, target):
        result = fake_forecast(base, train, target)
        result.iloc[0] = np.nan
        return result
    monkeypatch.setattr(calibration, "_forecast", missing)
    assert model.predict(target).iloc[0].isna().all()


def test_rank_validation_and_sparse_group_fallback(monkeypatch):
    assert calibration.upper_quantile(np.arange(20.)) == 18
    with pytest.raises(ValueError, match="insufficient"):
        calibration.upper_quantile([1.])
    monkeypatch.setattr(calibration, "_forecast", fake_forecast)
    source = table().groupby("delivery_day").head(1)
    model = calibration.CalibratedQuantileModel(grouping="hour").fit(source)
    target = source.tail(1).copy()
    target["delivery_day"] += pd.Timedelta(days=2)
    target["hour"] = 23
    assert model.predict(target).notna().all().all()
    with pytest.raises(ValueError, match="out-of-sample"):
        calibration.CalibratedQuantileModel().fit(source.head(5))


def test_selection_prefers_success_per_base_and_falls_back_to_pinball():
    settings = {str(i): value for i, value in enumerate(GRID)}
    scores = pd.DataFrame(index=list(settings), data={"pinball": 1., "coverage_80": .7})
    for base in BASES.values():
        eligible = [key for key, value in settings.items() if value.get("base") == base
                    and value.get("calibrated", True)]
        scores.loc[eligible[1], "coverage_80"] = .8
        scores.loc[eligible[1], "pinball"] = 1.005
    chosen = select_candidates(scores, settings)
    assert len(chosen) == 4
    assert chosen["calibrated_B2"]["grouping"] == "6h"
    scores["coverage_80"] = .7
    chosen = select_candidates(scores, settings)
    assert chosen["calibrated_B2"]["grouping"] == "global"


def test_tuning_persists_model_specific_selection(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from energy_price.evaluation import experiment
    from test_evaluation_backtest import MeanModel, synthetic_table

    expected = {"calibrated": {"window": "rolling_56d"}}
    module = SimpleNamespace(MODEL=MeanModel, GRID=[{"window": "expanding"}, {"window": "rolling_56d"}],
                             select_candidates=lambda scores, settings: expected)
    monkeypatch.setattr(experiment, "load", lambda name: module)
    monkeypatch.setattr(experiment, "period_days", lambda period: [pd.Timestamp("2026-04-10")])
    monkeypatch.setattr(experiment, "diagnostic_scores",
                        lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    tracked = []
    monkeypatch.setattr(experiment, "track_reference", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "track", lambda *args, **kwargs: tracked.append(args[2]))
    experiment.tune("baseline_model", synthetic_table(days=12), 1, tmp_path, do_track=True)
    assert json.loads((tmp_path / "tune_selection.json").read_text())["chosen"] == expected
    assert [config["window"] for config in tracked if config["selected"]] == ["rolling_56d"]
