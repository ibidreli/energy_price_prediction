"""Temporal isolation, base walk-forward, asymmetric correction and selection for issue #19 (#49)."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from energy_price.evaluation import experiment
from energy_price.evaluation.backtest import folds, run_model
from energy_price.evaluation.design import DESIGN, WINDOW_DAYS, WINDOW_MEAN
from energy_price.features.dataset import TARGET, quarter_hours
from energy_price.models.calibrated_bands import model as calibrated_bands
from energy_price.models.calibrated_bands.model import BASES, GRID, select_candidates
from energy_price.models.shared import calibration

DAY = pd.Timedelta(days=1)


def table():
    frame = quarter_hours(list(pd.date_range("2026-01-01", "2026-04-12")))
    frame["hour"] = frame["timestamp_local"].dt.hour
    frame[TARGET] = np.where(frame["hour"] < 12, -4., 12.)
    return frame


class Recorder:
    """Tiny base model: constant quantiles; records the training and target days of every base fit."""

    calls: list[dict] = []

    def __init__(self, **settings):
        self.settings = settings

    def fit(self, train):
        self.train_days = pd.DatetimeIndex(sorted(train["delivery_day"].unique()))
        return self

    def predict(self, target):
        assert TARGET not in target
        Recorder.calls.append({"train_first": self.train_days.min(), "train_last": self.train_days.max(),
                               "n_train_days": len(self.train_days),
                               "target_first": target["delivery_day"].min(),
                               "target_last": target["delivery_day"].max()})
        return pd.DataFrame({"q10": 0., "q50": 5., "q90": 8.}, index=target.index)


class LengthRecorder(Recorder):
    """P50 is the number of training days, so a window mean is visible in the forecast."""

    def predict(self, target):
        super().predict(target)
        n = float(len(self.train_days))
        return pd.DataFrame({"q10": n - 1, "q50": n, "q90": n + 1}, index=target.index)


@pytest.fixture
def recording_base(monkeypatch):
    """Replace the frozen base by a recording model with a given window and refit frequency, without cache."""
    Recorder.calls = []

    def use(window="expanding", refit_every=1, windows=None, model=Recorder):
        windows = windows or (window,)
        base = calibration.Base(model=model, settings={}, window=window, windows=windows, refit_every=refit_every)
        monkeypatch.setattr(calibration, "_base", lambda name: base)
        monkeypatch.setattr(calibration, "_cached_forecast", calibration._fit_predict)
        return Recorder.calls

    use()
    return use


def assert_rows_known_and_inside_window(call, fit_day, window_days):
    assert call["target_first"] >= fit_day
    assert call["train_last"] <= fit_day - 2 * DAY
    oldest = DESIGN.first_train_day if window_days is None else max(
        DESIGN.first_train_day, fit_day - (window_days + 1) * DAY)
    assert call["train_first"] == oldest


@pytest.mark.parametrize("window", ["expanding", "rolling_56d", "rolling_90d"])
@pytest.mark.parametrize("refit_every", [1, 7])
def test_base_fits_follow_base_folds_window_and_d_minus_2(recording_base, window, refit_every):
    calls = recording_base(window, refit_every)
    source = table()
    train = source[source["delivery_day"] <= "2026-04-10"]
    model = calibration.CalibratedQuantileModel(calibration_days=28).fit(train)

    days = list(pd.date_range("2026-03-14", "2026-04-10"))
    expected = [block[0] for block in folds(days, refit_every)]
    assert [call["target_first"] for call in calls] == expected
    assert sorted(model.calibration_["fit_day"].unique()) == expected
    for call, block in zip(calls, folds(days, refit_every)):
        assert call["target_last"] == block[-1]
        assert_rows_known_and_inside_window(call, block[0], WINDOW_DAYS[window])
        assert call["target_last"] - 2 * DAY >= call["train_last"]

    calls.clear()
    target = source[source["delivery_day"] == "2026-04-12"]
    model.predict(target)
    assert len(calls) == 1
    fit_day = pd.Timestamp("2026-04-12")  # last training day + 2
    assert calls[0]["train_last"] == pd.Timestamp("2026-04-10")
    assert_rows_known_and_inside_window(calls[0], fit_day, WINDOW_DAYS[window])


def test_predict_uses_training_rows_of_last_day_plus_two(recording_base, monkeypatch):
    recording_base("rolling_56d")
    seen = []
    original = calibration.training_rows

    def spy(table, fit_day, window_days):
        seen.append((len(table), fit_day, window_days))
        return original(table, fit_day, window_days)

    monkeypatch.setattr(calibration, "training_rows", spy)
    source = table()
    train = source[source["delivery_day"] <= "2026-04-10"]
    model = calibration.CalibratedQuantileModel(calibrated=False).fit(train)
    model.predict(source[source["delivery_day"] == "2026-04-12"])
    assert seen == [(len(train), pd.Timestamp("2026-04-12"), 56)]


def test_window_mean_base_averages_the_per_window_forecasts(recording_base):
    calls = recording_base(WINDOW_MEAN, 1, windows=("expanding", "rolling_56d"), model=LengthRecorder)
    source = table()
    train = source[source["delivery_day"] <= "2026-04-10"]
    model = calibration.CalibratedQuantileModel(calibration_days=28).fit(train)
    assert len(calls) == 2 * 28  # every fold is forecast once per window
    calls.clear()
    target = source[source["delivery_day"] == "2026-04-12"]
    raw = calibration.CalibratedQuantileModel(calibrated=False).fit(train).predict(target)
    assert sorted(call["n_train_days"] for call in calls) == [56, 100]
    assert (raw["q50"] == (56 + 100) / 2).all()
    assert raw.index.equals(target.index)
    assert model.calibration_["q50"].notna().all()


def test_base_reads_window_and_refit_from_frozen_model(monkeypatch):
    real = calibration._base("baseline_model:B2")
    frozen = import_module("energy_price.models.baseline_model.model").CHOSEN["B2"]["window"]
    assert real.window == frozen and real.windows == (frozen,)
    assert real.refit_every == DESIGN.refit_every
    assert "window" not in real.settings

    module = SimpleNamespace(MODEL=Recorder, WINDOWS=("rolling_56d", "rolling_90d"), REFIT_EVERY=7,
                             CHOSEN={"mean": {"depth": 2, "window": WINDOW_MEAN},
                                     "short": {"depth": 2, "window": "rolling_56d"},
                                     "unnamed": {"depth": 2}})
    monkeypatch.setattr(calibration, "import_module", lambda name: module)
    mean = calibration._base("fake:mean")
    assert (mean.window, mean.windows, mean.refit_every, mean.settings) == (
        WINDOW_MEAN, ("rolling_56d", "rolling_90d"), 7, {"depth": 2})
    short = calibration._base("fake:short")
    assert (short.window, short.windows, short.refit_every) == ("rolling_56d", ("rolling_56d",), 7)
    with pytest.raises(ValueError, match="window"):
        calibration._base("fake:unnamed")


def test_cache_key_includes_window_and_refit_frequency(monkeypatch):
    keys = []
    monkeypatch.setattr(calibration, "_cached_forecast",
                        lambda model, settings, window, refit_every, train, target: keys.append((window, refit_every))
                        or pd.DataFrame({"q10": 0., "q50": 5., "q90": 8.}, index=target.index))
    base = calibration.Base(model=Recorder, settings={}, window=WINDOW_MEAN,
                            windows=("rolling_56d", "rolling_90d"), refit_every=7)
    monkeypatch.setattr(calibration, "_base", lambda name: base)
    source = table()
    train = source[source["delivery_day"] <= "2026-04-10"]
    calibration.CalibratedQuantileModel(calibrated=False).fit(train).predict(
        source[source["delivery_day"] == "2026-04-12"])
    assert keys == [("rolling_56d", 7), ("rolling_90d", 7)]


def test_outer_future_prices_and_inner_calibration_targets_never_enter_training(recording_base):
    source = table()
    day = pd.Timestamp("2026-04-04")
    factory = lambda: calibration.CalibratedQuantileModel(grouping="hour")
    before, _, _ = run_model(source, [day], factory, "calibrated", jobs=1)
    poisoned = source.copy()
    poisoned.loc[poisoned["delivery_day"] > day - pd.Timedelta(days=2), TARGET] = 1e9
    after, _, _ = run_model(poisoned, [day], factory, "calibrated", jobs=1)
    pd.testing.assert_frame_equal(before[["q10", "q50", "q90"]], after[["q10", "q50", "q90"]])
    known = source[source["delivery_day"] <= day - pd.Timedelta(days=2)]
    model = factory().fit(known)
    history = model.calibration_
    assert history["delivery_day"].min() == known["delivery_day"].max() - pd.Timedelta(days=27)
    assert (history["train_last_day"] <= history["fit_day"] - pd.Timedelta(days=2)).all()
    assert (history["fit_day"] <= history["delivery_day"]).all()


def test_separate_tails_preserve_median_order_missing_rows_and_index(recording_base, monkeypatch):
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

    class Missing(Recorder):
        def predict(self, target):
            result = super().predict(target)
            result.iloc[0] = np.nan
            return result

    recording_base(model=Missing)
    assert model.predict(target).iloc[0].isna().all()


def test_rank_validation_and_sparse_group_fallback(recording_base):
    assert calibration.upper_quantile(np.arange(20.)) == 18
    with pytest.raises(ValueError, match="insufficient"):
        calibration.upper_quantile([1.])
    source = table().groupby("delivery_day").head(1)
    model = calibration.CalibratedQuantileModel(grouping="hour").fit(source)
    target = source.tail(1).copy()
    target["delivery_day"] += pd.Timedelta(days=2)
    target["hour"] = 23
    assert model.predict(target).notna().all().all()
    with pytest.raises(ValueError, match="out-of-sample"):
        calibration.CalibratedQuantileModel().fit(source.head(5))


def candidate_settings():
    return {str(i): {**value, "window": "expanding"} for i, value in enumerate(GRID)}


def test_selection_prefers_success_per_base_and_falls_back_to_mean_daily_pinball():
    settings = candidate_settings()
    scores = pd.DataFrame(index=list(settings), data={"mean_daily_pinball": 1., "coverage_80": .7})
    for base in BASES.values():
        eligible = [key for key, value in settings.items() if value.get("base") == base
                    and value.get("calibrated", True)]
        scores.loc[eligible[1], "coverage_80"] = .8
        scores.loc[eligible[1], "mean_daily_pinball"] = 1.005
    chosen = select_candidates(scores, settings)
    assert set(chosen) == {"calibrated_B2", "raw_B2", "calibrated_gbm_quantile", "raw_gbm_quantile"}
    assert chosen["calibrated_B2"]["grouping"] == "6h"
    assert chosen["raw_B2"] == {"base": BASES["B2"], "calibrated": False, "window": "expanding"}
    assert all(value["window"] == "expanding" for value in chosen.values())
    scores["coverage_80"] = .7
    chosen = select_candidates(scores, settings)
    assert chosen["calibrated_B2"]["grouping"] == "global"


def test_model_file_matches_the_select_contract():
    assert calibrated_bands.WINDOWS == ("expanding",)
    assert experiment.grid_settings(calibrated_bands) == GRID
    experiment.check_chosen("calibrated_bands", calibrated_bands)
    assert all(settings["window"] == "expanding" for settings in calibrated_bands.CHOSEN.values())
    assert "holdout" not in calibrated_bands.RUN_INFO
    chosen = select_candidates(
        pd.DataFrame({"mean_daily_pinball": 1., "coverage_80": .8}, index=list(candidate_settings())),
        candidate_settings())
    assert set(chosen) == set(calibrated_bands.CHOSEN)


def test_select_persists_model_specific_selection(tmp_path, monkeypatch):
    import json
    from test_evaluation_backtest import MeanModel, synthetic_table

    expected = {"calibrated": {"window": "rolling_56d"}}
    module = SimpleNamespace(MODEL=MeanModel, GRID=[{}], WINDOWS=("expanding", "rolling_56d"),
                             select_candidates=lambda scores, settings: expected)
    monkeypatch.setattr(experiment, "load", lambda name: module)
    monkeypatch.setattr(experiment, "development_table", lambda design=None: synthetic_table(days=12))
    monkeypatch.setattr(experiment, "diagnostic_scores",
                        lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    tracked = []
    monkeypatch.setattr(experiment, "track_reference", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "track", lambda *args, **kwargs: tracked.append(args[2]))
    experiment.select("baseline_model", jobs=1, out_dir=tmp_path, do_track=True, days=("2026-04-10", "2026-04-10"))
    assert json.loads((tmp_path / "select_selection.json").read_text())["chosen"] == expected
    assert [config["window"] for config in tracked if config["selected"]] == ["rolling_56d"]
