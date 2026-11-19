import numpy as np
import pandas as pd
import pytest

from energy_price.availability import swissgrid_cutoff
from energy_price.evaluation.backtest import (
    common_rows,
    compare,
    development_inputs,
    evaluate,
    folds,
    log_daily,
    missing_days,
    run_model,
    run_reference,
    training_rows,
    window_mean,
)
from energy_price.evaluation.design import DESIGN, WINDOW_DAYS
from energy_price.evaluation.experiment import load, run_variant
from energy_price.features.dataset import build_table, calendar_features, quarter_hours

TZ = "Europe/Zurich"
CLOCK_CHANGE_ORIGINS = ["2026-03-29", "2026-03-30", "2026-03-31", "2026-10-25", "2026-10-26", "2026-10-27"]


def synthetic_table(first: str = "2026-04-01", days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    table = quarter_hours(list(pd.date_range(first, periods=days, freq="D")))
    table = pd.concat([table, calendar_features(table["timestamp_local"])], axis=1)
    table["holiday_pv_share"] = 0.0
    table["radiation"] = rng.uniform(0, 800, len(table))
    table["radiation_change"] = rng.normal(0, 50, len(table))
    table["aep_ct_kwh"] = 10 - 2 * table["radiation"] / 100 + rng.normal(0, 4, len(table))
    return table


class MeanModel:
    """Smallest model that fulfils the interface: the training quantiles for every row."""

    def fit(self, table):
        self.q_ = table["aep_ct_kwh"].quantile([0.1, 0.5, 0.9]).to_numpy()
        return self

    def predict(self, table):
        return pd.DataFrame([self.q_] * len(table), index=table.index, columns=["q10", "q50", "q90"])


class RecordingModel(MeanModel):
    """Writes what it was trained on next to every forecast, so the check works across processes."""

    def fit(self, table):
        self.seen_ = {"train_last_local": table["timestamp_local"].max(),
                      "train_first_day": table["delivery_day"].min(),
                      "train_last_day": table["delivery_day"].max(),
                      "train_n_days": table["delivery_day"].nunique(), "train_rows": len(table)}
        return super().fit(table)

    def predict(self, table):
        return super().predict(table).assign(**self.seen_)


# --- Folds and training rows ----------------------------------------------------------------------


@pytest.mark.parametrize(("refit_every", "sizes"), [(1, [1] * 17), (7, [7, 7, 3])])
def test_folds_are_blocks_of_refit_every_days_in_order(refit_every, sizes):
    days = list(pd.date_range("2026-06-01", "2026-06-17", freq="D"))

    blocks = folds(days, refit_every)

    assert [len(b) for b in blocks] == sizes
    assert sum(blocks, []) == days


def test_folds_default_to_daily_refit():
    days = list(pd.date_range("2026-06-01", periods=3, freq="D"))

    assert folds(days) == [[day] for day in days]


@pytest.fixture(scope="module")
def full_year():
    """Every quarter hour of 2026 with a price, also after every origin: future rows exist."""
    return synthetic_table(first="2026-01-01", days=365)


@pytest.mark.parametrize("refit_every", [1, 7])
@pytest.mark.parametrize("window", list(WINDOW_DAYS))
@pytest.mark.parametrize("span", [("2026-03-23", "2026-04-05"), ("2026-10-19", "2026-11-01")])
def test_no_training_row_after_d_minus_2_for_any_origin(full_year, refit_every, window, span):
    days = list(pd.date_range(*span, freq="D"))
    assert {pd.Timestamp(d) for d in CLOCK_CHANGE_ORIGINS} & set(days)

    forecasts, _, runtime = run_model(full_year, days, RecordingModel, "rec", WINDOW_DAYS[window], refit_every,
                                      jobs=1)

    per_day = forecasts.groupby("delivery_day").first()
    assert list(per_day.index) == days
    for day, seen in per_day.iterrows():
        fit_day = max(f for f in runtime["fit_day"] if f <= day)
        assert seen["train_last_local"] < swissgrid_cutoff(day.date())
        assert seen["train_last_day"] == fit_day - pd.Timedelta(days=2) <= day - pd.Timedelta(days=2)
        if refit_every == 1:
            assert seen["train_last_day"] == day - pd.Timedelta(days=2)
        w = WINDOW_DAYS[window]
        if w is None:
            assert seen["train_first_day"] == DESIGN.first_train_day
        else:  # D-1-w to D-2, shorter only where 2026 has fewer days before
            first = max(fit_day - pd.Timedelta(days=w + 1), DESIGN.first_train_day)
            assert seen["train_first_day"] == first
            assert seen["train_n_days"] == (fit_day - pd.Timedelta(days=2) - first).days + 1
            assert seen["train_n_days"] == w or first == DESIGN.first_train_day
    assert set(runtime["fit_day"]) == {block[0] for block in folds(days, refit_every)}


@pytest.mark.parametrize(("fit_day", "window"), [  # every window holds 2026-03-29 (92) or 2026-10-25 (100)
    ("2026-03-31", 28), ("2026-03-31", 56), ("2026-04-02", 90),
    ("2026-10-27", 28), ("2026-10-27", 56), ("2026-10-28", 90),
])
def test_rolling_window_is_exactly_w_days_across_clock_changes(full_year, window, fit_day):
    fit_day = pd.Timestamp(fit_day)

    train = training_rows(full_year, fit_day, window)

    first, last = fit_day - pd.Timedelta(days=window + 1), fit_day - pd.Timedelta(days=2)
    assert train["delivery_day"].min() == first and train["delivery_day"].max() == last
    assert train["delivery_day"].nunique() == window
    assert len(train) == len(quarter_hours(list(pd.date_range(first, last, freq="D"))))


def test_run_model_reports_fit_time_and_training_size_per_fold():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-20", "2026-05-29", freq="D"))

    _, _, runtime = run_model(table, days, MeanModel, "mean", refit_every=7, jobs=1)

    assert list(runtime["fit_day"]) == [pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-27")]
    assert list(runtime["n_days"]) == [7, 3]
    assert (runtime["seconds"] >= 0).all() and (runtime["n_train"] > 0).all()
    assert set(runtime["model"]) == {"mean"}


# --- Raw data of the test period ------------------------------------------------------------------


def synthetic_inputs(last: str = "2026-10-31", price=lambda stamp: 1.0):
    from test_features_dataset import balance_of, inputs, prices_of, weather_for

    weather = pd.concat([weather_for(str(day.date())) for day in pd.date_range("2026-08-10", last, freq="D")],
                        ignore_index=True)
    return inputs(weather, prices_of("2026-06-01", last, price), balance=balance_of("2026-06-01", last))


def test_development_inputs_cut_all_raw_data_at_the_test_start():
    raw = synthetic_inputs()

    dev = development_inputs(raw)

    start = pd.Timestamp("2026-09-01").tz_localize(TZ)
    assert dev.prices["timestamp_local"].max() < start <= raw.prices["timestamp_local"].max()
    assert dev.balance["timestamp_local"].max() < start
    assert dev.weather["delivery_day"].max() == pd.Timestamp("2026-08-31")
    assert len(dev.prices) == len(quarter_hours(list(pd.date_range("2026-06-01", "2026-08-31", freq="D"))))


def test_development_features_are_identical_with_and_without_test_raw_data():
    days = list(pd.date_range("2026-08-20", "2026-08-31", freq="D"))
    with_test = synthetic_inputs("2026-10-31", price=lambda stamp: float(stamp.day))
    without = synthetic_inputs("2026-08-31", price=lambda stamp: float(stamp.day))

    full = build_table(with_test, days=days)
    cut = build_table(development_inputs(with_test), days=days)

    pd.testing.assert_frame_equal(cut, build_table(without, days=days))
    pd.testing.assert_frame_equal(cut, full)
    assert build_table(development_inputs(with_test))["delivery_day"].max() == pd.Timestamp("2026-08-31")


# --- Identical quarter hours for every model ------------------------------------------------------


def two_models(days, models=("A", "B")):
    rows = quarter_hours([pd.Timestamp(d) for d in days]).assign(aep_ct_kwh=1.0, q10=0.0, q50=1.0, q90=2.0)
    return pd.concat([rows.assign(model=m) for m in models], ignore_index=True)


def keys_per_model(rows):
    return {m: set(map(tuple, g[["delivery_day", "timestamp_utc"]].to_numpy())) for m, g in rows.groupby("model")}


def test_day_drops_when_a_model_has_no_rows_at_all_on_it():
    forecasts = two_models(["2026-06-01", "2026-06-02"])
    forecasts = forecasts[~((forecasts["model"] == "B") & (forecasts["delivery_day"] == "2026-06-02"))]

    rows = common_rows(forecasts)

    assert set(rows["delivery_day"]) == {pd.Timestamp("2026-06-01")}
    assert set(rows["model"]) == {"A", "B"}


def test_day_drops_when_a_model_misses_some_quarter_hours():
    forecasts = two_models(["2026-06-01", "2026-06-02"])
    noon = forecasts["timestamp_local"].dt.hour == 12
    missing = (forecasts["model"] == "B") & (forecasts["delivery_day"] == "2026-06-02") & noon

    rows = common_rows(forecasts[~missing])

    assert set(rows["delivery_day"]) == {pd.Timestamp("2026-06-01")}


def test_day_drops_when_a_model_has_a_quarter_hour_twice():
    forecasts = two_models(["2026-06-01", "2026-06-02"])
    twice = forecasts[(forecasts["model"] == "B") & (forecasts["delivery_day"] == "2026-06-02")].iloc[[5]]
    # one row too many, and one fewer elsewhere keeps the count at 96
    forecasts = pd.concat([forecasts.drop(twice.index + 1), twice], ignore_index=True)

    rows = common_rows(forecasts)

    assert set(rows["delivery_day"]) == {pd.Timestamp("2026-06-01")}


def test_day_drops_when_a_quarter_hour_belongs_to_another_day():
    forecasts = two_models(["2026-06-01", "2026-06-02"])
    first = forecasts.index[(forecasts["model"] == "B") & (forecasts["delivery_day"] == "2026-06-02")][0]
    forecasts.loc[first, "timestamp_utc"] -= pd.Timedelta(days=1)

    assert set(common_rows(forecasts)["delivery_day"]) == {pd.Timestamp("2026-06-01")}


def test_day_drops_for_all_models_with_a_missing_forecast_or_price():
    forecasts = two_models(["2026-06-01", "2026-06-02", "2026-06-03"])
    forecasts.loc[(forecasts["model"] == "B") & (forecasts["delivery_day"] == "2026-06-02"), "q50"] = np.nan
    forecasts.loc[forecasts["delivery_day"] == "2026-06-03", "aep_ct_kwh"] = np.nan

    rows = common_rows(forecasts)

    assert set(rows["delivery_day"]) == {pd.Timestamp("2026-06-01")}


@pytest.mark.parametrize(("day", "n"), [("2026-03-29", 92), ("2026-10-25", 100)])
def test_clock_change_days_count_with_all_their_quarter_hours(day, n):
    forecasts = two_models([day, "2026-06-01"])

    rows = common_rows(forecasts)
    short = common_rows(forecasts.drop(forecasts.index[(forecasts["model"] == "A")
                                                       & (forecasts["delivery_day"] == day)][-1]))

    assert rows.groupby(["model", "delivery_day"]).size().loc[("A", pd.Timestamp(day))] == n
    assert set(short["delivery_day"]) == {pd.Timestamp("2026-06-01")}


def test_all_models_keep_the_same_quarter_hours():
    forecasts = two_models(["2026-03-29", "2026-06-01", "2026-10-25"], models=("A", "B", "C"))
    forecasts = forecasts.drop(forecasts.index[(forecasts["model"] == "C")
                                               & (forecasts["delivery_day"] == "2026-06-01")][:3])

    keys = keys_per_model(common_rows(forecasts))

    assert keys["A"] == keys["B"] == keys["C"]
    assert {day for day, _ in keys["A"]} == {pd.Timestamp("2026-03-29"), pd.Timestamp("2026-10-25")}


# --- Mean over the windows ------------------------------------------------------------------------


def window_frames():
    base = quarter_hours([pd.Timestamp("2026-06-01")]).assign(aep_ct_kwh=5.0)
    rng = np.random.default_rng(1)
    frames = []
    for name in ("expanding", "rolling_56d", "rolling_90d"):
        q = np.sort(rng.normal(5, 3, (len(base), 3)), axis=1)
        frames.append(base.assign(q10=q[:, 0], q50=q[:, 1], q90=q[:, 2], model=name))
    return frames


def test_window_mean_averages_each_quantile_and_stays_sorted():
    frames = window_frames()
    shuffled = [frames[0], frames[1].sample(frac=1, random_state=0), frames[2]]

    mean = window_mean(shuffled, "setting|window_mean")

    expected = np.mean([f[["q10", "q50", "q90"]].to_numpy() for f in frames], axis=0)
    np.testing.assert_allclose(mean[["q10", "q50", "q90"]].to_numpy(), expected)
    assert ((mean["q10"] <= mean["q50"]) & (mean["q50"] <= mean["q90"])).all()
    assert set(mean["model"]) == {"setting|window_mean"}
    assert list(mean["timestamp_utc"]) == list(frames[0]["timestamp_utc"])


def test_window_mean_is_nan_where_a_window_has_no_forecast():
    frames = window_frames()
    frames[2].loc[[3, 7], "q50"] = np.nan

    mean = window_mean(frames)

    assert mean[["q10", "q50", "q90"]].isna().any(axis=1).tolist() == [i in (3, 7) for i in range(len(mean))]


def test_window_mean_rejects_windows_with_different_quarter_hours():
    frames = window_frames()
    frames[1] = frames[1].iloc[1:]

    with pytest.raises(ValueError, match="different quarter hours"):
        window_mean(frames)


# --- Scores and comparison ------------------------------------------------------------------------


def test_backtest_beats_the_reference_when_radiation_drives_the_price():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-20", "2026-05-29", freq="D"))

    settings = {"feature_set": "B2", "window": "expanding", "alpha": 0.0}
    linear, coefficients = run_variant(load("baseline_model"), table, days, "B2", settings, jobs=1, refit_every=7)
    scores, daily = evaluate(pd.concat([run_reference(table, days), linear], ignore_index=True))

    assert scores.loc["B2", "skill_vs_reference"] > 0.2
    assert scores.loc["B2", "mean_daily_pinball"] == pytest.approx(daily["B2"].mean())
    assert set(coefficients["fit_day"]) == {pd.Timestamp("2026-05-20"), pd.Timestamp("2026-05-27")}
    assert list(daily.index) == days


def test_any_model_with_fit_and_predict_runs_through_the_backtest():
    table = synthetic_table()
    days = list(pd.date_range("2026-05-20", "2026-05-29", freq="D"))

    forecasts, coefficients, runtime = run_model(table, days, MeanModel, "mean", jobs=1)

    assert coefficients is None
    assert set(forecasts["model"]) == {"mean"}
    assert forecasts[["q10", "q50", "q90"]].notna().all().all()
    assert sorted(forecasts["delivery_day"].unique()) == days
    assert list(runtime["fit_day"]) == days  # daily refit by default


def comparable(days=("2026-06-01", "2026-06-10")):
    table = synthetic_table(first="2026-04-01", days=80)
    days = list(pd.date_range(*days, freq="D"))
    model, _, _ = run_model(table, days, MeanModel, "mean", jobs=1)
    baseline, _, _ = run_model(table, days, MeanModel, "baseline_B2", window_days=28, jobs=1)
    return pd.concat([run_reference(table, days), model, baseline], ignore_index=True), days


def test_compare_scores_every_model_against_reference_and_b2_on_the_same_rows():
    forecasts, days = comparable()
    # one model misses a quarter hour on the first day: that day drops for every model
    gap = forecasts.index[(forecasts["model"] == "mean") & (forecasts["delivery_day"] == days[0])][10]

    table = compare(forecasts.drop(gap))

    assert {"period_start", "period_end", "compared_with", "low", "high", "skill", "mean_difference"} <= set(table)
    assert set(map(tuple, table[["model", "compared_with"]].to_numpy())) == {
        ("mean", "reference"), ("mean", "baseline_B2"), ("baseline_B2", "reference"), ("reference", "baseline_B2")}
    assert (table["period_start"] == "2026-06-02").all() and (table["period_end"] == "2026-06-10").all()
    assert (table["n_days"] == len(days) - 1).all()
    row = table.set_index(["model", "compared_with"]).loc[("baseline_B2", "reference")]
    assert row["skill"] == pytest.approx(1 - row["mean_daily_pinball"] / row["compared_mean_daily_pinball"])
    assert row["low"] <= row["mean_difference"] <= row["high"]


def test_compare_gives_no_interval_with_fewer_than_seven_days():
    forecasts, _ = comparable(("2026-06-01", "2026-06-05"))

    table = compare(forecasts)

    assert table["low"].isna().all() and table["high"].isna().all()


# --- W&B history ----------------------------------------------------------------------------------


class RecordingRun:
    """Stand-in for a W&B run that records define_metric and log calls."""

    def __init__(self):
        self.metrics, self.rows = {}, []

    def define_metric(self, name, **kwargs):
        self.metrics[name] = kwargs

    def log(self, row, step):
        self.rows.append((step, row))


def test_daily_pinball_is_logged_as_history_one_step_per_day():
    days = pd.date_range("2026-06-01", periods=3, freq="D", name="delivery_day")
    model = pd.Series([2.0, 5.0, 3.0], index=days)
    reference = pd.Series([2.5, 4.0, 3.0], index=days)
    run = RecordingRun()

    log_daily(run, "select", model, reference)

    assert run.metrics == {"select/daily_*": {"summary": "none"}}
    assert [step for step, _ in run.rows] == [0, 1, 2]
    assert [row["select/daily_pinball"] for _, row in run.rows] == [2.0, 5.0, 3.0]
    assert [row["select/daily_pinball_vs_reference"] for _, row in run.rows] == [-0.5, 1.0, 0.0]


def test_reference_logs_no_difference_to_itself():
    days = pd.date_range("2026-06-01", periods=2, freq="D", name="delivery_day")
    run = RecordingRun()

    log_daily(run, "select", pd.Series([1.0, 2.0], index=days))

    assert all(set(row) == {"select/daily_pinball"} for _, row in run.rows)


def test_skill_vs_reference_weighs_every_day_equally():
    # 2026-03-29 has 92, 2026-03-30 96 quarter hours: pooled and daily means differ.
    rows = quarter_hours([pd.Timestamp("2026-03-29"), pd.Timestamp("2026-03-30")])
    rows["aep_ct_kwh"] = 0.0
    reference = rows.assign(model="reference", q10=1.0, q50=1.0, q90=1.0)  # 0.5 per row on both days
    level = np.where(rows["delivery_day"] == pd.Timestamp("2026-03-29"), 0.0, 2.0)  # 0 on day 1, 1.0 on day 2
    model = rows.assign(model="B2", q10=level, q50=level, q90=level)

    scores, daily = evaluate(pd.concat([reference, model], ignore_index=True))

    assert daily["B2"].mean() == pytest.approx(daily["reference"].mean())
    assert scores.loc["B2", "skill_vs_reference"] == pytest.approx(0.0)


def test_missing_days_name_the_models_without_a_complete_forecast():
    rows = quarter_hours([pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02")])
    rows["aep_ct_kwh"] = 1.0
    full = rows.assign(q10=0.0, q50=1.0, q90=2.0)
    a = full.assign(model="a")
    b = full.assign(model="b")
    b.loc[b["delivery_day"] == pd.Timestamp("2026-06-02"), "q50"] = np.nan
    days = [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02"), pd.Timestamp("2026-06-03")]

    missing = missing_days(pd.concat([a, b], ignore_index=True), days)

    assert missing == {pd.Timestamp("2026-06-02"): ["b"], pd.Timestamp("2026-06-03"): ["a", "b"]}
