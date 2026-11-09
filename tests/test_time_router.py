"""Temporal poisoning, expert admission, DST, inference and one-shot protocol."""

import json

import numpy as np
import pandas as pd
import pytest

from energy_price.features.dataset import TARGET, quarter_hours
from energy_price.models.shared.calibration import ResidualCalibrator
from energy_price.models.shared.forecast_pool import validate_pool, wide_pool
from energy_price.models.shared.routing import (TimeRouterModel, block_ids, bootstrap_means,
                                               convex_weights, difference_interval)
from energy_price.evaluation.router_holdout import claim_once, check_final_evidence, readiness


def wide(start="2026-01-30", end="2026-04-10"):
    frame = quarter_hours(list(pd.date_range(start, end)))
    frame["hour"] = frame.timestamp_local.dt.hour
    frame[TARGET] = 5.
    frame["short"] = 1.
    frame["radiation"] = frame["radiation_change"] = frame["holiday_pv_share"] = 0.
    for name, values in (("anchor", (0., 3., 10.)), ("other", (4., 5., 6.))):
        for q, value in zip(("q10", "q50", "q90"), values):
            frame[f"{name}:{q}"] = value
    return frame


@pytest.mark.parametrize("strategy", ["default", "mean", "selection", "weights"])
def test_router_ignores_future_outcomes_and_direction_and_freezes_week(strategy):
    source = wide()
    day = pd.Timestamp("2026-04-04")
    poisoned = source.copy()
    poisoned.loc[poisoned.delivery_day > day - pd.Timedelta(days=2), TARGET] = -1e9
    poisoned["short"] = 0.
    def forecast(table):
        router = TimeRouterModel(["anchor", "other"], "anchor", [0, 8, 16, 24], strategy,
                                 fit_day=day, reps=200).fit(table)
        target = table[table.delivery_day.between(day, day + pd.Timedelta(days=6))]
        return router.predict(target)
    before, after = forecast(source), forecast(poisoned)
    pd.testing.assert_frame_equal(before, after)
    assert before.fit_day.nunique() == 1
    assert (before.error_last_day <= day - pd.Timedelta(days=2)).all()
    assert (before.q10 <= before.q50).all() and (before.q50 <= before.q90).all()
    if strategy == "selection":
        assert set(before.chosen_expert) == {"other"}


def test_new_expert_requires_no_selector_code_and_all_quantiles_choose_together():
    source = wide()
    for q in ("q10", "q50", "q90"):
        source[f"new_expert:{q}"] = 5.
    day = pd.Timestamp("2026-04-04")
    router = TimeRouterModel(["anchor", "other", "new_expert"], "anchor", fit_day=day, reps=200).fit(source)
    result = router.predict(source[source.delivery_day == day])
    assert set(result.chosen_expert) == {"new_expert"}
    assert (result[["q10", "q50", "q90"]] == 5.).all().all()
    assert json.loads(result.weights.iloc[0])["q50"]["new_expert"] == 1.


def test_no_evidence_keeps_default_and_bootstrap_retains_calendar_gap():
    source = wide()
    for q in ("q10", "q50", "q90"):
        source[f"other:{q}"] = source[f"anchor:{q}"]
    router = TimeRouterModel(["anchor", "other"], "anchor", fit_day="2026-04-04", reps=200).fit(source)
    assert set(router.choices_.values()) == {"anchor"}
    dates = pd.date_range("2026-01-01", periods=35)
    series = pd.Series(np.arange(35.), index=dates).drop(dates[12])
    assert np.array_equal(bootstrap_means(series, reps=200), bootstrap_means(series, reps=200))
    low, high = difference_interval(series, reps=200)
    assert low < series.mean() < high


@pytest.mark.parametrize("edges", [[0, 8, 8, 24], [1, 24], [0, 23], [0, 25, 24]])
def test_invalid_time_partition_rejected(edges):
    with pytest.raises(ValueError, match="partition"):
        block_ids(np.arange(24), edges)


def test_partition_and_crossing_quantiles_have_traceable_rearrangement():
    assert np.bincount(block_ids(np.arange(24), [0, 8, 16, 24])).tolist() == [8, 8, 8]
    source = wide()
    day = pd.Timestamp("2026-04-04")
    router = TimeRouterModel(["anchor", "other"], "anchor", fit_day=day).fit(source)
    # Different weights per quantile can cross even when each expert is ordered.
    target = source[source.delivery_day == day].copy()
    target[["anchor:q10", "anchor:q50", "anchor:q90"]] = [10., 11., 12.]
    target[["other:q10", "other:q50", "other:q90"]] = [0., 1., 2.]
    router.weights_[0] = np.array([[1, 0], [0, 1], [0, 1]])
    result = router.predict(target)
    assert result.quantiles_reordered.all()
    assert (result.preorder_q10 == 10).all()
    assert (result.q10 == 1).all() and (result.q90 == 10).all()


def test_simplex_pinball_solver_and_sparse_history_fallback():
    weights = convex_weights(np.tile([0., 10.], (30, 1)), np.full(30, 7.), .5)
    assert weights == pytest.approx([.3, .7])
    router = TimeRouterModel(["anchor", "other"], "anchor", strategy="weights", fit_day="2026-02-02").fit(wide())
    assert router.weights_[0] == pytest.approx(np.tile([1., 0.], (3, 1)))


def test_calibration_only_past_recorded_outputs_preserves_median():
    source = wide().rename(columns={f"anchor:{q}": q for q in ("q10", "q50", "q90")})
    day = pd.Timestamp("2026-04-04")
    before = ResidualCalibrator().fit(source, day)
    poisoned = source.copy()
    poisoned.loc[poisoned.delivery_day > day - pd.Timedelta(days=2), TARGET] = 1e9
    after = ResidualCalibrator().fit(poisoned, day)
    target = source[source.delivery_day == day]
    a, b = before.predict(target), after.predict(target)
    pd.testing.assert_frame_equal(a, b)
    assert a.q50.equals(target.q50)
    assert (a.calibration_last_day <= day - pd.Timedelta(days=2)).all()


def long_pool(frame):
    members = []
    for name in ("anchor", "other"):
        member = frame.copy()
        for q in ("q10", "q50", "q90"):
            member[q] = member[f"{name}:{q}"]
        member["fit_day"] = member.delivery_day
        member["train_last_day"] = member.delivery_day - pd.Timedelta(days=2)
        members.append(member.assign(model=name))
    return pd.concat(members, ignore_index=True)


def test_pool_detects_missing_rows_differing_labels_and_training_leakage():
    source = long_pool(wide(end="2026-02-01"))
    validate_pool(source, ["anchor", "other"])
    with pytest.raises(ValueError, match="missing quarter"):
        validate_pool(source.iloc[1:], ["anchor", "other"])
    changed = source.copy()
    changed.loc[0, TARGET] += 1
    with pytest.raises(ValueError, match="different targets"):
        validate_pool(changed, ["anchor", "other"])
    changed = source.copy()
    changed.loc[0, "train_last_day"] = changed.loc[0, "fit_day"] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="D-2"):
        validate_pool(changed, ["anchor", "other"])
    source.loc[0, "q10"] = np.nan
    assert wide_pool(source, ["anchor", "other"]).delivery_day.nunique() == 2


def test_dst_repeated_hour_is_preserved_by_utc_keys():
    source = long_pool(wide("2026-10-25", "2026-10-25"))
    result = wide_pool(source, ["anchor", "other"])
    assert len(result) == 100 and result.timestamp_utc.nunique() == 100


def test_holdout_readiness_is_not_finality_and_atomic_claim_is_once(tmp_path):
    source = wide("2026-09-01", "2026-10-31")
    ready = readiness(source)
    assert ready["complete"] and ready["n_expected_quarters"] == 5860
    with pytest.raises(ValueError, match="evidence"):
        check_final_evidence(None, ready, pd.Timestamp("2026-11-27"))
    evidence = tmp_path / "final.json"
    evidence.write_text(json.dumps({"final_prices_confirmed": True, "months": ["2026-09", "2026-10"],
                        "source_url": "https://example.org/final", "verified_by": "reviewer", "explanation": "final publication",
                        "verified_on": "2026-11-27", "data_fingerprint": ready["data_fingerprint"]}))
    check_final_evidence(evidence, ready, pd.Timestamp("2026-11-27"))
    with pytest.raises(ValueError, match="verification date"):
        check_final_evidence(evidence, ready, pd.Timestamp("2026-11-26"))
    ledger = tmp_path / "ledger.json"
    claim_once(ledger, {"status": "started"})
    with pytest.raises(RuntimeError, match="already started"):
        claim_once(ledger, {"status": "completed"})
    assert json.loads(ledger.read_text())["status"] == "started"


def test_complete_forward_pipeline_uses_historical_router_outputs_and_calibrates_once():
    from energy_price.evaluation.time_router import routed_history
    source = wide(end="2026-06-07")
    design = {"pool": {"anchor": {}, "other": {}}, "default": "anchor", "edges": [0, 8, 16, 24],
              "candidates": ["selection"], "min_history_days": 28, "bootstrap_reps": 200, "seed": 0,
              "calibration": {"calibration_days": 56, "grouping": "6h"}}
    before, _, _ = routed_history(source, design, "2026-06-07")
    poisoned = source.copy()
    poisoned.loc[poisoned.delivery_day >= "2026-05-31", TARGET] = 1e9
    poisoned["short"] = 0.
    after, _, _ = routed_history(poisoned, design, "2026-06-07")
    columns = ["q10", "q50", "q90", "weights", "fit_day", "error_last_day", "calibration_last_day"]
    def evaluated(frame):
        return frame[frame.delivery_day >= "2026-06-01"][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(evaluated(before), evaluated(after))
    raw = before[before.model == "selection"]
    past = raw[raw.delivery_day < "2026-06-01"]
    expected = ResidualCalibrator().fit(past, pd.Timestamp("2026-06-01")).predict(raw[raw.delivery_day >= "2026-06-01"])
    actual = before[(before.model == "calibrated_selection") & (before.delivery_day >= "2026-06-01")]
    np.testing.assert_allclose(expected[["q10", "q50", "q90"]], actual[["q10", "q50", "q90"]])


def test_design_analysis_refuses_test_rows_and_reports_current_pool(tmp_path, monkeypatch):
    from energy_price.evaluation import time_router
    monkeypatch.setattr(time_router.config, "POOL", {"anchor": {}, "other": {}})
    with pytest.raises(ValueError, match="April/May"):
        time_router.design_analysis(wide(end="2026-06-01"), tmp_path)
    result = time_router.design_analysis(wide("2026-04-01", "2026-05-31"), tmp_path)
    assert result["default"] == "other"
    assert result["block_cap"] == 24
    assert result["edges"] == list(range(25))
    assert result["layout"] == "hourly"
    assert result["power_is_diagnostic_only"] is True
    assert result["weekday_weekend_split"] is False
    assert set(result["pool"]) == {"anchor", "other"}
    hourly = pd.read_csv(tmp_path / "tune_hourly_best.csv")
    assert hourly.hour.tolist() == list(range(24))
    assert set(hourly.empirical_best) == {"other"}


@pytest.mark.parametrize("selection_mode", ["empirical", "significant"])
def test_hourly_schedule_selects_different_experts_within_same_day_and_week(selection_mode):
    source = wide()
    source[TARGET] = np.where(source.hour < 8, 3., 5.)
    day = pd.Timestamp("2026-04-04")
    router = TimeRouterModel(["anchor", "other"], "anchor", list(range(25)), fit_day=day,
                             selection_mode=selection_mode, reps=200).fit(source)
    target = source[source.delivery_day.between(day, day + pd.Timedelta(days=6))]
    result = router.predict(target)
    assert len(router.schedule_) == 24
    assert (result.loc[target.hour < 8, "chosen_expert"] == "anchor").all()
    assert (result.loc[target.hour >= 8, "chosen_expert"] == "other").all()
    for hour in range(24):
        assert result.loc[target.hour == hour, "chosen_expert"].nunique() == 1
    assert result.fit_day.nunique() == 1
    assert all(row["history_last_day"] <= day - pd.Timedelta(days=2) for row in router.schedule_)
    poisoned = source.copy()
    poisoned.loc[poisoned.delivery_day > day - pd.Timedelta(days=2), TARGET] = -1e9
    after = TimeRouterModel(["anchor", "other"], "anchor", list(range(25)), fit_day=day,
                            selection_mode=selection_mode, reps=200).fit(poisoned).predict(target)
    pd.testing.assert_frame_equal(result, after)


def test_hourly_winner_is_learned_only_from_same_local_hour():
    source = wide()
    day = pd.Timestamp("2026-04-04")
    before = TimeRouterModel(["anchor", "other"], "anchor", list(range(25)), fit_day=day,
                             selection_mode="empirical", reps=200).fit(source)
    changed = source.copy()
    changed.loc[changed.hour == 15, TARGET] = 3.
    after = TimeRouterModel(["anchor", "other"], "anchor", list(range(25)), fit_day=day,
                            selection_mode="empirical", reps=200).fit(changed)
    assert before.choices_[15] == "other" and after.choices_[15] == "anchor"
    assert all(before.choices_[h] == after.choices_[h] for h in range(24) if h != 15)


def test_empirical_and_significant_modes_expose_uncertainty_instead_of_hiding_winner():
    source = wide()
    weeks = (source.delivery_day - source.delivery_day.min()).dt.days // 7
    source[TARGET] = np.where(weeks % 2, 5., 3.)
    arguments = {"names": ["anchor", "other"], "default": "anchor", "fit_day": "2026-04-04", "reps": 1000}
    empirical = TimeRouterModel(**arguments, selection_mode="empirical").fit(source)
    significant = TimeRouterModel(**arguments, selection_mode="significant").fit(source)
    assert empirical.schedule_[0]["empirical_best"] == "other"
    assert empirical.choices_[0] == "other"
    assert significant.choices_[0] == "anchor"
    assert empirical.schedule_[0]["chosen_supported_vs_default"] is False
    assert empirical.diagnostics_[0]["ci95_low"] < 0 < empirical.diagnostics_[0]["ci95_high"]


def test_forward_hourly_tables_match_predicted_expert_for_all_four_quarters(tmp_path):
    from energy_price.evaluation.time_router import routed_history, write_routing_tables
    source = wide(end="2026-06-07")
    source[TARGET] = np.where(source.hour < 8, 3., 5.)
    design = {"pool": {"anchor": {}, "other": {}}, "default": "anchor", "edges": list(range(25)),
              "candidates": ["selection"], "selection_mode": "empirical", "min_history_days": 28,
              "bootstrap_reps": 100, "seed": 0, "calibration": {"calibration_days": 56, "grouping": "6h"}}
    forecasts, decisions, schedules = routed_history(source, design, "2026-06-07")
    write_routing_tables(decisions, schedules, "test", tmp_path)
    latest = pd.read_csv(tmp_path / "test_latest_hourly_schedule.csv", parse_dates=["fit_day", "history_last_day"])
    assert len(latest) == 24
    assert latest.start_hour.tolist() == list(range(24))
    assert (latest.history_last_day <= latest.fit_day - pd.Timedelta(days=2)).all()
    selected = forecasts[(forecasts.model == "selection") & (forecasts.delivery_day >= "2026-06-01")]
    mapped = selected.merge(latest[["start_hour", "chosen_expert"]], left_on="hour", right_on="start_hour")
    assert (mapped.chosen_expert_x == mapped.chosen_expert_y).all()
    assert set(mapped.chosen_expert_x) == {"anchor", "other"}
