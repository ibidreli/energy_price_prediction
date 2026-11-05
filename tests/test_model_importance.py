import numpy as np
import pandas as pd
import pytest

from energy_price.models.shared.importance import grouped_permutation_importance
from energy_price.evaluation.diagnostics import backtest_importance, breakdown_scores, importance_summary
from test_evaluation_backtest import synthetic_table


class RadiationModel:
    def fit(self, table):
        return self

    def predict(self, table):
        centre = table["radiation"] / 100
        return pd.DataFrame({"q10": centre - 1, "q50": centre, "q90": centre + 1}, index=table.index)


def test_importance_detects_signal_preserves_inputs_and_is_reproducible():
    table = synthetic_table(days=2)
    table["aep_ct_kwh"] = table["radiation"] / 100
    untouched = table.copy(deep=True)
    groups = {"radiation": ["radiation", "radiation_change"], "calendar": ["hour", "position"]}
    result = grouped_permutation_importance(RadiationModel(), table, groups, n_repeats=3)
    pd.testing.assert_frame_equal(table, untouched)
    pd.testing.assert_frame_equal(result, grouped_permutation_importance(
        RadiationModel(), table, groups, n_repeats=3))
    summary = importance_summary(result).set_index("group")
    assert summary.loc["radiation", "importance_mean"] > 0.2
    assert summary.loc["calendar", "importance_mean"] == 0


def test_columns_of_one_source_are_permuted_together():
    table = synthetic_table(days=2)
    table["radiation_change"] = 2 * table["radiation"]

    class JointModel(RadiationModel):
        def predict(self, frame):
            np.testing.assert_array_equal(frame["radiation_change"], 2 * frame["radiation"])
            return super().predict(frame)

    grouped_permutation_importance(JointModel(), table, {"radiation": ["radiation", "radiation_change"]})


def test_importance_never_refits_on_validation_targets():
    table = synthetic_table(days=18)
    days = list(pd.date_range("2026-04-10", "2026-04-18"))
    trained_until = []

    class RecordingModel(RadiationModel):
        def fit(self, frame):
            trained_until.append(frame["delivery_day"].max())
            return self

    result = backtest_importance(table, days, RecordingModel, {"radiation": ["radiation"]},
                                n_repeats=1, jobs=1)
    assert trained_until == [pd.Timestamp("2026-04-08"), pd.Timestamp("2026-04-15")]
    assert list(result["fit_day"]) == [pd.Timestamp("2026-04-10"), pd.Timestamp("2026-04-17")]


@pytest.mark.parametrize("groups", [{"target": ["aep_ct_kwh"]}, {"missing": ["not_a_column"]}, {}])
def test_importance_rejects_invalid_feature_groups(groups):
    with pytest.raises(ValueError):
        grouped_permutation_importance(RadiationModel(), synthetic_table(days=1), groups)


def test_actual_direction_is_only_joined_for_diagnostics():
    table = synthetic_table(days=2)
    original = table.copy(deep=True)
    forecasts = pd.concat([table, RadiationModel().predict(table)], axis=1).assign(model="gbm")
    balance = table[["timestamp_utc"]].assign(tsi_mw=np.where(table["hour"] < 12, -10, 10))
    result = breakdown_scores(forecasts, balance)
    pd.testing.assert_frame_equal(table, original)
    directions = result[result["dimension"] == "direction"].set_index("value")
    assert set(directions.index) == {"short", "long"}
    assert directions["n_quarter_hours"].sum() == len(table)


def test_importance_excludes_a_whole_day_with_incomplete_forecasts():
    table = synthetic_table(days=2)
    table.loc[0, "radiation"] = np.nan
    result = grouped_permutation_importance(RadiationModel(), table, {"radiation": ["radiation"]}, n_repeats=1)
    assert result["n_quarter_hours"].iloc[0] == 96
