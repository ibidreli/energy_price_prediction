"""Settings of the baseline in models/baseline_model/model.py."""

from energy_price.evaluation.design import WINDOW_DAYS
from energy_price.evaluation.experiment import grid_settings, load, split, windows_of
from energy_price.features.design import FEATURE_SETS


def test_grid_covers_feature_set_and_alpha_and_windows_include_28_days():
    module = load("baseline_model")

    assert len(grid_settings(module)) == 2 * 3
    assert {s["feature_set"] for s in module.GRID} == {"B1", "B2"}
    assert windows_of(module) == ("expanding", "rolling_28d", "rolling_56d", "rolling_90d")


def test_chosen_settings_build_a_model():
    module = load("baseline_model")

    for variant, settings in module.CHOSEN.items():
        window_days, params = split(settings)
        model = module.MODEL(**params)
        assert settings["window"] in WINDOW_DAYS and window_days is None
        assert model.feature_set == variant and model.feature_set in FEATURE_SETS
