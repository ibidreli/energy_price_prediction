"""Settings of the baseline in models/baseline_model/model.py."""

from energy_price.evaluation.experiment import WINDOWS, load, split
from energy_price.features.design import FEATURE_SETS


def test_grid_covers_feature_set_window_and_alpha():
    module = load("baseline_model")

    assert len(module.GRID) == 2 * 2 * 3
    assert {s["feature_set"] for s in module.GRID} == {"B1", "B2"}


def test_chosen_settings_build_a_model():
    module = load("baseline_model")

    for variant, settings in module.CHOSEN.items():
        window_days, params = split(settings)
        model = module.MODEL(**params)
        assert settings["window"] in WINDOWS and window_days is None
        assert model.feature_set == variant and model.feature_set in FEATURE_SETS
