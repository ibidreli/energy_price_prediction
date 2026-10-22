from types import SimpleNamespace

import pandas as pd

from energy_price.evaluation import experiment
from energy_price.evaluation.backtest import track
from energy_price.evaluation import tracking
from test_evaluation_backtest import MeanModel, synthetic_table


def test_tuning_writes_selection_diagnostics_and_tracks_each_candidate(tmp_path, monkeypatch):
    table = synthetic_table(days=12)
    module = SimpleNamespace(MODEL=MeanModel, GRID=[{"window": "expanding"}, {"window": "rolling_56d"}],
                             ISSUE=17, RUN_INFO={"availability": "unconfirmed"})
    baseline = SimpleNamespace(MODEL=MeanModel, CHOSEN={"B2": {"window": "expanding"}})
    monkeypatch.setattr(experiment, "load", lambda name: baseline if name == "baseline_model" else module)
    monkeypatch.setattr(experiment, "period_days", lambda period: [pd.Timestamp("2026-04-10")])
    monkeypatch.setattr(experiment, "diagnostic_scores", lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    calls = []
    monkeypatch.setattr(experiment, "track", lambda *args, **kwargs: calls.append((args, kwargs)))
    references = []
    monkeypatch.setattr(experiment, "track_reference", lambda *args, **kwargs: references.append((args, kwargs)))

    scores = experiment.tune("toy", table, 1, tmp_path, do_track=True)

    assert set(scores.index) == {"reference", "expanding", "rolling_56d", "baseline_B2"}
    assert (tmp_path / "tune_selection.json").exists()
    assert (tmp_path / "tune_forecasts.parquet").exists()
    assert (tmp_path / "tune_breakdowns.csv").exists()
    assert len(calls) == 3 and len(references) == 1
    assert all(kwargs["issue"] == 17 for _, kwargs in calls)
    assert calls[0][1]["daily"] is not None
    assert sum(args[2].get("selected", False) for args, _ in calls) == 1


def test_track_uploads_importance_tables_and_artifacts_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(tracking, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    artifact = tmp_path / "scores.csv"
    artifact.write_text("pinball\n1.0\n")
    importance = pd.DataFrame({"group": ["radiation"], "importance_mean": [0.5], "importance_std": [0.1]})
    track("tune", "gbm", {"model": "gbm_quantile", "window": "expanding"}, pd.Series({"pinball": 1.0}),
          1, ("2026-04-01", "2026-05-31"), issue=17,
          tables={"feature_importance": importance}, files=[artifact])


def test_evaluation_tracks_the_selected_model_and_cached_baseline_with_urls(tmp_path, monkeypatch):
    table = synthetic_table(first="2026-05-25", days=12)
    day = pd.Timestamp("2026-06-02")
    settings = {"window": "expanding"}
    module = SimpleNamespace(MODEL=MeanModel, CHOSEN={"toy": settings}, ISSUE=17)
    baseline_module = SimpleNamespace(CHOSEN={"B2": settings})
    monkeypatch.setattr(experiment, "load", lambda name: baseline_module if name == "baseline_model" else module)
    monkeypatch.setattr(experiment, "period_days", lambda *args: [day])
    target = table[table["delivery_day"] == day]
    baseline = pd.concat([target, MeanModel().fit(table[table["delivery_day"] < day]).predict(target)],
                         axis=1).assign(model="baseline_B2")
    monkeypatch.setattr(experiment, "baseline_forecasts", lambda *args: baseline)
    monkeypatch.setattr(experiment, "diagnostic_scores", lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    calls = []

    def recorded_track(*args, **kwargs):
        calls.append((args, kwargs))
        return f"https://example.test/{args[1]}"

    monkeypatch.setattr(experiment, "track", recorded_track)
    monkeypatch.setattr(experiment, "track_reference", lambda *args, **kwargs: "https://example.test/reference")
    scores = experiment.evaluate_period("toy_model", table, "test", 1, tmp_path, do_track=True)

    assert set(scores.index) == {"reference", "toy", "baseline_B2"}
    assert [args[1] for args, _ in calls] == ["toy", "baseline_B2"]
    assert all(kwargs["issue"] == 17 for _, kwargs in calls)
    import json
    urls = json.loads((tmp_path / "test_wandb_urls.json").read_text())
    assert set(urls) == {"reference", "toy", "baseline_B2"}
    assert urls["baseline_B2"] == "https://example.test/baseline_B2"
