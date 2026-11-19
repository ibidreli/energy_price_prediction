import contextlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from energy_price.evaluation import experiment
from energy_price.evaluation import tracking
from energy_price.evaluation.backtest import track
from energy_price.evaluation.design import DESIGN, Design, TestLockedError
from energy_price.features.dataset import FIRST_DAY
from test_evaluation_backtest import MeanModel, synthetic_inputs, synthetic_table

SMOKE_DAYS = ("2026-04-27", "2026-05-03")  # two validation blocks
SHORT = Design(first_train_day=FIRST_DAY, validation=("2026-04-27", "2026-05-03"), test=("2026-09-01", "2026-09-05"))


class ShiftModel(MeanModel):
    """Training quantiles plus a constant: a setting that changes the score."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def predict(self, table):
        return super().predict(table) + self.shift


class SeenModel(MeanModel):
    """Writes the last delivery day it was trained on and asked for next to every forecast."""

    def fit(self, table):
        self.last_train_ = table["delivery_day"].max()
        return super().fit(table)

    def predict(self, table):
        return super().predict(table).assign(last_train=self.last_train_, last_target=table["delivery_day"].max())


def toy_module(**extra):
    return SimpleNamespace(**{"MODEL": ShiftModel, "GRID": [{"shift": 0.0}, {"shift": 3.0}], "CHOSEN": {},
                              "ISSUE": 17, "RUN_INFO": {"availability": "unconfirmed"}, **extra})


BASELINE = SimpleNamespace(MODEL=ShiftModel, CHOSEN={"B2": {"shift": 0.0, "window": "expanding"}})


@pytest.fixture
def toy(monkeypatch):
    """Synthetic table and models; diagnostics and W&B replaced by recorders."""
    modules = {"baseline_model": BASELINE, "toy": toy_module()}
    monkeypatch.setattr(experiment, "load", lambda name: modules[name])
    monkeypatch.setattr(experiment, "development_table", lambda design=DESIGN: synthetic_table(days=70))
    monkeypatch.setattr(experiment, "diagnostic_scores",
                        lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    calls = []
    monkeypatch.setattr(experiment, "track", lambda *args, **kwargs: calls.append((args, kwargs)) or "url")
    monkeypatch.setattr(experiment, "track_reference", lambda *args, **kwargs: "url")
    return SimpleNamespace(modules=modules, calls=calls)


def test_select_runs_every_setting_with_every_window_and_their_mean(tmp_path, toy):
    scores = experiment.select("toy", jobs=1, out_dir=tmp_path, do_track=True, days=SMOKE_DAYS)

    candidates = {f"{shift}|{window}" for shift in (0.0, 3.0)
                  for window in ("expanding", "rolling_56d", "rolling_90d", "window_mean")}
    assert set(scores["model"]) == candidates | {"reference", "baseline_B2"}
    assert sorted(scores["block"].unique()) == ["2026-04", "2026-05", "all"]
    ranked = scores[scores["model"].isin(candidates)]
    assert ranked.groupby("block")["rank"].min().eq(1).all()
    assert scores[~scores["model"].isin(candidates)]["rank"].isna().all()

    selection = json.loads((tmp_path / "select_selection.json").read_text())
    overall = ranked[ranked["block"] == "all"].set_index("model")["mean_daily_pinball"]
    assert selection["variant"] == overall.idxmin()
    assert selection["settings"] == {"shift": 0.0, "window": selection["variant"].split("|")[1]}
    assert selection["period"] == list(SMOKE_DAYS) and selection["refit_every"] == 1

    runtime = pd.read_csv(tmp_path / "select_runtime.csv")
    assert len(runtime) == 7 * (2 * 3 + 1)  # daily refit: 7 days x (6 candidates + B2); the means need no fit
    assert {"fit_day", "n_train", "seconds", "window", "setting", "model"} <= set(runtime.columns)
    info = json.loads((tmp_path / "select_run_info.json").read_text())
    assert info["design"]["validation"] == ["2026-04-01", "2026-08-31"]
    assert info["windows"] == ["expanding", "rolling_56d", "rolling_90d"] and info["runtime"]["n_fits"] == 49
    for name in ("forecasts.parquet", "breakdowns.csv", "daily_pinball.csv", "scores.csv"):
        assert (tmp_path / f"select_{name}").exists()

    assert len(toy.calls) == len(candidates) + 1  # every candidate and B2
    assert all(kwargs["issue"] == 17 and kwargs["refit_every"] == 1 for _, kwargs in toy.calls)
    assert sum(args[2].get("selected", False) for args, _ in toy.calls) == 1
    assert all(args[5] == SMOKE_DAYS for args, _ in toy.calls)


def test_figures_show_reference_selected_candidate_and_b2(tmp_path, toy):
    experiment.select("toy", jobs=1, out_dir=tmp_path, days=SMOKE_DAYS)

    paths = experiment.figures("toy", tmp_path, tmp_path / "figures")

    assert [p.name for p in paths] == ["select_pinball_by_hour.png", "select_coverage_by_month.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_select_rejects_a_grid_with_a_window(tmp_path, toy):
    toy.modules["toy"] = toy_module()
    toy.modules["toy"].GRID = [{"shift": 0.0, "window": "expanding"}]

    with pytest.raises(ValueError, match="must not contain 'window'"):
        experiment.select("toy", jobs=1, out_dir=tmp_path, days=SMOKE_DAYS)


@pytest.mark.parametrize("days", [("2026-03-30", "2026-04-02"), ("2026-08-30", "2026-09-02")])
def test_select_days_must_lie_inside_the_validation_period(tmp_path, toy, days):
    with pytest.raises(ValueError, match="inside the validation period"):
        experiment.select("toy", jobs=1, out_dir=tmp_path, days=days)


@pytest.mark.parametrize(("model_refit", "cli_refit", "fit_days"), [
    (7, None, ["2026-04-27", "2026-05-04"]),
    (7, 1, [str(d.date()) for d in pd.date_range("2026-04-27", "2026-05-05")]),
    (None, 3, ["2026-04-27", "2026-04-30", "2026-05-03"]),
])
def test_refit_frequency_comes_from_the_cli_else_the_model(tmp_path, toy, model_refit, cli_refit, fit_days):
    toy.modules["toy"] = toy_module(WINDOWS=("expanding",), **({"REFIT_EVERY": model_refit} if model_refit else {}))

    scores = experiment.select("toy", jobs=1, out_dir=tmp_path, refit_every=cli_refit,
                               days=("2026-04-27", "2026-05-05"))

    runtime = pd.read_csv(tmp_path / "select_runtime.csv", parse_dates=["fit_day"])
    own = runtime[runtime["setting"] != "baseline_B2"]
    assert sorted(own["fit_day"].dt.strftime("%Y-%m-%d").unique()) == fit_days
    assert not scores["model"].str.endswith("window_mean").any()  # one window: no mean
    assert len(runtime[runtime["setting"] == "baseline_B2"]) == 9  # B2 keeps its own (daily) refit


def test_the_selection_hook_names_the_chosen_settings(tmp_path, toy):
    expected = {"calibrated": {"shift": 3.0, "window": "rolling_56d"}}
    toy.modules["toy"] = toy_module(select_candidates=lambda scores, settings: expected)

    experiment.select("toy", jobs=1, out_dir=tmp_path, do_track=True, days=SMOKE_DAYS)

    assert json.loads((tmp_path / "select_selection.json").read_text())["chosen"] == expected
    selected = [args[2] for args, _ in toy.calls if args[2].get("selected")]
    assert [(c["shift"], c["window"]) for c in selected] == [(3.0, "rolling_56d")]


# --- The test period stays unseen during select ---------------------------------------------------


def select_on_synthetic_raw_data(tmp_path, monkeypatch, price, days=("2026-08-24", "2026-08-31")):
    """select with the real development_table on synthetic raw data that reaches to 2026-10-31."""
    modules = {"baseline_model": BASELINE, "seen": SimpleNamespace(MODEL=SeenModel, GRID=[{}], CHOSEN={})}
    monkeypatch.setattr(experiment, "load", lambda name: modules[name])
    monkeypatch.setattr(experiment, "load_inputs", lambda: synthetic_inputs("2026-10-31", price))
    monkeypatch.setattr(experiment, "diagnostic_scores",
                        lambda frame: pd.DataFrame({"model": frame["model"].unique()}))
    scores = experiment.select("seen", jobs=1, out_dir=tmp_path, days=days)
    return scores, pd.read_parquet(tmp_path / "select_forecasts.parquet")


def test_select_never_reads_the_test_period(tmp_path, monkeypatch):
    _, forecasts = select_on_synthetic_raw_data(tmp_path, monkeypatch, price=lambda stamp: float(stamp.day))

    test_start = pd.Timestamp("2026-09-01")
    assert forecasts["delivery_day"].max() == pd.Timestamp("2026-08-31")
    own = forecasts[forecasts["model"].isin(["expanding", "rolling_56d", "rolling_90d"])]
    assert len(own) and (pd.to_datetime(own["last_train"]) < test_start).all()
    assert (pd.to_datetime(own["last_target"]) < test_start).all()


def test_the_development_table_cannot_reach_the_test_period(monkeypatch):
    monkeypatch.setattr(experiment, "load_inputs", lambda: synthetic_inputs("2026-10-31"))

    table = experiment.development_table()

    assert table["delivery_day"].max() == pd.Timestamp("2026-08-31")


def test_extreme_test_prices_do_not_change_the_select_scores(tmp_path, monkeypatch):
    def price(stamp):
        return float(stamp.hour + stamp.day % 5)

    normal, _ = select_on_synthetic_raw_data(tmp_path / "a", monkeypatch, price)
    extreme, _ = select_on_synthetic_raw_data(
        tmp_path / "b", monkeypatch, price=lambda stamp: 1e6 if stamp.month >= 9 else price(stamp))

    pd.testing.assert_frame_equal(normal, extreme)


# --- compare --------------------------------------------------------------------------------------


def test_compare_reruns_chosen_of_every_model_with_reference_and_b2(tmp_path, toy):
    toy.modules["toy"] = toy_module(CHOSEN={"toy": {"shift": 0.0, "window": "window_mean"},
                                            "toy_wide": {"shift": 3.0, "window": "rolling_56d"}})
    toy.modules["baseline_model"] = SimpleNamespace(MODEL=ShiftModel, CHOSEN={
        "B1": {"shift": 1.0, "window": "expanding"}, "B2": {"shift": 0.0, "window": "expanding"}})

    table = experiment.run_compare(jobs=1, out_dir=tmp_path, do_track=True, names=["baseline_model", "toy"],
                                   design=SHORT)

    assert set(table["model"]) == {"reference", "baseline_B2", "baseline_model/B1", "toy/toy", "toy/toy_wide"}
    assert set(table["compared_with"]) == {"reference", "baseline_B2"}
    assert (table["period_start"] == "2026-04-27").all() and (table["period_end"] == "2026-05-03").all()
    forecasts = pd.read_parquet(tmp_path / "validation_forecasts.parquet")
    assert (forecasts["delivery_day"] <= pd.Timestamp("2026-05-03")).all()
    for name in ("scores.csv", "daily_pinball.csv", "run_info.json", "runtime.csv", "model_scores.csv",
                 "wandb_urls.json"):
        assert (tmp_path / f"validation_{name}").exists()
    assert {args[1] for args, _ in toy.calls} == {"baseline_B2", "baseline_model/B1", "toy/toy", "toy/toy_wide"}


def test_compare_refuses_a_model_without_chosen_settings(tmp_path, toy):
    with pytest.raises(ValueError, match="CHOSEN of toy is empty"):
        experiment.run_compare(jobs=1, out_dir=tmp_path, names=["toy"], design=SHORT)


def test_model_names_cover_every_model_folder_but_the_template():
    names = experiment.model_names()

    assert "baseline_model" in names and "gbm_quantile" in names
    assert "_template" not in names and "shared" not in names


# --- final-test (scaffold, never run on real data in #36) -----------------------------------------


@pytest.fixture
def final(monkeypatch, toy):
    toy.modules["toy"] = toy_module(CHOSEN={"toy": {"shift": 0.0, "window": "expanding"}})
    monkeypatch.setattr(experiment, "git_status", lambda: "")
    monkeypatch.setattr(experiment, "load_inputs", lambda: None)
    monkeypatch.setattr(experiment, "build_table", lambda inputs: synthetic_table(first="2026-06-01", days=100))
    return toy


def test_final_test_is_locked_without_unlock(tmp_path, final):
    with pytest.raises(TestLockedError):
        experiment.final_test(protocol=tmp_path / "protocol.json", out_dir=tmp_path, names=["toy"], design=SHORT)
    assert not (tmp_path / "protocol.json").exists()


def test_final_test_refuses_a_dirty_git_tree(tmp_path, final, monkeypatch):
    monkeypatch.setattr(experiment, "git_status", lambda: " M src/energy_price/models/toy/model.py")

    with pytest.raises(RuntimeError, match="clean git tree"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=tmp_path / "protocol.json", names=["toy"],
                              design=SHORT)
    assert not (tmp_path / "protocol.json").exists()


def test_final_test_writes_the_protocol_before_scoring_and_runs_only_once(tmp_path, final, monkeypatch):
    protocol = tmp_path / "docs" / "test_protocol.json"
    scored = []
    original = experiment._compare_step

    def recorded(step, *args, **kwargs):
        scored.append((step, protocol.exists()))
        return original(step, *args, **kwargs)

    monkeypatch.setattr(experiment, "_compare_step", recorded)

    table = experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)

    assert scored == [("test", True)]
    record = json.loads(protocol.read_text())
    assert record["chosen"]["toy"] == {"toy": {"shift": 0.0, "window": "expanding"}}
    assert record["period"] == ["2026-09-01", "2026-09-05"] and record["git_commit"]
    assert record["design"]["test"] == ["2026-09-01", "2026-09-05"]
    assert (table["period_start"] == "2026-09-01").all() and (tmp_path / "test_scores.csv").exists()
    with pytest.raises(RuntimeError, match="only once"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)


def test_the_cli_keeps_steps_and_options_apart():
    with pytest.raises(SystemExit):
        experiment.main(["baseline_model"])  # a model needs the step select
    with pytest.raises(SystemExit):
        experiment.main(["compare", "--days", "2026-04-01", "2026-04-02"])
    with pytest.raises(SystemExit):
        experiment.main(["baseline_model", "tune"])


# --- W&B ------------------------------------------------------------------------------------------


def test_track_config_holds_design_window_refit_and_period(monkeypatch):
    configs = []

    @contextlib.contextmanager
    def start_run(job_type, issue=None, name=None, config=None):
        configs.append(config)
        yield SimpleNamespace(summary={}, url="url", define_metric=lambda *a, **k: None, log=lambda *a, **k: None)

    monkeypatch.setattr(tracking, "start_run", start_run)

    track("select", "toy", {"model": "toy", "window": "rolling_56d"}, pd.Series({"pinball": 1.0}), 153,
          ("2026-04-01", "2026-08-31"), refit_every=7)

    config = configs[0]
    assert config["window"] == "rolling_56d" and config["refit_every"] == 7
    assert config["period"] == ["2026-04-01", "2026-08-31"] and config["design"] == DESIGN.as_config()
    assert config["cv"]["scheme"] == "rolling_window" and config["split"] == "select"


def test_track_uploads_importance_tables_and_artifacts_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(tracking, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    artifact = tmp_path / "scores.csv"
    artifact.write_text("pinball\n1.0\n")
    importance = pd.DataFrame({"group": ["radiation"], "importance_mean": [0.5], "importance_std": [0.1]})
    track("select", "gbm", {"model": "gbm_quantile", "window": "expanding"}, pd.Series({"pinball": 1.0}),
          1, ("2026-04-01", "2026-08-31"), issue=17,
          tables={"feature_importance": importance}, files=[artifact])


# --- review findings: preflight of final-test, dropped days in select ----------------------------


def test_final_test_checks_chosen_settings_before_it_writes_the_protocol(tmp_path, final):
    final.modules["toy"] = toy_module(CHOSEN={})
    protocol = tmp_path / "protocol.json"

    with pytest.raises(ValueError, match="CHOSEN of toy is empty"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)
    assert not protocol.exists()


@pytest.mark.parametrize("settings", [{"shift": 0.0}, {"shift": 0.0, "window": "rolling_7d"}])
def test_final_test_rejects_a_chosen_window_before_it_writes_the_protocol(tmp_path, final, settings):
    final.modules["toy"] = toy_module(CHOSEN={"toy": settings})
    protocol = tmp_path / "protocol.json"

    with pytest.raises(ValueError, match="window"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)
    assert not protocol.exists()


def test_final_test_rejects_an_invalid_refit_frequency_before_it_writes_the_protocol(tmp_path, final):
    final.modules["toy"] = toy_module(CHOSEN={"toy": {"shift": 0.0, "window": "expanding"}}, REFIT_EVERY=0)
    protocol = tmp_path / "protocol.json"

    with pytest.raises(ValueError, match="refit_every"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)
    assert not protocol.exists()


def test_final_test_names_missing_test_days_before_it_writes_the_protocol(tmp_path, final, monkeypatch):
    def table(inputs):
        frame = synthetic_table(first="2026-06-01", days=94)  # ends 2026-09-02, test runs to 09-05
        frame.loc[frame["delivery_day"] == pd.Timestamp("2026-09-01"), "aep_ct_kwh"] = float("nan")
        return frame

    monkeypatch.setattr(experiment, "build_table", table)
    protocol = tmp_path / "protocol.json"

    with pytest.raises(ValueError, match="2026-09-01.*2026-09-03"):
        experiment.final_test(True, jobs=1, out_dir=tmp_path, protocol=protocol, names=["toy"], design=SHORT)
    assert not protocol.exists()


class GapModel(ShiftModel):
    """No forecast on 2026-04-30 for the shifted setting: a broken candidate."""

    def predict(self, table):
        out = super().predict(table)
        if self.shift:
            out.loc[table["delivery_day"] == pd.Timestamp("2026-04-30")] = float("nan")
        return out


def test_select_fails_when_a_candidate_lacks_too_many_days(tmp_path, toy):
    toy.modules["toy"] = toy_module(MODEL=GapModel)  # 1 of 7 days = 14 % > 5 %

    with pytest.raises(ValueError, match=r"3\.0\|expanding"):
        experiment.select("toy", jobs=1, out_dir=tmp_path, days=SMOKE_DAYS)


def test_select_records_and_reports_dropped_days(tmp_path, toy, monkeypatch, capsys):
    toy.modules["toy"] = toy_module(MODEL=GapModel)
    monkeypatch.setattr(experiment, "MAX_MISSING_SHARE", 0.5)

    experiment.select("toy", jobs=1, out_dir=tmp_path, days=SMOKE_DAYS)

    selection = json.loads((tmp_path / "select_selection.json").read_text())
    assert selection["n_days"] == 6
    [dropped] = selection["dropped_days"]
    assert dropped["day"] == "2026-04-30"
    assert set(dropped["models"]) == {"3.0|expanding", "3.0|rolling_56d", "3.0|rolling_90d", "3.0|window_mean"}
    info = json.loads((tmp_path / "select_run_info.json").read_text())
    assert info["selection"]["dropped_days"] == selection["dropped_days"]
    assert "2026-04-30" in capsys.readouterr().out
