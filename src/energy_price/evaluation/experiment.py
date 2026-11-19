"""Select, compare and finally test the models in ``src/energy_price/models/<name>/``.

Usage (from the repository root; ``make select MODEL=<name>``, ``make compare`` and
``make final-test UNLOCK_TEST=1`` do the same):

``python -m energy_price.evaluation.experiment <name> select``   every GRID setting x window on all validation days
``python -m energy_price.evaluation.experiment compare``         CHOSEN of every model, reference and B2 on the same days
``python -m energy_price.evaluation.experiment final-test --unlock-test``   only once, after the freeze (issue #53)

Options: ``--jobs N`` parallel folds, ``--track`` W&B runs (grouped by the issue number in the branch
name or ``ISSUE`` in model.py, docs/experiment_tracking.md), ``--refit-every K`` (select only) and
``--days START END`` (select only, a smoke test on part of the validation period). Results go to
``reports/<name>/select_*`` and ``reports/compare/`` (not versioned), figures to ``docs/figures/<name>/``.

``select`` and ``compare`` build their table only from ``development_inputs``: the raw data of the
test period never enters a development run. The design (periods, windows, refit) is ``evaluation.design``.

A model folder only needs a ``model.py`` with:

- ``MODEL``    class (or function) that returns a new, unfitted model for the keyword settings
- ``GRID``     list of settings compared by ``select``, WITHOUT the training window
- ``CHOSEN``   ``{variant name: settings incl. "window"}``, copied by hand from ``select_selection.json``;
               ``"window"`` is one of ``design.WINDOW_DAYS`` or ``"window_mean"``

Optional: ``WINDOWS`` (default ``DESIGN.windows``), ``REFIT_EVERY`` (default ``DESIGN.refit_every``),
``ISSUE`` for W&B grouping, ``RUN_INFO`` for notes, ``COEFFICIENTS_SHOWN`` for a coefficient figure,
``IMPORTANCE_GROUPS`` for permutation importance of the selected candidate, and
``select_candidates(scores, settings_by_name)`` returning named CHOSEN settings for experiments that
select separately for multiple base models; they are stored under ``chosen`` in the selection record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from functools import partial
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pandas as pd

from energy_price.evaluation.backtest import (
    BASELINE,
    REFERENCE,
    REFERENCE_WINDOW_DAYS,
    common_rows,
    compare,
    development_inputs,
    evaluate,
    folds,
    missing_days,
    run_info,
    run_model,
    run_reference,
    track,
    track_reference,
    window_mean,
    write_outputs,
)
from energy_price.evaluation.design import (
    DESIGN,
    WINDOW_MEAN,
    Design,
    TestLockedError,
    block_of,
    test_days,
    validation_days,
    window_days,
)
from energy_price.evaluation.diagnostics import backtest_importance, breakdown_scores, importance_summary
from energy_price.evaluation.figures import backtest_figures, coefficients_figure
from energy_price.evaluation.scores import daily_pinball, score
from energy_price.features.dataset import TARGET, build_table, load_inputs, quarter_hours
from energy_price.paths import FIGURES_DIR, REPORTS_DIR, ROOT

BASELINE_MODEL, BASELINE_VARIANT = "baseline_model", "B2"
MODELS_DIR = ROOT / "src" / "energy_price" / "models"
COMPARE_DIR = REPORTS_DIR / "compare"
TEST_PROTOCOL = ROOT / "docs" / "test_protocol.json"
# A candidate without a complete forecast on more than this share of the days is broken: select stops.
MAX_MISSING_SHARE = 0.05


def load(name: str) -> ModuleType:
    """The ``model.py`` of the model folder ``name``."""
    return import_module(f"energy_price.models.{name}.model")


def model_names() -> list[str]:
    """All model folders with a ``model.py``, without ``_template`` and ``shared``."""
    return sorted(p.parent.name for p in MODELS_DIR.glob("*/model.py") if not p.parent.name.startswith("_"))


def windows_of(module: ModuleType, design: Design = DESIGN) -> tuple[str, ...]:
    """Training windows compared for a model: ``WINDOWS`` in model.py, else the design default."""
    windows = tuple(getattr(module, "WINDOWS", design.windows))
    for window in windows:
        window_days(window)  # rejects unknown names
    if not windows:
        raise ValueError("WINDOWS must not be empty")
    return windows


def refit_every_of(module: ModuleType, override: int | None = None, design: Design = DESIGN) -> int:
    """Refit frequency of a model: the CLI value, else ``REFIT_EVERY`` in model.py, else the design default."""
    refit = override if override is not None else getattr(module, "REFIT_EVERY", design.refit_every)
    if refit < 1:
        raise ValueError("refit_every must be at least 1")
    return int(refit)


def label(settings: dict) -> str:
    return "|".join(str(v) for v in settings.values())


def split(settings: dict) -> tuple[int | None, dict]:
    """Training window in days and the remaining settings of a single-window setting."""
    params = dict(settings)
    if "window" not in params:
        raise ValueError(f"settings {settings} have no 'window'; CHOSEN settings must name their training window")
    return window_days(params.pop("window")), params


def grid_settings(module: ModuleType) -> list[dict]:
    """The GRID of a model, checked: not empty, distinct, without ``window`` (the windows come from the design)."""
    grid = list(module.GRID)
    if not grid:
        raise ValueError("GRID must not be empty")
    if any("window" in settings for settings in grid):
        raise ValueError("GRID settings must not contain 'window': select runs every setting with every window "
                         "of WINDOWS (default DESIGN.windows); remove it from GRID")
    if len({label(settings) for settings in grid}) != len(grid):
        raise ValueError("GRID must contain distinct settings")
    return grid


def development_table(design: Design = DESIGN) -> pd.DataFrame:
    """Feature table of the development period, built only from raw data before the test period."""
    table = build_table(development_inputs(load_inputs(), design))
    if not table.empty and table["delivery_day"].max() >= design.test_start:
        raise RuntimeError("the development table reaches into the test period")
    return table


def check_days(days: tuple[str, str] | None, design: Design = DESIGN) -> list[pd.Timestamp]:
    """All validation days, or the days from ``days`` (a smoke test) if they lie inside the validation period."""
    all_days = validation_days(design)
    if days is None:
        return all_days
    chosen = list(pd.date_range(*days, freq="D"))
    if not chosen or chosen[0] < all_days[0] or chosen[-1] > all_days[-1]:
        raise ValueError(f"--days must lie inside the validation period {design.validation[0]} to "
                         f"{design.validation[1]}")
    return chosen


def run_window(module: ModuleType, table: pd.DataFrame, days: list[pd.Timestamp], name: str, params: dict,
               window: str, refit_every: int, jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Forecasts, coefficients and runtime of one setting with one training window."""
    forecasts, coefficients, runtime = run_model(table, days, partial(module.MODEL, **params), name,
                                                 window_days(window), refit_every, jobs)
    if coefficients is not None:
        coefficients = coefficients.assign(model=name, window=window)
    return forecasts, coefficients, runtime.assign(window=window)


def run_chosen(module: ModuleType, table: pd.DataFrame, days: list[pd.Timestamp], name: str, settings: dict,
               jobs: int = -1, refit_every: int | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Forecasts, coefficients and runtime of a CHOSEN setting (incl. ``window``, which may be ``window_mean``).

    The refit frequency is ``refit_every`` or the model's own (``REFIT_EVERY``, else the design default).
    """
    refit = refit_every_of(module, refit_every)
    params = dict(settings)
    if params.get("window") != WINDOW_MEAN:
        split(params)  # checks that the window is named and known
        window = params.pop("window")
        return run_window(module, table, days, name, params, window, refit, jobs)
    params.pop("window")
    parts = [run_window(module, table, days, f"{name}|{window}", params, window, refit, jobs)
             for window in windows_of(module)]
    forecasts = window_mean([p[0] for p in parts], name)
    coefs = [p[1] for p in parts if p[1] is not None]
    coefficients = pd.concat(coefs, ignore_index=True).assign(model=name) if coefs else None
    return forecasts, coefficients, pd.concat([p[2] for p in parts], ignore_index=True).assign(model=name)


def run_variant(module: ModuleType, table: pd.DataFrame, days: list[pd.Timestamp], name: str, settings: dict,
                jobs: int = -1, refit_every: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Forecasts (and coefficients, if the model has them) of one CHOSEN setting, see ``run_chosen``."""
    forecasts, coefficients, _ = run_chosen(module, table, days, name, settings, jobs, refit_every)
    return forecasts, coefficients


def diagnostic_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Join actual direction only for evaluation, if the processed balance table exists."""
    path = ROOT / "data" / "processed" / "control_area_balance.parquet"
    balance = pd.read_parquet(path, columns=["timestamp_utc", "tsi_mw"]) if path.exists() else None
    return breakdown_scores(forecasts, balance)


def tracking_files(out_dir: Path, step: str) -> list[Path]:
    """Explicit allowlist: source code and aggregate results, never credentials or raw data.

    Include the source snapshot so an uncommitted run is reproducible beyond its git SHA.
    """
    return [*sorted((ROOT / "src" / "energy_price").rglob("*.py")),
            ROOT / "pyproject.toml", ROOT / "requirements.lock",
            *sorted(out_dir.glob(f"{step}_*.csv")), *sorted(out_dir.glob(f"{step}_*.json"))]


def block_scores(forecasts: pd.DataFrame, candidates: list[str], reference: str = REFERENCE) -> pd.DataFrame:
    """Scores per model over all days (``block`` = ``"all"``) and per validation block, on the common rows.

    ``rank`` orders the candidates within a block by ``mean_daily_pinball`` (1 = best); the reference
    and B2 have no rank.
    """
    rows = common_rows(forecasts)
    blocks = rows["delivery_day"].map(block_of)
    records = []
    for block, part in [("all", rows), *rows.groupby(blocks, sort=True)]:
        for model, group in part.groupby("model", sort=False):
            per_day = daily_pinball(group)
            records.append({"model": model, "block": block, **score(group), "n_days": len(per_day),
                            "mean_daily_pinball": per_day.mean()})
    out = pd.DataFrame(records)
    if reference in set(out["model"]):
        ref = out[out["model"] == reference].set_index("block")["mean_daily_pinball"]
        out["skill_vs_reference"] = 1 - out["mean_daily_pinball"] / out["block"].map(ref)
    is_candidate = out["model"].isin(candidates)
    out["rank"] = out[is_candidate].groupby("block")["mean_daily_pinball"].rank(method="min")
    return out


def select(name: str, jobs: int = -1, out_dir: Path | None = None, do_track: bool = False,
           refit_every: int | None = None, days: tuple[str, str] | None = None,
           design: Design = DESIGN) -> pd.DataFrame:
    """Every GRID setting x window (and the window mean of each setting) on all validation days.

    Also runs the reference and, for every model except the baseline, the baseline B2 on the same days.
    Scores per candidate over all days and per month block; the selected candidate has the lowest
    mean daily pinball loss over all validation days. Writes ``select_*`` to ``out_dir``.
    """
    module = load(name)
    out_dir = out_dir or REPORTS_DIR / name
    grid = grid_settings(module)
    windows = windows_of(module, design)
    refit = refit_every_of(module, refit_every, design)
    days = check_days(days, design)
    table = development_table(design)

    frames, coefficients, runtimes = [run_reference(table, days)], [], []
    candidates: dict[str, dict] = {}
    for index, settings in enumerate(grid, start=1):
        print(f"select {index}/{len(grid)}: {label(settings) or '(defaults)'}", flush=True)
        per_window = []
        for window in windows:
            variant = label({**settings, "window": window})
            forecast, coef, runtime = run_window(module, table, days, variant, settings, window, refit, jobs)
            per_window.append(forecast)
            candidates[variant] = {**settings, "window": window}
            runtimes.append(runtime.assign(setting=label(settings)))
            if coef is not None:
                coefficients.append(coef)
        frames += per_window
        if len(windows) > 1:
            variant = label({**settings, "window": WINDOW_MEAN})
            frames.append(window_mean(per_window, variant))
            candidates[variant] = {**settings, "window": WINDOW_MEAN}

    baseline_settings = baseline_refit = None
    if name != BASELINE_MODEL:
        baseline_module = load(BASELINE_MODEL)
        baseline_settings = baseline_module.CHOSEN[BASELINE_VARIANT]
        baseline_refit = refit_every_of(baseline_module, None, design)
        forecast, _, runtime = run_chosen(baseline_module, table, days, BASELINE, baseline_settings, jobs)
        frames.append(forecast)
        runtimes.append(runtime.assign(setting=BASELINE))

    forecasts = pd.concat(frames, ignore_index=True)
    dropped = check_dropped_days(forecasts, days, list(candidates))
    scores, daily = evaluate(forecasts)
    by_block = block_scores(forecasts, list(candidates))
    best = scores.loc[list(candidates), "mean_daily_pinball"].idxmin()
    period = (str(days[0].date()), str(days[-1].date()))
    selection = {"variant": best, "settings": candidates[best], "metric": "mean_daily_pinball",
                 "value": float(scores.loc[best, "mean_daily_pinball"]), "period": list(period),
                 "n_days": int(scores.loc[best, "n_days"]), "refit_every": refit, "windows": list(windows),
                 "dropped_days": dropped}
    selected = {best}
    if hasattr(module, "select_candidates"):
        selection["chosen"] = module.select_candidates(scores, candidates)
        selected = {key for key, value in candidates.items() if value in selection["chosen"].values()}

    runtime = pd.concat(runtimes, ignore_index=True)
    info = run_info({"model": name, "step": "select", "period": list(period), "design": design.as_config(),
                     "windows": list(windows), "refit_every": refit, "grid": grid, "selection": selection,
                     "baseline": {"settings": baseline_settings, "refit_every": baseline_refit},
                     "reference_window_days": REFERENCE_WINDOW_DAYS, "runtime": runtime_summary(runtime),
                     **getattr(module, "RUN_INFO", {})})
    write_outputs(out_dir, "select", forecasts, by_block, daily, info,
                  pd.concat(coefficients, ignore_index=True) if coefficients else None, runtime)
    (out_dir / "select_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    breakdown = diagnostic_scores(forecasts)
    breakdown.to_csv(out_dir / "select_breakdowns.csv", index=False)

    importance = None
    groups = getattr(module, "IMPORTANCE_GROUPS", None)
    if groups and candidates[best]["window"] == WINDOW_MEAN:
        print("select: no permutation importance for a window mean, it has no single fitted model", flush=True)
    elif groups:
        print("select: grouped permutation importance of the selected candidate on validation folds", flush=True)
        window, params = split(candidates[best])
        details = backtest_importance(table, days, partial(module.MODEL, **params), groups, window,
                                      getattr(module, "IMPORTANCE_REPEATS", 5), jobs=jobs, refit_every=refit)
        details.to_csv(out_dir / "select_feature_importance_repeats.csv", index=False)
        importance = importance_summary(details)
        importance.to_csv(out_dir / "select_feature_importance.csv", index=False)

    if do_track:
        issue = getattr(module, "ISSUE", None)
        n_folds = len(folds(days, refit))
        urls = {REFERENCE: track_reference("select", scores, len(days), period, issue=issue,
                                           daily=daily[REFERENCE])}
        for variant, settings in candidates.items():
            tables = {"breakdowns": breakdown[breakdown["model"] == variant].reset_index(drop=True)}
            if variant == best and importance is not None:
                tables["feature_importance"] = importance
            urls[variant] = track("select", variant, {"model": name, **settings, **getattr(module, "RUN_INFO", {}),
                                                      "selected": variant in selected},
                                  scores.loc[variant], n_folds, period, issue=issue, daily=daily[variant],
                                  reference_daily=daily[REFERENCE], tables=tables,
                                  files=tracking_files(out_dir, "select") if variant == best else None,
                                  refit_every=refit, design=design)
        if baseline_settings is not None:
            urls[BASELINE] = track("select", BASELINE, {"model": BASELINE_MODEL, **baseline_settings},
                                   scores.loc[BASELINE], len(folds(days, baseline_refit)), period, issue=issue,
                                   daily=daily[BASELINE], reference_daily=daily[REFERENCE],
                                   refit_every=baseline_refit, design=design)
        (out_dir / "select_wandb_urls.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
    return by_block


def check_dropped_days(forecasts: pd.DataFrame, days: list[pd.Timestamp], candidates: list[str]) -> list[dict]:
    """Days that drop out of the comparison, each with the models lacking a complete forecast.

    A dropped day is removed for every candidate, so one broken candidate could change the winner. The
    days are printed and recorded; a candidate missing more than ``MAX_MISSING_SHARE`` of the days stops
    the run with an error.
    """
    missing = missing_days(forecasts, days)
    share = {c: sum(c in models for models in missing.values()) / len(days) for c in candidates}
    broken = {c: s for c, s in share.items() if s > MAX_MISSING_SHARE}
    if broken:
        listed = ", ".join(f"{c} ({s:.0%})" for c, s in sorted(broken.items()))
        raise ValueError(f"candidates without a complete forecast on more than {MAX_MISSING_SHARE:.0%} of the "
                         f"days: {listed}")
    dropped = [{"day": str(day.date()), "models": models} for day, models in missing.items()]
    for record in dropped:
        print(f"select: WARNING {record['day']} drops out for all models, incomplete: "
              f"{', '.join(record['models'])}", flush=True)
    return dropped


def runtime_summary(runtime: pd.DataFrame) -> dict:
    """Number of fits and seconds per fit (fit and predict of one fold), summed over all candidates."""
    seconds = runtime["seconds"]
    return {"n_fits": int(len(runtime)), "seconds_total": float(seconds.sum()),
            "seconds_per_fit_mean": float(seconds.mean()), "seconds_per_fit_max": float(seconds.max())}


def chosen_forecasts(names: list[str], table: pd.DataFrame, days: list[pd.Timestamp],
                     jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Reference, baseline B2 and every CHOSEN variant of ``names`` (as ``<model>/<variant>``) on ``days``."""
    baseline_module = load(BASELINE_MODEL)
    forecast, _, runtime = run_chosen(baseline_module, table, days, BASELINE,
                                      baseline_module.CHOSEN[BASELINE_VARIANT], jobs)
    frames, runtimes = [run_reference(table, days), forecast], [runtime]
    settings = {BASELINE: {"model": BASELINE_MODEL, **baseline_module.CHOSEN[BASELINE_VARIANT]}}
    for name in names:
        module = load(name)
        if not module.CHOSEN:
            raise ValueError(f"CHOSEN of {name} is empty: run select and fix the selected settings first")
        for variant, chosen in module.CHOSEN.items():
            if name == BASELINE_MODEL and variant == BASELINE_VARIANT:
                continue  # already in as the baseline B2
            label_ = f"{name}/{variant}"
            print(f"compare: {label_}", flush=True)
            forecast, _, runtime = run_chosen(module, table, days, label_, chosen, jobs)
            frames.append(forecast)
            runtimes.append(runtime)
            settings[label_] = {"model": name, **chosen, "refit_every": refit_every_of(module)}
    return pd.concat(frames, ignore_index=True), pd.concat(runtimes, ignore_index=True), settings


def _compare_step(step: str, table: pd.DataFrame, days: list[pd.Timestamp], names: list[str], jobs: int,
                  out_dir: Path, do_track: bool, design: Design, extra: dict | None = None) -> pd.DataFrame:
    forecasts, runtime, settings = chosen_forecasts(names, table, days, jobs)
    comparison = compare(forecasts)
    scores, daily = evaluate(forecasts)
    period = (str(days[0].date()), str(days[-1].date()))
    info = run_info({"step": step, "period": list(period), "design": design.as_config(), "models": settings,
                     "reference_window_days": REFERENCE_WINDOW_DAYS, "runtime": runtime_summary(runtime),
                     **(extra or {})})
    write_outputs(out_dir, step, forecasts, comparison, daily, info, runtime=runtime)
    scores.to_csv(out_dir / f"{step}_model_scores.csv")
    if do_track:
        urls = {REFERENCE: track_reference(step, scores, len(days), period, daily=daily[REFERENCE])}
        for model, config in settings.items():
            module = load(config["model"])
            refit = config.get("refit_every", refit_every_of(module))
            urls[model] = track(step, model, {**config, **getattr(module, "RUN_INFO", {})}, scores.loc[model],
                                len(folds(days, refit)), period, issue=getattr(module, "ISSUE", None),
                                daily=daily[model], reference_daily=daily[REFERENCE],
                                files=tracking_files(out_dir, step), refit_every=refit, design=design)
        (out_dir / f"{step}_wandb_urls.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
    return comparison


def run_compare(jobs: int = -1, out_dir: Path = COMPARE_DIR, do_track: bool = False,
                names: list[str] | None = None, design: Design = DESIGN) -> pd.DataFrame:
    """CHOSEN of every model, reference and B2 rerun on all validation days and compared on identical rows."""
    table = development_table(design)
    return _compare_step("validation", table, validation_days(design), names or model_names(), jobs, out_dir,
                         do_track, design)


def git_status() -> str:
    """``git status --porcelain`` of the repository; empty for a clean tree."""
    return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout.strip()


def check_chosen(name: str, module: ModuleType, design: Design = DESIGN) -> None:
    """CHOSEN of a model is runnable: not empty, every setting names a known window, valid refit frequency."""
    if not module.CHOSEN:
        raise ValueError(f"CHOSEN of {name} is empty: run select and fix the selected settings first")
    for variant, settings in module.CHOSEN.items():
        window = settings.get("window")
        if window is None:
            raise ValueError(f"CHOSEN {name}/{variant} has no 'window'")
        if window == WINDOW_MEAN:
            windows_of(module, design)
        else:
            window_days(window)
    refit_every_of(module, None, design)
    if name == BASELINE_MODEL and BASELINE_VARIANT not in module.CHOSEN:
        raise ValueError(f"CHOSEN of {name} has no {BASELINE_VARIANT}, the comparison model")


def check_test_days(table: pd.DataFrame, days: list[pd.Timestamp]) -> None:
    """Every test day has all its quarter hours (92, 96 or 100) and a price in each; else name the days."""
    bad = []
    for day in days:
        rows = table[table["delivery_day"] == day]
        if len(rows) != len(quarter_hours([day])) or rows[TARGET].isna().any():
            bad.append(str(day.date()))
    if bad:
        raise ValueError(f"test days without all quarter hours or prices: {', '.join(bad)}")


def final_test(unlock_test: bool = False, jobs: int = -1, out_dir: Path = COMPARE_DIR,
               protocol: Path = TEST_PROTOCOL, names: list[str] | None = None, design: Design = DESIGN,
               do_track: bool = False) -> pd.DataFrame:
    """The one evaluation on the test period (issue #53), after the freeze. Scaffold, not run in #36.

    Requires ``unlock_test``, a clean git tree and no ``docs/test_protocol.json``. Before the protocol is
    written, every CHOSEN setting and refit frequency is checked and every test day must be in the table
    with all its quarter hours and prices, so a broken setup cannot use up the one run. The protocol
    (commit, CHOSEN of all models, design, time) is written BEFORE anything is scored, so a second run
    is refused.
    """
    if not unlock_test:
        raise TestLockedError("the test period is locked; run final-test --unlock-test exactly once after the freeze")
    if git_status():
        raise RuntimeError("final-test requires a clean git tree: commit the frozen CHOSEN settings first")
    if protocol.exists():
        raise RuntimeError(f"{protocol} exists: the test period has already been evaluated, it is run only once")
    days = test_days(design, unlock=True)
    names = names or model_names()
    for name in dict.fromkeys([BASELINE_MODEL, *names]):
        check_chosen(name, load(name), design)
    table = build_table(load_inputs())
    check_test_days(table, days)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", check=True).stdout.strip()
    record = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
              "design": design.as_config(), "period": [str(days[0].date()), str(days[-1].date())],
              "chosen": {name: load(name).CHOSEN for name in [BASELINE_MODEL, *names]},
              "refit_every": {name: refit_every_of(load(name)) for name in [BASELINE_MODEL, *names]}}
    protocol.parent.mkdir(parents=True, exist_ok=True)
    with protocol.open("x", encoding="utf-8") as handle:  # fails if a parallel run wrote it first
        json.dump(record, handle, indent=2)
    return _compare_step("test", table, days, names, jobs, out_dir, do_track, design,
                         {"protocol": str(protocol), "unlock_test": True})


def figures(name: str, out_dir: Path, fig_dir: Path, step: str = "select") -> list[Path]:
    """Pinball loss per hour and coverage per month of the reference, the selected candidate and B2.

    With ``COEFFICIENTS_SHOWN`` in model.py also the coefficients of the selected candidate, if it has them.
    """
    module = load(name)
    best = json.loads((out_dir / f"{step}_selection.json").read_text(encoding="utf-8"))["variant"]
    paths = backtest_figures(out_dir, step, fig_dir, [REFERENCE, best, BASELINE])
    shown = getattr(module, "COEFFICIENTS_SHOWN", None)
    path = out_dir / f"{step}_coefficients.csv"
    if shown and path.exists():
        coefficients = pd.read_csv(path, usecols=["feature", "model"])
        if set(shown) <= set(coefficients.loc[coefficients["model"] == best, "feature"]):
            paths.append(coefficients_figure(out_dir, step, fig_dir, best, shown))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="model folder in src/energy_price/models/ (with step select), "
                                       "or compare, or final-test")
    parser.add_argument("step", nargs="?", choices=["select"], help="select: GRID x windows on the validation days")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds, -1 uses all cores")
    parser.add_argument("--track", action="store_true", help="log the runs to W&B")
    parser.add_argument("--refit-every", type=int, default=None, help="select: refit every K days (default: model)")
    parser.add_argument("--days", nargs=2, metavar=("START", "END"), default=None,
                        help="select: only these validation days, for smoke tests")
    parser.add_argument("--unlock-test", action="store_true", help="final-test: release the test period once")
    parser.add_argument("--out", type=Path, default=None, help="default reports/<model>/ or reports/compare/")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", None, "display.max_rows", 200)

    if args.target in ("compare", "final-test"):
        if args.step or args.days or args.refit_every is not None:
            parser.error(f"{args.target} takes no step, --days or --refit-every")
        out = args.out or COMPARE_DIR
        if args.target == "compare":
            result = run_compare(args.jobs, out, args.track)
        else:
            result = final_test(args.unlock_test, args.jobs, out, do_track=args.track)
        print(result.round(4).to_string(index=False))
    else:
        if args.step != "select":
            parser.error("a model needs the step select, e.g. baseline_model select")
        if args.unlock_test:
            parser.error("--unlock-test belongs to final-test")
        out = args.out or REPORTS_DIR / args.target
        scores = select(args.target, args.jobs, out, args.track, args.refit_every,
                        tuple(args.days) if args.days else None)
        if args.days:
            print("smoke run (--days): no figures in docs/figures/")
        else:
            for path in figures(args.target, out, FIGURES_DIR / args.target):
                print(f"figure {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        overall = scores[scores["block"] == "all"].sort_values("mean_daily_pinball")
        print(overall.round(4).to_string(index=False))
        runtime = pd.read_csv(out / "select_runtime.csv")
        print("runtime per window (seconds per fit, fit and predict of one fold):")
        print(runtime.groupby("window")["seconds"].agg(["count", "mean", "max", "sum"]).round(3).to_string())
        print(json.dumps(runtime_summary(runtime)))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
