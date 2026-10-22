"""Run the backtest of any model in ``src/energy_price/models/<name>/`` against the reference.

Usage (from the repository root, ``make backtest MODEL=<name> STEP=<step>`` does the same):
``python -m energy_price.evaluation.experiment <name> tune``      compare the GRID on April and May
``python -m energy_price.evaluation.experiment <name> test``      CHOSEN settings on June to August, with figures
``python -m energy_price.evaluation.experiment <name> holdout --unlock-holdout``   only once, at the very end

``--track`` logs every variant as a W&B run, grouped by the issue number in the branch name
(``exp/15-...`` gives ``issue-15``, or set ``ISSUE`` in model.py, docs/experiment_tracking.md). Results go to
``reports/<name>/`` (not versioned), figures to ``docs/figures/<name>/``. Every model except the
baseline is also compared with the baseline B2 when ``make baseline`` has run.

A model folder only needs a ``model.py`` with:

- ``MODEL``    class (or function) that returns a new, unfitted model for the keyword settings
- ``CHOSEN``   ``{variant name: settings}`` used on the test period, fixed after tuning
- ``GRID``     list of settings compared on the tuning period

Settings are keyword arguments of ``MODEL``; the key ``window`` is taken by the backtest
(``"expanding"``: all prices since 2026-01-01, ``"rolling_56d"``: the last 56 days). Optional:
``COEFFICIENTS_SHOWN`` and ``COEFFICIENTS_OF`` for a coefficient figure, ``RUN_INFO`` for notes,
``ISSUE`` for W&B grouping and ``IMPORTANCE_GROUPS`` for tuning-fold permutation importance.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pandas as pd

from energy_price.evaluation.backtest import (
    PERIODS,
    REFERENCE,
    REFERENCE_WINDOW_DAYS,
    REFIT_DAYS,
    evaluate,
    folds,
    period_days,
    run_info,
    run_model,
    run_reference,
    track,
    track_reference,
    write_outputs,
)
from energy_price.features.dataset import build_table, load_inputs
from energy_price.paths import FIGURES_DIR, REPORTS_DIR, ROOT
from energy_price.evaluation.figures import backtest_figures, coefficients_figure
from energy_price.evaluation.diagnostics import backtest_importance, breakdown_scores, importance_summary

WINDOWS = {"expanding": None, "rolling_56d": 56}
BASELINE, BASELINE_VARIANT = "baseline_model", "B2"


def load(name: str) -> ModuleType:
    """The ``model.py`` of the model folder ``name``."""
    return import_module(f"energy_price.models.{name}.model")


def split(settings: dict) -> tuple[int | None, dict]:
    """Training window in days and the remaining settings for the model."""
    params = dict(settings)
    return WINDOWS[params.pop("window", "expanding")], params


def run_variant(module: ModuleType, table: pd.DataFrame, days: list[pd.Timestamp], name: str, settings: dict,
                jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Forecasts (and coefficients, if the model has them) of one variant, refitted every week."""
    window_days, params = split(settings)
    return run_model(table, days, partial(module.MODEL, **params), name, window_days, jobs)


def label(settings: dict) -> str:
    return "|".join(str(v) for v in settings.values())


def diagnostic_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Join actual direction only for evaluation, if the processed balance table exists."""
    path = ROOT / "data" / "processed" / "control_area_balance.parquet"
    balance = pd.read_parquet(path, columns=["timestamp_utc", "tsi_mw"]) if path.exists() else None
    return breakdown_scores(forecasts, balance)


def tracking_files(out_dir: Path, period: str) -> list[Path]:
    """Explicit allowlist: source code and aggregate results, never credentials or raw data.

    Include the source snapshot so an uncommitted run is reproducible beyond its git SHA.
    """
    return [*sorted((ROOT / "src" / "energy_price").rglob("*.py")),
            ROOT / "pyproject.toml", ROOT / "requirements.lock",
            *sorted(out_dir.glob(f"{period}_*.csv")), *sorted(out_dir.glob(f"{period}_*.json"))]


def tune(name: str, table: pd.DataFrame, jobs: int, out_dir: Path, do_track: bool = False) -> pd.DataFrame:
    """Every GRID entry on April/May, plus baseline, diagnostics and optional W&B runs."""
    module = load(name)
    days = period_days("tune")
    frames = [run_reference(table, days)]
    settings_by_name = {label(settings): settings for settings in module.GRID}
    if not settings_by_name or len(settings_by_name) != len(module.GRID):
        raise ValueError("GRID must be nonempty and contain distinct settings")
    for index, settings in enumerate(module.GRID, start=1):
        print(f"tune {index}/{len(module.GRID)}: {label(settings)}", flush=True)
        forecast, _ = run_variant(module, table, days, label(settings), settings, jobs)
        frames.append(forecast)
    baseline_name = f"baseline_{BASELINE_VARIANT}"
    baseline_settings = None
    if name != BASELINE:
        baseline_module = load(BASELINE)
        baseline_settings = baseline_module.CHOSEN[BASELINE_VARIANT]
        baseline, _ = run_variant(baseline_module, table, days, baseline_name, baseline_settings, jobs)
        frames.append(baseline)
    forecasts = pd.concat(frames, ignore_index=True)
    scores, daily = evaluate(forecasts)
    best = scores.loc[list(settings_by_name), "pinball"].idxmin()
    selection = {"variant": best, "settings": settings_by_name[best], "metric": "pinball",
                 "value": float(scores.loc[best, "pinball"]), "period": list(PERIODS["tune"])}
    info = run_info({"model": name, "period": "tune", "grid": module.GRID,
                     "selection": selection, "refit_days": REFIT_DAYS,
                     "reference_window_days": REFERENCE_WINDOW_DAYS, **getattr(module, "RUN_INFO", {})})
    write_outputs(out_dir, "tune", forecasts, scores, daily, info)
    (out_dir / "tune_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    breakdown = diagnostic_scores(forecasts)
    breakdown.to_csv(out_dir / "tune_breakdowns.csv", index=False)

    importance = None
    groups = getattr(module, "IMPORTANCE_GROUPS", None)
    if groups:
        print("tune: grouped permutation importance of the best candidate on validation folds", flush=True)
        window, params = split(settings_by_name[best])
        details = backtest_importance(table, days, partial(module.MODEL, **params), groups, window,
                                     getattr(module, "IMPORTANCE_REPEATS", 5), jobs=jobs)
        details.to_csv(out_dir / "tune_feature_importance_repeats.csv", index=False)
        importance = importance_summary(details)
        importance.to_csv(out_dir / "tune_feature_importance.csv", index=False)

    if do_track:
        issue = getattr(module, "ISSUE", None)
        n_folds = len(folds(days))
        track_reference("tune", scores, n_folds, issue=issue, daily=daily[REFERENCE])
        urls = {}
        for variant, settings in settings_by_name.items():
            tables = {"breakdowns": breakdown[breakdown["model"] == variant].reset_index(drop=True)}
            if variant == best and importance is not None:
                tables["feature_importance"] = importance
            urls[variant] = track("tune", variant, {"model": name, **settings,
                                  **getattr(module, "RUN_INFO", {}), "selected": variant == best},
                                  scores.loc[variant], n_folds, PERIODS["tune"], issue=issue,
                                  daily=daily[variant], reference_daily=daily[REFERENCE], tables=tables,
                                  files=tracking_files(out_dir, "tune") if variant == best else None)
        if baseline_settings is not None:
            urls[baseline_name] = track("tune", baseline_name, {"model": BASELINE, **baseline_settings},
                                       scores.loc[baseline_name], n_folds, PERIODS["tune"], issue=issue,
                                       daily=daily[baseline_name], reference_daily=daily[REFERENCE])
        (out_dir / "tune_wandb_urls.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
    return scores


def baseline_forecasts(name: str, period: str) -> pd.DataFrame | None:
    """Forecasts of the baseline B2 for ``period`` from ``make baseline``, for every other model."""
    path = REPORTS_DIR / BASELINE / f"{period}_forecasts.parquet"
    if name == BASELINE or not path.exists():
        return None
    forecasts = pd.read_parquet(path)
    return forecasts[forecasts["model"] == BASELINE_VARIANT].assign(model=f"baseline_{BASELINE_VARIANT}")


def evaluate_period(name: str, table: pd.DataFrame, period: str, jobs: int, out_dir: Path,
                    unlock_holdout: bool = False, do_track: bool = False) -> pd.DataFrame:
    """Reference and every variant of ``CHOSEN`` on ``period``; writes forecasts, scores and coefficients."""
    module = load(name)
    if not module.CHOSEN:
        raise ValueError("CHOSEN is empty: run tuning and fix the selected settings before testing")
    days = period_days(period, unlock_holdout)
    frames, coefficients = [run_reference(table, days)], []
    for variant, settings in module.CHOSEN.items():
        forecast, coef = run_variant(module, table, days, variant, settings, jobs)
        frames.append(forecast)
        if coef is not None:
            coefficients.append(coef.assign(model=variant))
    baseline = baseline_forecasts(name, period)
    if baseline is not None:
        frames.append(baseline)
    forecasts = pd.concat(frames, ignore_index=True)
    scores, daily = evaluate(forecasts)

    info = run_info({"model": name, "period": period, "days": [str(d.date()) for d in (days[0], days[-1])],
                     "chosen": module.CHOSEN, "reference_window_days": REFERENCE_WINDOW_DAYS,
                     "refit_days": REFIT_DAYS, **getattr(module, "RUN_INFO", {})})
    write_outputs(out_dir, period, forecasts, scores, daily, info,
                  pd.concat(coefficients, ignore_index=True) if coefficients else None)
    breakdown = diagnostic_scores(forecasts)
    breakdown.to_csv(out_dir / f"{period}_breakdowns.csv", index=False)

    if do_track:
        n_folds = len(folds(days))
        issue = getattr(module, "ISSUE", None)
        urls = {REFERENCE: track_reference(period, scores, n_folds, issue=issue, daily=daily[REFERENCE])}
        for variant, settings in module.CHOSEN.items():
            urls[variant] = track(period, variant, {"model": name, **settings, **getattr(module, "RUN_INFO", {})},
                  scores.loc[variant], n_folds, PERIODS[period], issue=issue,
                  daily=daily[variant], reference_daily=daily[REFERENCE],
                  tables={"breakdowns": breakdown[breakdown["model"] == variant].reset_index(drop=True)},
                  files=tracking_files(out_dir, period))
        if baseline is not None:
            variant = f"baseline_{BASELINE_VARIANT}"
            baseline_module = load(BASELINE)
            urls[variant] = track(period, variant, {"model": BASELINE,
                                  **baseline_module.CHOSEN[BASELINE_VARIANT],
                                  **getattr(baseline_module, "RUN_INFO", {})},
                                  scores.loc[variant], n_folds, PERIODS[period], issue=issue,
                                  daily=daily[variant], reference_daily=daily[REFERENCE],
                                  tables={"breakdowns": breakdown[breakdown["model"] == variant].reset_index(drop=True)})
        (out_dir / f"{period}_wandb_urls.json").write_text(json.dumps(urls, indent=2), encoding="utf-8")
    return scores


def figures(name: str, out_dir: Path, period: str, fig_dir: Path) -> list[Path]:
    """Pinball loss per hour, coverage per month and, if defined in model.py, the coefficients."""
    module = load(name)
    order = [REFERENCE, *module.CHOSEN, f"baseline_{BASELINE_VARIANT}"]
    paths = backtest_figures(out_dir, period, fig_dir, order)
    shown = getattr(module, "COEFFICIENTS_SHOWN", None)
    if shown and (out_dir / f"{period}_coefficients.csv").exists():
        paths.append(coefficients_figure(out_dir, period, fig_dir, module.COEFFICIENTS_OF, shown))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="folder name in src/energy_price/models/, e.g. baseline_model")
    parser.add_argument("step", choices=["tune", "test", "holdout"])
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds, -1 uses all cores")
    parser.add_argument("--track", action="store_true", help="log the runs to W&B")
    parser.add_argument("--unlock-holdout", action="store_true", help="release the holdout period once")
    parser.add_argument("--out", type=Path, default=None, help="default reports/<model>/")
    args = parser.parse_args(argv)
    out = args.out or REPORTS_DIR / args.model

    table = build_table(load_inputs())
    pd.set_option("display.width", 160)
    if args.step == "tune":
        scores = tune(args.model, table, args.jobs, out, args.track)
    else:
        scores = evaluate_period(args.model, table, args.step, args.jobs, out, args.unlock_holdout, args.track)
        for path in figures(args.model, out, args.step, FIGURES_DIR / args.model):
            print(f"figure {path.relative_to(ROOT)}")
    print(scores.round(3).to_string())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
