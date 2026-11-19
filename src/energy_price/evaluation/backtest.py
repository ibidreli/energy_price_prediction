"""Walk-forward backtest shared by all models: every delivery day is forecast as at D-1, 11:00.

The design (periods, refit frequency, windows) lives in ``evaluation.design``. For delivery day D a
model sees only prices and system balance up to the end of D-2 (``known_at_issue``). Inside a run a
model is refitted every ``refit_every`` days (default 1, daily): the fit for the first day of a fold
also serves the following days of that fold, which then use older data, never newer. A training
window keeps either all rows since 2026-01-01 (``expanding``) or the last w days D-1-w to D-2.

Development runs never see the test period: ``development_inputs`` cuts the raw data before
``build_table`` at the first test day, so neither features nor models can read it. The test period
(2026-09-01 to 2026-10-31) is a "Testperiode mit Vorbehalt": no price or forecast of it was ever
scored, but the EDA looked at its system balance and weather.

Every model runs through the same functions, so all models are trained, compared and scored alike:

- ``run_model`` fits any ``QuantileModel`` fold by fold and returns its forecasts and fit times,
- ``run_reference`` gives the historical reference every model has to beat,
- ``window_mean`` averages the forecasts of one setting over its training windows,
- ``common_rows`` keeps only quarter hours that every model forecast completely,
- ``evaluate`` and ``compare`` score all models on these identical rows,
- ``write_outputs`` and ``track`` store the results in ``reports/<model>/`` and in W&B.

``evaluation.experiment`` runs these steps for any model folder (``make select MODEL=<name>``).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from energy_price.availability import known_at_issue
from energy_price.evaluation.design import DESIGN, WINDOW_MEAN, Design
from energy_price.evaluation.scores import COLUMNS, block_bootstrap, daily_pinball, score, skill_score
from energy_price.features.dataset import TARGET, TZ, Inputs, quarter_hours
from energy_price.models.shared.base import QuantileModel
from energy_price.models.shared.reference import WINDOW_DAYS as REFERENCE_WINDOW_DAYS
from energy_price.models.shared.reference import reference_forecast
from energy_price.paths import ROOT

REFERENCE = "reference"
BASELINE = "baseline_B2"
KEYS = ["delivery_day", "timestamp_utc"]
QUANTILE_COLUMNS = list(COLUMNS.values())


def development_inputs(inputs: Inputs, design: Design = DESIGN) -> Inputs:
    """Raw inputs without anything from the test period on.

    Prices and system balance are cut at the local start of the first test day (``timestamp_local``),
    weather forecasts by their ``delivery_day``. ``build_table`` on the result cannot see the test period.
    """
    start = design.test_start
    local_start = start.tz_localize(TZ)

    def before(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["timestamp_local"] < local_start].reset_index(drop=True)

    weather = inputs.weather[inputs.weather["delivery_day"] < start].reset_index(drop=True)
    return dataclasses.replace(inputs, prices=before(inputs.prices), balance=before(inputs.balance), weather=weather)


def folds(days: list[pd.Timestamp], refit_every: int = DESIGN.refit_every) -> list[list[pd.Timestamp]]:
    """Consecutive blocks of ``refit_every`` delivery days; the first day of a block is the fit day."""
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    return [days[i:i + refit_every] for i in range(0, len(days), refit_every)]


def training_rows(table: pd.DataFrame, fit_day: pd.Timestamp, window_days: int | None) -> pd.DataFrame:
    """Rows with a price known at the issue time for ``fit_day``: up to D-2, optionally the last ``window_days``.

    With a window of w days the rows are those of the delivery days D-1-w to D-2.
    """
    known = known_at_issue(table, fit_day.date())
    known = known[known[TARGET].notna() & (known["delivery_day"] >= DESIGN.first_train_day)]
    if window_days is not None:
        known = known[known["delivery_day"] >= fit_day - pd.Timedelta(days=window_days + 1)]
    return known


def _fit_and_predict(table: pd.DataFrame, block: list[pd.Timestamp], make_model: Callable[[], QuantileModel],
                     window_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    started = time.perf_counter()
    train = training_rows(table, block[0], window_days)
    model = make_model().fit(train)
    target = table[table["delivery_day"].isin(block)]
    forecast = pd.concat([target, model.predict(target)], axis=1)
    n_train = getattr(model, "n_train_", len(train))
    runtime = {"fit_day": block[0], "n_days": len(block), "n_train": n_train,
               "seconds": time.perf_counter() - started}
    coef = None
    if hasattr(model, "coefficients"):
        coef = model.coefficients().assign(fit_day=block[0], n_train=n_train)
    return forecast, coef, runtime


def run_model(table: pd.DataFrame, days: list[pd.Timestamp], make_model: Callable[[], QuantileModel], name: str,
              window_days: int | None = None, refit_every: int = DESIGN.refit_every,
              jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Forecasts of a model for ``days``, refitted on every fold. Folds run in parallel.

    ``make_model`` returns a new, unfitted model (e.g. ``functools.partial(MyModel, alpha=0.1)``).
    ``window_days`` limits the training rows to the last days before D-2 (None: all since 2026-01-01).
    Returns the forecasts, the coefficients of every fit (None if the model has no ``coefficients()``)
    and the runtime per fold (``fit_day``, ``n_days``, ``n_train``, ``seconds`` for fit and predict).
    """
    # HGB uses OpenMP internally. One native thread per worker avoids nested parallelism
    # when the fits already run in separate processes.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs)(
            delayed(_fit_and_predict)(table, block, make_model, window_days) for block in folds(days, refit_every)
        )
    forecasts = pd.concat([r[0] for r in results], ignore_index=True).assign(model=name)
    coefs = [r[1] for r in results if r[1] is not None]
    coefficients = pd.concat(coefs).rename_axis("feature").reset_index() if coefs else None
    runtime = pd.DataFrame([r[2] for r in results]).assign(model=name)
    return forecasts, coefficients, runtime


def run_reference(table: pd.DataFrame, days: list[pd.Timestamp],
                  window_days: int = REFERENCE_WINDOW_DAYS) -> pd.DataFrame:
    """Forecasts of the historical reference for ``days`` (``energy_price.models.shared.reference``)."""
    prices = table[table[TARGET].notna()]
    frames = []
    for day in days:
        target = table[table["delivery_day"] == day]
        frames.append(pd.concat([target, reference_forecast(prices, target, day.date(), window_days)], axis=1))
    return pd.concat(frames, ignore_index=True).assign(model=REFERENCE)


def window_mean(frames: list[pd.DataFrame], name: str = WINDOW_MEAN) -> pd.DataFrame:
    """Per-row mean of P10, P50 and P90 over the forecasts of one setting with different windows.

    All frames must hold the same ``(delivery_day, timestamp_utc)`` rows, each once. A row where any
    window has no forecast stays NaN. The mean of sorted quantiles is sorted, which is checked.
    ``model`` becomes ``name``. Other columns are kept if they are equal in all frames (the table), else dropped.
    """
    if not frames:
        raise ValueError("no forecasts to average")
    indexed = []
    for frame in frames:
        if frame.duplicated(KEYS).any():
            raise ValueError("a forecast holds a quarter hour twice")
        indexed.append(frame.set_index(KEYS))
    base = indexed[0]
    for other in indexed[1:]:
        if len(other) != len(base) or not other.index.sort_values().equals(base.index.sort_values()):
            raise ValueError("the windows forecast different quarter hours")
    aligned = [other.reindex(base.index) for other in indexed]
    out = base.copy()
    for column in [c for c in base.columns if c not in QUANTILE_COLUMNS and c != "model"]:
        if not all(column in other.columns and other[column].equals(base[column]) for other in aligned[1:]):
            out = out.drop(columns=column)
    values = np.stack([other[QUANTILE_COLUMNS].to_numpy(dtype=float) for other in aligned])
    out[QUANTILE_COLUMNS] = values.mean(axis=0)  # NaN as soon as one window has none
    q = out[QUANTILE_COLUMNS].dropna()
    if ((q["q10"] > q["q50"]) | (q["q50"] > q["q90"])).any():
        raise AssertionError("the mean over the windows has crossing quantiles")
    return out.reset_index().assign(model=name)


def common_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Rows of the delivery days on which every model has a complete forecast of every quarter hour.

    A day counts only if every model has exactly the quarter hours of that local day (92, 96 or 100,
    from ``quarter_hours``), each exactly once, all with P10, P50, P90 and the price. Otherwise the day
    drops out for all models alike, e.g. a day with missing weather (2026-06-24) or a missing price.
    The result holds the same ``(delivery_day, timestamp_utc)`` for every model; a ValueError says
    otherwise.
    """
    if forecasts.empty:
        return forecasts
    good = _complete_days(forecasts)
    out = forecasts[forecasts["delivery_day"].isin(good.index[good.all(axis=1)])]

    keys = {m: set(map(tuple, g[KEYS].to_numpy())) for m, g in out.groupby("model")}
    if len({frozenset(k) for k in keys.values()}) > 1:
        raise ValueError("models are scored on different quarter hours")
    return out


def missing_days(forecasts: pd.DataFrame, days: list[pd.Timestamp] | None = None) -> dict[pd.Timestamp, list[str]]:
    """Delivery days that ``common_rows`` drops, each with the models that lack a complete forecast.

    A day of ``days`` without any row is missing for every model.
    """
    models = sorted(forecasts["model"].unique())
    good = _complete_days(forecasts) if not forecasts.empty else pd.DataFrame(columns=models, dtype=bool)
    if days is not None:
        good = good.reindex(pd.DatetimeIndex(days).union(good.index), fill_value=False)
    return {pd.Timestamp(day): [m for m in models if not row[m]]
            for day, row in good.iterrows() if not row.all()}


def _complete_days(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Per delivery day and model: True if the model has every quarter hour of the day exactly once, complete."""
    models = forecasts["model"].unique()
    days = sorted(forecasts["delivery_day"].unique())
    expected = quarter_hours([pd.Timestamp(day) for day in days])[KEYS]
    expected = expected.astype({key: forecasts[key].dtype for key in KEYS})
    n_expected = expected.groupby("delivery_day").size()

    rows = forecasts[[*KEYS, "model"]].copy()
    rows["complete"] = forecasts[QUANTILE_COLUMNS + [TARGET]].notna().all(axis=1)
    rows["expected"] = pd.MultiIndex.from_frame(rows[KEYS]).isin(pd.MultiIndex.from_frame(expected))
    grouped = rows.groupby(["delivery_day", "model"])
    per_model = pd.DataFrame({
        "n": grouped.size(),
        "n_unique": grouped["timestamp_utc"].nunique(),
        "ok": grouped["complete"].all() & grouped["expected"].all(),
    })
    n = per_model.index.get_level_values("delivery_day").map(n_expected).to_numpy()
    per_model["good"] = per_model["ok"] & (per_model["n"] == n) & (per_model["n_unique"] == n)
    return per_model["good"].unstack("model").reindex(columns=models).fillna(False).astype(bool)


def evaluate(forecasts: pd.DataFrame, reference: str = REFERENCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per model and daily pinball loss per model, on the common rows.

    ``mean_daily_pinball`` weights every delivery day equally; it is the selection criterion, and
    ``skill_vs_reference`` (1 - model / reference) is computed from it as well.
    """
    rows = common_rows(forecasts)
    if rows.empty:
        raise ValueError("no delivery day with a complete forecast of every model")
    scores = pd.DataFrame({m: score(g) for m, g in rows.groupby("model")}).T
    scores["n_days"] = rows.groupby("model")["delivery_day"].nunique()
    daily = pd.DataFrame({m: daily_pinball(g) for m, g in rows.groupby("model")})
    scores["mean_daily_pinball"] = daily.mean()
    if reference in scores.index:
        ref = scores.loc[reference, "mean_daily_pinball"]
        scores["skill_vs_reference"] = [skill_score(v, ref) for v in scores["mean_daily_pinball"]]
    if reference in daily.columns:
        better = daily.drop(columns=reference).lt(daily[reference], axis=0).mean()
        scores["share_days_better"] = better.reindex(scores.index)
    return scores, daily


def compare(forecasts: pd.DataFrame, reference: str = REFERENCE, baseline: str = BASELINE) -> pd.DataFrame:
    """Every model against the reference and against the baseline B2, on identical quarter hours.

    One row per model and comparison model (``compared_with``): scores of the model, the mean daily
    pinball loss of both, the skill 1 - model / compared (on the mean daily pinball, every day weighs
    the same) and the mean daily difference model minus compared with its 95 % block bootstrap interval
    (7-day blocks; NaN with fewer than 7 days). Below 0 the model is better. ``period_start`` and
    ``period_end`` are the first and last scored delivery day.
    """
    scores, daily = evaluate(forecasts, reference)
    against = [m for m in (reference, baseline) if m in daily.columns]
    if not against:
        raise ValueError(f"neither {reference} nor {baseline} is among the forecasts")
    start, end = (str(d.date()) for d in (daily.index.min(), daily.index.max()))
    records = []
    for model in daily.columns:
        for other in against:
            if other == model:
                continue
            diff = (daily[model] - daily[other]).to_numpy()
            low, high = block_bootstrap(diff) if len(diff) >= 7 else (np.nan, np.nan)
            records.append({
                "model": model, "compared_with": other, "period_start": start, "period_end": end,
                "n_days": len(diff), "pinball": scores.loc[model, "pinball"],
                "mean_daily_pinball": daily[model].mean(), "compared_mean_daily_pinball": daily[other].mean(),
                "skill": skill_score(daily[model].mean(), daily[other].mean()),
                "mean_difference": diff.mean(), "low": low, "high": high,
                "share_days_better": float((diff < 0).mean()),
                "mae_q50": scores.loc[model, "mae_q50"], "coverage_80": scores.loc[model, "coverage_80"],
                "width_80": scores.loc[model, "width_80"],
            })
    return pd.DataFrame(records)


def run_info(extra: dict | None = None) -> dict:
    """Code, data and package versions of a run, for ``reports/<model>/<step>_run_info.json``."""
    import joblib
    import sklearn

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        from energy_price.evaluation.tracking import data_version
        version = data_version()
    except ImportError:  # tracking extra not installed
        version = None
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "data_version": version,
        "python": platform.python_version(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                     "joblib": joblib.__version__},
        **(extra or {}),
    }


def write_outputs(out_dir: Path, step: str, forecasts: pd.DataFrame, scores: pd.DataFrame, daily: pd.DataFrame,
                  info: dict, coefficients: pd.DataFrame | None = None, runtime: pd.DataFrame | None = None) -> None:
    """Forecasts, scores, daily pinball loss, coefficients, runtime and run info of one step in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / f"{step}_forecasts.parquet", index=False)
    scores.to_csv(out_dir / f"{step}_scores.csv", index=not isinstance(scores.index, pd.RangeIndex))
    daily.to_csv(out_dir / f"{step}_daily_pinball.csv")
    if coefficients is not None:
        coefficients.to_csv(out_dir / f"{step}_coefficients.csv", index=False)
    if runtime is not None:
        runtime.to_csv(out_dir / f"{step}_runtime.csv", index=False)
    (out_dir / f"{step}_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def log_daily(run, split: str, daily: pd.Series, reference_daily: pd.Series | None = None) -> None:
    """Pinball loss per delivery day as W&B history, one step per day (step 0 is the first day).

    The workspace draws it as a line chart per run. With ``reference_daily`` the daily difference
    model minus reference is logged as well: below 0 the model was better on that day. The daily
    values get no summary, so the summary keeps only the scores of the whole period.
    """
    run.define_metric(f"{split}/daily_*", summary="none")
    diff = None if reference_daily is None else daily - reference_daily.reindex(daily.index)
    for step, day in enumerate(daily.index):
        row = {f"{split}/daily_pinball": float(daily[day])}
        if diff is not None:
            row[f"{split}/daily_pinball_vs_reference"] = float(diff[day])
        run.log(row, step=step)


def track(split: str, name: str, config: dict, metrics: pd.Series, n_folds: int, period: tuple[str, str],
          issue: int | None = None, daily: pd.Series | None = None,
          reference_daily: pd.Series | None = None, tables: dict[str, pd.DataFrame] | None = None,
          files: list[Path] | None = None, refit_every: int = DESIGN.refit_every,
          design: Design = DESIGN) -> str | None:
    """Log one model as a W&B run (job type eval, group ``issue-<issue>``, else from the branch name).

    The config holds the design, the training window, ``refit_every`` and the scored period next to the
    settings. ``metrics`` go to the summary; ``daily`` (and ``reference_daily``) to the history, see ``log_daily``.
    """
    from energy_price.evaluation.tracking import start_run

    window = config.get("window", "expanding")
    scheme = {"expanding": "expanding_window"}.get(window, "rolling_window" if window.startswith("rolling")
                                                   else window)
    cv = {"scheme": scheme, "n_folds": n_folds, "period_start": period[0], "period_end": period[1],
          "refit_every": refit_every}
    config = {**config, "window": window, "refit_every": refit_every, "period": list(period),
              "design": design.as_config(), "cv": cv, "split": split}
    with start_run("eval", issue=issue, name=name, config=config) as run:
        if daily is not None:
            log_daily(run, split, daily, reference_daily)
        for key, value in metrics.items():
            if pd.notna(value):
                run.summary[f"{split}/{key}"] = float(value)
        if tables or files:
            import wandb

            for key, frame in (tables or {}).items():
                table = wandb.Table(dataframe=frame)
                run.log({f"{split}/{key}": table})
                if key == "feature_importance":
                    run.log({f"{split}/feature_importance_plot": wandb.plot.bar(
                        table, "group", "importance_mean", title="Grouped permutation importance (pinball increase)")})
            if files:
                artifact = wandb.Artifact(f"{config.get('model', 'model')}-{split}-{run.id}", type="evaluation")
                for path in files:
                    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else f"outputs/{path.name}"
                    artifact.add_file(str(path), name=name)
                run.log_artifact(artifact)
        return run.url


def track_reference(split: str, scores: pd.DataFrame, n_folds: int, period: tuple[str, str],
                    issue: int | None = None, daily: pd.Series | None = None) -> str | None:
    """Log the reference of a step as a W&B run, next to the models of the same issue."""
    return track(split, REFERENCE, {"model": REFERENCE, "window": f"rolling_{REFERENCE_WINDOW_DAYS}d"},
                 scores.loc[REFERENCE], n_folds, period, issue, daily, refit_every=1)
