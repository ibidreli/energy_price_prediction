"""Walk-forward backtest shared by all models: every delivery day is forecast as at D-1, 11:00.

For delivery day D a model sees only prices up to the end of D-2 (``known_at_issue``). Models are
refitted once a week: the fit for the first day of a week also serves the following six days, so
those days use even older data, never newer. Settings are chosen on the tuning period and then
applied unchanged to the test period. The holdout (September and October) stays locked until it
is explicitly released with ``unlock_holdout``; it was not part of the EDA.

Every model runs through the same functions, so all models are trained, compared and scored alike:

- ``run_model`` fits any ``QuantileModel`` fold by fold and returns its forecasts,
- ``run_reference`` gives the historical reference every model has to beat,
- ``evaluate`` scores all models on the same quarter hours,
- ``write_outputs`` and ``track`` store the results in ``reports/<model>/`` and in W&B.

``evaluation.experiment`` runs these steps for any model folder (``make backtest MODEL=<name>``).
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from energy_price.availability import known_at_issue
from energy_price.evaluation.scores import COLUMNS, daily_pinball, score, skill_score
from energy_price.features.dataset import FIRST_DAY, TARGET
from energy_price.models.shared.base import QuantileModel
from energy_price.models.shared.reference import WINDOW_DAYS as REFERENCE_WINDOW_DAYS
from energy_price.models.shared.reference import reference_forecast
from energy_price.paths import ROOT

PERIODS = {
    "tune": ("2026-04-01", "2026-05-31"),
    "test": ("2026-06-01", "2026-08-31"),
    "holdout": ("2026-09-01", "2026-10-31"),
}
REFIT_DAYS = 7
REFERENCE = "reference"


class HoldoutLockedError(RuntimeError):
    """The holdout period was requested without ``unlock_holdout``."""


def period_days(period: str, unlock_holdout: bool = False) -> list[pd.Timestamp]:
    if period == "holdout" and not unlock_holdout:
        raise HoldoutLockedError("the holdout is locked; release it once with --unlock-holdout")
    start, end = PERIODS[period]
    return list(pd.date_range(start, end, freq="D"))


def folds(days: list[pd.Timestamp], refit_days: int = REFIT_DAYS) -> list[list[pd.Timestamp]]:
    """Consecutive blocks of ``refit_days`` delivery days; the first day of a block is the fit day."""
    return [days[i:i + refit_days] for i in range(0, len(days), refit_days)]


def training_rows(table: pd.DataFrame, fit_day: pd.Timestamp, window_days: int | None) -> pd.DataFrame:
    """Rows with a price known at the issue time for ``fit_day``: up to D-2, optionally the last ``window_days``."""
    known = known_at_issue(table, fit_day.date())
    known = known[known[TARGET].notna() & (known["delivery_day"] >= FIRST_DAY)]
    if window_days is not None:
        known = known[known["delivery_day"] >= fit_day - pd.Timedelta(days=window_days + 1)]
    return known


def _fit_and_predict(table: pd.DataFrame, block: list[pd.Timestamp], make_model: Callable[[], QuantileModel],
                     window_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train = training_rows(table, block[0], window_days)
    model = make_model().fit(train)
    target = table[table["delivery_day"].isin(block)]
    forecast = pd.concat([target, model.predict(target)], axis=1)
    coef = None
    if hasattr(model, "coefficients"):
        coef = model.coefficients().assign(fit_day=block[0], n_train=getattr(model, "n_train_", len(train)))
    return forecast, coef


def run_model(table: pd.DataFrame, days: list[pd.Timestamp], make_model: Callable[[], QuantileModel], name: str,
              window_days: int | None = None, jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Forecasts of a model for ``days``, refitted on every fold. Folds run in parallel.

    ``make_model`` returns a new, unfitted model (e.g. ``functools.partial(MyModel, alpha=0.1)``).
    ``window_days`` limits the training rows to the last days before D-2 (None: all since 2026-01-01).
    If the model has a ``coefficients()`` method, its coefficients of every fit are returned as well.
    """
    # HGB uses OpenMP internally. One native thread per worker avoids nested parallelism
    # when the weekly fits already run in separate processes.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs)(
            delayed(_fit_and_predict)(table, block, make_model, window_days) for block in folds(days)
        )
    forecasts = pd.concat([r[0] for r in results], ignore_index=True).assign(model=name)
    coefs = [r[1] for r in results if r[1] is not None]
    coefficients = pd.concat(coefs).rename_axis("feature").reset_index() if coefs else None
    return forecasts, coefficients


def run_reference(table: pd.DataFrame, days: list[pd.Timestamp],
                  window_days: int = REFERENCE_WINDOW_DAYS) -> pd.DataFrame:
    """Forecasts of the historical reference for ``days`` (``energy_price.models.shared.reference``)."""
    prices = table[table[TARGET].notna()]
    frames = []
    for day in days:
        target = table[table["delivery_day"] == day]
        frames.append(pd.concat([target, reference_forecast(prices, target, day.date(), window_days)], axis=1))
    return pd.concat(frames, ignore_index=True).assign(model=REFERENCE)


def common_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Rows on delivery days where every model has a complete forecast and every price is known.

    All models are scored on the same quarter hours. A day with missing weather (2026-06-24) or a
    missing price drops out for all models alike.
    """
    q = list(COLUMNS.values())
    ok = forecasts[q + [TARGET]].notna().all(axis=1)
    per_day_model = ok.groupby([forecasts["delivery_day"], forecasts["model"]]).all().unstack()
    good_days = per_day_model.index[per_day_model.all(axis=1)]
    return forecasts[forecasts["delivery_day"].isin(good_days)]


def evaluate(forecasts: pd.DataFrame, reference: str = REFERENCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores per model and daily pinball loss per model, on the common rows."""
    rows = common_rows(forecasts)
    scores = pd.DataFrame({m: score(g) for m, g in rows.groupby("model")}).T
    scores["n_days"] = rows.groupby("model")["delivery_day"].nunique()
    if reference in scores.index:
        ref = scores.loc[reference, "pinball"]
        scores["skill_vs_reference"] = [skill_score(v, ref) for v in scores["pinball"]]
    daily = pd.DataFrame({m: daily_pinball(g) for m, g in rows.groupby("model")})
    if reference in daily.columns:
        better = daily.drop(columns=reference).lt(daily[reference], axis=0).mean()
        scores["share_days_better"] = better.reindex(scores.index)
    return scores, daily


def run_info(extra: dict | None = None) -> dict:
    """Code, data and package versions of a run, for ``reports/<model>/<period>_run_info.json``."""
    import joblib
    import sklearn

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
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


def write_outputs(out_dir: Path, period: str, forecasts: pd.DataFrame, scores: pd.DataFrame, daily: pd.DataFrame,
                  info: dict, coefficients: pd.DataFrame | None = None) -> None:
    """Forecasts, scores, daily pinball loss, coefficients and run info of one period in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / f"{period}_forecasts.parquet", index=False)
    scores.to_csv(out_dir / f"{period}_scores.csv")
    daily.to_csv(out_dir / f"{period}_daily_pinball.csv")
    if coefficients is not None:
        coefficients.to_csv(out_dir / f"{period}_coefficients.csv", index=False)
    (out_dir / f"{period}_run_info.json").write_text(json.dumps(info, indent=2))


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
          files: list[Path] | None = None) -> str | None:
    """Log one model as a W&B run (job type eval, group ``issue-<issue>``, else from the branch name).

    ``metrics`` go to the summary; ``daily`` (and ``reference_daily``) to the history, see ``log_daily``.
    """
    from energy_price.evaluation.tracking import start_run

    scheme = "rolling_window" if config.get("window", "").startswith("rolling") else "expanding_window"
    cv = {"scheme": scheme, "n_folds": n_folds, "test_start": period[0], "test_end": period[1],
          "refit_days": REFIT_DAYS}
    with start_run("eval", issue=issue, name=name, config={**config, "cv": cv, "split": split}) as run:
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


def track_reference(split: str, scores: pd.DataFrame, n_folds: int, issue: int | None = None,
                    daily: pd.Series | None = None) -> None:
    """Log the reference of a period as a W&B run, next to the models of the same issue."""
    track(split, REFERENCE, {"model": REFERENCE, "window": f"rolling_{REFERENCE_WINDOW_DAYS}d"},
          scores.loc[REFERENCE], n_folds, PERIODS[split], issue, daily)
