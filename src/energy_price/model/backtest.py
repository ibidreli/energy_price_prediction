"""Walk-forward backtest of the baseline: every delivery day is forecast as at D-1, 11:00.

For delivery day D a model sees only prices up to the end of D-2 (``known_at_issue``). Models are
refitted once a week: the fit for the first day of a week also serves the following six days, so
those days use even older data, never newer. Settings are chosen on the tuning period and then
applied unchanged to the test period. The holdout (September and October) stays locked until it
is explicitly released with ``--unlock-holdout``; it was not part of the EDA.

Usage:
``python -m energy_price.model.backtest tune``    grid over feature set, window and alpha on April and May
``python -m energy_price.model.backtest test``    reference, B1 and B2 on June to August, with figures
``python -m energy_price.model.backtest holdout --unlock-holdout``

``--track`` logs every model as a W&B run in the group ``issue-6`` (docs/experiment_tracking.md).
Results are written to ``reports/baseline/`` (not versioned).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from energy_price.availability import known_at_issue
from energy_price.model.dataset import FIRST_DAY, ROOT, TARGET, build_table, load_inputs
from energy_price.model.models import LinearQuantileModel, reference_forecast
from energy_price.model.scores import COLUMNS, daily_pinball, score, skill_score

PERIODS = {
    "tune": ("2026-04-01", "2026-05-31"),
    "test": ("2026-06-01", "2026-08-31"),
    "holdout": ("2026-09-01", "2026-10-31"),
}
REFIT_DAYS = 7
REFERENCE_WINDOW_DAYS = 56
WINDOWS = {"expanding": None, "rolling_56d": 56}
ALPHAS = (0.0, 0.01, 0.1)
ISSUE = 6
OUT_DIR = ROOT / "reports" / "baseline"


@dataclass(frozen=True)
class Config:
    feature_set: str
    window: str
    alpha: float

    @property
    def name(self) -> str:
        return self.feature_set


# Chosen on the tuning period (April and May), see docs/baseline.md.
CHOSEN = {
    "B1": Config("B1", "expanding", 0.01),
    "B2": Config("B2", "expanding", 0.01),
}


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


def _fit_and_predict(table: pd.DataFrame, block: list[pd.Timestamp], config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = training_rows(table, block[0], WINDOWS[config.window])
    model = LinearQuantileModel(config.feature_set, config.alpha).fit(train)
    target = table[table["delivery_day"].isin(block)]
    forecast = pd.concat([target, model.predict(target)], axis=1)
    coef = model.coefficients().assign(fit_day=block[0], n_train=model.n_train_)
    return forecast, coef


def run_linear(table: pd.DataFrame, days: list[pd.Timestamp], config: Config, jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasts for ``days`` and the coefficients of every weekly fit. Folds run in parallel."""
    results = Parallel(n_jobs=jobs)(delayed(_fit_and_predict)(table, block, config) for block in folds(days))
    forecasts = pd.concat([r[0] for r in results], ignore_index=True).assign(model=config.name)
    coefficients = pd.concat([r[1] for r in results]).rename_axis("feature").reset_index()
    return forecasts, coefficients


def run_reference(table: pd.DataFrame, days: list[pd.Timestamp]) -> pd.DataFrame:
    prices = table[table[TARGET].notna()]
    frames = []
    for day in days:
        target = table[table["delivery_day"] == day]
        frames.append(pd.concat([target, reference_forecast(prices, target, day.date(), REFERENCE_WINDOW_DAYS)], axis=1))
    return pd.concat(frames, ignore_index=True).assign(model="reference")


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


def evaluate(forecasts: pd.DataFrame, reference: str = "reference") -> tuple[pd.DataFrame, pd.DataFrame]:
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
    """Code, data and package versions of a run, for ``reports/baseline/run_info.json``."""
    import joblib
    import sklearn

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        from energy_price.tracking import data_version
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
        "deterministic": "QuantileRegressor with the HiGHS solver has no random component; no seed needed",
        **(extra or {}),
    }


def track(split: str, name: str, config: dict, metrics: pd.Series, n_folds: int, period: tuple[str, str]) -> None:
    """Log one model as a W&B run (job type eval, group issue-6)."""
    from energy_price.tracking import start_run

    scheme = "rolling_window" if config.get("window", "").startswith("rolling") else "expanding_window"
    cv = {"scheme": scheme, "n_folds": n_folds, "test_start": period[0], "test_end": period[1],
          "refit_days": REFIT_DAYS}
    with start_run("eval", issue=ISSUE, name=name, config={**config, "cv": cv, "split": split}) as run:
        for key, value in metrics.items():
            if pd.notna(value):
                run.summary[f"{split}/{key}"] = float(value)


def tune(table: pd.DataFrame, jobs: int, out_dir: Path) -> pd.DataFrame:
    """Every combination of feature set, window and alpha on the tuning period, against the reference."""
    days = period_days("tune")
    configs = [Config(fs, w, a) for fs in ("B1", "B2") for w in WINDOWS for a in ALPHAS]
    frames = [run_reference(table, days)]
    for config in configs:
        forecast, _ = run_linear(table, days, config, jobs)
        frames.append(forecast.assign(model=f"{config.feature_set}|{config.window}|{config.alpha}"))
    scores, _ = evaluate(pd.concat(frames, ignore_index=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "tune_scores.csv")
    return scores


def evaluate_period(table: pd.DataFrame, period: str, jobs: int, out_dir: Path, unlock_holdout: bool = False,
                    do_track: bool = False) -> pd.DataFrame:
    """Reference, B1 and B2 with the chosen settings on ``period``; writes forecasts, scores and coefficients."""
    days = period_days(period, unlock_holdout)
    frames, coefficients = [run_reference(table, days)], []
    for config in CHOSEN.values():
        forecast, coef = run_linear(table, days, config, jobs)
        frames.append(forecast)
        coefficients.append(coef.assign(model=config.name))
    forecasts = pd.concat(frames, ignore_index=True)
    scores, daily = evaluate(forecasts)

    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_dir / f"{period}_forecasts.parquet", index=False)
    scores.to_csv(out_dir / f"{period}_scores.csv")
    daily.to_csv(out_dir / f"{period}_daily_pinball.csv")
    pd.concat(coefficients, ignore_index=True).to_csv(out_dir / f"{period}_coefficients.csv", index=False)
    info = run_info({"period": period, "days": [str(d.date()) for d in (days[0], days[-1])],
                     "chosen": {k: asdict(v) for k, v in CHOSEN.items()},
                     "reference_window_days": REFERENCE_WINDOW_DAYS, "refit_days": REFIT_DAYS})
    (out_dir / f"{period}_run_info.json").write_text(json.dumps(info, indent=2))

    if do_track:
        n_folds = len(folds(days))
        track(period, "reference", {"model": "reference", "window": f"rolling_{REFERENCE_WINDOW_DAYS}d"},
              scores.loc["reference"], n_folds, PERIODS[period])
        for config in CHOSEN.values():
            track(period, config.name, {"model": "linear_quantile", **asdict(config)},
                  scores.loc[config.name], n_folds, PERIODS[period])
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["tune", "test", "holdout"])
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds, -1 uses all cores")
    parser.add_argument("--track", action="store_true", help="log the runs to W&B")
    parser.add_argument("--unlock-holdout", action="store_true", help="release the holdout period once")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    table = build_table(load_inputs())
    pd.set_option("display.width", 160)
    if args.step == "tune":
        scores = tune(table, args.jobs, args.out)
    else:
        scores = evaluate_period(table, args.step, args.jobs, args.out, args.unlock_holdout, args.track)
    print(scores.round(3).to_string())
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
