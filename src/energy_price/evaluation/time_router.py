"""Issue #25: shared pool, April/May design, weekly routing and final calibration."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist
import subprocess

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import PERIODS, folds, period_days, run_info, track, write_outputs
from energy_price.evaluation.scores import daily_pinball, pinball, score
from energy_price.features.dataset import TARGET, build_table, load_inputs
from energy_price.models.shared.calibration import ResidualCalibrator
from energy_price.models.shared.forecast_pool import Q, generate_pool, pool_fingerprint, wide_pool
from energy_price.models.shared.routing import (TimeRouterModel, block_ids, bootstrap_means,
                                               daily_expert_losses, difference_interval)
from energy_price.models.time_router import model as config
from energy_price.paths import REPORTS_DIR, ROOT

OUT = REPORTS_DIR / "time_router"
DOC = ROOT / "src/energy_price/models/time_router"
DESIGN = DOC / "design.json"
NOMINATION = DOC / "nomination.json"


def policy_fingerprint():
    files = [Path(__file__), ROOT / "src/energy_price/models/shared/routing.py",
             ROOT / "src/energy_price/models/shared/calibration.py", DOC / "model.py",
             ROOT / "src/energy_price/evaluation/router_holdout.py"]
    return sha256(b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in files)).hexdigest()


def json_fingerprint(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def pool_phase(table, phase, out, jobs):
    days = (list(pd.date_range("2026-01-30", "2026-03-31")) if phase == "warm" else period_days(phase))
    # Content-addressed fold cache below always validates the actual inputs. A
    # top-level parquet is an output, never a trusted input for a later run.
    frame = generate_pool(table, days, config.POOL, jobs)
    frame.to_parquet(out / f"{phase}_pool.parquet", index=False)
    return frame


def design_analysis(tune, out):
    """Design exclusively from April/May; no test table is accepted here."""
    if not tune.delivery_day.between("2026-04-01", "2026-05-31").all():
        raise ValueError("design analysis accepts only April/May")
    names = list(config.POOL)
    losses = daily_expert_losses(tune, names, [0, 24]).droplevel(1)
    default = min(names, key=lambda n: (losses[n].mean(), n))
    hourly = []
    for name in names:
        for q, tau in zip(Q, (.1, .5, .9)):
            values = pd.Series(pinball(tune[TARGET], tune[f"{name}:{q}"], tau), index=tune.index)
            hourly.extend({"model": name, "quantile": q, "hour": int(h), "pinball": float(v)}
                          for h, v in values.groupby(tune.hour).mean().items())
    pd.DataFrame(hourly).to_csv(out / "tune_hourly.csv", index=False)
    pd.DataFrame({n: tune[TARGET] - tune[f"{n}:q50"] for n in names}).corr().to_csv(out / "tune_error_correlation.csv")
    losses.corr().to_csv(out / "tune_loss_correlation.csv")
    # A 1% reduction is the predeclared planning effect. Required days are an
    # approximate 80%-power calculation from block-bootstrap long-run variance,
    # accounting for (experts-1)*blocks simultaneous challenger comparisons.
    power, supported = [], []
    for layout, edges in config.LAYOUTS.items():
        daily = daily_expert_losses(tune, names, edges)
        comparisons = max(1, (len(names) - 1) * (len(edges) - 1))
        required = []
        for group in range(len(edges) - 1):
            block = daily.xs(group, level=1)
            effect = .01 * block[default].mean()
            for name in names:
                if name == default:
                    continue
                diff = block[name] - block[default]
                draws = bootstrap_means(diff)
                z = NormalDist().inv_cdf(1 - .05 / (2 * comparisons)) + NormalDist().inv_cdf(.8)
                days = max(1, int(np.ceil(len(diff) * (z * draws.std(ddof=1) / effect) ** 2)))
                required.append(days)
                low, high = difference_interval(diff, comparisons)
                power.append({"layout": layout, "block": group, "expert": name, "comparisons": comparisons,
                              "n_days": len(diff), "planning_effect": effect, "required_days_80pct": days,
                              "difference": diff.mean(), "family_low": low, "family_high": high})
        # All declared challenger/block comparisons must support the planning
        # effect. If none do, keep one pooled block and report inadequate power.
        if max(required, default=0) <= tune.delivery_day.nunique():
            supported.append(layout)
    pd.DataFrame(power).to_csv(out / "tune_power.csv", index=False)
    cap = max([len(config.LAYOUTS[n]) - 1 for n in supported], default=1)
    layout = "domain" if cap >= 3 else "global"
    return {"issue": 25, "pool": config.POOL, "pool_fingerprint": pool_fingerprint(config.POOL),
            "policy_fingerprint": policy_fingerprint(),
            "default": default, "tune_scores": {n: float(losses[n].mean()) for n in names},
            "layout": layout, "edges": config.LAYOUTS[layout], "block_cap": cap,
            "power_supported_layouts": supported, "planning_effect_fraction": .01, "planning_power": .8,
            "power_requirement": "all challenger/block comparisons; fallback pooled block if unsupported",
            "robustness_edges": config.LAYOUTS["6h"], "weekday_weekend_split": False,
            "candidates": [s["strategy"] for s in config.GRID], "bootstrap_reps": 5000,
            "bootstrap_block_days": 7, "seed": 0, "selection_history": "expanding OOS since 2026-01-30",
            "min_history_days": 28, "multiplicity": "Bonferroni across (pool_size-1)*blocks at each refit",
            "weight_method": "simplex pinball LP, 56 calendar days, 50% shrinkage to default",
            "calibration": config.CALIBRATION, "pool_freeze_date": config.POOL_FREEZE_DATE,
            "holdout_status": "locked; final data and final pool freeze required",
            "test_nomination": "lowest calibrated pinball among candidates; no causal claim from test",
            "test_is_blind": False, "info": run_info()}


def require_committed(path):
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True)
    if result.returncode or result.stdout.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError(f"commit {relative} before evaluating the next phase")


def read_design(require_commit=True):
    if require_commit:
        require_committed(DESIGN)
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    if design["pool"] != config.POOL or design["pool_fingerprint"] != pool_fingerprint(config.POOL):
        raise ValueError("pool/settings/expert code changed: redo April/May design before testing")
    if design["policy_fingerprint"] != policy_fingerprint():
        raise ValueError("routing/calibration policy changed: redo April/May design before testing")
    return design


def routed_history(wide, design, through, robustness=False):
    """One forward pass: historical routing is never refitted retrospectively.

    Design/default were selected on April/May, so those rows are development
    data. June onward uses the frozen design and only then-available errors.
    Calibration uses the recorded uncalibrated output of each historical fit.
    """
    names = list(design["pool"])
    edges = design["robustness_edges"] if robustness else design["edges"]
    strategies = ["default", *design["candidates"]]
    histories, output, decisions = {s: [] for s in strategies}, [], []
    phases = [("2026-01-30", "2026-03-31"), PERIODS["tune"], PERIODS["test"]]
    if pd.Timestamp(through) >= pd.Timestamp(PERIODS["holdout"][0]):
        phases.append(PERIODS["holdout"])
    for start, end in phases:
        if pd.Timestamp(start) > pd.Timestamp(through):
            break
        for days in folds(list(pd.date_range(start, min(pd.Timestamp(end), pd.Timestamp(through))))):
            fit_day = days[0]
            target = wide[wide.delivery_day.isin(days)].copy()
            if target.empty:
                continue
            known = wide[wide.delivery_day <= fit_day - pd.Timedelta(days=2)]
            for strategy in strategies:
                train = known
                if strategy == "weights":
                    train = known[known.delivery_day >= fit_day - pd.Timedelta(days=57)]
                router = TimeRouterModel(names, design["default"], edges, strategy, fit_day,
                                         min_days=design["min_history_days"], reps=design["bootstrap_reps"], seed=design["seed"])
                router.fit(train)
                # Explicitly remove outcome columns from the prediction call.
                predicted = router.predict(target.drop(columns=[TARGET, "short"], errors="ignore"))
                raw = pd.concat([target[["timestamp_utc", "timestamp_local", "delivery_day", "hour", TARGET]
                                        + (["short"] if "short" in target else [])], predicted], axis=1)
                past = pd.concat(histories[strategy], ignore_index=True) if histories[strategy] else None
                histories[strategy].append(raw)
                output.append(raw.assign(model=strategy))
                # Warm-up has insufficient residual history. No fabricated
                # calibrated rows; all scored phases have >56 days warm-up.
                if past is not None and past.delivery_day.nunique() >= 28:
                    calibrated = ResidualCalibrator(**design["calibration"]).fit(past, fit_day).predict(raw)
                    output.append(calibrated.assign(model=f"calibrated_{strategy}"))
                decisions.extend({"strategy": strategy, "fit_day": str(fit_day.date()),
                                  "error_last_day": str(router.error_last_day_), **item}
                                 for item in router.diagnostics_)
    return pd.concat(output, ignore_index=True), pd.DataFrame(decisions)


def summarise(forecasts, design, period, out, do_track=False):
    rows = forecasts[forecasts.delivery_day.between(*PERIODS[period])].copy()
    expected = {"default", *design["candidates"], "calibrated_default",
                *(f"calibrated_{s}" for s in design["candidates"])}
    if set(rows.model) != expected:
        raise ValueError("missing candidate/calibrated comparator")
    keys = [set(g.timestamp_utc) for _, g in rows.groupby("model")]
    if any(k != keys[0] for k in keys):
        raise ValueError("candidates do not share identical quarter hours")
    scores = pd.DataFrame({n: score(g) for n, g in rows.groupby("model")}).T
    scores["n_days"] = rows.groupby("model").delivery_day.nunique()
    daily = pd.DataFrame({n: daily_pinball(g) for n, g in rows.groupby("model")})
    info = run_info({"design": design, "period": period, "test_is_blind": False,
                     "calibration_stage": "once, after routing", "holdout_executed": period == "holdout"})
    write_outputs(out, period, rows, scores, daily, info)
    pairs = []
    for candidate in design["candidates"]:
        for prefix in ("", "calibrated_"):
            name, comparator = prefix + candidate, prefix + "default"
            diff = daily[name] - daily[comparator]
            low, high = difference_interval(diff)
            pairs.append({"model": name, "comparator": comparator, "mean_difference": diff.mean(),
                          "ci95_low": low, "ci95_high": high, "n_days": len(diff),
                          "coverage_difference": scores.loc[name, "coverage_80"] - scores.loc[comparator, "coverage_80"]})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / f"{period}_comparisons.csv", index=False)
    breakdowns = []
    rows["month"] = rows.delivery_day.dt.month
    rows["direction"] = rows["short"].map({0.: "long", 1.: "short"}) if "short" in rows else None
    for dimension in ("block", "month", "direction"):
        for (name, value), group in rows.groupby(["model", dimension]):
            breakdowns.append({"model": name, "dimension": dimension, "value": str(value), **score(group)})
    breakdowns = pd.DataFrame(breakdowns)
    breakdowns.to_csv(out / f"{period}_breakdowns.csv", index=False)
    if do_track:
        urls = {}
        for name in scores.index:
            urls[name] = track(period, f"time_router/{name}", {"model": "time_router", "variant": name,
                              "window": "expanding", "design": design}, scores.loc[name],
                              len(folds(period_days(period, period == "holdout"))), PERIODS[period], issue=25,
                              daily=daily[name], reference_daily=daily["calibrated_default" if name.startswith("calibrated_") else "default"],
                              tables={"comparisons": pairs[pairs.model == name].reset_index(drop=True),
                                      "breakdowns": breakdowns[breakdowns.model == name].reset_index(drop=True)},
                              files=[DESIGN, out / f"{period}_comparisons.csv", out / f"{period}_scores.csv",
                                     out / f"{period}_run_info.json", out / f"{period}_forecasts.parquet",
                                     out / f"{period}_decisions.csv"])
        dump(out / f"{period}_wandb_urls.json", urls)
    return scores, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["pool", "tune", "test", "readiness", "freeze", "holdout"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--final-data-evidence", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step in ("readiness", "freeze", "holdout"):
        from energy_price.evaluation.router_holdout import run
        return run(args)
    table = build_table(load_inputs())
    # June-August is never generated in pool/tune mode.
    phases = ["warm", "tune"] + (["test"] if args.step == "test" else [])
    design = read_design() if args.step == "test" else None
    frames = [pool_phase(table, phase, args.out, args.jobs) for phase in phases]
    wide = wide_pool(pd.concat(frames, ignore_index=True), list(config.POOL))
    if args.step == "pool":
        return
    if args.step == "tune":
        tune = wide[wide.delivery_day.between(*PERIODS["tune"])]
        design = design_analysis(tune, args.out)
        dump(DESIGN, design)
        print(json.dumps({k: design[k] for k in ("default", "layout", "block_cap", "power_supported_layouts")}, indent=2))
        return
    forecasts, decisions = routed_history(wide, design, PERIODS["test"][1])
    decisions.to_csv(args.out / "test_decisions.csv", index=False)
    scores, pairs = summarise(forecasts, design, "test", args.out, args.track)
    # A robustness diagnostic is not a fourth candidate and cannot replace the
    # predeclared main layout after seeing its test results.
    robust, robust_decisions = routed_history(wide, design, PERIODS["test"][1], robustness=True)
    robust_dir = args.out / "robustness_6h"
    robust_dir.mkdir(exist_ok=True)
    summarise(robust, design, "test", robust_dir)
    robust_decisions.to_csv(robust_dir / "test_decisions.csv", index=False)
    candidate = min(design["candidates"], key=lambda n: (scores.loc[f"calibrated_{n}", "pinball"], n))
    dump(NOMINATION, {"candidate": candidate, "design_sha256": json_fingerprint(DESIGN),
                      "selection_period": "June-August; not blind", "final_pool_frozen": False,
                      "holdout_executed": False, "info": run_info()})
    print(scores.round(5).to_string())
    print(pairs.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
