"""Explorative architecture check of a bandit over the models (``make bandit-exploration STEP=tune|test``).

Issue #21. This is not a model candidate and does not follow the standard model workflow, see
docs/exploration/bandit/README.md. ``tune`` sees only April and May and picks the settings per
strategy. ``test`` applies ``CHOSEN`` to June to August, with the state carried over from April.
There is no holdout step, so this check never touches the holdout.

Full outputs go to ``reports/bandit_exploration/``. The results that document the check are written
to ``docs/exploration/bandit/results/``, because ``reports/`` is not versioned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price.evaluation.backtest import PERIODS, period_days, run_info, run_reference
from energy_price.evaluation.experiment import load, run_variant
from energy_price.evaluation.figures import AXIS, CATEGORICAL, GRID as GRID_COLOUR, INK, INK_SECONDARY, RC, SURFACE
from energy_price.evaluation.scores import QUANTILES, pinball
from energy_price.exploration.bandit.strategies import ARMS, CHOSEN, EXPLORATION_FLOOR, GRID, SLOTS_PER_DAY, \
    TEST_SEEDS, TUNE_SEEDS, hour_groups, run_strategy, thompson
from energy_price.features.dataset import TARGET, build_table, load_inputs
from energy_price.paths import REPORTS_DIR, ROOT

RESULTS_DIR = ROOT / "docs" / "exploration" / "bandit" / "results"
BEST_SINGLE = "gbm_quantile"  # lowest pinball loss of the arms on April and May, so the benchmark needs no hindsight
BOOTSTRAP_REPS = 5000
BOOTSTRAP_BLOCK_DAYS = 7
THOMPSON = ("thompson_bandit", "thompson_bandit_no_floor", "thompson_full_info")
PAIRS = (("gbm_quantile", "reference"), ("B2", "reference"), ("gbm_quantile", "B2"))
SPEC = {"B1": ("baseline_model", "B1"), "B2": ("baseline_model", "B2"), "gbm_quantile": ("gbm_quantile", "gbm_quantile")}
FIGURE_LABELS = {
    "thompson_bandit": "Thompson bandit per slot, 5 % floor",
    "thompson_bandit_no_floor": "Thompson bandit per slot, no floor",
    "thompson_full_info": "Thompson with all losses visible",
    "slot_winner_from_tune": "Fixed slot winner from April/May",
    "hour_winner_from_tune": "Fixed hour winner from April/May",
    "mean_all": "Plain mean of all arms",
    "hedge_global": "Online weights, one set for the day",
    "hedge_hour": "Online weights per hour",
    "hedge_slot": "Online weights per slot",
    "follow_leader_slot": "Recent best per slot, no random choice",
    "HINDSIGHT_slot_winner_of_test": "Hindsight: slot winner of the test itself",
}


def member_forecasts(table: pd.DataFrame, days: list[pd.Timestamp], jobs: int) -> pd.DataFrame:
    """Walk-forward forecasts of all arms, from the shared backtest and the chosen settings of each model."""
    frames = [run_reference(table, days)]
    for arm, (folder, variant) in SPEC.items():
        module = load(folder)
        frames.append(run_variant(module, table, days, arm, module.CHOSEN[variant], jobs)[0])
    return pd.concat(frames, ignore_index=True)


def tune_and_test_forecasts(table: pd.DataFrame, jobs: int) -> pd.DataFrame:
    """Forecasts per period: one run over both would shift the weekly refit days of the official test."""
    return pd.concat([member_forecasts(table, period_days(period), jobs) for period in ("tune", "test")],
                     ignore_index=True)


def to_arrays(frame: pd.DataFrame, days: list[pd.Timestamp]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecasts (day, slot, arm, quantile), prices (day, slot) and the days complete for all arms."""
    local = frame["timestamp_local"]
    slot = local.dt.hour * 4 + local.dt.minute // 15
    position = {day: i for i, day in enumerate(days)}
    day = pd.to_datetime(frame["delivery_day"]).map(position)
    if pd.DataFrame({"model": frame["model"], "day": day, "slot": slot}).duplicated().any():
        raise ValueError("a quarter hour appears twice; a clock change day cannot be mapped to 96 slots")
    forecasts = np.full((len(days), SLOTS_PER_DAY, len(ARMS), len(QUANTILES)), np.nan)
    prices = np.full((len(days), SLOTS_PER_DAY), np.nan)
    for arm_index, arm in enumerate(ARMS):
        rows = frame["model"] == arm
        for q_index, column in enumerate(("q10", "q50", "q90")):
            forecasts[day[rows], slot[rows], arm_index, q_index] = frame.loc[rows, column]
        prices[day[rows], slot[rows]] = frame.loc[rows, TARGET]
    valid = ~np.isnan(forecasts).any(axis=(1, 2, 3)) & ~np.isnan(prices).any(axis=1)
    return forecasts, prices, valid


def pinball_losses(forecasts: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Pinball loss per forecast value, with the shape of ``forecasts``, with or without an arm axis."""
    actual = prices if forecasts.ndim == prices.ndim + 1 else prices[..., None]
    return np.stack([pinball(actual, forecasts[..., i], tau) for i, tau in enumerate(QUANTILES)], axis=-1)


def score(forecast: np.ndarray, prices: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Pinball loss, MAE of P50, coverage and width on the days of ``mask``."""
    p, y = forecast[mask], prices[mask]
    return {"pinball": float(pinball_losses(p, y).mean()), "mae_q50": float(np.abs(y - p[..., 1]).mean()),
            "coverage_80": float(((y >= p[..., 0]) & (y <= p[..., 2])).mean()),
            "width_80": float((p[..., 2] - p[..., 0]).mean())}


def daily_loss(forecast: np.ndarray, prices: np.ndarray) -> np.ndarray:
    return pinball_losses(forecast, prices).mean(axis=(1, 2))


def block_bootstrap(diff: np.ndarray, reps: int = BOOTSTRAP_REPS, block: int = BOOTSTRAP_BLOCK_DAYS,
                    seed: int = 0) -> tuple[float, float]:
    """95 % interval of a mean daily difference; blocks of days keep the dependence between days."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = rng.integers(0, n - block + 1, (reps, n // block + 1))
    index = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    low, high = np.percentile(diff[index].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def tune_selection(forecasts: np.ndarray, prices: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    """Every ``GRID`` setting on the days given; the lowest pinball loss per strategy is marked."""
    losses = pinball_losses(forecasts, prices)
    rows = []
    for name, settings_list in GRID.items():
        seeds = TUNE_SEEDS if name in THOMPSON else 1
        for settings in settings_list:
            values = [score(run_strategy(name, forecasts, losses, valid, settings, seed), prices, valid)["pinball"]
                      for seed in range(seeds)]
            rows.append({"strategy": name, **settings, "pinball": float(np.mean(values))})
    table = pd.DataFrame(rows)
    table["chosen"] = table.index.isin(table.groupby("strategy")["pinball"].idxmin())
    return table


def fixed_strategies(forecasts: np.ndarray, losses: np.ndarray, tune: np.ndarray,
                     test: np.ndarray) -> dict[str, np.ndarray]:
    """Plain mean, winners chosen on April and May, and the two hindsight bounds no forecast can reach."""
    joint = losses.mean(axis=-1)
    slots = np.arange(forecasts.shape[1])
    hours = hour_groups(forecasts.shape[1])
    slot_winner = joint[tune].mean(axis=0).argmin(axis=1)
    hour_winner = np.array([joint[tune][:, hours == h].mean(axis=(0, 1)).argmin() for h in np.unique(hours)])
    best_each = np.nan_to_num(joint, nan=np.inf).argmin(axis=2)
    return {
        "mean_all": np.sort(forecasts.mean(axis=2), axis=-1),
        "slot_winner_from_tune": forecasts[:, slots, slot_winner],
        "hour_winner_from_tune": forecasts[:, slots, hour_winner[hours]],
        "HINDSIGHT_slot_winner_of_test": forecasts[:, slots, joint[test].mean(axis=0).argmin(axis=1)],
        "HINDSIGHT_each_forecast": np.take_along_axis(forecasts, best_each[:, :, None, None], axis=2)[:, :, 0],
    }


def persistence(losses: np.ndarray, tune: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    """Do slot differences between models repeat? Tune against test, and odd against even test days."""
    joint = losses.mean(axis=-1)
    index = np.flatnonzero(test)
    odd, even = index[::2], index[1::2]
    hours = hour_groups(joint.shape[1])
    rows = []
    for a, b in PAIRS:
        diff = joint[:, :, ARMS.index(a)] - joint[:, :, ARMS.index(b)]
        slot_tune, slot_test = pd.Series(diff[tune].mean(axis=0)), pd.Series(diff[test].mean(axis=0))
        slot_odd, slot_even = pd.Series(diff[odd].mean(axis=0)), pd.Series(diff[even].mean(axis=0))
        sd = diff[test].std(axis=0, ddof=1)
        rows.append({
            "pair": f"{a} - {b}",
            "spearman_slots_tune_test": slot_tune.corr(slot_test, method="spearman"),
            "spearman_slots_odd_even": slot_odd.corr(slot_even, method="spearman"),
            "spearman_hours_tune_test": slot_tune.groupby(hours).mean().corr(slot_test.groupby(hours).mean(),
                                                                             method="spearman"),
            "same_sign_slots_tune_test": float((np.sign(slot_tune) == np.sign(slot_test)).mean()),
            "median_abs_slot_diff": float(np.median(np.abs(slot_test))),
            "median_slot_sd": float(np.median(sd)),
            "median_days_for_80pct_power": float(np.median((2.8 * sd / np.maximum(np.abs(slot_test), 1e-12)) ** 2)),
        })
    table = pd.DataFrame(rows)
    table["same_winner_slots_tune_test"] = float(
        (joint[tune].mean(axis=0).argmin(axis=1) == joint[test].mean(axis=0).argmin(axis=1)).mean())
    table["same_winner_slots_odd_even"] = float(
        (joint[odd].mean(axis=0).argmin(axis=1) == joint[even].mean(axis=0).argmin(axis=1)).mean())
    return table


def cross_fit_winner(forecasts: np.ndarray, losses: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Slot winners picked on the odd test days and applied to the even ones, and the other way round."""
    joint = losses.mean(axis=-1)
    index = np.flatnonzero(test)
    odd, even = index[::2], index[1::2]
    slots = np.arange(forecasts.shape[1])
    out = np.full((*forecasts.shape[:2], forecasts.shape[3]), np.nan)
    out[odd] = forecasts[odd][:, slots, joint[even].mean(axis=0).argmin(axis=1)]
    out[even] = forecasts[even][:, slots, joint[odd].mean(axis=0).argmin(axis=1)]
    return out


def compare(days_loss: np.ndarray, base: np.ndarray, test: np.ndarray) -> dict[str, float]:
    """Mean difference to the benchmark in percent and its bootstrap interval in ct/kWh."""
    diff = days_loss[test] - base[test]
    low, high = block_bootstrap(diff)
    return {"vs_best_single_pct": 100 * diff.mean() / base[test].mean(), "ci_low_ct": low, "ci_high_ct": high}


def evaluate_strategies(forecasts: np.ndarray, prices: np.ndarray, valid: np.ndarray, tune: np.ndarray,
                        test: np.ndarray, chosen: dict) -> pd.DataFrame:
    """Scores of all arms and strategies on the test days, sorted by pinball loss."""
    losses = pinball_losses(forecasts, prices)
    base = daily_loss(forecasts[:, :, ARMS.index(BEST_SINGLE)], prices)
    rows: dict[str, dict] = {}
    shown = {arm: forecasts[:, :, i] for i, arm in enumerate(ARMS)}
    shown |= fixed_strategies(forecasts, losses, tune, test)
    shown["cross_fit_slot_winner_within_test"] = cross_fit_winner(forecasts, losses, test)
    for name, settings in chosen.items():
        if name not in THOMPSON:
            shown[name] = run_strategy(name, forecasts, losses, valid, settings)
    for name, forecast in shown.items():
        rows[name] = {**score(forecast, prices, test), **compare(daily_loss(forecast, prices), base, test)}
    for name in THOMPSON:
        runs = [run_strategy(name, forecasts, losses, valid, chosen[name], seed) for seed in range(TEST_SEEDS)]
        scores = pd.DataFrame([score(run, prices, test) for run in runs])
        mean_daily = np.mean([daily_loss(run, prices) for run in runs], axis=0)
        rows[name] = {**scores.mean().to_dict(), **compare(mean_daily, base, test),
                      "seed_p5": float(scores["pinball"].quantile(0.05)),
                      "seed_p95": float(scores["pinball"].quantile(0.95))}
    return pd.DataFrame(rows).T.sort_values("pinball").rename_axis("strategy")


def bandit_shares(forecasts: np.ndarray, prices: np.ndarray, valid: np.ndarray, test: np.ndarray,
                  half_life: float) -> pd.DataFrame:
    """Share of each arm in the real bandit on the test days, mean over the seeds: does it learn anything?"""
    losses = pinball_losses(forecasts, prices)
    shares = []
    for seed in range(TEST_SEEDS):
        _, choices = thompson(forecasts, losses, valid, half_life, False, EXPLORATION_FLOOR, seed)
        shares.append(np.bincount(choices[test].ravel(), minlength=len(ARMS)) / choices[test].size)
    return pd.DataFrame({"arm": ARMS, "share_test": np.mean(shares, axis=0)})


def figure(table: pd.DataFrame, path: Path) -> None:
    """Difference to the benchmark with its interval. The bound of single forecasts would flatten the scale."""
    import matplotlib.pyplot as plt

    rows = table.loc[[name for name in table.index if name in FIGURE_LABELS]].iloc[::-1]
    y = np.arange(len(rows))
    mean_ct = rows["vs_best_single_pct"].to_numpy() * table.loc[BEST_SINGLE, "pinball"] / 100
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        ax.hlines(y, rows["ci_low_ct"], rows["ci_high_ct"], color=CATEGORICAL[0], linewidth=2)
        ax.scatter(mean_ct, y, s=48, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_yticks(y, [FIGURE_LABELS[name] for name in rows.index], color=INK, fontsize=9)
        ax.set_xlabel("Pinball loss minus gradient boosting alone, ct/kWh (left of 0 is better)",
                      color=INK_SECONDARY, fontsize=9)
        ax.set_title("A bandit over the models is worse; online weights give no clear gain\n"
                     "June to August 2026, 95 % block bootstrap interval", loc="left", color=INK, fontsize=11)
        ax.grid(axis="x", color=GRID_COLOUR, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8, length=0)
        ax.tick_params(axis="y", length=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def run_tune(table: pd.DataFrame, jobs: int, out: Path) -> None:
    days = period_days("tune")
    forecasts, prices, valid = to_arrays(member_forecasts(table, days, jobs), days)
    selection = tune_selection(forecasts, prices, valid)
    out.mkdir(parents=True, exist_ok=True)
    selection.to_csv(out / "tune_selection.csv", index=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selection.to_csv(RESULTS_DIR / "tune_selection.csv", index=False, float_format="%.4f")
    arms ={arm: score(forecasts[:, :, i], prices, valid)["pinball"] for i, arm in enumerate(ARMS)}
    print(pd.Series(arms).round(4).to_string())
    print(selection[selection["chosen"]].round(4).to_string(index=False))


def run_test(table: pd.DataFrame, jobs: int, out: Path) -> None:
    if set(CHOSEN) != set(GRID):
        raise ValueError("CHOSEN must hold one setting per strategy: run STEP=tune first")
    days = period_days("tune") + period_days("test")
    forecasts, prices, valid = to_arrays(tune_and_test_forecasts(table, jobs), days)
    stamps = pd.DatetimeIndex(days)
    tune = np.asarray(stamps <= PERIODS["tune"][1]) & valid
    test = np.asarray((stamps >= PERIODS["test"][0]) & (stamps <= PERIODS["test"][1])) & valid
    results = evaluate_strategies(forecasts, prices, valid, tune, test, CHOSEN)
    stable = persistence(pinball_losses(forecasts, prices), tune, test)
    shares = bandit_shares(forecasts, prices, valid, test, CHOSEN["thompson_bandit"]["half_life"])

    out.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (out, RESULTS_DIR):
        results.to_csv(folder / "test_results.csv", float_format="%.4f")
        stable.to_csv(folder / "test_persistence.csv", index=False, float_format="%.4f")
        shares.to_csv(folder / "test_bandit_shares.csv", index=False, float_format="%.4f")
    info = run_info({"arms": list(ARMS), "chosen": {k: str(v) for k, v in CHOSEN.items()}, "test_seeds": TEST_SEEDS})
    (out / "test_run_info.json").write_text(json.dumps(info, indent=2))
    figure(results, RESULTS_DIR / "test_strategies.png")

    pd.set_option("display.width", 180)
    print(f"days tune {tune.sum()}, test {test.sum()}, dropped {[str(d.date()) for d in stamps[~valid]]}")
    print(results.round(4).to_string())
    print(stable.round(3).T.to_string())
    print(shares.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["tune", "test"])
    parser.add_argument("--jobs", type=int, default=-1, help="parallel folds, -1 uses all cores")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "bandit_exploration")
    args = parser.parse_args(argv)
    table = build_table(load_inputs())
    if args.step == "tune":
        run_tune(table, args.jobs, args.out)
    else:
        run_test(table, args.jobs, args.out)


if __name__ == "__main__":
    main()
