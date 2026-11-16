"""Is a bandit or an online aggregation of existing models useful for the price forecast?

Exploratory, read-only check on tune (April, May) and test (June to August). The holdout is not used.
Every online strategy decides for delivery day D using only losses of days up to D-2. Settings are
chosen on tune and then scored on test, with the state carried over from tune.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MODELS = ["reference", "B1", "B2", "gbm", "regime"]
TAUS = np.array([0.1, 0.5, 0.9])
HALF_LIVES = [7, 14, 28, 56, np.inf]
ETAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
EPS = 1e-12


def load():
    frames = [pd.read_parquet(HERE / "forecasts" / f"{p}_{m}.parquet") for p in ("tune", "test") for m in MODELS]
    df = pd.concat(frames, ignore_index=True)
    target = [c for c in df.columns if c not in {"timestamp_local", "timestamp_utc", "delivery_day", "q10", "q50",
                                                 "q90", "model"}][0]
    local = df["timestamp_local"]
    df["slot"] = local.dt.hour * 4 + local.dt.minute // 15
    days = pd.date_range("2026-04-01", "2026-08-31", freq="D")
    index = {d: i for i, d in enumerate(days)}
    df["day_i"] = pd.to_datetime(df["delivery_day"]).map(index)
    forecasts = np.full((len(days), 96, len(MODELS), 3), np.nan)
    y = np.full((len(days), 96), np.nan)
    for k, m in enumerate(MODELS):
        g = df[df["model"] == m]
        forecasts[g["day_i"], g["slot"], k, 0] = g["q10"]
        forecasts[g["day_i"], g["slot"], k, 1] = g["q50"]
        forecasts[g["day_i"], g["slot"], k, 2] = g["q90"]
        y[g["day_i"], g["slot"]] = g[target]
    valid = ~np.isnan(forecasts).any(axis=(1, 2, 3)) & ~np.isnan(y).any(axis=1)
    return days, forecasts, y, valid


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score(p, y, mask):
    p, y = p[mask], y[mask]
    losses = np.stack([pinball(y, p[..., i], TAUS[i]) for i in range(3)], axis=-1)
    return {"pinball": losses.mean(), "mae": np.abs(y - p[..., 1]).mean(),
            "coverage": ((y >= p[..., 0]) & (y <= p[..., 2])).mean(), "width": (p[..., 2] - p[..., 0]).mean()}


def daily(p, y):
    losses = np.stack([pinball(y, p[..., i], TAUS[i]) for i in range(3)], axis=-1)
    return losses.mean(axis=(1, 2))


def gamma(half_life):
    return 1.0 if np.isinf(half_life) else 0.5 ** (1 / half_life)


def hedge(F, L, valid, groups, eta, half_life):
    """Exponential weights per group and quantile; the forecast is the weighted mean of the quantiles."""
    n_days, n_slots, n_arms, _ = F.shape
    n_groups = groups.max() + 1
    cum = np.zeros((n_groups, n_arms, 3))
    out = np.full((n_days, n_slots, 3), np.nan)
    g = gamma(half_life)
    weights_seen = np.zeros(n_arms)
    for d in range(n_days):
        cum *= g
        if d >= 2 and valid[d - 2]:
            for j in range(n_groups):
                cum[j] += L[d - 2, groups == j].mean(axis=0)
        w = np.exp(-eta * (cum - cum.min(axis=1, keepdims=True)))
        w /= w.sum(axis=1, keepdims=True)
        for s in range(n_slots):
            out[d, s] = (w[groups[s]] * F[d, s]).sum(axis=0)
        weights_seen += w.mean(axis=(0, 2))
    return np.sort(out, axis=-1), weights_seen / n_days


def follow_leader(F, L, valid, half_life):
    """Per slot the arm with the lowest decayed loss so far; no sampling."""
    n_days, n_slots, n_arms, _ = F.shape
    cum = np.zeros((n_slots, n_arms))
    out = np.full((n_days, n_slots, 3), np.nan)
    g = gamma(half_life)
    joint = L.mean(axis=-1)
    for d in range(n_days):
        cum *= g
        if d >= 2 and valid[d - 2]:
            cum += joint[d - 2]
        choice = cum.argmin(axis=1)
        out[d] = F[d, np.arange(n_slots), choice]
    return out


def thompson(F, L, valid, half_life, full_info, floor, rng):
    """Gaussian Thompson sampling per slot, delayed feedback, decayed observations, minimum share per arm."""
    n_days, n_slots, n_arms, _ = F.shape
    joint = L.mean(axis=-1)
    weight, total, square = np.zeros((3, n_slots, n_arms))
    out = np.full((n_days, n_slots, 3), np.nan)
    choices = np.zeros((n_days, n_slots), dtype=int)
    g = gamma(half_life)
    rows = np.arange(n_slots)
    for d in range(n_days):
        weight *= g
        total *= g
        square *= g
        if d >= 2 and valid[d - 2]:
            seen = joint[d - 2]
            if full_info:
                weight += 1
                total += seen
                square += seen ** 2
            else:
                c = choices[d - 2]
                weight[rows, c] += 1
                total[rows, c] += seen[rows, c]
                square[rows, c] += seen[rows, c] ** 2
        all_w = weight.sum(axis=1)
        pooled = np.where(all_w > 0, total.sum(axis=1) / np.maximum(all_w, EPS), 3.0)
        second = np.where(all_w > 0, square.sum(axis=1) / np.maximum(all_w, EPS), 34.0)
        sigma2 = np.maximum(second - pooled ** 2, 1.0)
        mean = (pooled[:, None] + total) / (1 + weight)
        var = sigma2[:, None] / (1 + weight)
        draw = rng.normal(mean, np.sqrt(var))
        choice = draw.argmin(axis=1)
        explore = rng.random(n_slots) < floor * n_arms
        choice[explore] = rng.integers(0, n_arms, explore.sum())
        choices[d] = choice
        out[d] = F[d, rows, choice]
    return out, choices


def block_bootstrap(diff, reps=5000, block=7, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = rng.integers(0, n - block + 1, (reps, n // block + 1))
    index = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    means = diff[index].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    days, F, y, valid = load()
    is_tune = (days >= "2026-04-01") & (days <= "2026-05-31")
    is_test = (days >= "2026-06-01") & (days <= "2026-08-31")
    tune_mask, test_mask = is_tune & valid, is_test & valid
    print("valid days tune/test:", tune_mask.sum(), test_mask.sum(), "dropped:",
          [str(d.date()) for d in days[~valid]])
    L = np.stack([pinball(y[..., None], F[..., i], TAUS[i]) for i in range(3)], axis=-1)  # day, slot, arm, q

    print("\n== Single models ==")
    single = {m: F[:, :, k] for k, m in enumerate(MODELS)}
    rows = {m: {**{f"tune_{a}": b for a, b in score(p, y, tune_mask).items() if a == "pinball"},
                **score(p, y, test_mask)} for m, p in single.items()}
    print(pd.DataFrame(rows).T.round(4).to_string())
    best_on_tune = min(MODELS, key=lambda m: score(single[m], y, tune_mask)["pinball"])
    print("best on tune:", best_on_tune)

    print("\n== Pairwise correlation of errors of the median and of the daily loss on test ==")
    err = pd.DataFrame({m: (y - F[:, :, k, 1])[test_mask].ravel() for k, m in enumerate(MODELS)})
    print(err.corr().round(3).to_string())
    daily_loss = pd.DataFrame({m: daily(single[m], y)[test_mask] for m in MODELS})
    print(daily_loss.corr().round(3).to_string())

    print("\n== Pinball per hour, test (tune in second table) ==")
    hour = np.arange(96) // 4
    for mask, label in ((test_mask, "test"), (tune_mask, "tune")):
        per_hour = pd.DataFrame({m: [L[mask][:, hour == h, k].mean() for h in range(24)]
                                 for k, m in enumerate(MODELS)})
        per_hour["winner"] = per_hour[MODELS].idxmin(axis=1)
        print(label)
        print(per_hour.round(2).to_string())

    print("\n== Persistence of slot differences ==")
    joint = L.mean(axis=-1)
    test_idx = np.flatnonzero(test_mask)
    odd, even = test_idx[::2], test_idx[1::2]
    for a, b in (("gbm", "reference"), ("B2", "reference"), ("gbm", "B2")):
        ia, ib = MODELS.index(a), MODELS.index(b)
        diff = joint[:, :, ia] - joint[:, :, ib]
        slot_tune, slot_test = diff[tune_mask].mean(axis=0), diff[test_mask].mean(axis=0)
        slot_odd, slot_even = diff[odd].mean(axis=0), diff[even].mean(axis=0)
        hour_tune = pd.Series(slot_tune).groupby(hour).mean()
        hour_test = pd.Series(slot_test).groupby(hour).mean()
        sd = diff[test_mask].std(axis=0, ddof=1)
        se = sd / np.sqrt(test_mask.sum())
        t = slot_test / se
        needed = (2.8 * sd / np.maximum(np.abs(slot_test), EPS)) ** 2
        print(f"{a} - {b}: spearman slots tune/test {pd.Series(slot_tune).corr(pd.Series(slot_test), method='spearman'):.2f}, "
              f"slots odd/even test {pd.Series(slot_odd).corr(pd.Series(slot_even), method='spearman'):.2f}, "
              f"hours tune/test {hour_tune.corr(hour_test, method='spearman'):.2f}; "
              f"same sign tune/test {(np.sign(slot_tune) == np.sign(slot_test)).mean():.0%}; "
              f"slots |t|>2: {(np.abs(t) > 2).mean():.0%}; median |diff| {np.median(np.abs(slot_test)):.3f}, "
              f"median sd {np.median(sd):.2f}; days for 80 % power median {np.median(needed):.0f}")
    win_tune = joint[tune_mask].mean(axis=0).argmin(axis=1)
    win_test = joint[test_mask].mean(axis=0).argmin(axis=1)
    win_odd = joint[odd].mean(axis=0).argmin(axis=1)
    win_even = joint[even].mean(axis=0).argmin(axis=1)
    print(f"same winner tune/test {(win_tune == win_test).mean():.0%}, odd/even test {(win_odd == win_even).mean():.0%}, "
          f"chance if uniform {1 / len(MODELS):.0%}")
    print("slot winners on test (share):",
          {m: round(float((win_test == k).mean()), 2) for k, m in enumerate(MODELS)})

    print("\n== Strategies ==")
    strategies = {}
    for m in MODELS:
        strategies[m] = single[m]
    strategies["mean_all"] = np.sort(F.mean(axis=2), axis=-1)

    # fixed choice from tune
    rows = np.arange(96)
    strategies["slot_winner_from_tune"] = F[:, rows, win_tune]
    win_hour = np.array([joint[tune_mask][:, hour == h].mean(axis=(0, 1)).argmin() for h in range(24)])
    strategies["hour_winner_from_tune"] = F[:, rows, win_hour[hour]]

    # hindsight bounds on test: not achievable
    strategies["ORACLE_slot_winner_of_test"] = F[:, rows, win_test]
    day_slot_oracle = joint.argmin(axis=2)
    strategies["ORACLE_each_forecast"] = np.take_along_axis(F, day_slot_oracle[:, :, None, None], axis=2)[:, :, 0]
    cross = np.full(F.shape[:2] + (3,), np.nan)
    cross[odd] = F[odd][:, rows, win_even]
    cross[even] = F[even][:, rows, win_odd]
    cross_score = score(cross, y, test_mask)
    print("cross-fit slot winner within test (odd picks even, even picks odd):", {k: round(v, 4) for k, v in cross_score.items()})

    # online strategies, settings chosen on tune only
    group_sets = {"global": np.zeros(96, dtype=int), "hour": hour, "slot": rows}
    for name, groups in group_sets.items():
        best = None
        for eta in ETAS:
            for h in HALF_LIVES:
                p, seen = hedge(F, L, valid, groups, eta, h)
                value = score(p, y, tune_mask)["pinball"]
                if best is None or value < best[0]:
                    best = (value, eta, h, p, seen)
        print(f"hedge_{name}: eta={best[1]} half_life={best[2]} tune={best[0]:.4f} "
              f"mean weights={dict(zip(MODELS, best[4].round(2)))}")
        strategies[f"hedge_{name}"] = best[3]

    best = min(((score(follow_leader(F, L, valid, h), y, tune_mask)["pinball"], h) for h in HALF_LIVES))
    print(f"follow_leader_slot: half_life={best[1]} tune={best[0]:.4f}")
    strategies["follow_leader_slot"] = follow_leader(F, L, valid, best[1])

    def thompson_mean(half_life, full_info, floor, seeds):
        results, shares = [], []
        for seed in range(seeds):
            p, choices = thompson(F, L, valid, half_life, full_info, floor, np.random.default_rng(seed))
            results.append(p)
            shares.append(np.bincount(choices[test_mask].ravel(), minlength=len(MODELS)) / choices[test_mask].size)
        return results, np.mean(shares, axis=0)

    variants = {}
    for label, full_info, floor in (("thompson_bandit_floor5", False, 0.05), ("thompson_bandit_nofloor", False, 0.0),
                                    ("thompson_fullinfo_floor5", True, 0.05)):
        best = None
        for h in HALF_LIVES:
            runs, _ = thompson_mean(h, full_info, floor, 20)
            value = np.mean([score(p, y, tune_mask)["pinball"] for p in runs])
            if best is None or value < best[0]:
                best = (value, h)
        runs, shares = thompson_mean(best[1], full_info, floor, 200)
        print(f"{label}: half_life={best[1]} tune={best[0]:.4f} test shares={dict(zip(MODELS, shares.round(2)))}")
        variants[label] = runs

    rows_out = {}
    base = {m: daily(single[m], y)[test_mask] for m in ("gbm", "reference")}
    for name, p in strategies.items():
        s = score(p, y, test_mask)
        d = daily(p, y)[test_mask]
        for ref in ("gbm", "reference"):
            low, high = block_bootstrap(d - base[ref])
            s[f"vs_{ref}_%"] = 100 * (d.mean() - base[ref].mean()) / base[ref].mean()
            s[f"vs_{ref}_ci"] = f"[{low:+.3f}, {high:+.3f}]"
        rows_out[name] = s
    for name, runs in variants.items():
        all_scores = pd.DataFrame([score(p, y, test_mask) for p in runs])
        s = all_scores.mean().to_dict()
        d = np.mean([daily(p, y)[test_mask] for p in runs], axis=0)
        for ref in ("gbm", "reference"):
            low, high = block_bootstrap(d - base[ref])
            s[f"vs_{ref}_%"] = 100 * (d.mean() - base[ref].mean()) / base[ref].mean()
            s[f"vs_{ref}_ci"] = f"[{low:+.3f}, {high:+.3f}]"
        s["seed_p5_p95"] = f"{np.percentile(all_scores['pinball'], 5):.4f} to {np.percentile(all_scores['pinball'], 95):.4f}"
        rows_out[name] = s
    table = pd.DataFrame(rows_out).T
    print(table.sort_values("pinball").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nCI: 95 % moving block bootstrap (7 days) of the mean daily difference in ct/kWh, negative = better.")
    print("Cross-fit vs gbm %:", round(100 * (cross_score['pinball'] - score(single['gbm'], y, test_mask)['pinball'])
                                       / score(single['gbm'], y, test_mask)['pinball'], 2))


if __name__ == "__main__":
    main()
