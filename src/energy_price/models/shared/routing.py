"""Weekly, D-2-safe selection and convex combination of OOS quantile forecasts."""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from energy_price.evaluation.scores import pinball
from energy_price.features.dataset import TARGET
from energy_price.models.shared.forecast_pool import Q


def block_ids(hours, edges):
    edges = np.asarray(edges)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or edges[-1] != 24 or (np.diff(edges) <= 0).any():
        raise ValueError("blocks must partition [0, 24) without gaps or overlaps")
    hours = np.asarray(hours)
    if ((hours < 0) | (hours >= 24)).any():
        raise ValueError("hour outside [0, 24)")
    return np.searchsorted(edges, hours, side="right") - 1


def bootstrap_means(daily, reps=5000, block=7, seed=0):
    """Moving calendar-day block bootstrap; missing days remain gaps.

    Whole days, never individual quarter hours, are resampled. NaN days in
    sampled calendar blocks contribute no observation to that replicate.
    """
    daily = daily.sort_index()
    values = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).to_numpy(float)
    if np.isfinite(values).sum() < 2 or len(values) < block:
        raise ValueError("insufficient daily history for block bootstrap")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values) - block + 1, (reps, int(np.ceil(len(values) / block))))
    index = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :len(values)]
    draws = values[index]
    counts = np.isfinite(draws).sum(axis=1)
    return np.nansum(draws, axis=1)[counts > 0] / counts[counts > 0]


def difference_interval(daily, comparisons=1, reps=5000, seed=0):
    draws = bootstrap_means(daily, reps=reps, seed=seed)
    alpha = .05 / comparisons
    return tuple(float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def daily_expert_losses(table, names, edges):
    groups = block_ids(table.hour, edges)
    losses = {name: np.mean([pinball(table[TARGET], table[f"{name}:{q}"], tau)
                            for q, tau in zip(Q, (.1, .5, .9))], axis=0) for name in names}
    return pd.DataFrame(losses, index=table.index).groupby([table.delivery_day, groups]).mean()


def convex_weights(x, y, tau):
    """Exact empirical pinball minimisation on the simplex (sparse HiGHS LP)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = x.shape
    # y-Xw = positive-negative, w>=0 and sum(w)=1.
    objective = np.r_[np.zeros(m), np.full(n, tau / n), np.full(n, (1 - tau) / n)]
    aeq = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(x), sparse.eye(n), -sparse.eye(n)]),
        sparse.csr_matrix(np.r_[np.ones(m), np.zeros(2 * n)][None, :]),
    ], format="csr")
    result = linprog(objective, A_eq=aeq, b_eq=np.r_[y, 1.], bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"convex quantile combination failed: {result.message}")
    weights = np.maximum(result.x[:m], 0)
    return weights / weights.sum()


class TimeRouterModel:
    """Fit/predict protocol on a wide OOS pool table, never on raw features.

    Each weekly decision uses observed expert errors through fit_day-2. One
    selected expert supplies all three quantiles. Learned convex weights are
    shrunk halfway to the default; this fixed safeguard is not tuned on test.
    """

    def __init__(self, names, default, edges=(0, 24), strategy="selection", fit_day=None,
                 min_days=28, reps=5000, seed=0, shrinkage=.5, selection_mode="significant"):
        block_ids([], edges)
        if default not in names or strategy not in ("default", "mean", "selection", "weights"):
            raise ValueError("invalid default expert or strategy")
        if selection_mode not in ("significant", "empirical"):
            raise ValueError("selection_mode must be significant or empirical")
        self.names, self.default, self.edges, self.strategy = list(names), default, list(edges), strategy
        self.selection_mode = selection_mode
        self.fit_day, self.min_days, self.reps, self.seed = fit_day, min_days, reps, seed
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = shrinkage

    def fit(self, table):
        if self.fit_day is None:
            raise ValueError("explicit weekly fit_day is required")
        cutoff = pd.Timestamp(self.fit_day) - pd.Timedelta(days=2)
        known = table[table.delivery_day <= cutoff].copy()
        cols = [f"{n}:{q}" for n in self.names for q in Q]
        known = known[np.isfinite(known[cols + [TARGET]]).all(axis=1)]
        self.error_last_day_ = known.delivery_day.max()
        self.weights_, self.choices_, self.diagnostics_, self.schedule_ = {}, {}, [], []
        default = np.eye(len(self.names))[self.names.index(self.default)]
        groups = block_ids(known.hour, self.edges)
        losses = daily_expert_losses(known, self.names, self.edges) if len(known) else None
        comparisons = max(1, (len(self.names) - 1) * (len(self.edges) - 1))
        for group in range(len(self.edges) - 1):
            w = np.tile(default, (3, 1))
            choice = self.default
            rows = known[groups == group]
            daily = losses.xs(group, level=1) if len(rows) else pd.DataFrame(columns=self.names)
            means = daily.mean()
            empirical_best = (min(self.names, key=lambda name: (means[name], name != self.default, name))
                              if len(rows) else self.default)
            reason = "default"
            eligible = []
            if self.strategy == "mean":
                w[:] = 1 / len(self.names)
                choice = "mixture"
                reason = "equal_weights"
            elif rows.delivery_day.nunique() >= self.min_days and self.strategy == "selection":
                for name in self.names:
                    if name == self.default:
                        continue
                    diff = daily[name] - daily[self.default]
                    draws = bootstrap_means(diff, reps=self.reps, seed=self.seed)
                    ordinary = np.quantile(draws, [.025, .975])
                    adjusted = np.quantile(draws, [.025 / comparisons, 1 - .025 / comparisons])
                    self.diagnostics_.append({"block": group, "expert": name, "mean_difference": diff.mean(),
                                              "ci95_low": ordinary[0], "ci95_high": ordinary[1],
                                              "family_low": adjusted[0], "family_high": adjusted[1], "n_days": len(diff)})
                    if adjusted[1] < 0:
                        eligible.append(name)
                if self.selection_mode == "empirical":
                    choice = empirical_best
                    reason = "lowest_past_hour_loss"
                elif eligible:
                    choice = min(eligible, key=lambda name: (daily[name].mean(), name))
                    reason = "supported_past_hour_advantage"
                else:
                    reason = "no_supported_advantage"
                w[:] = np.eye(len(self.names))[self.names.index(choice)]
            elif rows.delivery_day.nunique() >= self.min_days and self.strategy == "weights":
                for i, (q, tau) in enumerate(zip(Q, (.1, .5, .9))):
                    raw = convex_weights(rows[[f"{n}:{q}" for n in self.names]], rows[TARGET], tau)
                    w[i] = self.shrinkage * default + (1 - self.shrinkage) * raw
                choice = "mixture"
                reason = "learned_weights"
            elif self.strategy in ("selection", "weights"):
                reason = "insufficient_history"
            self.weights_[group], self.choices_[group] = w, choice
            self.schedule_.append({
                "block": group, "start_hour": self.edges[group], "end_hour": self.edges[group + 1],
                "fit_day": pd.Timestamp(self.fit_day), "history_last_day": rows.delivery_day.max(),
                "n_history_days": len(daily), "empirical_best": empirical_best, "chosen_expert": choice,
                "selection_mode": self.selection_mode, "reason": reason,
                "chosen_supported_vs_default": choice in eligible,
                "empirical_best_pinball": float(means[empirical_best]) if len(rows) else np.nan,
                "default_pinball": float(means[self.default]) if len(rows) else np.nan,
                "chosen_pinball": float(means[choice]) if choice in self.names and len(rows) else np.nan,
                **{f"history_pinball:{name}": float(means[name]) if len(rows) else np.nan for name in self.names},
            })
        return self

    def predict(self, table):
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit the router first")
        if len(table) and (table.delivery_day.min() < pd.Timestamp(self.fit_day)):
            raise ValueError("cannot predict before the weekly fit day")
        groups = block_ids(table.hour, self.edges)
        out = pd.DataFrame(index=table.index, columns=Q, dtype=float)
        for group, weights in self.weights_.items():
            mask = groups == group
            for i, q in enumerate(Q):
                out.loc[mask, q] = table.loc[mask, [f"{n}:{q}" for n in self.names]].to_numpy() @ weights[i]
        original = out[Q].to_numpy().copy()
        out[Q] = np.sort(original, axis=1)
        out["quantiles_reordered"] = (original != out[Q].to_numpy()).any(axis=1)
        out["chosen_expert"] = [self.choices_[g] for g in groups]
        out["weights"] = [json.dumps({q: dict(zip(self.names, self.weights_[g][i].tolist()))
                                       for i, q in enumerate(Q)}, sort_keys=True) for g in groups]
        for i, q in enumerate(Q):
            out[f"preorder_{q}"] = original[:, i]
        out["block"] = groups
        out["empirical_best"] = [self.schedule_[g]["empirical_best"] for g in groups]
        out["selection_reason"] = [self.schedule_[g]["reason"] for g in groups]
        out["fit_day"] = pd.Timestamp(self.fit_day)
        out["error_last_day"] = self.error_last_day_
        return out
