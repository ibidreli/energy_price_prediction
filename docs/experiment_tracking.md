# Experiment Tracking with Weights & Biases

All model experiments are logged to one shared W&B project (`energy-price-prediction`) in the team's entity. W&B now runs under the CoreWeave "Forge" brand; the `wandb` SDK and CLI are unchanged.

The repository contains the setup, never credentials: no API key and no links to the W&B workspace. The team name in `.env.example` is not a secret: access is controlled by team membership in W&B and by the project being **Private**. The GitHub repository is public, so every project in the team must stay Private.

## One-time setup per person

1. Accept the team invitation in W&B (sent to your FHNW address). Ask a team member if you have none.
2. Install the environment and log in. The API key is in your W&B user settings; `wandb login` stores it in `~/.netrc`, outside the repository.

   ```bash
   make venv
   .venv/bin/wandb login
   ```

3. Create your local settings file. The team name is already filled in:

   ```bash
   cp .env.example .env    # .env is ignored by git
   ```

4. Check the setup. It prints your W&B user and whether you are a member of the team entity, never the key.

   ```bash
   make tracking-check
   ```

## Logging a run

For model backtests, use `make backtest MODEL=gbm_quantile STEP=tune TRACK=1 JOBS=2`
(and, once the D−2 publication assumption is confirmed, `STEP=test`). Tracking works for
both tuning and evaluation. Each tuning candidate, the historical reference and baseline B2
get separate runs with summary scores and daily pinball curves. `gbm_quantile` explicitly
sets `ISSUE = 17`, so the runs share `issue-17` even when the branch name does not match
the usual convention.

The selected tuning run also contains grouped permutation importance as a table and bar
chart, score breakdowns, and an evaluation artifact with aggregate CSV/JSON results and
an explicit snapshot of the Python source and dependency declarations. The source snapshot
also makes uncommitted runs inspectable; `git_dirty` still records their actual state.
Raw datasets and local settings/credentials are excluded. Run URLs remain in ignored
`reports/<model>/tune_wandb_urls.json`, never in committed documentation.

Use `start_run` instead of calling `wandb.init` directly. It fills in the team entity and project, groups runs by hypothesis (the GitHub issue) and records which data and code the run used.

One hypothesis is one issue; every candidate model is its own run in that issue's group. Give each run a `name` and put the model into `config`, so the runs table shows which row is which model:

```python
from energy_price.evaluation.tracking import start_run

candidates = {"ridge": make_ridge(), "lgbm_quantile": make_lgbm()}
cv = {"scheme": "expanding_window", "n_folds": 5, "test_start": "2026-07-01"}

for model_name, model in candidates.items():
    with start_run("train", issue=15, name=model_name, config={"model": model_name, "cv": cv}) as run:
        fold_mae = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            ...  # fit scalers and the model on train_idx only
            fold_mae.append(mae)
            run.log({"fold": fold, "val/mae": mae})
        run.summary["val/mae_mean"] = np.mean(fold_mae)
        run.summary["val/mae_std"] = np.std(fold_mae)
```

`issue=15` can be left out on branches whose name carries the number (`exp/15-...`, `feature/15-...`, `<user>/exp-15-...`, up to three digits). The same code works in notebooks and scripts. `start_run` reads `.env` itself.

## Conventions

| Field | Convention | Set by |
|---|---|---|
| Project | `energy-price-prediction`; trial runs go elsewhere, see below | `start_run` |
| Group | `issue-<n>` from `issue=` or the branch name; else the branch name. Override with `group=`, e.g. for a sweep | `start_run` |
| Tags | Branch type and issue, e.g. `exp`, `issue-15`; add your own with `tags=[...]` | `start_run` |
| Name | Model or variant, e.g. `lgbm_quantile` | you |
| Job type | One of `eda`, `features`, `train`, `eval`, `sweep`; anything else is rejected | you |
| `config["data_version"]` | Hash over all checksum stamps from `make data`; equal hash means equal raw data | `start_run` |
| `config["git_commit"]`, `["git_dirty"]`, `["git_branch"]`, `["issue"]` | Code version of the run, filterable in the runs table. Passing these or `data_version` in your own `config` is an error | `start_run` |
| `config["cv"]` | Validation scheme, at least `scheme`, `n_folds`, `test_start` | you |
| Metrics | `<split>/<metric>`, e.g. `val/mae`, `val/pinball_q10`, `test/coverage_80`; fold means and spread as `val/<metric>_mean`, `_std` in `run.summary` | you |

`git_dirty: true` means the run used code that is not committed and cannot be reproduced from `git_commit`. Commit before a run you want to compare later. W&B also records the Python version, the host name and system metrics on its own.

## Validation without leakage

The prices are strongly autocorrelated, so a random K-fold split leaks the future into training and gives far too good scores. Use time-ordered splits only (expanding or rolling window), fit scalers and encoders inside each training fold, and respect the publication times in [data.md](data.md). Runs whose `config["cv"]` is missing or says anything other than a time-ordered scheme are not comparable.

## Trial runs

Keep the main project for runs you want to compare. For trying things out, set `WANDB_PROJECT=sandbox` in `.env`, or `WANDB_MODE=disabled` to log nothing. Delete failed or aborted runs from the main project.

## What never goes to W&B or into the repository

- API keys and tokens (W&B, ENTSO-E, anything else). Do not pass them in `config`, `notes` or logged tables: every team member can see them.
- `.env`, `wandb/` and `artifacts/` are in `.gitignore`. Do not force-add them.
- Links to the W&B workspace do not go into committed files. Share them in the team chat if needed.

## Offline and tests

- `WANDB_MODE=offline` logs into the local `wandb/` folder; upload later with `.venv/bin/wandb sync wandb/offline-run-*`.
- `WANDB_MODE=disabled` turns tracking off completely. The unit tests run this way, without network or login.
- Without `WANDB_ENTITY`, `start_run` refuses to run online or offline (a synced offline run would land in the personal account). Only `disabled` works without it.
- `start_run` reads only `WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_MODE` from `.env`. Set the mode there, not as an argument: `start_run` rejects `mode=`, `entity=` and `project=`.
- Inside a sweep (`wandb.agent`), call `start_run("sweep", ...)` in the training function. W&B takes entity, project and the swept parameters from the sweep.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `WANDB_ENTITY is not set` | `.env` missing or empty, see step 3 |
| `W&B login failed` | Run `.venv/bin/wandb login` (or `--relogin` for another account) |
| `not a member of WANDB_ENTITY` | Invitation not accepted yet, or a typo in the team name |
| Project does not exist yet | Normal before the first run. After the first run, check that the project is set to Private |
