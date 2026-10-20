"""Start Weights & Biases runs in the team project with the shared conventions.

Setup once per person (details in docs/experiment_tracking.md):

1. ``wandb login``: stores the personal API key in ~/.netrc, outside the repository.
2. Copy ``.env.example`` to ``.env`` (ignored by git) and fill in ``WANDB_ENTITY``.

Every run goes to the team entity and the project ``energy-price-prediction``. Runs of one
hypothesis share the group ``issue-<n>``, taken from ``issue=`` or the branch (``exp/15-...``).
The config gets a hash of the raw data checksums (``data/processed/.*_checksums``), the git
commit and whether the working tree had uncommitted changes.

Never put tokens or credentials into ``config``: it is uploaded and visible to the whole team.

Example::

    from energy_price.evaluation.tracking import start_run

    with start_run("train", config={"model": "lgbm_quantile", "alpha": 0.1}) as run:
        run.log({"val/pinball_loss": 0.42})

Usage: ``python -m energy_price.evaluation.tracking`` checks login and entity without starting a run.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import wandb
from dotenv import dotenv_values

from energy_price.paths import ENV_FILE, PROCESSED_DIR, ROOT

PROJECT = "energy-price-prediction"
# Only these keys are taken from .env, so a key or token put there by mistake stays unused.
ENV_KEYS = ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_MODE")
# One stamp per data source, written by `make data`; new sources are picked up automatically.
DATA_STAMP_GLOB = ".*_checksums"
# Branch naming from docs/project_management.md is <type>/<issue-id>-<description>. Branches in use
# also look like feature/4-..., <user>/exp-15-... or <user>/feat/12-..., so match anywhere after a "/".
# At most three digits, so a year as in eda/2026-analysis is not read as an issue.
ISSUE_PATTERN = re.compile(r"(?:^|/)(feat|feature|exp|eda|fix|docs)[/-](\d{1,3})(?:-|$)")
JOB_TYPES = ("eda", "features", "train", "eval", "sweep")
# Provenance written by start_run; a user config must not override it, e.g. when copied from an old run.
PROVENANCE_KEYS = {"data_version", "issue", "git_branch", "git_commit", "git_dirty"}
# Set by start_run itself; mode comes from WANDB_MODE so the entity check always sees it.
RESERVED_INIT_ARGS = {"entity", "project", "mode"}


class TrackingConfigError(RuntimeError):
    """W&B is not set up locally: the team entity is missing."""


def load_env(path: Path | None = None) -> None:
    """Copy ``ENV_KEYS`` from ``.env`` into the environment. Variables that are already set win."""
    for key, value in dotenv_values(path or ENV_FILE).items():
        if key in ENV_KEYS and value and key not in os.environ:
            os.environ[key] = value


def resolve_entity(environ: Mapping[str, str]) -> str | None:
    """Team entity from the environment; required unless tracking is disabled.

    Offline runs need it too: ``wandb sync`` uploads them to the entity stored in the run.
    """
    entity = environ.get("WANDB_ENTITY") or None
    if entity is None and environ.get("WANDB_MODE", "online") != "disabled":
        raise TrackingConfigError(
            "WANDB_ENTITY is not set, so the run would land in your personal account. "
            "Copy .env.example to .env (docs/experiment_tracking.md)."
        )
    return entity


def _git(cwd: Path, *args: str) -> str | None:
    """Output of a git command, or None outside a repository or without git."""
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_branch(cwd: Path = ROOT) -> str | None:
    """Current branch name, or None outside a repository or on a detached HEAD."""
    out = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    return None if out in (None, "", "HEAD") else out


def git_state(cwd: Path = ROOT) -> dict[str, Any]:
    """Branch, commit and whether tracked or new files differ from it; None outside a repository."""
    status = _git(cwd, "status", "--porcelain")
    return {
        "git_branch": git_branch(cwd),
        "git_commit": _git(cwd, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def branch_issue(branch: str | None) -> tuple[str, int] | None:
    """Branch type and issue number: ``exp/15-quantile-lgb`` gives ``("exp", 15)``, else None."""
    match = ISSUE_PATTERN.search(branch or "")
    if match is None:
        return None
    kind = "feat" if match[1] == "feature" else match[1]
    return kind, int(match[2])


def data_version(processed_dir: Path = PROCESSED_DIR) -> str | None:
    """Short hash over the checksum stamps written by ``make data``, None before the first build."""
    stamps = sorted(path for path in processed_dir.glob(DATA_STAMP_GLOB) if path.is_file())
    if not stamps:
        return None
    digest = hashlib.sha256()
    for stamp in stamps:
        digest.update(stamp.name.encode())
        digest.update(stamp.read_bytes())
    return digest.hexdigest()[:12]


def start_run(
    job_type: str,
    config: Mapping[str, Any] | None = None,
    tags: Iterable[str] = (),
    issue: int | None = None,
    group: str | None = None,
    **init_kwargs: Any,
) -> wandb.Run:
    """Start a run in the team project. ``job_type`` is one of ``JOB_TYPES``.

    The run is grouped by hypothesis: ``group`` if given, else ``issue-<n>`` with ``issue`` or the
    number in the branch name, else the branch name. Pass ``issue`` on branches without a number.

    ``WANDB_PROJECT`` overrides the project, e.g. ``sandbox`` for trying things out, and
    ``WANDB_MODE`` switches to offline or disabled. Further keyword arguments (``name``,
    ``notes``, ...) are passed to ``wandb.init``.
    """
    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type {job_type!r} is not one of {JOB_TYPES}")
    if reserved := RESERVED_INIT_ARGS & init_kwargs.keys():
        raise TypeError(
            f"start_run sets {sorted(reserved)} itself, see docs/experiment_tracking.md"
        )
    if overlap := PROVENANCE_KEYS & (config or {}).keys():
        raise ValueError(
            f"config must not contain {sorted(overlap)}: start_run records them itself. "
            "Pass the issue number as issue=..."
        )
    load_env()
    git = git_state()
    parsed = branch_issue(git["git_branch"])
    if issue is None and parsed is not None:
        issue = parsed[1]
    auto_tags = ([parsed[0]] if parsed else []) + ([f"issue-{issue}"] if issue is not None else [])
    return wandb.init(
        entity=resolve_entity(os.environ),
        project=os.environ.get("WANDB_PROJECT", PROJECT),
        job_type=job_type,
        group=group or (f"issue-{issue}" if issue is not None else git["git_branch"]),
        tags=[*auto_tags, *tags],
        config={"data_version": data_version(), "issue": issue, **git, **(config or {})},
        **init_kwargs,
    )


def main() -> None:
    load_env()
    try:
        entity = resolve_entity({**os.environ, "WANDB_MODE": "online"})
        api = wandb.Api()
        viewer = api.viewer
    except TrackingConfigError as err:
        sys.exit(str(err))
    except (
        Exception
    ) as err:  # wandb raises several unrelated types when the login is missing
        sys.exit(
            f"W&B login failed ({type(err).__name__}: {err}). Run `wandb login` first."
        )

    if entity not in viewer.teams:
        sys.exit(
            f"{viewer.username} is not a member of WANDB_ENTITY from .env. Ask the team for an invite."
        )
    print(f"Logged in as {viewer.username}, member of the team entity.")

    project = os.environ.get("WANDB_PROJECT", PROJECT)
    try:
        api.project(project, entity=entity).id
    except ValueError as err:
        if (
            err.__cause__ is not None
        ):  # wandb wraps API failures (network, permission) in ValueError
            sys.exit(f"Could not read project {project}: {err.__cause__}")
        print(
            f"Project {project} does not exist yet. The first run creates it: set it to Private."
        )
    else:
        print(f"Project {project} exists.")


if __name__ == "__main__":
    main()
