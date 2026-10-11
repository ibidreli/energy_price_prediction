import subprocess

import pytest

from energy_price import tracking as tr

NO_GIT = {"git_branch": None, "git_commit": None, "git_dirty": None}


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-b", "exp/15-quantile-lgb")
    (tmp_path / "model.py").write_text("x = 1\n")
    git(tmp_path, "add", "model.py")
    git(tmp_path, "commit", "-m", "init")
    return tmp_path


@pytest.fixture
def init_calls(tmp_path, monkeypatch):
    """Capture the arguments of wandb.init instead of starting a run."""
    calls = []
    monkeypatch.setattr(tr, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(tr, "PROCESSED_DIR", tmp_path)
    monkeypatch.setenv("WANDB_ENTITY", "some-team")
    monkeypatch.setattr(tr.wandb, "init", lambda **kwargs: calls.append(kwargs))
    return calls


def on_branch(monkeypatch, branch):
    monkeypatch.setattr(tr, "git_state", lambda: {**NO_GIT, "git_branch": branch})


@pytest.mark.parametrize(
    ("branch", "issue"),
    [
        ("exp/15-quantile-lgb", ("exp", 15)),
        ("eda/3-erste-analyse", ("eda", 3)),
        ("feature/4-reservoir-data", ("feat", 4)),
        ("mannaluca02/exp-15-quantile-lgb", ("exp", 15)),
        ("user/feat/12-pv-features", ("feat", 12)),
        ("fix/7", ("fix", 7)),
        ("eda/zweite-runde", None),
        ("mannaluca02/eda-metrik-definieren-f-r-die-auswertung", None),
        ("mannaluca02/stauseefuellstand-ausgleichspreis-eignung", None),
        ("exp/15quantile", None),
        ("eda/2026-analysis", None),
        ("user/docs-2024-report", None),
        ("main", None),
        (None, None),
    ],
)
def test_branch_issue_reads_the_branch_names_in_use(branch, issue):
    assert tr.branch_issue(branch) == issue


def test_git_state_on_a_clean_branch(repo):
    state = tr.git_state(repo)

    assert state["git_branch"] == "exp/15-quantile-lgb"
    assert len(state["git_commit"]) == 40
    assert state["git_dirty"] is False


@pytest.mark.parametrize("change", ["modify", "new_file"])
def test_git_state_is_dirty_with_uncommitted_changes(repo, change):
    target = repo / ("model.py" if change == "modify" else "new_model.py")
    target.write_text("x = 2\n")

    assert tr.git_state(repo)["git_dirty"] is True


def test_git_state_keeps_the_commit_on_a_detached_head(repo):
    commit = tr.git_state(repo)["git_commit"]
    git(repo, "checkout", "--detach")

    assert tr.git_state(repo) == {"git_branch": None, "git_commit": commit, "git_dirty": False}


def test_git_state_is_empty_outside_a_repository(tmp_path):
    assert tr.git_state(tmp_path) == NO_GIT


@pytest.mark.parametrize("environ", [{"WANDB_ENTITY": ""}, {"WANDB_MODE": "online"}, {"WANDB_MODE": "offline"}])
def test_missing_entity_is_an_error_unless_tracking_is_disabled(environ):
    # Offline too: `wandb sync` would upload the run to the personal account.
    with pytest.raises(tr.TrackingConfigError, match="WANDB_ENTITY"):
        tr.resolve_entity(environ)


def test_missing_entity_is_allowed_when_tracking_is_disabled():
    assert tr.resolve_entity({"WANDB_MODE": "disabled"}) is None


def test_entity_comes_from_the_environment():
    assert tr.resolve_entity({"WANDB_ENTITY": "some-team"}) == "some-team"


def test_load_env_takes_only_the_tracking_keys_and_keeps_set_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("WANDB_ENTITY=from-file\nWANDB_PROJECT=sandbox\nWANDB_API_KEY=should-stay-unused\n")
    environ = {"WANDB_ENTITY": "already-set"}
    monkeypatch.setattr(tr.os, "environ", environ)

    tr.load_env(env)

    assert environ == {"WANDB_ENTITY": "already-set", "WANDB_PROJECT": "sandbox"}


def test_load_env_without_file_changes_nothing(tmp_path, monkeypatch):
    environ = {}
    monkeypatch.setattr(tr.os, "environ", environ)
    tr.load_env(tmp_path / "missing.env")
    assert environ == {}


def test_data_version_is_none_before_make_data(tmp_path):
    assert tr.data_version(tmp_path) is None


def test_data_version_changes_only_with_the_stamp_content(tmp_path):
    (tmp_path / ".raw_checksums").write_text("aaa  data/ausgleichpreis/2026/01.xml\n")
    first = tr.data_version(tmp_path)
    (tmp_path / "unrelated.txt").write_text("ignored")
    (tmp_path / ".raw_checksums.tmp").write_text("half-written by make")

    assert tr.data_version(tmp_path) == first
    (tmp_path / ".cab_checksums").write_text("bbb  data/control_area_balance/x.csv\n")
    assert tr.data_version(tmp_path) != first


def test_data_version_includes_new_sources_like_reservoirs(tmp_path):
    (tmp_path / ".raw_checksums").write_text("aaa  data/ausgleichpreis/2026/01.xml\n")
    before = tr.data_version(tmp_path)
    (tmp_path / ".reservoir_checksums").write_text("ccc  data/reservoirs/snapshots/x.json\n")

    assert tr.data_version(tmp_path) != before


def test_run_is_grouped_by_the_issue_in_the_branch(init_calls, monkeypatch):
    on_branch(monkeypatch, "feature/4-reservoir-data")
    tr.start_run("train", config={"model": "ridge"})

    (call,) = init_calls
    assert call["group"] == "issue-4"
    assert call["tags"] == ["feat", "issue-4"]
    assert call["config"]["issue"] == 4
    assert call["config"]["git_branch"] == "feature/4-reservoir-data"
    assert call["config"]["model"] == "ridge"
    assert call["entity"] == "some-team"


def test_explicit_issue_wins_over_a_branch_without_number(init_calls, monkeypatch):
    on_branch(monkeypatch, "mannaluca02/stauseefuellstand-ausgleichspreis-eignung")
    tr.start_run("eda", issue=9, tags=["reservoir"])

    assert init_calls[0]["group"] == "issue-9"
    assert init_calls[0]["tags"] == ["issue-9", "reservoir"]


def test_explicit_group_wins(init_calls, monkeypatch):
    on_branch(monkeypatch, "exp/15-quantile-lgb")
    tr.start_run("sweep", group="lgbm-sweep-1")

    assert init_calls[0]["group"] == "lgbm-sweep-1"
    assert init_calls[0]["tags"] == ["exp", "issue-15"]


@pytest.mark.parametrize(("branch", "group"), [("eda/zweite-runde", "eda/zweite-runde"), (None, None)])
def test_without_issue_the_group_falls_back_to_the_branch(init_calls, monkeypatch, branch, group):
    on_branch(monkeypatch, branch)
    tr.start_run("eda")

    assert init_calls[0]["group"] == group
    assert init_calls[0]["config"]["issue"] is None


def test_start_run_without_network_keeps_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    with tr.start_run("train", config={"alpha": 0.1}) as run:
        assert run.config["alpha"] == 0.1
        assert "data_version" in run.config
        assert "git_commit" in run.config


def test_start_run_refuses_to_log_online_without_entity(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    with pytest.raises(tr.TrackingConfigError):
        tr.start_run("train")


def test_start_run_rejects_an_unknown_job_type():
    with pytest.raises(ValueError, match="trian"):
        tr.start_run("trian")


@pytest.mark.parametrize("key", ["issue", "git_commit", "git_dirty", "data_version"])
def test_config_must_not_override_the_provenance(key):
    # e.g. a config copied from an old run would carry a stale git_commit
    with pytest.raises(ValueError, match=key):
        tr.start_run("train", config={key: "stale", "model": "ridge"})


@pytest.mark.parametrize("key", ["mode", "entity", "project"])
def test_start_run_rejects_arguments_it_sets_itself(key):
    # mode="online" would otherwise bypass the entity check made with WANDB_MODE=disabled
    with pytest.raises(TypeError, match=key):
        tr.start_run("train", **{key: "x"})
