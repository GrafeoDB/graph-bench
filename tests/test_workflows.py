"""The GitHub workflows: their triggers, jobs and the commands they run."""

import json
import re
from pathlib import Path
from typing import Any

import yaml

# CodSpeed's Graviton (arm64) macro runner, the label the plain `codspeed-macro` used to alias.
GRAVITON = "codspeed-macro-arm64-graviton-ubuntu-22-04"
SCRIPTS = Path(__file__).resolve().parent.parent / ".github" / "scripts"
WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
ACTIONS = Path(__file__).resolve().parent.parent / ".github" / "actions"


def _load(name: str) -> dict[Any, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """The steps of a job, with each `uses: ./.github/actions/NAME` replaced by that composite action's steps.

    The composite's `${{ inputs.selection }}` is replaced by the `selection` the job passes.
    """
    steps = []
    for step in job["steps"]:
        uses = step.get("uses", "")
        if not uses.startswith("./.github/actions/"):
            steps.append(step)
            continue
        action = yaml.safe_load((ACTIONS / uses.rsplit("/", 1)[1] / "action.yml").read_text(encoding="utf-8"))
        selection = step.get("with", {}).get("selection", "")
        for inner in action["runs"]["steps"]:
            steps.append(json.loads(json.dumps(inner).replace("${{ inputs.selection }}", selection)))
    return steps


def _runs(job: dict[str, Any]) -> list[str]:
    """Every shell command of a job, plus the `run` input of action steps (CodSpeed's)."""
    runs = []
    for step in _steps(job):
        if "run" in step:
            runs.append(step["run"])
        if "run" in step.get("with", {}):
            runs.append(step["with"]["run"])
    return runs


def _uses(job: dict[str, Any]) -> list[str]:
    return [step["uses"] for step in _steps(job) if "uses" in step]


def test_bench_workflow_shape() -> None:
    workflow = _load("bench.yml")
    triggers = _triggers(workflow)
    assert {"push", "schedule", "workflow_dispatch"} <= set(triggers)
    assert triggers["push"]["branches"] == ["main"]

    codspeed = workflow["jobs"]["codspeed"]
    assert codspeed["runs-on"] == GRAVITON
    action = next(step for step in codspeed["steps"] if step.get("uses", "").startswith("CodSpeedHQ/action@"))
    assert action["with"]["mode"] == "walltime"
    # CodSpeed authenticates with GitHub OIDC: the job may mint an id token, and no token secret is passed.
    assert codspeed["permissions"]["id-token"] == "write"
    assert codspeed["permissions"]["contents"] == "read"
    assert "token" not in action["with"]
    assert "graph-bench codspeed --selection push" in action["with"]["run"]
    runs = _runs(codspeed)
    assert any("graph-bench data key --selection push" in run for run in runs)
    assert any("graph-bench list --selection push" in run for run in runs)
    assert any(use.startswith("actions/upload-artifact@") for use in _uses(codspeed))

    readme = workflow["jobs"]["readme"]
    assert readme["runs-on"] == GRAVITON
    condition = readme["if"]
    assert "schedule" in condition
    assert "workflow_dispatch" in condition
    assert "push" not in condition
    runs = _runs(readme)
    assert any("graph-bench bench --selection full" in run for run in runs)
    assert any("graph-bench report --format readme" in run and "--write README.md" in run for run in runs)
    pr = next(step for step in readme["steps"] if step.get("uses", "").startswith("peter-evans/create-pull-request@"))
    assert pr["with"]["branch"] == "bot/results"
    assert pr["with"]["title"] == "Update benchmark results"


def test_bench_workflow_measures_released_databases() -> None:
    workflow = _load("bench.yml")
    for name, job in workflow["jobs"].items():
        runs = _runs(job)
        # Grafeo is measured as its latest PyPI release, not built from main.
        assert not any("setup --grafeo" in run for run in runs), name
        assert not any(step.get("with", {}).get("repository") == "GrafeoDB/grafeo" for step in job["steps"]), name
        assert not any(use.startswith(("dtolnay/rust-toolchain@", "Swatinem/rust-cache@")) for use in _uses(job)), name
        # The locked sync takes the non-database extras only; the database packages then install one at a time.
        sync = next(i for i, run in enumerate(runs) if "uv sync --frozen" in run)
        assert "--extra embedded" not in runs[sync], name
        install = next(i for i, run in enumerate(runs) if "install-databases.sh" in run)
        assert sync < install, name
        assert not any("--pre" in run or "--prerelease" in run for run in runs), name
        assert not any("uv lock" in run or "uv sync --upgrade" in run for run in runs), name


def test_bench_jobs_run_on_graviton_with_the_arch_in_cache_keys() -> None:
    workflow = _load("bench.yml")
    for name, job in workflow["jobs"].items():
        assert job["runs-on"] == GRAVITON, name
        caches = [step for step in _steps(job) if step.get("uses", "").startswith("actions/cache@")]
        assert caches, name
        for step in caches:
            assert "${{ runner.arch }}" in step["with"]["key"], name
    action = next(
        step["uses"] for step in workflow["jobs"]["codspeed"]["steps"] if step.get("uses", "").startswith("CodSpeedHQ/")
    )
    assert tuple(int(p) for p in action.split("@v")[1].split(".")) >= (3,)  # walltime mode needs >= 3.1.0


def test_install_script_goes_on_when_a_package_fails() -> None:
    script = (SCRIPTS / "install-databases.sh").read_text(encoding="utf-8")
    # One package at a time, from `graph-bench packages`; a failure is logged with the installer's reason and the
    # script still exits 0, so the run reports that database as failed instead of the job stopping.
    assert "graph-bench packages" in script
    assert 'uv pip install --upgrade --only-binary :all: "$requirement"' in script
    assert "install-failures.txt" in script
    assert "::warning" in script
    assert "set -e" not in script
    assert script.rstrip().endswith("exit 0")
    assert "--pre" not in script


def test_pins_workflow_opens_a_pull_request() -> None:
    workflow = _load("pins.yml")
    triggers = _triggers(workflow)
    assert {"schedule", "workflow_dispatch"} <= set(triggers)
    assert triggers["schedule"][0]["cron"]
    job = workflow["jobs"]["pins"]
    assert job["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert any("graph_bench.pins_update --write" in run for run in _runs(job))
    pr = next(step for step in job["steps"] if step.get("uses", "").startswith("peter-evans/create-pull-request@"))
    assert pr["with"]["branch"] == "bot/pins"
    assert pr["with"]["add-paths"] == "graph_bench/databases/pins.py"


def test_ci_workflow_shape() -> None:
    workflow = _load("ci.yml")
    assert {"push", "pull_request"} <= set(_triggers(workflow))
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
        assert job["strategy"]["matrix"]["python-version"] == ["3.12", "3.13"]
    runs = [run for job in workflow["jobs"].values() for run in _runs(job)]
    assert any("uv sync --frozen --extra embedded --extra bench --extra dev" in run for run in runs)
    assert any("ruff check ." in run for run in runs)
    assert any("ruff format --check ." in run for run in runs)
    assert any("pytest" in run for run in runs)


def test_setup_uv_pinned_to_full_version() -> None:
    # astral-sh/setup-uv publishes no floating major tag (no `v10`), so a major pin fails at "Set up job".
    pins = [
        use
        for name in ("ci.yml", "bench.yml", "pins.yml", "release.yml")
        for job in _load(name)["jobs"].values()
        for use in _uses(job)
        if use.startswith("astral-sh/setup-uv@")
    ]
    assert pins
    for use in pins:
        assert re.fullmatch(r"astral-sh/setup-uv@v\d+\.\d+\.\d+", use), use


def test_release_workflow_triggers_and_state() -> None:
    workflow = _load("release.yml")
    triggers = _triggers(workflow)
    assert {"schedule", "workflow_dispatch"} <= set(triggers)
    assert triggers["schedule"][0]["cron"]
    assert triggers["workflow_dispatch"]["inputs"]["force"]["type"] == "boolean"
    detect, post = workflow["jobs"]["detect"], workflow["jobs"]["post"]
    assert any("graph_bench.release latest" in run for run in _runs(detect))
    assert any("FORCE" in run and "bot/state" in run for run in _runs(detect))
    assert post["needs"] == "detect"
    assert post["if"] == "needs.detect.outputs.run == 'true'"
    assert post["runs-on"] == GRAVITON
    assert post["permissions"]["contents"] == "write"
    assert any("graph-bench bench --selection full" in run for run in _runs(post))
    # Shared steps: the same composite action as bench.yml's README job, not a copy.
    assert "./.github/actions/prepare" in [step.get("uses") for step in post["steps"]]
    assert any("install-databases.sh" in run for run in _runs(post))
    assert not any("--pre" in run for run in _runs(post))


def test_release_post_is_conditional_on_the_secret_and_records_after() -> None:
    post = _load("release.yml")["jobs"]["post"]
    steps = post["steps"]
    names = [step.get("name", "") for step in steps]
    notify = steps[names.index("Post to Discord")]
    record = steps[names.index("Record the posted version on bot/state")]
    assert notify["env"] == {"DISCORD_BENCH_WEBHOOK": "${{ secrets.DISCORD_BENCH_WEBHOOK }}"}
    assert "--webhook-env DISCORD_BENCH_WEBHOOK" in notify["run"]
    assert "graph-bench notify" in notify["run"]
    assert notify["if"] == "needs.detect.outputs.webhook == 'true'"
    assert record["if"] == notify["if"]
    # The version is recorded after a successful post (a step without `if: always()` does not run after a failure).
    assert names.index("Record the posted version on bot/state") > names.index("Post to Discord")
    assert "always()" not in record["if"]
    assert "git push origin bot/state" in record["run"]
    # The secret is only ever read from the secrets context, never written into a file.
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "discord.com/api/webhooks" not in text
    detect = _load("release.yml")["jobs"]["detect"]
    gate = next(step for step in detect["steps"] if step.get("id") == "webhook")
    assert gate["env"]["DISCORD_BENCH_WEBHOOK"] == "${{ secrets.DISCORD_BENCH_WEBHOOK }}"


def test_prepare_action_installs_released_databases() -> None:
    action = yaml.safe_load((ACTIONS / "prepare" / "action.yml").read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    runs = [step["run"] for step in action["runs"]["steps"] if "run" in step]
    assert all(step["shell"] == "bash" for step in action["runs"]["steps"] if "run" in step)
    assert "uv sync --frozen --extra bench" in runs[0]
    assert any("install-databases.sh" in run for run in runs)
