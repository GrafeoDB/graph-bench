"""The GitHub workflows: their triggers, jobs and the commands they run."""

import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _load(name: str) -> dict[Any, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _runs(job: dict[str, Any]) -> list[str]:
    """Every shell command of a job, plus the `run` input of action steps (CodSpeed's)."""
    runs = []
    for step in job["steps"]:
        if "run" in step:
            runs.append(step["run"])
        if "run" in step.get("with", {}):
            runs.append(step["with"]["run"])
    return runs


def _uses(job: dict[str, Any]) -> list[str]:
    return [step["uses"] for step in job["steps"] if "uses" in step]


def test_bench_workflow_shape() -> None:
    workflow = _load("bench.yml")
    triggers = _triggers(workflow)
    assert {"push", "schedule", "workflow_dispatch"} <= set(triggers)
    assert triggers["push"]["branches"] == ["main"]

    codspeed = workflow["jobs"]["codspeed"]
    assert codspeed["runs-on"] == "codspeed-macro"
    action = next(step for step in codspeed["steps"] if step.get("uses", "").startswith("CodSpeedHQ/action@"))
    assert action["with"]["mode"] == "walltime"
    assert action["with"]["token"] == "${{ secrets.CODSPEED_TOKEN }}"
    assert "graph-bench codspeed --selection push" in action["with"]["run"]
    runs = _runs(codspeed)
    assert any("graph-bench setup --grafeo grafeo" in run for run in runs)
    assert any("graph-bench data key --selection push" in run for run in runs)
    assert any("graph-bench list --selection push" in run for run in runs)
    assert any(use.startswith("actions/upload-artifact@") for use in _uses(codspeed))

    readme = workflow["jobs"]["readme"]
    assert readme["runs-on"] == "codspeed-macro"
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
        for name in ("ci.yml", "bench.yml")
        for job in _load(name)["jobs"].values()
        for use in _uses(job)
        if use.startswith("astral-sh/setup-uv@")
    ]
    assert pins
    for use in pins:
        assert re.fullmatch(r"astral-sh/setup-uv@v\d+\.\d+\.\d+", use), use
