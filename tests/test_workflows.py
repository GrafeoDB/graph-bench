"""The GitHub workflows: their triggers, jobs and the commands they run."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from graph_bench.suites import catalog

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


# The jobs on the macro runner, one leg per roster database, and the job on ubuntu-latest that merges their
# results, with the artifact prefix of the legs and the merged artifact's name.
MATRIX_JOBS = [
    ("bench.yml", "codspeed", "codspeed-results", "results-push"),
    ("bench.yml", "readme", "readme-results", "results-full"),
    ("release.yml", "post", "post-results", "results-release"),
]


def _macro_jobs(name: str) -> dict[str, dict[str, Any]]:
    jobs = _load(name)["jobs"].items()
    return {n: job for n, job in jobs if str(job.get("runs-on", "")).startswith("codspeed-macro")}


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
    assert "--db ${{ matrix.db }}" in action["with"]["run"]
    assert any("graph-bench data key --selection push" in run for run in runs)
    assert any("graph-bench list --selection push" in run for run in runs)
    assert any(use.startswith("actions/upload-artifact@") for use in _uses(codspeed))

    readme = workflow["jobs"]["readme"]
    assert readme["runs-on"] == GRAVITON
    # On schedule and dispatch the full legs run after the push legs, whatever their outcome, unless cancelled.
    assert readme["needs"] == ["resolve", "codspeed"]
    condition = readme["if"]
    assert "schedule" in condition
    assert "workflow_dispatch" in condition
    assert "push" not in condition
    assert "!cancelled()" in condition
    runs = _runs(readme)
    assert any("graph-bench bench --selection full --db ${{ matrix.db }}" in run for run in runs)
    # The README section and the pull request come from the merged results, on ubuntu-latest.
    assert not any("graph-bench report" in run for run in runs)
    merged = workflow["jobs"]["readme-results"]
    events = "(github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')"
    assert merged["if"] == f"${{{{ !cancelled() && needs.readme.result != 'skipped' && {events} }}}}"
    assert merged["permissions"] == {"contents": "write", "pull-requests": "write"}
    runs = _runs(merged)
    report = next(i for i, run in enumerate(runs) if "graph-bench report --format readme" in run)
    assert "--in merged/results.json" in runs[report] and "--write README.md" in runs[report]
    assert report > next(i for i, run in enumerate(runs) if "graph-bench merge" in run)
    pr = next(step for step in merged["steps"] if step.get("uses", "").startswith("peter-evans/create-pull-request@"))
    assert pr["with"]["branch"] == "bot/results"
    assert pr["with"]["title"] == "Update benchmark results"
    assert pr["with"]["add-paths"].split() == ["README.md", "docs/results.md"]
    # The overview page is written next to the README section, from the same merged results.
    overview = next(i for i, run in enumerate(runs) if "graph-bench report --format overview" in run)
    assert "--in merged/results.json" in runs[overview] and "--write docs/results.md" in runs[overview]


def test_bench_workflow_measures_released_databases() -> None:
    for name, job in _macro_jobs("bench.yml").items():
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
    assert set(_macro_jobs("bench.yml")) == {"codspeed", "readme"}
    for name, job in _macro_jobs("bench.yml").items():
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
    check = next(run for run in _runs(detect) if "FORCE" in run)
    assert "bot/state" in check
    # Numeric comparison through graph_bench.release, never string equality; a failed lookup stops the job.
    assert "graph_bench.release newer" in check
    assert '"$latest" = "$posted"' not in check
    assert "sort -V" not in check
    assert "ls-remote --exit-code" in check
    assert "2>/dev/null" not in check
    assert post["needs"] == "detect"
    assert post["if"] == "needs.detect.outputs.run == 'true'"
    assert post["runs-on"] == GRAVITON
    assert post["permissions"]["contents"] == "write"
    assert any("graph-bench bench --selection full --db ${{ matrix.db }}" in run for run in _runs(post))
    # Shared steps: the same composite action as bench.yml's README job, not a copy.
    assert "./.github/actions/prepare" in [step.get("uses") for step in post["steps"]]
    assert any("install-databases.sh" in run for run in _runs(post))
    assert not any("--pre" in run for run in _runs(post))


def test_release_post_is_conditional_on_the_secret_and_records_after() -> None:
    """The merging job posts and records; the macro-runner legs only benchmark and never see the secret."""
    jobs = _load("release.yml")["jobs"]
    assert not any("secrets." in json.dumps(step) for step in jobs["post"]["steps"])
    assert not any("graph-bench notify" in run for run in _runs(jobs["post"]))
    post = jobs["post-results"]
    assert post["needs"] == ["detect", "post"]
    # A cancelled run posts and records nothing.
    assert post["if"] == "${{ !cancelled() && needs.post.result != 'skipped' && needs.detect.outputs.run == 'true' }}"
    assert post["runs-on"] == "ubuntu-latest"
    assert post["permissions"] == {"contents": "write"}
    steps = post["steps"]
    names = [step.get("name", "") for step in steps]
    notify = steps[names.index("Post to Discord")]
    record = steps[names.index("Record the posted version on bot/state")]
    assert notify["env"] == {"DISCORD_BENCH_WEBHOOK": "${{ secrets.DISCORD_BENCH_WEBHOOK }}"}
    assert "--webhook-env DISCORD_BENCH_WEBHOOK" in notify["run"]
    assert "graph-bench notify" in notify["run"]
    assert "--in merged/results.json" in notify["run"]
    assert "../merged/results.json" in record["run"]
    assert names.index("Merge the results") < names.index("Post to Discord")
    # Both need the secret and Grafeo's results (see test_release_post_needs_grafeo_results).
    assert notify["if"] == "needs.detect.outputs.webhook == 'true' && steps.grafeo.outputs.ran == 'true'"
    assert record["if"] == notify["if"]
    # The version is recorded after a successful post (a step without `if: always()` does not run after a failure).
    assert names.index("Record the posted version on bot/state") > names.index("Post to Discord")
    assert "always()" not in record["if"]
    # The push comes from a checkout of bot/state of its own (not a worktree), and a failed push fails the step.
    assert record["working-directory"] == "state"
    assert "git push origin HEAD:refs/heads/bot/state" in record["run"]
    assert "exit 1" in record["run"].split("git push", 1)[1]
    assert "worktree" not in record["run"]
    checkouts = [s for s in steps if s.get("uses", "").startswith("actions/checkout@") and "with" in s]
    assert {c["with"]["path"] for c in checkouts} == {"state"}
    assert any(c["with"].get("ref") == "bot/state" for c in checkouts)
    # Retried posting resumes from the recorded progress.
    assert "--progress notify-progress.txt" in notify["run"]
    # The secret is only ever read from the secrets context, never written into a file.
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "discord.com/api/webhooks" not in text
    detect = _load("release.yml")["jobs"]["detect"]
    gate = next(step for step in detect["steps"] if step.get("id") == "webhook")
    assert gate["env"]["DISCORD_BENCH_WEBHOOK"] == "${{ secrets.DISCORD_BENCH_WEBHOOK }}"


def test_release_post_needs_grafeo_results(tmp_path: Path) -> None:
    """Without Grafeo in the merged results (its leg left none, or Grafeo did not start), the release run warns,
    posts nothing and records nothing, so the next daily run benchmarks that release again. The check is the
    step's own Python, run here against a merged results.json with and without Grafeo."""
    steps = _load("release.yml")["jobs"]["post-results"]["steps"]
    names = [step.get("name", "") for step in steps]
    check = steps[names.index("Did Grafeo run?")]
    assert check["id"] == "grafeo"
    assert names.index("Merge the results") < names.index("Did Grafeo run?") < names.index("Post to Discord")
    assert "::warning" in check["run"] and 'echo "ran=false"' in check["run"] and 'echo "ran=true"' in check["run"]
    code = re.search(r'python3 -c "(.+?)"; then', check["run"])
    assert code, check["run"]
    (tmp_path / "merged").mkdir()
    for databases, ran in (({"neug": {}, "grafeo": {}}, 0), ({"neug": {}}, 1)):
        (tmp_path / "merged" / "results.json").write_text(json.dumps({"databases": databases}), encoding="utf-8")
        done = subprocess.run([sys.executable, "-c", code.group(1)], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == ran, (databases, done.stderr)


def test_every_leg_installs_the_grafeo_version_of_the_run() -> None:
    """The grafeo version is resolved once per run (bench.yml's `resolve` job, release.yml's `detect`) and every
    leg's prepare step installs exactly it, so a grafeo release during the run cannot split the legs (merge would
    refuse them). Without the input the install script keeps installing the newest stable grafeo."""
    bench = _load("bench.yml")["jobs"]
    resolve = bench["resolve"]
    assert resolve["runs-on"] == "ubuntu-latest"
    assert resolve["outputs"] == {"grafeo": "${{ steps.grafeo.outputs.version }}"}
    lookup = next(step for step in resolve["steps"] if step.get("id") == "grafeo")
    assert "python -m graph_bench.release latest" in lookup["run"] and lookup["env"] == {"PYTHONPATH": "."}
    pinned = {
        ("bench.yml", "codspeed"): "${{ needs.resolve.outputs.grafeo }}",
        ("bench.yml", "readme"): "${{ needs.resolve.outputs.grafeo }}",
        ("release.yml", "post"): "${{ needs.detect.outputs.version }}",
    }
    for (name, job_name), version in pinned.items():
        job = _load(name)["jobs"][job_name]
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert version.split(".")[1] in needs, (name, job_name)
        prepare = next(step for step in job["steps"] if step.get("uses") == "./.github/actions/prepare")
        assert prepare["with"]["grafeo-version"] == version, (name, job_name)
    assert "needs.resolve.result == 'success'" in bench["readme"]["if"]
    action = yaml.safe_load((ACTIONS / "prepare" / "action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["grafeo-version"] == {
        "description": action["inputs"]["grafeo-version"]["description"],
        "required": False,
        "default": "",
    }
    install = next(step for step in action["runs"]["steps"] if "install-databases.sh" in step.get("run", ""))
    assert install["env"] == {"GRAFEO_VERSION": "${{ inputs.grafeo-version }}"}
    script = (SCRIPTS / "install-databases.sh").read_text(encoding="utf-8")
    assert 'if [ -n "${GRAFEO_VERSION:-}" ] && [ "${requirement%% *}" = "grafeo" ]; then' in script
    assert 'requirement="grafeo==$GRAFEO_VERSION${requirement#grafeo}"' in script


@pytest.mark.skipif(sys.platform != "linux", reason="runs install-databases.sh's pin line in Linux bash")
def test_install_script_pins_grafeo() -> None:
    """The pin line of install-databases.sh turns `grafeo` into `grafeo==<version>` and leaves the rest alone."""
    script = (SCRIPTS / "install-databases.sh").read_text(encoding="utf-8")
    start = script.index('  if [ -n "${GRAFEO_VERSION:-}" ]')
    pin = script[start : script.index("  fi\n", start) + len("  fi\n")]
    probe = f'for requirement in grafeo grafeo-x duckdb; do\n{pin}  echo "$requirement"\ndone\n'
    env = {**os.environ, "GRAFEO_VERSION": "0.6.4"}
    done = subprocess.run(["bash", "-c", probe], env=env, capture_output=True, text=True)
    assert done.stdout.split() == ["grafeo==0.6.4", "grafeo-x", "duckdb"], done.stderr
    env.pop("GRAFEO_VERSION")
    done = subprocess.run(["bash", "-c", probe], env=env, capture_output=True, text=True)
    assert done.stdout.split() == ["grafeo", "grafeo-x", "duckdb"], done.stderr


def test_prepare_action_installs_released_databases() -> None:
    action = yaml.safe_load((ACTIONS / "prepare" / "action.yml").read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    runs = [step["run"] for step in action["runs"]["steps"] if "run" in step]
    assert all(step["shell"] == "bash" for step in action["runs"]["steps"] if "run" in step)
    assert "uv sync --frozen --extra bench" in runs[0]
    assert any("install-databases.sh" in run for run in runs)


def test_macro_runner_jobs_never_run_at_once() -> None:
    """One macro-runner job at a time, across bench.yml and release.yml: one concurrency group at workflow level on
    both (a job-level group keeps one pending job and cancels older pending ones, so matrix legs would cancel each
    other), no job-level groups, and every macro-runner matrix runs one leg at a time."""
    for name in ("bench.yml", "release.yml"):
        workflow = _load(name)
        assert workflow["concurrency"] == {"group": "macro-runner", "cancel-in-progress": False}, name
        for job_name, job in workflow["jobs"].items():
            assert "concurrency" not in job, (name, job_name)
        for job_name, job in _macro_jobs(name).items():
            assert job["strategy"]["max-parallel"] == 1, (name, job_name)
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "release-post" not in text


def test_macro_runner_jobs_are_one_leg_per_roster_database() -> None:
    """Each macro-runner job is a matrix over `catalog.CI_DATABASES`, in that order, that goes on when a leg
    fails. Every leg passes the run deadline from `graph-bench ci-settings` (never a number in the YAML), takes
    the results of the legs before it in the roster as `--yardstick` (`graph-bench ci-yardsticks`, so a re-run or
    an out-of-order leg never takes a later one; none is fine) and uploads `<prefix>-<db>`."""
    found = []
    for name, job_name, _, prefix in MATRIX_JOBS:
        job = _load(name)["jobs"][job_name]
        strategy = job["strategy"]
        assert strategy["matrix"] == {"db": list(catalog.CI_DATABASES)}, (name, job_name)
        assert strategy["fail-fast"] is False, (name, job_name)
        steps = _steps(job)
        settings = next(step for step in steps if step.get("id") == "ci")
        assert 'graph-bench ci-settings >> "$GITHUB_OUTPUT"' in settings["run"]
        download = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/download-"))
        assert steps[download]["with"] == {"pattern": f"{prefix}-*", "path": "yardstick"}, (name, job_name)
        assert steps[download].get("continue-on-error") is True
        args = next(step for step in steps if step.get("id") == "yardstick")
        assert "args=$(uv run --no-sync graph-bench ci-yardsticks --db ${{ matrix.db }} yardstick)" in args["run"]
        assert 'echo "args=$args" >> "$GITHUB_OUTPUT"' in args["run"]
        assert "yardstick/*/results.json" not in args["run"], (name, job_name)
        command = next(run for run in _runs(job) if "--selection" in run and "--out out" in run)
        assert "--db ${{ matrix.db }}" in command
        assert "${{ steps.yardstick.outputs.args }}" in command
        runs_at = next(
            i for i, step in enumerate(steps) if command in (step.get("run"), step.get("with", {}).get("run"))
        )
        assert download < runs_at, (name, job_name)
        # The deadline counts from the job's first step: the start epoch is recorded before the checkout, and the
        # step right before the bench passes what is left of RUN_DEADLINE_S, never below 0 (a negative value would
        # exit 2 with no results.json; 0 writes one with every case at "run deadline").
        start = job["steps"][0]
        assert start["id"] == "start" and start["run"] == 'echo "epoch=$(date +%s)" >> "$GITHUB_OUTPUT"'
        left = steps[runs_at - 1]
        assert left["id"] == "deadline", (name, job_name)
        assert "${{ steps.ci.outputs.deadline_s }} - ($(date +%s) - ${{ steps.start.outputs.epoch }})" in left["run"]
        assert 'if [ "$left" -lt 0 ]; then left=0; fi' in left["run"]
        assert 'echo "seconds=$left" >> "$GITHUB_OUTPUT"' in left["run"]
        assert "--deadline-s ${{ steps.deadline.outputs.seconds }}" in command
        assert "steps.ci.outputs.deadline_s" not in command
        upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
        assert upload["with"]["name"] == prefix + "-${{ matrix.db }}"
        assert upload["if"] == "always()"
        # "Re-run failed jobs" uploads the same name again in the same run.
        assert upload["with"]["overwrite"] is True
        found.append((name, job_name))
    for name in ("bench.yml", "release.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "2880" not in text and "48 * 60" not in text, name
    macro = {(name, job) for name in ("bench.yml", "release.yml") for job in _macro_jobs(name)}
    assert sorted(found) == sorted(macro)


def test_merge_jobs_run_on_ubuntu_after_every_leg() -> None:
    """A job on ubuntu-latest per matrix runs after all its legs, whatever their outcome, and merges the legs'
    results (`graph-bench merge` with the roster as `--db`, so a leg without results shows as "no results");
    with no results at all it fails instead of merging nothing. It is skipped with its legs (the grafeo lookup
    failed), so it never fails for lack of results there."""
    for name, job_name, merge_name, prefix in MATRIX_JOBS:
        job = _load(name)["jobs"][merge_name]
        assert job["runs-on"] == "ubuntu-latest", merge_name
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        assert job_name in needs, merge_name
        # Whatever the legs' outcome, but not after a cancel (no partial PR, post or record).
        assert job["if"].startswith(f"${{{{ !cancelled() && needs.{job_name}.result != 'skipped'"), merge_name
        assert "always()" not in job["if"], merge_name
        steps = _steps(job)
        download = next(step for step in steps if step.get("uses", "").startswith("actions/download-artifact@"))
        # Not into results/: the committed local snapshot, results/local/results.json, would match the glob.
        assert download["with"] == {"pattern": f"{prefix}-*", "path": "legs"}, merge_name
        merge = next(run for run in _runs(job) if "graph-bench merge" in run)
        assert 'graph-bench merge --out merged --db "${{ steps.ci.outputs.databases }}"' in merge, merge_name
        assert "legs/*/results.json" in merge and "::error" in merge and "exit 1" in merge, merge_name
        assert "results/*/results.json" not in merge, merge_name
        assert any('graph-bench ci-settings >> "$GITHUB_OUTPUT"' in run for run in _runs(job)), merge_name
        # The merge needs graph-bench only, not the databases or the datasets.
        assert not any("install-databases.sh" in run or "data prepare" in run for run in _runs(job)), merge_name
        upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
        expected = {"name": prefix, "path": "merged/results.json", "if-no-files-found": "warn", "overwrite": True}
        assert upload["with"] == expected, merge_name


def test_readme_results_mixes_in_the_local_snapshot() -> None:
    """The README job merges the local snapshot when it is committed (and only then), and checks its versions
    against the newest stable releases: the databases that differ become a `::notice`, and the check never fails
    the job."""
    job = _load("bench.yml")["jobs"]["readme-results"]
    merge = next(run for run in _runs(job) if "graph-bench merge" in run)
    assert "if [ -f results/local/results.json ]" in merge
    assert "--local results/local/results.json" in merge
    steps = _steps(job)
    check_at = next(i for i, step in enumerate(steps) if "graph-bench local-check" in step.get("run", ""))
    check = steps[check_at]
    assert "graph-bench local-check --local results/local/results.json" in check["run"]
    assert "if [ ! -f results/local/results.json ]" in check["run"]
    assert "::notice title=Local snapshot out of date::" in check["run"]
    assert check.get("continue-on-error") is True
    pr_at = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("peter-evans/"))
    merge_at = next(i for i, step in enumerate(steps) if "graph-bench merge" in step.get("run", ""))
    assert merge_at < check_at < pr_at
    # The other merge jobs never mix in the snapshot.
    for name, _, merge_name, _ in MATRIX_JOBS:
        if merge_name != "readme-results":
            assert not any("--local" in run for run in _runs(_load(name)["jobs"][merge_name])), merge_name


def test_local_bench_script_runs_all_ten_in_one_linux_container() -> None:
    """local-bench.sh: one ubuntu:24.04 container, the repo mounted read-only and copied in, the full selection
    over every built-in database (no --db), the output kept in the container and copied out at the end (a Windows
    bind mount fails appends with ENODATA), to results/local/results.json, which git keeps."""
    script = (SCRIPTS / "local-bench.sh").read_text(encoding="utf-8")
    assert "ubuntu:24.04" in script
    assert ":/src:ro" in script
    assert "graph-bench bench --selection full --out /work/out" in script
    assert "--db" not in script
    assert "install-databases.sh" in script
    assert " cp " in script and "results/local/results.json" in script
    assert "/work/out/results.json" in script
    # A failed run still copies out what it left, before the EXIT trap removes the container, and keeps its status.
    assert '"$IMAGE" bash -c "$inside" || status=$?' in script
    assert script.index("|| status=$?") < script.index("$DOCKER cp") < script.rindex('exit "$status"')
    # ...but never over the committed snapshot: a failed run's file is results.partial.json.
    assert '[ "$status" -eq 0 ] || target=results/local/results.partial.json' in script
    assert "target=results/local/results.json" in script
    assert '$DOCKER cp "$NAME:/work/out/results.json" "$target"' in script
    assert script.index("target=results/local/results.json") < script.index("|| target=results/local/results.partial")
    assert script.index("|| target=results/local/results.partial") < script.index("$DOCKER cp")
    ignore = Path(__file__).resolve().parent.parent / ".gitignore"
    lines = ignore.read_text(encoding="utf-8").splitlines()
    assert lines.index("results/*") < lines.index("!results/local/") < lines.index("!results/local/results.json")
    if shutil.which("bash") is not None:
        # A relative path, so a Windows bash (Git Bash or WSL) reads it too.
        done = subprocess.run(["bash", "-n", "local-bench.sh"], cwd=SCRIPTS, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr


def test_macro_runner_jobs_end_before_the_runner_does() -> None:
    """The Graviton macro runner dies about 60 minutes after boot (Task T): every job on it has a 55-minute
    timeout, so GitHub cancels an overlong run with its log kept instead of the runner vanishing with it."""
    found = []
    for name in ("bench.yml", "release.yml"):
        for job_name, job in _macro_jobs(name).items():
            assert job.get("timeout-minutes") == 55, (name, job_name)
            found.append((name, job_name))
    assert sorted(found) == [("bench.yml", "codspeed"), ("bench.yml", "readme"), ("release.yml", "post")]


def test_full_runs_start_the_heartbeat() -> None:
    """Both full-selection jobs run the bench next to heartbeat.sh, keep the bench's exit status through `tee`
    and can push (the heartbeat writes bot/heartbeat with the job token)."""
    jobs = [_load("bench.yml")["jobs"]["readme"], _load("release.yml")["jobs"]["post"]]
    for job in jobs:
        bench = next(run for run in _runs(job) if "graph-bench bench --selection full" in run)
        assert "bash .github/scripts/heartbeat.sh out" in bench
        assert "set -o pipefail" in bench
        assert 'kill "$heartbeat"' in bench
        assert job["permissions"]["contents"] == "write"
    script = (SCRIPTS / "heartbeat.sh").read_text(encoding="utf-8")
    # It must never fail the job: no `set -e`, and push errors are reported, not raised.
    assert "set -e" not in script
    assert "bot/heartbeat" in script


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("git") is None, reason="heartbeat.sh reads Linux /proc")
def test_heartbeat_pushes_one_beat(tmp_path: Path) -> None:
    """One beat lands on bot/heartbeat of the remote: the time line, memory, each worker's last line and the bench
    output's tail."""
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)
    subprocess.run(["git", "init", "--quiet", str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(remote)], check=True)
    worker = work / "out" / "raw" / "grafeo-1" / "worker.jsonl"
    worker.parent.mkdir(parents=True)
    worker.write_text('{"event": "start"}\n{"event": "case", "id": "hops/hop_2"}\n', encoding="utf-8")
    log = tmp_path / "bench.log"
    log.write_text("line 1\nloading snb-medium\n", encoding="utf-8")
    ps = tmp_path / "ps.txt"
    jit = "eyIucnVubmVyIjoiZXlKQloyVnVkRWxrSWpvaU1UQTNPVEVpTENKQloy"
    ps.write_text(
        f"110592 07:37 /actions-runner/bin/Runner.Listener run --jitconfig {jit}\n"
        "225280 06:08 python -m graph_bench.runner.worker --db grafeo\n"
        "5120 00:01 tool --token=hunter2 --other\n",
        encoding="utf-8",
    )
    env = {**os.environ, "HEARTBEAT_ONCE": "1", "GITHUB_RUN_ID": "42", "HEARTBEAT_PS": str(ps)}
    done = subprocess.run(
        ["bash", str(SCRIPTS / "heartbeat.sh"), "out", str(log)], cwd=work, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    shown = subprocess.run(
        ["git", "--git-dir", str(remote), "show", "bot/heartbeat:heartbeat.txt"], capture_output=True, text=True
    )
    assert shown.returncode == 0, shown.stderr
    text = shown.stdout
    assert re.search(r"^=== \d{4}-\d\d-\d\dT\S+Z run 42 ", text, re.M), text
    assert "Mem:" in text
    assert 'grafeo-1: {"event": "case", "id": "hops/hop_2"}' in text
    assert "  loading snb-medium" in text
    # The branch is public: a credential argument never lands on it.
    assert "Runner.Listener run --jitconfig [redacted]" in text
    assert "--token=[redacted]" in text
    assert jit[:20] not in text
    assert "hunter2" not in text
    assert "graph_bench.runner.worker --db grafeo" in text


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("git") is None, reason="heartbeat.sh reads Linux /proc")
def test_heartbeat_survives_a_failing_push(tmp_path: Path) -> None:
    """With no remote to push to, the beat is reported once on stderr and the script still exits 0."""
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--quiet", str(work)], check=True)
    env = {**os.environ, "HEARTBEAT_ONCE": "1"}
    script = str(SCRIPTS / "heartbeat.sh")
    done = subprocess.run(["bash", script, "out"], cwd=work, env=env, capture_output=True, text=True)
    assert done.returncode == 0
    assert "heartbeat: push to bot/heartbeat failed" in done.stderr
