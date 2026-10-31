import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from graph_bench import cli, grafeo_setup
from graph_bench.grafeo_setup import install_grafeo

SHA = "a" * 40


class Recorder:
    def __init__(self, status: str = "") -> None:
        self.commands: list[list[str]] = []
        self.status = status  # `git status --porcelain` output: empty for a clean checkout

    def __call__(self, cmd: list[str]) -> str:
        self.commands.append(cmd)
        if grafeo_setup.VERSION_SCRIPT in cmd:
            return "0.6.0\n"
        if cmd[:1] == ["git"] and "status" in cmd:
            return self.status
        if cmd[:1] == ["git"]:
            return SHA + "\n"
        return ""


@pytest.fixture(autouse=True)
def _uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grafeo_setup.shutil, "which", lambda name: "uv-bin" if name == "uv" else "uvx-bin")


def test_install_wheel_commands(tmp_path: Path) -> None:
    wheel = tmp_path / "grafeo-0.6.0-cp312-win_amd64.whl"
    wheel.write_bytes(b"wheel")
    run = Recorder()
    info = install_grafeo(wheel, run=run, state_dir=tmp_path / "state")
    assert run.commands[0] == [
        "uv-bin",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        str(wheel),
    ]
    assert run.commands[1][:3] == [sys.executable, "-c", grafeo_setup.VERSION_SCRIPT]
    assert info["version"] == "0.6.0"
    assert info["source"] == str(wheel)
    assert len(info["sha"]) == 64


def test_install_checkout_builds_then_installs(tmp_path: Path) -> None:
    src = tmp_path / "grafeo"
    bindings = src / "crates" / "bindings" / "python"
    bindings.mkdir(parents=True)
    state = tmp_path / "state"
    wheels = state / "wheels"
    wheels.mkdir(parents=True)
    old = wheels / "grafeo-0.5.0-x.whl"
    new = wheels / "grafeo-0.6.0-x.whl"
    old.write_bytes(b"o")
    new.write_bytes(b"n")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    run = Recorder()
    info = install_grafeo(src, run=run, state_dir=state)
    assert run.commands[0] == [
        "uvx-bin",
        "maturin",
        "build",
        "--release",
        "-i",
        sys.executable,
        "--out",
        str(wheels),
        "-m",
        str(bindings / "Cargo.toml"),
    ]
    assert run.commands[1] == [
        "uv-bin",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--reinstall",
        "--no-deps",
        str(new),
    ]
    assert ["git", "-C", str(src), "rev-parse", "HEAD"] in run.commands
    assert info == {"version": "0.6.0", "source": str(src), "sha": SHA, "git_sha": SHA}


def test_install_dirty_checkout_marks_the_sha(tmp_path: Path) -> None:
    """Uncommitted changes to tracked files: the sha alone would not identify the build."""
    state = tmp_path / "state"
    (state / "wheels").mkdir(parents=True)
    (state / "wheels" / "grafeo-1.whl").write_bytes(b"x")
    run = Recorder(status=" M crates/grafeo-engine/src/lib.rs\n")
    info = install_grafeo(tmp_path, run=run, state_dir=state)
    assert info["sha"] == SHA + "-dirty"
    assert ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=no"] in run.commands


def test_checkout_without_nested_bindings_uses_root(tmp_path: Path) -> None:
    run = Recorder()
    state = tmp_path / "state"
    (state / "wheels").mkdir(parents=True)
    (state / "wheels" / "grafeo-1.whl").write_bytes(b"x")
    install_grafeo(tmp_path, run=run, state_dir=state)
    assert run.commands[0][-1] == str(tmp_path / "Cargo.toml")


def test_checkout_build_without_wheel_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no wheel"):
        install_grafeo(tmp_path, run=Recorder(), state_dir=tmp_path / "state")


def test_rejects_other_paths(tmp_path: Path) -> None:
    other = tmp_path / "x.txt"
    other.write_text("x")
    with pytest.raises(ValueError, match=r"expected a .whl file or a grafeo checkout"):
        install_grafeo(other, run=Recorder(), state_dir=tmp_path / "state")
    with pytest.raises(ValueError, match=r"expected a .whl file or a grafeo checkout"):
        install_grafeo(tmp_path / "missing", run=Recorder(), state_dir=tmp_path / "state")


def test_writes_state_file(tmp_path: Path) -> None:
    wheel = tmp_path / "g.whl"
    wheel.write_bytes(b"w")
    state = tmp_path / "state"
    info = install_grafeo(wheel, run=Recorder(), state_dir=state)
    assert json.loads((state / "grafeo.json").read_text()) == info
    assert set(info) == {"version", "source", "sha", "git_sha"}
    assert info["git_sha"] is None


def test_missing_uv_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grafeo_setup.shutil, "which", lambda name: None)
    wheel = tmp_path / "g.whl"
    wheel.write_bytes(b"w")
    with pytest.raises(RuntimeError, match="uv"):
        install_grafeo(wheel, run=Recorder(), state_dir=tmp_path / "state")


def test_default_run_reports_stderr() -> None:
    assert grafeo_setup._run([sys.executable, "-c", "print('hi')"]).strip() == "hi"
    with pytest.raises(RuntimeError, match="boom"):
        grafeo_setup._run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])


def test_run_decodes_utf8_output_regardless_of_locale() -> None:
    # Child writes raw UTF-8 bytes; a locale-codec decode (cp1252 on Windows) would mangle or reject them.
    code = "import sys; sys.stdout.buffer.write('\\u2192 \\u00e9\\n'.encode('utf-8'))"
    out = grafeo_setup._run([sys.executable, "-c", code])
    assert "→" in out
    assert "é" in out


def _patch_bench(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    calls: list[Path] = []
    monkeypatch.setattr(
        cli, "install_grafeo", lambda p, **_: calls.append(p) or {"version": "0", "source": "", "sha": ""}
    )
    monkeypatch.setattr(cli.supervisor, "run_bench", lambda *a, **k: None)
    return calls


def test_bench_honours_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_bench(monkeypatch)
    monkeypatch.setenv("GRAPH_BENCH_GRAFEO", str(tmp_path / "env"))
    args = ["bench", "--selection", "push", "--db", "grafeo", "--out", str(tmp_path / "out")]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [tmp_path / "env"]


def test_bench_flag_beats_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_bench(monkeypatch)
    monkeypatch.setenv("GRAPH_BENCH_GRAFEO", str(tmp_path / "env"))
    args = [
        "bench",
        "--selection",
        "push",
        "--db",
        "grafeo",
        "--out",
        str(tmp_path / "o"),
        "--grafeo",
        str(tmp_path / "f"),
    ]
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert calls == [tmp_path / "f"]


def test_bench_without_grafeo_does_not_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_bench(monkeypatch)
    monkeypatch.delenv("GRAPH_BENCH_GRAFEO", raising=False)
    args = ["bench", "--selection", "push", "--db", "grafeo", "--out", str(tmp_path / "o")]
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert calls == []


def test_setup_command_prints_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"version": "0.6.0", "source": "s", "sha": "h"}
    monkeypatch.setattr(cli, "install_grafeo", lambda p, **_: info)
    result = CliRunner().invoke(cli.app, ["setup", "--grafeo", str(tmp_path)])
    assert result.exit_code == 0
    assert json.loads(result.output) == info


def test_setup_command_reports_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["setup", "--grafeo", str(tmp_path / "nope")])
    assert result.exit_code == 2
    assert "expected a .whl file" in result.output


def test_bench_validates_before_installing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_bench(monkeypatch)
    args = ["bench", "--selection", "push", "--db", "nope", "--out", str(tmp_path), "--grafeo", str(tmp_path)]
    assert CliRunner().invoke(cli.app, args).exit_code != 0
    args = ["bench", "--selection", "bogus", "--db", "grafeo", "--out", str(tmp_path), "--grafeo", str(tmp_path)]
    assert CliRunner().invoke(cli.app, args).exit_code != 0
    assert calls == []


def test_wheel_records_the_given_git_sha(tmp_path: Path) -> None:
    """A wheel names no commit, so `--grafeo-sha` is recorded beside the file's SHA-256."""
    wheel = tmp_path / "g.whl"
    wheel.write_bytes(b"w")
    info = install_grafeo(wheel, git_sha="9164e307", run=Recorder(), state_dir=tmp_path / "state")
    assert info["git_sha"] == "9164e307"
    assert info["sha"] == hashlib.sha256(b"w").hexdigest()


@pytest.mark.parametrize("bad", ["main", "12345", "9164E307", "9164e307 ", "g164e307"])
def test_git_sha_must_be_a_commit(tmp_path: Path, bad: str) -> None:
    wheel = tmp_path / "g.whl"
    wheel.write_bytes(b"w")
    with pytest.raises(ValueError, match="grafeo-sha"):
        install_grafeo(wheel, git_sha=bad, run=Recorder(), state_dir=tmp_path / "state")
