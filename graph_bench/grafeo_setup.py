"""Install the Grafeo under test (a wheel or a local checkout) and record what was installed."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

STATE_DIR = Path(".graph-bench")
STATE_FILE = "grafeo.json"
# A git commit as `--grafeo-sha` takes it: an abbreviated or full hex sha, with `-dirty` for a modified build.
GIT_SHA = re.compile(r"[0-9a-f]{7,40}(-dirty)?")
VERSION_SCRIPT = "import importlib.metadata as m; print(m.version('grafeo'))"
_ERR_TAIL = 2000


def _run(cmd: list[str]) -> str:
    """Run a command and return its stdout; failures carry the command and the stderr tail."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8", errors="replace").stdout
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-_ERR_TAIL:]
        raise RuntimeError(f"command failed ({exc.returncode}): {' '.join(cmd)}\n{tail}") from exc


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"'{name}' was not found on PATH; install uv (https://docs.astral.sh/uv/)")
    return found


def _install_wheel(wheel: Path, run: Callable[[list[str]], str]) -> None:
    run([_tool("uv"), "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", str(wheel)])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_grafeo(
    source: Path,
    *,
    git_sha: str | None = None,
    run: Callable[[list[str]], str] = _run,
    state_dir: Path = STATE_DIR,
) -> dict[str, str | None]:
    """Install Grafeo from a `.whl` file or a checkout; write and return `state_dir/grafeo.json`.

    A checkout is built with maturin; the build inherits the environment, so `CARGO_TARGET_DIR` is honoured. Its
    sha is the checked-out commit, with `-dirty` appended when tracked files have uncommitted changes. A wheel's
    sha is the SHA-256 of the file. `git_sha` names the grafeo commit a wheel was built from (a wheel does not
    record it, and local builds of a release branch share one version string); a checkout's `git_sha` is its sha.
    """
    if git_sha is not None and not GIT_SHA.fullmatch(git_sha):
        raise ValueError(f"--grafeo-sha must be a git commit (7 to 40 hex digits, optional -dirty): {git_sha!r}")
    if source.is_file() and source.suffix == ".whl":
        _install_wheel(source, run)
        sha = _sha256(source)
    elif source.is_dir():
        nested = source / "crates" / "bindings" / "python"
        bindings = nested if nested.is_dir() else source
        wheels = state_dir / "wheels"
        build = ["maturin", "build", "--release", "-i", sys.executable, "--out", str(wheels)]
        run([_tool("uvx"), *build, "-m", str(bindings / "Cargo.toml")])
        built = sorted(wheels.glob("*.whl"), key=lambda p: p.stat().st_mtime) if wheels.is_dir() else []
        if not built:
            raise RuntimeError(f"maturin produced no wheel in {wheels}")
        _install_wheel(built[-1], run)
        sha = run(["git", "-C", str(source), "rev-parse", "HEAD"]).strip()
        # Uncommitted changes to tracked files (as `git describe --dirty` counts them): the build is not that commit.
        if run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]).strip():
            sha += "-dirty"
        if git_sha is not None and not sha.startswith(git_sha.removesuffix("-dirty")):
            raise ValueError(f"--grafeo-sha {git_sha} does not match the checkout's commit {sha}")
        git_sha = sha
    else:
        raise ValueError(f"expected a .whl file or a grafeo checkout: {source}")
    # The distribution version, as `supervisor.grafeo_source` compares it (grafeo.__version__ is the Cargo
    # version, spelled differently for pre-releases).
    version = run([sys.executable, "-c", VERSION_SCRIPT]).strip()
    info: dict[str, str | None] = {"version": version, "source": str(source), "sha": sha, "git_sha": git_sha}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE).write_text(json.dumps(info), encoding="utf-8")
    return info
