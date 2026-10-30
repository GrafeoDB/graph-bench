"""Install the Grafeo under test (a wheel or a local checkout) and record what was installed."""

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

STATE_DIR = Path(".graph-bench")
STATE_FILE = "grafeo.json"
_ERR_TAIL = 2000


def _run(cmd: list[str]) -> str:
    """Run a command and return its stdout; failures carry the command and the stderr tail."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
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
    source: Path, *, run: Callable[[list[str]], str] = _run, state_dir: Path = STATE_DIR
) -> dict[str, str]:
    """Install Grafeo from a `.whl` file or a checkout; write and return `state_dir/grafeo.json`.

    A checkout is built with maturin; the build inherits the environment, so `CARGO_TARGET_DIR` is honoured.
    """
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
    else:
        raise ValueError(f"expected a .whl file or a grafeo checkout: {source}")
    version = run([sys.executable, "-c", "import grafeo; print(grafeo.__version__)"]).strip()
    info = {"version": version, "source": str(source), "sha": sha}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE).write_text(json.dumps(info), encoding="utf-8")
    return info
