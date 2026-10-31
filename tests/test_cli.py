from typer.testing import CliRunner

from graph_bench import __version__
from graph_bench.cli import app


def test_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_packages_lists_every_database_package_without_bounds() -> None:
    result = CliRunner().invoke(app, ["packages"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    names = {line.split(";")[0].strip() for line in lines}
    assert {"grafeo", "ladybug", "neug", "duckdb", "pyoxigraph", "helix-db", "helix-db-embedded", "JPype1"} <= names
    assert {"falkordblite", "turingdb", "pyarrow", "arcadedb-embedded"} <= names
    assert not any(ch in line.split(";")[0] for line in lines for ch in "<>=")
    assert "falkordblite ; sys_platform == 'linux'" in lines
    assert "graph-bench" not in names
