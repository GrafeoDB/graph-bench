"""The newest stable Grafeo release on PyPI, for the release-post workflow.

`python -m graph_bench.release latest` prints it. Pre-releases (a, b, rc, dev, post and local versions) and
releases whose files are all yanked are ignored.
"""

import json
import re
import sys
import urllib.request
from typing import Any

PYPI_URL = "https://pypi.org/pypi/grafeo/json"
_STABLE = re.compile(r"\d+(\.\d+)*")


def newest_stable(releases: dict[str, list[dict[str, Any]]]) -> str | None:
    """The highest stable version among PyPI's `releases` mapping (version -> files), or None."""
    stable = [
        version
        for version, files in releases.items()
        if _STABLE.fullmatch(version) and any(not f.get("yanked", False) for f in files)
    ]
    return max(stable, key=lambda v: tuple(int(p) for p in v.split(".")), default=None)


def fetch_latest(url: str = PYPI_URL) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        releases = json.load(response)["releases"]
    version = newest_stable(releases)
    if version is None:
        raise ValueError("PyPI lists no stable grafeo release")
    return version


def main(argv: list[str]) -> int:
    if argv != ["latest"]:
        print("usage: python -m graph_bench.release latest", file=sys.stderr)
        return 2
    try:
        print(fetch_latest())
    except (OSError, ValueError, KeyError) as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
