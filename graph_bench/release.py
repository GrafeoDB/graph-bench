"""The newest stable Grafeo release on PyPI, for the release-post workflow.

`python -m graph_bench.release latest` prints it; `newer LATEST POSTED` exits 0 when LATEST is strictly newer
(numeric order). Pre-releases (a, b, rc, dev, post and local versions) and releases whose files are all yanked
are ignored.
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


def parse_stable(version: str) -> tuple[int, ...]:
    """A stable version as a tuple of ints with trailing zeros dropped (0.6 equals 0.6.0); ValueError otherwise."""
    if not _STABLE.fullmatch(version):
        raise ValueError(f"{version!r} is not a stable version")
    parts = [int(p) for p in version.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_newer(latest: str, posted: str) -> bool:
    """Whether `latest` is strictly greater than `posted`; an empty `posted` (never posted) is older than any."""
    if not posted.strip():
        return True
    return parse_stable(latest) > parse_stable(posted.strip())


def fetch_latest(url: str = PYPI_URL) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        releases = json.load(response)["releases"]
    version = newest_stable(releases)
    if version is None:
        raise ValueError("PyPI lists no stable grafeo release")
    return version


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "newer":
        # Exit 0 when argv[1] is strictly greater than argv[2] (empty: never posted), 1 when not, 2 on bad input.
        try:
            return 0 if is_newer(argv[1], argv[2]) else 1
        except ValueError as exc:
            print(f"release: {exc}", file=sys.stderr)
            return 2
    if argv != ["latest"]:
        print("usage: python -m graph_bench.release latest | newer LATEST POSTED", file=sys.stderr)
        return 2
    try:
        print(fetch_latest())
    except (OSError, ValueError, KeyError) as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
