"""The newest stable release of a PyPI package: Grafeo's for the release-post workflow, the databases' for
`graph-bench local-check`.

`python -m graph_bench.release latest` prints Grafeo's; `newer LATEST POSTED` exits 0 when LATEST is strictly newer
(numeric order). Pre-releases (a, b, rc, dev and local versions) and releases whose files are all yanked are
ignored, and so are post releases unless asked for (`post=True`: arcadedb-embedded publishes `.postN` releases,
which an install of the newest stable picks).
"""

import json
import re
import sys
import urllib.request
from typing import Any

PYPI_URL = "https://pypi.org/pypi/{package}/json"
_STABLE = re.compile(r"\d+(\.\d+)*")
_POST = re.compile(r"(\d+(?:\.\d+)*)(?:\.post(\d+))?")


def _order(version: str) -> tuple[tuple[int, ...], int]:
    """Numeric order of a stable or post version: release numbers, then the post number (-1 for none)."""
    match = _POST.fullmatch(version)
    assert match is not None
    return tuple(int(p) for p in match.group(1).split(".")), int(match.group(2)) if match.group(2) else -1


def newest_stable(releases: dict[str, list[dict[str, Any]]], *, post: bool = False) -> str | None:
    """The highest stable version among PyPI's `releases` mapping (version -> files), or None; with `post`, post
    releases (`1.2.post1`) count too."""
    pattern = _POST if post else _STABLE
    stable = [
        version
        for version, files in releases.items()
        if pattern.fullmatch(version) and any(not f.get("yanked", False) for f in files)
    ]
    return max(stable, key=_order, default=None)


def same_version(a: str, b: str) -> bool:
    """Whether two versions are equal, trailing zeros aside (0.6 equals 0.6.0); else plain string equality."""
    if _POST.fullmatch(a) and _POST.fullmatch(b):
        (nums_a, post_a), (nums_b, post_b) = _order(a), _order(b)
        return (_trim(nums_a), post_a) == (_trim(nums_b), post_b)
    return a == b


def _trim(parts: tuple[int, ...]) -> tuple[int, ...]:
    trimmed = list(parts)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed)


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


def fetch_latest(package: str = "grafeo", *, post: bool = False) -> str:
    """The newest stable release of `package` on PyPI (`newest_stable`); raises OSError on a network error and
    ValueError when there is none."""
    with urllib.request.urlopen(PYPI_URL.format(package=package), timeout=30) as response:
        releases = json.load(response)["releases"]
    version = newest_stable(releases, post=post)
    if version is None:
        raise ValueError(f"PyPI lists no stable {package} release")
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
