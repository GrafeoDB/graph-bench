#!/usr/bin/env bash
# The local snapshot: all ten databases, the full selection, in one Linux (x86-64) container on the dev box, so one
# OS covers every database (FalkorDB and TuringDB need Linux). Writes results/local/results.json, which is
# committed; the readme-results job of bench.yml mixes its non-roster databases into the README
# (`graph-bench merge --local`), marked and never compared with the CI ones. Refresh it when
# `graph-bench local-check` reports a changed version.
#
# Usage, from anywhere in the repository (Git Bash on Windows, or any Linux or macOS shell with Docker):
#   bash .github/scripts/local-bench.sh
# Environment:
#   DOCKER          the docker command; default "docker" (for example "docker --context desktop-linux")
#   GRAFEO_VERSION  pin grafeo to this version; default: its newest stable release, like every other database
#
# The repository is mounted read-only and copied into the container, and the run's output stays inside the
# container until the end: appending to a Windows bind mount fails with ENODATA. Only results.json is copied out,
# also when the run fails part way (whatever it left), before the container is removed; the script then exits
# with the run's status.
# The dataset cache lives in a named volume (graph-bench-local-cache), so a refresh reuses it.
set -euo pipefail

DOCKER=${DOCKER:-docker}
IMAGE=ubuntu:24.04
NAME=graph-bench-local-$$
root=$(cd "$(dirname "$0")/../.." && pwd)
# Docker Desktop on Windows needs the Windows form of the path; Git Bash must not rewrite /src and the like.
host_root=$(cd "$root" && (pwd -W 2>/dev/null || pwd))
export MSYS_NO_PATHCONV=1

inside=$(cat <<'SCRIPT'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl git > /dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /work
# A copy of the checkout without the host's virtual environment and run output.
tar -C /src --exclude=./.venv --exclude=./out --exclude=./results -cf - . | tar -C /work -xf -
git config --global --add safe.directory /work
cd /work
export UV_PYTHON=3.13
uv sync --frozen --extra bench
bash .github/scripts/install-databases.sh
uv run --no-sync graph-bench data prepare --selection full
uv run --no-sync graph-bench bench --selection full --out /work/out
SCRIPT
)

cleanup() { $DOCKER rm -f "$NAME" > /dev/null 2>&1 || true; }
trap cleanup EXIT

status=0
$DOCKER run --name "$NAME" --platform linux/amd64 \
  -v "$host_root:/src:ro" \
  -v graph-bench-local-cache:/root/.cache/graph-bench \
  -e GRAFEO_VERSION="${GRAFEO_VERSION:-}" \
  "$IMAGE" bash -c "$inside" || status=$?

mkdir -p "$root/results/local"
if (cd "$root" && $DOCKER cp "$NAME:/work/out/results.json" results/local/results.json); then
  echo "wrote $root/results/local/results.json; commit it, then regenerate the README (bench.yml, readme-results)"
else
  echo "local-bench: the run left no results.json" >&2
  [ "$status" -ne 0 ] || status=1
fi
if [ "$status" -ne 0 ]; then
  echo "local-bench: the run in the container exited with status $status" >&2
fi
exit "$status"
