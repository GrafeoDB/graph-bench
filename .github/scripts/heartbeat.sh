#!/usr/bin/env bash
# A runner heartbeat for the long benchmark jobs. GitHub keeps no log of a job whose runner is lost, so every
# HEARTBEAT_INTERVAL seconds this force-pushes one file, heartbeat.txt, to the branch bot/heartbeat: the run, the
# time, memory, disk, load, the largest processes, the case each worker is on and the last lines of the bench
# output. After a lost runner the branch shows where the box died and in what state.
#
# Usage: heartbeat.sh OUT_DIR [BENCH_LOG]   (started in the background next to `graph-bench bench`)
# Pushes through the checkout's credentials (needs `contents: write`). It never fails the job: every error is
# reported once on stderr and the loop goes on. HEARTBEAT_ONCE=1 writes and pushes a single beat (tests).
set -u

out=${1:?usage: heartbeat.sh OUT_DIR [BENCH_LOG]}
bench_log=${2:-}
interval=${HEARTBEAT_INTERVAL:-60}
branch=${HEARTBEAT_BRANCH:-bot/heartbeat}
remote=${HEARTBEAT_REMOTE:-origin}
keep=${HEARTBEAT_KEEP:-180}  # beats kept in the file: three hours at the default interval
history=$(mktemp)
warned=""

export GIT_AUTHOR_NAME="graph-bench heartbeat" GIT_AUTHOR_EMAIL="heartbeat@graph-bench.invalid"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

beat() {
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) run ${GITHUB_RUN_ID:-local} attempt ${GITHUB_RUN_ATTEMPT:-1} job ${GITHUB_JOB:-local}"
  echo "load: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
  free -m 2>/dev/null | sed -n '1,2p'
  df -h / "${RUNNER_TEMP:-/tmp}" "$out" 2>/dev/null | awk '!seen[$0]++'
  echo "largest processes (RSS MB, elapsed, command):"
  ps -eo rss=,etime=,args= --sort=-rss 2>/dev/null | head -5 | awk '{printf "  %d %s", $1 / 1024, $2; for (i = 3; i <= NF && i < 9; i++) printf " %s", substr($i, 1, 80); print ""}'
  for log in "$out"/raw/*/worker.jsonl; do
    [ -f "$log" ] || continue
    echo "$(basename "$(dirname "$log")"): $(tail -n 1 "$log" | cut -c1-240)"
  done
  if [ -n "$bench_log" ] && [ -f "$bench_log" ]; then
    echo "bench output:"
    tail -n 3 "$bench_log" | cut -c1-240 | sed 's/^/  /'
  fi
  sudo -n dmesg 2>/dev/null | grep -i -E 'out of memory|oom-kill|killed process' | tail -n 3 | sed 's/^/kernel: /'
}

push() {
  local blob tree commit
  blob=$(git hash-object -w "$history") &&
    tree=$(printf '100644 blob %s\theartbeat.txt\n' "$blob" | git mktree) &&
    commit=$(git commit-tree "$tree" -m "heartbeat ${GITHUB_RUN_ID:-local}") &&
    git push --quiet --force "$remote" "$commit:refs/heads/$branch"
}

while true; do
  beat >> "$history" 2>&1
  # keep the newest `keep` beats
  awk -v keep="$keep" '/^=== /{n++} {line[NR]=$0; beat[NR]=n} END{for (i = 1; i <= NR; i++) if (beat[i] > n - keep) print line[i]}' \
    "$history" > "$history.new" && mv "$history.new" "$history"
  if ! push 2> "$history.err"; then
    [ -z "$warned" ] && echo "heartbeat: push to $branch failed: $(head -c 300 "$history.err")" >&2
    warned=1
  fi
  [ "${HEARTBEAT_ONCE:-}" = 1 ] && break
  sleep "$interval"
done
