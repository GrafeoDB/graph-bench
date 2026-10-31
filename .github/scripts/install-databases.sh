#!/usr/bin/env bash
# Install the database packages one at a time at their newest stable release (`graph-bench packages`), keeping
# going when one has no installable build for this runner (for example falkordblite on glibc older than 2.38).
# A failed package is logged with the installer's reason and listed in install-failures.txt; the benchmark run then
# shows that database as failed ("not installed") instead of the whole job stopping. uv.lock is not touched, and
# uv never picks a pre-release. Wheels only (--only-binary :all:): a package with no wheel for this platform fails at
# once with uv's reason, instead of falling back to a source build that needs a compiler and may not finish.
set -u
uv run --no-sync graph-bench packages > database-requirements.txt
: > install-failures.txt
while IFS= read -r requirement; do
  [ -n "$requirement" ] || continue
  if uv pip install --upgrade --only-binary :all: "$requirement" > install-attempt.log 2>&1; then
    echo "installed: $requirement"
  else
    reason=$(grep -v '^\s*$' install-attempt.log | tail -n 6 | tr '\n' ' ')
    echo "::warning title=Not installable on $(uname -m)::$requirement: $reason"
    echo "NOT INSTALLED: $requirement: $reason"
    echo "$requirement: $reason" >> install-failures.txt
  fi
done < database-requirements.txt
uv pip freeze | tee installed-versions.txt
if [ -s install-failures.txt ]; then
  echo "Packages without an installable build on $(uname -m):"
  cat install-failures.txt
fi
exit 0
