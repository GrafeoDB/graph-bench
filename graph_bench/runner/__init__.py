"""Running databases: timing, the per-database worker process, the startup children and the supervisor."""

import sys

from graph_bench.databases.base import Database, NotSupported

# Run-time limits (Task T, user decision of 2026-10-31). CodSpeed's Graviton macro runner dies about 60 minutes
# after boot, so a run must fit in that; the limits apply to every database alike (FAIRNESS.md):
# - a case (query, algorithm, write) stops `CASE_CEILING_S` after its `start` line: the worker starts no warm-up or
#   timed run once that time is reached, and records the timing over at least `MIN_ITERATIONS` timed runs, else
#   ⏱ "too slow: over 60 s"; the supervisor kills a single call still running `CEILING_GRACE_S` later;
# - a load (either Load row, the Memory and Disk record's load, an answer-only fixture's load) still running
#   `LOAD_CEILING_S` after its start is stopped: ⏱ "load over 5 min", and so is every case of that dataset;
# - time box: no new timed run once the timed runs total `TIME_BOX_S` and at least `MIN_ITERATIONS` ran
#   (`MIN_LOAD_ITERATIONS` for loads); fast cases keep all their runs;
# - `DEFAULT_CASE_TIMEOUT_S` is the silence limit (`--case-timeout`), down from 600 s.
CASE_CEILING_S = 60.0
LOAD_CEILING_S = 300.0
TIME_BOX_S = 5.0
MIN_ITERATIONS = 3
MIN_LOAD_ITERATIONS = 1
DEFAULT_CASE_TIMEOUT_S = 120.0
# How long past a ceiling the supervisor waits before it kills: room for the worker's own stop (the read-back of the
# run in progress, the counts guard, the entry line), so a worker that stops itself is never killed for it.
CEILING_GRACE_S = 5.0
# A startup child that has not answered `STARTUP_CHILD_TIMEOUT_S` after it started is killed and the Startup entry
# is ⏱. It is fixed, not derived from the silence limit: during the Startup row the supervisor raises its silence
# limit above it, so the worker always kills its own child first and no child outlives its worker.
STARTUP_CHILD_TIMEOUT_S = 120.0
# How far a raised silence limit lies beyond the limit it makes room for (a load's ceiling and grace, a startup
# child's limit and grace), so that the inner limit, which records the right marker, always fires first.
SILENCE_MARGIN_S = 1.0
# The deadline CI passes as `--deadline-s` (Task R1). The GitHub job times out at 55 minutes and the CodSpeed
# runner dies about 60 minutes after boot; at 48 minutes the supervisor starts nothing new, marks every case not yet
# run ⏱ "run deadline: 48 min" and writes results.json, so a job always leaves results. `--deadline-s` has no
# default: a local run is not limited.
RUN_DEADLINE_S = 48 * 60


def over_ceiling(ceiling_s: float) -> str:
    """The error of a case stopped at its ceiling: "too slow: over 60 s"."""
    return f"too slow: over {ceiling_s:g} s"


def load_over(ceiling_s: float) -> str:
    """The error of a load stopped at its ceiling, and of every case of its dataset: "load over 5 min"."""
    if ceiling_s >= 60 and ceiling_s % 60 == 0:
        return f"load over {ceiling_s / 60:g} min"
    return f"load over {ceiling_s:g} s"


def deadline_over(deadline_s: float) -> str:
    """The error of a case, load or record not started because the run deadline had passed: "run deadline: 48 min"
    for a whole number of minutes, else seconds ("run deadline: 90 s")."""
    if deadline_s >= 60 and deadline_s % 60 == 0:
        return f"run deadline: {deadline_s / 60:g} min"
    return f"run deadline: {deadline_s:g} s"


def message(exc: BaseException) -> str:
    """The error text an entry records for `exc`: a `NotSupported` reason as it is, else "<Type>: <text>"."""
    text = str(exc)
    if isinstance(exc, NotSupported):
        return text or "not supported"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# What a handler around a database call never swallows: the user stopping the run, the interpreter exiting.
CONTROL_FLOW = (KeyboardInterrupt, SystemExit, GeneratorExit)


def reraise_control(exc: BaseException) -> None:
    """First line of a handler around a database call written `except BaseException as exc`.

    A database's panic (pyo3's `PanicException`) derives from `BaseException`, not `Exception`, so a handler
    that catches `Exception` lets it kill the process. Such a handler catches everything and calls this:
    `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` are raised again, anything else falls through to be
    recorded like any other database error ("PanicException: <message>").
    """
    if isinstance(exc, CONTROL_FLOW):
        raise exc


def close_quietly(db: Database, context: str = "closing") -> None:
    """`db.close()` where an error must not replace the result or the error already being recorded.

    A close that raises (or panics) writes its message to stderr as "<context>: <message>" and the run goes on;
    `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` still stop it.
    """
    try:
        db.close()
    except BaseException as exc:
        reraise_control(exc)
        sys.stderr.write(f"{context}: {message(exc)}\n")
