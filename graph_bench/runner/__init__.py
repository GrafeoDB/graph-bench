"""Running databases: timing, the per-database worker process, the startup children and the supervisor."""

import sys

from graph_bench.databases.base import Database, NotSupported


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
