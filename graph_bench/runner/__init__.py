"""Running databases: timing, the per-database worker process, the startup children and the supervisor."""

from graph_bench.databases.base import NotSupported


def message(exc: BaseException) -> str:
    """The error text an entry records for `exc`: a `NotSupported` reason as it is, else "<Type>: <text>"."""
    text = str(exc)
    if isinstance(exc, NotSupported):
        return text or "not supported"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
