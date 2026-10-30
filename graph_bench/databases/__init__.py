"""Database names, their data models and display names.

The static maps below cover the built-in databases without importing their drivers, so `list` works
without the packages installed. Other names are looked up in `GRAPH_BENCH_EXTRA_DATABASES`
(`"name=module:Class,name2=module:Class"`), whose classes carry `models` and `display_name` ClassVars.
"""

import importlib
import os
from typing import Any

from graph_bench.types import Model

KNOWN: tuple[str, ...] = ("grafeo", "ladybug")

MODELS: dict[str, frozenset[Model]] = {
    "grafeo": frozenset({"property_graph"}),
    "ladybug": frozenset({"property_graph"}),
}

DISPLAY: dict[str, str] = {"grafeo": "Grafeo", "ladybug": "LadybugDB"}

EXTRA_ENV = "GRAPH_BENCH_EXTRA_DATABASES"


def _extra_targets() -> dict[str, str]:
    targets = {}
    for part in os.environ.get(EXTRA_ENV, "").split(","):
        name, sep, target = part.partition("=")
        if sep and name.strip():
            targets[name.strip()] = target.strip()
    return targets


def _extra_class(name: str) -> Any:
    target = _extra_targets().get(name)
    if target is None:
        known = ", ".join(sorted({*KNOWN, *_extra_targets()}))
        raise KeyError(f"unknown database {name!r} (known: {known}; add others with {EXTRA_ENV}=name=module:Class)")
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def model_support(name: str) -> frozenset[Model]:
    """The data models a database supports."""
    if name in MODELS:
        return MODELS[name]
    return frozenset(_extra_class(name).models)


def display_name(name: str) -> str:
    """The name shown in the matrix, e.g. "LadybugDB"."""
    if name in DISPLAY:
        return DISPLAY[name]
    return str(_extra_class(name).display_name)
