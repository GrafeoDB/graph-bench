"""Database names, their data models, display names and implementing classes.

The static maps below cover the built-in databases without importing their drivers, so `list` works
without the packages installed. Other names are looked up in `GRAPH_BENCH_EXTRA_DATABASES`
(`"name=module:Class,name2=module:Class"`), whose classes carry `models` and `display_name` ClassVars.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

from graph_bench.types import Model

if TYPE_CHECKING:
    from graph_bench.databases.base import Database

KNOWN: tuple[str, ...] = ("grafeo", "ladybug", "neug", "falkordb", "turingdb", "helixdb", "oxigraph")

MODELS: dict[str, frozenset[Model]] = {
    "grafeo": frozenset({"property_graph", "sql_pgq", "rdf"}),
    "ladybug": frozenset({"property_graph"}),
    "neug": frozenset({"property_graph"}),
    "falkordb": frozenset({"property_graph"}),
    "turingdb": frozenset({"property_graph"}),
    "helixdb": frozenset({"property_graph"}),
    "oxigraph": frozenset({"rdf"}),
}

DISPLAY: dict[str, str] = {
    "grafeo": "Grafeo",
    "ladybug": "LadybugDB",
    "neug": "NeuG",
    "falkordb": "FalkorDB",
    "turingdb": "TuringDB",
    "helixdb": "HelixDB",
    "oxigraph": "Oxigraph",
}

EXTRA_ENV = "GRAPH_BENCH_EXTRA_DATABASES"

REGISTRY: dict[str, str] = {
    "grafeo": "graph_bench.databases.grafeo:GrafeoDatabase",
    "ladybug": "graph_bench.databases.ladybug:LadybugDatabase",
    "neug": "graph_bench.databases.neug:NeugDatabase",
    "falkordb": "graph_bench.databases.falkordb:FalkorDatabase",
    "turingdb": "graph_bench.databases.turingdb:TuringDatabase",
    "helixdb": "graph_bench.databases.helixdb:HelixDatabase",
    "oxigraph": "graph_bench.databases.oxigraph:OxigraphDatabase",
}


def _extra_targets() -> dict[str, str]:
    targets = {}
    for part in os.environ.get(EXTRA_ENV, "").split(","):
        name, sep, target = part.partition("=")
        if sep and name.strip():
            targets[name.strip()] = target.strip()
    return targets


def registry() -> dict[str, str]:
    """`REGISTRY` extended by `GRAPH_BENCH_EXTRA_DATABASES`, read now."""
    return {**REGISTRY, **_extra_targets()}


def _import(name: str, targets: dict[str, str]) -> Any:
    target = targets.get(name)
    if target is None:
        known = ", ".join(sorted({*KNOWN, *registry()}))
        raise KeyError(f"unknown database {name!r} (known: {known}; add others with {EXTRA_ENV}=name=module:Class)")
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def _extra_class(name: str) -> Any:
    return _import(name, _extra_targets())


def get_database(name: str) -> type[Database]:
    """The class of a database, importing its module (and so its driver) only now.

    Raises `KeyError` for an unknown name and `ImportError` when the driver is not installed.
    """
    return _import(name, registry())


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
