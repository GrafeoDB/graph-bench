"""Canonicalization and verification of query answers against a reference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graph_bench.suites.catalog import Case
    from graph_bench.types import AnswerVerdict

SNB_EPS = 1e-9
LDBC_EPS = 1e-4

_INT_LIMIT = 2**53
_UNREACHABLE_LIMIT = 2**62
_BINDING_LABELS = ("binding", "row", "col")
_BINDING_UNITS = ("bindings", "rows", "cols")


def _canon_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) < _INT_LIMIT:
        return int(value)
    if isinstance(value, (list, tuple)):
        items = [_canon_value(v) for v in value]
        return sorted(items, key=_sort_key)
    return value


def _sort_key(value: Any) -> tuple[str, str]:
    """Sort key: rounded repr first, so float noise does not reorder rows; full repr breaks ties.

    Only used for ordering, never for output. The tie-break keeps the result a function of the input
    multiset, so permutations of the same rows canonicalize identically.
    """
    return (repr(_round_floats(value)), repr(value))


def canonical(rows: Iterable[Sequence[Any]], *, ordered: bool) -> list[list[Any]]:
    """Normalize result rows so that equivalent answers compare equal."""
    result = [[_canon_value(v) for v in row] for row in rows]
    if not ordered:
        result.sort(key=_sort_key)
    return result


def canonical_map(values: Mapping[Any, Any], *, drop_unreachable: bool) -> dict[int, Any]:
    """Normalize a node-to-value mapping, optionally dropping unreachable entries."""
    result: dict[int, Any] = {}
    for key, value in values.items():
        if drop_unreachable and _is_unreachable(value):
            continue
        result[int(key)] = _canon_value(value)
    return result


def _is_unreachable(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isinf(value):
        return True
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= _UNREACHABLE_LIMIT


def _round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, Mapping):
        return {str(k): _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


def digest(answer: Any) -> str:
    """Stable short hash of an answer, insensitive to float noise below 6 significant digits."""
    payload = json.dumps(_round_floats(answer), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _float_close(actual: float, reference: float, rel_eps: float) -> bool:
    if actual == reference:
        return True
    if reference == 0:
        return abs(actual) <= rel_eps
    return abs(actual - reference) <= rel_eps * abs(reference)


def _join(prefix: str, message: str) -> str:
    return f"{prefix}: {message}" if prefix else message


def _describe(answer: Any, reference: Any, rel_eps: float, prefix: str, depth: int) -> str | None:
    if isinstance(answer, Mapping) and isinstance(reference, Mapping):
        return _describe_dict(answer, reference, rel_eps, prefix, depth)
    if isinstance(answer, (list, tuple)) and isinstance(reference, (list, tuple)):
        return _describe_list(answer, reference, rel_eps, prefix, depth)
    if _is_number(answer) and _is_number(reference) and (isinstance(answer, float) or isinstance(reference, float)):
        if _float_close(float(answer), float(reference), rel_eps):
            return None
        return _join(prefix, f"{answer} != {reference}")
    # A bool never equals a number here, even though True == 1 in Python.
    same_kind = isinstance(answer, bool) == isinstance(reference, bool)
    if same_kind and answer == reference:
        return None
    return _join(prefix, f"{answer!r} != {reference!r}")


def _describe_dict(
    answer: Mapping[Any, Any], reference: Mapping[Any, Any], rel_eps: float, prefix: str, depth: int
) -> str | None:
    if set(answer) != set(reference):
        missing = [k for k in reference if k not in answer]
        extra = [k for k in answer if k not in reference]
        detail = []
        if missing:
            detail.append(f"missing keys {sorted(missing, key=repr)}")
        if extra:
            detail.append(f"unexpected keys {sorted(extra, key=repr)}")
        return _join(prefix, ", ".join(detail))
    for key in reference:
        label = f"key {key}"
        child_prefix = f"{prefix}, {label}" if prefix else label
        message = _describe(answer[key], reference[key], rel_eps, child_prefix, depth)
        if message is not None:
            return message
    return None


def _describe_list(
    answer: Sequence[Any], reference: Sequence[Any], rel_eps: float, prefix: str, depth: int
) -> str | None:
    unit = _BINDING_UNITS[depth] if depth < len(_BINDING_UNITS) else "items"
    if len(answer) != len(reference):
        return _join(prefix, f"{len(answer)} {unit} != {len(reference)} {unit}")
    label_word = _BINDING_LABELS[depth] if depth < len(_BINDING_LABELS) else "item"
    for index, (a, b) in enumerate(zip(answer, reference, strict=True)):
        label = f"{label_word} {index}"
        child_prefix = f"{prefix}, {label}" if prefix else label
        message = _describe(a, b, rel_eps, child_prefix, depth + 1)
        if message is not None:
            return message
    return None


def verify(answer: Any, reference: Any, *, rel_eps: float) -> str | None:
    """Return None when answer matches reference, otherwise a description of the first difference.

    Floats are compared with a relative tolerance against the reference (absolute when the reference is 0).
    Booleans are never treated as numbers.
    """
    return _describe(answer, reference, rel_eps, "", 0)


def check_case(case: Case, answers: list[Any], reference_file: Path) -> tuple[AnswerVerdict, str | None]:
    """Check a case's canonical per-binding answers against its reference file.

    WCC labels are first normalized to the smallest member of each component. Returns the verdict and,
    for "disagree", the first difference.
    """
    # Imported here: the catalog and the suites import this module.
    from graph_bench.suites import Algorithm
    from graph_bench.suites.catalog import load_reference
    from graph_bench.suites.graphalytics import min_member

    item = case.item
    if isinstance(item, Algorithm) and item.name == "wcc":
        answers = [min_member(answer) for answer in answers]
    _, reference = load_reference(reference_file)
    message = verify(answers, reference, rel_eps=item.rel_eps)
    return ("agree", None) if message is None else ("disagree", message)
