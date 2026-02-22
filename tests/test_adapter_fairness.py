r"""
Fairness compliance tests for adapter BaseAdapter usage.

Verifies that every registered adapter follows the rules in FAIRNESS.md:
- Inherits from BaseAdapter
- Calls super() in overridden fallback methods (so native fails gracefully)
- Does not override shared helpers (_build_networkx_graph)
- Does not raise batch_size above 1000
"""

import inspect

import pytest

from graph_bench.adapters.base import AdapterRegistry, BaseAdapter

# All registered adapter names and classes
_ADAPTER_ITEMS = [
    (name, AdapterRegistry.get(name))
    for name in AdapterRegistry.list()
]

# Concrete fallback methods in BaseAdapter that adapters may override
# but MUST call super() somewhere in the override body.
_FALLBACK_METHODS = [
    "traverse_bfs",
    "traverse_dfs",
    "pagerank",
    "community_detection",
    "local_clustering_coefficient",
    "weakly_connected_components",
    "sssp",
    "bfs_levels",
    "vector_search",
    "create_vector_index",
]

# Shared helpers that should never be overridden.
_PRIVATE_HELPERS = [
    "_build_networkx_graph",
    "_cosine_similarity",
    "_synchronous_label_propagation",
]

# Methods whose batch_size default must be <= 1000.
_BATCH_METHODS = [
    "insert_nodes",
    "insert_edges",
]


# ── Parametrize over every registered adapter ──────────────────


def _adapter_ids() -> list[str]:
    return [name for name, _ in _ADAPTER_ITEMS]


def _adapter_classes() -> list[type[BaseAdapter]]:
    return [cls for _, cls in _ADAPTER_ITEMS]


@pytest.mark.parametrize(
    "adapter_cls",
    _adapter_classes(),
    ids=_adapter_ids(),
)
class TestAdapterInheritsBase:
    """Every adapter must be a subclass of BaseAdapter."""

    def test_is_subclass(self, adapter_cls: type[BaseAdapter]) -> None:
        assert issubclass(adapter_cls, BaseAdapter), (
            f"{adapter_cls.__name__} does not inherit from BaseAdapter"
        )


@pytest.mark.parametrize(
    "adapter_cls",
    _adapter_classes(),
    ids=_adapter_ids(),
)
class TestFallbackMethodsCallSuper:
    """Overridden fallback methods must contain a super() call."""

    @pytest.mark.parametrize("method_name", _FALLBACK_METHODS)
    def test_override_calls_super(
        self, adapter_cls: type[BaseAdapter], method_name: str,
    ) -> None:
        base_method = getattr(BaseAdapter, method_name, None)
        adapter_method = getattr(adapter_cls, method_name, None)

        if adapter_method is None or base_method is None:
            pytest.skip(f"{method_name} not defined")

        # If the adapter does not override the method, it inherits the
        # BaseAdapter implementation, which is fine.
        if adapter_method is base_method:
            return

        source = inspect.getsource(adapter_method)
        has_super = "super()" in source or "super()." in source
        has_raise = "raise NotImplementedError" in source

        assert has_super or has_raise, (
            f"{adapter_cls.__name__}.{method_name} overrides BaseAdapter "
            f"but does not call super() or raise NotImplementedError. "
            f"Native overrides must fall back to super() when the native "
            f"implementation is unavailable (see FAIRNESS.md)."
        )


@pytest.mark.parametrize(
    "adapter_cls",
    _adapter_classes(),
    ids=_adapter_ids(),
)
class TestPrivateHelpersNotOverridden:
    """Shared helpers must not be overridden by adapters."""

    @pytest.mark.parametrize("method_name", _PRIVATE_HELPERS)
    def test_not_overridden(
        self, adapter_cls: type[BaseAdapter], method_name: str,
    ) -> None:
        base_method = getattr(BaseAdapter, method_name, None)
        adapter_method = getattr(adapter_cls, method_name, None)

        if base_method is None:
            pytest.skip(f"{method_name} not on BaseAdapter")

        assert adapter_method is base_method, (
            f"{adapter_cls.__name__} overrides {method_name} which is a "
            f"shared BaseAdapter helper. Do not override this method "
            f"(see FAIRNESS.md)."
        )


@pytest.mark.parametrize(
    "adapter_cls",
    _adapter_classes(),
    ids=_adapter_ids(),
)
class TestBatchSizeDefaults:
    """batch_size defaults must not exceed 1000."""

    @pytest.mark.parametrize("method_name", _BATCH_METHODS)
    def test_batch_size_within_limit(
        self, adapter_cls: type[BaseAdapter], method_name: str,
    ) -> None:
        method = getattr(adapter_cls, method_name, None)
        if method is None:
            pytest.skip(f"{method_name} not defined")

        sig = inspect.signature(method)
        param = sig.parameters.get("batch_size")
        if param is None or param.default is inspect.Parameter.empty:
            pytest.skip(f"{method_name} has no batch_size default")

        assert param.default <= 1000, (
            f"{adapter_cls.__name__}.{method_name} has batch_size={param.default}, "
            f"which exceeds the maximum of 1000 (see FAIRNESS.md)."
        )
