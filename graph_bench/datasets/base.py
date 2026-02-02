r"""
Base dataset loader interface.

    from graph_bench.datasets.base import BaseDatasetLoader
"""

from abc import ABC, abstractmethod
from typing import Any

from graph_bench.protocols import GraphDatabaseAdapter
from graph_bench.types import ScaleConfig

__all__ = ["BaseDatasetLoader"]


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        ...

    @abstractmethod
    def generate(self, scale: ScaleConfig) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate dataset, returning (nodes, edges)."""
        ...

    def load_into(self, adapter: GraphDatabaseAdapter, scale: ScaleConfig, *, batch_size: int = 1000) -> None:
        """Load dataset directly into adapter."""
        nodes, edges = self.generate(scale)
        adapter.insert_nodes(nodes, label="Person", batch_size=batch_size)
        adapter.insert_edges(edges, batch_size=batch_size)
