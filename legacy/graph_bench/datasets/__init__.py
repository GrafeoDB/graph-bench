r"""
Dataset generators for graph-bench.

Provides synthetic and standard benchmark datasets
for consistent graph database evaluation.

Available datasets:
    - SyntheticSocialNetwork: Simple synthetic social network (LPG)
    - LDBCSocialNetwork: LDBC SNB - industry standard benchmark (LPG)
    - PokecSocialNetwork: Slovak social network dataset (LPG)
    - LUBM: Lehigh University Benchmark - university domain (RDF/LPG)

    from graph_bench.datasets import LDBCSocialNetwork, LUBM

    # LPG dataset
    ldbc = LDBCSocialNetwork(scale_factor=1)
    nodes, edges = ldbc.generate(SCALES["medium"])

    # RDF dataset
    lubm = LUBM(universities=5)
    triples = lubm.generate_rdf(SCALES["medium"])
"""

from graph_bench.datasets.base import BaseDatasetLoader
from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork
from graph_bench.datasets.lubm import LUBM
from graph_bench.datasets.pokec import PokecSocialNetwork
from graph_bench.datasets.synthetic import SyntheticSocialNetwork

__all__ = [
    "BaseDatasetLoader",
    "LDBCSocialNetwork",
    "LUBM",
    "PokecSocialNetwork",
    "SyntheticSocialNetwork",
]
