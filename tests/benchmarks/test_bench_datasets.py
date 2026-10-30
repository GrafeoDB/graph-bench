r"""
CodSpeed benchmarks for the dataset generators.

Run with:
    uv run pytest tests/benchmarks --codspeed --no-cov
"""

import pytest

from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork
from graph_bench.datasets.lubm import LUBM
from graph_bench.datasets.pokec import PokecSocialNetwork
from graph_bench.datasets.synthetic import SyntheticSocialNetwork
from graph_bench.types import ScaleConfig

SCALE = ScaleConfig(name="bench", nodes=1_000, edges=5_000)


@pytest.mark.benchmark
def test_synthetic_social_generate():
    nodes, edges = SyntheticSocialNetwork(seed=42, use_faker=False).generate(SCALE)
    assert len(nodes) == SCALE.nodes
    assert len(edges) == SCALE.edges


@pytest.mark.benchmark
def test_pokec_generate():
    nodes, edges = PokecSocialNetwork(seed=42).generate(SCALE)
    assert len(nodes) == SCALE.nodes
    assert edges


@pytest.mark.benchmark
def test_ldbc_snb_generate_sf01():
    nodes, edges = LDBCSocialNetwork(scale_factor=0.1, seed=42).generate(SCALE)
    assert nodes
    assert edges


@pytest.mark.benchmark
def test_lubm_generate():
    nodes, edges = LUBM(universities=1, seed=42).generate(SCALE)
    assert nodes
    assert edges


@pytest.mark.benchmark
def test_lubm_generate_rdf():
    triples = LUBM(universities=1, seed=42).generate_rdf(SCALE)
    assert triples
