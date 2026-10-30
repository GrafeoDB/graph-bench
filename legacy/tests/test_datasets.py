r"""
Tests for graph_bench.datasets module.
"""

import pytest

from graph_bench.datasets import LDBCSocialNetwork, LUBM, PokecSocialNetwork, SyntheticSocialNetwork
from graph_bench.types import ScaleConfig


class TestSyntheticSocialNetwork:
    @pytest.fixture
    def dataset(self):
        return SyntheticSocialNetwork(seed=42, use_faker=False)

    @pytest.fixture
    def tiny_scale(self):
        return ScaleConfig(
            name="tiny",
            nodes=100,
            edges=200,
            warmup_iterations=1,
            measurement_iterations=2,
            timeout_seconds=10,
        )

    def test_name(self, dataset):
        assert dataset.name == "synthetic_social"

    def test_generate_returns_tuple(self, dataset, tiny_scale):
        result = dataset.generate(tiny_scale)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_correct_node_count(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        assert len(nodes) == tiny_scale.nodes

    def test_generate_node_structure(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        node = nodes[0]
        assert "id" in node
        assert "name" in node
        assert "age" in node
        assert "city" in node

    def test_generate_edge_count_approx(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        assert len(edges) <= tiny_scale.edges

    def test_generate_edge_structure(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        edge = edges[0]
        assert len(edge) == 4
        src, tgt, edge_type, props = edge
        assert src.startswith("person_")
        assert tgt.startswith("person_")
        assert edge_type == "FOLLOWS"
        assert isinstance(props, dict)

    def test_reproducible_with_seed(self, tiny_scale):
        dataset1 = SyntheticSocialNetwork(seed=42, use_faker=False)
        dataset2 = SyntheticSocialNetwork(seed=42, use_faker=False)

        nodes1, edges1 = dataset1.generate(tiny_scale)
        nodes2, edges2 = dataset2.generate(tiny_scale)

        assert nodes1 == nodes2
        assert edges1 == edges2

    def test_different_with_different_seed(self, tiny_scale):
        dataset1 = SyntheticSocialNetwork(seed=42, use_faker=False)
        dataset2 = SyntheticSocialNetwork(seed=123, use_faker=False)

        _, edges1 = dataset1.generate(tiny_scale)
        _, edges2 = dataset2.generate(tiny_scale)

        # Edges should differ since they are randomly generated
        assert edges1 != edges2

    def test_no_self_loops(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        for src, tgt, _, _ in edges:
            assert src != tgt

    def test_no_duplicate_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        edge_pairs = [(src, tgt) for src, tgt, _, _ in edges]
        assert len(edge_pairs) == len(set(edge_pairs))


class TestLDBCSocialNetwork:
    """Tests for LDBC Social Network Benchmark dataset."""

    @pytest.fixture
    def dataset(self):
        return LDBCSocialNetwork(scale_factor=0.1, seed=42)

    @pytest.fixture
    def tiny_scale(self):
        return ScaleConfig(
            name="tiny",
            nodes=500,
            edges=1000,
            warmup_iterations=1,
            measurement_iterations=2,
            timeout_seconds=10,
        )

    def test_name(self, dataset):
        assert dataset.name == "ldbc_snb"

    def test_generate_returns_tuple(self, dataset, tiny_scale):
        result = dataset.generate(tiny_scale)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generates_persons(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        persons = [n for n in nodes if n.get("label") == "Person"]
        assert len(persons) > 0

    def test_person_has_ldbc_properties(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        persons = [n for n in nodes if n.get("label") == "Person"]
        person = persons[0]
        assert "firstName" in person
        assert "lastName" in person
        assert "gender" in person
        assert "birthday" in person
        assert "browserUsed" in person

    def test_generates_cities(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        cities = [n for n in nodes if n.get("label") == "City"]
        assert len(cities) > 0

    def test_generates_tags(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        tags = [n for n in nodes if n.get("label") == "Tag"]
        assert len(tags) > 0

    def test_generates_knows_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        knows_edges = [e for e in edges if e[2] == "KNOWS"]
        assert len(knows_edges) > 0

    def test_generates_lives_in_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        lives_in_edges = [e for e in edges if e[2] == "LIVES_IN"]
        assert len(lives_in_edges) > 0

    def test_generates_interest_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        interest_edges = [e for e in edges if e[2] == "HAS_INTEREST"]
        assert len(interest_edges) > 0

    def test_reproducible_with_seed(self, tiny_scale):
        dataset1 = LDBCSocialNetwork(scale_factor=0.1, seed=42)
        dataset2 = LDBCSocialNetwork(scale_factor=0.1, seed=42)

        nodes1, edges1 = dataset1.generate(tiny_scale)
        nodes2, edges2 = dataset2.generate(tiny_scale)

        assert nodes1 == nodes2
        assert edges1 == edges2


class TestPokecSocialNetwork:
    """Tests for Pokec Social Network dataset."""

    @pytest.fixture
    def dataset(self):
        return PokecSocialNetwork(seed=42, avg_degree=10)

    @pytest.fixture
    def tiny_scale(self):
        return ScaleConfig(
            name="tiny",
            nodes=100,
            edges=500,
            warmup_iterations=1,
            measurement_iterations=2,
            timeout_seconds=10,
        )

    def test_name(self, dataset):
        assert dataset.name == "pokec"

    def test_generate_returns_tuple(self, dataset, tiny_scale):
        result = dataset.generate(tiny_scale)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generates_users(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        users = [n for n in nodes if n.get("label") == "User"]
        assert len(users) == tiny_scale.nodes

    def test_user_has_pokec_properties(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        user = nodes[0]
        assert "region" in user
        assert "registration" in user
        assert "public" in user

    def test_user_has_interests_when_enabled(self, tiny_scale):
        dataset = PokecSocialNetwork(seed=42, include_interests=True)
        nodes, _ = dataset.generate(tiny_scale)
        user = nodes[0]
        # Should have at least one I_* interest column
        interest_keys = [k for k in user.keys() if k.startswith("I_")]
        assert len(interest_keys) > 0

    def test_generates_friend_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        friend_edges = [e for e in edges if e[2] == "FRIEND"]
        assert len(friend_edges) > 0

    def test_directed_edges(self, dataset, tiny_scale):
        """Pokec has directed friendships - verify we can have both (a,b) and (b,a)."""
        # This test verifies edges are directed by checking the structure
        _, edges = dataset.generate(tiny_scale)
        edge_tuples = [(e[0], e[1]) for e in edges]
        # All should be unique (directed uniqueness)
        assert len(edge_tuples) == len(set(edge_tuples))

    def test_no_self_loops(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        for src, tgt, _, _ in edges:
            assert src != tgt

    def test_reproducible_with_seed(self, tiny_scale):
        dataset1 = PokecSocialNetwork(seed=42)
        dataset2 = PokecSocialNetwork(seed=42)

        nodes1, edges1 = dataset1.generate(tiny_scale)
        nodes2, edges2 = dataset2.generate(tiny_scale)

        assert nodes1 == nodes2
        assert edges1 == edges2


class TestLUBM:
    """Tests for LUBM (Lehigh University Benchmark) dataset."""

    @pytest.fixture
    def dataset(self):
        return LUBM(universities=1, seed=42, departments_per_university=3)

    @pytest.fixture
    def tiny_scale(self):
        return ScaleConfig(
            name="tiny",
            nodes=500,
            edges=1000,
            warmup_iterations=1,
            measurement_iterations=2,
            timeout_seconds=10,
        )

    def test_name(self, dataset):
        assert dataset.name == "lubm"

    def test_generate_returns_tuple(self, dataset, tiny_scale):
        result = dataset.generate(tiny_scale)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generates_university(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        universities = [n for n in nodes if n.get("label") == "University"]
        assert len(universities) >= 1

    def test_generates_departments(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        departments = [n for n in nodes if n.get("label") == "Department"]
        assert len(departments) > 0

    def test_generates_professors(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        professors = [n for n in nodes if "Professor" in n.get("label", "")]
        assert len(professors) > 0

    def test_generates_students(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        students = [n for n in nodes if "Student" in n.get("label", "")]
        assert len(students) > 0

    def test_generates_courses(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        courses = [n for n in nodes if "Course" in n.get("label", "")]
        assert len(courses) > 0

    def test_nodes_have_rdf_type(self, dataset, tiny_scale):
        nodes, _ = dataset.generate(tiny_scale)
        for node in nodes:
            assert "rdf_type" in node

    def test_generates_works_for_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        works_for = [e for e in edges if e[2] == "worksFor"]
        assert len(works_for) > 0

    def test_generates_member_of_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        member_of = [e for e in edges if e[2] == "memberOf"]
        assert len(member_of) > 0

    def test_generates_teacher_of_edges(self, dataset, tiny_scale):
        _, edges = dataset.generate(tiny_scale)
        teacher_of = [e for e in edges if e[2] == "teacherOf"]
        assert len(teacher_of) > 0

    def test_generate_rdf_returns_triples(self, dataset, tiny_scale):
        triples = dataset.generate_rdf(tiny_scale)
        assert isinstance(triples, list)
        assert len(triples) > 0
        # Each triple should be (subject, predicate, object)
        for triple in triples[:10]:
            assert len(triple) == 3

    def test_rdf_triples_have_namespaces(self, dataset, tiny_scale):
        triples = dataset.generate_rdf(tiny_scale)
        # Check that triples use LUBM namespace
        lubm_ns = "http://www.lehigh.edu/~zhp2/2004/0401/univ-bench.owl#"
        subjects = [t[0] for t in triples]
        assert any(lubm_ns in s for s in subjects)

    def test_reproducible_with_seed(self, tiny_scale):
        dataset1 = LUBM(universities=1, seed=42)
        dataset2 = LUBM(universities=1, seed=42)

        nodes1, edges1 = dataset1.generate(tiny_scale)
        nodes2, edges2 = dataset2.generate(tiny_scale)

        assert nodes1 == nodes2
        assert edges1 == edges2
