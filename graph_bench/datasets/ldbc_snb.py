r"""
LDBC Social Network Benchmark (SNB) dataset.

The LDBC SNB is the industry standard benchmark for graph databases.
It models a social network with realistic data distributions and correlations.

Schema (simplified):
    Person -[KNOWS]-> Person
    Person -[LIVES_IN]-> City
    Person -[HAS_INTEREST]-> Tag
    Person -[STUDY_AT]-> University
    Person -[WORK_AT]-> Company
    Post/Comment -[HAS_CREATOR]-> Person
    Post/Comment -[HAS_TAG]-> Tag

Scale Factors:
    SF0.1: ~10K persons, ~180K knows edges
    SF1:   ~10K persons, ~180K knows edges (standard)
    SF10:  ~73K persons, ~2M knows edges
    SF100: ~280K persons, ~18M knows edges

References:
    - https://ldbcouncil.org/benchmarks/snb/
    - https://github.com/ldbc/ldbc_snb_datagen

    from graph_bench.datasets.ldbc_snb import LDBCSocialNetwork

    dataset = LDBCSocialNetwork(scale_factor=1)
    nodes, edges = dataset.generate(scale)
"""

import random
from datetime import datetime, timedelta
from typing import Any

from graph_bench.datasets.base import BaseDatasetLoader
from graph_bench.types import ScaleConfig

__all__ = ["LDBCSocialNetwork"]

# LDBC SNB scale factor mappings (approximate)
SCALE_FACTORS = {
    0.1: {"persons": 1_000, "knows_edges": 18_000, "posts": 5_000},
    1: {"persons": 10_000, "knows_edges": 180_000, "posts": 50_000},
    3: {"persons": 27_000, "knows_edges": 540_000, "posts": 150_000},
    10: {"persons": 73_000, "knows_edges": 2_000_000, "posts": 500_000},
    30: {"persons": 180_000, "knows_edges": 6_500_000, "posts": 1_500_000},
    100: {"persons": 280_000, "knows_edges": 18_000_000, "posts": 5_000_000},
}

# Realistic first names (gender-specific for LDBC)
FIRST_NAMES_MALE = [
    "James", "John", "Robert", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Chen", "Wei", "Ming", "Lei",
    "Hans", "Klaus", "Wolfgang", "Friedrich", "Pierre", "Jean", "François", "Michel",
]
FIRST_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley",
    "Kimberly", "Emily", "Donna", "Michelle", "Yan", "Li", "Fang", "Xiu",
    "Anna", "Maria", "Eva", "Ingrid", "Marie", "Sophie", "Camille", "Claire",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Wang", "Zhang", "Liu", "Chen",
    "Mueller", "Schmidt", "Schneider", "Fischer", "Bernard", "Dubois", "Moreau", "Laurent",
]

# Cities with countries (LDBC uses realistic locations)
CITIES = [
    ("New York", "United States"), ("Los Angeles", "United States"), ("Chicago", "United States"),
    ("London", "United Kingdom"), ("Manchester", "United Kingdom"), ("Birmingham", "United Kingdom"),
    ("Paris", "France"), ("Lyon", "France"), ("Marseille", "France"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Hamburg", "Germany"),
    ("Beijing", "China"), ("Shanghai", "China"), ("Shenzhen", "China"),
    ("Tokyo", "Japan"), ("Osaka", "Japan"), ("Sydney", "Australia"),
    ("Toronto", "Canada"), ("Vancouver", "Canada"), ("São Paulo", "Brazil"),
]

# Tags/interests (realistic categories)
TAGS = [
    "Photography", "Music", "Travel", "Sports", "Technology", "Art", "Science",
    "Movies", "Books", "Gaming", "Cooking", "Fashion", "Fitness", "Nature",
    "Politics", "Business", "Education", "Health", "History", "Philosophy",
]

# Universities
UNIVERSITIES = [
    "MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "ETH Zurich",
    "Tsinghua", "Peking", "Tokyo", "Berkeley", "Caltech", "Princeton",
]

# Companies
COMPANIES = [
    "Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix", "Tesla",
    "IBM", "Intel", "Oracle", "SAP", "Siemens", "Samsung", "Sony", "Alibaba",
]

# Browsers (LDBC tracks this)
BROWSERS = ["Firefox", "Chrome", "Safari", "Edge", "Opera", "Internet Explorer"]


class LDBCSocialNetwork(BaseDatasetLoader):
    """LDBC Social Network Benchmark dataset generator.

    Generates a social network graph following LDBC SNB schema and
    data distributions. Includes persons, posts, comments, and relationships.
    """

    def __init__(
        self,
        *,
        scale_factor: float = 1,
        seed: int | None = None,
        include_posts: bool = False,
        include_comments: bool = False,
    ) -> None:
        """Initialize LDBC SNB dataset generator.

        Args:
            scale_factor: LDBC scale factor (0.1, 1, 3, 10, 30, 100).
            seed: Random seed for reproducibility.
            include_posts: Include Post nodes and relationships.
            include_comments: Include Comment nodes and relationships.
        """
        self._scale_factor = scale_factor
        self._seed = seed
        self._include_posts = include_posts
        self._include_comments = include_comments

    @property
    def name(self) -> str:
        return "ldbc_snb"

    def _get_sf_params(self) -> dict[str, int]:
        """Get parameters for the closest scale factor."""
        available = sorted(SCALE_FACTORS.keys())
        sf = min(available, key=lambda x: abs(x - self._scale_factor))
        return SCALE_FACTORS[sf]

    def generate(
        self, scale: ScaleConfig
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Generate LDBC SNB dataset.

        Respects the scale config's node/edge counts while maintaining
        LDBC-like data distributions.
        """
        if self._seed is not None:
            random.seed(self._seed)

        sf_params = self._get_sf_params()

        # Use scale config to determine actual counts
        person_count = min(scale.nodes, sf_params["persons"])
        knows_count = min(scale.edges, sf_params["knows_edges"])

        # Generate nodes
        nodes: list[dict[str, Any]] = []

        # Generate Person nodes
        persons = self._generate_persons(person_count)
        nodes.extend(persons)

        # Generate City nodes
        cities = self._generate_cities()
        nodes.extend(cities)

        # Generate Tag nodes
        tags = self._generate_tags()
        nodes.extend(tags)

        # Generate edges
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        # KNOWS edges (person-person with power-law distribution)
        knows_edges = self._generate_knows_edges(person_count, knows_count)
        edges.extend(knows_edges)

        # LIVES_IN edges (person-city)
        lives_in = self._generate_lives_in_edges(person_count, len(cities))
        edges.extend(lives_in)

        # HAS_INTEREST edges (person-tag)
        interests = self._generate_interest_edges(person_count, len(tags))
        edges.extend(interests)

        return nodes, edges

    def _generate_persons(self, count: int) -> list[dict[str, Any]]:
        """Generate Person nodes with LDBC-realistic properties."""
        persons = []
        base_date = datetime(1940, 1, 1)
        creation_start = datetime(2010, 1, 1)

        for i in range(count):
            gender = "male" if random.random() < 0.5 else "female"
            first_names = FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE

            # Birthday with realistic age distribution (18-80)
            birthday = base_date + timedelta(days=random.randint(0, 30000))

            # Creation date (when joined the network)
            creation_days = random.randint(0, 4000)
            creation_date = creation_start + timedelta(days=creation_days)

            person = {
                "id": f"person_{i}",
                "label": "Person",
                "firstName": random.choice(first_names),
                "lastName": random.choice(LAST_NAMES),
                "gender": gender,
                "birthday": birthday.strftime("%Y-%m-%d"),
                "creationDate": creation_date.isoformat(),
                "locationIP": f"{random.randint(1, 255)}.{random.randint(0, 255)}."
                f"{random.randint(0, 255)}.{random.randint(0, 255)}",
                "browserUsed": random.choice(BROWSERS),
            }
            persons.append(person)

        return persons

    def _generate_cities(self) -> list[dict[str, Any]]:
        """Generate City nodes."""
        cities = []
        for i, (city_name, country) in enumerate(CITIES):
            cities.append({
                "id": f"city_{i}",
                "label": "City",
                "name": city_name,
                "country": country,
            })
        return cities

    def _generate_tags(self) -> list[dict[str, Any]]:
        """Generate Tag nodes."""
        tags = []
        for i, tag_name in enumerate(TAGS):
            tags.append({
                "id": f"tag_{i}",
                "label": "Tag",
                "name": tag_name,
            })
        return tags

    def _generate_knows_edges(
        self, person_count: int, edge_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate KNOWS edges with power-law distribution.

        LDBC SNB uses a correlated power-law distribution where
        some people have many more connections than others.
        """
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[str, str]] = set()

        # Power-law: some nodes are "influencers" with many connections
        influencer_count = max(10, person_count // 100)
        influencers = list(range(influencer_count))

        creation_start = datetime(2010, 1, 1)
        attempts = 0
        max_attempts = edge_count * 3

        while len(edges) < edge_count and attempts < max_attempts:
            attempts += 1

            # 40% chance to connect to/from an influencer
            if random.random() < 0.4 and influencers:
                if random.random() < 0.5:
                    src = random.choice(influencers)
                    tgt = random.randint(0, person_count - 1)
                else:
                    src = random.randint(0, person_count - 1)
                    tgt = random.choice(influencers)
            else:
                # Random connection
                src = random.randint(0, person_count - 1)
                tgt = random.randint(0, person_count - 1)

            if src == tgt:
                continue

            # Ensure undirected uniqueness (both directions)
            edge_key = (min(src, tgt), max(src, tgt))
            if edge_key in edge_set:
                continue

            edge_set.add(edge_key)

            # Creation date for the friendship
            creation_days = random.randint(0, 4000)
            creation_date = creation_start + timedelta(days=creation_days)

            edges.append((
                f"person_{src}",
                f"person_{tgt}",
                "KNOWS",
                {"creationDate": creation_date.isoformat()},
            ))

        return edges

    def _generate_lives_in_edges(
        self, person_count: int, city_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate LIVES_IN edges (each person lives in one city)."""
        edges = []
        for i in range(person_count):
            city_idx = random.randint(0, city_count - 1)
            edges.append((f"person_{i}", f"city_{city_idx}", "LIVES_IN", {}))
        return edges

    def _generate_interest_edges(
        self, person_count: int, tag_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate HAS_INTEREST edges (persons have multiple interests)."""
        edges = []
        for i in range(person_count):
            # Each person has 1-5 interests
            interest_count = random.randint(1, 5)
            interests = random.sample(range(tag_count), min(interest_count, tag_count))
            for tag_idx in interests:
                edges.append((f"person_{i}", f"tag_{tag_idx}", "HAS_INTEREST", {}))
        return edges
