r"""
LDBC Social Network Benchmark (SNB) dataset.

The LDBC SNB is the industry standard benchmark for graph databases.
It models a social network with realistic data distributions and correlations.

Schema:
    Person -[KNOWS]-> Person
    Person -[LIVES_IN]-> City
    Person -[HAS_INTEREST]-> Tag
    Person -[STUDY_AT]-> University     (props: classYear)
    Person -[WORK_AT]-> Company         (props: workFrom)
    Person -[CREATED]-> Post/Comment    (reverse helper for forward traversal)
    City -[IS_PART_OF]-> Country
    Post -[HAS_CREATOR]-> Person
    Post -[HAS_TAG]-> Tag
    Post -[IS_LOCATED_IN]-> Country
    Post -[IN_FORUM]-> Forum            (reverse helper)
    Comment -[HAS_CREATOR]-> Person
    Comment -[HAS_TAG]-> Tag
    Comment -[IS_LOCATED_IN]-> Country
    Comment -[REPLY_OF]-> Post/Comment
    Forum -[CONTAINER_OF]-> Post
    Forum -[HAS_MODERATOR]-> Person
    Forum -[HAS_MEMBER]-> Person        (props: joinDate)
    University -[IS_LOCATED_IN]-> City
    Company -[IS_LOCATED_IN]-> Country
    Message -[HAS_REPLY]-> Comment      (reverse helper)

Scale Factors:
    SF0.1: ~1K persons, ~18K knows, ~5K posts
    SF1:   ~10K persons, ~180K knows, ~50K posts
    SF10:  ~73K persons, ~2M knows, ~500K posts
    SF100: ~280K persons, ~18M knows, ~5M posts

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

__all__ = ["LDBCSocialNetwork", "scale_name_to_factor"]

# LDBC SNB scale factor mappings (official)
# Reference: https://ldbcouncil.org/ldbc_snb_docs/ldbc-snb-specification.pdf
# These values match the official LDBC SNB datagen output
SCALE_FACTORS = {
    0.1: {"persons": 1_000, "knows_edges": 18_000, "posts": 5_000},
    1: {"persons": 10_000, "knows_edges": 180_000, "posts": 50_000},
    3: {"persons": 27_000, "knows_edges": 540_000, "posts": 150_000},
    10: {"persons": 73_000, "knows_edges": 2_000_000, "posts": 500_000},
    30: {"persons": 180_000, "knows_edges": 6_500_000, "posts": 1_500_000},
    100: {"persons": 280_000, "knows_edges": 18_000_000, "posts": 5_000_000},
}

# Map scale config names to LDBC scale factors
_SCALE_NAME_TO_FACTOR: dict[str, float] = {
    "sf01": 0.1,
    "sf1": 1,
    "sf3": 3,
    "sf10": 10,
    "sf30": 30,
    "sf100": 100,
}


def scale_name_to_factor(name: str) -> float:
    """Convert a scale config name (e.g. 'sf3') to an LDBC scale factor (e.g. 3)."""
    return _SCALE_NAME_TO_FACTOR.get(name, 1)

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

# Sample content for Posts and Comments
POST_CONTENT = [
    "Just visited an amazing new restaurant downtown!",
    "Excited to start my new project at work today.",
    "The weather is beautiful this morning.",
    "Check out this incredible photo I took yesterday.",
    "Thinking about switching careers, any advice?",
    "Happy birthday to my best friend!",
    "Just finished reading an incredible book.",
    "Anyone else watching the game tonight?",
    "New blog post about technology trends.",
    "Can't believe how fast this year is going.",
]

COMMENT_CONTENT = [
    "Great post! Totally agree.",
    "Thanks for sharing this!",
    "I had a similar experience.",
    "Interesting perspective.",
    "That's awesome, congrats!",
    "I disagree, but respect your opinion.",
    "Love this! More please.",
    "Where was this? Looks amazing!",
]

FORUM_TITLES = [
    "Wall of {name}", "Group: Photography Enthusiasts", "Group: Tech News",
    "Group: Travel Stories", "Group: Sports Fans", "Group: Book Club",
    "Group: Music Lovers", "Group: Foodies", "Group: Science Forum",
    "Group: Art Gallery", "Group: Fitness Tips", "Group: Movie Reviews",
]


class LDBCSocialNetwork(BaseDatasetLoader):
    """LDBC Social Network Benchmark dataset generator.

    Generates a social network graph following the full LDBC SNB schema
    with Person, Post, Comment, Forum, Country, University, Company entities
    and all relationships defined in the specification.
    """

    def __init__(
        self,
        *,
        scale_factor: float = 1,
        seed: int | None = None,
    ) -> None:
        self._scale_factor = scale_factor
        self._seed = seed

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
        """Generate full LDBC SNB dataset.

        Generates all entity types and relationships per the LDBC SNB spec.
        Respects scale config node/edge counts while maintaining LDBC distributions.
        """
        if self._seed is not None:
            random.seed(self._seed)

        sf_params = self._get_sf_params()

        # Use scale config to determine actual counts.
        # Posts/comments scale 1:1 with persons (not 5:1 LDBC ratio)
        # to keep setup time reasonable for adapter-based benchmarks.
        person_count = min(scale.nodes, sf_params["persons"])
        knows_count = min(scale.edges, sf_params["knows_edges"])
        post_count = min(person_count, sf_params["posts"])
        comment_count = post_count * 2
        forum_count = max(10, person_count // 10)

        # ── Nodes ──
        nodes: list[dict[str, Any]] = []

        persons = self._generate_persons(person_count)
        nodes.extend(persons)

        cities = self._generate_cities()
        nodes.extend(cities)

        tags = self._generate_tags()
        nodes.extend(tags)

        countries = self._generate_countries()
        nodes.extend(countries)

        universities = self._generate_universities()
        nodes.extend(universities)

        companies = self._generate_companies()
        nodes.extend(companies)

        forums = self._generate_forums(forum_count, person_count)
        nodes.extend(forums)

        posts = self._generate_posts(post_count, person_count, len(countries))
        nodes.extend(posts)

        comments = self._generate_comments(comment_count, person_count, post_count, len(countries))
        nodes.extend(comments)

        # ── Edges ──
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        # Person-Person: KNOWS (power-law)
        edges.extend(self._generate_knows_edges(person_count, knows_count))

        # Person-City: LIVES_IN
        edges.extend(self._generate_lives_in_edges(person_count, len(cities)))

        # Person-Tag: HAS_INTEREST
        edges.extend(self._generate_interest_edges(person_count, len(tags)))

        # City-Country: IS_PART_OF
        edges.extend(self._generate_city_country_edges())

        # Person-University: STUDY_AT (with classYear)
        edges.extend(self._generate_study_at_edges(person_count, len(universities)))

        # Person-Company: WORK_AT (with workFrom)
        edges.extend(self._generate_work_at_edges(person_count, len(companies)))

        # University-City: IS_LOCATED_IN
        edges.extend(self._generate_university_location_edges(len(universities), len(cities)))

        # Company-Country: IS_LOCATED_IN
        edges.extend(self._generate_company_location_edges(len(companies), len(countries)))

        # Post edges: HAS_CREATOR, HAS_TAG, IS_LOCATED_IN, CONTAINER_OF, reverse helpers
        edges.extend(self._generate_post_edges(
            post_count, person_count, len(tags), len(countries), forum_count,
        ))

        # Comment edges: HAS_CREATOR, REPLY_OF, HAS_TAG, IS_LOCATED_IN, reverse helpers
        edges.extend(self._generate_comment_edges(
            comment_count, person_count, post_count, len(tags), len(countries),
        ))

        # Forum edges: HAS_MODERATOR, HAS_MEMBER
        edges.extend(self._generate_forum_membership_edges(forum_count, person_count))

        return nodes, edges

    # ─── Node generators ───────────────────────────────────────────────

    def _generate_persons(self, count: int) -> list[dict[str, Any]]:
        """Generate Person nodes with LDBC-realistic properties."""
        persons = []
        base_date = datetime(1940, 1, 1)
        creation_start = datetime(2010, 1, 1)

        for i in range(count):
            gender = "male" if random.random() < 0.5 else "female"
            first_names = FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE

            birthday = base_date + timedelta(days=random.randint(0, 30000))
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

    def _generate_countries(self) -> list[dict[str, Any]]:
        """Generate Country nodes (derived from CITIES list)."""
        seen: dict[str, int] = {}
        countries = []
        for _, country_name in CITIES:
            if country_name not in seen:
                idx = len(seen)
                seen[country_name] = idx
                countries.append({
                    "id": f"country_{idx}",
                    "label": "Country",
                    "name": country_name,
                })
        return countries

    def _generate_universities(self) -> list[dict[str, Any]]:
        """Generate University nodes."""
        return [
            {"id": f"university_{i}", "label": "University", "name": name}
            for i, name in enumerate(UNIVERSITIES)
        ]

    def _generate_companies(self) -> list[dict[str, Any]]:
        """Generate Company nodes."""
        return [
            {"id": f"company_{i}", "label": "Company", "name": name}
            for i, name in enumerate(COMPANIES)
        ]

    def _generate_forums(self, count: int, person_count: int) -> list[dict[str, Any]]:
        """Generate Forum nodes."""
        creation_start = datetime(2010, 1, 1)
        forums = []
        for i in range(count):
            creation_date = creation_start + timedelta(days=random.randint(0, 4000))
            title_template = random.choice(FORUM_TITLES)
            if "{name}" in title_template:
                person_idx = random.randint(0, person_count - 1)
                title = title_template.format(name=f"Person_{person_idx}")
            else:
                title = title_template
            forums.append({
                "id": f"forum_{i}",
                "label": "Forum",
                "title": title,
                "creationDate": creation_date.isoformat(),
            })
        return forums

    def _generate_posts(
        self, count: int, person_count: int, country_count: int,
    ) -> list[dict[str, Any]]:
        """Generate Post nodes with LDBC properties."""
        creation_start = datetime(2010, 1, 1)
        posts = []
        for i in range(count):
            creation_date = creation_start + timedelta(days=random.randint(0, 4000))
            has_image = random.random() < 0.3
            content = "" if has_image else random.choice(POST_CONTENT)
            image_file = f"photo{random.randint(1, 1000)}.jpg" if has_image else ""
            posts.append({
                "id": f"post_{i}",
                "label": "Post",
                "content": content,
                "imageFile": image_file,
                "creationDate": creation_date.isoformat(),
                "length": len(content),
                "creatorId": f"person_{random.randint(0, person_count - 1)}",
                "countryId": f"country_{random.randint(0, country_count - 1)}",
            })
        return posts

    def _generate_comments(
        self, count: int, person_count: int, post_count: int, country_count: int,
    ) -> list[dict[str, Any]]:
        """Generate Comment nodes with LDBC properties."""
        creation_start = datetime(2010, 6, 1)  # comments start after posts
        comments = []
        for i in range(count):
            creation_date = creation_start + timedelta(days=random.randint(0, 4000))
            content = random.choice(COMMENT_CONTENT)
            # 70% reply to a post, 30% reply to another comment
            if i > 0 and random.random() < 0.3:
                reply_to = f"comment_{random.randint(0, max(0, i - 1))}"
            else:
                reply_to = f"post_{random.randint(0, post_count - 1)}"
            comments.append({
                "id": f"comment_{i}",
                "label": "Comment",
                "content": content,
                "creationDate": creation_date.isoformat(),
                "length": len(content),
                "creatorId": f"person_{random.randint(0, person_count - 1)}",
                "countryId": f"country_{random.randint(0, country_count - 1)}",
                "replyOf": reply_to,
            })
        return comments

    # ─── Edge generators ───────────────────────────────────────────────

    def _generate_knows_edges(
        self, person_count: int, edge_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate KNOWS edges with power-law distribution."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        edge_set: set[tuple[int, int]] = set()

        influencer_count = max(10, person_count // 100)
        influencers = list(range(influencer_count))

        creation_start = datetime(2010, 1, 1)
        attempts = 0
        max_attempts = edge_count * 3

        while len(edges) < edge_count and attempts < max_attempts:
            attempts += 1

            if random.random() < 0.4 and influencers:
                if random.random() < 0.5:
                    src = random.choice(influencers)
                    tgt = random.randint(0, person_count - 1)
                else:
                    src = random.randint(0, person_count - 1)
                    tgt = random.choice(influencers)
            else:
                src = random.randint(0, person_count - 1)
                tgt = random.randint(0, person_count - 1)

            if src == tgt:
                continue

            edge_key = (min(src, tgt), max(src, tgt))
            if edge_key in edge_set:
                continue

            edge_set.add(edge_key)

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
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(person_count):
            city_idx = random.randint(0, city_count - 1)
            edges.append((f"person_{i}", f"city_{city_idx}", "LIVES_IN", {}))
        return edges

    def _generate_interest_edges(
        self, person_count: int, tag_count: int
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate HAS_INTEREST edges (persons have multiple interests)."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(person_count):
            interest_count = random.randint(1, 5)
            interests = random.sample(range(tag_count), min(interest_count, tag_count))
            for tag_idx in interests:
                edges.append((f"person_{i}", f"tag_{tag_idx}", "HAS_INTEREST", {}))
        return edges

    def _generate_city_country_edges(self) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate IS_PART_OF edges (City -> Country)."""
        country_map: dict[str, int] = {}
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i, (_, country_name) in enumerate(CITIES):
            if country_name not in country_map:
                country_map[country_name] = len(country_map)
            edges.append((f"city_{i}", f"country_{country_map[country_name]}", "IS_PART_OF", {}))
        return edges

    def _generate_study_at_edges(
        self, person_count: int, university_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate STUDY_AT edges (0-1 per person, with classYear)."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(person_count):
            if random.random() < 0.4:  # 40% of persons studied at a university
                uni_idx = random.randint(0, university_count - 1)
                class_year = random.randint(1990, 2020)
                edges.append((
                    f"person_{i}", f"university_{uni_idx}", "STUDY_AT",
                    {"classYear": class_year},
                ))
        return edges

    def _generate_work_at_edges(
        self, person_count: int, company_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate WORK_AT edges (0-2 per person, with workFrom year)."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(person_count):
            job_count = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
            companies_chosen = random.sample(
                range(company_count), min(job_count, company_count),
            )
            for comp_idx in companies_chosen:
                work_from = random.randint(2000, 2024)
                edges.append((
                    f"person_{i}", f"company_{comp_idx}", "WORK_AT",
                    {"workFrom": work_from},
                ))
        return edges

    def _generate_university_location_edges(
        self, university_count: int, city_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate IS_LOCATED_IN edges (University -> City)."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(university_count):
            city_idx = i % city_count
            edges.append((f"university_{i}", f"city_{city_idx}", "IS_LOCATED_IN", {}))
        return edges

    def _generate_company_location_edges(
        self, company_count: int, country_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate IS_LOCATED_IN edges (Company -> Country)."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        for i in range(company_count):
            country_idx = i % country_count
            edges.append((f"company_{i}", f"country_{country_idx}", "IS_LOCATED_IN", {}))
        return edges

    def _generate_post_edges(
        self,
        post_count: int,
        person_count: int,
        tag_count: int,
        country_count: int,
        forum_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate all Post-related edges.

        Spec-direction edges:
          Post -[HAS_CREATOR]-> Person
          Post -[HAS_TAG]-> Tag (1-3 per post)
          Post -[IS_LOCATED_IN]-> Country
          Forum -[CONTAINER_OF]-> Post

        Reverse helper edges (for efficient forward traversal):
          Person -[CREATED]-> Post
          Post -[IN_FORUM]-> Forum
        """
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        for i in range(post_count):
            post_id = f"post_{i}"
            creator_idx = random.randint(0, person_count - 1)
            creator_id = f"person_{creator_idx}"
            country_idx = random.randint(0, country_count - 1)
            forum_idx = i % forum_count

            # Spec-direction: Post -> Person
            edges.append((post_id, creator_id, "HAS_CREATOR", {}))
            # Reverse helper: Person -> Post
            edges.append((creator_id, post_id, "CREATED", {}))

            # Post -> Tag (1-3 tags)
            num_tags = random.randint(1, 3)
            chosen_tags = random.sample(range(tag_count), min(num_tags, tag_count))
            for tag_idx in chosen_tags:
                edges.append((post_id, f"tag_{tag_idx}", "HAS_TAG", {}))

            # Post -> Country
            edges.append((post_id, f"country_{country_idx}", "IS_LOCATED_IN", {}))

            # Forum -> Post (CONTAINER_OF)
            edges.append((f"forum_{forum_idx}", post_id, "CONTAINER_OF", {}))
            # Reverse helper: Post -> Forum
            edges.append((post_id, f"forum_{forum_idx}", "IN_FORUM", {}))

        return edges

    def _generate_comment_edges(
        self,
        comment_count: int,
        person_count: int,
        post_count: int,
        tag_count: int,
        country_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate all Comment-related edges.

        Spec-direction edges:
          Comment -[HAS_CREATOR]-> Person
          Comment -[REPLY_OF]-> Post or Comment
          Comment -[HAS_TAG]-> Tag (0-2 per comment)
          Comment -[IS_LOCATED_IN]-> Country

        Reverse helper edges:
          Person -[CREATED]-> Comment
          Post/Comment -[HAS_REPLY]-> Comment
        """
        edges: list[tuple[str, str, str, dict[str, Any]]] = []

        for i in range(comment_count):
            comment_id = f"comment_{i}"
            creator_idx = random.randint(0, person_count - 1)
            creator_id = f"person_{creator_idx}"
            country_idx = random.randint(0, country_count - 1)

            # Comment -> Person (HAS_CREATOR)
            edges.append((comment_id, creator_id, "HAS_CREATOR", {}))
            # Reverse: Person -> Comment (CREATED)
            edges.append((creator_id, comment_id, "CREATED", {}))

            # Comment -> Post or Comment (REPLY_OF)
            if i > 0 and random.random() < 0.3:
                parent_id = f"comment_{random.randint(0, max(0, i - 1))}"
            else:
                parent_id = f"post_{random.randint(0, post_count - 1)}"
            edges.append((comment_id, parent_id, "REPLY_OF", {}))
            # Reverse: parent -> Comment (HAS_REPLY)
            edges.append((parent_id, comment_id, "HAS_REPLY", {}))

            # Comment -> Tag (0-2 tags)
            num_tags = random.randint(0, 2)
            if num_tags > 0:
                chosen_tags = random.sample(range(tag_count), min(num_tags, tag_count))
                for tag_idx in chosen_tags:
                    edges.append((comment_id, f"tag_{tag_idx}", "HAS_TAG", {}))

            # Comment -> Country
            edges.append((comment_id, f"country_{country_idx}", "IS_LOCATED_IN", {}))

        return edges

    def _generate_forum_membership_edges(
        self, forum_count: int, person_count: int,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Generate Forum HAS_MODERATOR and HAS_MEMBER edges."""
        edges: list[tuple[str, str, str, dict[str, Any]]] = []
        creation_start = datetime(2010, 1, 1)

        for i in range(forum_count):
            forum_id = f"forum_{i}"

            # Each forum has one moderator
            moderator_idx = random.randint(0, person_count - 1)
            edges.append((forum_id, f"person_{moderator_idx}", "HAS_MODERATOR", {}))

            # Each forum has 5-20 members (including the moderator)
            member_count = random.randint(5, min(20, person_count))
            members = set([moderator_idx])
            while len(members) < member_count:
                members.add(random.randint(0, person_count - 1))
            for member_idx in members:
                join_date = creation_start + timedelta(days=random.randint(0, 4000))
                edges.append((
                    forum_id, f"person_{member_idx}", "HAS_MEMBER",
                    {"joinDate": join_date.isoformat()},
                ))

        return edges
