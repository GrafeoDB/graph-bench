"""LDBC SNB-like social network generator (ported from the legacy generator).

Edges keep LDBC direction only; reverse helper edges (CREATED, IN_FORUM, HAS_REPLY) are dropped.
Dates are INT64 epoch milliseconds (UTC). Output is deterministic for a given (scale, seed).
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from graph_bench.datasets.schema import Column, Dataset, EdgeTable, NodeTable

# (persons, knows edges), from legacy sf01 / sf1.
SCALES: dict[str, tuple[int, int]] = {"small": (1_000, 18_000), "medium": (10_000, 180_000)}

EDGE_TYPES = (
    "KNOWS",
    "LIVES_IN",
    "HAS_INTEREST",
    "IS_PART_OF",
    "STUDY_AT",
    "WORK_AT",
    "IS_LOCATED_IN",
    "HAS_CREATOR",
    "HAS_TAG",
    "CONTAINER_OF",
    "REPLY_OF",
    "HAS_MODERATOR",
    "HAS_MEMBER",
)

# fmt: off
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
CITIES = [
    ("New York", "United States"), ("Los Angeles", "United States"), ("Chicago", "United States"),
    ("London", "United Kingdom"), ("Manchester", "United Kingdom"), ("Birmingham", "United Kingdom"),
    ("Paris", "France"), ("Lyon", "France"), ("Marseille", "France"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Hamburg", "Germany"),
    ("Beijing", "China"), ("Shanghai", "China"), ("Shenzhen", "China"),
    ("Tokyo", "Japan"), ("Osaka", "Japan"), ("Sydney", "Australia"),
    ("Toronto", "Canada"), ("Vancouver", "Canada"), ("São Paulo", "Brazil"),
]
TAGS = [
    "Photography", "Music", "Travel", "Sports", "Technology", "Art", "Science",
    "Movies", "Books", "Gaming", "Cooking", "Fashion", "Fitness", "Nature",
    "Politics", "Business", "Education", "Health", "History", "Philosophy",
]
UNIVERSITIES = [
    "MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "ETH Zurich",
    "Tsinghua", "Peking", "Tokyo", "Berkeley", "Caltech", "Princeton",
]
COMPANIES = [
    "Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix", "Tesla",
    "IBM", "Intel", "Oracle", "SAP", "Siemens", "Samsung", "Sony", "Alibaba",
]
FORUM_TITLES = [
    "Wall of {name}", "Group: Photography Enthusiasts", "Group: Tech News",
    "Group: Travel Stories", "Group: Sports Fans", "Group: Book Club",
    "Group: Music Lovers", "Group: Foodies", "Group: Science Forum",
    "Group: Art Gallery", "Group: Fitness Tips", "Group: Movie Reviews",
]
# fmt: on
BROWSERS = ["Firefox", "Chrome", "Safari", "Edge", "Opera", "Internet Explorer"]
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

_I = "INT64"
_S = "STRING"


def _cols(*pairs: tuple[str, str]) -> tuple[Column, ...]:
    return tuple(Column(n, t) for n, t in pairs)  # type: ignore[arg-type]


_NODE_SCHEMAS: dict[str, tuple[Column, ...]] = {
    "Person": _cols(
        ("id", _S),
        ("firstName", _S),
        ("lastName", _S),
        ("gender", _S),
        ("birthday", _I),
        ("creationDate", _I),
        ("locationIP", _S),
        ("browserUsed", _S),
    ),
    "City": _cols(("id", _S), ("name", _S), ("country", _S)),
    "Tag": _cols(("id", _S), ("name", _S)),
    "Country": _cols(("id", _S), ("name", _S)),
    "University": _cols(("id", _S), ("name", _S)),
    "Company": _cols(("id", _S), ("name", _S)),
    "Forum": _cols(("id", _S), ("title", _S), ("creationDate", _I)),
    "Post": _cols(("id", _S), ("content", _S), ("imageFile", _S), ("creationDate", _I), ("length", _I)),
    "Comment": _cols(("id", _S), ("content", _S), ("creationDate", _I), ("length", _I)),
}
_EDGE_PROPS: dict[str, tuple[Column, ...]] = {
    "KNOWS": _cols(("creationDate", _I)),
    "STUDY_AT": _cols(("classYear", _I)),
    "WORK_AT": _cols(("workFrom", _I)),
    "HAS_MEMBER": _cols(("joinDate", _I)),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

Edge = tuple[str, str, str, dict[str, Any]]


def _ms(dt: datetime) -> int:
    return (dt.replace(tzinfo=UTC) - _EPOCH) // timedelta(milliseconds=1)


def _label_of(node_id: str) -> str:
    return node_id.split("_", 1)[0].capitalize()


class _Gen:
    """Holds the seeded RNG and builds node rows and edge tuples."""

    def __init__(self, persons: int, knows: int, seed: int) -> None:
        self.rng = random.Random(seed)
        self.persons = persons
        self.knows = knows
        self.posts = persons
        self.comments = persons * 2
        self.forums = max(10, persons // 10)
        self.countries = list(dict.fromkeys(c for _, c in CITIES))
        self.nodes: dict[str, list[dict[str, Any]]] = {}
        self.edges: list[Edge] = []

    def _date(self, start: datetime, days: int = 4000) -> int:
        return _ms(start + timedelta(days=self.rng.randint(0, days)))

    def run(self) -> None:
        self._gen_nodes()
        self._gen_edges()

    def _gen_nodes(self) -> None:
        r = self.rng
        start = datetime(2010, 1, 1)
        persons = []
        for i in range(self.persons):
            gender = "male" if r.random() < 0.5 else "female"
            names = FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE
            birthday = _ms(datetime(1940, 1, 1) + timedelta(days=r.randint(0, 30000)))
            persons.append(
                {
                    "id": f"person_{i}",
                    "firstName": r.choice(names),
                    "lastName": r.choice(LAST_NAMES),
                    "gender": gender,
                    "birthday": birthday,
                    "creationDate": self._date(start),
                    "locationIP": f"{r.randint(1, 255)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(0, 255)}",
                    "browserUsed": r.choice(BROWSERS),
                }
            )
        self.nodes["Person"] = persons
        self.nodes["City"] = [{"id": f"city_{i}", "name": c, "country": k} for i, (c, k) in enumerate(CITIES)]
        self.nodes["Tag"] = [{"id": f"tag_{i}", "name": n} for i, n in enumerate(TAGS)]
        self.nodes["Country"] = [{"id": f"country_{i}", "name": n} for i, n in enumerate(self.countries)]
        self.nodes["University"] = [{"id": f"university_{i}", "name": n} for i, n in enumerate(UNIVERSITIES)]
        self.nodes["Company"] = [{"id": f"company_{i}", "name": n} for i, n in enumerate(COMPANIES)]
        forums = []
        for i in range(self.forums):
            created = self._date(start)
            template = r.choice(FORUM_TITLES)
            title = template
            if "{name}" in template:
                title = template.format(name=f"Person_{r.randint(0, self.persons - 1)}")
            forums.append({"id": f"forum_{i}", "title": title, "creationDate": created})
        self.nodes["Forum"] = forums
        posts = []
        for i in range(self.posts):
            created = self._date(start)
            has_image = r.random() < 0.3
            content = "" if has_image else r.choice(POST_CONTENT)
            image = f"photo{r.randint(1, 1000)}.jpg" if has_image else ""
            posts.append(
                {
                    "id": f"post_{i}",
                    "content": content,
                    "imageFile": image,
                    "creationDate": created,
                    "length": len(content),
                }
            )
        self.nodes["Post"] = posts
        comments = []
        for i in range(self.comments):
            created = self._date(datetime(2010, 6, 1))
            content = r.choice(COMMENT_CONTENT)
            comments.append({"id": f"comment_{i}", "content": content, "creationDate": created, "length": len(content)})
        self.nodes["Comment"] = comments

    def _add(self, src: str, dst: str, etype: str, **props: Any) -> None:
        self.edges.append((src, dst, etype, props))

    def _gen_edges(self) -> None:
        r = self.rng
        n = self.persons
        start = datetime(2010, 1, 1)
        self._gen_knows(start)
        for i in range(n):
            self._add(f"person_{i}", f"city_{r.randint(0, len(CITIES) - 1)}", "LIVES_IN")
        for i in range(n):
            for t in r.sample(range(len(TAGS)), min(r.randint(1, 5), len(TAGS))):
                self._add(f"person_{i}", f"tag_{t}", "HAS_INTEREST")
        seen: dict[str, int] = {}
        for i, (_, country) in enumerate(CITIES):
            seen.setdefault(country, len(seen))
            self._add(f"city_{i}", f"country_{seen[country]}", "IS_PART_OF")
        for i in range(n):
            if r.random() < 0.4:
                uni = r.randint(0, len(UNIVERSITIES) - 1)
                self._add(f"person_{i}", f"university_{uni}", "STUDY_AT", classYear=r.randint(1990, 2020))
        for i in range(n):
            jobs = r.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
            for c in r.sample(range(len(COMPANIES)), min(jobs, len(COMPANIES))):
                self._add(f"person_{i}", f"company_{c}", "WORK_AT", workFrom=r.randint(2000, 2024))
        for i in range(len(UNIVERSITIES)):
            self._add(f"university_{i}", f"city_{i % len(CITIES)}", "IS_LOCATED_IN")
        for i in range(len(COMPANIES)):
            self._add(f"company_{i}", f"country_{i % len(self.countries)}", "IS_LOCATED_IN")
        nc = len(self.countries)
        for i in range(self.posts):
            pid = f"post_{i}"
            self._add(pid, f"person_{r.randint(0, n - 1)}", "HAS_CREATOR")
            for t in r.sample(range(len(TAGS)), min(r.randint(1, 3), len(TAGS))):
                self._add(pid, f"tag_{t}", "HAS_TAG")
            self._add(pid, f"country_{r.randint(0, nc - 1)}", "IS_LOCATED_IN")
            self._add(f"forum_{i % self.forums}", pid, "CONTAINER_OF")
        for i in range(self.comments):
            cid = f"comment_{i}"
            self._add(cid, f"person_{r.randint(0, n - 1)}", "HAS_CREATOR")
            if i > 0 and r.random() < 0.3:
                parent = f"comment_{r.randint(0, i - 1)}"
            else:
                parent = f"post_{r.randint(0, self.posts - 1)}"
            self._add(cid, parent, "REPLY_OF")
            for t in r.sample(range(len(TAGS)), min(r.randint(0, 2), len(TAGS))):
                self._add(cid, f"tag_{t}", "HAS_TAG")
            self._add(cid, f"country_{r.randint(0, nc - 1)}", "IS_LOCATED_IN")
        for i in range(self.forums):
            fid = f"forum_{i}"
            moderator = r.randint(0, n - 1)
            self._add(fid, f"person_{moderator}", "HAS_MODERATOR")
            members = {moderator}
            target = r.randint(5, min(20, n))
            while len(members) < target:
                members.add(r.randint(0, n - 1))
            for m in sorted(members):
                self._add(fid, f"person_{m}", "HAS_MEMBER", joinDate=self._date(start))

    def _gen_knows(self, start: datetime) -> None:
        r = self.rng
        n = self.persons
        influencers = list(range(max(10, n // 100)))
        seen: set[tuple[int, int]] = set()
        count = attempts = 0
        while count < self.knows and attempts < self.knows * 3:
            attempts += 1
            if r.random() < 0.4:
                if r.random() < 0.5:
                    src, dst = r.choice(influencers), r.randint(0, n - 1)
                else:
                    src, dst = r.randint(0, n - 1), r.choice(influencers)
            else:
                src, dst = r.randint(0, n - 1), r.randint(0, n - 1)
            key = (min(src, dst), max(src, dst))
            if src == dst or key in seen:
                continue
            seen.add(key)
            count += 1
            self._add(f"person_{src}", f"person_{dst}", "KNOWS", creationDate=self._date(start))


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def generate(scale: str, root: Path, seed: int = 42) -> Dataset:
    """Generate the SNB dataset for `scale` into `root` (CSVs plus schema.json)."""
    persons, knows = SCALES[scale]
    gen = _Gen(persons, knows, seed)
    gen.run()
    root.mkdir(parents=True, exist_ok=True)

    node_tables = []
    for label, columns in _NODE_SCHEMAS.items():
        table = NodeTable(label, columns, f"{label}.csv")
        header = [c.name for c in columns]
        _write_csv(root / table.file, header, [[row[c.name] for c in columns] for row in gen.nodes[label]])
        node_tables.append(table)

    grouped: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    for edge in gen.edges:
        grouped[(edge[2], _label_of(edge[0]), _label_of(edge[1]))].append(edge)
    edge_tables = []
    for (etype, src, dst), edges in sorted(grouped.items()):
        columns = _EDGE_PROPS.get(etype, ())
        table = EdgeTable(etype, src, dst, columns, f"{etype}__{src}__{dst}.csv")
        _write_csv(
            root / table.file,
            ["src", "dst", *[c.name for c in columns]],
            [[s, d, *[props[c.name] for c in columns]] for s, d, _, props in edges],
        )
        edge_tables.append(table)

    ds = Dataset("snb", scale, root, tuple(node_tables), tuple(edge_tables))
    ds.save()
    return ds


Adjacency = dict[str, dict[str, list[tuple[str, dict[str, Any]]]]]


@dataclass
class SnbGraph:
    """In-memory view: nodes by label and id, edges by type in both directions."""

    nodes: dict[str, dict[str, dict[str, Any]]]
    out: Adjacency
    incoming: Adjacency

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "SnbGraph":
        nodes: dict[str, dict[str, dict[str, Any]]] = {}
        for nt in ds.nodes:
            nodes[nt.label] = {row["id"]: row for row in ds.rows(nt)}
        out: Adjacency = defaultdict(lambda: defaultdict(list))
        inc: Adjacency = defaultdict(lambda: defaultdict(list))
        for et in ds.edges:
            for row in ds.rows(et):
                props = {k: v for k, v in row.items() if k not in ("src", "dst")}
                out[et.type][row["src"]].append((row["dst"], props))
                inc[et.type][row["dst"]].append((row["src"], props))
        return cls(nodes, {k: dict(v) for k, v in out.items()}, {k: dict(v) for k, v in inc.items()})
