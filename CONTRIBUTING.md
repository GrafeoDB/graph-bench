# Contributing

graph-bench compares embedded graph databases under the rules in [FAIRNESS.md](FAIRNESS.md). Read those first:
every change to an adapter or a benchmark has to keep them.

## Setup

```bash
git clone https://github.com/GrafeoDB/graph-bench
cd graph-bench
uv sync --extra embedded --extra bench --extra dev
```

## Tests and lint

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

`pytest` runs `tests/` with coverage and fails below 88%. The adapter tests (`tests/test_db_*.py`) generate the
push selection's data into a temporary directory once per session and run every push benchmark against the
database; they skip when its package is not installed. CI (`.github/workflows/ci.yml`) runs the same three
commands on Python 3.12 and 3.13.

To run the benchmark by hand, see the [README](README.md#running-it). To try a local Grafeo build, install it
with `uv run --no-sync graph-bench setup --grafeo path/to/grafeo` and keep using `uv run --no-sync`, so uv does
not put the PyPI release back.

## Layout

```text
graph_bench/
  cli.py          the graph-bench command
  databases/      one adapter per database, the Database base class, the name registry
  datasets/       dataset generators, the CSV schema, the cache
  suites/         the benchmarks (queries per dialect, algorithms), the selections, the reference answers
  answers.py      canonicalization and the answer check
  runner/         the supervisor and the per-database worker process, timing
  report/         ranking, the README matrix, the summary block for grafeo's README
benches/          the pytest-codspeed entry point (graph-bench codspeed)
tests/            the test suite
docs/             known disagreements, workload profiles, design notes
legacy/           the pre-0.3.0 benchmark, not installed or tested
```

## Adding a database

### 1. Write the adapter

Create `graph_bench/databases/<name>.py` with a subclass of `Database` (`graph_bench/databases/base.py`):

```python
from collections.abc import Mapping
from typing import Any, ClassVar

from graph_bench.databases.base import Database, DatabaseInfo, NotSupported
from graph_bench.datasets.schema import Dataset
from graph_bench.types import Model


class MyDatabase(Database):
    name: ClassVar[str] = "mydb"
    display_name: ClassVar[str] = "MyDB"
    models: ClassVar[frozenset[Model]] = frozenset({"property_graph"})
    dialect: ClassVar[str] = "cypher"  # or a key of its own for per-database query texts

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]:
        """Environment the worker sets before the driver is imported, e.g. a thread count variable."""
        return {}

    def info(self) -> DatabaseInfo:
        """Version, process kind, storage and how the thread setting was applied (`threads_note`)."""
        ...

    def load(self, dataset: Dataset) -> None:
        """Create the instance (inside the load clock) and load the dataset's CSV files."""
        ...

    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """Run one query and return its rows as tuples of scalars."""
        ...

    def counts(self) -> tuple[int, int]:
        """The number of nodes and edges loaded, counted through the database's own query interface; untimed."""
        ...

    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]:
        """Native "bfs", "wcc" or "pr": vertex id -> value. Leave it out (NotSupported) without a native one."""
        raise NotSupported(name)
```

- `__init__` only stores `workdir` and `threads`; create the database instance in `load()`, so its creation is
  part of the load time. Each load gets a fresh instance.
- `threads` is 1 (single-threaded, every current selection) or `None` (all cores). Apply it through the
  database's documented setting, or `process_env` when it is an environment variable, and say how in
  `threads_note`. A database whose thread count can't be set returns `threads_configurable=False`.
- Load with the database's fastest documented bulk path, CSV parsing included.
- `query()` returns the rows as Python tuples of scalars. Materializing them is part of the timed region;
  canonicalizing them is not.
- Queries use the shared openCypher text (`dialect = "cypher"`). Where the database needs another text, give it
  its own `dialect` and add that key to the query's `text` mapping in `graph_bench/suites/snb.py`, with the
  same meaning and the same answer.
- `counts()` checks the load: the worker calls it right after each timed load, outside the clock, and a
  mismatch with the dataset's node and edge counts makes the load ≠. Count with a documented query (e.g. Cypher
  `MATCH (n) RETURN count(n)` and `MATCH ()-[r]->() RETURN count(r)`). Without it (the default raises
  `NotSupported`) the load stays unchecked.
- Raise `NotSupported` for a capability the database lacks; the case is reported as ⊘.
- Add the package as an optional extra in `pyproject.toml` (and to `embedded` when it installs from PyPI on
  Linux, macOS and Windows), then run `uv lock`.

### 2. Register it

In `graph_bench/databases/__init__.py`, add the name to `KNOWN` and an entry to each map:

- `MODELS`: the data models it supports (`property_graph`, `sql_pgq`, `rdf`);
- `DISPLAY`: the name shown in the matrix;
- `REGISTRY`: `"module:Class"`, imported only when the database runs, so `graph-bench list` works without the
  driver installed.

A database outside this repository can be tried without registering it:
`GRAPH_BENCH_EXTRA_DATABASES="mydb=my_module:MyDatabase"`.

### 3. Pass the push agreement test

Add `tests/test_db_<name>.py` following `tests/test_db_ladybug.py`: `pytest.importorskip` the driver, run
`run_bench("push", ["<name>"], ...)` on the `push_data_dir` fixture, and require every entry to be `ok` with an
`agree` answer (for a load, its node and edge counts), except the worker's model gate (`not_supported` with
exactly "model <model> not supported") and the ids recorded for the database in `docs/known-disagreements.md`. Add a test of the thread settings (`process_env`, `info().threads_note`).

### 4. Record disagreements

When the database gives a wrong answer, fails, or runs far too long, do not change the query or the adapter to
get around it. Add a section to [docs/known-disagreements.md](docs/known-disagreements.md): the case, the data,
the binding, the database's answer, the reference answer and the cause (engine or adapter). An `- id: <db>/<benchmark>`
line in the section allowlists the case in the agreement test; cases that agree but are slow carry no such line.

## Adapter rules

From [FAIRNESS.md](FAIRNESS.md#adapter-rules), in short:

- documented paths only, called the way the vendor recommends;
- documentation-recommended settings only when they keep the comparison like for like;
- nothing that gives one database an advantage: no precomputed or cached answers, no benchmark-specific code;
- native algorithms only, never a NetworkX fallback;
- a database's bugs are recorded, not worked around.

## Style

- Python 3.12 or later, `ruff` with a line length of 120 (`ruff check .` and `ruff format .`).
- No em-dashes or en-dashes in code, comments or docs: use commas or colons.
- `uv` in docs and commands, never `pip`.

## Pull requests

1. Keep a pull request to one adapter or one change of shared code.
2. Run the tests and lint above.
3. For an adapter change, say what changed and link the database documentation it follows.
