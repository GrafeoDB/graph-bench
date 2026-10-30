# Embedded Comparison Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace graph-bench's single-process benchmark with a per-process, answer-checked comparison harness that renders a capability matrix. It runs Grafeo and LadybugDB on the matrix categories Load, Complex queries, Hops and Algorithms, and exposes the interface agreed with grafeo-31: `setup`, `data`, `list`, `bench`, `codspeed`, `report`.

**Architecture:**
- A catalog of benchmarks, each with a category and a data model, expands a selection (`push`/`full`) into cases.
- A supervisor runs one single-threaded worker process per database, one database after another. Each worker:
  - loads the generated CSV datasets once;
  - runs each case once untimed as a preflight, which checks the answer against a precomputed Python/NetworkX reference and enforces the time budget;
  - times only the cases that pass;
  - streams JSON lines back.
- The supervisor resumes after crashes and timeouts, applies the 10x-top-3 rule over all results, and writes `results.json`.
- `report` turns that file into the README matrix (overview + per-category detail) and the grafeo summary block.
- `codspeed` runs the same cases through pytest-codspeed, one pytest process per database. Each marker maps to a pytest outcome.

**Tech Stack:** Python 3.12+, typer, networkx, numpy/scipy, pytest, pytest-codspeed, grafeo (>=0.6), real_ladybug (>=0.15), GitHub Actions on `codspeed-macro`.

**Spec:** `docs/superpowers/specs/2026-10-30-embedded-comparison-design.md`

## Global Constraints

- **Repo rules:**
  - No em-dashes or en-dashes anywhere (code, docs, output); use commas or colons.
  - Use `uv`, never `pip`, in docs and commands.
  - Do not run `git commit`, `git push` or `git tag`. The user commits; each task ends with a checkpoint where tests pass and the tree is left for review.
- **Python:** `requires-python = ">=3.12"`, ruff line length 120, ruff rules `E,F,W,I,UP,B,SIM,RUF`.
- **Legacy code:** it lives in `legacy/` after Task 1 and is never imported by new code. New code may copy from it.
- **Answer check:** no NetworkX fallback inside a measured case. A database without an implementation raises `NotSupported` and the entry is `not_supported`.
- **Threads:** single-threaded only, for every database, until grafeo 0.6.4. Keep the `threads` field and the `-allcores` id suffix in the code (`threads: int | None`) but no selection uses `None` yet.
  - Grafeo: worker env `RAYON_NUM_THREADS=1`, threads_note `"single-threaded executor"`.
  - Ladybug: `Database(max_num_threads=1)` and `Connection(num_threads=1)`, threads_note `"max_num_threads=1"`.
- **Storage:** in-memory where the database supports it (user decision, 2026-10-30); a disk-only database uses a fresh directory under `--workdir` (default the system temp dir). `DatabaseInfo.storage` records `"in-memory"` or `"on-disk"`, and the README footer lists it per database. Queries run after load, warmed up.
- **Categories:**
  - `Category = Literal["load", "batch_writes", "scans", "hops", "pattern_matching", "complex_queries", "algorithms", "vector", "hybrid", "rdf", "sql_pgq", "crash_restore", "deriva", "project_suites"]`, in this matrix order.
  - Display names: Load, Batch writes, Scans, Hops, Pattern matching, Complex queries, Algorithms, Vector search, Hybrid, RDF/SPARQL, SQL/PGQ, Crash and restore, Deriva workload, Project suites.
  - `Model = Literal["property_graph", "sql_pgq", "rdf"]`.
- **Benchmark ids:**
  - queries: `benches/test_<category>.py::test_<benchmark_id>[<db>-<scale>]`, with `-allcores` appended inside the brackets only for `threads=None`;
  - loads: `benches/test_load.py::test_load_<dataset>[<db>-<scale>]`.
- **Markers:**

  | Marker | Meaning | status / answer | pytest outcome, reason prefix |
  |---|---|---|---|
  | ⊘ | not available | `not_supported` | skipped, `not available: ` |
  | ✗ | failing | `failed` | failed, `error: ` |
  | ≠ | wrong answer | `ok` / `disagree` | failed, `wrong answer: ` |
  | ⏱ | too slow | `timeout` | failed, `too slow: ` |

- **Budget rule:** the budget for a case is `10 * mean(top3 median_ns)` over databases with `ok`/`agree` (the mean over those available if fewer than 3). The hard cap is `--case-timeout` (default 600 s) when no earlier time exists. It is applied as a preflight cutoff during the run and again over the final results.
- **results.json:** `schema_version = 1`, field names exactly as in the spec's "results.json" section.
- **Exit codes:** `bench`/`codspeed` exit 0 whenever results.json was written, whatever the timings or answers. Non-zero only for harness errors.

## Review Focus

1. **A database that crashes or hangs inside one native call:** the user expects that case marked ✗/⏱ and every later case still run. (Task 6: `test_supervisor_resumes_after_crash`, `test_supervisor_marks_hang_as_timeout`.)
2. **Run order deciding a marker:** a slow database run first (no budget yet) must still end up ⏱ when it is beyond 10x of the final top 3. (Task 6: `test_budget_rule_is_order_independent`.)
3. **Numeric type drift:**
   - `INT64` vs Python `int` vs `float` `2.0`;
   - `inf`/`None` for unreachable BFS vertices;
   - collected lists in different orders.
   The user expects these to count as the same answer. (Task 3: `test_canonical_numbers_and_lists`, `test_bfs_unreachable_forms_agree`.)
4. **Ties under `ORDER BY ... LIMIT`:** different but valid rows must not be reported as wrong answers, so every limited query orders fully. (Task 4: `test_limited_queries_have_total_order`.)
5. **CodSpeed's `CODSPEED_PROFILE_FOLDER`:** it must reach every pytest subprocess unchanged, or CodSpeed silently loses the results. A wrong or too-slow result must never be measured. (Task 11: `test_codspeed_passes_profile_folder`, `test_codspeed_marker_outcomes`.)

Also covered, lower risk: stale dataset cache after a generator change (Task 2: `test_key_changes_with_generator_source`).

---

## File Structure

```
legacy/                                  # Task 1: everything of the old benchmark, unchanged
graph_bench/
  __init__.py                            # __version__
  types.py                               # Model, Category, Status, AnswerVerdict, Marker; Timing
  datasets/
    schema.py                            # Column, NodeTable, EdgeTable, Dataset (CSV + schema.json)
    snb.py                               # SNB generator (ported from legacy) -> Dataset; SnbGraph in-memory view
    graphalytics.py                      # symmetric KNOWS graph -> Dataset
    cache.py                             # data_dir(), dataset_key(), prepare()
  suites/
    catalog.py                           # Query, Algorithm, Selection, Case, SELECTIONS, cases()
    snb.py                               # SNB query texts, bindings(), reference()
    graphalytics.py                      # algorithm defs, bindings(), reference()
  answers.py                             # canonical(), digest(), verify()
  databases/
    __init__.py                          # REGISTRY, get_database()
    base.py                              # Database ABC, DatabaseInfo, NotSupported
    grafeo.py
    ladybug.py
  runner/
    timing.py                            # measure()
    worker.py                            # python -m graph_bench.runner.worker
    supervisor.py                        # run_bench(), merge into Results
  results.py                             # Entry, DatabaseInfo serialization, Results
  report/
    ranking.py
    summary.py
    readme.py
  grafeo_setup.py                        # install_grafeo()
  cli.py                                 # typer app
benches/
  conftest.py
  _cases.py                              # make_test(): preflight, marker -> pytest outcome, benchmark()
  test_load.py, test_complex_queries.py, test_hops.py, test_algorithms.py
tests/
  fakes.py                               # FakeDatabase for runner tests
  test_*.py
.github/workflows/ci.yml                 # lint + unit tests (new)
.github/workflows/bench.yml              # CodSpeed + README bot PR (new)
```

---

### Task 1: Move the legacy benchmark aside and lay down the new package

**Files:**
- Move (git mv) into `legacy/`:
  - `graph_bench/`, `tests/`, `scripts/`, `results/`
  - `RESULTS_EMBEDDED.md`, `RESULTS_SERVER.md`, `DOCKER.md`, `docker-compose.yml`, `falkordblite_bench.db`
  - `.github/workflows/ci.yml` → `legacy/workflows/ci.yml`
  - `.github/workflows/grafeo-latest.yml` → `legacy/workflows/grafeo-latest.yml`
- Create: `legacy/README.md`
- Create: `graph_bench/__init__.py`, `graph_bench/types.py`, `graph_bench/cli.py`, `tests/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `graph_bench.__version__: str`
  - `graph_bench.types`:
    - `Model` and `Category` exactly as in Global Constraints, plus `CATEGORY_ORDER: tuple[Category, ...]` and `CATEGORY_NAMES: dict[Category, str]`
    - `Status = Literal["ok", "failed", "timeout", "not_supported"]`
    - `AnswerVerdict = Literal["agree", "disagree", "unchecked"]`
    - `Marker = Literal["", "⊘", "✗", "≠", "⏱"]` and `marker(status: Status, answer: AnswerVerdict) -> Marker`, per the Global Constraints marker table
    - `@dataclass(frozen=True) Timing(median_ns: float, p99_ns: float, min_ns: int, iterations: int)`
  - `graph_bench.cli.app: typer.Typer` and `main() -> None`. Later tasks add commands to `app`.

- [ ] **Step 1: Move the legacy tree.** Use `git mv` for each path above. Write `legacy/README.md` (a few lines):
  - this is the pre-0.3.0 benchmark, kept for reference until the new comparison replaces it;
  - it is not installed or tested;
  - run it with `cd legacy && uv run --no-sync python -m graph_bench.cli.main ...`.
- [ ] **Step 2: Rewrite `pyproject.toml`.**
  - Keep the name, version, authors, license and urls.
  - Core dependencies: `networkx>=3.0`, `numpy>=1.25`, `scipy>=1.11`, `psutil>=5.9`, `typer>=0.15.0`.
  - Extras:
    - `grafeo = ["grafeo>=0.6.0,<0.7"]`
    - `ladybug = ["real_ladybug>=0.15"]`
    - `embedded = ["graph-bench[grafeo,ladybug]"]`
    - `bench = ["pytest>=9.0.2", "pytest-codspeed>=4.0"]`
    - `dev = ["pytest>=9.0.2", "pytest-cov>=7.0.0", "ruff>=0.15.0", "prek>=0.3.2", "ty>=0.0.15"]`
  - Script: `graph-bench = "graph_bench.cli:main"`.
  - Ruff: `extend-exclude = ["legacy"]`.
  - Pytest: `testpaths = ["tests"]`. Markers: `grafeo`, `ladybug`, `slow`. Keep `--cov-fail-under=88` with coverage `omit` adding `graph_bench/databases/grafeo.py` and `graph_bench/databases/ladybug.py`, since they are covered only by the optional integration tests.
- [ ] **Step 3: Write the failing test** `tests/test_cli.py::test_help_lists_commands`: `CliRunner().invoke(app, ["--help"])` exits 0.
- [ ] **Step 4: Run it** with `uv run --no-sync pytest tests/test_cli.py -v --no-cov`. Expected: FAIL (no module `graph_bench.cli`).
- [ ] **Step 5: Implement** `graph_bench/__init__.py`, `types.py` and `cli.py`, with `app = typer.Typer(no_args_is_help=True)` and a `version` command.
- [ ] **Step 6: Sync and verify.**
  - `uv sync --extra embedded --extra bench --extra dev`
  - `uv run --no-sync pytest tests -v --no-cov`
  - `uv run --no-sync ruff check graph_bench tests`

  Expected: PASS, ruff clean. `git status` shows the renames as `R`.

---

### Task 2: Datasets, schema and cache

**Files:**
- Create: `graph_bench/datasets/{__init__,schema,snb,graphalytics,cache}.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces, in `schema.py`:
  - `Column(name: str, type: Literal["INT64", "DOUBLE", "STRING"])`
  - `NodeTable(label: str, columns: tuple[Column, ...], file: str)`. The first column is `id`, the primary key.
  - `EdgeTable(type: str, src: str, dst: str, columns: tuple[Column, ...], file: str)`. CSV header: `src,dst,<columns>`.
  - `Dataset(name: str, scale: str, root: Path, nodes: tuple[NodeTable, ...], edges: tuple[EdgeTable, ...])` with:
    - `save() -> None`, which writes `schema.json`;
    - `@classmethod load(root: Path) -> Dataset`;
    - `rows(table: NodeTable | EdgeTable) -> Iterator[dict[str, Any]]`, values typed per column.
  - CSVs are UTF-8, comma-separated, with a header and no quoting of numbers.
- Produces, in `snb.py`:
  - `SCALES = {"small": (1_000, 18_000), "medium": (10_000, 180_000)}`: (persons, knows edges), from legacy `sf01`/`sf1`.
  - `generate(scale: str, root: Path, seed: int = 42) -> Dataset`
  - `SnbGraph.from_dataset(ds: Dataset) -> SnbGraph`: an in-memory view with `nodes: dict[str, dict[str, dict]]` (label → id → props) and `out: dict[str, dict[str, list[tuple[str, dict]]]]` (edge type → src id → [(dst id, props)]), plus `incoming` with the same shape.
- Produces, in `graphalytics.py`: `generate(snb: Dataset, root: Path) -> Dataset`. Node table `V(id INT64)`, edge table `E(V->V)`, with each KNOWS pair in both directions, deduplicated, vertex ids the integer suffix of `person_<n>`.
- Produces, in `cache.py`:
  - `data_dir() -> Path`: `GRAPH_BENCH_DATA_DIR` or `~/.cache/graph-bench`.
  - `dataset_key(selection: str) -> str`: sha256 over this package's `datasets/*.py` and `suites/*.py` sources, the selection's scales, bindings and seed. Its first 16 hex chars are the key.
  - `prepare(selection: str) -> dict[tuple[str, str], Dataset]`: keyed by (dataset name, scale), stored under `data_dir()/<key>/<name>-<scale>/`. Reuses an existing complete directory, marked by a `.complete` file.

- [ ] **Step 1: Port the generator.** Copy the legacy `datasets/ldbc_snb.py` generation logic into `snb.py`, with these changes:
  - Use a local `random.Random(seed)` instead of the global `random`.
  - Store dates as INT64 epoch milliseconds.
  - Keep only these edge types in LDBC direction: `KNOWS, LIVES_IN, HAS_INTEREST, IS_PART_OF, STUDY_AT, WORK_AT, IS_LOCATED_IN, HAS_CREATOR, HAS_TAG, CONTAINER_OF, REPLY_OF, HAS_MODERATOR, HAS_MEMBER`. Drop reverse helper edges such as `CREATED`.
  - Derive each edge's (src label, dst label) from the id prefix.
- [ ] **Step 2: Write the failing tests** in `tests/test_datasets.py`:
  - `test_generate_is_deterministic`: two `generate("small", tmp)` calls give byte-identical CSVs.
  - `test_schema_roundtrip`: `Dataset.load(ds.root) == ds`, and `rows()` yields `int` for INT64 columns.
  - `test_edge_tables_split_by_endpoint_labels`: `REPLY_OF` appears as two `EdgeTable`s, `(Comment, Post)` and `(Comment, Comment)`.
  - `test_graphalytics_is_symmetric`: for every (a, b) in E, (b, a) is in E; no self-loops; no duplicates.
  - `test_key_changes_with_generator_source`: monkeypatch the source reader to return modified text, and the key changes.
  - `test_prepare_reuses_complete_dir`: a second `prepare` doesn't call `generate`.
- [ ] **Step 3: Run** `uv run --no-sync pytest tests/test_datasets.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 4: Implement** `schema.py`, `snb.py`, `graphalytics.py` and `cache.py` to the interfaces above.
- [ ] **Step 5: Run** the same command. Expected: PASS.

---

### Task 3: Answer canonicalization and verification

**Files:**
- Create: `graph_bench/answers.py`
- Test: `tests/test_answers.py`

**Interfaces:**
- Produces:
  - `canonical(rows: Iterable[Sequence[Any]], *, ordered: bool) -> list[list[Any]]`:
    - integral floats with abs < 2**53 become `int`;
    - `bool` stays `bool`;
    - list/tuple values are sorted by `repr` of their canonical elements;
    - rows are sorted by `repr` when `ordered=False`.
  - `canonical_map(values: Mapping[Any, Any], *, drop_unreachable: bool) -> dict[int, Any]`: keys become `int`. With `drop_unreachable`, values that are `None`, `inf` or `>= 2**62` are dropped.
  - `digest(answer: Any) -> str`: sha256 of `json.dumps(answer, sort_keys=True)` with floats rounded to 6 significant digits, first 16 hex chars.
  - `verify(answer: Any, reference: Any, *, rel_eps: float) -> str | None`: `None` when equal, with floats compared by `abs(a - b) <= rel_eps * abs(b)` (b is the reference; when b == 0, compare `abs(a) <= rel_eps`). Otherwise a short description of the first difference, e.g. `"binding 3, row 0, col 2: 17 != 18"`.
  - Constants: `SNB_EPS = 1e-9`, `LDBC_EPS = 1e-4`.

- [ ] **Step 1: Write the failing tests:**
  - `test_canonical_numbers_and_lists`: `canonical([(2.0, [3, 1])], ordered=True) == [[2, [1, 3]]]`.
  - `test_canonical_unordered_sorts_rows`.
  - `test_bfs_unreachable_forms_agree`: `canonical_map({1: 0, 2: None}, drop_unreachable=True) == canonical_map({1: 0, 2: float("inf")}, drop_unreachable=True) == canonical_map({1: 0, 2: 9223372036854775807}, drop_unreachable=True) == {1: 0}`.
  - `test_verify_epsilon`:
    - `verify({1: 0.15001}, {1: 0.15}, rel_eps=1e-4) is None`;
    - `verify({1: 0.151}, {1: 0.15}, rel_eps=1e-4)` contains `"1"`.
  - `test_verify_reports_first_difference`: a nested list difference names the binding and row.
  - `test_digest_stable_across_float_noise`: `digest([0.1 + 0.2]) == digest([0.3])`.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_answers.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** `answers.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.

---

### Task 4: Catalog, SNB and Graphalytics suites, selections, `list` and `data` commands

**Files:**
- Create: `graph_bench/suites/{__init__,catalog,snb,graphalytics}.py`
- Modify: `graph_bench/cli.py` (commands `list`, `data prepare`, `data key`)
- Test: `tests/test_catalog.py`, `tests/test_references.py`

**Interfaces:**
- Consumes: `Dataset`, `SnbGraph`, `prepare`, `dataset_key` (Task 2); `canonical`, `canonical_map` (Task 3).
- Produces, in `catalog.py`:
  - `Query(id: str, title: str, category: Category, model: Model, dataset: str, text: Mapping[str, str], ordered: bool, rel_eps: float)`. `text` maps dialect to query text; `"cypher"` is required. `dataset` is `"snb"`. `title` is the matrix row label, e.g. `"IS1 person profile"`.
  - `Algorithm(id: str, title: str, category: Category, model: Model, dataset: str, name: Literal["bfs", "wcc", "pr"], rel_eps: float)`. `dataset` is `"graphalytics"` and `category` is `"algorithms"`.
  - `Selection(name, categories: frozenset[Category], scales: tuple[str, ...], threads: tuple[int | None, ...], warmup: int, iterations: int, load_iterations: int, bindings: int)`.
  - `SELECTIONS`:
    - `push`: `{"load", "complex_queries", "hops", "algorithms"}`, `("small",)`, `(1,)`, warmup=1, iterations=5, load_iterations=1, bindings=10.
    - `full`: all categories, `("small", "medium")`, `(1,)`, warmup=2, iterations=10, load_iterations=3, bindings=20.
    - `warmup` counts the preflight as its first warm-up run.
  - `Case(item: Query | Algorithm, db: str, scale: str, threads: int | None)` with:
    - `bench_id` property, per Global Constraints;
    - `test_name` property: `test_<item.id>`.
  - `cases(selection: str, dbs: Sequence[str]) -> list[Case]`: every item in the selection's categories for every db, sorted by `bench_id`. Items whose model the db doesn't support are still listed: they become ⊘ entries without starting the db, so the matrix shows them.
  - `load_ids(selection: str, dbs: Sequence[str]) -> list[str]`: the `test_load_<dataset>` ids.
  - `ITEMS: tuple[Query | Algorithm, ...]`.
- Produces, in `snb.py`:
  - `QUERIES: tuple[Query, ...]` for `snb_is1` … `snb_is7`, `snb_ic2`, `snb_ic6` (category `complex_queries`) and `snb_ic13` (category `hops`), all model `property_graph`.
  - `bindings(query_id: str, graph: SnbGraph, n: int) -> list[dict[str, Any]]`
  - `reference(query_id: str, graph: SnbGraph, binding: dict) -> list[list[Any]]`, already canonical.
- Produces, in `graphalytics.py`:
  - `ALGORITHMS`: ids `graphalytics_bfs`, `graphalytics_wcc`, `graphalytics_pr`.
  - `bindings(algo_id, ds) -> list[dict]`:
    - BFS: `[{"source": <vertex with max degree, lowest id on ties>}]`;
    - others: `[{}]`.
  - `reference(algo_id, ds, binding) -> dict[int, Any]`:
    - BFS: depth, unreachable dropped;
    - WCC: vertex → min vertex id of its component;
    - PR: `networkx.pagerank(G, alpha=0.85, tol=1e-10, max_iter=1000)`.
  - WCC answers from databases are normalized the same way (vertex → min member) before `verify`.
- Produces, in the cache: `prepare()` also writes `references/<item_id>-<scale>.json = {"bindings": [...], "answers": [...]}` for every item in the selection.
- Produces, CLI:
  - `graph-bench list --selection S [--db a,b]` prints the sorted `bench_id`s plus load ids, one per line, nothing else.
  - `graph-bench data prepare --selection S`
  - `graph-bench data key --selection S` prints only the key.

**SNB query semantics:**
- Follow the LDBC SNB Interactive v1 spec, simplified to the generated schema.
- Every `LIMIT` query orders fully, with the id as the last key.
- Returned values are scalars; collected lists are allowed but compared as multisets.
- Write each query in openCypher in `text["cypher"]`. Add a `text["ladybug"]` override where Kùzu syntax differs, e.g. `(m:Post:Comment)` for "Post or Comment", or `* SHORTEST` for IC13.

| id | parameters | returns | order / limit |
|---|---|---|---|
| snb_is1 | personId | firstName, lastName, birthday, locationIP, browserUsed, cityId, gender, creationDate | single row |
| snb_is2 | personId | messageId, content-or-imageFile, creationDate, originalPostId, originalPosterId, originalPosterFirstName, originalPosterLastName | creationDate desc, messageId desc, limit 10 |
| snb_is3 | personId | friendId, firstName, lastName, friendshipCreationDate | friendshipCreationDate desc, friendId asc |
| snb_is4 | messageId | creationDate, content-or-imageFile | single row |
| snb_is5 | messageId | creatorId, firstName, lastName | single row |
| snb_is6 | messageId | forumId, forumTitle, moderatorId, moderatorFirstName, moderatorLastName | single row |
| snb_is7 | messageId (a Post) | commentId, content, creationDate, replyAuthorId, firstName, lastName, replyAuthorKnowsOriginalAuthor | creationDate desc, replyAuthorId asc |
| snb_ic2 | personId, maxDate | friendId, firstName, lastName, messageId, content-or-imageFile, creationDate | creationDate desc, messageId asc, limit 20 |
| snb_ic6 | personId, tagName | otherTagName, postCount | postCount desc, otherTagName asc, limit 10 |
| snb_ic13 | person1Id, person2Id | shortestPathLength over KNOWS (undirected), -1 if none | single row |

- [ ] **Step 1: Write the failing tests** in `tests/test_catalog.py`:
  - `test_push_selection_cases`: `cases("push", ["grafeo", "ladybug"])` has 2 × 13 cases, all `threads == 1` and `scale == "small"`.
  - `test_full_selection_single_threaded`: every case in `cases("full", ...)` has `threads == 1`.
  - `test_bench_id_format`:
    - `Case(snb_is1, "ladybug", "small", 1).bench_id == "benches/test_complex_queries.py::test_snb_is1[ladybug-small]"`;
    - IC13's id starts with `benches/test_hops.py::`;
    - the `threads=None` variant ends with `[ladybug-small-allcores]`.
  - `test_list_output_is_stable`: the `CliRunner` output of `list --selection push --db grafeo,ladybug` equals the sorted lines and is identical across two invocations.
  - `test_limited_queries_have_total_order`: every query whose cypher text contains `LIMIT` also contains `ORDER BY`, and its last order key ends in `id` (regex `(\w+\.)?\w*[iI]d\s+(ASC|DESC)?\s*LIMIT`).
  - `test_every_query_has_cypher_text`.
- [ ] **Step 2: Write the failing tests** in `tests/test_references.py`, using the `small` dataset (module-scoped fixture, `GRAPH_BENCH_DATA_DIR=tmp_path`):
  - `test_reference_is1_matches_props`: the row equals the person's generated props.
  - `test_reference_ic13_self_and_unreachable`: binding (a, a) → `[[0]]`; for a person with no KNOWS edges → `[[-1]]`.
  - `test_bfs_reference_depths`: on a hand-built 4-vertex graph.
  - `test_wcc_reference_min_member`: on a hand-built graph with two components.
  - `test_pr_reference_sums_to_one`.
  - `test_prepare_writes_references`: `references/snb_is1-small.json` has `bindings` of length 10.
- [ ] **Step 3: Run** `uv run --no-sync pytest tests/test_catalog.py tests/test_references.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 4: Implement** the catalog, suites, reference functions, the `prepare()` extension and the CLI commands. Pick bindings deterministically, so `n` evenly spaced elements of the id-sorted candidate list (e.g. persons with at least one friend for IS3).
- [ ] **Step 5: Run** the same command. Expected: PASS.

---

### Task 5: Database contract, fake database, timing and worker

**Files:**
- Create: `graph_bench/databases/{__init__,base}.py`, `graph_bench/runner/{__init__,timing,worker}.py`, `graph_bench/results.py`
- Create: `tests/fakes.py`
- Test: `tests/test_worker.py`, `tests/test_timing.py`

**Interfaces:**
- Consumes: `Dataset` (Task 2), `Case`, `cases`, `SELECTIONS` (Task 4), `canonical`, `digest` (Task 3).
- Produces, in `base.py`:

```python
class NotSupported(Exception): ...

@dataclass(frozen=True)
class DatabaseInfo:
    name: str
    version: str
    process: Literal["in-process", "local process"]
    threads_configurable: bool
    threads_note: str
    storage: Literal["in-memory", "on-disk"]
    java: str | None = None

class Database(ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]              # "Grafeo", "LadybugDB"
    models: ClassVar[frozenset[Model]]
    dialect: ClassVar[str] = "cypher"
    process: ClassVar[Literal["in-process", "local process"]] = "in-process"

    @classmethod
    def process_env(cls, threads: int | None) -> dict[str, str]: return {}
    def __init__(self, workdir: Path, threads: int | None) -> None: ...
    @abstractmethod
    def info(self) -> DatabaseInfo: ...
    @abstractmethod
    def load(self, dataset: Dataset) -> None: ...
    @abstractmethod
    def query(self, text: str, params: Mapping[str, Any]) -> list[tuple[Any, ...]]: ...
    def algorithm(self, name: str, params: Mapping[str, Any]) -> dict[Any, Any]: raise NotSupported(name)
    def close(self) -> None: ...
```

- Produces, in `databases/__init__.py`:
  - `REGISTRY: dict[str, str] = {"grafeo": "graph_bench.databases.grafeo:GrafeoDatabase", "ladybug": "graph_bench.databases.ladybug:LadybugDatabase"}`, extended by the env var `GRAPH_BENCH_EXTRA_DATABASES="name=module:Class,..."`;
  - `get_database(name: str) -> type[Database]`, importing lazily;
  - `model_support(name) -> frozenset[Model]` and `display_name(name) -> str`, which read a static `MODELS`/`DISPLAY` map without importing the driver, so `list` works without the packages installed.
- Produces, in `timing.py`: `measure(fn: Callable[[], Any], *, warmup: int, iterations: int) -> tuple[Timing, Any]`. It returns the last call's result, disables gc during each timed call and collects between calls.
- Produces, in `results.py`:
  - `Entry` dataclass with every spec field (`id`, `category`, `model`, `benchmark`, `database`, `scale`, `threads`, `status`, `error`, `timing`, `preflight_ns`, `load_ns`, `answer`, `reference`, `answer_digest`, `detail`). `id` is `Case.bench_id`.
  - `Results` dataclass with the top-level fields;
  - `Results.write(path)`, `Results.read(path)`, `SCHEMA_VERSION = 1`.
- Produces, in `answers.py` (extends Task 3): `check_case(case: Case, answers: list[Any], reference_file: Path) -> tuple[AnswerVerdict, str | None]`. It applies the WCC min-member normalization and `verify` with the item's `rel_eps`.
- Produces, in `worker.py`, the CLI `python -m graph_bench.runner.worker --db NAME --selection S --threads (1|all) --workdir DIR --out FILE.jsonl [--skip ID ...] [--budget ID=SECONDS ...] [--case-timeout SECONDS]`:
  - **First line:** `{"kind": "database", **asdict(info)}`.
  - **Model not supported:** for a case whose model the db doesn't support, emit a `not_supported` entry with no load and no start line.
  - **Per dataset needed:**
    - load `load_iterations` times, each into a fresh database instance (closing the previous one; a fresh subdirectory for on-disk databases);
    - emit `{"kind": "entry", "id": <load id>, "status": "ok", "load_ns": <median>}`;
    - keep the last load.
  - **Per case**, emit `{"kind": "start", "id": ...}` first. This is how the supervisor knows which case hung. Then, where the run function runs the query or algorithm for every binding of the reference file:
    1. **Preflight:** run once, untimed but wall-clocked.
       - `NotSupported` → `not_supported`.
       - Any other exception → `failed` with the message.
       - Preflight time > the case's budget (from `--budget`, else `--case-timeout`) → `timeout`, with error `"too slow: <t> s > budget <b> s"`.
       - Answer check `disagree` → `status ok`, `answer disagree`, `detail` set, timing null.
       - In all four cases, skip timing.
    2. **Timing:** otherwise, `measure(run, warmup=selection.warmup - 1, iterations=selection.iterations)`.
    3. **Entry:** emit `{"kind": "entry", "id", "status", "error", "timing", "answer", "answer_digest", "detail"}` and write the canonical answers to `<out stem>/answers/<sanitized id>.json`.
  - **Each line** is flushed immediately.
  - The budget check is post hoc on the preflight. A preflight that never returns is caught by the supervisor's kill (Task 6).
- Produces, in `tests/fakes.py`: `FakeDatabase`, name `"fake"`, section `property_graph`. It answers from the reference files, so it agrees. The env var `GRAPH_BENCH_FAKE` holds a comma list of `crash:<item_id>` (`os._exit(3)`), `hang:<item_id>` (sleep 3600), `slow:<item_id>` (sleep 0.2 s per call), `wrong:<item_id>` (perturbed answer) and `unsupported:<item_id>`. A second fake, `FakeDatabase2` (name `"fake2"`, same behaviour, read from `GRAPH_BENCH_FAKE2`), lets the supervisor tests compare databases.

- [ ] **Step 1: Write the failing tests:**
  - `tests/test_timing.py::test_measure_counts_iterations`: `iterations == 5`, `min_ns <= median_ns`, and the result is returned.
  - `tests/test_worker.py`, each running the worker in a subprocess with `GRAPH_BENCH_EXTRA_DATABASES=fake=tests.fakes:FakeDatabase`:
    - `test_worker_emits_database_line_first`
    - `test_worker_emits_start_before_entry`
    - `test_worker_marks_not_supported`
    - `test_worker_skips_ids`: with `--skip`, the skipped ids are absent.
    - `test_worker_writes_answers`
    - `test_worker_wrong_answer_not_timed`: `wrong:snb_is5` → `answer == "disagree"`, `timing is None`.
    - `test_worker_budget_marks_too_slow`: the fake sleeps 0.2 s on `snb_is2` (`slow:snb_is2`), with `--budget <id>=0.05` → `timeout`, error starting `"too slow"`, `timing is None`.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_timing.py tests/test_worker.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** `base.py`, the registry, `timing.py`, `results.py`, `worker.py` and `fakes.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.

---

### Task 6: Supervisor, `bench` command and results.json

**Files:**
- Create: `graph_bench/runner/supervisor.py`
- Modify: `graph_bench/cli.py` (command `bench`)
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: the worker protocol, `Entry`, `Results` (Task 5); `verify` (Task 3); reference files (Task 4).
- Produces:
  - `run_bench(selection: str, dbs: Sequence[str], out_dir: Path, *, workdir: Path, case_timeout_s: float = 600.0) -> Results`:
    - one worker per (db, threads value), run sequentially in the order given;
    - env = `os.environ` + `db_cls.process_env(threads)`;
    - passes `--budget id=seconds` for every case that already has `ok`/`agree` entries from earlier databases, computed per the budget rule from their `preflight_ns` (the same quantity the preflight is compared against);
    - reads lines as they arrive;
    - a database whose driver can't be imported gets `failed` entries with error `"not installed: <exc>"` for its supported-model cases, and `not_supported` for the rest.
  - `apply_budget_rule(results: Results) -> None`: after all databases, recomputes each benchmark's budget from the final top 3 (on `timing.median_ns` in bench mode, `preflight_ns` in codspeed mode) and turns any `ok`/`agree` entry above it into `timeout`, with error `"too slow: <x>x the top-3 average"` and its timing kept. Entries with a single run are left as they are. `reference` is `"networkx"` for Graphalytics and `"python"` otherwise; the worker sets `answer`.
  - Metadata: `graph_bench_commit` from `git rev-parse HEAD` (or `"unknown"`), `grafeo` from `.graph-bench/grafeo.json` when present (else the installed version with source `"pypi"`), `machine` from `platform` and `psutil.cpu_count`.
  - CLI: `graph-bench bench --selection S [--db a,b] [--grafeo PATH] [--case-timeout SECONDS] [--workdir DIR] --out DIR`:
    - writes `DIR/results.json` and prints its path;
    - exit 0 once written;
    - `--grafeo` calls `install_grafeo` from Task 9; until Task 9 exists it errors "not available yet".
- **Resume rule:**
  - If a worker exits non-zero, the last `start` id without an entry becomes `failed` with error `"worker exited with code <n>"`.
  - If no line arrives for `case_timeout_s`, the supervisor kills the worker and that id becomes `timeout`.
  - In both cases a new worker starts with `--skip` for every id already recorded.
  - Loads are re-run by the new worker, but their entries are not duplicated.

- [ ] **Step 1: Write the failing tests:**
  - `test_supervisor_all_agree`: fake db → every entry `ok` and `agree`; results.json round-trips.
  - `test_supervisor_resumes_after_crash`: `GRAPH_BENCH_FAKE=crash:snb_is3` → snb_is3 `failed` with "exited with code 3", and every later case `ok`.
  - `test_supervisor_marks_hang_as_timeout`: `hang:snb_is4`, `case_timeout_s=2` → `timeout`, later cases `ok`.
  - `test_supervisor_marks_wrong_answer`: `wrong:snb_is5` → `disagree`, with `detail` set.
  - `test_supervisor_passes_budgets`: with dbs `["fake", "fake2"]`, the second worker's argv contains `--budget` for the cases fake completed.
  - `test_budget_rule_is_order_independent`: a hand-built `Results` where the slow db comes first, with 4 dbs at 1, 1, 1 and 50 ms → the 50 ms entry is `timeout`; the same set in another order gives the same markers.
  - `test_missing_driver_marks_failed`: a registry entry pointing at a nonexistent module → `failed` with "not installed".
  - `test_bench_exit_zero_on_disagree`: the CLI exits 0.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_supervisor.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** the supervisor and the `bench` command.
- [ ] **Step 4: Run** the same command. Expected: PASS.

---

### Task 7: Grafeo adapter

**Files:**
- Create: `graph_bench/databases/grafeo.py`
- Test: `tests/test_db_grafeo.py` (marker `grafeo`, `pytest.importorskip("grafeo")`)

**Interfaces:**
- Consumes: the `Database` contract (Task 5).
- Produces: `GrafeoDatabase(Database)`, name `"grafeo"`, display name `"Grafeo"`, models `{"property_graph"}` (sql_pgq and rdf come in later plans), dialect `"cypher"`.
  - **`process_env`:** threads=1 → `{"RAYON_NUM_THREADS": "1"}`.
  - **`info`:** threads_configurable `True` with the threads notes from Global Constraints.
  - **`load`:**
    - `grafeo.GrafeoDB()` (in-memory), storage `"in-memory"`;
    - nodes per table via `batch_create_nodes_with_props(label, rows)`, in chunks of 10_000;
    - keeps `ids: dict[(label, id), int]`;
    - edges via `batch_create_edges([(src, dst, type, props)])`, in chunks of 10_000;
    - `create_property_index("id")`.
  - **`query`:** `db.execute_cypher(text, params)`, rows as tuples in RETURN column order.
  - **`algorithm`:**
    - bfs: `db.algorithms.bfs_layers(<internal id of source>)` mapped back to vertex ids;
    - wcc: `db.algorithms.connected_components(key="id")`;
    - pr: `db.algorithms.pagerank(damping=0.85, max_iterations=1000, tolerance=1e-10)` mapped back to vertex ids.
  - **`close`:** `db.close()`.
- Grafeo 0.6.0's `RETURN n` returns maps. That doesn't matter here, because the suites return scalars only.

- [ ] **Step 1: Write the failing tests:**
  - `test_grafeo_push_agrees`: run `run_bench("push", ["grafeo"], tmp)`; every entry is `ok` and `agree`, or `not_supported`. Print failing details to make debugging easier.
  - `test_grafeo_single_thread_env`: `process_env(1) == {"RAYON_NUM_THREADS": "1"}`.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_db_grafeo.py -v --no-cov -m grafeo`. Expected: FAIL.
- [ ] **Step 3: Implement.** Where a query needs Grafeo-specific syntax, add `text["grafeo"]` in `suites/snb.py`. Use only syntax Grafeo documents, and never change a query's meaning. If Grafeo returns a wrong answer, leave the test red for that query and record it in `docs/known-disagreements.md` (query, observed vs reference). It is a Grafeo bug to report, never one to paper over.
- [ ] **Step 4: Run** the same command. Expected: PASS, except entries listed in `docs/known-disagreements.md`, which the test reads as an allowlist.

---

### Task 8: LadybugDB adapter

**Files:**
- Create: `graph_bench/databases/ladybug.py`
- Test: `tests/test_db_ladybug.py` (marker `ladybug`, `pytest.importorskip("real_ladybug")`)

**Interfaces:**
- Produces: `LadybugDatabase(Database)`, name `"ladybug"`, display name `"LadybugDB"`, models `{"property_graph"}`, dialect `"ladybug"` (falls back to `"cypher"` text when no override exists).
  - **`load`:**
    - an in-memory `Database` (no path or `":memory:"`, whichever the installed real_ladybug documents), `max_num_threads=1`, and `Connection(db, num_threads=1)`; storage `"in-memory"`;
    - `CREATE NODE TABLE <label>(<cols>, PRIMARY KEY(id))` per node table;
    - one `CREATE REL TABLE <type>(FROM a TO b, FROM c TO d, <cols>)` per edge type, grouping the EdgeTables;
    - `COPY <table> FROM '<csv>' (HEADER=true)` per table; for rel groups, `COPY <type> FROM '<csv>' (from='<src>', to='<dst>', HEADER=true)`.
    - For the graphalytics dataset, also `INSTALL ALGO; LOAD EXTENSION ALGO; CALL project_graph('g', ['V'], ['E'])`.
  - **`query`:** `conn.execute(text, parameters=dict(params))`, rows from `get_all()`.
  - **`algorithm`:**
    - pr: `CALL page_rank('g', dampingFactor := 0.85, maxIterations := 1000, tolerance := 1e-10) RETURN node.id, rank`;
    - wcc: `CALL weakly_connected_components('g') RETURN node.id, group_id`;
    - bfs: `MATCH p = (a:V {id: $s})-[:E* SHORTEST 1..10000]->(b:V) RETURN b.id, length(p)`, plus the source at depth 0.
    - Check the exact ALGO parameter names against the installed version and adjust; the meaning (damping 0.85, converge) is fixed.
  - **`info`:** version from `real_ladybug.__version__`, threads_note `f"max_num_threads={n}"`.

- [ ] **Step 1: Write the failing test** `test_ladybug_push_agrees`, mirroring Task 7, with the same allowlist mechanism.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_db_ladybug.py -v --no-cov -m ladybug`. Expected: FAIL.
- [ ] **Step 3: Implement** the adapter and the `text["ladybug"]` overrides.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: End-to-end check.** Run `uv run --no-sync graph-bench bench --selection push --db grafeo,ladybug --out out/push`. Expected: `out/push/results.json` exists, with 2 × 13 query entries plus load entries.

---

### Task 9: `setup --grafeo`

**Files:**
- Create: `graph_bench/grafeo_setup.py`
- Modify: `graph_bench/cli.py` (command `setup`; `bench --grafeo` and `GRAPH_BENCH_GRAFEO` call it)
- Test: `tests/test_grafeo_setup.py`

**Interfaces:**
- Produces: `install_grafeo(source: Path, *, run: Callable[[list[str]], str] = _run, state_dir: Path = Path(".graph-bench")) -> dict[str, str]`, returning and writing `state_dir/grafeo.json = {"version", "source", "sha"}`:
  - **`.whl` file:** `uv pip install --python <sys.executable> --reinstall --no-deps <wheel>`. `source` is the wheel path, `sha` is the wheel's sha256.
  - **Directory:** the bindings dir is `source/crates/bindings/python` if it exists, else `source` itself (it must contain a `pyproject.toml` with `maturin`).
    1. `uvx maturin build --release --out <state_dir>/wheels -m <bindings>/Cargo.toml`
    2. install the newest wheel in that folder as above
    3. `source` is the directory, `sha` is `git -C <source> rev-parse HEAD`
  - **Version:** read by running `<sys.executable> -c "import grafeo; print(grafeo.__version__)"` in a subprocess.
  - **Anything else:** raises `ValueError("expected a .whl file or a grafeo checkout: <path>")`.
- Produces, CLI: `graph-bench setup --grafeo PATH` prints the JSON. `bench`/`codspeed` call `install_grafeo` first when `--grafeo` or `GRAPH_BENCH_GRAFEO` is given.

- [ ] **Step 1: Write the failing tests** (they record the commands through a fake `run`):
  - `test_install_wheel_commands`
  - `test_install_checkout_builds_then_installs`
  - `test_rejects_other_paths`
  - `test_writes_state_file`
  - `test_bench_honours_env_var`: monkeypatch `install_grafeo` and check it was called with the env path.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_grafeo_setup.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Manual check.** Run `uv run --no-sync graph-bench setup --grafeo H:\GrafeoDB\grafeo`. Expected: a JSON line with the checkout's sha, and `import grafeo` reports the local version.

---

### Task 10: `report --format summary|readme`

**Files:**
- Create: `graph_bench/report/{__init__,ranking,summary,readme}.py`
- Modify: `graph_bench/cli.py` (command `report`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Results` (Task 5).
- Produces, in `ranking.py`:
  - `Ranked(category, benchmark, title, scale, threads, order: list[tuple[str, float]])`: the databases with `status == "ok"` and `answer == "agree"`, by `timing.median_ns` (by `preflight_ns` in codspeed mode);
  - `rank(results: Results) -> list[Ranked]`.
- Produces, in `summary.py`, `render_summary(results: Results, *, results_url: str = "https://github.com/GrafeoDB/graph-bench#results") -> str`, the spec's "Summary block" exactly:
  - It names no database other than Grafeo anywhere.
  - Header: `**graph-bench** · Grafeo <version> (<sha[:7]>) · <cpu>, <cores> cores, <os> · <YYYY-MM-DD> · <N> embedded databases`, where N is the number of databases in `results.databases`.
  - Uses threads=1 entries at the largest scale present.
  - One row per category in `CATEGORY_ORDER` where Grafeo has at least one entry.
  - Columns:
    - `DBs`: the count of databases with at least one `ok`/`agree` entry in the row;
    - `Grafeo fastest`: `k / n`;
    - `Grafeo median rank`: an integer if whole, else one decimal.
  - Under the table:
    - `Single-threaded, <scale> scale; answers checked; wrong answers not ranked.`;
    - then `Grafeo: <marker> on <benchmark>` lines for Grafeo's own markers, plus the legend line when any exist;
    - then `Full results, with every database named: <url>`.
  - Wrapped in `<!-- graph-bench:start -->` / `<!-- graph-bench:end -->`.
- Produces, in `readme.py`:
  - `render_readme(results: Results) -> str`, the matrix:
    1. **Overview table:** a row per category in `CATEGORY_ORDER` that has entries, a column per database (display names, Grafeo first, then alphabetical). A cell holds `✓ k/n`, where k is the correct and in-budget count and n the category's benchmark count, followed by marker counts such as `≠1 ⏱2`. A category where the db has nothing but ⊘ shows a bare `⊘`.
    2. **Detail tables:** one per category, with a row per benchmark (`title`) and a column per database. Cells hold the median in ms with 3 significant digits and the fastest in bold, or the marker alone. Load is its own category table, with seconds.
    3. **Footer:** the legend `⊘ not available · ✗ failing · ≠ wrong answer · ⏱ beyond 10x the top-3 average`, the run metadata, the per-database threads notes and the database versions.
  - `update_readme(path: Path, body: str) -> None`: replaces the text between `<!-- results:start -->` and `<!-- results:end -->`, raising if the markers are missing.
- Produces, CLI: `graph-bench report --format summary|readme --in results.json [--write README.md]`. It prints to stdout unless `--write` is given, which is only valid for `readme`.

- [ ] **Step 1: Write the failing tests** against a handwritten `Results` fixture with 2 categories, 4 benchmarks and 3 databases (`grafeo`, `ladybug`, `fake`), covering each marker at least once:
  - `test_summary_exact_output`: the full string equals an expected literal in the test, for the agreed format.
  - `test_summary_names_no_other_database`: neither `LadybugDB` nor `fake` appears in the summary, and `3 embedded databases` does.
  - `test_summary_lists_grafeo_markers`: a Grafeo `≠` entry yields the line `Grafeo: ≠ on <benchmark>` and the legend.
  - `test_summary_excludes_disagree_from_ranking`
  - `test_summary_has_no_combined_score`: no line contains "overall" or "total score".
  - `test_readme_overview_counts`: the overview cell for a db with one ≠ and one ok in a 2-benchmark category reads `✓ 1/2 ≠1`.
  - `test_readme_detail_markers`: each of ⊘ ✗ ≠ ⏱ appears in the detail table cell for its entry, and the fastest time is bold.
  - `test_update_readme_replaces_between_markers`
  - `test_update_readme_requires_markers`
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_report.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Notify grafeo-31.** Run `uv run --no-sync graph-bench report --format summary --in out/push/results.json`, then message `grafeo-31` with the exact output and the command sequence (`setup --grafeo <wheel>`, `bench --selection full --out DIR`, `report --format summary --in DIR/results.json`).

---

### Task 11: pytest-codspeed entry point and `codspeed` command

**Files:**
- Create: `benches/conftest.py`, `benches/_cases.py` (shared test factory), one module per category in `CATEGORY_ORDER` that has catalog items (`benches/test_load.py`, `test_complex_queries.py`, `test_hops.py`, `test_algorithms.py` in this plan)
- Modify: `graph_bench/cli.py` (command `codspeed`), `graph_bench/runner/supervisor.py` (reuse the budget rule and the results writer)
- Test: `tests/test_codspeed.py`

**Interfaces:**
- Consumes: `cases`, `load_ids` (Task 4), `get_database`, `check_case` (Task 5), `apply_budget_rule`, `Results` (Task 6), `install_grafeo` (Task 9).
- Produces, in `benches/conftest.py`:
  - Options: `--db NAME` (required), `--selection S`, `--threads 1|all`, `--gb-out FILE.jsonl`, `--workdir DIR`, `--gb-budget ID=SECONDS` (repeatable), `--case-timeout SECONDS`.
  - A session fixture opening and loading the database once per dataset.
  - A `pytest_generate_tests` hook parametrizing each `test_<id>` function over the cases for that db, with ids `<db>-<scale>[-allcores]`.
- Produces, in `benches/_cases.py`: `make_test(item) -> Callable`. Each generated test runs the worker's preflight logic (Task 5) and maps the outcome to pytest, always appending the entry in the worker's jsonl format first (`preflight_ns` set, timing null):
  - model not supported or `NotSupported` → `pytest.skip("not available: ...")`;
  - exception → `pytest.fail("error: ...")`;
  - over budget → `pytest.fail("too slow: ...")`;
  - `disagree` → `pytest.fail("wrong answer: <detail>")`;
  - otherwise `benchmark(run)`.

  `benchmark` is called only on the last path, so CodSpeed never receives a measurement for a marked case.
- Produces, in each `benches/test_<category>.py`: `globals()[f"test_{item.id}"] = make_test(item)` for that category's items. `test_load.py` holds `test_load_snb` and `test_load_graphalytics`, using `benchmark.pedantic(load, setup=<fresh workdir>, rounds=<load_iterations>)`.
- Check that the installed pytest-codspeed supports `pedantic` and that `benchmark(fn)` returns fn's result. If either is missing, raise the version floor in pyproject rather than working around it.
- Produces, CLI: `graph-bench codspeed --selection S [--db a,b] [--grafeo PATH] [--case-timeout SECONDS] --out DIR`:
  - one `[sys.executable, "-m", "pytest", "benches", "--codspeed", "--db", db, ...]` subprocess per (db, threads), env = `os.environ` + `process_env(threads)`;
  - passes `--gb-budget` from the earlier databases' `preflight_ns`, per the budget rule;
  - never removes or sets `CODSPEED_PROFILE_FOLDER`;
  - the same hang handling as the supervisor: on silence beyond `--case-timeout`, kill, mark the case `timeout` and rerun pytest with `--deselect` for the done ids;
  - then `apply_budget_rule` on `preflight_ns` and write `DIR/results.json` with `mode = "codspeed"`;
  - exit 0 once written, whatever the pytest exit codes.

- [ ] **Step 1: Write the failing tests:**
  - `test_codspeed_names_match_list`: `pytest benches --collect-only -q --db fake --selection push` lists exactly the ids `graph-bench list --selection push --db fake` prints.
  - `test_codspeed_passes_profile_folder`: monkeypatch the subprocess launcher and set `CODSPEED_PROFILE_FOLDER=/x`; every call's env contains it unchanged.
  - `test_codspeed_writes_results_without_timings`: a fake db run without `--codspeed` (the benchmark fixture still calls fn) gives results.json with `mode == "codspeed"`, timing null and answers `agree`.
  - `test_codspeed_marker_outcomes`: with `GRAPH_BENCH_FAKE=unsupported:snb_is1,crash_soft:snb_is2,wrong:snb_is3,slow:snb_is4` and `--gb-budget <is4 id>=0.05`, the pytest report shows is1 skipped with reason `not available:`, is2/is3/is4 failed with `error:` / `wrong answer:` / `too slow:`, and a spy on `benchmark` shows it was never called for those four. `crash_soft` raises `RuntimeError` instead of exiting; add it to `tests/fakes.py`.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_codspeed.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run with the real databases.** Run `uv run --no-sync graph-bench codspeed --selection push --db grafeo,ladybug --out out/cs`. Expected: results.json written, and pytest output shows names like `test_snb_is1[ladybug-small]`.

---

### Task 12: CI workflows, docs and handoff

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/bench.yml`
- Modify: `README.md` (rewrite: what it is, the matrix categories, the ten planned databases and their status, the marker legend with its CodSpeed mapping, commands, a `<!-- results:start -->` / `<!-- results:end -->` section), `FAIRNESS.md` (trim to embedded use; add the spec's rules 1 to 8, the preflight, and the planned all-cores change with grafeo 0.6.4), `CONTRIBUTING.md` (how to add an adapter: subclass `Database`, register, pass the push agreement test), `CHANGELOG.md`
- Test: `tests/test_workflows.py`

**Interfaces:**
- `ci.yml`:
  - on push and pull_request, ubuntu-latest, Python 3.12 and 3.13;
  - `uv sync --frozen --extra embedded --extra bench --extra dev`, `ruff check`, `ruff format --check`, `pytest`;
  - the integration tests run there too, since grafeo and real_ladybug install from PyPI.
- `bench.yml`:
  - triggers: `push` to main, `schedule` (daily cron), `workflow_dispatch`;
  - job `codspeed`, on `codspeed-macro`:
    1. checkout graph-bench, then `GrafeoDB/grafeo` at `main` into `grafeo/`
    2. `dtolnay/rust-toolchain@stable` and `Swatinem/rust-cache` (workspace `grafeo`)
    3. `astral-sh/setup-uv`, `uv sync --frozen --extra embedded --extra bench`
    4. `uv run --no-sync graph-bench setup --grafeo grafeo`
    5. `actions/cache` on `~/.cache/graph-bench`, keyed by `graph-bench data key --selection push`
    6. `data prepare`, then `list --selection push` (printed in the log)
    7. `CodSpeedHQ/action` (current major, checked at implementation) with `mode: walltime`, `run: uv run --no-sync graph-bench codspeed --selection push --out out`
    8. upload `out/results.json` as an artifact
  - job `readme`: `schedule` and `workflow_dispatch` only, on `codspeed-macro`:
    1. the same setup, but with the selection `full`
    2. `graph-bench bench --selection full --out out`
    3. `graph-bench report --format readme --in out/results.json --write README.md`
    4. `peter-evans/create-pull-request` on branch `bot/results`, title `Update benchmark results`
  - Comment at the top of `bench.yml`: "graph-bench's own CodSpeed project. Separate from grafeo's CodSpeed benchmarks, which it does not replace."
  - Comment on the `codspeed-macro` lines: the user enables the CodSpeed app for graph-bench before the workflow can upload.

- [ ] **Step 1: Write the failing test** `tests/test_workflows.py::test_bench_workflow_shape`. Load `bench.yml` with `yaml.safe_load` (add `pyyaml` to `dev`) and assert:
  - the triggers include `push`, `schedule` and `workflow_dispatch`;
  - the `codspeed` job's `run` uses `graph-bench codspeed --selection push`;
  - the `readme` job runs only on schedule/dispatch.
- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_workflows.py -v --no-cov`. Expected: FAIL.
- [ ] **Step 3: Write the workflows and docs.** Docs follow the repo rules (no dashes as punctuation, `uv` only).
- [ ] **Step 4: Full verification.** Run `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest`. Expected: all pass, coverage >= 88.
- [ ] **Step 5: Hand off.** Message `grafeo-31`: the foundation works with Grafeo + Ladybug, with the exact commands and file paths, and the load benchmark names `test_load_snb` / `test_load_graphalytics` (refined from the earlier `test_load`). Tell the user the workflow needs the CodSpeed app enabled on the graph-bench repo.
