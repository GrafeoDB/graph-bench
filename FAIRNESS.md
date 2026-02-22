# Fairness Policy

This document describes the rules that govern graph-bench benchmarks.
Every adapter, whether maintained by the project or contributed externally, must comply with these rules.
Pull requests that optimise an adapter are welcome and will be reviewed and merged provided they follow the guidelines below.

## Standard Adapter Parameters

All adapters inherit from `BaseAdapter` and must respect the default method signatures.

| Parameter | Default | Rule |
| --------- | ------- | ---- |
| `batch_size` (node insert) | 1000 | May be lowered to work around driver limits (e.g. TuGraph uses 200), but must not be raised above 1000 |
| `batch_size` (edge insert) | 1000 | Same as above (Grafeo Server uses 500 to avoid gRPC deadline) |
| Indexing | Optional | Adapters may create a property index on `id` after bulk insert. No other pre-computation is allowed during `insert_nodes` / `insert_edges` |
| Warmup iterations | Per scale | Defined in `ScaleConfig`, not adjustable per adapter |
| Measurement iterations | Per scale | Defined in `ScaleConfig`, not adjustable per adapter |
| Timeout | Per scale | 60 s (small) to 3600 s (sf100). CLI `--timeout` overrides apply equally to all adapters in a run |

## Allowed Optimisations

The following optimisations are explicitly permitted.

1. **Native algorithms**: if the database ships a built-in algorithm (PageRank, BFS, SSSP, etc.), the adapter should use it instead of the NetworkX fallback. The algorithm must be a standard feature of the database, not a benchmark-specific plug-in.

2. **Native traversal**: adapters may use database-native BFS/DFS/shortest-path primitives. When the native primitive does not support a required parameter (e.g. `edge_type` filtering), the adapter must fall back to the `BaseAdapter` implementation.

3. **Batched writes**: adapters may batch `INSERT` / `CREATE` operations using `UNWIND`, multi-pattern `CREATE`, or the database's native bulk API, within the `batch_size` limit.

4. **Property indexes**: a single index on the `id` property is allowed to speed up `MATCH` lookups. It must be created after insert, not before.

5. **Connection pooling / multiplexing**: adapters may maintain connection pools, session-per-thread maps, or reuse HTTP/2 streams, as long as these are standard features of the client driver.

6. **Native query language**: each adapter uses the database's own query language (Cypher, AQL, GQL, etc.). This is not an unfair advantage, it reflects real-world usage.

## Disallowed Optimisations

The following are not permitted.

1. **Pre-computation during setup**: adapters must not pre-compute answers, build auxiliary data structures, or cache results during `insert_nodes` / `insert_edges` / `clear` that would give an unfair advantage to subsequent read or algorithm benchmarks.

2. **Result caching**: no caching of query results between benchmark iterations. Each measurement iteration must perform the full operation.

3. **Benchmark-specific code paths**: adapters must not detect which benchmark is running and switch behaviour accordingly. All optimisations must be general-purpose.

4. **Custom plug-ins or extensions**: only built-in, officially shipped features of the database may be used. External plug-ins, user-defined procedures, or custom modules written specifically for the benchmark are not allowed. Standard extensions that ship with the default install (e.g. Neo4j GDS, Memgraph MAGE) are permitted.

5. **Parallel execution within a single operation**: unless the database driver does it transparently, adapters must not spawn threads or processes to parallelise a single benchmark call (e.g. running multiple MATCH queries in parallel to speed up a single `pagerank()` call). Concurrent traversal using the driver's built-in multiplexing (e.g. gRPC HTTP/2 streams) is allowed.

6. **Raising `batch_size` above 1000**: the default batch size is 1000. Adapters may lower it for stability but must not raise it.

7. **Skipping data verification**: adapters must not skip writes or silently drop data to appear faster. All `insert_nodes` / `insert_edges` calls must attempt to insert every item.

## Algorithm Fallback Rules

`BaseAdapter` provides NetworkX-based fallback implementations for graph algorithms (PageRank, community detection, BFS levels, WCC, LCC, SSSP) and a brute-force fallback for vector search.

- Databases **without** native support use the fallback automatically. The extraction + computation overhead is included in the benchmark time: this is by design and represents the real cost of missing native support.
- Databases **with** native support should override the fallback methods. The override must produce semantically equivalent results (e.g. same community structure, correct shortest-path distances).
- If a native implementation fails at runtime, the adapter may fall back to `super()`. This fallback penalty is also included in the timed result.

## Test Inclusion / Exclusion Criteria

### Including a benchmark

A benchmark is included in a run when:

- It matches the `--category` filter (or no filter is set, meaning all categories run).
- It matches the `--benchmarks` name filter (or no filter is set).
- The adapter successfully connects to the database.

### Excluding / skipping a benchmark

A benchmark is excluded when:

- The adapter fails to connect (the entire database is skipped).
- The category or name does not match the CLI filter.

There is no mechanism for an adapter to selectively skip individual benchmarks. If a benchmark times out or throws an exception, it is recorded as `TIMEOUT` or `FAILED`, not silently omitted.

### Timeout and failure handling

- Each benchmark runs under the scale-defined timeout (enforced via `ThreadPoolExecutor`).
- `TIMEOUT` and `FAILED` results are recorded and included in the report.
- `continue_on_error = True` by default: one failure does not abort the remaining benchmarks.

## Dataset Rules

- All adapters in a run receive the **same generated dataset** for each benchmark.
- Dataset generation is deterministic (seeded RNG) to ensure reproducibility.
- LDBC Graphanalytics benchmarks use a reduced-scale synthetic directed graph (1/10th of the scale's node/edge count, capped at 10K nodes / 50K edges).
- LDBC SNB benchmarks use a full multi-label social network dataset matching the LDBC specification.

## Contributing Adapter Optimisations

Pull requests that improve adapter performance are encouraged. To be merged, a PR must:

1. **Follow this fairness policy**: no disallowed optimisations.
2. **Touch a single adapter**: each PR may only optimise one adapter. Changes that affect `BaseAdapter` or shared infrastructure belong in a separate PR.
3. **Reuse `BaseAdapter`**: adapters must inherit from `BaseAdapter` and call `super()` for any operation they do not natively support. Do not re-implement fallback logic that already exists in `base.py`.
4. **Pass all existing benchmarks**: no regressions in correctness.
5. **Be general-purpose**: the change must benefit all benchmarks, not just a specific one.
6. **Document the change**: explain what was optimised and why (e.g. "use native SSSP instead of NetworkX fallback").
7. **Include before/after results**: run `graph-bench run -d <adapter> -s small` before and after, and include the comparison in the PR description.

If in doubt about whether an optimisation is fair, open an issue to discuss it before submitting a PR.
