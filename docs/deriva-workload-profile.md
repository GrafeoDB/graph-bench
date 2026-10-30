# Deriva workload profile (input for the "Deriva workload" category)

Source: the deriva-d8 session, 2026-10-30, from deriva v0.9.0 (branch minor/0.9.0) code and recorded timings. Nothing was run against Deriva's own graph files. Code lives under `H:\Deriva\deriva`.

## What matters

- **Graph time vs LLM time:** graph time is small next to LLM time. In a cached 7-repo session, extraction took 62.7 s and derivation 61.5 s, and the top 15 query shapes added up to 8.5 s. In a live-LLM session (3619 s) they added up to 0.8 s.
- **When graph latency matters:** for cached or replayed runs and for re-extraction.
- **The pattern to benchmark:** many tiny single-row writes plus filter-heavy reads, not a bulk load.
- **Latency-critical, in order:**
  1. row-by-row upsert throughput during extraction
  2. batch property write-back after the algorithms
  3. `DETACH DELETE` of a whole repo
  4. the edge-property dedupe lookup with many Model relationships

## Roundtrip stages (one process opens one file per workspace key)

| Stage | What | Kind |
|---|---|---|
| a | open or create the file, property indexes on `id` (Graph) and `identifier` (Model) | DDL |
| b | read the Repository node's `extraction_fingerprint` | point read |
| c | if stale: count, then `MATCH (n:Graph) WHERE n.repository_name=$repo DETACH DELETE n` | bulk delete |
| d | extraction steps: a single-row node upsert by id (`replace=True`, labels `[Graph, <Type>]`), a single-row edge upsert by edge id with endpoints looked up by `id`, interleaved with existence reads `MATCH (n) WHERE n.id = $node_id RETURN count(n) > 0 as exists` (10,527 calls in the 7-repo session) | incremental upsert |
| e | per-step sweep: read nodes with `extractSteps` and this step's edges, batch-set properties, delete what the step no longer produces | read + targeted delete |
| f | set `extraction_fingerprint` | point update |
| g | prep: on a projection of the `Graph` label, run PageRank (undirected, 0.85, 100 iter, tol 1e-6), Louvain (resolution 1.0), k-core, articulation points and degree centrality, then batch write-back by id | algorithms + batch update |
| h | clear the Model namespace (`DETACH DELETE` on labels starting with `Model`) | bulk delete |
| i | derivation: about 17 candidate read queries (filter-heavy, some 2-hop), then single-row `merge_node` of Model elements; relationships via a dedupe check on an edge property (no index), then MATCH source, MATCH target, CREATE edge | reads + incremental writes |
| j | refine: whole-Model reads, deduplication, orphan removal | read + update/delete |
| k | export: all enabled Model elements and relationships | full read |

Switching repos means closing one file and opening another. That switch is worth timing.

## Shape

- **Labels:** every node carries a namespace label (`Graph` or `Model`) plus a type label.
- **Edge types:** stored as `Graph:CONTAINS` and so on.
- **Scoping:** by the `repository_name` property and by ids of the form `prefix::repo::path`.
- **Graph node types:** Repository, Directory, File, TypeDefinition, Method, ExternalDependency, Technology, BusinessConcept, DataEntity, DataDomain, DeployableUnit, BuildModule, Deployment, Software, Environment, Lifecycle, LifecycleEvent, Role, Test.
- **Graph edge types:** CONTAINS, CALLS, REFERENCES, INCLUDES, IMPORTS, USES, INHERITS, CONFIGURES, SERVES, NAMED_IN, REPRESENTS, RUNS, DEPLOYS, CONSUMES, HOSTS, TRIGGERS, ASSIGNED, BUILDS, NESTS.
- **Model:** ArchiMate element types (ApplicationComponent, DataObject, ...) and relationship types (Serving, Realization, Assignment, Composition, Access, Aggregation, Association, ...).
- **Small repo:** 1,040 nodes and 2,058 edges.
  - TypeDefinition 350, Method 185, File 166, ExternalDependency 62, Directory 42.
  - CONTAINS 695, USES 319, INCLUDES 199.
  - Model: about 170 elements and 240 relationships.
- **Large repo:** 4,476 nodes and 8,741 edges.
  - Method 2,135, TypeDefinition 1,556, File 574.
  - CONTAINS 4,306, CALLS 1,984, REFERENCES 830, INCLUDES 734, IMPORTS 507.
  - The largest workspace file is about 1.7x this.
- **Properties:**
  - native scalars and lists, no JSON blobs;
  - Graph nodes have 20 to 31 properties, 0.6 to 1.6 KB median, up to 80 KB (one TypeDefinition) and about 11 KB (Environment);
  - Model nodes have 6 properties of 0.5 to 1 KB.
- **No embeddings, no vector search, no full-text index.**

## Dominant query shapes

| id | Shape |
|---|---|
| Q1 | node upsert by id (API): `upsert_nodes([Graph, Type], [{id, ...}], key="id", replace=True)`, 5,898 times for Method alone |
| Q2 | edge upsert (API): `upsert_edges("Graph:CONTAINS", [{__src, __dst, id, ...}], key="id", endpoint_key="id", endpoint_labels=["Graph"])`, 11,655 times for CONTAINS alone |
| Q3 | `MATCH (n) WHERE n.id = $node_id RETURN count(n) > 0 as exists` |
| Q4 | `MATCH (n:Graph) WHERE n.repository_name = $repo_name DETACH DELETE n` |
| Q5 | sweep reads: `MATCH (n) WHERE n.extractSteps IS NOT NULL AND (n.repository_name = $repo OR n.id CONTAINS $marker) RETURN n.id AS id, n.extractSteps AS steps` and `MATCH (n)-[r]->(b) WHERE r.extractStep = $step AND (...) RETURN n.id, type(r), b.id` |
| Q6 | candidate reads: label conjunctions, `EXISTS { MATCH ... }` with STARTS WITH, regex `=~ '(?i).*(route\|api\|endpoint\|controller).*'`, `ANY(d IN COALESCE(n.decorators, []) WHERE d =~ ...)`, `'Environment' IN labels(n)`, IN-lists, `ORDER BY n.pagerank DESC`. Texts are in `deriva/adapters/database/data/derivation_config.json` (`input_graph_query`) |
| Q7 | `MATCH (n) WHERE n.id IN $source_ids MATCH (n)-[*1..2]-(neighbor) WHERE neighbor.active = true OR neighbor.active IS NULL RETURN DISTINCT neighbor.id as id` |
| Q8 | pagerank, louvain, kcore, articulation_points, degree_centrality on a label projection, then batch write-back by id |
| Q9 | ``MATCH (source:Model {identifier: $source}) MATCH (target:Model {identifier: $target}) CREATE (source)-[r:`Model:Serving` {identifier: $identifier, name: $name, documentation: $documentation, properties: $properties}]->(target)``, after the edge-property dedupe check |
| Q10 | `MATCH (e:Model) WHERE e.enabled = true RETURN e.identifier, e.name, [lbl IN labels(e) WHERE lbl <> 'Model'][0] as element_type, e.documentation, e.properties, ...` |

## Files to read when building the generator

- `deriva/adapters/grafeo/manager.py`: every DB call, indexes, upserts, projections, clear.
- `deriva/adapters/graph/manager.py`: the Graph namespace API, `graph_metric` (l.459), `clear_graph_for_repo`, fingerprint.
- `deriva/adapters/archimate/manager.py`: Model writes, `add_relationship` (l.162), `clear_model`.
- `deriva/services/extraction.py`: fingerprint (l.220), step sweep (l.585-637).
- `deriva/services/derivation.py`: prep algorithms (l.1494-1522), `clear_model` (l.159).
- `deriva/modules/derivation/base.py`: 2-hop query (l.1486-1525).
- `deriva/modules/derivation/refine/`
- `deriva/adapters/database/data/derivation_config.json`: candidate query texts.
- `deriva/adapters/graph/models.py`: node property schemas.
- `workspace/benchmarks/<session>/timings.json` (`top_queries`): for example `bench_20261026_142434` (cached, with extraction) and `bench_20261026_154956` (live LLM). It keeps only the 15 shapes with the highest total time, so cheap high-count shapes may be missing.

## Proposed benchmarks (for the later plan)

`deriva_upsert_rows` (Q1 + Q2 + Q3 interleaved, row by row), `deriva_repo_delete` (Q4), `deriva_sweep` (Q5), `deriva_candidates` (Q6, a fixed set of the config's texts), `deriva_two_hop` (Q7), `deriva_prep` (Q8 with write-back), `deriva_model_write` (Q9 with dedupe), `deriva_export` (Q10), `deriva_roundtrip` (the full stage sequence for small and large repos, including the file switch). Each is checked against a Python reference over the generated repo graph.

Fairness notes for that plan:
- Q1 and Q2 are Grafeo API calls. Every database runs the equivalent in its own idiom (a MERGE query where there's no upsert API), and the matrix footnotes it.
- Louvain and k-core answers vary between valid solutions, so they are checked by modularity / k-core validity rather than exact equality.
