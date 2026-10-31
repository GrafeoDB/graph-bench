# Grafeo: tracked in the grafeo repository

Grafeo issues are tracked by the Grafeo team and go to the grafeo repository, not through this directory. Nothing is
drafted here. The open items below are copied from `docs/known-disagreements.md` for reference, with the Grafeo
version they were measured on (0.6.0, release/0.6.1 builds).

## Open items (correctness or defect)

- IC13 shortest path does not finish (`grafeo/snb_ic13`): `OPTIONAL MATCH ... shortestPath` is planned as a
  `LeftJoin` whose right side is an uncorrelated `ShortestPath` over all node pairs, then filtered. The same pattern
  with a plain `MATCH` answers in about 1 ms. Cause: query planner.
- SQL/PGQ parser (`grafeo/snb_sql_is4`, `snb_sql_is6`, `snb_sql_ic13`): the label disjunction `Post|Comment` and the
  `ANY SHORTEST` path prefix do not parse. Confirmed by grafeo-31 on release/0.6.1.

## Open items (performance only, answers agree)

- `snb_is7` and `snb_is4`: uncorrelated `LeftJoin` and a label-disjunction scan that skips the property index.
- Medium-scale SPARQL IS1, IS4, IS6 against Oxigraph (reported to grafeo-31); Cypher `snb_ic13` on `snb-medium`
  (grafeo-28 profiles it).
- LOAD CSV is 2x to 3x slower than the batch API (per-row cost of an indexed `MATCH` plus `CREATE`).
- SPARQL IS6 property path (`REPLY_OF*`) is slow on `snb-small`.
