# Neo4j: nothing to file

Neo4j (embedded Community, 2026.10.1) agrees with the references on every checked case in
`docs/known-disagreements.md`. No correctness bug or defect was found, so there is no issue draft.

## Not filed (by design or performance only)

- IC13 with the same person twice raises "The shortest path algorithm does not work when the start and end nodes are
  the same." by default. This is documented behaviour with a documented setting
  (`dbms.cypher.forbid_shortestpath_common_nodes`, which the adapter sets to false), and the answers then agree.
- Unlabeled id lookups start from a label scan (IS5, IS6), slow cases under the 10x budget rule, the fixed cost of
  `neo4j-admin import`, and GDS notes: performance or design only.
