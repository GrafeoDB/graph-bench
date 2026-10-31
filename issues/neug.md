# NeuG: nothing to file

NeuG 0.3.0 agrees with the references on every checked case in `docs/known-disagreements.md`. No correctness bug or
defect was found, so there is no issue draft.

## Not filed (by design or performance only)

- PageRank is slow under the 10x budget rule. `page_rank` has no tolerance option, so it likely runs a fixed number of
  rounds; the answer agrees. Performance only.
- Load is slow (a checkpoint per COPY statement). Counts and queries agree. Performance only.
