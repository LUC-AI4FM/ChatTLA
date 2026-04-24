# Ralph-gen TLA+ dataset expansion

Source: 9 parallel Opus agents, each running a ralph loop against
`scripts/data/ralph_gen_tla.py validate` (SANY + TLC + Diamond gate with
`min_kill_rate=0.5, min_coverage=0.3`).

- 168 pre-existing rows (tlaplus/examples, ChatTLA, FormaLLM) carried over
- 476 new ralph-gen rows across 12 topics
- Final splits: 500 train, 50 dev, stratified round-robin on (topic, difficulty)

Per-topic accepted (partitions under `benchmarks/tla/gen/`):

| topic                    | accepted |
|--------------------------|----------|
| mutual_exclusion         |       55 |
| scheduling               |       51 |
| workflows                |       60 |
| communication_protocols  |       50 |
| concurrency_primitives   |       50 |
| distributed_systems      |       57 |
| transactions             |       40 |
| databases                |       12 |
| caches                   |       15 |
| puzzles                  |       15 |
| data_structures          |       20 |
| counters_small           |       51 |

Every accepted row passes: SANY parse, TLC verify (≤25s), Diamond mutation
kill-rate ≥ 0.5, coverage ≥ 0.3 on the inline-Next mutation operator set.
Random 15-row re-verify sample on train: 15/15 clean.
