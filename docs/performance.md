# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 41,760 boards, 40,093 drive units, 150 reels, 1,445
shop-floor events, 39,527 delivery rows, about 42,000 genealogy edges for the approved model. Reproduce with `make demo`, then
`make load-test`.

## Reads: a recall's impact, one unit's genealogy and the plant summary, in rotation, closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 327 | 32.7 req/s | 18.3 ms | 58.3 ms | 64.5 ms | 0.00% |
| 8 | 854 | 84.7 req/s | 69.8 ms | 179.3 ms | 430.9 ms | 0.00% |
| 32 | 914 | 89.1 req/s | 300.2 ms | 402.6 ms | 506.1 ms | 0.00% |

Hardware: arm64, 12 cores, client and server on the same machine. Every read is SQL over indexed tables; the plant summary is the
heaviest (counts over 80,000 serial rows and a self-join of the reel events). The probabilities are computed once, in the scope job,
and stored as genealogy edges, so tracing a unit never re-runs the model.

## Compute

| Work | Time |
|---|---|
| Generate sixty days of the plant, its records and the truth (in-process) | 0.1–0.4 s |
| Load it into Postgres (the generate job: 82,000 serial rows, events, deliveries, returns) | 2.1 s |
| Scope job: rebuild the records from the tables, learn the record models, every posterior, the optimiser and three baselines, store the edges | 0.5 s |
| The engine alone, from records in memory | 0.05 s |
| Evaluation: three plants, every strategy, the policy table and the broken-assumption variants | 2 s |

Not measured: a plant with millions of serialised units a year. The engine is vectorised per line and per reel and the exact lot
posterior is quadratic in a delivery's reels, so the cost there would be in reading the records from the database, not in the model.
