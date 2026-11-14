# Changelog

All notable changes to this project are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every number below is a result of the project's own runs (*measured*), recorded when the pull request merged.

## [0.1.0] - unreleased

### I/O fix

- A reopened file stream made every buffer pool miss cost about 0.65 ms (*measured*); `DiskManager` now reads and writes pages with `pread` and `pwrite` on one open file. Every latency recorded before this fix is invalid. (#22)

### Planner and joins

- Join-order headroom tool (`join_headroom`) times every connected join order; at SF=1.0 only Q5 (1.82x, *measured*) and Q12 (1.13x, *measured*) missed the best order. (#23)
- Hash joins build from their smaller input (SF=1.0 Q5 15% faster, *measured*). (#24)
- Column references are bound to their index once, at plan time. (#25)
- Range selectivity from each column's min and max, and filter-aware join row counts; the classical planner now picks Q5's best order. (#26)
- Join output rows are built by copying bytes instead of rebuilding values (SF=1.0 Q5 56% faster, *measured*). (#27)
- Scan costs calibrated in microseconds per page and per row; with indexes, Q14 and Q10 read through an index on purpose. (#31)

### Buffer pool and eviction

- LRU without allocations on the hot path (every query 23-54% faster, *measured*). (#28)
- A realistic miss cost (`--miss-penalty-us`, `--direct-io`) and a page-level hit rate reported beside the fetch-level one. (#29)
- Workloads with reuse (`stream`, `lookup`, `mixed`); fixed a B+tree range scan that missed a key's first entry when its copies crossed a leaf split. (#30)
- Eviction headroom against Belady's optimal policy; LIRS and ARC, counting page visits, are the best classical policies. (#32)
- CLOCK, LRU-2, ARC and LIRS in the engine (`--eviction-policy`), deciding exactly as the simulator on every workload item. (#33)

### Vectorized path

- Batches keep each row's bytes and decode columns only when an operator reads them; vectorized mode is now faster than tuple mode on 6 of the 8 TPC-H queries (*measured*). (#34)

### The checks

- Learned eviction V3 check: no-go (best cell 1.8% over ARC at a 10% pool against a 5% bar, *measured*). (#35)
- Policy-switching headroom check: no-go (at most 2.55% in hindsight against a 5% bar, *measured*). (#36)
- Vector-search check: part 1 a narrow go (a coverage claim, not a speed claim), part 2 no verdict, recorded as a fail. (#38)
- AI-analyst check: go. On 300 test questions the checks warned about 20 of the model's 35 silent wrong answers (0.571), refused none of its 227 right ones and warned on 18 of them (0.079) (*measured*). (#39)

### Repository

- `bench/` grouped by topic (TPC-H, join, eviction, vector check), no behavior change. (#37)

### Earlier

Pull requests #1-#21 built the engine from the ground up:

- Storage: raw 8 KB page I/O, slotted pages for variable-length rows, heap files, and a buffer pool with pin counts and pluggable eviction.
- A B+tree index with point lookups and range scans, a catalog with per-column statistics, and a hand-written SQL lexer and recursive-descent parser.
- A cost-based planner with exhaustive join ordering, Volcano-style executors (scans, filter, hash and nested-loop joins, sort, aggregate, projection, limit), automatic indexes, and a hybrid vectorized path.
- The TPC-H loader and 8 queries checked against SQLite, and two learned components, a join-order model and an eviction policy (V1 and V2), kept as measured research and off by default.
