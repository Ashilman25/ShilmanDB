# Changelog

All notable changes to this project are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every number below is a result of the project's own runs (*measured*), recorded when the pull request merged.

## [Unreleased]

### App

- A first slice of the app in `app/`, started with `.venv-ai/bin/python -m app`: one local Python process, serving on `127.0.0.1` only, holds the engine (through the analyst check's bridge) and the local model (`gemma4-26b` through MLX), with a TypeScript page and no framework.
- A SQL runner that checks typed SQL for the engine's known traps before it runs, and the AI analyst: the first answer at once, then the second phrasing and the verdict beside it, on the built-in TPC-H and product-review data.
- A screen laid out like a database tool: a schema sidebar with row counts, columns, types and keys; a tab per table with a data grid and column notes; a Query tab with SQL highlighting and completion, a results grid and chart, and a Checks inspector; a status bar.
- The analyst check's rates are shown only while the tested setup holds (its files, the model settings and today's engine); re-asked questions say they aren't part of the test, and typed or edited SQL shows no rates.
- A prompt cache, on only while a replay proves it changes nothing: the replay matched all 300 test answers in each of two runs, and the median wait fell from 2.51 to 1.50-1.62 s for the first answer and from 5.29 to 3.32-3.52 s for the verdict (*measured*).
- 388 tests for the app, run in CI with the bench tests (`pytest bench app`); no test loads the model.
- The app's files are organized into folders (`core/`, `model/`, `answers/`, `server/`, `web/`); the commands are unchanged.

## [0.1.0] - 2026-11-15

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

### Baseline, docs and a professional repo

- Baseline runners for DuckDB 2.0.0 and Postgres 17.11 (its own server, the same 8 indexes), and `run_benchmarks.py --duckdb` / `--postgres`.
- A third answer result, DRIFT (only decimal numbers differ, by at most 1e-10 relative), and a reference per scale factor: SQLite up to SF=1, DuckDB and Postgres both from SF=10.
- SF=10 data and runs. Today's engine answers all 8 queries at SF=10: every answer PASS except Q1, a DRIFT of 2.24e-13 relative from adding decimals as doubles (*measured*).
- Five charts made by script (latency, vectorized, eviction, ceilings, analyst), in light and dark: CSVs in `docs/data/`, images in `docs/images/`.
- New `docs/benchmarks.md` (every config, every table, the method) and `docs/research.md` (the learned components and the checks), and a rewritten README.
- Apache-2.0 license, CI (macOS Debug, macOS Release with the Python tests, Linux Debug), a contributing guide, and issue and pull request templates.
- Removed `plot_results.py` and `docs/benchmarking_results.md`, replaced by the chart scripts and `docs/benchmarks.md`.

### Earlier

Pull requests #1-#21 built the engine from the ground up:

- Storage: raw 8 KB page I/O, slotted pages for variable-length rows, heap files, and a buffer pool with pin counts and pluggable eviction.
- A B+tree index with point lookups and range scans, a catalog with per-column statistics, and a hand-written SQL lexer and recursive-descent parser.
- A cost-based planner with exhaustive join ordering, Volcano-style executors (scans, filter, hash and nested-loop joins, sort, aggregate, projection, limit), automatic indexes, and a hybrid vectorized path.
- The TPC-H loader and 8 queries checked against SQLite, and two learned components, a join-order model and an eviction policy (V1 and V2), kept as measured research and off by default.
