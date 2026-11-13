# bench/

Benchmarks and experiment tools. Code is grouped by topic; each topic keeps its Python tests in its own `tests/` folder.

```
bench/
├── CMakeLists.txt     builds every tool into build/bench/<name>
├── tpch/              TPC-H benchmarks against SQLite
│   ├── load_tpch.cpp            the loader and benchmark binary
│   ├── tpch_workload.hpp        schemas, indexes, the 8 queries, SameResult
│   ├── tpch_streams.hpp         stream / lookup / mixed workload generators
│   ├── page_owners.hpp          page → table or index map (load_tpch --dump-page-owners)
│   ├── generate_tpch_data.sh    builds dbgen and writes bench/tpch_data/
│   ├── run_benchmarks.py        the main runner (SQLite + every config)
│   ├── run_sqlite_benchmarks.py SQLite timings and the answer key
│   ├── check_stream_results.py  checks workload answers against SQLite
│   ├── plot_results.py          charts from a results folder
│   ├── run_shilmandb_benchmarks.py  the older runner
│   └── tests/
├── join/              join-order experiments
│   ├── join_headroom.cpp        times every connected join order
│   ├── analyze_join_headroom.py
│   ├── generate_join_training_data.cpp
│   └── tests/
├── eviction/          eviction experiments
│   ├── sim/           trace simulator (C++): eviction_sim, eviction_samples, eviction_switch
│   ├── headroom/      classical policies vs Belady on recorded traces
│   ├── engine/        the classical policies run inside the engine
│   ├── v3_check/      small learned policies on sampled candidates
│   ├── switch_check/  switching between policies as the workload changes
│   └── tests/
├── analyst/           AI-analyst check: local model writes SQL, the engine checks the answer (C++ bridge + Python)
│   ├── an_common.py             shared paths, fixed values, crash-safe JSON-lines files, to_sqlite
│   ├── scoring.py               does a shown result match the answer key
│   ├── analyst_bridge.cpp       loads a dataset once, answers SQL lines with rows and profile as JSON
│   ├── bridge_client.py         talks to the bridge: 30 s timeout, restart on timeout or crash
│   ├── prepare_datasets.py      TPC-H SF=0.1 and the reviews as .tbl + tables.tsv + manifest + SQLite copy
│   ├── datasets/                tpch.json, reviews.json: column sentences, keys, grains, glossary
│   ├── describe_datasets.py     checks those against the data, writes each description.md
│   ├── probe_subset.py          runs 41 SQL shapes on the bridge and SQLite, writes subset_rules.md
│   ├── questions.py             the question file: validation and the dev / test split
│   ├── answer_key.py            checks the key (writers, engines, empty, ties) and freezes it
│   ├── prompts.py               the model's prompts and the reply parser
│   ├── prompt_examples.json     3 worked examples per dataset for the prompt
│   ├── model_mlx.py             the five model configs and the MLX wrapper (.venv-ai)
│   ├── freeze.py                hashes the frozen inputs before the test run; verify names changes
│   ├── run_pipeline.py          runs one model on one split, one resumable record per question (.venv-ai)
│   └── tests/                   (fake_bridge.py stands in for the bridge in client tests)
└── vector_check/      vector search check: pgvector strategy grid and learned stopping (Python)
    ├── vc_common.py             shared paths, text rules, distances, top-k, recall@10
    ├── questions.csv            the 210 written questions used as queries
    ├── prepare_vector_data.py   the 70K / 500K tables and the 10,000 query reviews
    ├── embed_vector_check.py    embeds queries and the 500K slice (.venv-ai), drift check
    ├── vector_filters.py        the 15 filters per query (SQL and mask) and the part 1 queries
    ├── vector_truth.py          exact top-10 truth for every query × filter and unfiltered
    ├── pg_vector_load.py        loads a table into the local Postgres, builds its B-trees and HNSW index
    ├── pg_vector_grid.py        the 33 pgvector options, plan reading, and the timed strategy grid
    ├── hnsw_search.py           numpy HNSW search over a saved graph, counting distance computations
    ├── hnsw_graph.py            builds the HNSW graph with faiss and saves it (own process)
    ├── stopping_check.py        REM, the logged plain search, the stopping model and its replay
    ├── analyze_vector_check.py  applies the fixed bars (500K only) and writes report.md, cells.csv, charts
    ├── run_vector_check.py      the driver: runs every step for one size, prints each command
    └── tests/                   (fixtures/ holds EXPLAIN plans captured from the live server)
```

`results/` and `tpch_data/` hold local data (gitignored) and stay at the top of `bench/`.

## Conventions

- C++ includes are relative to `bench/`, like `src/`: `#include "tpch/tpch_workload.hpp"`.
- Binaries always land in `build/bench/`, whatever folder their source is in.
- Scripts run from any directory: `.venv/bin/python bench/tpch/run_benchmarks.py --help`.
- One script loads another by path, so a moved script needs those paths fixed.
- Test files have unique names (pytest imports them by file name).
- All Python tests: `.venv/bin/python -m pytest bench -q -W error`.

## Moved on 2026-11-11

Only folders changed; no file was renamed.

| Old | New |
|---|---|
| `bench/load_tpch.cpp`, `tpch_workload.hpp`, `tpch_streams.hpp`, `page_owners.hpp`, `generate_tpch_data.sh` | `bench/tpch/` |
| `bench/run_benchmarks.py`, `run_sqlite_benchmarks.py`, `check_stream_results.py`, `plot_results.py`, `run_shilmandb_benchmarks.py` | `bench/tpch/` |
| `bench/join_headroom.cpp`, `generate_join_training_data.cpp`, `bench/headroom/analyze_join_headroom.py` | `bench/join/` |
| `bench/eviction_*.{hpp,cpp}` | `bench/eviction/sim/` |
| `bench/headroom/{run,analyze}_eviction_headroom.py`, `export_v2_weights.py` | `bench/eviction/headroom/` |
| `bench/headroom/{run,analyze}_engine_eviction.py` | `bench/eviction/engine/` |
| `bench/headroom/{run,train,analyze}_v3_check.py` | `bench/eviction/v3_check/` |
| `bench/headroom/{run,analyze}_switch_check.py` | `bench/eviction/switch_check/` |
| `bench/test_*.py`, `bench/headroom/test_*.py` | the `tests/` folder of each topic |
