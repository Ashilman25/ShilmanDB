# ShilmanDB

A relational database engine built from scratch in C++17, featuring two learned ML components: a neural network join order optimizer and a learned buffer pool eviction policy. Targets analytical workloads (TPC-H) and benchmarks against SQLite.

ShilmanDB is an educational and research project exploring how machine learning can enhance traditional database internals. It is not a production database -- it is a ground-up implementation of every layer of a relational engine, from raw page I/O to SQL parsing, with ML integration at two key decision points.

## Quick Start

```bash
# Clone and build (Debug, no ML)
bash scripts/build.sh

# Run tests
cd build && ctest --output-on-failure && cd ..

# Generate TPC-H data (one-time)
bash bench/generate_tpch_data.sh

# Run a query (via the TPC-H benchmark loader)
./build/bench/load_tpch --sf 0.01 --db-file /tmp/tpch.db \
    --data-dir bench/tpch_data/sf0.01/
```

To use the learned ML components:

```bash
# Build with LibTorch
bash scripts/build.sh Release ON

# Run with both learned components
./build/bench/load_tpch --sf 0.01 --db-file /tmp/tpch.db \
    --data-dir bench/tpch_data/sf0.01/ \
    --use-learned-join --join-model-path ml/join_optimizer/models/join_order_model.pt \
    --use-learned-eviction --eviction-model-path ml/eviction_policy/models/eviction_model.pt
```

## Architecture

```
                          SQL string
                              |
                              v
                     +----------------+
                     |     Lexer      |   Tokenizes input
                     +----------------+
                              |
                              v
                     +----------------+
                     |     Parser     |   Recursive descent -> AST
                     +----------------+
                              |
                              v
                     +----------------+     +-------------------------+
                     |    Planner     |<--->| Learned Join Optimizer  |
                     +----------------+     | (MLP, TorchScript)      |
                              |             +-------------------------+
                              v
                     +----------------+
                     |   Executor     |   Volcano-style iterator tree
                     |  (9 operators) |   SeqScan, HashJoin, Sort, ...
                     +----------------+
                              |
                              v
                     +----------------+     +-------------------------+
                     | Buffer Pool    |<--->| Learned Eviction Policy |
                     |   Manager      |     | (EmbedPool, TorchScript)|
                     +----------------+     +-------------------------+
                              |
                              v
                     +----------------+
                     |  Disk Manager  |   8 KB page I/O
                     +----------------+
                              |
                              v
                         [db file]
```

**Layers (bottom-up):**

| Layer | Directory | Purpose |
|-------|-----------|---------|
| Storage | `src/storage/` | DiskManager (page I/O), SlottedPage (variable-length tuples), TableHeap (heap file) |
| Buffer | `src/buffer/` | BufferPoolManager with pin-count tracking, pluggable eviction policy |
| Index | `src/index/` | B+tree with point lookup and range scan |
| Catalog | `src/catalog/` | Table schemas, indexes, per-column statistics for cost estimation |
| Types | `src/types/` | Value (tagged union), Tuple, type coercion system |
| Parser | `src/parser/` | Hand-written lexer + recursive descent parser producing AST |
| Planner | `src/planner/` | AST to physical plan tree with join ordering (exhaustive or learned) |
| Executor | `src/executor/` | Volcano-style operators: SeqScan, IndexScan, Filter, HashJoin, NestedLoopJoin, Sort, Aggregate, Projection, Limit |
| Engine | `src/engine/` | Database facade: `ExecuteSQL()`, `LoadTable()` |

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## ML Components

### Learned Join Order Optimizer

An MLP that predicts the optimal join order for multi-table queries, replacing exhaustive search over all permutations.

- **Architecture:** 48 -> 256 -> 128 -> 64 -> 6 (Wide MLP, 54K params)
- **Training:** ListMLE ranking loss on 6,561 TPC-H-derived samples with perturbation-based data augmentation
- **Accuracy:** 94.1% exact-match, mean cost regret 0.019, prefix-2 accuracy 96.3%
- **Safety:** 1.5x cost fallback -- if the learned order costs more than 1.5x the exhaustive-search optimum, the heuristic order is used instead

### Learned Buffer Pool Eviction Policy

An embedding-based model that predicts page reuse distances to make smarter eviction decisions than LRU.

- **Architecture:** EmbedPool -- `nn.Embedding(4096, 16)` + frequency features -> MLP -> predicted log(reuse_distance) (66K params)
- **Training:** Zipfian access trace (500K accesses, 4096-page vocabulary)
- **Hit rate:** +3.64% improvement over LRU (Belady optimal ceiling is +7.39%, model captures ~49% of the theoretical gap)
- **Inference:** Single batched forward pass over all eviction candidates

Both models are exported as TorchScript (`.pt`) files and loaded at runtime via LibTorch. All ML code is behind `#ifdef SHILMANDB_HAS_LIBTORCH` -- the engine compiles and runs without ML dependencies.

## Benchmarks

Benchmarked on TPC-H queries Q1, Q3, Q5, Q6 at scale factor 0.01, comparing four ShilmanDB configurations against SQLite (disk and in-memory modes).

| Query | ShilmanDB (LRU) | SQLite (disk) | Ratio |
|-------|-----------------|---------------|-------|
| Q1 (aggregation) | 1,173 ms | 28 ms | ~42x |
| Q3 (3-table join) | 1,432 ms | 5.1 ms | ~281x |
| Q5 (5-table join) | 1,518 ms | 8.8 ms | ~172x |
| Q6 (scan + filter) | 1,149 ms | 4.6 ms | ~250x |

The performance gap is expected. ShilmanDB is an educational row-store with no query optimizer (beyond join ordering), no columnar storage, no vectorized execution, and no page compression. SQLite has decades of optimization. The value is in the architecture and ML integration, not raw throughput.

**Learned component impact at SF=0.01:**
- Learned join optimizer: ~5% improvement on Q5 (5-table join), where join ordering matters most
- Learned eviction: marginal latency impact (small dataset fits mostly in buffer pool)
- All 16/16 correctness verifications PASS (every config x every query matches SQLite results)

For detailed benchmark analysis, see [docs/benchmarking_results.md](docs/benchmarking_results.md).

Charts: `bench/results/query_latency.png`, `bench/results/join_order_accuracy.png`, `bench/results/hit_rate.png`

## Build Instructions

### Prerequisites

- macOS ARM (Apple Silicon) -- primary target
- C++17 compiler (Apple Clang or GCC)
- CMake 3.20+
- Ninja build system
- Python 3.10+ with venv (for ML training and benchmarking scripts)

For ML builds additionally:
- LibTorch 2.11.0 (run `bash scripts/setup_libtorch.sh`)
- libomp (`brew install libomp`)

### Build

```bash
# Debug build, no ML
bash scripts/build.sh

# Debug build, no ML (explicit)
bash scripts/build.sh Debug OFF

# Release build with ML
bash scripts/build.sh Release ON

# Manual CMake (full control)
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSHILMANDB_ENABLE_ML=OFF \
    -DSHILMANDB_BUILD_BENCH=ON
cmake --build build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SHILMANDB_ENABLE_ML` | OFF | Link LibTorch for learned components. Requires `third_party/libtorch/` |
| `SHILMANDB_BUILD_TESTS` | ON | Build GoogleTest suite |
| `SHILMANDB_BUILD_BENCH` | OFF during dev | Build TPC-H benchmark loader |

### LibTorch Setup (for ML builds)

```bash
bash scripts/setup_libtorch.sh
# Also requires:
brew install libomp
sudo ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /opt/llvm-openmp/lib/libomp.dylib
```

## Testing

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run a single test
cd build && ctest --output-on-failure -R disk_manager_test

# Run tests matching a pattern
cd build && ctest --output-on-failure -R buffer

# Run a test binary directly (shows individual test case names)
./build/test/storage/disk_manager_test
```

318 tests across 26 test files. All pass in Debug. Four assert-based tests fail in Release builds (NDEBUG strips `assert()` -- pre-existing, not a regression).

## TPC-H Benchmarking

```bash
# Generate TPC-H data (one-time setup)
bash bench/generate_tpch_data.sh

# Run full benchmark suite (ShilmanDB + SQLite, all configs)
python bench/run_benchmarks.py --sf 0.01 --runs 5

# Generate charts from results
.venv/bin/python bench/plot_results.py
```

## Project Structure

```
ShilmanDB/
  src/
    buffer/         Buffer pool manager, LRU and learned eviction policies
    catalog/        Table metadata, schemas, statistics
    common/         Config constants, exceptions, RID type
    engine/         Database facade (ExecuteSQL, LoadTable)
    executor/       Volcano-style query operators (9 executors)
    index/          B+tree index with point lookup and range scan
    parser/         Hand-written SQL lexer and recursive descent parser
    planner/        Query planner, join order optimizer (heuristic + learned)
    storage/        Disk manager, slotted pages, table heap
    types/          Value (tagged union), Tuple, type system
  test/             26 test files, 318 tests (GoogleTest)
  bench/            TPC-H data generation, benchmark harness, result plotting
  ml/
    join_optimizer/ Training, evaluation, export for join order MLP
    eviction_policy/ Training, evaluation, export for eviction model
  scripts/          Build script, LibTorch setup, result comparison
  docs/             Architecture and benchmark documentation
  _docs/            Product spec and implementation plan
  third_party/      LibTorch (gitignored)
```

## License

MIT
