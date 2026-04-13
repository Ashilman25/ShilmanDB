# ShilmanDB Architecture

## Data Flow

A SQL query follows this path through the engine:

```
  "SELECT l_returnflag, SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag"
      |
      v
  +------------------+
  |      Lexer       |  Tokenizes SQL into SELECT, IDENTIFIER, LPAREN, ...
  |  (src/parser/)   |
  +------------------+
      |
      v
  +------------------+
  |     Parser       |  Recursive descent: tokens -> AST (SelectStatement)
  |  (src/parser/)   |  Hand-written, no parser generator
  +------------------+
      |  SelectStatement with select_list, from_clause, joins,
      |  where_clause, group_by, order_by, limit
      v
  +------------------+     +----------------------------+
  |     Planner      |---->| JoinOrderOptimizer         |
  |  (src/planner/)  |     | Exhaustive search (N<=6)   |
  +------------------+     | or LearnedJoinOptimizer    |
      |                    | (MLP via TorchScript)      |
      |                    +----------------------------+
      |  Physical plan tree: SeqScanPlanNode -> FilterPlanNode
      |  -> HashJoinPlanNode -> AggregatePlanNode -> ...
      v
  +------------------+
  | ExecutorFactory   |  Walks plan tree, constructs executor for each node
  |  (src/executor/) |
  +------------------+
      |
      v
  +------------------+
  |    Executors      |  Volcano model: Init() -> Next(tuple) -> Close()
  |  (src/executor/) |  9 operators, pull-based iteration
  +------------------+
      |  Leaf executors (SeqScan, IndexScan) read from storage
      v
  +------------------+     +----------------------------+
  | BufferPoolManager|---->| EvictionPolicy (interface) |
  |  (src/buffer/)   |     | LRUEvictionPolicy          |
  +------------------+     | LearnedEvictionPolicy       |
      |                    +----------------------------+
      |  FetchPage / NewPage / UnpinPage
      |  Pin-count tracking, dirty-page write-back
      v
  +------------------+
  |   DiskManager    |  Raw 8 KB page reads/writes
  |  (src/storage/)  |  Sequential file I/O
  +------------------+
      |
      v
    [file]
```

## Layer Descriptions

### Storage Layer (`src/storage/`)

The foundation. Manages raw data on disk.

**DiskManager** (`disk_manager.hpp/cpp`)
- Reads and writes fixed 8 KB pages by page ID
- Allocates new pages sequentially
- Single database file per DiskManager instance

**Page** (`page.hpp`)
- In-memory representation of an 8 KB disk page
- Tracks page ID, pin count, dirty flag
- Provides raw `char[PAGE_SIZE]` data buffer

**SlottedPage** (`slotted_page.hpp/cpp`)
- All-static page class operating on raw `char*` buffers
- Variable-length tuple storage within a page
- Slot directory grows from the front, tuple data grows from the back
- Supports insert, delete, and lookup by slot number

**TableHeap** (`table_heap.hpp/cpp`)
- Chains pages into a linked list forming a heap file
- Provides an iterator for full-table scans
- Insert returns a RID (page_id, slot_id) for the new tuple

### Buffer Layer (`src/buffer/`)

Caches disk pages in memory. All upper layers access pages exclusively through the buffer pool.

**BufferPoolManager** (`buffer_pool_manager.hpp/cpp`)
- Fixed-size pool of page frames
- `FetchPage(page_id)` -- returns pinned page from cache or disk
- `NewPage()` -- allocates and pins a new page
- `UnpinPage(page_id, is_dirty)` -- decrements pin count, marks dirty
- `FlushPage(page_id)` -- writes dirty page to disk
- Pin-count tracking prevents eviction of in-use pages
- Dirty flag is a one-way latch (false -> true only)
- Hit/miss counters for benchmark instrumentation
- Access tracing via `EnableTracing(path)` for offline analysis

**EvictionPolicy** (`eviction_policy.hpp`)
- Abstract interface: `RecordAccess(frame_id)`, `Evict()`, `Remove(frame_id)`, `SetEvictable(frame_id, bool)`
- `RecordAccess(frame_id, page_id)` overload for learned policy (default delegates to single-arg)

**LRUEvictionPolicy** (`lru_eviction_policy.hpp/cpp`)
- Standard LRU using a doubly-linked list + hash map
- O(1) access recording, O(1) eviction

**LearnedEvictionPolicy** (`learned_eviction_policy.hpp/cpp`) [ML build only]
- Loads TorchScript model at construction
- Maintains sliding window of recent page accesses (64 entries)
- Bidirectional frame-to-page / page-to-frame mapping
- On `Evict()`: builds (N, 64) tensor of candidate windows, runs batched inference, evicts the page with highest predicted reuse distance
- Falls back to first-evictable frame on inference failure

### Index Layer (`src/index/`)

B+tree index supporting fixed-size key types (INTEGER, BIGINT, DECIMAL, DATE).

**BTreeIndex** (`btree_index.hpp/cpp`)
- `Insert(key, rid)`, `Delete(key, rid)`, `PointLookup(key)`
- `Begin()` / `Begin(low_key)` for full and range scans
- Iterator yields `pair<Value, RID>`
- Leaf pages linked for sequential scan

**BTreeInternalPage / BTreeLeafPage** (`btree_internal_page.hpp/cpp`, `btree_leaf_page.hpp/cpp`)
- All-static page classes operating on raw `char*` buffers
- Binary search within pages
- Page splits on overflow

### Catalog (`src/catalog/`)

Metadata store for tables, schemas, indexes, and column statistics.

**Catalog** (`catalog.hpp/cpp`)
- `CreateTable(name, schema)` -- registers a table and creates its heap file
- `GetTable(name)` -- returns TableInfo with schema, TableHeap pointer, stats
- `CreateIndex(index_name, table_name, column_name)` -- builds B+tree, auto-populates from existing data
- `GetTableIndexes(table_name)` -- returns all indexes for a table
- `UpdateTableStats(table_name)` -- full scan to compute row count and per-column distinct counts
- Stats are used by the planner for join ordering cost estimation

**Schema** (`schema.hpp/cpp`)
- Column definitions: name, type (INTEGER, BIGINT, DECIMAL, VARCHAR, DATE), nullable flag
- Column lookup by name, index resolution

### Types (`src/types/`)

The type system underpinning all data in the engine.

**Value** (`value.hpp/cpp`)
- Tagged union: INTEGER (int32), BIGINT (int64), DECIMAL (double), VARCHAR (string), DATE (int32 days)
- Comparison operators with automatic type coercion (INTEGER -> BIGINT -> DECIMAL)
- Arithmetic operators (+, -, *, /) with promotion rules
- `CastTo(TypeId)` for explicit conversion
- `FromString(TypeId, string)` for parsing from text

**Tuple** (`tuple.hpp/cpp`)
- A row of Values, one per column
- Accessed by column index

### Parser (`src/parser/`)

Converts SQL text into an abstract syntax tree.

**Lexer** (`lexer.hpp/cpp`)
- Tokenizes SQL into keywords, identifiers, literals, operators
- Handles string literals, integer/float literals, date literals

**Parser** (`parser.hpp/cpp`)
- Hand-written recursive descent (no parser generator)
- Produces `SelectStatement` with:
  - `select_list` (expressions, aliases, `*`)
  - `from_clause` (table references)
  - `joins` (INNER/LEFT/RIGHT/CROSS with ON conditions)
  - `where_clause`, `group_by`, `having`, `order_by`, `limit`

**AST** (`ast.hpp`)
- Expression hierarchy: `ColumnRef`, `Literal`, `BinaryOp`, `UnaryOp`, `Aggregate`, `StarExpr`
- All expressions support `Clone()` for deep copying

### Planner (`src/planner/`)

Transforms the AST into a physical execution plan.

**Planner** (`planner.hpp/cpp`)
- Input: `SelectStatement` (moved by value, zero-copy)
- Output: tree of `PlanNode` objects
- Pipeline (bottom-up): SeqScan -> Filter (WHERE pushdown) -> Join tree -> Aggregate -> Filter (HAVING) -> Projection -> Sort -> Limit
- Equi-joins use HashJoin, all others use NestedLoopJoin

**JoinOrderOptimizer** (`join_order_optimizer.hpp/cpp`)
- Exhaustive search over all permutations for N <= 6 tables
- Cost model based on catalog statistics (row counts, selectivity estimates)
- 720 permutations max -- microsecond-fast

**LearnedJoinOptimizer** (`learned_join_optimizer.hpp/cpp`) [ML build only]
- Loads TorchScript model (Wide MLP: 48 -> 256 -> 128 -> 64 -> 6)
- 48-element feature vector: 6 tables x 3 stats + 15 pairs x 2 stats
- Baked standardization (mean/std stored as model buffers)
- Predicts join order, planner compares cost to exhaustive search
- 1.5x safety fallback: if learned cost > 1.5x exhaustive optimum, heuristic wins

### Executor (`src/executor/`)

Volcano-style pull-based query execution. Each operator implements `Init()`, `Next(Tuple*)`, `Close()`.

| Executor | Type | Description |
|----------|------|-------------|
| **SeqScan** | Leaf | Full table scan via TableHeap iterator |
| **IndexScan** | Leaf | B+tree range scan with optional predicate |
| **Filter** | Streaming | Evaluates predicate, passes matching tuples |
| **Projection** | Streaming | Evaluates expression list, produces output columns |
| **Limit** | Streaming | Passes first N tuples, then stops |
| **Sort** | Materializing | Drains child in Init(), sorts in memory, emits in Next() |
| **Aggregate** | Materializing | Groups and accumulates (COUNT, SUM, AVG, MIN, MAX) |
| **HashJoin** | Join | Builds hash table on left child, probes with right |
| **NestedLoopJoin** | Join | Streams left, materializes right, evaluates predicate |

**ExecutorFactory** (`executor_factory.hpp/cpp`)
- Recursively walks PlanNode tree
- Constructs the corresponding executor for each node type

**ExpressionEvaluator** (`expression_evaluator.hpp`)
- `EvaluateExpression(expr, tuple, schema)` -- resolves column refs, evaluates arithmetic/comparison
- `CombineTuples` -- concatenates tuples for join output
- `IsTruthy` / `MakeBool` -- boolean conversion for filter predicates

### Engine (`src/engine/`)

Top-level facade that wires everything together.

**Database** (`database.hpp/cpp`)
- Constructor creates DiskManager, BufferPoolManager, Catalog
- ML constructor additionally creates LearnedJoinOptimizer and/or LearnedEvictionPolicy
- `ExecuteSQL(sql)` -- Parser -> Planner -> ExecutorFactory -> Init/Next/Close -> QueryResult
- `LoadTable(name, schema, tbl_file_path)` -- bulk loads delimited files (e.g., TPC-H `.tbl` files)
- Exception-safe: Close() always called even if Next() throws

## ML Integration Points

There are exactly two points where learned components plug into the engine:

### 1. Join Order Selection (Planner)

```
Planner::Plan()
    |
    +-- Has LearnedJoinOptimizer?
         |
         Yes --> PredictJoinOrder(features)
         |       |
         |       +-- Build 48-element feature vector from catalog stats
         |       +-- Run TorchScript forward pass
         |       +-- Decode output into join order
         |       |
         |       +-- Compare learned cost vs exhaustive-search cost
         |       +-- If learned_cost <= 1.5x * exhaustive_cost: use learned order
         |       +-- Else: fall back to heuristic order
         |
         No ---> JoinOrderOptimizer::FindBestOrder()
                 (exhaustive permutation search)
```

The learned optimizer is a non-owning raw pointer passed per-query from Database to Planner. Database owns the `unique_ptr<LearnedJoinOptimizer>`.

### 2. Page Eviction (Buffer Pool Manager)

```
BufferPoolManager::FetchPage() / NewPage()
    |
    +-- Page in pool?
    |    |
    |    Yes --> RecordAccess(frame_id, page_id)  // update eviction policy
    |    |
    |    No ---> Need free frame?
    |             |
    |             Yes --> eviction_policy_->Evict()
    |             |       |
    |             |       +-- [LRU] Return least recently used evictable frame
    |             |       +-- [Learned] Build (N, 64) tensor of candidate windows
    |             |       |            Run batched inference
    |             |       |            Evict argmax(predicted_reuse_distance)
    |             |       |            Falls back to first-evictable on error
    |             |       |
    |             |       +-- Write back dirty page, load new page
    |             |
    |             No ---> Assign to free frame
```

The eviction policy is injected via `unique_ptr<EvictionPolicy>` into the BPM constructor. BPM owns the policy for its entire lifetime. The rest of the engine is unaware of which policy is in use.

### ML Build Guard

All ML code is conditionally compiled:

```cpp
#ifdef SHILMANDB_HAS_LIBTORCH
// ML-specific includes, classes, and code paths
#endif
```

The `SHILMANDB_HAS_LIBTORCH` macro is set by CMake when `SHILMANDB_ENABLE_ML=ON`. The engine compiles and passes all 318 tests without LibTorch -- ML is strictly additive.

### ML Model Artifacts

```
ml/
  join_optimizer/
    models/
      join_order_model.pt       TorchScript (Wide MLP + baked standardization)
      join_order_model_best.pt  Best state dict checkpoint
      feature_stats.pt          Mean, std, model variant name
    train_join_model.py         Training script (ListMLE loss, cosine LR)
    evaluate_join_model.py      3-metric evaluation (exact, regret, prefix-2)
    export_model.py             TorchScript export with 3 verification checks
    generate_training_data.py   Orchestrates C++ data generator at multiple SFs

  eviction_policy/
    models/
      eviction_model.pt         TorchScript (EmbedPool)
      eviction_model_best.pt    Best state dict checkpoint
      model_config.pt           max_pages, embed_dim, window_size
    train_eviction_model.py     Training script (MSE loss on log-reuse-distance)
    evaluate_eviction_model.py  3-metric evaluation (MSE, ranking, hit rate)
    export_model.py             TorchScript export
    generate_access_trace.py    Generates page access traces for training data
```

Both models are trained in Python (PyTorch), exported to TorchScript, and loaded at runtime by C++ via LibTorch. The Python venv at `.venv/` must have a PyTorch version matching the LibTorch version (currently 2.11.0).
