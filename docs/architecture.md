# ShilmanDB Architecture

ShilmanDB is a relational database engine written from scratch in C++17. This page explains how it is built today: the path a query takes, what each layer does, the vectorized path, the buffer pool and its eviction policies, the planner, and the two learned components. The code under `src/` is the final word; if this page and the code disagree, the code is right.

Numbers on this page are marked by how we know them: *measured* (we ran it on this engine, an Apple M5 Max), *from docs* (copied from project notes), or *inferred* (worked out from other numbers). Constants that are simply set in the code (page size, batch size, the 1.5× fallback) are not marked.

## Data Flow

A query takes this path through the engine:

```
  "SELECT l_returnflag, SUM(l_extendedprice) FROM lineitem GROUP BY l_returnflag"
      |
      v
  +-------------------+
  |  Lexer + Parser   |  Hand-written lexer and recursive descent parser
  |   (src/parser/)   |  -> SelectStatement (AST)
  +-------------------+
      |
      v
  +-------------------+     +------------------------------------+
  |      Planner      |---->| JoinOrderOptimizer (classical)     |
  |  (src/planner/)   |     | LearnedJoinOptimizer (ML, opt-in)  |
  |                   |---->| IndexSelector + CostModel          |
  +-------------------+     | BindPlan (column positions)        |
      |                     +------------------------------------+
      |  physical plan tree: scans -> filters -> joins -> aggregate
      |  -> HAVING filter -> projection -> sort -> limit
      v
  +-------------------+
  |  ExecutorFactory  |  Builds one executor per plan node, in tuple or
  |  (src/executor/)  |  vectorized mode
  +-------------------+
      |
      v
  +-------------------+
  |     Executors     |  Volcano: Init() -> Next(tuple) -> Close()
  |  (src/executor/)  |  Vectorized ones also have NextBatch(chunk)
  +-------------------+
      |  scans read pages
      v
  +-------------------+     +------------------------------------+
  | BufferPoolManager |---->| EvictionPolicy                     |
  |   (src/buffer/)   |     | LRU (default), CLOCK, LRU-2, ARC,  |
  +-------------------+     | LIRS; learned V1 / V2 (ML, opt-in) |
      |                     +------------------------------------+
      v
  +-------------------+
  |    DiskManager    |  8 KB pages, pread / pwrite on one file
  |  (src/storage/)   |
  +-------------------+
      |
      v
    [file]
```

Each layer only depends on the layers below it. The `Database` class in `src/engine/` wires them together.

## Layers at a Glance

| Layer | Folder | Main classes | Job |
|---|---|---|---|
| Storage | `src/storage/` | `DiskManager`, `Page`, `SlottedPage`, `TableHeap` | Pages on disk, rows inside pages, tables as chains of pages |
| Buffer | `src/buffer/` | `BufferPoolManager`, `EvictionPolicy` and its seven policies | Keeps pages in memory, picks which page to drop |
| Index | `src/index/` | `BTreeIndex`, `BTreeInternalPage`, `BTreeLeafPage` | B+tree for point and range lookups |
| Catalog | `src/catalog/` | `Catalog`, `Schema`, `TableStats` | Tables, indexes, column statistics |
| Types | `src/types/` | `Value`, `Tuple` | Values and the row format |
| Parser | `src/parser/` | `Lexer`, `Parser`, AST nodes | SQL text to a syntax tree |
| Planner | `src/planner/` | `Planner`, `JoinOrderOptimizer`, `CostModel`, `IndexSelector`, `BindPlan` | Syntax tree to a physical plan |
| Executor | `src/executor/` | 9 tuple operators, 4 vectorized ones, `DataChunk`, `ExecutorFactory` | Runs the plan |
| Engine | `src/engine/` | `Database` | The front door: `ExecuteSQL`, `LoadTable` |

## Storage (`src/storage/`)

**DiskManager** reads and writes fixed 8 KB pages by page id, with `pread` / `pwrite` on one file. New pages are handed out in order. It also has three tools for benchmarking:

- `GetReadStats()`: how many reads succeeded and how long they took (always on).
- `SetReadPenalty(us)`: makes every read last at least this long, to stand in for slower storage.
- `SetDirectIO(true)`: skips the operating system's file cache (macOS `F_NOCACHE`; throws elsewhere).

**Page** is one 8 KB frame in memory with its page id, pin count and dirty flag.

**SlottedPage** works on a raw page buffer. A slot directory grows from the front and row bytes grow from the back, so rows can have different lengths.

**TableHeap** chains pages into a heap file. `InsertTuple` returns a `RID` (page id, slot). It has two ways to read:

- `Iterator`: one row at a time, used by the tuple-mode scan. It pins and unpins the page on every call, so no pin is held between calls.
- `ReadRows(pos, max_rows, on_row)`: hands each live row's bytes to a callback, one pin per page. The vectorized scan uses this to fill a batch.

## Buffer Pool (`src/buffer/`)

Every layer above storage reaches pages only through the `BufferPoolManager`.

**BufferPoolManager** holds a fixed number of frames (1,024 by default, 8 MB).

- `FetchPage(page_id)` returns a pinned page, from memory or from disk.
- `NewPage(&page_id)` allocates and pins a new page.
- `UnpinPage(page_id, is_dirty)` drops a pin. The dirty flag only goes from clean to dirty: unpinning with `false` never clears it.
- `FlushPage`, `FlushAllPages` write dirty pages out.
- `DropAllPages()` writes out dirty pages, empties the pool and resets the policy, like a restart with an empty cache. It throws if any page is still pinned. Benchmarks call it before each workload run so every policy starts cold.

A page with a pin count above zero is never evicted.

### Eviction policies

`EvictionPolicy` is an abstract interface. The buffer pool tells it about accesses (`RecordAccess`), pins and unpins (`SetEvictable`), and asks it for a victim (`Evict`). `Reset()` makes it forget everything.

| Policy | Class | Idea | Notes |
|---|---|---|---|
| LRU (default) | `LRUEvictionPolicy` | Evict the page used longest ago | A list plus a per-frame array, O(1) per call |
| CLOCK | `ClockEvictionPolicy` | A hand sweeps the frames; a page with its bit set gets a second chance | Cheap approximation of LRU |
| LRU-2 | `Lru2EvictionPolicy` | Evict the page whose second-to-last use is oldest | Kept in a `std::map`, so it costs more time per access |
| ARC | `ArcEvictionPolicy` | Balances "seen once" against "seen again" pages, with ghost lists of recently evicted pages | Needs the incoming page id in `Evict(page)` |
| LIRS | `LirsEvictionPolicy` | Keeps pages with short reuse distance; the rest share a small queue (about 1% of the pool) | Needs at least 2 frames |
| Learned V1 | `LearnedEvictionPolicy` | A model over a window of the last 64 page accesses | ML build only, for comparison |
| Learned V2 | `LearnedEvictionPolicyV2` | A model over 4 features per frame | ML build only |

**Picking a policy.** `EvictionPolicyKind` names them (`LRU`, `CLOCK`, `LRU2`, `ARC`, `LIRS`, `LEARNED_V1`, `LEARNED_V2`). `MakeEvictionPolicy(kind, pool_size, model_path)` builds one; a learned kind needs an ML build and a model path, or it throws `DatabaseException`. `ParseEvictionPolicyKind` turns `"lru"`, `"clock"`, `"lru2"`, `"arc"`, `"lirs"`, `"learned-v1"` or `"learned-v2"` into a kind. The benchmark loader exposes this as `load_tpch --eviction-policy`.

### Access types and the visit window

A scan pins the same page many times in a row (the heap iterator pins its page several times for every row). If every one of those pins counted as a "use", frequency-based policies like LRU-2 and ARC would think every scanned page was hot. So the buffer pool works out an `AccessType` for each fetch and gives it to the classical policies (those whose `UsesAccessTypes()` returns true):

| Type | When | What policies do |
|---|---|---|
| `NEW_USE` | A miss, a new page, or a return to a page after more than `kVisitWindow` (64) page changes | Counts as a real use |
| `TOUCH` | A return to a resident page within 64 page changes | Same visit: refreshes recency but is not a second use |
| `REPEAT` | The same page as the fetch just before | Ignored |

LRU and the learned policies keep the older interface and get every fetch as a plain access.

### Stats: compare misses, not the raw hit rate

The pool counts hits, misses and **visits** (a run of fetches to one page, with the same 64-page-change window). Two numbers are worth comparing:

- **Misses**, from `GetMissCount()`.
- **The page-level hit rate**: `1 - misses / visits`.

**Never compare the fetch-level hit rate** (`hits / (hits + misses)`). Because a page is pinned once per dereference, each page gets about 160 fetches (*measured*), and the fetch-level rate comes out near 0.994 for every configuration (*measured*), so it hides every difference. A single TPC-H query has no page reuse at all: its page hit rate is 0.0000 (*measured*). Repeated workloads (the loader's `stream`, `lookup` and `mixed`) are where policies can differ.

Two more benchmarking tools sit in the pool, both off by default: `SetPolicyTiming(true)` wraps the policy in a `TimedEvictionPolicy` that times every call, and `EnableTracing(path)` writes a `timestamp,page_id` line for every fetch (the timestamp is the fetch number) so a trace can be replayed in a simulator.

A miss served from the OS file cache costs about 0.56 µs (*measured*), so on cached data eviction barely moves wall-clock time. To see what eviction is worth on slower storage, run with a read penalty or direct I/O (above).

## Index (`src/index/`)

**BTreeIndex** is a B+tree over one fixed-size key column (INTEGER, BIGINT, DECIMAL or DATE).

- `Insert(key, rid)`, `Delete(key, rid)`, `PointLookup(key)` (returns every RID for the key).
- `Begin()` and `Begin(low_key)` start full and range scans. The iterator yields `(key, RID)` pairs and walks the linked leaves.
- Duplicate keys can span a leaf split. Reads descend with `LookupLeftmost` (the first separator `>= key`), so a key's first entry is always found; inserts descend with `Lookup`.

`BTreeInternalPage` and `BTreeLeafPage` work on raw page buffers, with binary search inside a page.

## Catalog (`src/catalog/`)

**Catalog** holds tables (`TableInfo`: schema, heap, stats) and indexes (`IndexInfo`). `CreateIndex` builds a B+tree and fills it from the rows already in the table. The catalog lives in memory only.

**TableStats** is what the planner estimates with. `UpdateTableStats` fills it with one full scan:

- `row_count`
- `page_count`: real heap pages
- `distinct_counts`: per column
- `min_max`: smallest and largest value of each INTEGER, BIGINT, DECIMAL and DATE column

`Database::LoadTable` calls `UpdateTableStats` after every load.

**Schema** lists columns (name, type, nullable flag) and finds a column by name.

## Types (`src/types/`)

**Value** is a tagged union: INTEGER (int32), BIGINT (int64), DECIMAL (double), VARCHAR (string), DATE (int32 days). Arithmetic and comparisons promote by rank, INTEGER → BIGINT → DECIMAL (`CommonType`). `CastTo` only widens (narrowing throws); `WidenTo` widens when it can and otherwise returns the value unchanged.

**Tuple** is a row stored as bytes: the fixed-size part first, then each string, reached by an offset from the row start. It is built from values with `Tuple(values, schema)`, and `GetValue(schema, i)` decodes one column. `Tuple::Concat` glues two rows into a join row (left columns, then right) by copying bytes.

## Parser (`src/parser/`)

A hand-written lexer and a recursive descent parser turn a `SELECT` into a `SelectStatement`: the select list (expressions, aliases, `*`), `FROM` tables, `JOIN ... ON` clauses (`INNER` or `LEFT`), `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY` and `LIMIT`.

Expressions are `ColumnRef`, `Literal`, `BinaryOp` (comparisons, `AND`/`OR`, arithmetic, `LIKE`), `UnaryOp` (`NOT`, negation), `Aggregate` (COUNT, SUM, AVG, MIN, MAX), `StarExpr` and `CaseExpression`. `BETWEEN` is rewritten to `>= AND <=` and `IN (...)` to a chain of `=` joined by `OR`. The negated forms work too: `NOT BETWEEN` and `NOT LIKE` wrap the same expression in `NOT`, and `NOT IN (...)` becomes a chain of `<>` joined by `AND`. `DATE 'YYYY-MM-DD'` becomes a date literal. Every node has `Clone()`.

SQL is read-only: tables come in through `Database::LoadTable`, not `CREATE` or `INSERT`.

## Planner (`src/planner/`)

`Planner::Plan` takes the statement by value and moves its expressions into plan nodes. The plan is built bottom up:

```
scan (SeqScan or IndexScan) per table
  -> Filter (that table's own WHERE conditions)
  -> join tree (hash or nested loop joins, with filters for non-equality join conditions)
  -> Aggregate -> Filter (HAVING) -> Projection -> Sort -> Limit
```

### Splitting the WHERE clause

The `WHERE` clause is split at its top-level `AND`s. A condition on one table goes down to that table's scan. A condition on two or more tables waits for the join that brings in its last table.

### Filtered row counts

`CostModel::EstimateSelectivity` guesses what fraction of a table's rows pass its filters:

- `col = value` keeps one value out of the column's distinct values (`1 / distinct`).
- Range conditions on one column (`<`, `>`, `<=`, `>=`, and so `BETWEEN`) are merged into one interval and priced against the column's min and max from `TableStats`.
- `a OR b` counts as `a + b - a × b`.
- Columns are treated as independent, so their fractions multiply.
- Where the stats can't help, fixed guesses apply: 0.33 for a range with no min/max or a column-vs-column comparison, 0.1 for anything else (including `<>` and `LIKE`), and never below 0.001.

`EstimateFilteredRows` is the table's row count times that fraction. These filtered counts feed both the join order search and the scan choice.

### Join ordering

`JoinOrderOptimizer::FindBestOrder` tries every order of the tables (up to 6 tables, at most 720 orders) and keeps the cheapest. Above 6 tables it keeps the order the query wrote.

- **Connected orders only.** Two tables are linked when an `ON` or a two-table `WHERE` condition ties them. Only orders where each new table links to one already joined compete, so the search never picks a cross product the query didn't ask for. If no order is connected, it tries them all. The learned path's check counts only `ON` links, so a query joined only in `WHERE` always gets the classical order.
- **The cost** of an order is the estimated row count after each join, added up. Each join multiplies the rows so far by the new table's filtered rows and by `1 / max(distinct values)` for each pair of columns its `ON` compares. Only `JOIN ... ON` conditions price a join; a join written as a `WHERE` condition links tables for the connected check but is priced as a cross product.

### Building each join

At each join, every pending condition that uses the new table, at least one table already joined, and no table not yet joined goes into that join. They are then split:

- **Equality conditions** (`a.x = b.y`) become the `HashJoin` condition.
- **Everything else** goes into a `Filter` right above the hash join.
- If there is no equality condition, the join is a `NestedLoopJoin` with the conditions as its predicate.

Keep this split. Putting a non-equality condition inside the hash join condition once made TPC-H Q19 drastically slower.

### Scan choice and scan costs

`IndexSelector::SelectScanStrategy` picks a sequential scan or an index scan for each table by comparing costs in microseconds:

```
SeqScanCost   = pages × seq_page_us + rows × scan_row_us
IndexScanCost = tree_height × random_page_us
              + max(1, rows × selectivity) × (random_page_us + index_row_us)
```

The prices are in `ScanCosts` (`planner/cost_model.hpp`) and come from runs on this engine:

| Field | Value | What it is |
|---|---:|---|
| `seq_page_us`, `random_page_us` | 0.6 µs | A page read from the OS file cache (*measured*: about 0.56 µs) |
| | 50 µs | A page read with direct I/O, picked inside the measured range (*measured*: 25–150 µs, very noisy) |
| `scan_row_us` | 0.06 µs | CPU per row a sequential scan reads and filters (*measured*, SF=0.01) |
| `index_row_us` | 0.07 µs | CPU per row an index scan returns (*measured*, SF=0.01) |

`Database::GetScanCosts()` reads the disk manager every time: the page price is the largest of 0.6 µs, 50 µs if direct I/O is on, and the read penalty. So setting a penalty or direct I/O changes the planner's choices with no extra call. `SetScanCosts` overrides it.

The model is simple on purpose: one page read per matching index row, pool hits ignored, the same price for sequential and random reads. Even so, it found that TPC-H Q14 should read `lineitem` through its date index: with indexes, Q14 at SF=1.0 runs in 0.12 s against SQLite's 0.40 s on disk (*measured*).

### Column binding

`BindPlan` runs last, after every planning decision. It walks the plan and stores each column reference's position (`bound_index`), so executors read a column by index instead of searching by name on every row.

- Filter, Aggregate, Projection and Sort bind against their child's output schema. HashJoin and NestedLoopJoin bind against their own (combined) output schema.
- A name binds to the **first** column with that name; the table qualifier is ignored. An unknown name stays unbound and falls back to a name lookup at run time.
- A literal compared to a bound column (`=`, `<>`, `<`, `>`, `<=`, `>=`) is converted once to the column's type when that is a widening (or a date string to a DATE). A conversion that fails leaves the literal alone, so a bad date still fails at run time.
- A `Clone()` of an expression is unbound.

### What the planner reports

`JoinOrderInfo` records the last join decision: the tables, the links between them, the order, who chose it (`JoinOrderSource`: `SINGLE_TABLE`, `CLASSICAL`, `LEARNED`, `LEARNED_REJECTED` or `FORCED`), and the row estimates per table and per join. `Database::ExplainJoinOrder` plans a query without running it and returns this. `QueryOptions::forced_join_order` makes the planner use a given order (it throws if that order would need a cross product).

## Executor (`src/executor/`)

Execution is Volcano-style and pull-based. Every operator has `Init()`, `Next(Tuple*)` and `Close()`, and every `Next` asserts that `Init` ran first.

| Executor | Kind | What it does |
|---|---|---|
| SeqScan | Leaf | Full table scan through the heap iterator |
| IndexScan | Leaf | B+tree point or range scan |
| Filter | Streaming | Passes rows whose predicate is true |
| Projection | Streaming | Computes the output columns |
| Limit | Streaming | Passes the first N rows |
| Sort | Materializing | Reads all input in `Init`, sorts, then emits |
| Aggregate | Materializing | Groups and accumulates COUNT, SUM, AVG, MIN, MAX; groups come out in key order |
| HashJoin | Join | Builds a hash table (keyed by `Value`) on the smaller input, probes with the other; output is left columns then right |
| NestedLoopJoin | Join | Streams the left input, keeps the right in memory |

**Expression evaluation.** One template, `EvaluateExpression(expr, row)`, serves both modes. `TupleRow` reads from a `Tuple`; `ChunkRow` reads from one row of a `DataChunk`. A bound column is read by position.

**Profiling.** With `QueryOptions::analyze`, every executor is wrapped in a `ProfilingExecutor` that counts rows and times `Init`/`Next`/`Close`. The query result then carries per-operator rows and times. It works in tuple mode only and adds its own overhead, so use it for proportions and row counts, never to time a query.

## The Vectorized Path

Vectorized mode moves rows in batches of up to 1,024 instead of one at a time. Pass `ExecutionMode::VECTORIZED` to `ExecuteSQL` to use it.

### DataChunk: row bytes with lazy decode

A `DataChunk` holds one batch. It is filled one of two ways, never both:

- **From rows** (`AppendRow(data, length)`): the row bytes are copied into the chunk and nothing is decoded. The scan fills chunks this way, straight from the page.
- **From values** (`AppendValues`): every column already decoded, no row bytes. Projection and Aggregate output this way.

Columns are decoded only when someone needs them:

- `EnsureDecoded(cols)` decodes the listed columns, for the active rows only.
- `DecodeCell(col, row)` decodes one cell from the row bytes without caching it.

So a filter that reads one column of a ten-column table only ever decodes that column, and only for rows it actually checks.

### The selection vector

A filter does not copy surviving rows. It writes a **selection vector**: the positions of the rows that passed. Later operators read through it. Two filters in a row compose: the second one checks only the rows the first kept and writes a smaller selection, still pointing into the original storage. Within a batch a selection only narrows, so a decoded column stays valid.

### The four vectorized operators

| Operator | Kind | How it works |
|---|---|---|
| `VectorizedSeqScanExecutor` | Streaming | Fills a chunk with `TableHeap::ReadRows`: one pin per page per batch |
| `VectorizedFilterExecutor` | Streaming | Evaluates the predicate cell by cell (with the same short-circuiting as tuple mode) and sets the selection vector; decodes nothing up front |
| `VectorizedProjectionExecutor` | Streaming | Decodes the columns its expressions read, computes the outputs into a fresh chunk |
| `VectorizedAggregateExecutor` | Materializing | Reads all input in `Init`, decodes the columns it needs, groups in a `std::map` (key order, like tuple mode) |

Both Aggregate operators return one row of zero defaults (COUNT 0, SUM 0.0) for an aggregate with no `GROUP BY` over empty input. TPC-H results depend on this, and tests check that both modes agree.

### Two interfaces per operator

Every vectorized operator has both `NextBatch(DataChunk*)` and `Next(Tuple*)`. `Next` drains an internal chunk one row at a time and refills it with `NextBatch` when empty. So a vectorized operator works under a tuple-mode parent, and the database's `ExecuteSQL` loop (which calls `Next` on the root) works in both modes. Going the other way, the base `Executor::NextBatch` fills a chunk by calling `Next` in a loop, so any tuple operator can feed a batch consumer.

### How ExecutorFactory picks

`ExecutorFactory::CreateExecutor(plan, ctx, mode)` walks the plan and builds an executor per node, passing the mode down.

- In `VECTORIZED` mode, SeqScan, Filter, Projection and Aggregate get their vectorized versions.
- IndexScan, HashJoin, NestedLoopJoin, Sort and Limit stay tuple-mode. When one of them sits over a vectorized child, the factory puts a `BatchToTupleAdapter` between them. The adapter hands out rows one at a time through the selection vector (`MaterializeTuple`), without flattening the chunk.
- In `TUPLE` mode every node gets its tuple executor.

Since the chunk started keeping row bytes, vectorized mode is faster than tuple mode with indexes on 6 of the 8 TPC-H queries we run at SF=0.01 (it takes 0.58–0.87 of that time) and is within 7% of it on the other two, Q14 and Q19 (*measured*).

## Engine (`src/engine/`)

**Database** owns the disk manager, buffer pool and catalog, plus the join model if one is loaded.

- `Database(db_file, pool_size)`: LRU eviction, no learned join.
- `Database(db_file, pool_size, EvictionPolicyKind, eviction_model_path, join_model_path)`: any eviction policy, and the join model if `join_model_path` is set (ML build only, else it throws).
- An ML-build constructor takes on/off flags for the learned join and learned eviction, plus `eviction_version` (`"v2"` by default; anything else is V1).
- `ExecuteSQL(sql, mode)` or `ExecuteSQL(sql, QueryOptions)`: parse, plan, build executors, run, and return a `QueryResult` (schema, rows, and a `QueryProfile` with the join decision). `Close()` is called even if `Next()` throws.
- `ExplainJoinOrder(sql, options)`: parse and plan only.
- `LoadTable(name, schema, file, delimiter)`: bulk-loads a delimited file (such as TPC-H `.tbl`) and computes the table's stats.

`QueryOptions` holds the mode, a forced join order, `analyze`, and `use_learned_join` (true by default; false plans with the classical optimizer even when a model is loaded).

## Learned Components

ShilmanDB has two learned components. Both are **off by default** and exist only in an ML build (`-DSHILMANDB_ENABLE_ML=ON`, which links LibTorch and defines `SHILMANDB_HAS_LIBTORCH`; all ML code sits behind that macro). Without it the engine builds and passes its tests with no ML at all.

Both models are trained in Python with PyTorch (`ml/join_optimizer/`, `ml/eviction_policy/`), exported to TorchScript, and loaded in C++ through LibTorch. The PyTorch in `.venv/` and the LibTorch in `third_party/` must be the same version (both 2.11.0). The trained `.pt` files are not checked in; the scripts in `ml/` produce them.

**Measured, both lose or tie against the classical answers.** They are kept as research results, not as the recommended setup.

### Learned join order

`LearnedJoinOptimizer::PredictJoinOrder` runs a small MLP over a 48-number feature vector (6 tables × 3 stats, plus 15 table pairs × 2 stats) from `JoinOrderOptimizer::BuildFeatureVector`. The model was trained on whole-table stats, so it gets those, not the filtered counts.

```
Planner::Plan, with a model loaded
  -> model proposes an order
  -> not connected by ON?      use the classical order
  -> learned cost <= 1.5 x     use the model's order     (source LEARNED)
     classical best cost?
  -> otherwise                 use the classical order   (source LEARNED_REJECTED)
  -> model throws?             use the classical order
```

The classical search always runs too, as the safety check, so a bad prediction can cost at most 1.5× the classical plan's estimated cost. The `Database` owns the model and gives the planner a non-owning pointer per query.

### Learned eviction

Both learned policies plug in through the same `EvictionPolicy` interface, so the buffer pool doesn't know which one it has.

- **V1** (`LearnedEvictionPolicy`, for comparison only) keeps the last 64 page accesses and, on eviction, scores every candidate with a page-embedding model.
- **V2** (`LearnedEvictionPolicyV2`, the default learned one) gives each evictable frame 4 features: log time since its last use, log use count, log time since its second-to-last use, and a dirty flag fixed at 0. It evicts the frame the model says will be reused furthest in the future.

Each falls back to a simple choice if inference fails. V2 spends about 17–183 µs deciding each eviction (pools of 128–2,048 frames), saves no misses on TPC-H, and makes queries 3.5–6.9× slower at SF=0.01 (*measured*).

## Current Limits

The main gaps today (all from the code):

- **No NULLs.** Aggregates over empty input return zero defaults, not NULL.
- **`LEFT JOIN` runs as an inner join, silently.** The parser records it; nothing downstream reads it.
- **No functions** (`extract`, `substring`, casts), no `DISTINCT`, subqueries, `WITH`, set operations or window functions.
- **Table qualifiers are ignored when binding**, so two joined tables with the same column name read the wrong column. TPC-H's prefixed names (`l_`, `o_`, ...) avoid it.
- **A single-table `WHERE` condition that names no column is dropped** (`WHERE 1 = 0` returns every row).
- **Joins written in `WHERE`** link tables for the classical connected check but aren't priced as joins by the cost formula, and the learned path's check counts only `ON` links, so such a query always gets the classical order.
- **Read-only SQL.** No `CREATE`, `INSERT`, `UPDATE` or `DELETE`; tables are bulk-loaded.
- **Single-threaded.** One query at a time, no latches.
- **No crash safety.** No write-ahead log or recovery, and the catalog lives in memory, so reopening a file doesn't bring its tables back.
- **A row bigger than a page** makes `TableHeap::InsertTuple` loop forever (it only refuses rows over 65,535 bytes).
