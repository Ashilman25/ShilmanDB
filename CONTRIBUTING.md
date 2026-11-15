# Contributing to ShilmanDB

Thanks for looking. ShilmanDB is a relational database engine written from scratch in C++17. This page covers how to build it, how to run the tests, the code conventions, how to measure a speed change, and the gate every change must pass.

## Build

You need CMake, Ninja and a C++17 compiler. The main target is macOS on Apple Silicon.

```bash
# Debug: asserts on, no ML, no bench tools
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSHILMANDB_BUILD_BENCH=OFF
cmake --build build-debug

# Release with the bench tools (use this for any timing)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHILMANDB_BUILD_BENCH=ON
cmake --build build
```

Re-run the configure line only when you add a new `.cpp` file or change a CMake option; otherwise `cmake --build` is enough. Files under `src/planner/` and `src/buffer/` are listed by name in the top-level `CMakeLists.txt`, so a new `.cpp` there goes into that list.

**The learned components are optional and off by default.** `-DSHILMANDB_ENABLE_ML=ON` links LibTorch (run `scripts/setup_libtorch.sh` first). Nothing else needs it.

## Test

The integration tests that check TPC-H answers need data. Generate the small scale factor once:

```bash
bash bench/tpch/generate_tpch_data.sh 0.01
```

Without it those tests skip instead of failing, so a green run without data checks less than it looks.

Run `ctest` from inside a build directory (from the repo root it leaves a stray `Testing/` folder):

```bash
cd build-debug && ctest -j 12 --output-on-failure
cd build && ctest -j 12 --output-on-failure -E NextBeforeInitAsserts
```

The four `NextBeforeInitAsserts` tests check `assert()`, which Release compiles out, so they are excluded there.

`ctest -R <pattern>` matches gtest test names (PascalCase fixtures such as `DiskManagerTest`), not CMake target names.

The Python tests (bench tools and their analysis scripts) run in a virtual environment at `.venv/`. It needs Python 3.12 or newer (the pinned `scipy` needs it):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest bench -q -W error
```

Tests that need a service or package you don't have (Postgres, LibTorch, a trained model) skip.

## Conventions

- **C++17.** No C++20 features: for example, use `find() != end()`, not `contains()`.
- Headers are `.hpp`, sources are `.cpp`, and every header starts with `#pragma once`.
- All code lives in the `shilmandb` namespace.
- Includes use quoted paths relative to `src/`: `#include "common/config.hpp"`.
- ML code is guarded by `#ifdef SHILMANDB_HAS_LIBTORCH`, in headers and sources both.
- **Test temp files carry the process ID** (`"shilmandb_dm_test_" + std::to_string(::getpid()) + ".db"`), so `ctest -j` can run tests side by side. A fixed name collides.
- Comments are short and plain: a small comment above a function, 15 words or less.
- New bench code goes in a topic folder under `bench/` with its own `tests/` folder (see `bench/README.md`).

## Measuring a speed change

- Use the Release build (`build/`).
- 1 warmup and 5 timed runs; report the median.
- Run one benchmark at a time. macOS moves work between performance and efficiency cores, so parallel runs skew each other.
- `load_tpch` times only `ExecuteSQL` (parse, plan and execute), not the data load.
- For the buffer pool, compare misses (or the page-level hit rate). Never the fetch-level `raw_hit_rate`: an iterator fetches the same page many times while it reads it, so that number is close to 1 for every setting and hides real differences.

## The gate

Every change keeps these true, and the pull request template asks for each:

1. `ctest` passes in Debug and in Release.
2. `.venv/bin/python -m pytest bench -q -W error` passes.
3. TPC-H answers match the reference (SQLite up to SF=1). Integers, text and dates match exactly; floating point stays within the existing tolerances.
4. Any latency change is measured as above, with the before and after numbers.
5. Buffer pool misses are unchanged, or the change says why they moved (for example, a planner change that picks a different plan on purpose).

## Reporting issues

Use the issue forms: a bug report asks what you ran, what you expected and what happened; a feature request asks for the problem first.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
