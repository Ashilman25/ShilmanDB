# ShilmanDB app

A local app for asking questions of ShilmanDB's built-in data. It has two parts: a SQL runner, and an AI analyst that writes the SQL for a question and then checks the answer with evidence only the engine has. The checks are the ones the analyst check in `bench/analyst/` measured, and the app shows that test's rates only while today's setup is the tested one.

Everything runs on this computer. The server listens on `127.0.0.1` only.

## Run it

From the repo root:

```bash
.venv-ai/bin/python -m app
```

| Flag | What it does |
|---|---|
| `--port N` | Serve on port N, or exit with code 2 if it's taken. Without it the app takes the first free port of 20, from 8765 up (*from docs*: `DEFAULT_PORT` and `PORT_TRIES` in `app/__main__.py`). |
| `--no-browser` | Print the link without opening a browser. |
| `--no-model` | Start without the model: the SQL runner only. |
| `--recorded-model PATH` | Answer from a saved run file instead of the model (see "Working on the page"). Can't be used with `--no-model`. |
| `--recorded-pace` | With `--recorded-model`, each reply takes as long as it did when it was recorded. |
| `--no-cache` | Don't use the prompt cache, even when its gate is open. |

At launch the app prints its link (`ShilmanDB app: http://127.0.0.1:<port>/?token=...`) and opens it in the browser. The token in the link becomes a cookie on the first visit, so open the printed link, not a bare address. Then the two datasets load (each starts an engine bridge), and the model loads on its own thread. The page shows each one as loading, ready, missing or failed, and the SQL runner works as soon as a dataset is ready.

Press Ctrl-C to quit. The server, the workers and the bridges close in that order.

## What it needs

- **The engine bridge**, `build/bench/analyst_bridge`. If it's missing the app says so and prints the build command:
  ```bash
  cmake --build build --target analyst_bridge
  ```
  (`build/` is the Release + bench build in `CLAUDE.md`.)
- **The analyst check's dataset files** under `bench/results/analyst_check/`: each dataset's `tables.tsv`, `.tbl` files, `description.md` and `manifest.json`, plus `subset_rules.md`. They're gitignored. A missing dataset shows as missing, with what to run; `docs/playbooks/ai-analyst-check-playbook.md` has the full steps. In short:
  ```bash
  .venv/bin/python bench/analyst/prepare_datasets.py
  .venv/bin/python bench/analyst/describe_datasets.py
  .venv/bin/python bench/analyst/probe_subset.py   # subset_rules.md
  ```
- **`.venv-ai`** with mlx 0.32.4 and mlx_lm 0.32.2 (*measured* 2026-11-14). Nothing is installed by the app.
- **The model** in the Hugging Face cache (`~/.cache/huggingface/hub`): `mlx-community/gemma-4-26b-a4b-it-4bit` at the revision `bench/analyst/model_mlx.py` pins. It loads offline, so it never downloads.
- **Memory:** about 16 GB free (*inferred* from the peaks). The model peaked at 15.7 GB (*measured* 2026-11-14), and the replay smoke peaked at 15.6 GB RSS (*measured* 2026-11-15), on a 64 GB Mac (*from docs*: `CLAUDE.md`). With `--no-model` or `--recorded-model` the app needs little more than the bridges.

The model loads in about 3.2 s (*measured* 2026-11-14), or 5.0 s with both prompt caches filled (*measured* 2026-11-15).

## Two venvs

- **Run in `.venv-ai`** (Python 3.13.7 with MLX, *measured* 2026-11-14). Only the model needs it, so `--no-model` and `--recorded-model` also run in `.venv`.
- **Test in `.venv`:**
  ```bash
  .venv/bin/python -m pytest app -q -W error
  ```
  No test loads MLX. The end-to-end tests (`app/tests/test_app_end_to_end.py`) use the real bridge and skip when it isn't built.

The page is TypeScript in `app/web/src/`, compiled to `app/static/js/`. The compiled files are committed, so running and testing need no Node. To rebuild after an edit:

```bash
npm ci --prefix app/web
npm run --prefix app/web build
```

## The tested line and the measured lock

`app/measured.json` records what the analyst check tested: the sha1 of every file the answers depend on (the check's code and dataset files, plus the pipeline, bridge client and bridge source the app runs), the sha1 of the engine's `src/` tree, the model settings, and the rates. The rates come from `docs/research.md`; the app reads them from the file and never retypes them (*from docs*: `rates` in `app/measured.json`, e.g. `test_questions` = 300).

At launch the app checks all of this once. The lock breaks, and the status says why, when:

| Reason the status shows | What happened |
|---|---|
| `<file> is missing` / `<file> changed` | A locked file is gone or differs from the tested one. |
| `the model settings changed` | The model's entry in `model_mlx.py` isn't the tested one. |
| `the engine changed since the test` | Anything under `src/` changed. |
| `this machine's MLX no longer reproduces the test's answers` | A replay on this exact MLX found the plain, uncached model gives different answers. |
| `measured.json is missing or can't be read` | No record to check against. |

While the lock holds, each answer carries a line with the test's rates, and the page has a "How the checks were tested" panel. While it's broken, the rates hide: the line says the setup differs from the tested one and why, and the panel gets no rates. The analyst still answers and still runs every check. Only the claim about how often the checks catch mistakes goes away.

## The prompt cache and the replay

Every analyst prompt for a dataset starts with the same long text: the dataset's description, the rules and the examples, before the question. The cache keeps the model's state after that shared start, one per dataset (4,096 tokens for TPC-H, 2,048 for the reviews, *measured* 2026-11-15). It's filled in pieces of the same size, on the same boundaries, as mlx_lm's own generation would fill them (*from docs*: `CHUNK` = 2,048 in `app/prompt_cache.py`), and each question gets a fresh copy, so the model does the same arithmetic either way. That's the argument; the replay is the proof.

**The gate.** The cache is used only with the real model, only without `--no-cache`, and only when `bench/results/app/cache_proof.json` matches this setup. The status shows the reason (300 is `rates.test_questions` in `measured.json`, *from docs*):

| Reason | Meaning |
|---|---|
| `the replay hasn't been run` | No proof file. |
| `the replay found differences` | A replayed answer differed. |
| `the replay didn't cover all 300 test questions` | A partial replay (`--limit`). |
| `mlx changed since the replay` | mlx or mlx_lm versions differ. |
| `the app's cache code changed since the replay` | `model_cache.py` or `prompt_cache.py` changed. |
| `a tested file changed since the replay` | A locked file or the model settings changed. |
| `the engine changed since the replay` | `src/` changed. |
| `the replay matched 300 of 300` | On. |

**The replay** reruns the analyst check's 300 saved test answers (*from docs*: `measured.json`) with the cached model and compares what the model wrote, reply by reply (never times or engine replies). Run it in `.venv-ai`:

```bash
.venv-ai/bin/python -m app.replay_check
```

| Flag | What it does |
|---|---|
| `--limit N` | Replay only the first N questions per dataset. |
| `--out DIR` | Where the proof goes (default `bench/results/app/`). |
| `--uncached-only` | Run the plain model with no cache and write `uncached_check.json` instead. |

Exit codes (*from docs*: `app/replay_check.py`): 0 every question identical, 1 one differs, 2 it can't start (it lists why: a freeze that doesn't verify, a missing test run, questions or dataset file), 130 stopped by Ctrl-C (nothing written). When a cached answer differs it reruns that question uncached, so the proof says whether the cache or MLX itself is to blame. It prints a timing table: the replay, the saved run on the same questions, and the saved run on all of them.

| Run | Result |
|---|---|
| Replay smoke, `--limit 2` | identical 4 of 4 in 16.1 s (*measured* 2026-11-15) |
| Full replay | 15-20 min (*inferred*; not run yet) |

**Timings on TPC-H through the app** (*measured* 2026-11-15):

| | First answer | Verdict |
|---|---:|---:|
| Cache off | 2.2-2.5 s | 4.6-5.5 s |
| Cache on (forced by a scratch proof) | 1.4-1.7 s | 3.0-3.7 s |

Both gave the same verdicts. For scale, the analyst check's saved run on all 300 test questions: first answer 2.51 s median, 3.63 s p90; verdict 5.29 s median, 7.64 s p90 (*measured* 2026-11-14).

## Working on the page

The recorded model answers from a saved run file, so the page can be worked on without MLX or the memory it needs:

```bash
.venv/bin/python -m app --recorded-model bench/results/analyst_check/runs/dev_gemma4-26b.jsonl
```

It finds each reply by the prompt's sha1, so only the run's own questions get answers (the examples on the page come from the same dev questions). `runs/test_gemma4-26b.jsonl` works too. Add `--recorded-pace` to see the stages at real speed. Edit `app/web/src/`, rebuild with `npm run --prefix app/web build`, and reload the page.

## Files

| File | What it holds |
|---|---|
| `__main__.py` | The command: flags, port, startup and shutdown. |
| `server.py` | The HTTP server: token cookie, Host and Origin checks, the JSON API, the long poll. |
| `jobs.py` | The two workers (model and SQL) and each job's snapshot. |
| `engine.py` | The bridges, the model, the lock and the cache gate. |
| `datasets.py` | Paths, loading a dataset, and the hint for missing files. |
| `analyst.py` | The analyst check's pipeline split into a first and a second answer. |
| `sql_runner.py` | Typed SQL: the engine-trap check, then the run. |
| `views.py` | Results ready to draw: rows, chart, plan, meta line, error text. |
| `check_views.py` | Check rows, verdicts, terms, the tested line and panel. |
| `copy_rules.py` | The characters no shown text may hold, and the em dash fix for model text. |
| `measured.py` / `measured.json` | The lock and the cache gate, and the tested record. |
| `model_cache.py` / `prompt_cache.py` | The prompt cache. Only `model_cache.py` imports MLX. |
| `replay_check.py` | The replay and its proof. |
| `recorded_model.py` | Replies from a saved run. |
| `sixd.py` | Loads the analyst check's modules from `bench/analyst/`. |
| `examples.json` | Example questions on the page. |
| `web/` | The page's TypeScript and its build. |
| `static/` | What the server sends: `index.html`, `app.css`, the compiled `js/`. |
| `tests/` | The tests, with a two-table TPC-H fixture and a two-question run file. |

`bench/results/app/` (gitignored) is the app's output: `bridge_<dataset>.log` (the bridges' stderr), `cache_proof.json`, `uncached_check.json`, and `replay_bridge_<dataset>.log`.

## Gotchas

- **A rebuilt dataset description must match the tested one byte for byte.** `description.md`, `manifest.json` and `subset_rules.md` are locked, so any difference breaks the lock and the rates hide.
- **When the columnar core changes `src/`, the rates hide** until the analyst check's test is rerun on the new engine and `measured.json` is updated. That's on purpose: the rates were measured on today's engine.
- **A `--limit` replay with the default `--out` replaces a full proof with a partial one,** so the gate turns the cache off. That's safe, but rerun the full replay to turn it back on. Use `--out` somewhere else for a quick check.
- **The model must load and answer on one thread** (MLX's generation stream is per thread). The model worker does both, so don't call the model from anywhere else.
- **The token cookie is named for the port,** so two launches on two ports don't clash.
- **The engine reports an unknown column as a run error, not a plan error** (`Run error: Column not found: nope`), since the planner leaves names it can't bind for the executor.
