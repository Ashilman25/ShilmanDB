from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


# Load vc_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "vc_common.py"
    loaded = sys.modules.get("vc_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("vc_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_common"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_common()

HERE = Path(__file__).resolve().parent
REPO_ROOT = vc.REPO_ROOT
STEPS = ("prepare", "queries", "embed", "truth", "load", "grid", "graph", "rem", "trajectories", "train", "replay", "analyze")
# Allowed --queries per size; 500K may only go down from its full 50.
QUERY_BOUNDS = {"70k": (1, 100), "500k": (30, 50)}

QUERIES_NOTE = "note: data/query_reviews.csv exists; pass --redo to resample"
EMBED_NOTE = "note: embeddings/query_reviews.npy is newer than data/query_reviews.csv; pass --redo to re-embed"
SLICE_NOTE = "note: embeddings/slice.npy exists; pass --redo to re-embed"


# argparse type for --steps: known names, in my fixed order, each once.
def _steps(text: str) -> list[str]:
    given = [s for s in text.split(",") if s]
    unknown = [s for s in given if s not in STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step {', '.join(unknown)}; steps are {', '.join(STEPS)}")
    if not given:
        raise argparse.ArgumentTypeError("no steps given")
    return [s for s in STEPS if s in given]


# argparse type for --jobs: a whole number of at least 1.
def _jobs(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}") from None

    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


# argparse type for --part2-limit: three whole numbers of at least 1, as A,B,C.
def _limit(text: str) -> str:
    try:
        values = [int(x) for x in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"wants TRAIN,VAL,TEST whole numbers, got {text!r}") from None

    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError(f"wants three whole numbers of at least 1, got {text!r}")
    return ",".join(map(str, values))


# Parse and check the command line; paths come back absolute.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vector search check's scripts in order for one table size.")
    parser.add_argument("--size", choices=vc.SIZES, help="table size; needed unless the only step is analyze")
    parser.add_argument("--steps", type=_steps, default=list(STEPS), help=f"from {','.join(STEPS)} (default: all)")
    parser.add_argument("--queries", type=int, default=None, help="filtered-search queries per group for truth and grid")
    parser.add_argument("--part2-limit", type=_limit, default=None, metavar="TRAIN,VAL,TEST", help="first queries of each review set for the stopping steps")
    parser.add_argument("--jobs", type=_jobs, default=max(1, (os.cpu_count() or 1) - 2), help="processes for rem, trajectories and replay")
    parser.add_argument("--redo", action="store_true", help="redo the grid, the query sample and the embeddings")
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--ai-python", type=Path, default=REPO_ROOT / ".venv-ai" / "bin" / "python")
    parser.add_argument("--results", type=Path, default=vc.RESULTS)
    parser.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="print every command and run nothing")
    args = parser.parse_args(argv)

    if args.size is None and args.steps != ["analyze"]:
        parser.error("--size is required unless --steps is only analyze")
    if args.size is not None and args.queries is not None:
        lo, hi = QUERY_BOUNDS[args.size]
        if not lo <= args.queries <= hi:
            tail = " (it may only go down)" if args.size == "500k" else ""
            parser.error(f"--queries at {args.size} must be {lo}-{hi}{tail}")

    # A venv's python is a link, so I make it absolute without following links.
    args.python = Path(os.path.abspath(args.python))
    args.ai_python = Path(os.path.abspath(args.ai_python))
    args.results, args.ai = args.results.resolve(), args.ai.resolve()
    return args


# True when a exists and was written after b.
def _newer(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.stat().st_mtime_ns > b.stat().st_mtime_ns


# Every (step, command, note) to run, in order; a skipped command has a note instead.
def build_commands(args) -> list[tuple[str, list[str] | None, str | None]]:
    py, ai_py = str(args.python), str(args.ai_python)
    results = ["--results", str(args.results)]
    both = [*results, "--ai", str(args.ai)]
    size = ["--size", args.size] if args.size else []
    queries = ["--queries", str(args.queries)] if args.queries is not None else []
    limit = ["--limit", args.part2_limit] if args.part2_limit else []
    jobs = ["--jobs", str(args.jobs)]
    redo = ["--redo"] if args.redo else []

    prepare = str(HERE / "prepare_vector_data.py")
    embed = str(HERE / "embed_vector_check.py")
    stop = str(HERE / "stopping_check.py")
    p = vc.paths(args.results, args.ai)
    reviews_csv = p.data / "query_reviews.csv"

    embeds = [(None, EMBED_NOTE) if not args.redo and _newer(p.embeddings / "query_reviews.npy", reviews_csv) else ([ai_py, embed, "queries", *both], None)]
    if args.size == "500k":
        embeds.append((None, SLICE_NOTE) if not args.redo and (p.embeddings / "slice.npy").exists() else ([ai_py, embed, "slice", *results], None))

    per_step = {
        "prepare": [([py, prepare, "table", *size, *both], None)],
        "queries": [(None, QUERIES_NOTE) if not args.redo and reviews_csv.exists() else ([py, prepare, "queries", *both], None)],
        "embed": embeds,
        "truth": [([py, str(HERE / "vector_truth.py"), *size, *queries, *both], None)],
        "load": [([py, str(HERE / "pg_vector_load.py"), *size, *both], None)],
        "grid": [([py, str(HERE / "pg_vector_grid.py"), *size, *queries, *redo, *both], None)],
        "graph": [([py, str(HERE / "hnsw_graph.py"), *size, *both], None)],
        "rem": [([py, stop, "rem", *size, *jobs, *limit, *both], None)],
        "trajectories": [([py, stop, "trajectories", *size, *jobs, *limit, *both], None)],
        "train": [([py, stop, "train", *size, *limit, *both], None)],
        "replay": [([py, stop, "replay", *size, *jobs, *limit, *both], None)],
        "analyze": [([py, str(HERE / "analyze_vector_check.py"), *results], None)],
    }
    return [(step, cmd, note) for step in args.steps for cmd, note in per_step[step]]


# True when analyze exited 1 and wrote report.md: warnings, not a crash.
def _warned(step: str, code: int, report: Path) -> bool:
    return step == "analyze" and code == 1 and report.exists()


# Run one command; returns 0, or the exit code to stop with after saying why.
def _run(step: str, cmd: list[str], report: Path) -> int:
    try:
        proc = subprocess.Popen(cmd)
    except OSError as e:
        print(f"== {step}: can't start {cmd[0]}: {e}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print(f"== {step}: stopped by Ctrl-C", file=sys.stderr, flush=True)
        return 130

    try:
        code = proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
        print(f"== {step}: stopped by Ctrl-C", file=sys.stderr, flush=True)
        return 130

    if code < 0:
        print(f"== {step}: killed by signal {-code}, stopping", file=sys.stderr, flush=True)
        return 128 - code
    
    if _warned(step, code, report):
        print("== analyze: report.md has warnings (NO VERDICT); see its first lines", file=sys.stderr, flush=True)
    elif code:
        print(f"== {step}: failed with exit code {code}, stopping", file=sys.stderr, flush=True)
    return code


# How to resume from a failed step; a full rerun would restart the grid.
# Past embed, --redo only deletes the grid's finished cells, so I say drop it.
def resume_line(args, step: str) -> str:
    rest = ",".join(args.steps[args.steps.index(step):])
    if args.redo and STEPS.index(step) > STEPS.index("embed"):
        return f"to resume: rerun with --steps {rest} (and your other flags except --redo, which would delete the grid's finished cells)"
    return f"to resume: rerun with --steps {rest} (and your other flags)"


# After a step stops: analyzer warnings end quietly; any other failure still gets analyze.
# Ctrl-C (130) skips analyze. Returns the exit code to stop with.
def _stopped(args, commands, step: str, code: int, report: Path) -> int:
    if _warned(step, code, report):
        return 1
    if step != "analyze" and code != 130 and "analyze" in args.steps:
        cmd = [*next(c for s, c, _ in commands if s == "analyze"), "--failed-step", f"{args.size}:{step}"]
        print(f"== analyze: start {time.strftime('%H:%M:%S')} (after the {step} step failed)", flush=True)
        print(shlex.join(cmd), flush=True)
        _run("analyze", cmd, report)
    print(resume_line(args, step), file=sys.stderr, flush=True)
    return code


# Print and run each step's commands, stopping at the first failure. Returns the exit code.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)

    if not args.dry_run:
        missing = sorted({cmd[0] for _, cmd, _ in commands if cmd and not Path(cmd[0]).exists()})
        if missing:
            print(f"error: python not found: {', '.join(missing)}", file=sys.stderr)
            return 1

    # An old report would outlive a failed step and look current.
    report = vc.paths(args.results, args.ai).report
    if "analyze" in args.steps and not args.dry_run:
        report.unlink(missing_ok=True)

    for step in args.steps:
        started = time.monotonic()
        print(f"== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
        for _, cmd, note in (row for row in commands if row[0] == step):
            print(note if cmd is None else shlex.join(cmd), flush=True)
            if cmd is not None and not args.dry_run:
                code = _run(step, cmd, report)
                if code:
                    return _stopped(args, commands, step, code, report)

        print(f"== {step}: done in {time.monotonic() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
