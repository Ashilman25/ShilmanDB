from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
import time
from pathlib import Path


# Load an_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    spec.loader.exec_module(module)
    return module


an = _load_common()
qm = an.load_sibling("an_questions", "questions.py")

STEPS = ("prepare", "describe", "probe", "key", "generate", "analyze")
DEFAULT_STEPS = "prepare,describe,probe,key"
BRIDGE = an.BRIDGE
BRIDGE_STEPS = ("probe", "key", "generate")
BUILD_BRIDGE = ("cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHILMANDB_ENABLE_ML=ON -DSHILMANDB_BUILD_BENCH=ON && cmake --build build --target analyst_bridge")


# argparse type for --steps: known names, in my fixed order, each once.
def _steps(text: str) -> list[str]:
    given = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in given if s not in STEPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step {', '.join(unknown)}; steps are {', '.join(STEPS)}")
    if not given:
        raise argparse.ArgumentTypeError("no steps given")
    return [s for s in STEPS if s in given]


# argparse type for --limit-per-dataset: a whole number of at least 1.
def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return n


# Parse the command line; paths stay exactly as given.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the analyst check's scripts in order, from the repo root. Relative paths are taken from the repo root.")
    ap.add_argument("--steps", type=_steps, default=DEFAULT_STEPS, metavar="A,B",help=f"comma-separated from {','.join(STEPS)}; they always run in that order (default {DEFAULT_STEPS})")
    ap.add_argument("--split", choices=qm.SPLITS, help="needed by generate; analyze defaults to test")
    ap.add_argument("--model", metavar="ID", help="a config id; needed by generate, passed to analyze only with --split test")
    ap.add_argument("--limit-per-dataset", type=_positive_int, metavar="N", help="generate only: the first N questions per dataset")
    ap.add_argument("--python", default=".venv/bin/python", help="runs every step but generate")
    ap.add_argument("--ai-python", default=".venv-ai/bin/python", help="runs generate")
    ap.add_argument("--results", help="results folder, passed to every script only when given")
    ap.add_argument("--dry-run", action="store_true", help="make the same checks, print the commands, run nothing")
    return ap.parse_args(argv)


# A script next to this one, as a path from the repo root.
def _script(name: str) -> str:
    return (an.HERE / name).relative_to(an.REPO_ROOT).as_posix()


# Each chosen step's command, in my fixed order. Raises ValueError
# when generate is chosen without both --split and --model.
def build_commands(args) -> list[tuple[str, list[str]]]:
    if "generate" in args.steps and not (args.split and args.model):
        raise ValueError("generate needs --split and --model")

    py, ai_py = args.python, args.ai_python
    split = args.split or "test"
    results = ["--results", args.results] if args.results is not None else []
    limit = ["--limit-per-dataset", str(args.limit_per_dataset)] if args.limit_per_dataset is not None else []
    # Dev compares every config, so the analyzer takes --model only on test.
    analyze_model = ["--model", args.model] if args.model and split == "test" else []

    per_step = {
        "prepare": [py, _script("prepare_datasets.py")],
        "describe": [py, _script("describe_datasets.py")],
        "probe": [py, _script("probe_subset.py")],
        "key": [py, _script("answer_key.py"), "check"],
        "generate": [ai_py, _script("run_pipeline.py"), "--split", args.split, "--model", args.model, *limit],
        "analyze": [py, _script("analyze_analyst_check.py"), "--split", split, *analyze_model],
    }
    return [(step, [*per_step[step], *results]) for step in STEPS if step in args.steps]


# The bridge path as I print it: from the repo root when it's inside.
def _shown(path: Path) -> str:
    path = Path(path)
    return path.relative_to(an.REPO_ROOT).as_posix() if path.is_relative_to(an.REPO_ROOT) else str(path)


# Run one command from the repo root.
# Returns its exit code, 128 + signal, or 130 on Ctrl-C.
def _run(step: str, cmd: list[str]) -> int:
    try:
        proc = subprocess.Popen(cmd, cwd=an.REPO_ROOT)
    except OSError as e:
        print(f"== {step}: can't start {cmd[0]}: {e}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print(f"== {step}: stopped by Ctrl-C", file=sys.stderr, flush=True)
        return 130

    try:
        code = proc.wait()
    except KeyboardInterrupt:
        # The step got the Ctrl-C too: I wait, and kill it on a second one.
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
    if code:
        print(f"== {step}: failed with exit code {code}, stopping", file=sys.stderr, flush=True)
    return code


# Check the steps, split, model and bridge, then print and run each step.
# Returns 0, 2 when it won't start, or the first failing step's code.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        commands = build_commands(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if any(s in BRIDGE_STEPS for s in args.steps) and not Path(BRIDGE).is_file():
        print(f"error: {_shown(BRIDGE)} is missing (probe, key and generate use it); build it from the repo root with:", file=sys.stderr)
        print(BUILD_BRIDGE, file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"== dry run, nothing runs; commands run from {an.REPO_ROOT}")
        for step, cmd in commands:
            print(f"== {step}")
            print(shlex.join(cmd))
        return 0

    for step, cmd in commands:
        started = time.monotonic()
        print(f"== {step}: start {time.strftime('%H:%M:%S')}", flush=True)
        print(shlex.join(cmd), flush=True)
        code = _run(step, cmd)
        if code:
            return code
        print(f"== {step}: done in {time.monotonic() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
