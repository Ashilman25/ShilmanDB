from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from datetime import datetime, timezone
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

# Files under the results folder, then files next to this script.
FROZEN_RESULTS = [
    "questions.jsonl", "datasets/tpch/description.md", "datasets/reviews/description.md",
    "datasets/tpch/manifest.json", "datasets/reviews/manifest.json", "probe.json", "subset_rules.md",
]
FROZEN_CODE = [
    "prompts.py", "prompt_examples.json", "sql_facts.py", "checks.py", "scoring.py",
    "datasets/tpch.json", "datasets/reviews.json", "an_common.py", "describe_datasets.py", "model_mlx.py",
]
# Files next to this script whose changes are shown, never a mismatch.
WATCHED = ["analyze_analyst_check.py", "run_pipeline.py", "bridge_client.py", "analyst_bridge.cpp", "freeze.py"]


# MODEL_CONFIGS from model_mlx.py, read with ast so mlx never has to be installed.
def read_model_configs(path: Path) -> dict:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    node = None
    for stmt in tree.body:  # the last top-level assignment wins, as when Python runs it
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "MODEL_CONFIGS" for t in targets):
            node = stmt.value

    if node is None:
        raise ValueError(f"no top-level MODEL_CONFIGS in {path}")

    try:
        configs = ast.literal_eval(node)
        json.dumps(configs)
    except (ValueError, TypeError) as e:
        raise ValueError(f"MODEL_CONFIGS in {path} isn't a plain JSON-like literal ({e})") from e
    if not isinstance(configs, dict) or not all(isinstance(c, dict) for c in configs.values()):
        raise ValueError(f"MODEL_CONFIGS in {path} isn't a dict of config dicts")
    return configs


# Each frozen name with the path it lives at.
def _frozen_paths(results: Path, code_dir: Path) -> list[tuple[str, Path]]:
    return [(n, Path(results) / n) for n in FROZEN_RESULTS] + [(n, Path(code_dir) / n) for n in FROZEN_CODE]


# One model config as freeze.json stores it, after a JSON round trip.
def _model_record(config_id: str, model_configs: dict) -> dict:
    if config_id not in model_configs:
        raise ValueError(f"unknown model config {config_id!r}; known: {', '.join(model_configs)}")
    return json.loads(json.dumps({**model_configs[config_id], "id": config_id}))


# sha1 of every frozen file, the model config, and each watched file found.
# Raises if any frozen file is missing.
def compute(results: Path, code_dir: Path, config_id: str, model_configs: dict) -> dict:
    model = _model_record(config_id, model_configs)
    named = _frozen_paths(results, code_dir)
    missing = [str(path) for _, path in named if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"can't freeze; missing {', '.join(missing)}")

    code = Path(code_dir)
    watched = {name: an.sha1_file(code / name) for name in WATCHED if (code / name).is_file()}
    return {"files": {name: an.sha1_file(path) for name, path in named}, "model": model, "watched": watched}


# Every test-split answers file under the results folder.
def _test_runs(results: Path) -> list[Path]:
    pattern = an.paths(results).run_file("test", "*")
    return sorted(pattern.parent.glob(pattern.name))


# Write freeze.json into the results folder and return its path.
# Refuses while a test run exists, so a freeze can't be redone under it.
def write(results: Path, code_dir: Path, config_id: str, model_configs: dict) -> Path:
    runs = _test_runs(results)
    if runs:
        listed = ", ".join(map(str, runs))
        raise FileExistsError(f"won't freeze: a test run exists, and a new freeze would no longer match it: {listed}")

    record = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update(compute(results, code_dir, config_id, model_configs))
    path = an.paths(results).freeze
    an.write_text_atomic(path, json.dumps(record, indent=2) + "\n")
    return path


# The keys two model configs disagree on, as "key old -> new".
def _config_diff(old: dict, new: dict) -> str:
    def show(config: dict, key: str) -> str:
        return json.dumps(config[key], sort_keys=True) if key in config else "unset"

    keys = sorted((set(old) | set(new)) - {"id"})
    return ", ".join(f"{k} {show(old, k)} -> {show(new, k)}" for k in keys if show(old, k) != show(new, k))


# freeze.json as a dict, or None and the line saying why it's unusable.
def _load_freeze(results: Path) -> tuple[dict | None, str | None]:
    path = an.paths(results).freeze
    if not path.is_file():
        return None, "missing freeze.json"

    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:  # bad JSON or bad UTF-8
        return None, f"freeze.json doesn't parse: {e}"
    except OSError as e:
        return None, f"can't read freeze.json: {e}"

    if not isinstance(frozen, dict):
        return None, "freeze.json isn't a JSON object"
    return frozen, None


# The model id in a parsed freeze.json, or None if it has no string id.
def _model_id(frozen: dict) -> str | None:
    model = frozen.get("model")
    return model["id"] if isinstance(model, dict) and isinstance(model.get("id"), str) else None


# The frozen model id, or None when freeze.json is missing, broken or has none.
def frozen_model_id(results: Path) -> str | None:
    frozen, _ = _load_freeze(results)
    return None if frozen is None else _model_id(frozen)


# sha1 of freeze.json's bytes, or None when it can't be read.
def freeze_sha1(results: Path) -> str | None:
    try:
        return an.sha1_file(an.paths(results).freeze)
    except OSError:
        return None


# A line per watched file changed or missing; [] when freeze.json is unusable.
def watched_changes(results: Path, code_dir: Path) -> list[str]:
    frozen, _ = _load_freeze(results)
    if frozen is None:
        return []  # verify already says why
    watched = frozen.get("watched")
    if not isinstance(watched, dict):
        return ["freeze.json doesn't record the watched files"]

    lines = []
    for name in WATCHED:
        path = Path(code_dir) / name
        if name not in watched:
            lines.append(f"{name} wasn't recorded in freeze.json")
        elif not path.is_file():
            lines.append(f"{name} is missing")
        elif an.sha1_file(path) != watched[name]:
            lines.append(f"{name} changed")
    return lines


# One line per difference between freeze.json and what's on disk now; [] when clean.
def verify(results: Path, code_dir: Path, model_configs: dict) -> list[str]:
    frozen, problem = _load_freeze(results)
    if frozen is None:
        return [problem]

    problems = []
    files = frozen.get("files")
    if not isinstance(files, dict):
        problems.append("freeze.json has no files")
    else:
        named = _frozen_paths(results, code_dir)
        for name, file in named:
            if name not in files:
                problems.append(f"{name} isn't in freeze.json")
            elif not file.is_file():
                problems.append(f"{name} is missing")
            elif an.sha1_file(file) != files[name]:
                problems.append(f"{name} changed")
        known = {name for name, _ in named}
        problems += [f"{name} is in freeze.json but no longer frozen" for name in files if name not in known]

    model_id = _model_id(frozen)
    if model_id is None:
        problems.append("freeze.json has no model id")
    elif model_id not in model_configs:
        problems.append(f"model {model_id} is no longer in MODEL_CONFIGS")
    elif diff := _config_diff(frozen["model"], _model_record(model_id, model_configs)):
        problems.append(f"model {model_id} changed: {diff}")
    return problems


# Write freeze.json, or check it against today's files; 1 on any refusal or mismatch.
# Watched files are printed as notes and never change the exit code.
def main(argv: list[str] | None = None, code_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze what the test run depends on, or check nothing changed since.")
    sub = ap.add_subparsers(dest="command", required=True)
    write_cmd = sub.add_parser("write", help="hash the frozen files and one model config into freeze.json")
    write_cmd.add_argument("--model", required=True, help="a config id from MODEL_CONFIGS in model_mlx.py")
    write_cmd.add_argument("--results", type=Path, default=an.RESULTS)
    verify_cmd = sub.add_parser("verify", help="print every difference from freeze.json; exit 1 on any")
    verify_cmd.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    code_dir = Path(code_dir or an.HERE)

    try:
        configs = read_model_configs(code_dir / "model_mlx.py")
    except (OSError, SyntaxError, ValueError) as e:
        print(f"can't read model configs: {e}", file=sys.stderr)
        return 1

    if args.command == "write":
        try:
            path = write(args.results, code_dir, args.model, configs)
        except (OSError, ValueError) as e:
            print(f"freeze failed: {e}", file=sys.stderr)
            return 1
        print(f"froze {len(FROZEN_RESULTS) + len(FROZEN_CODE)} files and model {args.model} -> {path}")
        return 0

    freeze_path = an.paths(args.results).freeze
    for line in watched_changes(args.results, code_dir):
        print(f"note (watched, not frozen): {line}")
        
    problems = verify(args.results, code_dir, configs)
    for problem in problems:
        print(problem, file=sys.stderr)

    if problems:
        print(f"{len(problems)} {'difference' if len(problems) == 1 else 'differences'} from {freeze_path}", file=sys.stderr)
        return 1

    print(f"everything matches {freeze_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
