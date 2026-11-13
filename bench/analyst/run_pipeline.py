from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
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
pr = an.load_sibling("an_prompts", "prompts.py")
qm = an.load_sibling("an_questions", "questions.py")
fz = an.load_sibling("an_freeze", "freeze.py")
bc = an.load_sibling("an_bridge_client", "bridge_client.py")

to_sqlite = an.to_sqlite
EXAMPLES = an.HERE / "prompt_examples.json"


@dataclass(frozen=True)
class Context:
    description: str      # the dataset's description.md
    rules: str            # subset_rules.md
    examples: list[dict]  # the dataset's prompt examples


# The rephrase part with nothing answered: skipped, or the base for an empty rewrite.
def _skipped() -> dict:
    return {
        "raw": None, "text": None, "seconds": None, "prompt_sha1": None,
        "attempts": [], "final": "skipped", "sql": None, "result": None
    }


# What a retry prompt says went wrong with a failed bridge reply.
def _problem(reply: dict) -> str:
    err = reply.get("error") or {}
    kind = err.get("kind", "request")
    
    if kind == "timeout":
        return pr.TOO_SLOW
    if kind == "crash":
        return pr.CRASHED
    return f"{kind} error: {err.get('message', '')}"


# How the attempts ended; sql, result and claims come only from a shown attempt.
def _outcome(attempts: list[dict], final: str, parsed=None) -> dict:
    shown = final == "shown"
    return {
        "attempts": attempts,
        "final": final,
        "sql": parsed.sql if shown else None,
        "result": attempts[-1]["reply"] if shown else None,
        "claims": {"grain": parsed.grain, "columns": parsed.columns} if shown else None,
        "reason": parsed.reason if final == "cannot" else None,
    }


# Ask for SQL and run it, retrying on a bad reply or an engine error.
# Ends shown, cannot (the model refused) or failed (out of attempts).
def run_attempts(model, base_prompt: str, bridge) -> dict:
    attempts = []
    prompt = base_prompt
    for _ in range(1 + an.MAX_RETRIES):
        raw, seconds = model.generate(prompt)
        parsed = pr.parse_reply(raw)
        attempt = {
            "raw": raw, "kind": parsed.kind, "sql": parsed.sql, "reply": None,
            "seconds": seconds, "prompt_sha1": an.sha1_text(prompt)
        }
        attempts.append(attempt)

        if parsed.kind == "cannot":
            return _outcome(attempts, "cannot", parsed)
        
        if parsed.kind == "sql":
            attempt["reply"] = bridge.query(parsed.sql)
            if attempt["reply"].get("ok") is True:
                return _outcome(attempts, "shown", parsed)
            
            problem = _problem(attempt["reply"])
        else:
            problem = pr.NOT_A_REPLY
        prompt = pr.build_retry_prompt(base_prompt, pr.strip_thinking(raw), problem)
    return _outcome(attempts, "failed")


# Ask for the question in other words, then answer the rewritten question.
def _rephrase(question: str, model, bridge, ctx: Context) -> dict:
    prompt = pr.build_rephrase_prompt(question)
    raw, seconds = model.generate(prompt)
    text = pr.parse_rephrase(raw)
    rec = {
        **_skipped(), 
        "raw": raw, 
        "text": text, 
        "seconds": seconds, 
        "prompt_sha1": an.sha1_text(prompt)
    }
    if text is None:
        return rec

    out = run_attempts(model, pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, text), bridge)
    return {**rec, **{k: out[k] for k in ("attempts", "final", "sql", "result")}}


# Every model second plus every query's bridge time, in seconds.
def _seconds_total(primary: dict, rephrase: dict) -> float:
    attempts = primary["attempts"] + rephrase["attempts"]
    model_s = sum(a["seconds"] for a in attempts) + (rephrase["seconds"] or 0.0)
    bridge_ms = sum(a["reply"].get("ms", 0) for a in attempts if a["reply"] is not None)
    return model_s + bridge_ms / 1000


# One question's record: the answer, the rephrased answer, and SQLite's run of a shown query.
def run_question(q: dict, model, bridge, sqlite_db: Path, ctx: Context) -> dict:
    restarts = bridge.restarts
    base = pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, q["question"])
    primary = run_attempts(model, base, bridge)
    shown = primary["final"] == "shown"
    rephrase = _rephrase(q["question"], model, bridge, ctx) if shown else _skipped()

    rec = {
        "qid": q["qid"], "dataset": q["dataset"], "split": q["split"], "model": model.config_id,
        "prompt_sha1": an.sha1_text(base), "primary": primary, "rephrase": rephrase
    }
    if shown:
        rec["sqlite_cause"] = an.sqlite_query(Path(sqlite_db), to_sqlite(primary["sql"]), row_cap=an.ROW_CAP)
    rec["seconds_total"] = _seconds_total(primary, rephrase)
    rec["bridge_restarts"] = bridge.restarts - restarts
    return rec


# Raise ValueError if a whole line of the run file isn't a JSON object.
# Bytes after the last newline are a cut write, which repair_jsonl drops.
def _check_run_file(path: Path) -> None:
    if not path.is_file():
        return
    
    data = path.read_bytes()
    for n, line in enumerate(data[:data.rfind(b"\n") + 1].split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except ValueError:
            record = None
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{n}: not a JSON object; fix the line or move the run file away")


# Raise if the run can't start: results under ai_queries, a missing input file,
# a split with no questions, or a run file that doesn't read.
def check_inputs(results: Path, split: str, datasets: list[str], config_id: str | None = None, bridge_binary: Path | None = None) -> None:
    an.refuse_ai_root(Path(results))
    p = an.paths(results)
    needed = [p.questions, p.rules, EXAMPLES] + ([Path(bridge_binary)] if bridge_binary is not None else [])
    for ds in datasets:
        d = p.dataset_dir(ds)
        needed += [d / "description.md", d / "tables.tsv", d / f"{ds}.sqlite"]
        
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)}")

    qs = qm.read_questions(p.questions)
    for ds in datasets:
        if not any(q.get("dataset") == ds and q.get("split") == split for q in qs):
            raise ValueError(f"no {split} questions for {ds} in {p.questions}; freeze the answer key first")
        
    if config_id is not None:
        _check_run_file(p.run_file(split, config_id))


# The real bridge on an.BRIDGE, its stderr appended to log_path.
def _logging_bridge_factory(log_path: Path):
    Path(an.refuse_ai_root(log_path)).parent.mkdir(parents=True, exist_ok=True)
    return lambda tables_tsv: bc.Bridge(tables_tsv, command=[str(an.BRIDGE)], log_path=log_path)


# Each dataset's prompt examples from prompt_examples.json.
def _load_examples(path: Path, datasets: list[str]) -> dict[str, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected {{dataset: [examples]}}")
    
    missing = [ds for ds in datasets if not isinstance(data.get(ds), list)]
    if missing:
        raise ValueError(f"{path} has no examples for {', '.join(missing)}")
    return data


# Answer every question still missing from the run file; returns how many were written.
# The first limit_per_dataset questions per dataset are picked before done ones are skipped.
def run(split: str, model, results: Path, datasets: list[str], limit_per_dataset: int | None = None, bridge_factory=None) -> int:
    if split not in qm.SPLITS:
        raise ValueError(f"split must be one of {', '.join(qm.SPLITS)}, got {split!r}")
    
    datasets = [an.check_dataset(ds) for ds in datasets]
    check_inputs(results, split, datasets, model.config_id, an.BRIDGE if bridge_factory is None else None)
    p = an.paths(results)

    out = p.run_file(split, model.config_id)
    an.repair_jsonl(out)
    done = {r.get("qid") for r in an.read_jsonl(out)}
    qs = qm.read_questions(p.questions)
    todo = {}
    
    for ds in datasets:
        picked = sorted((q for q in qs if q.get("dataset") == ds and q.get("split") == split), key=lambda q: q["qid"])
        todo[ds] = [q for q in picked[:limit_per_dataset] if q["qid"] not in done]
        
    if not any(todo.values()):
        return 0

    rules = p.rules.read_text(encoding="utf-8")
    examples = _load_examples(EXAMPLES, datasets)
    factory = _logging_bridge_factory(out.with_suffix(".bridge.log")) if bridge_factory is None else bridge_factory
    written = 0
    for ds, pending in todo.items():
        if not pending:
            continue
        d = p.dataset_dir(ds)
        ctx = Context((d / "description.md").read_text(encoding="utf-8"), rules, examples[ds])

        start = time.perf_counter()
        with factory(d / "tables.tsv") as bridge:
            print(f"{ds} bridge ready in {time.perf_counter() - start:.1f} s", flush=True)
            for q in pending:
                rec = run_question(q, model, bridge, d / f"{ds}.sqlite", ctx)
                an.append_jsonl(out, rec)
                written += 1
                restarts = f" (+{rec['bridge_restarts']} restarts)" if rec["bridge_restarts"] else ""
                print(f"{q['qid']} {rec['primary']['final']} {rec['seconds_total']:.1f} s{restarts}", flush=True)
    return written


# The --datasets list: names split on commas, each checked, repeats dropped.
def _dataset_list(text: str) -> list[str]:
    try:
        names = [an.check_dataset(name.strip()) for name in text.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return list(dict.fromkeys(names))


# An int of at least 1.
def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return n


# Check the model and, for test, the freeze; then load the model and run.
# 2 on a refusal, 1 on missing inputs or a failed run.
def main(argv: list[str] | None = None, code_dir: Path | None = None, bridge_factory=None, model_factory=None) -> int:
    ap = argparse.ArgumentParser(description="Ask a local model for SQL per question and run it on the engine.")
    ap.add_argument("--split", required=True, choices=qm.SPLITS)
    ap.add_argument("--model", required=True, help="a config id from MODEL_CONFIGS in model_mlx.py")
    ap.add_argument("--datasets", type=_dataset_list, default=list(an.DATASETS), help="comma-separated, in run order (default tpch,reviews)")
    ap.add_argument("--limit-per-dataset", type=_positive_int, default=None, help="answer only the first N questions per dataset")
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    code_dir = Path(code_dir or an.HERE)

    try:
        configs = fz.read_model_configs(code_dir / "model_mlx.py")
    except (OSError, SyntaxError, ValueError) as e:
        print(f"can't read model configs: {e}", file=sys.stderr)
        return 1
    if args.model not in configs:
        print(f"unknown model {args.model!r}; known: {', '.join(configs)}", file=sys.stderr)
        return 2

    if args.split == "test":
        problems = fz.verify(args.results, code_dir, configs)
        frozen = fz.frozen_model_id(args.results)
        if frozen is not None and frozen != args.model:
            problems.append(f"freeze.json is for model {frozen}")
        if problems:
            print(f"test generation needs a clean freeze for --model {args.model}: {'; '.join(problems)}", file=sys.stderr)
            return 2

    try:
        check_inputs(args.results, args.split, args.datasets, args.model, an.BRIDGE if bridge_factory is None else None)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    start = time.perf_counter()
    if model_factory is None:
        model_factory = an.load_sibling("an_model_mlx", "model_mlx.py").MlxModel
    model = model_factory(args.model)
    print(f"model {args.model} loaded in {time.perf_counter() - start:.1f} s", flush=True)

    try:
        written = run(args.split, model, args.results, args.datasets, args.limit_per_dataset, bridge_factory)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as e:
        print(f"run failed: {e}", file=sys.stderr)
        return 1
    noun = "record" if written == 1 else "records"
    print(f"wrote {written} {noun} -> {an.paths(args.results).run_file(args.split, args.model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
