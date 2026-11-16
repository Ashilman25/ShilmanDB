from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from itertools import zip_longest
from pathlib import Path

from app import sixd
from app.datasets import DATASETS, MODEL_ID, AppPaths, default_paths, load_dataset
from app.measured import cache_code_sha1, file_hashes, mlx_versions, tree_sha1
from app.model_cache import CachedModel

an, rp, fz = sixd.an, sixd.rp, sixd.fz

ATTEMPT_FIELDS = ("raw", "kind", "sql")
PROOF = "cache_proof.json"
UNCACHED_CHECK = "uncached_check.json"


# Raised with every reason the replay can't start.
class Refused(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


# The first place two attempt lists differ, as "<label> attempt N <field>"; None when they match.
def _attempt_difference(label: str, saved: list[dict], replayed: list[dict]) -> str | None:
    for i, (a, b) in enumerate(zip_longest(saved, replayed, fillvalue={}), 1):
        for field in ATTEMPT_FIELDS:
            if a.get(field) != b.get(field):
                return f"{label} attempt {i} {field}"
            
    return None


# The first difference between a saved record and its replay, as "<qid> <where> differs"; None when identical.
# Only what the model wrote and how it ended is compared, never times or engine replies.
def compare_record(saved: dict, replayed: dict) -> str | None:
    where = _attempt_difference("primary", saved["primary"]["attempts"], replayed["primary"]["attempts"])
    if where is None and saved["primary"]["final"] != replayed["primary"]["final"]:
        where = "primary final"
        
    for field in ("raw", "text", "final"):
        if where is None and saved["rephrase"][field] != replayed["rephrase"][field]:
            where = f"rephrase {field}"
            
    if where is None:
        where = _attempt_difference("rephrase", saved["rephrase"]["attempts"], replayed["rephrase"]["attempts"])
    return None if where is None else f"{saved['qid']} {where} differs"


# Model seconds plus bridge seconds over some attempts.
def _attempt_seconds(attempts: list[dict]) -> float:
    return sum(a["seconds"] + (a["reply"] or {}).get("ms", 0) / 1000 for a in attempts)


# Median and nearest-rank p90, as the analyst check's report computes them; None for no values.
def _median_p90(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p90": None}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered), 
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1]
    }


# Seconds to the first answer (primary attempts) and to the verdict (plus the rephrase), median and p90.
def timings(records: list[dict]) -> dict:
    first, verdict = [], []
    for rec in records:
        answer = _attempt_seconds(rec["primary"]["attempts"])
        rephrase = rec["rephrase"]
        first.append(answer)
        verdict.append(answer + (rephrase["seconds"] or 0.0) + _attempt_seconds(rephrase["attempts"]))
        
    return {"first_answer": _median_p90(first), "verdict": _median_p90(verdict)}


# An int of at least 1.
def _positive_int(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return n


# The tested model's config; Refused when the freeze isn't clean for it.
def _check_freeze(paths: AppPaths) -> dict:
    try:
        configs = fz.read_model_configs(paths.code / "model_mlx.py")
    except (OSError, SyntaxError, ValueError) as e:
        raise Refused([f"can't read the model settings: {e}"]) from None

    problems = list(fz.verify(paths.results, paths.code, configs))
    frozen = fz.frozen_model_id(paths.results)
    if frozen is not None and frozen != MODEL_ID:
        problems.append(f"freeze.json is for model {frozen}")
    if MODEL_ID not in configs:
        problems.append(f"model {MODEL_ID} isn't in MODEL_CONFIGS")
    if problems:
        raise Refused(problems)
    return configs[MODEL_ID]


# Saved test records per dataset, in dataset order and by qid, the first limit of each.
def _saved_by_dataset(paths: AppPaths, limit: int | None) -> dict[str, list[dict]]:
    run = paths.run_file("test")
    if not run.is_file():
        raise Refused([f"no saved test run at {run}"])
    
    records = an.read_jsonl(run)
    picked = {}
    for ds in DATASETS:
        mine = sorted((r for r in records if r.get("dataset") == ds), key=lambda r: r["qid"])
        if mine[:limit]:
            picked[ds] = mine[:limit]
    if not picked:
        raise Refused([f"{run} holds no test records"])
    return picked


# Each replayed question as run_question takes it, by qid; Refused for a qid the questions file lacks.
def _questions(paths: AppPaths, saved: dict[str, list[dict]]) -> dict[str, dict]:
    path = paths.results / "questions.jsonl"
    texts = {q.get("qid"): q.get("question") for q in rp.qm.read_questions(path)}
    missing = [r["qid"] for records in saved.values() for r in records if not isinstance(texts.get(r["qid"]), str)]
    if missing:
        raise Refused([f"{path} has no question for {', '.join(missing)}"])
    
    return {
        r["qid"]: {
            "qid": r["qid"], 
            "dataset": r["dataset"], 
            "split": r["split"], 
            "question": texts[r["qid"]]
        } for records in saved.values() for r in records
    }


# Each replayed dataset loaded, with its SQLite file; Refused for one that can't be.
def _datasets(paths: AppPaths, names) -> dict:
    loaded, problems = {}, []
    for ds in names:
        try:
            loaded[ds] = load_dataset(paths, ds)
        except Exception as e:
            problems.append(f"{ds}: {e}")
            continue
        sqlite_db = paths.dataset_dir(ds) / f"{ds}.sqlite"
        if not sqlite_db.is_file():
            problems.append(f"{ds}: missing {sqlite_db}")
            
    if problems:
        raise Refused(problems)
    return loaded


# Seconds as the table shows them.
def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


# The timing table: the replay, the saved run on the same questions, the saved run on every question.
def _timing_lines(rows: list[tuple[str, dict]]) -> list[str]:
    lines = [f"{'':<28}{'first answer':<22}verdict", f"{'':<28}{'median':<11}{'p90':<11}{'median':<11}p90"]
    for label, t in rows:
        fa, vd = t["first_answer"], t["verdict"]
        lines.append(f"{label:<28}{_s(fa['median']):<11}{_s(fa['p90']):<11}{_s(vd['median']):<11}{_s(vd['p90'])}")
        
    return lines


# The real bridge on paths.bridge, its stderr appended to a log per dataset in out.
def _default_bridge_factory(paths: AppPaths, out: Path):
    def make(ds: str, tables_tsv: Path):
        out.mkdir(parents=True, exist_ok=True)
        return sixd.bc.Bridge(tables_tsv, command=[str(paths.bridge)], log_path=out / f"replay_bridge_{ds}.log")
    return make


# The prompt cache model, or the plain MlxModel with --uncached-only.
def _default_model_factory(uncached_only: bool):
    if uncached_only:
        return lambda contexts: sixd.load_model_mlx().MlxModel(MODEL_ID)
    return lambda contexts: CachedModel(MODEL_ID, contexts)


# Every question replayed; returns the replayed records and the differences, cached and uncached.
def _replay(paths, saved, questions, loaded, model, bridge_factory, uncached_only) -> tuple[list[dict], list[str], list[str]]:
    replayed, differences, uncached = [], [], []
    for ds, records in saved.items():
        ds_obj = loaded[ds]
        sqlite_db = paths.dataset_dir(ds) / f"{ds}.sqlite"
        
        with bridge_factory(ds, ds_obj.tables_tsv) as bridge:
            for old in records:
                q = questions[old["qid"]]
                start = time.perf_counter()
                rec = rp.run_question(q, model, bridge, sqlite_db, ds_obj.context)
                diff = compare_record(old, rec)
                replayed.append(rec)
                print(f"{q['qid']} {'same' if diff is None else 'differs'} {time.perf_counter() - start:.1f} s", flush=True)
                if diff is None:
                    continue

                differences.append(diff)
                print(f"  {diff}", flush=True)
                if uncached_only:
                    uncached.append(diff)
                    continue
                
                again = compare_record(old, rp.run_question(q, getattr(model, "plain", model), bridge, sqlite_db, ds_obj.context))
                print(f"  uncached rerun: {'same' if again is None else again}", flush=True)
                if again is not None:
                    uncached.append(again)
    return replayed, differences, uncached


# Replay the saved test answers, write the proof, and print the result.
# 0 when every question is identical, 1 when one differs, 2 on a refusal, 130 on Ctrl-C.
def main(argv: list[str] | None = None, paths: AppPaths | None = None, model_factory=None, bridge_factory=None) -> int:
    paths = paths or default_paths()
    ap = argparse.ArgumentParser(description="Replay the tested model's saved test answers and check each reply is the same.")
    ap.add_argument("--limit", type=_positive_int, default=None, help="replay only the first N questions per dataset")
    ap.add_argument("--out", type=Path, default=paths.app_results, help=f"where the proof goes (default {paths.app_results})")
    ap.add_argument("--uncached-only", action="store_true", help="run the plain model with no cache and write uncached_check.json instead")
    args = ap.parse_args(argv)

    try:
        model_config = _check_freeze(paths)
        saved = _saved_by_dataset(paths, args.limit)
        questions = _questions(paths, saved)
        loaded = _datasets(paths, saved)
    except Refused as e:
        print("The replay can't start:")
        for problem in e.problems:
            print(f"  {problem}")
        return 2

    try:
        return _run(args, paths, model_config, saved, questions, loaded, model_factory, bridge_factory)
    except KeyboardInterrupt:
        print("Stopped by Ctrl-C; nothing was written.", file=sys.stderr)
        return 130


# Load the model, replay every question, then write the proof in one go.
def _run(args, paths, model_config, saved, questions, loaded, model_factory, bridge_factory) -> int:
    factory = model_factory or _default_model_factory(args.uncached_only)
    start = time.perf_counter()
    model = factory({ds: loaded[ds].context for ds in saved})
    print(f"Model {MODEL_ID} loaded in {time.perf_counter() - start:.1f} s.", flush=True)
    cached_tokens = dict(getattr(model, "cached_tokens", {}))
    for ds in saved:
        print(f"{ds}: {cached_tokens.get(ds, 0)} cached tokens", flush=True)

    bridges = bridge_factory or _default_bridge_factory(paths, args.out)
    replayed, differences, uncached = _replay(paths, saved, questions, loaded, model, bridges, args.uncached_only)

    total = len(replayed)
    identical = total - len(differences)
    saved_all = an.read_jsonl(paths.run_file("test"))
    times = {
        "replay": timings(replayed), 
        "saved": timings([r for records in saved.values() for r in records]),
        "saved_all": timings(saved_all)
    }
    
    proof = {
        "identical": identical, "total": total, "differences": differences, "uncached_differences": uncached,
        "versions": mlx_versions(), "cache_code_sha1": cache_code_sha1(paths.app_dir), "files": file_hashes(paths),
        "engine_src_sha1": tree_sha1(paths.src), "model": model_config, "cached_tokens": cached_tokens,
        "limit": args.limit, "timings": times,
    }
    path = args.out / (UNCACHED_CHECK if args.uncached_only else PROOF)
    an.write_text_atomic(path, json.dumps(proof, indent=2) + "\n")

    print(f"identical {identical} of {total}")
    for line in _timing_lines([(f"replay ({total})", times["replay"]), (f"saved run, same {total}", times["saved"]), (f"saved run, all {len(saved_all)}", times["saved_all"])]):
        print(line)
    print(f"Wrote {path}")
    return 0 if identical == total else 1


if __name__ == "__main__":
    sys.exit(main())
