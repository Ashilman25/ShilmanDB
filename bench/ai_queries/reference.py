import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .common import paths, write_text_atomic
from .labels import TierLabels, load_tier
from .questions import C2_CLASSES, QUESTIONS
from .queries import load_aliases, normalize_model, question_scopes

PER_STRATUM = 50
PER_CLASS = 17
MIN_ROWS_PER_SEC = 5.0
BAKEOFF_QIDS = ("F3", "F8", "C2")
_QORDER = {qid: i for i, qid in enumerate(QUESTIONS)}


#The form two answers are compared in
def canonical(qid: str, answer: str, aliases: Mapping[str, str] | None = None) -> str:
    kind = QUESTIONS[qid].kind
    if kind == "yesno":
        return answer.strip().lower()
    
    if kind == "extract":
        return normalize_model(answer, aliases or {})
    
    return answer.strip()


def _stratum(qid: str, answer: str, aliases: Mapping[str, str]) -> str:
    if QUESTIONS[qid].kind == "extract":
        return "none" if normalize_model(answer, aliases) == "none" else "some"
    return canonical(qid, answer)


#Per question: 50 big-yes and 50 big-no (C2: 17 per class; E7: 50 none and 50 not)
def reference_sample(big: TierLabels, keys_by_q: Mapping[str, set[str]], seed: int = 11, aliases: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    aliases = aliases or {}
    out: dict[str, list[str]] = {}
    for qid, idx in _QORDER.items():
        if qid not in keys_by_q or qid not in big.answers:
            continue
        
        rng = np.random.default_rng([seed, idx])
        keys = sorted(k for k in keys_by_q[qid] if k in big.answers[qid])
        names = {
            "yesno": ["yes", "no"], 
            "class": list(C2_CLASSES), 
            "extract": ["none", "some"]
        }[QUESTIONS[qid].kind]
        
        want = PER_CLASS if qid == "C2" else PER_STRATUM
        picked: list[str] = []
        
        for name in names:
            pool = [k for k in keys if _stratum(qid, big.answers[qid][k], aliases) == name]
            take = min(want, len(pool))
            picked += [pool[i] for i in rng.choice(len(pool), size=take, replace=False)] if take else []
        out[qid] = picked
    return out


#n distinct review texts (by text key), drawn from the table in its own order
def bakeoff_rows(reviews: pd.DataFrame, n: int = 200, seed: int = 7) -> list[str]:
    keys = list(dict.fromkeys(reviews["text_key"]))
    rng = np.random.default_rng(seed)
    return [keys[i] for i in rng.choice(len(keys), size=min(n, len(keys)), replace=False)]


#  batches for the reference model 


def _batch_name(i: int) -> str:
    return f"batch_{i:03d}.jsonl"


#Write the sample as JSONL batches: questions in QUESTIONS order, keys in sample order
def export_batches(sample: Mapping[str, list[str]], inputs: Mapping[str, str], out_dir: Path, batch_size: int = 50) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for qid, q in QUESTIONS.items():
        for key in sample.get(qid, []):
            lines.append(json.dumps({
                "qid": qid, "text_key": key, "question": q.wording,
                "options": list(q.options) if q.kind != "extract" else [],
                "review": inputs[key]
            }, ensure_ascii=False))

    paths_out = []
    for i, start in enumerate(range(0, len(lines), batch_size), start=1):
        path = out_dir / _batch_name(i)
        path.write_text("\n".join(lines[start:start + batch_size]) + "\n", encoding="utf-8")
        paths_out.append(path)
    return paths_out


def _exported_pairs(batch_dir: Path) -> set[tuple[str, str]]:
    pairs = set()
    for path in sorted(Path(batch_dir).glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                pairs.add((rec["qid"], rec["text_key"]))
    return pairs


#Read answers_*.jsonl (`{"qid", "text_key", "answer"}` per line).
def import_answers(in_dir: Path, batch_dir: Path | None = None, allow_missing: bool = False, check_exported: bool = True) -> pd.DataFrame:
    in_dir = Path(in_dir)
    exported = None
    if check_exported:
        batch_dir = Path(batch_dir) if batch_dir else in_dir.parent / "batches"
        if not batch_dir.is_dir():
            raise ValueError(f"batch folder {batch_dir} does not exist")
        exported = _exported_pairs(batch_dir)
        
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(in_dir.glob("answers_*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue

            def bad(why: str) -> ValueError:
                return ValueError(f"{path}: line {n}: {why}")

            try:
                rec = json.loads(line)
            except ValueError:
                raise bad("not valid JSON") from None
            
            if not isinstance(rec, dict) or not all(isinstance(rec.get(f), str) for f in ("qid", "text_key", "answer")):
                raise bad("needs string fields qid, text_key and answer")
            
            qid, key = rec["qid"], rec["text_key"]
            if qid not in QUESTIONS:
                raise bad(f"unknown qid {qid!r}")
            
            if exported is not None and (qid, key) not in exported:
                raise bad(f"{key} was not exported for {qid}")
            
            if (qid, key) in seen:
                raise bad(f"duplicate answer for {qid} {key}")
            
            q = QUESTIONS[qid]
            answer = canonical(qid, rec["answer"])
            if q.kind != "extract" and answer not in q.options:
                raise bad(f"answer {rec['answer']!r} is not one of {list(q.options)}")
            
            seen.add((qid, key))
            rows.append({
                "qid": qid, 
                "text_key": key, 
                "answer": answer
            })
            
    if exported is not None and not allow_missing and (missing := sorted(exported - seen)):
        raise ValueError(f"{len(missing)} exported answers are missing, e.g. {missing[:3]}")
    return pd.DataFrame(rows, columns=["qid", "text_key", "answer"])


# agreement 


#Per question: (rows that agree, rows compared).
def compare(local: TierLabels, ref: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> dict[str, tuple[int, int]]:
    aliases = aliases or {}
    out = {}
    for qid in QUESTIONS:
        mine = local.answers.get(qid, {})
        sub = ref[ref["qid"] == qid]
        hits = [canonical(qid, mine[k], aliases) == canonical(qid, a, aliases) for k, a in zip(sub["text_key"], sub["answer"]) if k in mine]
        if hits:
            out[qid] = (sum(hits), len(hits))
    return out


#Share of compared rows where the local answer matches (see `compare`)
def agreement(local: TierLabels, ref: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> dict[str, float]:
    return {qid: hit / n for qid, (hit, n) in compare(local, ref, aliases).items()}


#  the blinded spot-check 

_SPOT_COLUMNS = ["qid", "text_key", "question", "review", "answer_a", "answer_b", "a_is", "your_answer"]


def export_spotcheck(local: TierLabels, ref: pd.DataFrame, inputs: Mapping[str, str], out_csv: Path, n: int = 50, seed: int = 13, aliases: Mapping[str, str] | None = None) -> pd.DataFrame:
    aliases = aliases or {}
    rows = []
    for r in ref.itertuples():
        mine = local.answers.get(r.qid, {}).get(r.text_key)
        if mine is not None and canonical(r.qid, mine, aliases) != canonical(r.qid, r.answer, aliases):
            rows.append((_QORDER[r.qid], r.text_key, r.qid, mine, r.answer))
            
    rows.sort(key=lambda t: t[:2])
    rng = np.random.default_rng(seed)
    chosen = [rows[i] for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False)] if rows else []
    flips = rng.random(len(chosen)) < 0.5
    out = []
    
    for (_, key, qid, mine, theirs), flip in zip(chosen, flips):
        a, b, a_is = (theirs, mine, "reference") if flip else (mine, theirs, "local")
        out.append({
            "qid": qid, "text_key": key, "question": QUESTIONS[qid].wording, "review": inputs[key],
            "answer_a": a, "answer_b": b, "a_is": a_is, "your_answer": ""
        })
        
    df = pd.DataFrame(out, columns=_SPOT_COLUMNS)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


#Count who the reader sided with. 
def score_spotcheck(filled_csv: Path, aliases: Mapping[str, str] | None = None) -> dict:
    df = pd.read_csv(filled_csv, dtype=str, keep_default_na=False)
    counts = {"local_right": 0, "reference_right": 0, "neither": 0, "blank": 0}
    
    for r in df.itertuples():
        if not r.your_answer.strip():
            counts["blank"] += 1
            continue

        mine = canonical(r.qid, r.your_answer, aliases)
        a_hit = mine == canonical(r.qid, r.answer_a, aliases)
        b_hit = mine == canonical(r.qid, r.answer_b, aliases)
        if not (a_hit or b_hit):
            counts["neither"] += 1
            continue
        
        for hit, side in ((a_hit, r.a_is), (b_hit, "reference" if r.a_is == "local" else "local")):
            if hit:
                counts[f"{side}_right"] += 1
    return counts


# the bake-off 

#Agreement per question and on average for each model, with the rows compared per question.
def bakeoff_score(candidates: Mapping[str, TierLabels], ref: pd.DataFrame, rows_per_sec: Mapping[str, float], aliases: Mapping[str, str] | None = None) -> pd.DataFrame:
    rows = []
    ref_rows = {qid: int((ref["qid"] == qid).sum()) for qid in BAKEOFF_QIDS}
    
    for model, labels in candidates.items():
        cmp = compare(labels, ref, aliases)
        speed = float(rows_per_sec[model])
        
        got = {qid: cmp.get(qid, (0, 0)) for qid in BAKEOFF_QIDS}
        complete = all(ref_rows[q] > 0 and got[q][1] == ref_rows[q] for q in BAKEOFF_QIDS)
        
        rows.append({
            "model": model,
            **{qid: got[qid][0] / got[qid][1] if got[qid][1] else float("nan") for qid in BAKEOFF_QIDS},
            **{f"rows_{qid}": got[qid][1] for qid in BAKEOFF_QIDS},
            **{f"ref_rows_{qid}": ref_rows[qid] for qid in BAKEOFF_QIDS},
            "mean_agreement": float(np.mean([got[q][0] / got[q][1] for q in BAKEOFF_QIDS]))
            if complete else float("nan"),
            "rows_per_sec": speed, "complete": complete,
            "fast_enough": speed >= MIN_ROWS_PER_SEC, "pick": False
        })
        
    df = pd.DataFrame(rows)
    ok = df[df["complete"] & df["fast_enough"]]
    if not ok.empty:
        best = ok.sort_values(["mean_agreement", "rows_per_sec", "model"], ascending=[False, False, True], kind="stable")
        df.loc[best.index[0], "pick"] = True
    return df


#1000 / the rows-weighted median ms per row, from a tier's label files
def rows_per_second(tier_dir: Path) -> float:
    from .llm import latency_summary
    s = latency_summary(tier_dir)
    ms = float((s["median_ms"] * s["rows"]).sum() / s["rows"].sum())
    return 1000.0 / ms


#  command line 


def _ref(root: str):
    p = paths(Path(root))
    return p, p.reference


def _cmd_export(args: argparse.Namespace) -> None:
    from .data_prep import read_tables
    from .llm import first_inputs
    
    p, ref = _ref(args.root)
    reviews, products = read_tables(p.data)
    
    big = load_tier(p.labels / "big", "big")
    aliases = load_aliases(p.root / "e7_aliases.json")
    sample = reference_sample(big, question_scopes(reviews, products), args.seed, aliases)
    ref.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame([(q, k) for q, ks in sample.items() for k in ks], columns=["qid", "text_key"]).to_csv(ref / "sample.csv", index=False)
    files = export_batches(sample, first_inputs(reviews), ref / "batches")
    print(f"wrote {sum(map(len, sample.values()))} questions in {len(files)} batches to {ref / 'batches'}")


def _cmd_import(args: argparse.Namespace) -> None:
    p, ref = _ref(args.root)
    
    answers = import_answers(ref / "answers", ref / "batches", args.allow_missing)
    answers.to_csv(ref / "reference_answers.csv", index=False)
    big = load_tier(p.labels / "big", "big")
    aliases = load_aliases(p.root / "e7_aliases.json")
    
    cmp = compare(big, answers, aliases)
    out = pd.DataFrame({"qid": list(cmp), "agreement": [h / n for h, n in cmp.values()], "rows": [n for _, n in cmp.values()]})
    out.to_csv(ref / "agreement.csv", index=False)
    print(f"imported {len(answers)} answers; wrote {ref / 'agreement.csv'}")
    print(out.to_string(index=False))


def _cmd_bakeoff_export(args: argparse.Namespace) -> None:
    from .data_prep import read_tables
    from .llm import first_inputs
    
    p, ref = _ref(args.root)
    reviews, _ = read_tables(p.data)
    
    rows = bakeoff_rows(reviews, args.n, args.seed)
    ref.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text_key": rows}).to_csv(ref / "bakeoff_rows.csv", index=False)
    files = export_batches({q: rows for q in BAKEOFF_QIDS}, first_inputs(reviews), ref / "bakeoff" / "batches")
    print(f"wrote {len(rows)} rows to {ref / 'bakeoff_rows.csv'} and {len(files)} batches to {ref / 'bakeoff' / 'batches'}")


def _cmd_bakeoff_score(args: argparse.Namespace) -> None:
    from .llm import slug
    
    p, ref = _ref(args.root)
    answers = import_answers(ref / "bakeoff" / "answers", ref / "bakeoff" / "batches", args.allow_missing)
    candidates, speeds = {}, {}
    
    for model in args.models.split(","):
        tier_dir = p.labels / "bakeoff" / slug(model)
        candidates[model] = load_tier(tier_dir, model)
        speeds[model] = rows_per_second(tier_dir)
        
    df = bakeoff_score(candidates, answers, speeds, load_aliases(p.root / "e7_aliases.json"))
    df.to_csv(ref / "bakeoff_score.csv", index=False)
    print(df.to_string(index=False))
    for m in df.loc[~df["complete"], "model"]:
        print(f"note: {m} did not label every reference row of every bake-off question, so it can't be picked")
        
    picked = df.loc[df["pick"], "model"]
    print(f"pick: {picked.iloc[0] if len(picked) else 'none (no model is fast enough)'}")
    print(f"wrote {ref / 'bakeoff_score.csv'}")


def _cmd_spotcheck_export(args: argparse.Namespace) -> None:
    from .data_prep import read_tables
    from .llm import first_inputs
    
    p, ref = _ref(args.root)
    reviews, _ = read_tables(p.data)
    answers = pd.read_csv(ref / "reference_answers.csv", dtype=str, keep_default_na=False)
    
    big = load_tier(p.labels / "big", "big")
    df = export_spotcheck(big, answers, first_inputs(reviews), ref / "spotcheck.csv", args.n, args.seed, load_aliases(p.root / "e7_aliases.json"))
    print(f"wrote {len(df)} disagreements to {ref / 'spotcheck.csv'}; fill in your_answer")


def _cmd_spotcheck_score(args: argparse.Namespace) -> None:
    p, ref = _ref(args.root)
    counts = score_spotcheck(Path(args.file) if args.file else ref / "spotcheck.csv", load_aliases(p.root / "e7_aliases.json"))
    write_text_atomic(ref / "spotcheck_score.json", json.dumps(counts, indent=2) + "\n")
    print(f"{counts}\nwrote {ref / 'spotcheck_score.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--root", default=str(paths().root))
        s.set_defaults(func=func)
        return s

    add("export", _cmd_export, "write the reference sample and its batch files").add_argument("--seed", type=int, default=11)
    add("import", _cmd_import, "read the reference answers and write agreement.csv").add_argument("--allow-missing", action="store_true")
    b = add("bakeoff-export", _cmd_bakeoff_export, "pick the bake-off rows and write their batches")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--seed", type=int, default=7)
    b = add("bakeoff-score", _cmd_bakeoff_score, "score the bake-off models and print the pick")
    b.add_argument("--models", required=True, help="comma-separated, as given to llm bakeoff")
    b.add_argument("--allow-missing", action="store_true")
    b = add("spotcheck-export", _cmd_spotcheck_export, "write the blinded disagreement sheet")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--seed", type=int, default=13)
    add("spotcheck-score", _cmd_spotcheck_score, "score the filled sheet").add_argument("--file", default=None)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
