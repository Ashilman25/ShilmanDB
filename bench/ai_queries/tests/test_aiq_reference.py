import json

import pandas as pd
import pytest

from bench.ai_queries import reference as R
from bench.ai_queries.labels import TierLabels, append_rows, label_columns
from bench.ai_queries.llm import slug
from bench.ai_queries.questions import QUESTIONS
from bench.ai_queries.tests.aiq_world import make_world
from bench.ai_queries.questions import C2_CLASSES


def labels(answers):
    return TierLabels("big", answers, {})


def test_reference_sample_strata():
    keys = [f"k{i:04d}" for i in range(400)]
    answers = {
        "F3": {k: "yes" if i % 2 else "no" for i, k in enumerate(keys)},
        "F8": {k: "yes" if i < 12 else "no" for i, k in enumerate(keys)},
        "C2": {k: C2_CLASSES[i % 6] for i, k in enumerate(keys)},
        "E7": {k: "none" if i % 2 else "iPhone 13." for i, k in enumerate(keys)},
    }
    scopes = {q: set(keys) for q in answers}
    s = R.reference_sample(labels(answers), scopes)
    assert len(s["F3"]) == 100 and sum(answers["F3"][k] == "yes" for k in s["F3"]) == 50
    assert len(s["F8"]) == 62 and sum(answers["F8"][k] == "yes" for k in s["F8"]) == 12
    assert len(s["C2"]) == 102
    assert all(sum(answers["C2"][k] == c for k in s["C2"]) == 17 for c in C2_CLASSES)
    assert sum(answers["E7"][k] == "none" for k in s["E7"]) == 50 and len(s["E7"]) == 100
    assert s == R.reference_sample(labels(answers), scopes)


def sample_and_inputs():
    keys = [f"k{i}" for i in range(6)]
    sample = {"F3": keys[:3], "C2": keys[3:], "E7": keys[:2]}
    return sample, {k: f"Title: t\nReview: {k}" for k in keys}


def test_export_import_roundtrip(tmp_path):
    sample, inputs = sample_and_inputs()
    files = R.export_batches(sample, inputs, tmp_path / "batches", batch_size=4)
    assert [f.name for f in files] == ["batch_001.jsonl", "batch_002.jsonl"]
    recs = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert [r["qid"] for r in recs] == ["C2"] * 3 + ["F3"] * 3 + ["E7"] * 2
    answers = tmp_path / "answers"
    answers.mkdir()
    for f, name in zip(files, ["answers_001.jsonl", "answers_002.jsonl"]):
        lines = []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            ans = {"yesno": "Yes ", "class": R.C2_CLASSES[0], "extract": "iPhone 13"}[R.QUESTIONS[r["qid"]].kind]
            lines.append(json.dumps({"qid": r["qid"], "text_key": r["text_key"], "answer": ans}))
        (answers / name).write_text("\n".join(lines) + "\n")
    df = R.import_answers(answers)
    assert len(df) == 8 and set(df[df.qid == "F3"].answer) == {"yes"}
    bad = (answers / "answers_002.jsonl")
    lines = bad.read_text().splitlines()
    lines[1] = json.dumps({"qid": "F3", "text_key": "k0", "answer": "maybe"})
    bad.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"answers_002\.jsonl: line 2"):
        R.import_answers(answers)


def test_import_rejects_unexported_and_missing(tmp_path):
    sample, inputs = sample_and_inputs()
    R.export_batches(sample, inputs, tmp_path / "batches")
    ans = tmp_path / "answers"
    ans.mkdir()
    (ans / "answers_001.jsonl").write_text(json.dumps({"qid": "F3", "text_key": "zzz", "answer": "yes"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        R.import_answers(ans)
    (ans / "answers_001.jsonl").write_text(json.dumps({"qid": "F3", "text_key": "k0", "answer": "yes"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        R.import_answers(ans)
    assert len(R.import_answers(ans, allow_missing=True)) == 1


def test_agreement():
    keys = [f"k{i}" for i in range(10)]
    local = labels({"F3": {k: "yes" for k in keys}, "E7": {"a": "iPhone 13."}})
    ref = pd.DataFrame({"qid": ["F3"] * 10 + ["E7"], "text_key": keys + ["a"],
                        "answer": ["yes"] * 9 + ["no"] + ["iphone 13"]})
    ag = R.agreement(local, ref, {})
    assert ag["F3"] == 0.9 and ag["E7"] == 1.0


def test_spotcheck_blinds_and_scores(tmp_path):
    keys = [f"k{i:02d}" for i in range(30)]
    local = labels({"F3": {k: "yes" for k in keys}})
    ref = pd.DataFrame({"qid": "F3", "text_key": keys, "answer": "no"})
    inputs = {k: "review " + k for k in keys}
    out = tmp_path / "s.csv"
    df = R.export_spotcheck(local, ref, inputs, out, n=20)
    assert len(df) == 20 and set(df.a_is) == {"local", "reference"}
    assert df.equals(R.export_spotcheck(local, ref, inputs, tmp_path / "t.csv", n=20))
    filled = pd.read_csv(out, dtype=str, keep_default_na=False)
    for i in range(20):
        a = filled.loc[i, "answer_a"]
        filled.loc[i, "your_answer"] = (
            a if i < 8 else filled.loc[i, "answer_b"] if i < 12 else "maybe" if i < 14 else "")
    filled.to_csv(out, index=False)
    score = R.score_spotcheck(out)
    assert score["blank"] == 6 and score["neither"] == 2
    assert score["local_right"] + score["reference_right"] == 12


def test_bakeoff_pick():
    keys = [f"k{i}" for i in range(10)]
    ref = pd.concat([pd.DataFrame({"qid": q, "text_key": keys, "answer": "yes"}) for q in ("F3", "F8")]
                    + [pd.DataFrame({"qid": "C2", "text_key": keys, "answer": R.C2_CLASSES[0]})])

    def cand(n_right):
        return labels({"F3": {k: "yes" if i < n_right else "no" for i, k in enumerate(keys)},
                       "F8": {k: "yes" if i < n_right else "no" for i, k in enumerate(keys)},
                       "C2": {k: R.C2_CLASSES[0 if i < n_right else 1] for i, k in enumerate(keys)}})

    df = R.bakeoff_score({"slow": cand(10), "fast": cand(8), "worse": cand(5)},
                         ref, {"slow": 2.0, "fast": 6.0, "worse": 9.0})
    assert df.set_index("model")["pick"].to_dict() == {"slow": False, "fast": True, "worse": False}
    assert df.set_index("model")["fast_enough"].to_dict() == {"slow": False, "fast": True, "worse": True}


def test_bakeoff_incomplete_model_is_never_picked():
    keys = [f"k{i}" for i in range(10)]
    ref = pd.concat([pd.DataFrame({"qid": q, "text_key": keys, "answer": "yes"}) for q in ("F3", "F8")]
                    + [pd.DataFrame({"qid": "C2", "text_key": keys, "answer": R.C2_CLASSES[0]})])
    full = labels({"F3": {k: "no" for k in keys}, "F8": {k: "no" for k in keys},
                   "C2": {k: R.C2_CLASSES[1] for k in keys}})
    skipped_rows = labels({"F3": {k: "yes" for k in keys[:5]}, "F8": {k: "yes" for k in keys},
                           "C2": {k: R.C2_CLASSES[0] for k in keys}})
    no_c2 = labels({"F3": {k: "yes" for k in keys}, "F8": {k: "yes" for k in keys}})
    df = R.bakeoff_score({"full": full, "skipped": skipped_rows, "noc2": no_c2}, ref,
                         {"full": 9.0, "skipped": 9.0, "noc2": 9.0}).set_index("model")
    assert df["complete"].to_dict() == {"full": True, "skipped": False, "noc2": False}
    assert df["pick"].to_dict() == {"full": True, "skipped": False, "noc2": False}
    assert df.loc["skipped", "rows_F3"] == 5 and df.loc["noc2", "rows_C2"] == 0


def test_bakeoff_tie_goes_to_speed_then_name():
    keys = [f"k{i}" for i in range(4)]
    ref = pd.concat([pd.DataFrame({"qid": q, "text_key": keys, "answer": "yes"}) for q in ("F3", "F8")]
                    + [pd.DataFrame({"qid": "C2", "text_key": keys, "answer": R.C2_CLASSES[0]})])
    same = labels({"F3": {k: "yes" for k in keys}, "F8": {k: "yes" for k in keys},
                   "C2": {k: R.C2_CLASSES[0] for k in keys}})
    df = R.bakeoff_score({"b": same, "a": same, "c": same}, ref, {"a": 7.0, "b": 7.0, "c": 6.0}).set_index("model")
    assert df["pick"].to_dict() == {"a": True, "b": False, "c": False}


def test_import_needs_batch_folder_unless_told_otherwise(tmp_path):
    ans = tmp_path / "answers"
    ans.mkdir()
    (ans / "answers_001.jsonl").write_text(json.dumps({"qid": "F3", "text_key": "k0", "answer": "yes"}) + "\n")
    with pytest.raises(ValueError, match="does not exist"):
        R.import_answers(ans)
    assert len(R.import_answers(ans, check_exported=False)) == 1


def _write_tier(tier_dir, tier, with_probs):
    for qid, q in QUESTIONS.items():
        cols = label_columns(q)
        rows = []
        for k, a in tier.answers[qid].items():
            row = {"text_key": k, "answer": a, "in_tokens": 10, "out_tokens": 1, "latency_ms": 100.0}
            row.update({f"p{i}": float(p) for i, p in enumerate(tier.probs[qid].get(k, []))})
            rows.append(row)
        append_rows(tier_dir / f"{qid}.csv", rows, cols)


def _answer_files(batch_dir, answer_dir, tier, flip_first=False):
    answer_dir.mkdir(parents=True)
    for f in sorted(batch_dir.glob("batch_*.jsonl")):
        out, seen = [], set()
        for line in f.read_text().splitlines():
            r = json.loads(line)
            ans = tier.answers[r["qid"]][r["text_key"]]
            if flip_first and r["qid"] == "F3" and r["qid"] not in seen:
                ans = "no" if ans == "yes" else "yes"
            seen.add(r["qid"])
            out.append(json.dumps({"qid": r["qid"], "text_key": r["text_key"], "answer": ans}))
        (answer_dir / f.name.replace("batch_", "answers_")).write_text("\n".join(out) + "\n")


def test_cli_end_to_end(tmp_path, capsys):
    w = make_world(n=300, seed=3)
    root = tmp_path / "results"
    p = R.paths(root)
    p.data.mkdir(parents=True)
    w.reviews.to_csv(p.data / "reviews.csv", index=False)
    w.products.to_csv(p.data / "products.csv", index=False)
    _write_tier(p.labels / "big", w.big, True)
    cand_dir = p.labels / "bakeoff" / slug("org/Cand-Model")
    _write_tier(cand_dir, w.small, True)
    (cand_dir / "run_info.json").write_text(json.dumps({"model": "org/Cand-Model"}))
    ref = p.reference
    base = ["--root", str(root)]

    R.main(["export", *base])
    _answer_files(ref / "batches", ref / "answers", w.big, flip_first=True)
    R.main(["import", *base])
    ag = pd.read_csv(ref / "agreement.csv")
    assert list(ag["qid"]) == list(QUESTIONS)
    assert (ag["agreement"] <= 1).all() and (ag["rows"] > 0).all()
    assert ag.loc[ag.qid == "F3", "agreement"].iloc[0] < 1

    R.main(["bakeoff-export", *base, "--n", "30"])
    rows = pd.read_csv(ref / "bakeoff_rows.csv", dtype=str)["text_key"].tolist()
    assert len(rows) == 30
    _answer_files(ref / "bakeoff" / "batches", ref / "bakeoff" / "answers", w.big)
    R.main(["bakeoff-score", *base, "--models", "org/Cand-Model"])
    score = pd.read_csv(ref / "bakeoff_score.csv")
    assert score["pick"].tolist() == [True] and score["complete"].tolist() == [True]
    assert score["rows_per_sec"].iloc[0] == pytest.approx(10.0)
    assert "pick: org/Cand-Model" in capsys.readouterr().out

    R.main(["spotcheck-export", *base, "--n", "10"])
    sheet = pd.read_csv(ref / "spotcheck.csv", dtype=str, keep_default_na=False)
    assert len(sheet) > 0
    sheet["your_answer"] = [a if i % 2 == 0 else "" for i, a in enumerate(sheet["answer_a"])]
    sheet.to_csv(ref / "spotcheck.csv", index=False)
    R.main(["spotcheck-score", *base])
    score = json.loads((ref / "spotcheck_score.json").read_text())
    assert sum(score.values()) == len(sheet) and score["blank"] == len(sheet) // 2
