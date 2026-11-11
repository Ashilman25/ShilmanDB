"""Tests for bench/vector_check/questions.csv and vc_common.read_questions."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parents[1] / "vc_common.py"
spec = importlib.util.spec_from_file_location("vc_common", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["vc_common"] = mod
spec.loader.exec_module(mod)

# The ten question wordings from the AI-queries check, copied character for character.
AI53 = {
    "F1": "Does the reviewer say the product stopped working or broke within a few weeks of getting it?",
    "C2": "What is this review mainly about: shipping or delivery; product quality or a defect; "
          "a wrong, fake or missing item; customer service or returns; price or value; other?",
    "F3": "Does the reviewer complain that the item arrived late or that shipping was slow?",
    "F4": "Does the reviewer say the item was counterfeit, fake, or not the genuine brand?",
    "F5a": "Does the review talk about battery life?",
    "F5b": "Is the reviewer positive about the product overall?",
    "F6": "Does the reviewer say the case does not fit their phone?",
    "E7": 'Which phone model does the reviewer say they use? Answer with the model name only, or "none".',
    "F8": "Does the reviewer describe a safety hazard, such as overheating, melting, smoke, sparks, "
          "fire or a swelling battery?",
    "F10": "Does the reviewer praise the seller's or maker's customer service?",
}


def test_counts_and_ids():
    q = mod.read_questions()
    assert len(q) == 210 and q.qid.is_unique
    assert q.qid.tolist()[:200] == [f"q{i:03d}" for i in range(1, 201)]
    assert q.qid.tolist()[200:] == ["F1", "C2", "F3", "F4", "F5a", "F5b", "F6", "E7", "F8", "F10"]


def test_groups_and_columns():
    q = mod.read_questions()
    assert q.columns.tolist() == ["qid", "group", "text"]
    assert q.group.tolist() == ["written"] * 200 + ["ai53"] * 10


def test_ai53_wordings_verbatim():
    q = mod.read_questions().set_index("qid")
    for qid, text in AI53.items():
        assert q.loc[qid, "text"] == text


def test_written_lengths_and_unique():
    w = mod.read_questions().query("group == 'written'")
    assert w.text.map(lambda t: 4 <= len(t.split()) <= 20).all()
    assert w.text.str.lower().is_unique
    assert (w.text == w.text.str.strip()).all()
    assert not w.text.str.contains("\n").any()


def test_written_never_repeat_ai53():
    q = mod.read_questions()
    written = set(q.query("group == 'written'").text.str.lower())
    assert not written & {t.lower() for t in AI53.values()}


def test_read_questions_keeps_every_value_a_string(tmp_path):
    path = tmp_path / f"questions_{os.getpid()}.csv"
    path.write_text('text,qid,group\nNA,q1,written\nNone,q2,written\n"say ""none"", 7",q3,ai53\n')
    q = mod.read_questions(path)
    assert q.columns.tolist() == ["qid", "group", "text"]
    assert q.text.tolist() == ["NA", "None", 'say "none", 7']
    assert q.qid.tolist() == ["q1", "q2", "q3"]
    assert all(isinstance(v, str) for col in q for v in q[col])
    # Whole columns of numbers, which pandas would turn into ints and floats on its own.
    numeric = tmp_path / f"numeric_{os.getpid()}.csv"
    numeric.write_text("qid,group,text\n007,1,1.5\n010,2,2\n")
    q = mod.read_questions(numeric)
    assert q.qid.tolist() == ["007", "010"]
    assert q.group.tolist() == ["1", "2"]
    assert q.text.tolist() == ["1.5", "2"]
