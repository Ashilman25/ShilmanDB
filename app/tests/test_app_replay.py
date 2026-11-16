"""Tests for app/replay_check.py."""
from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from app import measured, replay_check, sixd
from app.datasets import MODEL_ID, default_paths
from app.recorded_model import RecordedModel
from app.replay_check import compare_record, timings
from app.tests.fakes import FakeBridge

an, fz = sixd.an, sixd.fz

# The questions the fixture run file answers.
QUESTION = "How many nations does each region hold?"
CANNOT_QUESTION = "Which nation has the most customers?"
LINE = re.compile(r"(t\d{3}) (same|differs) \d+\.\d s")


# One attempt as run_attempts writes it.
def _attempt(raw: str, seconds: float, ms: float | None = None, kind: str = "sql", sql: str | None = "SELECT 1") -> dict:
    reply = None if ms is None else {"ok": True, "ms": ms}
    return {"raw": raw, "kind": kind, "sql": sql, "reply": reply, "seconds": seconds, "prompt_sha1": None}


# A record with two primary attempts and a rephrase with one.
def _record(qid: str = "t013") -> dict:
    return {
        "qid": qid, "dataset": "tpch", "split": "test",
        "primary": {"attempts": [_attempt("bad", 1.0, kind="malformed", sql=None), _attempt("good", 2.0, ms=500.0)],
                    "final": "shown"},
        "rephrase": {"raw": "In other words?", "text": "Other words", "seconds": 0.5, "final": "shown",
                     "attempts": [_attempt("again", 1.5, ms=250.0)]},
    }


def test_compare_record_identical_is_none():
    assert compare_record(_record(), _record()) is None
    changed = _record()
    changed["seconds_total"] = 99.0
    changed["primary"]["attempts"][1]["seconds"] = 7.0
    changed["primary"]["attempts"][1]["reply"] = {"ok": True, "ms": 1.0}
    assert compare_record(_record(), changed) is None  # times and replies aren't compared


def test_compare_record_first_difference():
    replayed = _record()
    replayed["primary"]["attempts"][1]["raw"] = "other"
    assert compare_record(_record(), replayed) == "t013 primary attempt 2 raw differs"


# Each place a record can differ, with the message, in the order they're checked.
@pytest.mark.parametrize("where, change", [
    ("primary attempt 1 raw", lambda r: r["primary"]["attempts"][0].update(raw="x")),
    ("primary attempt 1 kind", lambda r: r["primary"]["attempts"][0].update(kind="sql")),
    ("primary attempt 2 sql", lambda r: r["primary"]["attempts"][1].update(sql="SELECT 2")),
    ("primary attempt 3 raw", lambda r: r["primary"]["attempts"].append(_attempt("more", 1.0))),
    ("primary attempt 2 raw", lambda r: r["primary"]["attempts"].pop()),
    ("primary final", lambda r: r["primary"].update(final="failed")),
    ("rephrase raw", lambda r: r["rephrase"].update(raw="x")),
    ("rephrase text", lambda r: r["rephrase"].update(text="x")),
    ("rephrase final", lambda r: r["rephrase"].update(final="failed")),
    ("rephrase attempt 1 raw", lambda r: r["rephrase"]["attempts"][0].update(raw="x")),
    ("rephrase attempt 1 kind", lambda r: r["rephrase"]["attempts"][0].update(kind="cannot")),
    ("rephrase attempt 1 sql", lambda r: r["rephrase"]["attempts"][0].update(sql="SELECT 3")),
])
def test_compare_record_names_each_place(where, change):
    replayed = _record()
    change(replayed)
    assert compare_record(_record(), replayed) == f"t013 {where} differs"


def test_compare_record_reports_the_first_of_several():
    replayed = _record()
    replayed["rephrase"]["raw"] = "x"
    replayed["primary"]["final"] = "failed"
    assert compare_record(_record(), replayed) == "t013 primary final differs"


def test_timings_on_hand_made_records():
    skipped = {"raw": None, "text": None, "seconds": None, "final": "skipped", "attempts": []}
    cannot = {"qid": "t002", "primary": {"attempts": [_attempt("no", 1.0, kind="cannot")], "final": "cannot"},
              "rephrase": skipped}
    result = timings([_record(), cannot, _record("t014")])

    # first answers 3.5, 1.0, 3.5; verdicts 3.5 + 0.5 + 1.5 + 0.25 = 5.75, 1.0, 5.75
    assert result == {"first_answer": {"median": 3.5, "p90": 3.5}, "verdict": {"median": 5.75, "p90": 5.75}}
    assert timings([cannot]) == {"first_answer": {"median": 1.0, "p90": 1.0}, "verdict": {"median": 1.0, "p90": 1.0}}
    assert timings([]) == {"first_answer": {"median": None, "p90": None}, "verdict": {"median": None, "p90": None}}


def test_timings_p90_by_nearest_rank():
    records = []
    for seconds in range(1, 11):
        rec = _record()
        rec["primary"]["attempts"] = [_attempt("a", float(seconds))]
        records.append(rec)
    result = timings(records)
    assert result["first_answer"] == {"median": 5.5, "p90": 9.0}


def test_timings_on_saved_test_run():
    run = default_paths().run_file("test")
    if not run.is_file():
        pytest.skip("the saved test run isn't here")
    result = timings(an.read_jsonl(run))
    assert {k: round(v, 2) for k, v in result["first_answer"].items()} == {"median": 2.51, "p90": 3.63}
    assert {k: round(v, 2) for k, v in result["verdict"].items()} == {"median": 5.29, "p90": 7.64}


# --- main, on the fixture tree ---

# A FakeBridge used as a context manager, as the real one is.
class ContextBridge(FakeBridge):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


# The saved bridge replies of a run file, by SQL.
def _saved_replies(run: Path) -> dict:
    replies = {}
    for rec in an.read_jsonl(run):
        for a in rec["primary"]["attempts"] + rec["rephrase"]["attempts"]:
            if a["sql"] is not None and a["reply"] is not None:
                replies[a["sql"]] = a["reply"]
    return replies


# A SQLite file of the fixture's .tbl files, as run_question reads it.
def _build_sqlite(dataset_dir: Path) -> None:
    with sqlite3.connect(dataset_dir / "tpch.sqlite") as conn:
        for line in (dataset_dir / "tables.tsv").read_text(encoding="utf-8").splitlines():
            table, tbl, columns = line.split("\t")
            names = [c.split(":")[0] for c in columns.split(",")]
            conn.execute(f"CREATE TABLE {table} ({', '.join(names)})")
            for row in (dataset_dir / tbl).read_text(encoding="utf-8").splitlines():
                values = row.split("|")[:len(names)]
                conn.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(names))})", values)
    conn.close()


@pytest.fixture
def replay(fixture_paths, monkeypatch):
    runs = fixture_paths.results / "runs"
    shutil.copy(runs / "dev_gemma4-26b.jsonl", runs / "test_gemma4-26b.jsonl")
    questions = [{"qid": "t001", "dataset": "tpch", "split": "test", "question": QUESTION},
                 {"qid": "t002", "dataset": "tpch", "split": "test", "question": CANNOT_QUESTION}]
    (fixture_paths.results / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    _build_sqlite(fixture_paths.dataset_dir("tpch"))
    monkeypatch.setattr(fz, "verify", lambda results, code_dir, configs: [])

    run = fixture_paths.run_file("test")
    bridges = []

    def bridge_factory(name, tables_tsv):
        bridges.append((name, tables_tsv, ContextBridge(_saved_replies(run))))
        return bridges[-1][2]

    models = []

    def model_factory(contexts):
        models.append((contexts, RecordedModel(run)))
        return models[-1][1]

    return fixture_paths, bridge_factory, model_factory, bridges, models


# main on the fixture with the given flags, its exit code and printed lines.
def _run(replay_args, argv, capsys, model_factory=None):
    paths, bridge_factory, default_model, _, _ = replay_args
    code = replay_check.main(argv, paths=paths, model_factory=model_factory or default_model, bridge_factory=bridge_factory)
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.err


def test_replay_refuses_without_clean_freeze(replay, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(fz, "verify", lambda results, code_dir, configs: ["checks.py changed"])
    code, lines, err = _run(replay, ["--out", str(tmp_path / "out")], capsys)
    assert code == 2
    assert "checks.py changed" in [line.strip() for line in lines + err.splitlines()]
    assert replay[4] == [] and not (tmp_path / "out").exists()


def test_replay_proof_with_recorded_model(replay, capsys, tmp_path):
    paths, _, _, bridges, models = replay
    out = tmp_path / "out"
    code, lines, _ = _run(replay, ["--out", str(out)], capsys)

    assert code == 0
    proof = json.loads((out / "cache_proof.json").read_text(encoding="utf-8"))
    assert (proof["identical"], proof["total"]) == (2, 2)
    assert proof["differences"] == [] and proof["uncached_differences"] == []
    assert proof["versions"] == measured.mlx_versions()
    assert proof["cache_code_sha1"] == measured.cache_code_sha1(paths.app_dir)
    assert proof["files"] == measured.file_hashes(paths)
    assert proof["engine_src_sha1"] == measured.tree_sha1(paths.src)
    assert proof["model"] == fz.read_model_configs(paths.code / "model_mlx.py")[MODEL_ID]
    assert proof["limit"] is None and proof["cached_tokens"] == {}
    assert set(proof["timings"]) == {"replay", "saved", "saved_all"}
    assert proof["timings"]["saved"] == proof["timings"]["saved_all"] == timings(an.read_jsonl(paths.run_file("test")))
    assert not list(out.glob("*.tmp")) and not list(out.glob(".*.tmp"))

    assert [m.groups() for m in map(LINE.fullmatch, lines) if m] == [("t001", "same"), ("t002", "same")]
    assert "identical 2 of 2" in lines
    assert str(out / "cache_proof.json") in lines[-1]
    assert len(models) == 1 and list(models[0][0]) == ["tpch"]
    assert [(name, tsv) for name, tsv, _ in bridges] == [("tpch", paths.dataset_dir("tpch") / "tables.tsv")]
    assert bridges[0][2].started and bridges[0][2].closed

    record = measured.read_json(default_paths().measured)
    gate = measured.cache_gate(paths, record, proof, proof["versions"], proof["cache_code_sha1"],
                               fz.read_model_configs(paths.code / "model_mlx.py")[MODEL_ID])
    assert gate == measured.CacheGate(False, "the replay didn't cover all 300 test questions")


def test_replay_limit_per_dataset(replay, capsys, tmp_path):
    code, lines, _ = _run(replay, ["--out", str(tmp_path), "--limit", "1"], capsys)
    proof = json.loads((tmp_path / "cache_proof.json").read_text(encoding="utf-8"))
    assert code == 0 and (proof["identical"], proof["total"], proof["limit"]) == (1, 1, 1)
    assert [m.group(1) for m in map(LINE.fullmatch, lines) if m] == ["t001"]
    saved = an.read_jsonl(replay[0].run_file("test"))
    assert proof["timings"]["saved"] == timings(saved[:1]) and proof["timings"]["saved_all"] == timings(saved)


# Answers like the saved run, except the first prompt gets another reply; plain is the saved run.
class DriftingModel:
    def __init__(self, run: Path):
        self.plain = RecordedModel(run)
        self.config_id = MODEL_ID
        self.cached_tokens = {"tpch": 4096}

    def generate(self, prompt: str):
        raw, seconds = self.plain.generate(prompt)
        return (raw + " ", seconds) if QUESTION in prompt and "## Your previous reply" not in prompt else (raw, seconds)


def test_replay_differing_question_reruns_uncached(replay, capsys, tmp_path):
    run = replay[0].run_file("test")
    code, lines, _ = _run(replay, ["--out", str(tmp_path)], capsys, model_factory=lambda contexts: DriftingModel(run))
    proof = json.loads((tmp_path / "cache_proof.json").read_text(encoding="utf-8"))

    assert code == 1
    assert (proof["identical"], proof["total"]) == (1, 2)
    assert proof["differences"] == ["t001 primary attempt 1 raw differs"]
    assert proof["uncached_differences"] == []  # the plain model matched, so the cache did it
    assert proof["cached_tokens"] == {"tpch": 4096}
    assert "t001 primary attempt 1 raw differs" in " ".join(lines)
    assert any("tpch" in line and "4096" in line for line in lines)
    assert "identical 1 of 2" in lines


# Always another reply, plain included.
class AlwaysOff(RecordedModel):
    def generate(self, prompt: str):
        raw, seconds = super().generate(prompt)
        return raw + " ", seconds


def test_replay_uncached_differences_listed(replay, capsys, tmp_path):
    run = replay[0].run_file("test")
    code, _, _ = _run(replay, ["--out", str(tmp_path)], capsys, model_factory=lambda contexts: AlwaysOff(run))
    proof = json.loads((tmp_path / "cache_proof.json").read_text(encoding="utf-8"))
    assert code == 1 and proof["identical"] == 0
    assert proof["uncached_differences"] == ["t001 primary attempt 1 raw differs", "t002 primary attempt 1 raw differs"]


def test_replay_uncached_only_never_writes_proof(replay, capsys, tmp_path):
    run = replay[0].run_file("test")
    reruns = []

    class Watched(RecordedModel):
        @property
        def plain(self):
            reruns.append(1)
            return self

    code, lines, _ = _run(replay, ["--out", str(tmp_path), "--uncached-only"], capsys,
                          model_factory=lambda contexts: AlwaysOff(run))
    assert code == 1
    assert not (tmp_path / "cache_proof.json").exists()
    check = json.loads((tmp_path / "uncached_check.json").read_text(encoding="utf-8"))
    assert (check["identical"], check["total"]) == (0, 2)
    assert check["differences"] == check["uncached_differences"] == ["t001 primary attempt 1 raw differs",
                                                                     "t002 primary attempt 1 raw differs"]
    assert str(tmp_path / "uncached_check.json") in lines[-1]

    code, _, _ = _run(replay, ["--out", str(tmp_path), "--uncached-only"], capsys, model_factory=lambda contexts: Watched(run))
    assert code == 0 and reruns == [] and not (tmp_path / "cache_proof.json").exists()


def test_replay_missing_question_refuses(replay, capsys, tmp_path):
    paths = replay[0]
    (paths.results / "questions.jsonl").write_text(json.dumps({"qid": "t001", "question": QUESTION}) + "\n", encoding="utf-8")
    code, lines, err = _run(replay, ["--out", str(tmp_path / "out")], capsys)
    assert code == 2
    assert any("t002" in line for line in lines + err.splitlines())
    assert replay[4] == [] and not (tmp_path / "out").exists()


def test_replay_missing_run_file_refuses(replay, capsys, tmp_path):
    replay[0].run_file("test").unlink()
    code, lines, err = _run(replay, ["--out", str(tmp_path / "out")], capsys)
    assert code == 2 and replay[4] == []
    assert any("test_gemma4-26b.jsonl" in line for line in lines + err.splitlines())


def test_replay_failure_mid_pass_writes_nothing(replay, capsys, tmp_path):
    paths, bridge_factory, model_factory, _, _ = replay

    class Broken(ContextBridge):
        def query(self, sql):
            raise KeyboardInterrupt

    out = tmp_path / "out"
    code = replay_check.main(["--out", str(out)], paths=paths, model_factory=model_factory,
                             bridge_factory=lambda name, tsv: Broken({}))
    assert code == 130
    assert "nothing was written" in capsys.readouterr().err
    assert not out.exists()


def test_replay_saved_records_untouched(replay, capsys, tmp_path):
    run = replay[0].run_file("test")
    before = run.read_bytes()
    saved = copy.deepcopy(an.read_jsonl(run))
    _run(replay, ["--out", str(tmp_path)], capsys)
    assert run.read_bytes() == before and an.read_jsonl(run) == saved
