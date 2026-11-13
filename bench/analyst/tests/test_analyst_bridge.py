"""Tests for bench/analyst/bridge_client.py and the analyst_bridge binary."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_bridge_client", "bridge_client.py")

FAKE = [sys.executable, str(HERE / "fake_bridge.py")]
needs_bridge = pytest.mark.skipif(not mod.BRIDGE.exists(), reason="analyst_bridge not built")

NAME = 'say "hi" \\ back\tslash\x01 \U0001F389'  # quote, backslash, tab, control byte, emoji
T_COLS = "tk:BIGINT,name:VARCHAR,d:DATE,x:DECIMAL"
T_ROWS = [["1", "plain", "1995-03-15", "0.139"], ["2", NAME, "1995-12-31", "1.0"], ["3", "other", "1996-01-02", "2.5"]]
U_COLS = "uk:BIGINT,v:INTEGER"
U_ROWS = [["1", "10"], ["2", "20"], ["2", "5"], ["4", "7"]]


# Write a tables.tsv and one .tbl per table (named relative to the tsv).
# Each table is (columns, rows); every .tbl line ends with "|".
def _dataset(folder: Path, tables: dict[str, tuple[str, list[list[str]]]]) -> Path:
    pid = os.getpid()
    lines = []
    for name, (cols, rows) in tables.items():
        tbl = folder / f"{name}_{pid}.tbl"
        tbl.write_bytes("".join("|".join(r) + "|\n" for r in rows).encode("utf-8"))
        lines.append(f"{name}\t{tbl.name}\t{cols}")

    tsv = folder / f"tables_{pid}.tsv"
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tsv


# Tables t and u, joined on tk = uk.
def _tiny(folder: Path) -> Path:
    return _dataset(folder, {"t": (T_COLS, T_ROWS), "u": (U_COLS, U_ROWS)})


# The fake bridge ignores its tables, so any file does.
def _dummy_tsv(folder: Path) -> Path:
    tsv = folder / f"fake_{os.getpid()}.tsv"
    tsv.write_text("", encoding="utf-8")
    return tsv


# ── one_line ──────────────────────────────────────────────────────────

def test_one_line_collapses_newlines_and_tabs():
    assert mod.one_line("SELECT a,\n\tb\r\nFROM t ;") == "SELECT a, b FROM t"


def test_one_line_drops_only_one_semicolon():
    assert mod.one_line("  SELECT 1;  ") == "SELECT 1"
    assert mod.one_line("SELECT 1;;") == "SELECT 1;"
    assert mod.one_line("SELECT 'a;b'") == "SELECT 'a;b'"


def test_one_line_keeps_spaces_inside_strings():
    assert mod.one_line("SELECT * FROM t WHERE p = 'a  b'") == "SELECT * FROM t WHERE p = 'a  b'"
    assert mod.one_line("SELECT 'it''s  x'  FROM t") == "SELECT 'it''s  x' FROM t"   # '' is an escaped quote
    assert mod.one_line("WHERE a = 'x\ny'") == "WHERE a = 'x y'"
    assert mod.one_line("WHERE a = 'x\r\n\ty'") == "WHERE a = 'x   y'"   # each break or tab is one space


def test_one_line_drops_comments():
    assert (mod.one_line("SELECT COUNT(*) FROM big -- every row\nWHERE n < 10")
            == "SELECT COUNT(*) FROM big WHERE n < 10")
    assert (mod.one_line("SELECT a FROM t -- the customer's rows\nWHERE b = 'x  y'")
            == "SELECT a FROM t WHERE b = 'x  y'")                    # a quote in a comment doesn't count
    assert mod.one_line("SELECT 'a--b' FROM t") == "SELECT 'a--b' FROM t"   # -- in a string is text
    assert mod.one_line("SELECT a FROM t -- last line, no newline") == "SELECT a FROM t"
    assert mod.one_line("SELECT a FROM t; -- done") == "SELECT a FROM t"
    assert mod.one_line("SELECT a -- pick a\rFROM t") == "SELECT a FROM t"   # \r ends a comment too
    assert mod.one_line("SELECT a - b FROM t") == "SELECT a - b FROM t"


# ── client against the fake bridge ────────────────────────────────────

def test_fake_roundtrip(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE) as b:
        assert b.ready["ready"] is True
        r = b.query("SELECT 1")
        assert r["ok"] and r["id"] == "1" and r["rows"] == [["SELECT 1"]]
        assert b.query("SELECT\n  2 ;")["rows"] == [["SELECT 2"]]   # sent as one line
        assert b.restarts == 0


def test_timeout_restarts_and_continues(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE, timeout_s=0.5) as b:
        r = b.query("SLEEP")
        assert r["error"]["kind"] == "timeout" and r["error"]["message"] == "over 0.5 s"
        assert r["ok"] is False and r["id"] == "1" and r["ms"] >= 500
        assert b.restarts == 1 and b.query("SELECT 2")["ok"]


def test_crash_restarts_and_continues(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE) as b:
        r = b.query("DIE")
        assert r["error"]["kind"] == "crash" and r["ok"] is False and r["id"] == "1"
        assert "1" in r["error"]["message"] and r["ms"] >= 0
        assert b.restarts == 1 and b.query("SELECT 3")["ok"]


def test_memory_over_the_cap_restarts_and_continues(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE, max_rss_mb=100) as b:
        old = b._proc
        r = b.query("HOG")
        assert r["error"] == {"kind": "memory", "message": "over 100 MB"}
        assert r["ok"] is False and r["id"] == "1" and r["ms"] >= 500
        assert old.poll() is not None and b._proc is not old   # the killed bridge is reaped
        assert b.restarts == 1 and b.query("SELECT 6")["rows"] == [["SELECT 6"]]


# Count the client's memory reads, still reading the real RSS.
def _count_rss_reads(monkeypatch) -> list[int]:
    reads, real = [], mod._rss_mb
    monkeypatch.setattr(mod, "_rss_mb", lambda pid: reads.append(pid) or real(pid))
    return reads


def test_fast_query_never_reads_memory(tmp_path, monkeypatch):
    reads = _count_rss_reads(monkeypatch)
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE, max_rss_mb=1) as b:
        assert b.query("SELECT 7")["ok"] and b.restarts == 0
    assert reads == []


def test_slow_query_under_the_cap_reads_memory_every_half_second(tmp_path, monkeypatch):
    reads = _count_rss_reads(monkeypatch)
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE) as b:
        assert b.query("HOG")["rows"] == [["HOG"]] and b.restarts == 0
        assert set(reads) == {b._proc.pid}
    assert 2 <= len(reads) <= 5   # a 2 s query, read from 0.5 s on


def test_ps_that_cant_run_skips_the_memory_read(tmp_path, monkeypatch):
    calls = []

    def no_ps(*args, **kwargs):
        calls.append(args)
        raise OSError(12, "Cannot allocate memory")

    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE, max_rss_mb=100) as b:
        monkeypatch.setattr(mod.subprocess, "run", no_ps)
        r = b.query("HOG")                    # over the cap, but no reading says so
        assert r["ok"] and r["rows"] == [["HOG"]] and b.restarts == 0
    assert calls                              # ps was tried


def test_rss_mb():
    assert 1 < mod._rss_mb(os.getpid()) < 100_000
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    assert mod._rss_mb(gone.pid) == 0   # an ended process reads as nothing


def test_dead_process_before_write_is_a_crash(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE) as b:
        b._proc.kill()
        b._proc.wait()
        assert b.query("SELECT 4")["error"]["kind"] == "crash"
        assert b.restarts == 1 and b.query("SELECT 5")["rows"] == [["SELECT 5"]]


def test_wrong_reply_id_raises(tmp_path):
    with mod.Bridge(_dummy_tsv(tmp_path), command=FAKE) as b:
        with pytest.raises(RuntimeError):
            b.query("WRONGID")


def test_failed_load_raises(tmp_path):
    die_at_start = [sys.executable, "-c", "import sys; sys.exit(1)"]
    with pytest.raises(RuntimeError):
        mod.Bridge(_dummy_tsv(tmp_path), command=die_at_start).start()


def test_defaults_and_unique_db_file():
    a, b = mod.Bridge(Path("x.tsv")), mod.Bridge(Path("x.tsv"))
    assert a.timeout_s == 30.0 and a.command == [str(mod.BRIDGE)] and a.restarts == 0
    assert a.max_rss_mb == mod.an.MAX_RSS_MB == 4096
    assert a.db_file != b.db_file and str(os.getpid()) in a.db_file.name


# ── the real bridge ───────────────────────────────────────────────────

@needs_bridge
def test_real_bridge_types_profile_and_escaping(tmp_path):
    with mod.Bridge(_tiny(tmp_path)) as b:
        assert b.ready["ready"] is True and b.ready["tables"] == 2
        assert b.ready["rows"] == {"t": 3, "u": 4} and b.ready["load_ms"] >= 0
        assert b.db_file.exists()

        r = b.query("SELECT name, d, x FROM t ORDER BY x")
        assert r["ok"] and [c["type"] for c in r["columns"]] == ["VARCHAR", "DATE", "DECIMAL"]
        assert [c["name"] for c in r["columns"]] == ["name", "d", "x"]
        assert r["rows"][0][1] == "1995-03-15" and r["rows"][0][2] == pytest.approx(0.139)  # not rounded to 0.14
        assert r["rows"][1][0] == NAME   # every byte comes back
        assert r["row_count"] == 3 and r["truncated"] is False and r["ms"] >= 0

        ints = b.query("SELECT uk, v FROM u ORDER BY v")
        assert [c["type"] for c in ints["columns"]] == ["BIGINT", "INTEGER"]
        assert ints["rows"] == [[2, 5], [4, 7], [1, 10], [2, 20]]
        assert all(isinstance(x, int) for row in ints["rows"] for x in row)

        j = b.query("SELECT SUM(v) FROM t JOIN u ON tk = uk")
        assert j["ok"] and j["rows"] == [[pytest.approx(35.0)]]
        join = j["profile"]["join"]
        assert join["tables"] == ["t", "u"] and join["edges"] == [[0, 1]]
        assert sorted(join["order"]) == [0, 1] and isinstance(join["source"], str)
        assert len(join["est_step_rows"]) == 2 and len(join["est_scan_rows"]) == 2
        assert j["profile"]["step_rows"][-1] == 3 and sorted(j["profile"]["scan_rows"]) == [3, 4]
        ops = j["profile"]["operators"]
        assert ops[0]["depth"] == 0 and any(o["name"].startswith("HashJoin") for o in ops)
        assert set(ops[0]) == {"name", "depth", "rows", "total_ms", "self_ms"}

    assert not b.db_file.exists()


@needs_bridge
def test_real_bridge_error_kinds(tmp_path):
    with mod.Bridge(_tiny(tmp_path)) as b:
        assert b.query("SELEC 1")["error"]["kind"] == "parse"
        assert b.query("SELECT * FROM nosuch")["error"]["kind"] == "plan"
        bad = b.query("SELECT zzz FROM t")
        assert bad["ok"] is False and bad["error"]["kind"] in ("plan", "run") and bad["error"]["message"]
        assert b.query("SELECT COUNT(*) FROM u")["rows"] == [[4]]
        assert b.restarts == 0


@needs_bridge
def test_real_bridge_comment_before_where_keeps_the_filter(tmp_path):
    with mod.Bridge(_tiny(tmp_path)) as b:
        r = b.query("SELECT COUNT(*) FROM u -- the customer's rows\nWHERE v < 10")
        assert r["ok"] and r["rows"] == [[2]]


@needs_bridge
def test_real_bridge_row_cap(tmp_path):
    rows = [[str(n)] for n in range(10050)]
    with mod.Bridge(_dataset(tmp_path, {"big": ("n:INTEGER", rows)})) as b:
        r = b.query("SELECT n FROM big")
        assert len(r["rows"]) == 10000 and r["row_count"] == 10050 and r["truncated"] is True
        small = b.query("SELECT n FROM big WHERE n < 10")
        assert small["row_count"] == 10 and small["truncated"] is False


@needs_bridge
def test_real_bridge_crash_restarts(tmp_path):
    with mod.Bridge(_tiny(tmp_path)) as b:
        b._proc.kill()
        assert b.query("SELECT COUNT(*) FROM t")["error"]["kind"] == "crash"
        assert b.restarts == 1 and b.query("SELECT COUNT(*) FROM t")["rows"] == [[3]]
    assert not b.db_file.exists()


@needs_bridge
def test_real_bridge_memory_cap_restarts(tmp_path):
    rows = [[str(n)] for n in range(4000)]
    tsv = _dataset(tmp_path, {"x": ("n:INTEGER", rows), "y": ("m:INTEGER", rows)})
    with mod.Bridge(tsv, timeout_s=20) as b:
        b.max_rss_mb = mod._rss_mb(b._proc.pid) + 100   # a little above the loaded bridge
        old = b._proc
        r = b.query("SELECT n, m FROM x, y")           # 16M rows, about 1.3 GB if it ran to the end
        assert r["error"] == {"kind": "memory", "message": f"over {b.max_rss_mb:g} MB"} and r["ms"] < 10_000
        assert old.poll() is not None and b._proc is not old   # the killed bridge is reaped
        assert b.restarts == 1 and b.query("SELECT COUNT(*) FROM x")["rows"] == [[4000]]
    assert not b.db_file.exists()


@needs_bridge
@pytest.mark.parametrize("case", ["type", "missing_file", "bad_number", "fields", "no_tables"])
def test_real_bridge_bad_tables_fail_start(tmp_path, case):
    pid = os.getpid()
    tbl = tmp_path / f"a_{pid}.tbl"
    tbl.write_text("1|\n" if case != "bad_number" else "oops|\n", encoding="utf-8")
    line = {
        "type": f"a\t{tbl.name}\tn:FLOAT",
        "missing_file": f"a\tnope_{pid}.tbl\tn:INTEGER",
        "bad_number": f"a\t{tbl.name}\tn:INTEGER",
        "fields": f"a\t{tbl.name}",
        "no_tables": "",
    }[case]
    tsv = tmp_path / f"bad_{pid}.tsv"
    tsv.write_text(line + "\n", encoding="utf-8")
    log = tmp_path / f"bridge_{pid}.log"

    b = mod.Bridge(tsv, log_path=log)
    with pytest.raises(RuntimeError):
        b.start()
    b.close()
    assert "analyst_bridge: " in log.read_text(encoding="utf-8")
    assert not b.db_file.exists()


# Run the binary by hand, outside the client.
def _raw_bridge(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([str(mod.BRIDGE), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


@needs_bridge
def test_real_bridge_request_lines_and_default_db_file(tmp_path):
    proc = _raw_bridge(["--tables", str(_tiny(tmp_path))])
    ready = json.loads(proc.stdout.readline())
    default_db = Path(f"/tmp/analyst_bridge_{proc.pid}.db")
    assert ready["ready"] is True and default_db.exists()

    out, err = proc.communicate(b"no tab here\n7\tSELECT COUNT(*) FROM u\r\n", timeout=30)
    first, second = (json.loads(line) for line in out.splitlines())
    assert first["id"] == "" and first["ok"] is False and first["error"]["kind"] == "request"
    assert first["ms"] == 0
    assert second["id"] == "7" and second["rows"] == [[4]]   # trailing \r is stripped
    assert proc.returncode == 0 and not default_db.exists()


@needs_bridge
@pytest.mark.parametrize("args", [[], ["--bogus"], ["--tables"], ["--tables", "x.tsv", "--pool-size", "lots"]])
def test_real_bridge_bad_arguments(args):
    proc = _raw_bridge(args)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 1 and err.decode().startswith("analyst_bridge: ")
