"""Tests for bench/tpch/generate_tpch_data.sh, run on a copy with stub git, make and dbgen."""
from __future__ import annotations

import csv
import importlib.util
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "generate_tpch_data.sh"

GIT_STUB = """#!/bin/bash
echo "git $*" >> "$STUB_LOG"
exit 1
"""

# make dbgen leaves a fake dbgen that reads -s, -C and -S, writing into $DSS_PATH.
# Whole: the 8 tables, each holding its -s value.
# orders also gets a blank line, a whitespace line and a second row.
# FAKE_SKIP names a table it leaves out, FAKE_LOCK one it writes with mode 000.
# Chunked: two rows per chunk file, nation and region whole.
# FAKE_FAIL_CHUNK names a chunk number that exits 3.
# FAKE_SKIP_CHUNK names a <table>.<chunk> it leaves out.
# FAKE_SLEEP is how long each chunk process then sleeps (only FAKE_SLEEP_CHUNK, if set).
# FAKE_PID_LOG gets "<chunk> <pid>" from each chunk process once its files are written.
FAKE_DBGEN = r"""#!/bin/bash
sf=""; chunks=1; chunk=0
while [ $# -gt 0 ]; do
    case "$1" in
        -s) sf="$2"; shift 2 ;;
        -C) chunks="$2"; shift 2 ;;
        -S) chunk="$2"; shift 2 ;;
        *) shift ;;
    esac
done
out="${DSS_PATH:-.}"
echo "dbgen -s $sf -C $chunks -S $chunk $out" >> "${DBGEN_LOG:-/dev/null}"
if [ "$chunks" -gt 1 ]; then
    [ "$chunk" = "${FAKE_FAIL_CHUNK:-}" ] && exit 3
    for t in customer lineitem orders part partsupp supplier; do
        [ "$t.$chunk" = "${FAKE_SKIP_CHUNK:-}" ] && continue
        printf '%s %s\n%s %s\n' "$t" "$chunk" "$t" "$chunk" > "$out/$t.tbl.$chunk"
    done
    for t in nation region; do echo "$t from $chunk" > "$out/$t.tbl"; done
    [ -n "${FAKE_PID_LOG:-}" ] && echo "$chunk $$" >> "$FAKE_PID_LOG"
    [ -n "${FAKE_SLEEP_CHUNK:-}" ] && [ "$chunk" != "$FAKE_SLEEP_CHUNK" ] && exit 0
    [ -n "${FAKE_SLEEP:-}" ] && exec sleep "$FAKE_SLEEP"
    exit 0
fi
for t in customer lineitem nation orders part partsupp region supplier; do
    [ "$t" = "${FAKE_SKIP:-}" ] && continue
    echo "$sf" > "$out/$t.tbl"
    [ "$t" = orders ] && printf '\n \t\nsecond\n' >> "$out/$t.tbl"
    [ "$t" = "${FAKE_LOCK:-}" ] && chmod 000 "$out/$t.tbl"
done
exit 0
"""

MAKE_STUB = """#!/bin/bash
echo "make $*" >> "$STUB_LOG"
if [ "${1:-}" = "dbgen" ]; then
    cp "$FAKE_DBGEN_SRC" dbgen
    chmod +x dbgen
fi
"""

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]

MAKEFILE = "CC = gcc\nDATABASE =\nMACHINE =\nWORKLOAD =\n"
BM_UTILS = "#include <stdio.h>\n#include <malloc.h>\n    struct stat fstats;\n"
BM_UTILS_PATCHED = "#include <stdio.h>\n#include <stdlib.h>\n    struct stat fstats = {0};\n"


# A copy of the script in tmp_path/repo, stub tools first on PATH, DBGEN_DIR in tmp_path.
def setup(tmp_path: Path, with_dbgen: bool) -> Tuple[Path, Path, dict]:
    script_dir = tmp_path / "repo" / "bench" / "tpch"
    script_dir.mkdir(parents=True)
    shutil.copy(SCRIPT, script_dir / "generate_tpch_data.sh")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("git", GIT_STUB), ("make", MAKE_STUB)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    fake_dbgen = tmp_path / "fake_dbgen.sh"
    fake_dbgen.write_text(FAKE_DBGEN)

    dbgen_dir = tmp_path / "dbgen"
    if with_dbgen:
        dbgen_dir.mkdir()
        (dbgen_dir / "makefile").write_text(MAKEFILE)
        (dbgen_dir / "bm_utils.c").write_text(BM_UTILS)

    log = tmp_path / "stub.log"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}"
    env["DBGEN_DIR"] = str(dbgen_dir)
    env["STUB_LOG"] = str(log)
    env["DBGEN_LOG"] = str(tmp_path / "dbgen.log")
    env["FAKE_DBGEN_SRC"] = str(fake_dbgen)
    return script_dir / "generate_tpch_data.sh", log, env


def run(script: Path, args: List[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True,
                          text=True, timeout=30)


def stub_calls(log: Path) -> List[str]:
    return log.read_text().splitlines() if log.exists() else []


def test_generator_rejects_unknown_sf(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=False)
    result = run(script, ["0.01", "3"], env)
    assert result.returncode == 1
    assert "unknown scale factor" in result.stdout + result.stderr
    assert stub_calls(log) == []
    assert not (tmp_path / "repo" / "bench" / "tpch_data").exists()


def test_generator_default_makes_three(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=True)
    result = run(script, [], env)
    assert result.returncode == 0, result.stdout + result.stderr

    data = tmp_path / "repo" / "bench" / "tpch_data"
    assert sorted(p.name for p in data.iterdir()) == ["sf0.01", "sf0.1", "sf1.0"]
    for sf in ("0.01", "0.1", "1.0"):
        assert sorted(p.stem for p in (data / f"sf{sf}").glob("*.tbl")) == TABLES
        assert (data / f"sf{sf}" / "row_counts.csv").exists()
        assert (data / f"sf{sf}" / "lineitem.tbl").read_text().strip() == sf
    assert stub_calls(log) == ["make clean", "make dbgen"]

    dbgen = tmp_path / "dbgen"
    assert (dbgen / "makefile").read_text() == (
        "CC = cc\nDATABASE = NONE\nMACHINE = LINUX\nWORKLOAD = TPCH\n")
    assert (dbgen / "bm_utils.c").read_text() == BM_UTILS_PATCHED


def test_generator_sf10_only(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=True)
    result = run(script, ["10"], env)
    assert result.returncode == 0, result.stdout + result.stderr

    data = tmp_path / "repo" / "bench" / "tpch_data"
    assert [p.name for p in data.iterdir()] == ["sf10"]
    assert (data / "sf10" / "lineitem.tbl").read_text().strip() == "10"
    assert "Generated SF=10 data in" in result.stdout


def test_generator_zeroes_dbgen_stat_once(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    bm_utils = tmp_path / "dbgen" / "bm_utils.c"
    assert run(script, ["0.01"], env).returncode == 0
    assert bm_utils.read_text() == BM_UTILS_PATCHED

    # A cached, already patched folder stays the same on the next run
    result = run(script, ["0.01"], env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert bm_utils.read_text() == BM_UTILS_PATCHED


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode 000 file")
def test_generator_fails_on_unreadable_table(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    env["FAKE_LOCK"] = "nation"
    result = run(script, ["0.01"], env)
    assert result.returncode == 1
    assert "nation.tbl" in result.stderr
    assert "Done." not in result.stdout


def test_generator_fails_on_missing_table(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    env["FAKE_SKIP"] = "partsupp"
    result = run(script, ["0.01"], env)
    assert result.returncode == 1
    assert "partsupp.tbl" in result.stderr
    assert "Done." not in result.stdout


def test_generator_fails_when_stat_patch_misses(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=True)
    (tmp_path / "dbgen" / "bm_utils.c").write_text("#include <malloc.h>\n    struct stat st;\n")
    result = run(script, ["0.01"], env)
    assert result.returncode == 1
    assert "could not patch dbgen's bm_utils.c (fstats)" in result.stderr
    assert stub_calls(log) == []


# TPC-H's row counts at SF=100, in the script's table order.
SF100_ROWS = {"customer": 15_000_000, "lineitem": 600_037_902, "nation": 25, "orders": 150_000_000,
              "part": 20_000_000, "partsupp": 80_000_000, "region": 5, "supplier": 1_000_000}
CHUNKED = ["customer", "lineitem", "orders", "part", "partsupp", "supplier"]


# Calls one of the script's functions with the script sourced, so its main doesn't run.
def call_function(script: Path, call: str, *args: str, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "$1"; shift; {call} "$@"', "bash", str(script), *args],
                          env=env, capture_output=True, text=True, timeout=60)


# The script's main at SF=100 with every count faked to TPC-H's, so the check passes.
# extra: more shell functions defined after sourcing, to inject a failure.
def run_sf100_with_tpch_counts(script: Path, env: dict, extra: str = "") -> subprocess.CompletedProcess:
    fake_count = 'count_rows() { tpch_sf100_rows "$(basename "$1" .tbl)"; }'
    return subprocess.run(["bash", "-c", f'source "$1"; {fake_count}; {extra}\nmain 100', "bash", str(script)],
                          env=env, capture_output=True, text=True, timeout=60)


def read_rows(path: Path) -> List[List[str]]:
    with path.open(newline="") as f:
        return list(csv.reader(f))


# Nothing of SF=100 is left: no tables, no counts, no chunk folders.
def assert_sf100_kept_nothing(tmp_path: Path) -> None:
    sf_dir = tmp_path / "repo" / "bench" / "tpch_data" / "sf100"
    assert not sf_dir.exists() or not list(sf_dir.rglob("*"))


def test_generate_accepts_100_and_rejects_others(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=False)
    # 100 passes the check and goes on to clone dbgen, which the stub git refuses
    result = run(script, ["100"], env)
    assert result.returncode != 0
    assert "unknown scale factor" not in result.stderr
    assert stub_calls(log)[0].startswith("git clone")

    for bad in ("100.0", "50", "1000"):
        log.unlink()
        log.write_text("")
        result = run(script, ["0.01", bad], env)
        assert result.returncode == 1
        assert f"unknown scale factor '{bad}' (choose from 0.01 0.1 1.0 10 100)" in result.stderr
        assert stub_calls(log) == []


def test_chunks_join_in_numeric_order(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    for i in range(1, 13):
        (root / str(i)).mkdir(parents=True)
        (root / str(i) / f"orders.tbl.{i}").write_text(f"order {i}\n")
    out = tmp_path / "out"
    out.mkdir()
    result = call_function(SCRIPT, "join_chunks", "orders", "12", str(root), str(out))
    assert result.returncode == 0, result.stderr
    # 1..12, not 1, 10, 11, 12, 2, ...
    assert (out / "orders.tbl").read_text() == "".join(f"order {i}\n" for i in range(1, 13))
    assert not list(root.rglob("orders.tbl.*"))


def test_join_stops_at_a_missing_chunk(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    for i in (1, 3):
        (root / str(i)).mkdir(parents=True)
        (root / str(i) / f"part.tbl.{i}").write_text(f"part {i}\n")
    out = tmp_path / "out"
    out.mkdir()
    result = call_function(SCRIPT, "join_chunks", "part", "3", str(root), str(out))
    assert result.returncode != 0
    assert "part.tbl.2" in result.stderr


def test_generate_sf100_runs_16_chunks_and_joins_them(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    result = run_sf100_with_tpch_counts(script, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Generated SF=100 data in" in result.stdout

    # 16 processes, chunks 1..16, each writing into its own folder
    calls = (tmp_path / "dbgen.log").read_text().splitlines()
    assert len(calls) == 16
    assert sorted(int(c.split()[6]) for c in calls) == list(range(1, 17))
    assert all(c.startswith("dbgen -s 100 -C 16 -S ") for c in calls)
    assert len({c.split()[7] for c in calls}) == 16

    sf_dir = tmp_path / "repo" / "bench" / "tpch_data" / "sf100"
    assert sorted(p.name for p in sf_dir.iterdir()) == sorted([f"{t}.tbl" for t in TABLES] + ["row_counts.csv"])
    for table in CHUNKED:
        assert (sf_dir / f"{table}.tbl").read_text() == "".join(f"{table} {i}\n" * 2 for i in range(1, 17))
    assert (sf_dir / "nation.tbl").read_text() == "nation from 1\n"
    assert (sf_dir / "region.tbl").read_text() == "region from 1\n"
    assert read_rows(sf_dir / "row_counts.csv") == [["table", "rows"]] + [[t, str(SF100_ROWS[t])] for t in TABLES]


def test_generate_sf100_refuses_wrong_counts(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    result = run(script, ["100"], env)
    assert result.returncode == 1
    assert "lineitem: expected 600037902, got 32" in result.stderr
    assert "nation: expected 25, got 1" in result.stderr
    assert "Done." not in result.stdout
    assert_sf100_kept_nothing(tmp_path)


def test_generate_sf100_refuses_a_failed_chunk(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    env["FAKE_FAIL_CHUNK"] = "5"
    result = run_sf100_with_tpch_counts(script, env)
    assert result.returncode == 1
    assert "dbgen chunk 5 of 16 failed" in result.stderr
    assert_sf100_kept_nothing(tmp_path)


def test_generate_sf100_refuses_a_missing_chunk(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    env["FAKE_SKIP_CHUNK"] = "orders.9"
    result = run_sf100_with_tpch_counts(script, env)
    assert result.returncode == 1
    assert "orders.tbl.9" in result.stderr
    assert_sf100_kept_nothing(tmp_path)


def test_generator_writes_row_counts(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    result = run(script, ["0.01"], env)
    assert result.returncode == 0, result.stdout + result.stderr

    sf_dir = tmp_path / "repo" / "bench" / "tpch_data" / "sf0.01"
    rows = read_rows(sf_dir / "row_counts.csv")
    # orders' blank and whitespace-only lines aren't rows
    assert rows == [["table", "rows"]] + [[t, "2" if t == "orders" else "1"] for t in TABLES]

    # The same numbers the Python runners count, with the file gone
    (sf_dir / "row_counts.csv").unlink()
    counted = baseline_common().tbl_row_counts(sf_dir)
    assert counted == {t: int(n) for t, n in rows[1:] if t in counted}
    assert len(counted) == 7


def test_generator_failure_leaves_no_old_row_counts(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    sf_dir = tmp_path / "repo" / "bench" / "tpch_data" / "sf0.01"
    sf_dir.mkdir(parents=True)
    (sf_dir / "row_counts.csv").write_text("table,rows\nlineitem,9\n")
    env["FAKE_SKIP"] = "partsupp"
    assert run(script, ["0.01"], env).returncode == 1
    assert not (sf_dir / "row_counts.csv").exists()


# bench/tpch/baseline_common.py, loaded by path once.
def baseline_common():
    name = "baseline_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPT.parent / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# A failure anywhere after the chunks start still keeps nothing.
@pytest.mark.parametrize("failure", [
    'count_tables() { return 1; }',                                                 # writing the counts
    'mv() { case "$2" in */row_counts.csv) return 1 ;; esac; command mv "$@"; }',  # the final move
])
def test_generate_sf100_keeps_nothing_on_a_late_failure(tmp_path: Path, failure: str) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    result = run_sf100_with_tpch_counts(script, env, failure)
    assert result.returncode == 1
    assert "SF=100: kept nothing in" in result.stderr
    assert "Generated SF=100" not in result.stdout
    assert_sf100_kept_nothing(tmp_path)


def test_generate_sf100_stops_started_chunks_when_a_launch_fails(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    env["FAKE_SLEEP"] = "20"
    fail_ninth = 'mkdir() { case "$*" in */chunks/9) return 1 ;; esac; command mkdir "$@"; }'
    start = time.monotonic()
    result = run_sf100_with_tpch_counts(script, env, fail_ninth)
    # The 8 started processes were killed, not waited out
    assert time.monotonic() - start < 10
    assert result.returncode == 1
    assert "SF=100: kept nothing in" in result.stderr
    assert_sf100_kept_nothing(tmp_path)


def test_count_rows_fails_on_a_read_error(tmp_path: Path) -> None:
    blank = tmp_path / "blank.tbl"
    blank.write_text("\n  \n")
    result = call_function(SCRIPT, "count_rows", str(blank))
    assert (result.returncode, result.stdout) == (0, "0\n")

    result = call_function(SCRIPT, "count_rows", str(tmp_path / "missing.tbl"))
    assert result.returncode != 0
    assert result.stdout == ""


def test_generator_stops_when_a_count_fails(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    result = subprocess.run(["bash", "-c", 'source "$1"; count_rows() { return 1; }; main 0.01', "bash", str(script)],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    sf_dir = tmp_path / "repo" / "bench" / "tpch_data" / "sf0.01"
    assert not list(sf_dir.glob("row_counts.csv*"))
    assert "Done." not in result.stdout


# True while a process with this pid exists.
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# {chunk: pid} from the fake dbgen's pid log.
def _chunk_pids(pid_log: Path) -> dict:
    if not pid_log.exists():
        return {}
    return {int(c): int(p) for c, p in (line.split() for line in pid_log.read_text().splitlines())}


# Starts argv, waits until ready() holds, sends sig, and returns the exit code.
# Output goes to files, so a child that outlives the script can't block the test.
def _stop_when(argv: List[str], env: dict, tmp_path: Path, ready, sig: int) -> int:
    with open(tmp_path / "out.log", "w") as out, open(tmp_path / "err.log", "w") as err:
        # The script starts with default signal handling, even if pytest ignores SIGINT
        proc = subprocess.Popen(argv, env=env, stdout=out, stderr=err,
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        try:
            deadline = time.monotonic() + 15
            while not ready():
                assert proc.poll() is None, (tmp_path / "err.log").read_text()
                assert time.monotonic() < deadline, "the chunks never started"
                time.sleep(0.05)
            proc.send_signal(sig)
            return proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_generate_sf100_keeps_nothing_when_stopped(tmp_path: Path, sig: int) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    pid_log = tmp_path / "chunk_pids.log"
    env.update(FAKE_SLEEP="30", FAKE_PID_LOG=str(pid_log))
    try:
        code = _stop_when(["bash", str(script), "100"], env, tmp_path,
                          lambda: len(_chunk_pids(pid_log)) == 16, sig)
        assert code == 1
        assert "SF=100: kept nothing in" in (tmp_path / "err.log").read_text()
        assert_sf100_kept_nothing(tmp_path)
        # Every sleeping chunk was killed and reaped
        assert not [pid for pid in _chunk_pids(pid_log).values() if _alive(pid)]
    finally:
        for pid in _chunk_pids(pid_log).values():
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


def test_a_stop_signals_only_chunks_still_running(tmp_path: Path) -> None:
    script, _, env = setup(tmp_path, with_dbgen=True)
    pid_log, kill_log = tmp_path / "chunk_pids.log", tmp_path / "kill.log"
    env.update(FAKE_SLEEP="30", FAKE_SLEEP_CHUNK="1", FAKE_PID_LOG=str(pid_log))
    # A kill that logs what it's asked to signal
    spy = f'kill() {{ echo "$@" >> {shlex.quote(str(kill_log))}; builtin kill "$@"; }}'

    # Ready once chunks 2..16 have exited and been reaped, so their pids are free
    def ready() -> bool:
        pids = _chunk_pids(pid_log)
        return len(pids) == 16 and not any(_alive(pid) for chunk, pid in pids.items() if chunk != 1)

    try:
        code = _stop_when(["bash", "-c", f'source "$1"; {spy}\nmain 100', "bash", str(script)],
                          env, tmp_path, ready, signal.SIGTERM)
        assert code == 1
        assert kill_log.read_text().split() == [str(_chunk_pids(pid_log)[1])]
        assert not _alive(_chunk_pids(pid_log)[1])
        assert_sf100_kept_nothing(tmp_path)
    finally:
        for pid in _chunk_pids(pid_log).values():
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
