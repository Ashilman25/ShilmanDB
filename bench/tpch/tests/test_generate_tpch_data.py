"""Tests for bench/tpch/generate_tpch_data.sh, run on a copy with stub git, make and dbgen."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "generate_tpch_data.sh"

GIT_STUB = """#!/bin/bash
echo "git $*" >> "$STUB_LOG"
exit 1
"""

# make dbgen leaves a fake dbgen that writes the 8 tables, lineitem.tbl holding its -s value.
# FAKE_SKIP names a table it leaves out, FAKE_LOCK one it writes with mode 000.
FAKE_DBGEN = """#!/bin/bash
for t in customer lineitem nation orders part partsupp region supplier; do
    [ "$t" = "${FAKE_SKIP:-}" ] && continue
    echo "$2" > "$t.tbl"
    [ "$t" = "${FAKE_LOCK:-}" ] && chmod 000 "$t.tbl"
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
        assert sorted(p.stem for p in (data / f"sf{sf}").iterdir()) == TABLES
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
