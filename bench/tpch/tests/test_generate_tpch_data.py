"""Tests for bench/tpch/generate_tpch_data.sh, run on a copy with stub git, make and dbgen."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

SCRIPT = Path(__file__).resolve().parents[1] / "generate_tpch_data.sh"

GIT_STUB = """#!/bin/bash
echo "git $*" >> "$STUB_LOG"
exit 1
"""

# make dbgen leaves a fake dbgen that writes lineitem.tbl holding its -s value
MAKE_STUB = """#!/bin/bash
echo "make $*" >> "$STUB_LOG"
if [ "${1:-}" = "dbgen" ]; then
    printf '#!/bin/bash\\necho "$2" > lineitem.tbl\\n' > dbgen
    chmod +x dbgen
fi
"""

MAKEFILE = "CC = gcc\nDATABASE =\nMACHINE =\nWORKLOAD =\n"
BM_UTILS = "#include <stdio.h>\n#include <malloc.h>\n"


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
        assert (data / f"sf{sf}" / "lineitem.tbl").read_text().strip() == sf
    assert stub_calls(log) == ["make clean", "make dbgen"]

    dbgen = tmp_path / "dbgen"
    assert (dbgen / "makefile").read_text() == (
        "CC = cc\nDATABASE = NONE\nMACHINE = LINUX\nWORKLOAD = TPCH\n")
    assert (dbgen / "bm_utils.c").read_text() == "#include <stdio.h>\n#include <stdlib.h>\n"


def test_generator_sf10_only(tmp_path: Path) -> None:
    script, log, env = setup(tmp_path, with_dbgen=True)
    result = run(script, ["10"], env)
    assert result.returncode == 0, result.stdout + result.stderr

    data = tmp_path / "repo" / "bench" / "tpch_data"
    assert [p.name for p in data.iterdir()] == ["sf10"]
    assert (data / "sf10" / "lineitem.tbl").read_text().strip() == "10"
    assert "Generated SF=10 data in" in result.stdout
