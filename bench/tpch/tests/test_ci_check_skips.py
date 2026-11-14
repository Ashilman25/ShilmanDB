"""Tests for scripts/ci_check_skips.sh, the CI check that TPC-H answer tests really ran."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ci_check_skips.sh"

PASSED_LINES = """\
Test project /work/build
      Start  12: SqliteComparisonTest.Q1
 12/930 Test  #12: SqliteComparisonTest.Q1 ..........   Passed    0.40 sec
      Start  13: DiskManagerTest.DirectIORoundTrip
 13/930 Test  #13: DiskManagerTest.DirectIORoundTrip ...***Skipped   0.01 sec
"""

FOOTER = """
100% tests passed, 0 tests failed out of 930

Total Test time (real) =   8.10 sec
"""


# A ctest log whose "did not run" section lists these entries.
def ctest_log(*skipped: str) -> str:
    text = PASSED_LINES + FOOTER
    if skipped:
        text += "\nThe following tests did not run:\n"
        text += "".join(f"\t{100 + i:3d} - {name} (Skipped)\n" for i, name in enumerate(skipped))
    return text


def run(tmp_path: Path, text: str) -> subprocess.CompletedProcess:
    log = tmp_path / "ctest.log"
    log.write_text(text)
    return subprocess.run(["bash", str(SCRIPT), str(log)], capture_output=True, text=True)


def test_tpch_skip_fails_and_names_it(tmp_path):
    result = run(tmp_path, ctest_log("SqliteComparisonTest.Q1"))
    assert result.returncode == 1
    assert "SqliteComparisonTest.Q1" in result.stdout + result.stderr


def test_other_skip_passes(tmp_path):
    result = run(tmp_path, ctest_log("DiskManagerTest.DirectIO"))
    assert result.returncode == 0


def test_no_skips_passes(tmp_path):
    result = run(tmp_path, ctest_log())
    assert result.returncode == 0


@pytest.mark.parametrize("name", [
    "SqliteComparisonTest.Q6",
    "EvictionPolicyTpchTest.ClockMatchesLru",
    "TpchEstimatesTest.TpchScanChoices",
    "TpchStreamsTest.StreamAnswers",
    "JoinOrderParityTest.Q5",
    "VectorizedTpchParityTest.Q19",
])
def test_each_tpch_suite_is_caught(tmp_path, name):
    result = run(tmp_path, ctest_log("DiskManagerTest.DirectIO", name))
    assert result.returncode == 1
    out = result.stdout + result.stderr
    assert name in out
    assert "DiskManagerTest.DirectIO" not in out


def test_only_tpch_names_are_printed_all_of_them(tmp_path):
    result = run(tmp_path, ctest_log("TpchStreamsTest.A", "ValueTest.B", "JoinOrderParityTest.C"))
    assert result.returncode == 1
    out = result.stdout + result.stderr
    assert "TpchStreamsTest.A" in out and "JoinOrderParityTest.C" in out
    assert "ValueTest.B" not in out


def test_tpch_name_outside_the_section_is_ignored(tmp_path):
    # The progress lines name SqliteComparisonTest.Q1, but nothing TPC-H was skipped
    result = run(tmp_path, ctest_log("DiskManagerTest.DirectIORoundTrip"))
    assert result.returncode == 0


def test_section_ends_before_failed_list(tmp_path):
    text = ctest_log("DiskManagerTest.DirectIO") + "\nThe following tests FAILED:\n\t 12 - SqliteComparisonTest.Q1 (Failed)\n"
    result = run(tmp_path, text)
    assert result.returncode == 0


def test_missing_log_is_an_error(tmp_path):
    result = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "nope.log")], capture_output=True, text=True)
    assert result.returncode == 2


def test_no_argument_is_an_error():
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
