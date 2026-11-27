"""Unit tests for scripts/compare_results.py (PASS, DRIFT, FAIL and the reference flags)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[3] / "scripts" / "compare_results.py"
spec = importlib.util.spec_from_file_location("compare_results_under_test", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)

sys.modules["compare_results_under_test"] = mod
spec.loader.exec_module(mod)


def write_csv(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# Runs main() with argv and returns (exit code, stdout).
def run_main(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    with pytest.raises(SystemExit) as exit_info:
        mod.main(argv)
    return exit_info.value.code, capsys.readouterr().out


# cell_status
def test_pass_within_cent() -> None:
    assert mod.cell_status("100.00", "100.009", 0.01)[0] == "PASS"


def test_drift_on_large_sum() -> None:
    status, abs_diff, rel_diff = mod.cell_status("1000000000000.00", "999999999999.70", 0.01)
    assert status == "DRIFT"
    assert abs_diff == pytest.approx(0.30, abs=1e-3)
    assert rel_diff == pytest.approx(3e-13, rel=1e-2)


def test_one_row_on_1e12_fails() -> None:
    # One $810 row on a 1e12 sum must not hide as drift.
    assert mod.cell_status("1000000000000.00", "999999999190.00", 0.01)[0] == "FAIL"


def test_integer_cells_never_drift() -> None:
    status, _, rel_diff = mod.cell_status("100000000000", "100000000001", 0.01)
    assert status == "FAIL"
    assert rel_diff < mod.DRIFT_RELATIVE


def test_mixed_integer_and_decimal_never_drift() -> None:
    assert mod.cell_status("100000000000", "100000000000.5", 0.01)[0] == "FAIL"


def test_exponent_form_never_drifts() -> None:
    assert mod.cell_status("1e+16", "1.0000000000001e+16", 0.01)[0] == "FAIL"


def test_text_cells_pass_or_fail_only() -> None:
    assert mod.cell_status("BUILDING", "BUILDING", 0.01)[0] == "PASS"
    assert mod.cell_status("BUILDING", "MACHINERY", 0.01)[0] == "FAIL"
    assert mod.cell_status("1995-03-15", "1995-03-16", 0.01)[0] == "FAIL"


def test_non_finite_cells_compare_as_text() -> None:
    assert mod.cell_status("nan", "nan", 0.01)[0] == "PASS"
    assert mod.cell_status("inf", "1.5", 0.01)[0] == "FAIL"


def test_cells_match_is_pass_only() -> None:
    assert mod.cells_match("100.00", "100.009", 0.01)
    assert not mod.cells_match("1000000000000.00", "999999999999.70", 0.01)


# compare_one
def test_drift_never_covers_text_or_rowcount(tmp_path: Path) -> None:
    ref = write_csv(tmp_path / "ref.csv", ["name,total", "a,1000000000000.00"])
    text = write_csv(tmp_path / "text.csv", ["name,total", "b,999999999999.70"])
    result = mod.compare_one(str(ref), str(text), "+index", "10", "Q1", 0.01)
    assert result.status == "FAIL"
    assert result.passed is False

    rows = write_csv(tmp_path / "rows.csv",
                     ["name,total", "a,999999999999.70", "a,999999999999.70"])
    result = mod.compare_one(str(ref), str(rows), "+index", "10", "Q1", 0.01)
    assert result.status == "FAIL"
    assert result.error is not None


def test_column_count_mismatch_fails(tmp_path: Path) -> None:
    ref = write_csv(tmp_path / "ref.csv", ["name,total", "a,1.00"])
    cand = write_csv(tmp_path / "cand.csv", ["name,total", "a,1.00,extra"])
    result = mod.compare_one(str(ref), str(cand), "+index", "10", "Q1", 0.01)
    assert result.status == "FAIL"
    assert result.error == "ShilmanDB row 0 has 3 columns, expected 2"


def test_file_status_is_worst_cell(tmp_path: Path) -> None:
    ref = write_csv(tmp_path / "ref.csv", ["small,big", "100.00,1000000000000.00"])
    cand = write_csv(tmp_path / "cand.csv", ["small,big", "100.005,999999999999.70"])
    result = mod.compare_one(str(ref), str(cand), "+index", "10", "Q1", 0.01)
    assert result.status == "DRIFT"
    assert result.passed is False
    assert result.max_abs_diff == pytest.approx(0.30, abs=1e-3)
    assert result.max_rel_diff == pytest.approx(3e-13, rel=1e-2)


def test_missing_file_is_skip(tmp_path: Path) -> None:
    cand = write_csv(tmp_path / "cand.csv", ["x", "1"])
    result = mod.compare_one(str(tmp_path / "nope.csv"), str(cand), "+index", "10", "Q1", 0.01)
    assert result.status == "SKIP"
    assert result.error is not None


# main
def test_reference_config_subfolder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "duckdb" / "Q1_sf10.csv", ["total", "5.00"])
    write_csv(tmp_path / "cand" / "+index" / "Q1_sf10.csv", ["total", "5.001"])
    code, out = run_main([
        "--shilmandb-dir", str(tmp_path / "cand"),
        "--reference-dir", str(tmp_path / "ref"),
        "--reference-config", "duckdb",
        "--reference-name", "DuckDB",
        "--sf", "10", "--queries", "Q1", "--configs", "+index",
    ], capsys)
    assert code == 0
    assert any(line.strip().startswith("PASS  [+index] Q1 SF=10") for line in out.splitlines())


def test_sqlite_dir_alias_still_works(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "Q6_sf0.01.csv", ["revenue", "1193053.23"])
    write_csv(tmp_path / "cand" / "baseline" / "Q6_sf0.01.csv", ["revenue", "1193053.225"])
    common = ["--shilmandb-dir", str(tmp_path / "cand"), "--sf", "0.01",
              "--queries", "Q6", "--configs", "baseline"]
    new_code, new_out = run_main(common + ["--reference-dir", str(tmp_path / "ref")], capsys)
    old_code, old_out = run_main(common + ["--sqlite-dir", str(tmp_path / "ref")], capsys)
    assert new_code == old_code == 0
    assert new_out == old_out
    assert "  PASS  [baseline] Q6 SF=0.01" in old_out.splitlines()


def test_drift_exit_code_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "Q1_sf10.csv", ["sum_charge", "1000000000000.00"])
    write_csv(tmp_path / "cand" / "drift" / "Q1_sf10.csv", ["sum_charge", "999999999999.70"])
    write_csv(tmp_path / "cand" / "wrong" / "Q1_sf10.csv", ["sum_charge", "999999999190.00"])
    common = ["--shilmandb-dir", str(tmp_path / "cand"), "--reference-dir", str(tmp_path / "ref"),
              "--sf", "10", "--queries", "Q1", "--configs"]

    code, out = run_main(common + ["drift"], capsys)
    assert code == 0
    assert any(line.startswith("  DRIFT [drift] Q1 SF=10  (max abs ") for line in out.splitlines())
    assert ("         row 0, col 'sum_charge': expected='1000000000000.00'  "
            "actual='999999999999.70'  (abs 0.3, rel 3e-13)") in out.splitlines()

    code, out = run_main(common + ["wrong"], capsys)
    assert code == 1
    assert any(line.startswith("  FAIL  [wrong] Q1 SF=10  (1 mismatch(es))") for line in out.splitlines())


def test_header_mismatch_prints_fail_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "Q1_sf0.01.csv", ["a", "1"])
    write_csv(tmp_path / "cand" / "baseline" / "Q1_sf0.01.csv", ["b", "1"])
    code, out = run_main(["--shilmandb-dir", str(tmp_path / "cand"),
                          "--reference-dir", str(tmp_path / "ref"),
                          "--sf", "0.01", "--queries", "Q1", "--configs", "baseline"], capsys)
    lines = out.splitlines()
    assert code == 1
    index = lines.index("  FAIL  [baseline] Q1 SF=0.01")
    assert lines[index + 1] == "        Header mismatch: SQLite=['a'], ShilmanDB=['b']"


def test_sf_all_keeps_three_scale_factors() -> None:
    assert mod.ALL_SCALE_FACTORS == ["0.01", "0.1", "1.0"]
    assert mod.KNOWN_SCALE_FACTORS == ["0.01", "0.1", "1.0", "10", "100"]


# --no-drift: a cell that would be DRIFT is FAIL; nothing else changes
NO_DRIFT_DETAIL = ("         row 0, col 'sum_charge': expected='1000000000000.00'  "
                   "actual='999999999999.00'  (abs 1, rel 1e-12)")


def test_no_drift_turns_drift_into_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1e-12 relative: DRIFT by default, FAIL with the rule
    ref = write_csv(tmp_path / "ref" / "Q1_sf100.csv", ["name,sum_charge", "a,1000000000000.00"])
    cand = write_csv(tmp_path / "cand" / "p" / "Q1_sf100.csv", ["name,sum_charge", "a,999999999999.00"])
    assert mod.compare_one(str(ref), str(cand), "p", "100", "Q1", 0.01).status == "DRIFT"
    result = mod.compare_one(str(ref), str(cand), "p", "100", "Q1", 0.01, no_drift=True)
    assert result.status == "FAIL"
    assert result.drifts == []
    assert result.mismatches == [NO_DRIFT_DETAIL[7:]]
    assert result.max_abs_diff == pytest.approx(1.0)
    assert result.max_rel_diff == pytest.approx(1e-12, rel=1e-6)

    common = ["--shilmandb-dir", str(tmp_path / "cand"), "--reference-dir", str(tmp_path / "ref"),
              "--sf", "100", "--queries", "Q1", "--configs", "p"]
    code, out = run_main(common, capsys)
    assert code == 0
    assert any(line.startswith("  DRIFT [p] Q1 SF=100  (max abs 1, max rel 1e-12)") for line in out.splitlines())

    code, out = run_main(common + ["--no-drift"], capsys)
    lines = out.splitlines()
    assert code == 1
    index = lines.index("  FAIL  [p] Q1 SF=100  (1 mismatch(es))")
    assert lines[index + 1] == NO_DRIFT_DETAIL
    assert "  0/1 comparisons passed  (1 failed)" in lines


def test_no_drift_lists_drift_cells_with_the_mismatches(tmp_path: Path) -> None:
    ref = write_csv(tmp_path / "ref.csv", ["n,big", "1,1000000000000.00", "2,5.00"])
    cand = write_csv(tmp_path / "cand.csv", ["n,big", "1,999999999999.00", "2,7.00"])
    result = mod.compare_one(str(ref), str(cand), "p", "100", "Q1", 0.01, no_drift=True)
    assert result.status == "FAIL"
    assert result.mismatches == [
        "  row 0, col 'big': expected='1000000000000.00'  actual='999999999999.00'  (abs 1, rel 1e-12)",
        "  row 1, col 'big': expected='5.00'  actual='7.00'",
    ]


def test_no_drift_keeps_pass_fail_and_skip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "Q1_sf100.csv", ["total", "5.00"])
    write_csv(tmp_path / "ref" / "Q3_sf100.csv", ["total", "5.00"])
    write_csv(tmp_path / "cand" / "p" / "Q1_sf100.csv", ["total", "5.009"])       # within the tolerance
    write_csv(tmp_path / "cand" / "p" / "Q3_sf100.csv", ["total", "5.00", "6.00"])  # row count
    common = ["--shilmandb-dir", str(tmp_path / "cand"), "--reference-dir", str(tmp_path / "ref"),
              "--sf", "100", "--queries", "Q1", "Q3", "Q5", "--configs", "p"]
    code, out = run_main(common, capsys)
    flag_code, flag_out = run_main(common + ["--no-drift"], capsys)
    assert code == flag_code == 1
    # Only the header says the rule is on
    extra = [line for line in flag_out.splitlines() if line not in out.splitlines()]
    assert extra == ["  No drift:      on (a cell that would be DRIFT is FAIL)"]
    assert [line for line in flag_out.splitlines() if line not in extra] == out.splitlines()
    assert "  PASS  [p] Q1 SF=100" in out.splitlines()
    assert "  FAIL  [p] Q3 SF=100" in out.splitlines()
    assert "  SKIP  [p] Q5 SF=100" in out.splitlines()


def test_without_no_drift_the_header_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_csv(tmp_path / "ref" / "Q1_sf10.csv", ["total", "5.00"])
    write_csv(tmp_path / "cand" / "p" / "Q1_sf10.csv", ["total", "5.00"])
    _, out = run_main(["--shilmandb-dir", str(tmp_path / "cand"), "--reference-dir", str(tmp_path / "ref"),
                       "--sf", "10", "--queries", "Q1", "--configs", "p"], capsys)
    assert out.splitlines()[:9] == [
        "ShilmanDB vs SQLite Result Comparison",
        f"  ShilmanDB dir: {tmp_path / 'cand'}",
        f"  SQLite dir:    {tmp_path / 'ref'}",
        "  Configs:       ['p']",
        "  Scale factors: ['10']",
        "  Queries:       ['Q1']",
        "  Tolerance:     0.01",
        "",
        "  PASS  [p] Q1 SF=10",
    ]
