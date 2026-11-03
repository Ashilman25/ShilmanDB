"""Unit tests for bench/headroom/run_eviction_headroom.py. Nothing here runs load_tpch or eviction_sim."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).with_name("run_eviction_headroom.py")
spec = importlib.util.spec_from_file_location("run_eviction_headroom", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_eviction_headroom"] = mod
spec.loader.exec_module(mod)

BENCH = Path(__file__).resolve().parents[1]
spec_rb = importlib.util.spec_from_file_location("run_benchmarks_for_test", BENCH / "run_benchmarks.py")
rb = importlib.util.module_from_spec(spec_rb)
sys.modules["run_benchmarks_for_test"] = rb
spec_rb.loader.exec_module(rb)


def make_paths(tmp_path: Path | None = None) -> "mod.Paths":
    return mod.Paths(
        binary=Path("build/bench/load_tpch"), sim=Path("build/bench/eviction_sim"),
        traces=Path("T"), out=Path("O"), data_dir=Path("bench/tpch_data"), db_file=Path("/tmp/evh.db"),
    )


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_names_for_every_kind_and_seed():
    assert mod.TraceSpec("0.01", "index_lookup", 2).name == "index_lookup_s2"
    assert mod.TraceSpec("0.01", "baseline_stream", 1).name == "baseline_stream_s1"
    s = mod.TraceSpec("0.01", "index100_mixed", 1)
    assert (s.workload, s.indexes, s.planner_read_us) == ("mixed", True, 100)
    s = mod.TraceSpec("0.01", "baseline_stream", 1)
    assert (s.workload, s.indexes, s.planner_read_us) == ("stream", False, None)


def test_grid_pools():
    assert mod.grid_pools("0.01") == [72, 128, 145, 362, 724, 1086, 1303, 1448]
    for sf in ("0.1", "1.0"):
        want = sorted({rb.pool_for_fraction(sf, f) for f in mod.FRACTIONS} | {rb.POOL_SIZES[sf]})
        assert mod.grid_pools(sf) == want


def test_trace_specs_counts():
    assert len(mod.trace_specs("0.01", [1, 2, 3], None)) == 14
    assert len(mod.trace_specs("0.1", [1, 2, 3], None)) == 14
    assert len(mod.trace_specs("1.0", [1], None)) == 6
    only = mod.trace_specs("0.01", [1], ["index_lookup"])
    assert [s.name for s in only] == ["index_lookup_s1"]


def test_record_cmd():
    p = make_paths()
    cmd = mod.record_cmd(mod.TraceSpec("0.01", "baseline_stream", 1), p)
    assert flag(cmd, "--create-indexes") == "false"
    assert flag(cmd, "--warmup") == "1" and flag(cmd, "--runs") == "2"
    assert "--trace-collapse" in cmd and "--planner-read-us" not in cmd
    assert flag(cmd, "--pool-size") == "128"
    assert flag(cmd, "--enable-tracing") == "T/sf0.01/baseline_stream_s1.csv"
    assert flag(cmd, "--results-dir") == "T/sf0.01/baseline_stream_s1/"
    assert flag(cmd, "--workload") == "stream" and flag(cmd, "--seed") == "1"
    idx = mod.record_cmd(mod.TraceSpec("0.01", "index_lookup", 2), p)
    assert flag(idx, "--create-indexes") == "true" and flag(idx, "--seed") == "2"
    chk = mod.record_cmd(mod.TraceSpec("0.01", "index100_stream", 1), p)
    assert flag(chk, "--planner-read-us") == "100"


def test_validate_cmd():
    cmd = mod.validate_cmd(mod.TraceSpec("0.1", "index_mixed", 1), 1442, make_paths())
    assert "--enable-tracing" not in cmd and "--trace-collapse" not in cmd
    assert flag(cmd, "--pool-size") == "1442"
    assert flag(cmd, "--results-dir") == "O/sf0.1/validate/index_mixed_s1/pool1442/"


def test_sim_cmd():
    p = make_paths()
    s = mod.TraceSpec("0.01", "index_stream", 1)
    cmd = mod.sim_cmd(s, p, 1448)
    assert flag(cmd, "--v2-weights") == "O/v2_weights.txt" and flag(cmd, "--v2-max-frames") == "1448"
    assert "--raw" not in cmd
    assert flag(cmd, "--pools") == "72,128,145,362,724,1086,1303,1448"
    assert flag(cmd, "--out-prefix") == "O/sf0.01/index_stream_s1"
    assert flag(cmd, "--items") == "T/sf0.01/index_stream_s1/stream.csv"
    assert "--v2-weights" not in mod.sim_cmd(s, p, 0)
    assert "--raw" in mod.sim_cmd(s, p, 0, raw=True)
    assert flag(mod.sim_cmd(s, p, 0, jobs=4), "--jobs") == "4"


def test_window_sim_cmd():
    s = mod.TraceSpec("0.01", "baseline_stream", 1)
    cmd = mod.sim_cmd(s, make_paths(), 1448, window=8, policies="lru2_visit,arc_visit,lirs_visit")
    assert flag(cmd, "--visit-window") == "8"
    assert flag(cmd, "--out-prefix") == "O/sf0.01/baseline_stream_s1_w8"
    assert flag(cmd, "--name") == "baseline_stream_s1_w8"
    assert flag(cmd, "--policies") == "lru2_visit,arc_visit,lirs_visit"
    assert "--v2-weights" not in cmd


def test_v2check_cmd():
    cmd = mod.v2check_cmd(mod.TraceSpec("0.01", "index_lookup", 1), 72, make_paths())
    assert "--use-learned-eviction" in cmd
    assert flag(cmd, "--eviction-model-path") == str(mod.REPO_ROOT / "ml/eviction_policy/models/eviction_model_v2.pt")
    assert flag(cmd, "--results-dir") == "O/sf0.01/v2check/index_lookup_s1/pool72/"
    assert "--enable-tracing" not in cmd


def test_defaults():
    d = mod.DEFAULTS
    assert d["0.01"]["v2_cap"] == 1448 and d["0.01"]["validate_seeds"] == [1, 2, 3]
    assert d["0.1"]["v2_cap"] == 2048 and d["0.1"]["validate_seeds"] == [1]
    assert d["1.0"]["v2_cap"] == 0 and d["1.0"]["validate_fractions"] == [0.1, 0.5, 0.9]


def test_default_steps():
    assert "fullcheck" in mod.default_steps("0.01") and "v2check" in mod.default_steps("0.01")
    assert "fullcheck" not in mod.default_steps("0.1") and "v2check" not in mod.default_steps("0.1")


def run_main(monkeypatch, capsys, argv, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    rc = mod.main(argv + ["--traces-dir", str(tmp_path / "t"), "--out-dir", str(tmp_path / "o")])
    return rc, calls, capsys.readouterr().out


def test_dry_run_runs_nothing(monkeypatch, capsys, tmp_path):
    rc, calls, out = run_main(monkeypatch, capsys, ["--sf", "0.01", "--dry-run"], tmp_path)
    assert rc == 0 and calls == []
    assert out.count("--enable-tracing") == 15  # 14 recordings + the uncollapsed one
    assert "--visit-window 8" in out and "== record: start" in out
    assert not (tmp_path / "t").exists()


def test_failure_stops_with_exit_code(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 3))
    with pytest.raises(SystemExit) as stop:
        mod.main(["--sf", "0.1", "--steps", "record", "--traces-dir", str(tmp_path / "t"), "--out-dir", str(tmp_path / "o")])
    assert stop.value.code == 3


def test_collapse_compare(tmp_path):
    full, col = tmp_path / "full.csv", tmp_path / "col.csv"
    full.write_text("timestamp,page_id\n1,5\n2,5\n3,6\n4,6\n5,5\n")
    col.write_text("timestamp,page_id\n1,5\n3,6\n5,5\n")
    assert mod.compare_collapsed(full, col) == "identical"
    col.write_text("timestamp,page_id\n1,5\n3,7\n5,5\n")
    msg = mod.compare_collapsed(full, col)
    assert msg.startswith("differs at line 3")
    col.write_text("timestamp,page_id\n1,5\n3,6\n")
    assert mod.compare_collapsed(full, col) != "identical"


def test_failed_collapse_check_keeps_the_full_trace(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(mod, "compare_collapsed", lambda full, col: "differs at line 7")
    full = tmp_path / "t" / "sf0.01" / f"{mod.FULLCHECK_TRACE}_full.csv"
    full.parent.mkdir(parents=True)
    full.write_text("timestamp,page_id\n")
    rc = mod.main(["--sf", "0.01", "--steps", "fullcheck", "--traces-dir", str(tmp_path / "t"), "--out-dir", str(tmp_path / "o")])
    assert rc == 0 and full.exists() and "the collapse check failed" in capsys.readouterr().out
    verdict = tmp_path / "o" / "sf0.01" / "fullcheck" / "collapse_check.txt"
    assert verdict.read_text() == "differs at line 7\n"


def test_defaults_work_from_any_directory(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = mod.parse_args(["--sf", "0.01"])
    for p in (args.binary, args.sim, args.traces_dir, args.out_dir, args.data_dir):
        assert p.is_absolute() and str(p).startswith(str(mod.REPO_ROOT))
    assert mod.V2_MODEL.startswith(str(mod.REPO_ROOT))


def _dry_run(capsys, sf):
    assert mod.main(["--sf", sf, "--dry-run"]) == 0
    return capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("sf", ["0.1", "1.0"])
def test_dry_run_records_plan_check_traces_at_every_sf(capsys, sf):
    lines = _dry_run(capsys, sf)
    records = [l for l in lines if l.startswith("$ ") and "--enable-tracing" in l and "_full" not in l]
    assert len(records) == 14
    checks = [l for l in records if "--planner-read-us 100" in l]
    assert len(checks) == 2
    assert any("index100_stream_s1.csv" in l and "--workload stream" in l for l in checks)
    assert any("index100_mixed_s1.csv" in l and "--workload mixed" in l for l in checks)


def test_dry_run_sf1_validates_three_fractions_without_v2(capsys):
    lines = _dry_run(capsys, "1.0")
    validates = [l for l in lines if l.startswith("$ ") and "--enable-tracing" not in l and "load_tpch" in l]
    # 4 main kinds, seed 1, fractions 0.1 / 0.5 / 0.9
    assert len(validates) == 12
    pools = {mod.pool_for_fraction("1.0", f) for f in (0.1, 0.5, 0.9)}
    assert {int(l.split("--pool-size ")[1].split()[0]) for l in validates} == pools
    assert not any("--v2-weights" in l for l in lines)
