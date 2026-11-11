"""Unit tests for bench/headroom/run_switch_check.py. Nothing here runs a binary."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).with_name("run_switch_check.py")
spec = importlib.util.spec_from_file_location("run_switch_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_switch_check"] = mod
spec.loader.exec_module(mod)


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def make_paths():
    return mod.Paths(tool=Path("build/bench/eviction_switch"), python=Path("py"), traces=Path("T"),
                     sim_dir=Path("S5"), out=Path("O"))


def write_layout(path: Path, n: int = 4, runs: int = 1000) -> None:
    lines = ["trace,phase,component,first_run,runs"]
    lines += [f"x,{i},c{i},{i * runs},{runs}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n")


def test_pools_match_recorded_pools():
    assert mod.pools("0.01") == [145, 362, 724]
    assert mod.pools("0.1") == [1442, 3604, 7208]


def test_sf_1_is_refused():
    with pytest.raises(SystemExit):
        mod.main(["--sf", "1.0", "--dry-run"])


def test_phases_components():
    assert mod.PHASES[3] == ("index_stream_s3", "index_lookup_s3", "index_mixed_s3", "index_stream_s1")
    assert mod.phases_name(1) == "phases_s1"


def test_window_for():
    assert mod.window_for(1_000_000, 64) == 15_624
    assert mod.window_for(1_000_000, 32) == 31_248
    assert mod.window_for(1_000_000, 4) == 250_000
    assert all(mod.window_for(1_234_567, k) % 4 == 0 for k in mod.WINDOW_KS)


def test_mean_phase_length_needs_12_phases(tmp_path):
    files = []
    for k in (1, 2, 3):
        f = tmp_path / f"l{k}.csv"
        write_layout(f, runs=1000 * k)
        files.append(f)
    assert mod.mean_phase_length(files) == 2000
    with pytest.raises(ValueError):
        mod.mean_phase_length(files[:2])
    with pytest.raises(ValueError, match="missing.csv"):
        mod.mean_phase_length([files[0], files[1], tmp_path / "missing.csv"])
    write_layout(files[0], n=3)
    with pytest.raises(ValueError):
        mod.mean_phase_length(files)


def test_layout_with_missing_files_exits_1(tmp_path, monkeypatch, capsys):
    tool = tmp_path / "tool"
    tool.write_text("")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    code = mod.main(["--sf", "0.01", "--steps", "layout", "--tool", str(tool), "--out-dir", str(tmp_path / "out")])
    assert code == 1
    assert "cannot read layout file" in capsys.readouterr().err


def test_write_windows(tmp_path):
    w = mod.write_windows(tmp_path / "windows.csv", 1_000_000)
    assert w[64] == 15_624 and set(w) == set(mod.WINDOW_KS)
    lines = (tmp_path / "windows.csv").read_text().splitlines()
    assert lines[0] == "window_k,window,mean_phase_runs"
    assert len(lines) == 6


def test_hedge_settings():
    pairs = mod.hedge_settings("0.01", None)
    assert len(pairs) == 12 and (5, 0) in pairs and (20, 0.1) in pairs
    chosen = {"hedge": {"window_k": 16, "eta": 10}, "fixed_share": {"window_k": 8, "eta": 5, "alpha": 0.03},
              "two_window": {"window_k": 16}}
    assert mod.hedge_settings("0.1", chosen) == [(10, 0), (5, 0.03)]


def test_replay_cmd_flags():
    w = {k: mod.window_for(1_000_000, k) for k in mod.WINDOW_KS}
    cmd = mod.replay_cmd("0.01", make_paths(), 1, w, [(5, 0), (10, 0.03)], None)
    assert flag(cmd, "--windows") == "64:15624,32:31248,16:62500,8:125000,4:250000"
    assert flag(cmd, "--hedge") == "5:0,10:0.03"
    assert flag(cmd, "--margin") == "2" and flag(cmd, "--threshold") == "0.05"
    assert flag(cmd, "--start") == "lirs_visit"
    assert flag(cmd, "--pools") == "145,362,724"
    assert flag(cmd, "--components") == "index_stream_s1,index_lookup_s1,index_mixed_s1,index_stream_s2"
    assert flag(cmd, "--traces-dir") == "T/sf0.01"
    assert flag(cmd, "--out-prefix") == "O/sf0.01/phases_s1"
    assert "--jobs" not in cmd
    assert flag(mod.replay_cmd("0.01", make_paths(), 1, w, [(5, 0)], 4), "--jobs") == "4"


def test_other_cmds():
    d = mod.describe_cmd("0.01", make_paths(), 2)
    assert "--describe" in d and "--pools" not in d
    v = mod.validate_cmd("0.01", make_paths(), 3)
    assert "--validate" in v and flag(v, "--pools") == "145,362,724"
    a = mod.analyze_cmd("0.1", make_paths())
    assert flag(a, "--sf") == "0.1" and flag(a, "--out") == "O" and flag(a, "--sim-dir") == "S5"


def run_main(tmp_path, *extra):
    tool = tmp_path / "tool"
    return mod.main(["--sf", "0.1", "--steps", "replay", "--dry-run", "--tool", str(tool),
                     "--out-dir", str(tmp_path / "out"), *extra])


def test_sf01_replay_refuses_missing_or_bad_chosen(tmp_path):
    assert run_main(tmp_path) == 1
    out = tmp_path / "out"
    out.mkdir()
    (out / "chosen.json").write_text(json.dumps({"sf": "0.1", "families": {}}))
    assert run_main(tmp_path) == 1


def test_replay_needs_windows(tmp_path, capsys):
    assert mod.main(["--sf", "0.01", "--steps", "replay", "--dry-run", "--tool", str(tmp_path / "t"),
                     "--out-dir", str(tmp_path / "out")]) == 0
    assert "<from windows.csv>" in capsys.readouterr().out


def test_dry_run_runs_nothing(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("subprocess.run called")

    monkeypatch.setattr(subprocess, "run", boom)
    code = mod.main(["--sf", "0.01", "--dry-run", "--tool", str(tmp_path / "t"), "--out-dir", str(tmp_path / "out")])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("$ ") >= 3 + 3 + 3 + 1
    assert not (tmp_path / "out").exists()


def test_missing_tool_refused(tmp_path):
    assert mod.main(["--sf", "0.01", "--steps", "validate", "--tool", str(tmp_path / "nope"),
                     "--out-dir", str(tmp_path / "out")]) == 1


def test_sf0_1_chosen_missing_refuses_before_any_step(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("subprocess.run called")

    monkeypatch.setattr(subprocess, "run", boom)
    tool = tmp_path / "t"
    tool.write_text("")
    for extra in ([], ["--dry-run"]):
        code = mod.main(["--sf", "0.1", "--steps", "layout,validate,replay", "--tool", str(tool),
                         "--out-dir", str(tmp_path / "out"), *extra])
        assert code == 1
        assert "chosen.json" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_replay_clears_old_results_before_running(tmp_path, monkeypatch):
    out = tmp_path / "out" / "sf0.01"
    out.mkdir(parents=True)
    rows = "".join(f"{k},{1600 // k},1600\n" for k in mod.WINDOW_KS)
    (out / "windows.csv").write_text("window_k,window,mean_phase_runs\n" + rows)
    stale = [out / f"phases_s{k}_{kind}.csv" for k in (1, 2, 3) for kind in mod.REPLAY_OUTPUTS]
    for f in stale:
        f.write_text("old\n")
    tool = tmp_path / "t"
    tool.write_text("")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    with pytest.raises(SystemExit):
        mod.main(["--sf", "0.01", "--steps", "replay", "--tool", str(tool), "--out-dir", str(tmp_path / "out")])
    assert not any(f.exists() for f in stale)
    assert (out / "windows.csv").exists()
