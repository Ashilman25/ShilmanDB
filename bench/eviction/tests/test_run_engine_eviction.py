"""Unit tests for bench/eviction/engine/run_engine_eviction.py. Nothing here runs load_tpch."""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "engine" / "run_engine_eviction.py"
spec = importlib.util.spec_from_file_location("run_engine_eviction", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["run_engine_eviction"] = mod
spec.loader.exec_module(mod)

TRI = (0.1, 0.25, 0.5)


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def make_paths(root: Path = Path("X")) -> "mod.Paths":
    return mod.Paths(binary=Path("build/bench/load_tpch"), out=root / "o", sim=root / "s", data_dir=Path("bench/tpch_data"), db_dir=root / "db")


def plan(sf: str, step: str, *extra: str) -> list:
    return mod.plan_runs(mod.parse_args(["--sf", sf, *extra]), step)


def cmd_of(run, job: int = 0) -> list[str]:
    return mod.run_cmd(run, make_paths(), job)


def read_kv(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["key"]: r["value"] for r in csv.DictReader(f)}


def fake_binary(tmp_path: Path, text: str = "fake") -> Path:
    b = tmp_path / "load_tpch"
    b.write_text(text)
    return b


def no_runs(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    return calls


def test_cached_sf001_is_560_runs():
    runs = plan("0.01", "cached")
    assert len(runs) == 560
    assert len({r.spec.name for r in runs}) == 14
    assert {r.pool for r in runs} == set(mod.grid_pools("0.01")) and len(mod.grid_pools("0.01")) == 8
    assert {r.policy for r in runs} == set(mod.POLICIES) == {"lru", "clock", "lru2", "arc", "lirs"}
    for r in runs:
        cmd = cmd_of(r)
        assert flag(cmd, "--warmup") == "1" and flag(cmd, "--runs") == "2"
        assert flag(cmd, "--eviction-policy") == r.policy
        assert flag(cmd, "--pool-size") == str(r.pool)
        assert flag(cmd, "--workload") == r.spec.workload and flag(cmd, "--seed") == str(r.spec.seed)
        assert flag(cmd, "--results-dir") == f"X/o/sf0.01/cached/{r.spec.name}/{r.policy}/pool{r.pool}/"
        if r.spec.kind.startswith("index100"):
            assert flag(cmd, "--planner-read-us") == "100"
        else:
            assert "--planner-read-us" not in cmd
        for absent in ("--miss-penalty-us", "--policy-timing", "--keep-load-pool", "--eviction-model-path", "--enable-tracing"):
            assert absent not in cmd
    # plan-check traces exist for seed 1 only
    assert {r.spec.name for r in runs if r.spec.kind in mod.PLAN_CHECK_KINDS} == {"index100_stream_s1", "index100_mixed_s1"}


def test_cached_sf01_and_sf1():
    runs = plan("0.1", "cached")
    assert len(runs) == 190
    main = [r for r in runs if r.spec.kind in mod.MAIN_KINDS]
    checks = [r for r in runs if r.spec.kind in mod.PLAN_CHECK_KINDS]
    assert len(main) == 160 and {r.pool for r in main} == set(mod.grid_pools("0.1"))
    assert len(checks) == 30 and {r.pool for r in checks} == {mod.pool_for_fraction("0.1", f) for f in TRI}
    assert {r.spec.seed for r in runs} == {1}

    args = mod.parse_args(["--sf", "1.0"])
    runs = mod.plan_runs(args, "cached")
    assert len(runs) == 60
    assert {r.spec.kind for r in runs} == set(mod.MAIN_KINDS)
    assert {r.pool for r in runs} == {mod.pool_for_fraction("1.0", f) for f in TRI}
    assert mod.jobs_for(args, "cached") == 4 and mod.jobs_for(args, "timing") == 1
    assert mod.jobs_for(mod.parse_args(["--sf", "0.1"]), "cached") == 1
    assert mod.jobs_for(mod.parse_args(["--sf", "1.0", "--jobs", "2"]), "cached") == 2


def test_penalty_runs():
    runs = plan("0.01", "penalty")
    assert len(runs) == 60
    assert {r.pool for r in runs} == {mod.pool_for_fraction("0.01", f) for f in TRI}
    assert {r.spec.kind for r in runs} == set(mod.MAIN_KINDS) and {r.spec.seed for r in runs} == {1}
    for r in runs:
        cmd = cmd_of(r)
        assert flag(cmd, "--miss-penalty-us") == "100" and flag(cmd, "--runs") == "2"
        assert "--planner-read-us" not in cmd  # the planner follows the penalty by itself

    runs = plan("0.1", "penalty")
    assert len(runs) == 12
    assert {r.policy for r in runs} == {"lru", "arc", "lirs"}
    assert {r.pool for r in runs} == {mod.pool_for_fraction("0.1", 0.25)}
    assert all(flag(cmd_of(r), "--runs") == "1" and flag(cmd_of(r), "--miss-penalty-us") == "100" for r in runs)

    assert "penalty" not in mod.parse_args(["--sf", "1.0"]).steps
    optional = plan("1.0", "penalty", "--steps", "penalty")
    assert sorted((r.spec.kind, r.policy) for r in optional) == [
        ("index_mixed", "arc"), ("index_mixed", "lru"), ("index_stream", "lirs"), ("index_stream", "lru")]
    assert {r.pool for r in optional} == {mod.pool_for_fraction("1.0", 0.25)}
    assert all(flag(cmd_of(r), "--runs") == "1" for r in optional)


def test_timing_runs():
    for sf, count in (("0.01", 40), ("0.1", 20), ("1.0", 5)):
        runs = plan(sf, "timing")
        assert len(runs) == count
        assert all("--policy-timing" in cmd_of(r) and "--miss-penalty-us" not in cmd_of(r) for r in runs)
        assert {r.policy for r in runs} == set(mod.POLICIES)
    assert {r.pool for r in plan("0.01", "timing")} == {128, mod.pool_for_fraction("0.01", 0.25)}
    assert {r.spec.name for r in plan("1.0", "timing")} == {"index_stream_s1"}


def test_learned_and_warm_sf001_only():
    runs = plan("0.01", "learned")
    v2 = [r for r in runs if r.policy == "learned-v2"]
    v1 = [r for r in runs if r.policy == "learned-v1"]
    assert len(runs) == 5 and len(v2) == 4 and len(v1) == 1
    assert {r.pool for r in v2} == {72, 128}
    assert {r.spec.name for r in v2} == {"baseline_stream_s1", "index_lookup_s1"}
    for r in v2:
        cmd = cmd_of(r)
        assert flag(cmd, "--eviction-policy") == "learned-v2"
        assert flag(cmd, "--eviction-model-path") == str(mod.REPO_ROOT / "ml/eviction_policy/models/eviction_model_v2.pt")
    cmd = cmd_of(v1[0])
    assert flag(cmd, "--eviction-policy") == "learned-v1"
    assert flag(cmd, "--eviction-model-path") == str(mod.REPO_ROOT / "ml/eviction_policy/models/eviction_model.pt")
    assert flag(cmd, "--results-dir").endswith("/learned/baseline_stream_s1/learned-v1/pool72/")

    warm = plan("0.01", "warm")
    assert len(warm) == 64
    assert {r.policy for r in warm} == {"clock", "lru2", "arc", "lirs"}
    assert {r.spec.name for r in warm} == {"baseline_stream_s1", "index_lookup_s1"}
    assert {r.pool for r in warm} == set(mod.grid_pools("0.01"))
    assert all("--keep-load-pool" in cmd_of(r) for r in warm)
    assert all("--keep-load-pool" not in cmd_of(r) for r in plan("0.01", "cached"))

    for sf in ("0.1", "1.0"):
        steps = mod.parse_args(["--sf", sf]).steps
        assert "learned" not in steps and "warm" not in steps and "check" in steps and "analyze" in steps
        for step in ("learned", "warm"):
            with pytest.raises(SystemExit):
                mod.parse_args(["--sf", sf, "--steps", step])
    assert mod.parse_args(["--sf", "0.01"]).steps == mod.ALL_STEPS


def test_check_command(monkeypatch, capsys, tmp_path):
    p = make_paths(tmp_path)
    links = [tmp_path / "o/sf0.1/check_folders/a", tmp_path / "o/sf0.1/check_folders/b"]
    cmd = mod.check_cmd("0.1", p, links)
    assert cmd[1].endswith("bench/tpch/check_stream_results.py")
    assert flag(cmd, "--sf") == "0.1" and flag(cmd, "--data-dir") == "bench/tpch_data"
    assert flag(cmd, "--output") == str(tmp_path / "o/sf0.1/stream_correctness.csv")
    assert cmd[-2:] == [str(x) for x in links]
    assert mod.check_link_name(Path("/r/sf0.1/penalty/index_mixed_s1/lirs/pool3604")) == "penalty-index_mixed_s1-lirs-pool3604"

    # Dry run: one folder per planned run, named so they never collide
    out = tmp_path / "dry"
    assert mod.main(["--sf", "0.1", "--steps", "penalty,check", "--dry-run", "--out-dir", str(out)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if "check_stream_results.py" in l)
    assert f"--output {out}/sf0.1/stream_correctness.csv" in line
    assert line.count(f"{out}/sf0.1/check_folders/penalty-") == 12

    # Real run: every finished run folder of this SF on disk, nothing else
    calls = no_runs(monkeypatch)
    out = tmp_path / "real"
    done = []
    for step, trace, policy, pool, sf, finished in (
        ("cached", "baseline_stream_s1", "lirs", 362, "0.01", True),
        ("warm", "index_lookup_s1", "arc", 72, "0.01", True),
        ("cached", "baseline_stream_s1", "arc", 362, "0.01", False),  # no driver_info.csv: never finished
        ("cached", "baseline_stream_s1", "lru", 512, "0.1", True),
    ):
        folder = out / f"sf{sf}" / step / trace / policy / f"pool{pool}"
        folder.mkdir(parents=True)
        for name in ("stream_queries.csv", "stream_results.csv") + (("driver_info.csv",) if finished else ()):
            (folder / name).write_text("x\n")
        if finished and sf == "0.01":
            done.append(folder)
    assert mod.main(["--sf", "0.01", "--steps", "check", "--out-dir", str(out)]) == 0
    assert len(calls) == 1
    cmd = calls[0][0]
    assert flag(cmd, "--output") == str(out / "sf0.01/stream_correctness.csv")
    given = [Path(c) for c in cmd[cmd.index("--output") + 2:]]
    assert len(given) == 2
    assert {g.name for g in given} == {"cached-baseline_stream_s1-lirs-pool362", "warm-index_lookup_s1-arc-pool72"}
    assert {g.resolve() for g in given} == {f.resolve() for f in done}


def test_check_without_folders_runs_nothing(monkeypatch, capsys, tmp_path):
    calls = no_runs(monkeypatch)
    assert mod.main(["--sf", "0.01", "--steps", "check", "--out-dir", str(tmp_path / "o")]) == 0
    assert calls == [] and "nothing to check" in capsys.readouterr().out


def test_db_file_per_job_and_driver_info(monkeypatch, capsys, tmp_path):
    seen = []

    def fake_run(cmd, **kw):
        db = Path(flag(cmd, "--db-file"))
        seen.append((db.name, db.exists()))
        db.write_text("db")  # what load_tpch leaves behind
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    base = ["--sf", "0.01", "--steps", "cached", "--kinds", "baseline_stream", "--seeds", "1", "--policies", "lru,arc,lirs",
            "--fractions", "0.25", "--binary", str(fake_binary(tmp_path)), "--out-dir", str(tmp_path / "o"), "--db-dir", str(tmp_path / "db")]
    assert mod.main(base + ["--jobs", "2"]) == 0
    assert len(seen) == 3
    assert {name for name, _ in seen} <= {"eve_sf0.01_0.db", "eve_sf0.01_1.db"}
    assert not any(existed for _, existed in seen)       # removed before each run
    assert not list((tmp_path / "db").glob("*.db"))      # and after
    folders = [tmp_path / "o/sf0.01/cached/baseline_stream_s1" / p / "pool362" for p in ("lru", "arc", "lirs")]
    for folder in folders:
        info = read_kv(folder / "driver_info.csv")
        assert info["jobs"] == "2" and info["step"] == "cached"

    seen.clear()
    assert mod.main(base + ["--redo"]) == 0
    assert [name for name, _ in seen] == ["eve_sf0.01_0.db"] * 3
    assert all(read_kv(f / "driver_info.csv")["jobs"] == "1" for f in folders)
    assert "3 runs, 1 at a time" in capsys.readouterr().out


def test_dry_run_runs_nothing(monkeypatch, capsys, tmp_path):
    calls = no_runs(monkeypatch)
    out = tmp_path / "o"
    assert mod.main(["--sf", "0.01", "--dry-run", "--out-dir", str(out), "--db-dir", str(tmp_path / "db")]) == 0
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert calls == []
    assert not out.exists() and not (tmp_path / "db").exists()
    engine = [l for l in lines if l.startswith("$ ") and "--eviction-policy" in l]
    assert len(engine) == 560 + 60 + 40 + 5 + 64
    assert all(f"--db-file {tmp_path}/db/eve_sf0.01_0.db" in l for l in engine)
    assert sum("check_stream_results.py" in l for l in lines if l.startswith("$ ")) == 1
    assert sum("analyze_engine_eviction.py" in l for l in lines if l.startswith("$ ")) == 1
    for step in mod.ALL_STEPS:
        assert f"== {step}: start" in text and f"== {step}: done in" in text
    assert "== total:" in text
    assert "(560 runs, 1 at a time)" in text


def test_dry_run_with_jobs_shows_each_database(capsys, tmp_path):
    assert mod.main(["--sf", "1.0", "--steps", "cached", "--dry-run", "--out-dir", str(tmp_path / "o"), "--db-dir", "/tmp"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("$ ")]
    assert len(lines) == 60
    assert {l.split("--db-file ")[1].split()[0] for l in lines} == {f"/tmp/eve_sf1.0_{j}.db" for j in range(4)}


def test_fractions_override():
    runs = plan("0.01", "cached", "--fractions", "0.25")
    assert len(runs) == 14 * 5 and {r.pool for r in runs} == {362}
    assert {r.pool for r in plan("0.01", "timing", "--fractions", "0.25")} == {362}
    assert {r.pool for r in plan("0.01", "learned", "--fractions", "0.25")} == {362}
    assert {r.pool for r in plan("0.1", "cached", "--fractions", "default,0.1")} == {512, 1442}
    assert len(plan("0.1", "cached", "--fractions", "default,0.1")) == (4 + 2) * 2 * 5
    for bad in ("0", "1.5", "x", ","):
        with pytest.raises(SystemExit):
            mod.parse_args(["--sf", "0.01", "--fractions", bad])


def test_policies_kinds_and_seeds_options():
    runs = plan("0.01", "cached", "--policies", "lirs")
    assert len(runs) == 14 * 8 and {r.policy for r in runs} == {"lirs"}
    assert {r.policy for r in plan("0.01", "learned", "--policies", "lirs")} == {"learned-v2", "learned-v1"}
    assert {r.policy for r in plan("0.1", "penalty", "--policies", "lru,clock")} == {"lru", "clock"}
    runs = plan("0.01", "cached", "--kinds", "index_lookup")
    assert len(runs) == 3 * 8 * 5 and {r.spec.kind for r in runs} == {"index_lookup"}
    runs = plan("0.1", "cached", "--seeds", "2")
    assert len(runs) == 160 and {r.spec.seed for r in runs} == {2}  # the plan-check traces have only seed 1
    for bad in (["--policies", "foo"], ["--policies", "learned-v2"], ["--kinds", "foo"], ["--steps", "record"], ["--jobs", "0"]):
        with pytest.raises(SystemExit):
            mod.parse_args(["--sf", "0.01", *bad])


def test_failure_stops_with_exit_code(monkeypatch, capsys, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 3))
    base = ["--sf", "0.01", "--steps", "cached", "--kinds", "baseline_stream", "--seeds", "1", "--fractions", "0.25",
            "--binary", str(fake_binary(tmp_path)), "--out-dir", str(tmp_path / "o"), "--db-dir", str(tmp_path / "db")]
    with pytest.raises(SystemExit) as stop:
        mod.main(base)
    assert stop.value.code == 3 and len(calls) == 1
    assert not list((tmp_path / "o").rglob("driver_info.csv"))
    with pytest.raises(SystemExit) as stop:
        mod.main(base + ["--jobs", "2"])
    assert stop.value.code == 3


def test_check_fail_stops_the_driver(monkeypatch, capsys, tmp_path):
    folder = tmp_path / "o/sf0.01/cached/baseline_stream_s1/lru/pool72"
    folder.mkdir(parents=True)
    for name in ("driver_info.csv", "stream_queries.csv", "stream_results.csv"):
        (folder / name).write_text("x\n")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 1))
    with pytest.raises(SystemExit) as stop:
        mod.main(["--sf", "0.01", "--steps", "check,analyze", "--out-dir", str(tmp_path / "o")])
    assert stop.value.code == 1 and len(calls) == 1  # analyze never ran


def test_missing_binary_fails_before_any_run(monkeypatch, capsys, tmp_path):
    calls = no_runs(monkeypatch)
    rc = mod.main(["--sf", "0.01", "--steps", "cached", "--binary", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")])
    assert rc == 1 and calls == [] and "no load_tpch binary" in capsys.readouterr().err


def write_complete(folder: Path, run, binary: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run_info.csv").write_text(f"key,value\npool_size,{run.pool}\neviction,{mod.RUN_INFO_NAME[run.policy]}\n")
    (folder / "stream_totals.csv").write_text("pass,warmup\n0,1\n1,0\n2,0\n")
    for name in ("stream.csv", "stream_queries.csv", "stream_results.csv"):
        (folder / name).write_text("x\n")
    mod.write_driver_info(folder, run.step, 1, binary, mod.file_sha256(binary))


def test_skips_complete_runs(monkeypatch, capsys, tmp_path):
    calls = no_runs(monkeypatch)
    binary = fake_binary(tmp_path)
    base = ["--sf", "0.01", "--steps", "cached", "--kinds", "baseline_stream", "--seeds", "1", "--policies", "lirs",
            "--fractions", "0.25", "--binary", str(binary), "--out-dir", str(tmp_path / "o"), "--db-dir", str(tmp_path / "db")]
    run = plan("0.01", "cached", *base[2:])[0]
    folder = mod.run_folder(run, make_paths(tmp_path))
    write_complete(folder, run, binary)

    assert mod.main(base) == 0
    assert calls == [] and "skip" in capsys.readouterr().out
    assert mod.main(base + ["--redo"]) == 0 and len(calls) == 1

    write_complete(folder, run, binary)
    binary.write_text("a different build")
    assert mod.main(base) == 0 and len(calls) == 2  # built again: run again

    write_complete(folder, run, binary)
    (folder / "stream_totals.csv").write_text("pass,warmup\n0,1\n1,0\n")  # a pass short
    assert mod.main(base) == 0 and len(calls) == 3


def test_defaults_work_from_any_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = mod.parse_args(["--sf", "0.01"])
    for p in (args.binary, args.out_dir, args.sim_dir, args.data_dir):
        assert p.is_absolute() and str(p).startswith(str(mod.REPO_ROOT))
    assert args.db_dir == Path("/tmp")
    assert all(str(m).startswith(str(mod.REPO_ROOT)) for m in mod.MODELS.values())
    assert str(mod.check_cmd("0.01", make_paths(), [])[1]).startswith(str(mod.REPO_ROOT))
