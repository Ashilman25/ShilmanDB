"""Tests for the eviction_switch command line. Skipped if it isn't built."""
import csv
import subprocess
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[3] / "build" / "bench" / "eviction_switch"

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="build/bench/eviction_switch is not built")

PREFIX = [0, 1, 2, 3]
BODIES = {
    "a": [4 + (i * i + i // 3) % 6 for i in range(32)],
    "b": [(i * 7 + i // 4) % 10 for i in range(32)],
}
# a page never repeats back to back, so the trace has one run per fetch
for _body in BODIES.values():
    for _i in range(1, len(_body)):
        if _body[_i] == _body[_i - 1]:
            _body[_i] = (_body[_i] + 1) % 10


def write_component(directory, name, prefix=PREFIX):
    pages = prefix + BODIES[name]
    n = len(prefix)
    (directory / f"{name}.csv").write_text("timestamp,page_id\n" + "".join(f"{i},{p}\n" for i, p in enumerate(pages)))
    (directory / name).mkdir(exist_ok=True)
    (directory / name / "stream.csv").write_text(
        "pass,warmup,seq,kind,query,trace_start,trace_end\n"
        f"0,0,0,tpch,Q1,{n},{n + 16}\n"
        f"0,0,1,tpch,Q6,{n + 16},{n + 32}\n"
    )


def run(tmp_path, *args, components="a,b", name="t"):
    cmd = [str(BIN), "--traces-dir", str(tmp_path), "--components", components, "--name", name,
           "--out-prefix", str(tmp_path / "out"), "--jobs", "2"]
    return subprocess.run(cmd + list(args), capture_output=True, text=True)


def rows(tmp_path, suffix):
    with open(tmp_path / f"out_{suffix}.csv", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def traces(tmp_path):
    write_component(tmp_path, "a")
    write_component(tmp_path, "b")
    return tmp_path


def check_switches(directory, res):
    cands = {"lru", "clock", "lru2_visit", "arc_visit", "lirs_visit"}
    total_runs = sum(int(x["runs"]) for x in rows(directory, "layout"))
    key_cols = ["switcher", "policy", "pool", "window_k", "eta", "alpha"]
    seen = {}
    for x in rows(directory, "switches"):
        assert x["from"] in cands and x["to"] in cands and x["from"] != x["to"]
        run = int(x["run"])
        assert 0 <= run < total_runs
        runs = seen.setdefault(tuple(x[c] for c in key_cols), [])
        assert not runs or run > runs[-1]
        runs.append(run)
    for x in res:
        if x["switches"] != "":
            assert len(seen.get(tuple(x[c] for c in key_cols), [])) == int(x["switches"])


def test_describe_writes_layout(traces):
    r = run(traces, "--describe")
    assert r.returncode == 0, r.stderr
    layout = rows(traces, "layout")
    assert [x["component"] for x in layout] == ["a", "b"]
    assert layout[0]["first_run"] == "0" and layout[0]["trace"] == "t"
    assert int(layout[1]["first_run"]) == int(layout[0]["runs"])


def test_replay_writes_every_switcher(traces):
    r = run(traces, "--windows", "8:40", "--hedge", "10:0,10:0.1", "--pools", "4")
    assert r.returncode == 0, r.stderr
    res = rows(traces, "results")
    assert list(res[0].keys()) == ["trace", "switcher", "policy", "pool", "window_k", "window", "eta", "alpha",
                                   "margin", "threshold", "misses", "switches"]
    names = [x["switcher"] for x in res]
    assert sorted(names) == sorted(["fixed"] * 5 + ["belady", "free_bound", "hindsight", "two_window",
                                                    "hedge", "fixed_share"])
    fixed = [int(x["misses"]) for x in res if x["switcher"] == "fixed"]
    bound = [x for x in res if x["switcher"] == "free_bound"][0]
    assert int(bound["misses"]) <= min(fixed)
    assert bound["window_k"] == "8" and bound["window"] == "40" and bound["switches"] == ""
    hedge = [x for x in res if x["switcher"] == "hedge"][0]
    assert hedge["eta"] == "10" and hedge["alpha"] == "0" and hedge["margin"] == "2"
    share = [x for x in res if x["switcher"] == "fixed_share"][0]
    assert share["alpha"] == "0.1"
    assert [x for x in res if x["switcher"] == "two_window"][0]["threshold"] == "0.05"

    phases = rows(traces, "phases")
    for x in res:
        if x["switcher"] != "fixed":
            continue
        mine = [p for p in phases if p["switcher"] == "fixed" and p["policy"] == x["policy"]]
        assert len(mine) == 2 and sum(int(p["misses"]) for p in mine) == int(x["misses"])
    assert not [p for p in phases if p["switcher"] == "free_bound"]
    occ = rows(traces, "occupancy")
    assert {o["switcher"] for o in occ} == {"hindsight", "two_window", "hedge", "fixed_share"}
    with open(traces / "out_occupancy.csv", newline="") as f:
        assert next(csv.reader(f)) == ["trace", "switcher", "policy", "pool", "window_k", "eta", "alpha",
                                       "phase", "candidate", "runs"]
    check_switches(traces, res)


def test_output_does_not_depend_on_jobs(traces):
    args = ["--windows", "8:40,4:24", "--hedge", "10:0", "--pools", "4,5"]
    assert run(traces, *args).returncode == 0
    first = {s: (traces / f"out_{s}.csv").read_text() for s in ("results", "phases", "occupancy", "switches")}
    r = subprocess.run([str(BIN), "--traces-dir", str(traces), "--components", "a,b", "--name", "t",
                        "--out-prefix", str(traces / "out"), "--jobs", "1", *args], capture_output=True, text=True)
    assert r.returncode == 0
    for s, text in first.items():
        assert (traces / f"out_{s}.csv").read_text() == text


def test_validate_writes_passes(traces):
    r = run(traces, "--validate", "--pools", "4,5")
    assert r.returncode == 0, r.stderr
    val = rows(traces, "validation")
    assert len(val) == 2 * 2 * 6
    assert {x["trace"] for x in val} == {"a", "b"}
    assert {x["policy"] for x in val} == {"lru", "clock", "lru2_visit", "arc_visit", "lirs_visit", "belady"}
    assert list(val[0].keys()) == ["trace", "policy", "pool", "pass", "warmup", "misses"]


def test_refuses_different_layout(tmp_path):
    write_component(tmp_path, "a")
    write_component(tmp_path, "b", prefix=[0, 1, 2, 9])
    r = run(tmp_path, "--describe")
    assert r.returncode == 1 and "b" in r.stderr and r.stderr.startswith("eviction_switch:")


@pytest.mark.parametrize("window", ["8:42", "8:1000", "0:40"])
def test_refuses_bad_window(traces, window):
    r = run(traces, "--windows", window, "--pools", "4")
    assert r.returncode == 1 and r.stderr.startswith("eviction_switch:")


def test_refuses_unknown_start(traces):
    r = run(traces, "--windows", "8:40", "--pools", "4", "--start", "nope")
    assert r.returncode == 1 and "nope" in r.stderr


def test_refuses_missing_windows_and_bad_hedge(traces):
    assert run(traces, "--pools", "4").returncode == 1
    assert run(traces, "--windows", "8:40", "--pools", "4", "--hedge", "10").returncode == 1
    assert run(traces, "--windows", "8:40", "--pools", "4", "--hedge", "10:1").returncode == 1
    assert run(traces, "--windows", "8:40", "--pools", "4", "--bogus", "1").returncode == 1


@pytest.mark.parametrize("flags", [("--visit-window", "4294967360"), ("--visit-window", "+64"), ("--visit-window", " 64"),
                                   ("--jobs", "99999999999999999999"), ("--pools", "4,-4")])
def test_refuses_numbers_that_do_not_fit(traces, flags):
    r = run(traces, "--windows", "8:40", "--pools", "4", *flags)
    assert r.returncode == 1 and r.stderr.startswith("eviction_switch:")


def write_pages(directory, name, body):
    pages = PREFIX + body
    n = len(PREFIX)
    half = n + len(body) // 2
    (directory / f"{name}.csv").write_text("timestamp,page_id\n" + "".join(f"{i},{p}\n" for i, p in enumerate(pages)))
    (directory / name).mkdir(exist_ok=True)
    (directory / name / "stream.csv").write_text(
        "pass,warmup,seq,kind,query,trace_start,trace_end\n"
        f"0,0,0,tpch,Q1,{n},{half}\n"
        f"0,0,1,tpch,Q6,{half},{len(pages)}\n"
    )


def test_switchers_switch_when_phases_differ(tmp_path):
    shifting = [30 + 8 * s + i for s in range(20) for _ in range(3) for i in range(8)]
    loop = [100 + i for _ in range(30) for i in range(12)]
    write_pages(tmp_path, "shift", shifting)
    write_pages(tmp_path, "loop", loop)
    r = run(tmp_path, "--windows", "8:80", "--hedge", "10:0,10:0.1", "--pools", "10", components="shift,loop")
    assert r.returncode == 0, r.stderr
    res = rows(tmp_path, "results")
    hind = [x for x in res if x["switcher"] == "hindsight"][0]
    assert int(hind["switches"]) >= 1
    sw = rows(tmp_path, "switches")
    assert any(x["switcher"] == "hindsight" for x in sw)
    check_switches(tmp_path, res)
