"""Tests for the eviction_sim command line (learned and sampled policies). Skipped if it isn't built."""
import csv
import subprocess
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[3] / "build" / "bench" / "eviction_sim"
INPUTS = {"basic": 3, "richer_no_owner": 11, "richer": 29}

pytestmark = pytest.mark.skipif(not SIM.exists(), reason="build/bench/eviction_sim is not built")


def write_trace(tmp_path):
    pages = [0, 1, 2, 3, 0, 1, 4, 2, 0, 3, 1, 4, 2, 0, 1, 3, 4, 0, 2, 1] * 2
    trace = tmp_path / "trace.csv"
    trace.write_text("timestamp,page_id\n" + "".join(f"{i},{p}\n" for i, p in enumerate(pages)))
    items = tmp_path / "stream.csv"
    items.write_text(
        "pass,warmup,seq,kind,query,trace_start,trace_end\n"
        "0,0,0,tpch,Q1,0,20\n"
        "0,0,1,tpch,Q6,20,40\n"
    )
    return trace, items


def write_model(tmp_path, pool=3, features="basic", name="model.txt"):
    n = INPUTS[features]
    zeros = " ".join("0" for _ in range(n))
    ones = " ".join("1" for _ in range(n))
    text = (
        f"model mlp\nfeatures {features}\ninputs {n}\npool {pool}\nsf 0.01\n"
        f"mean {n} {zeros}\nstd {n} {ones}\n"
        f"layer 1 {n} {zeros} 0.5\n"
        f"ref 1\n{zeros} 0.5\n"
    )
    path = tmp_path / name
    path.write_text(text)
    return path


def run(tmp_path, *args, pools="3", policies=None):
    trace, items = write_trace(tmp_path)
    cmd = [str(SIM), "--trace", str(trace), "--items", str(items), "--name", "t", "--pools", pools,
           "--out-prefix", str(tmp_path / "out"), "--jobs", "1"]
    if policies:
        cmd += ["--policies", policies]
    return subprocess.run(cmd + list(args), capture_output=True, text=True)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_sampled_lru_full_sample_equals_lru(tmp_path):
    r = run(tmp_path, "--sample-k", "1000000", policies="lru,sampled_lru")
    assert r.returncode == 0, r.stderr
    rows = read_rows(tmp_path / "out_items.csv")
    by = {p: [x["misses"] for x in rows if x["policy"] == p] for p in ("lru", "sampled_lru")}
    assert len(by["lru"]) == 2
    assert by["lru"] == by["sampled_lru"]


def test_learned_refuses_wrong_pool_and_missing_owners(tmp_path):
    model = write_model(tmp_path)
    r = run(tmp_path, "--model", str(model), pools="4", policies="learned")
    assert r.returncode == 1 and "pool 3" in r.stderr

    rich = write_model(tmp_path, features="richer", name="rich.txt")
    r = run(tmp_path, "--model", str(rich), policies="learned")
    assert r.returncode == 1 and "page-owners" in r.stderr

    r = run(tmp_path, policies="learned")
    assert r.returncode == 1 and "learned needs --model" in r.stderr


def test_time_decisions_writes_timing_csv(tmp_path):
    model = write_model(tmp_path)
    r = run(tmp_path, "--model", str(model), "--time-decisions", "--sample-k", "2", policies="learned")
    assert r.returncode == 0, r.stderr
    path = tmp_path / "out_timing.csv"
    assert path.read_text().splitlines()[0] == (
        "trace,policy,pool,k,decisions,median_ns,mean_ns,p90_ns,macs_per_decision,nodes_per_decision,features,model")
    rows = read_rows(path)
    assert len(rows) == 1 and int(rows[0]["decisions"]) > 0
    assert rows[0]["macs_per_decision"] == "6" and rows[0]["features"] == "basic"
    assert rows[0]["model"] == str(model)


def test_time_decisions_needs_learned(tmp_path):
    r = run(tmp_path, "--time-decisions", policies="lru")
    assert r.returncode == 1 and "learned" in r.stderr


def test_check_model_prints_error(tmp_path):
    model = write_model(tmp_path)
    r = subprocess.run([str(SIM), "--check-model", str(model)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0].startswith("max ref error")
    assert "macs per candidate 3" in r.stdout and "inputs 3" in r.stdout and "pool 3" in r.stdout
    bad = tmp_path / "bad.txt"
    bad.write_text("model nope\n")
    assert subprocess.run([str(SIM), "--check-model", str(bad)], capture_output=True, text=True).returncode == 1


def test_trace_page_missing_from_owners_exits_1(tmp_path):
    owners = tmp_path / "owners.csv"
    owners.write_text("page_id,kind,object,owner\n" + "".join(
        f"{p},table,0,lineitem\n" for p in (0, 1, 3, 4)))
    r = run(tmp_path, "--page-owners", str(owners), policies="lru")
    assert r.returncode == 1 and "page 2" in r.stderr


def test_macs_per_decision_counts_only_candidates_in_pool(tmp_path):
    model = write_model(tmp_path)
    r = run(tmp_path, "--model", str(model), "--time-decisions", "--sample-k", "8", policies="learned")
    assert r.returncode == 0, r.stderr
    rows = read_rows(tmp_path / "out_timing.csv")
    assert rows[0]["k"] == "8" and rows[0]["macs_per_decision"] == "9"


SAMPLES = SIM.with_name("eviction_samples")


@pytest.mark.skipif(not SAMPLES.exists(), reason="build/bench/eviction_samples is not built")
def test_samples_refuses_empty_items(tmp_path):
    trace, _ = write_trace(tmp_path)
    items = tmp_path / "empty.csv"
    items.write_text("pass,warmup,seq,kind,query,trace_start,trace_end\n")
    owners = tmp_path / "owners.csv"
    owners.write_text("page_id,kind,object,owner\n" + "".join(f"{p},table,0,lineitem\n" for p in range(5)))
    r = subprocess.run([str(SAMPLES), "--trace", str(trace), "--items", str(items), "--name", "t", "--sf", "0.01",
                        "--pool", "3", "--replay", "lru", "--page-owners", str(owners),
                        "--out", str(tmp_path / "s")], capture_output=True, text=True)
    assert r.returncode == 1 and "no items in" in r.stderr
