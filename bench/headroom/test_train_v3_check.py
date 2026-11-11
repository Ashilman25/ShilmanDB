"""Tests for train_v3_check.py on synthetic samples."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SPEC_PATH = Path(__file__).with_name("train_v3_check.py")
spec = importlib.util.spec_from_file_location("train_v3_check", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["train_v3_check"] = mod
spec.loader.exec_module(mod)

SIM = Path(__file__).resolve().parents[2] / "build" / "bench" / "eviction_sim"
POOL = 40
SF = "0.01"


def make_rows(rng, n):
    x = rng.random((n, 29), dtype=np.float32) * np.float32(100)
    x[:, 3:] *= np.float32(0.1)
    y = ((x[:, 0] + 3 * x[:, 1] - x[:, 4] + rng.normal(0, 5, n)) > 100).astype(np.float32)
    return np.concatenate([y[:, None], x], axis=1).astype("<f4")


def write_samples(tmp_path, rows_each=3000, pool=POOL, sf=SF, trace_override=None, shorten=False):
    rng = np.random.default_rng(1)
    for trace in mod.TRAIN_TRACES:
        for replay in mod.REPLAYS:
            prefix = mod.sample_prefix(tmp_path, trace, replay, pool)
            rows = make_rows(rng, rows_each)
            data = rows.tobytes()
            if shorten:
                data = data[:-4]
            Path(str(prefix) + ".bin").write_bytes(data)
            meta = {"format": 1, "sf": sf, "trace": trace_override or trace, "replay": replay, "pool": pool,
                    "k": 32, "seed": 1, "decisions": rows_each, "evictions": rows_each, "replay_misses": 1,
                    "rows": rows_each, "columns": ["label"] + [f"f{i}" for i in range(29)]}
            Path(str(prefix) + ".json").write_text(json.dumps(meta))


def test_cells_match_spec_parameter_counts():
    names = ["mlp_small_basic", "mlp_small_richer", "mlp_medium_basic", "mlp_medium_richer",
             "mlp_large_basic", "mlp_large_richer"]
    assert [mod.param_count(mod.CELLS[n]) for n in names] == [945, 1033, 8833, 9153, 99841, 106497]
    assert list(mod.CELLS) == ["mlp_small_basic", "mlp_small_richer", "mlp_medium_basic", "mlp_medium_richer",
                                          "mlp_large_basic", "mlp_large_richer", "gbdt_basic", "gbdt_richer"]


@pytest.mark.parametrize("bad", ["index_mixed_s1", "index_stream_s3"])
def test_load_refuses_test_traces(tmp_path, bad):
    write_samples(tmp_path, trace_override=bad)
    with pytest.raises(SystemExit):
        mod.load_samples(tmp_path, SF, POOL)


def test_load_refuses_wrong_pool(tmp_path):
    write_samples(tmp_path)
    with pytest.raises(SystemExit):
        mod.load_samples(tmp_path, SF, POOL + 1)
    # files written for another pool in the JSON
    other = tmp_path / "other"
    other.mkdir()
    write_samples(other)
    for trace in mod.TRAIN_TRACES:
        for replay in mod.REPLAYS:
            p = Path(str(mod.sample_prefix(other, trace, replay, POOL)) + ".json")
            meta = json.loads(p.read_text())
            meta["pool"] = POOL + 5
            p.write_text(json.dumps(meta))
    with pytest.raises(SystemExit):
        mod.load_samples(other, SF, POOL)


def test_load_refuses_wrong_sf_and_missing_file(tmp_path):
    write_samples(tmp_path)
    with pytest.raises(SystemExit):
        mod.load_samples(tmp_path, "0.1", POOL)
    Path(str(mod.sample_prefix(tmp_path, "index_lookup_s2", "belady", POOL)) + ".bin").unlink()
    with pytest.raises(SystemExit) as e:
        mod.load_samples(tmp_path, SF, POOL)
    assert "index_lookup_s2_belady" in str(e.value)


def test_load_refuses_truncated_samples(tmp_path):
    write_samples(tmp_path, shorten=True)
    with pytest.raises(SystemExit) as e:
        mod.load_samples(tmp_path, SF, POOL)
    assert ".bin" in str(e.value)


def test_load_returns_all_rows(tmp_path):
    write_samples(tmp_path, rows_each=100)
    x, y = mod.load_samples(tmp_path, SF, POOL)
    assert x.shape == (800, 29) and y.shape == (800,) and x.dtype == np.float32


def trained_mlp(tmp_path, name="mlp_small_richer", n=20000):
    rows = make_rows(np.random.default_rng(5), n)
    x, y = rows[:, 1:].copy(), rows[:, 0].copy()
    spec = mod.CELLS[name]
    return spec, x, y, mod.train_mlp(x, y, spec)


def test_standardization_from_training_rows_only(tmp_path):
    write_samples(tmp_path, rows_each=500)
    x, y = mod.load_samples(tmp_path, SF, POOL)
    spec = mod.CELLS["mlp_small_basic"]
    trained = mod.train_mlp(x, y, spec)
    np.testing.assert_allclose(trained["mean"], x[:, :3].mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(trained["std"], x[:, :3].std(axis=0), rtol=1e-4)


def test_mlp_export_round_trip(tmp_path):
    spec, x, y, trained = trained_mlp(tmp_path)
    path = tmp_path / "m.txt"
    mod.export_model(spec, trained, path, POOL, SF, x[:256])
    probe = np.random.default_rng(9).random((500, 29), dtype=np.float32) * np.float32(50)
    mine = mod.score_model_file(path, probe)
    torch_logits = mod.torch_logits(trained, probe)
    np.testing.assert_allclose(mine, torch_logits, atol=1e-5, rtol=1e-5)
    assert path.read_text().split()[:4] == ["model", "mlp", "features", "richer"]


def test_gbdt_export_round_trip(tmp_path):
    rows = make_rows(np.random.default_rng(6), 6000)
    x, y = rows[:, 1:].copy(), rows[:, 0].copy()
    spec = mod.CELLS["gbdt_richer"]
    booster = mod.train_gbdt(x, y, spec)
    path = tmp_path / "g.txt"
    mod.export_model(spec, booster, path, POOL, SF, x[:256])
    probe = x[1000:1600]
    mine = mod.score_model_file(path, probe)
    np.testing.assert_allclose(mine, booster.predict(probe, raw_score=True), atol=1e-5)


def test_no_owner_cell_uses_11_inputs(tmp_path):
    spec = mod.ablation_cell("gbdt_richer")
    assert spec.features == "richer_no_owner" and spec.kind == "gbdt"
    assert mod.ablation_cell("mlp_medium_richer").hidden == mod.CELLS["mlp_medium_richer"].hidden
    rows = make_rows(np.random.default_rng(7), 4000)
    x, y = rows[:, 1:].copy(), rows[:, 0].copy()
    path = tmp_path / "g.txt"
    mod.export_model(spec, mod.train_gbdt(x, y, spec), path, POOL, SF, x[:256])
    assert path.read_text().split()[:8] == ["model", "gbdt", "features", "richer_no_owner", "inputs", "11", "pool", str(POOL)]


def test_export_refuses_non_finite_threshold(tmp_path):
    rows = make_rows(np.random.default_rng(8), 2000)
    x, y = rows[:, 1:].copy(), rows[:, 0].copy()
    spec = mod.CELLS["gbdt_basic"]
    booster = mod.train_gbdt(x, y, spec)
    dump = booster.dump_model()
    node = dump["tree_info"][0]["tree_structure"]
    node["threshold"] = float("inf")
    with pytest.raises(ValueError, match="threshold"):
        mod.tree_nodes(dump["tree_info"][0]["tree_structure"])


@pytest.mark.skipif(not SIM.exists(), reason="build/bench/eviction_sim is not built")
def test_cpp_loader_accepts_exported_files(tmp_path):
    spec, x, y, trained = trained_mlp(tmp_path, n=8000)
    rows = make_rows(np.random.default_rng(6), 6000)
    gx, gy = rows[:, 1:].copy(), rows[:, 0].copy()
    gspec = mod.CELLS["gbdt_richer"]
    files = {"mlp": (tmp_path / "m.txt", spec, trained, x), "gbdt": (tmp_path / "g.txt", gspec, mod.train_gbdt(gx, gy, gspec), gx)}
    for path, s, t, data in files.values():
        mod.export_model(s, t, path, POOL, SF, data[:256])
        out = subprocess.run([str(SIM), "--check-model", str(path)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        err = float(next(l.split()[-1] for l in out.stdout.splitlines() if l.startswith("max ref error")))
        assert err < 1e-5


def test_main_writes_models_and_log(tmp_path):
    (tmp_path / "s").mkdir()
    write_samples(tmp_path / "s", rows_each=400)
    out = tmp_path / "models"
    argv = ["--samples-dir", str(tmp_path / "s"), "--sf", SF, "--pool", str(POOL), "--out-dir", str(out),
            "--cells", "mlp_small_basic,gbdt_richer_no_owner"]
    assert mod.main(argv) == 0
    assert (out / f"mlp_small_basic_p{POOL}.txt").exists()
    assert (out / f"gbdt_richer_no_owner_p{POOL}.txt").exists()
    lines = (out / "train_log.csv").read_text().splitlines()
    assert lines[0] == "cell,pool,rows,positive_share,final_loss,label_agreement,seconds" and len(lines) == 3
    assert mod.main(argv[:-1] + ["mlp_small_basic"]) == 0
    lines = (out / "train_log.csv").read_text().splitlines()
    assert len(lines) == 3 and sum(l.startswith("mlp_small_basic,") for l in lines) == 1


def test_load_refuses_sidecar_for_another_prefix(tmp_path):
    write_samples(tmp_path)
    prefix = mod.sample_prefix(tmp_path, mod.TRAIN_TRACES[0], "lru", POOL)
    js = Path(str(prefix) + ".json")
    meta = json.loads(js.read_text())
    meta["replay"] = "belady"
    js.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="lru"):
        mod.load_samples(tmp_path, SF, POOL)
    meta["replay"] = "lru"
    meta["trace"] = mod.TRAIN_TRACES[1]
    js.write_text(json.dumps(meta))
    with pytest.raises(SystemExit):
        mod.load_samples(tmp_path, SF, POOL)


def test_scoring_in_chunks_matches_one_pass(tmp_path):
    spec, x, y, trained = trained_mlp(tmp_path, n=3000)
    whole = mod.torch_logits(trained, x, chunk=10**9)
    for chunk in (1, 700, 2999):
        np.testing.assert_allclose(mod.torch_logits(trained, x, chunk=chunk), whole, atol=1e-6)
    assert mod.log_stats(spec, trained, x, y, chunk=500) == pytest.approx(mod.log_stats(spec, trained, x, y, chunk=10**9), abs=1e-3)


def test_gbdt_training_returns_train_scores(tmp_path):
    rows = make_rows(np.random.default_rng(6), 3000)
    x, y = rows[:, 1:].copy(), rows[:, 0].copy()
    spec = mod.CELLS["gbdt_richer"]
    booster = mod.train_gbdt(x, y, spec)
    assert booster.train_scores.shape == (3000,)
    np.testing.assert_allclose(booster.train_scores, booster.predict(x, raw_score=True), atol=1e-12)
    loss, agree = mod.log_stats(spec, booster, x, y)
    assert 0 < loss < 1 and agree > 0.5
