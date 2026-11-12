"""Tests for bench/vector_check/hnsw_graph.py; faiss runs only in a child process."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HNSW_GRAPH = HERE.parent / "hnsw_graph.py"
INFO_KEYS = ["n", "dim", "m", "ef_construction", "entry_point", "max_level", "threads", "build_s", "sha256"]
ARRAY_FILES = ["levels.npy", "offsets.npy", "neighbors.npy", "cum.npy"]


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_hnsw_graph", "hnsw_graph.py")
hs = mod.hs
vc = mod.vc


# Seeded unit float32 rows.
def unit_rows(rng, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# A small graph in faiss's layout and dtypes.
# Layer 0 is a ring; node 0 is also on level 1.
def ring_graph(n: int) -> dict:
    cum = np.array([0, 4, 6], dtype=np.int32)
    levels = np.ones(n, dtype=np.int32)
    levels[0] = 2
    offsets = np.zeros(n + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum(cum[levels])
    neighbors = np.full(int(offsets[-1]), -1, dtype=np.int32)
    for i in range(n):
        start = int(offsets[i])
        neighbors[start:start + 2] = [(i - 1) % n, (i + 1) % n]
    return {"levels": levels, "offsets": offsets, "neighbors": neighbors, "cum": cum, "entry": 0, "max_level": 1}


# The info the command line passes to export, with made-up timings.
def sample_info(threads: int = 3, build_s: float = 1.25) -> dict:
    return {"dim": vc.DIM, "m": 16, "ef_construction": 64, "threads": threads, "build_s": build_s}


# Results and ai folders under tmp_path, with the 70k base written as test.npy.
def write_world(tmp_path: Path, base: np.ndarray | None) -> dict:
    results, ai = tmp_path / "results", tmp_path / "ai"
    (ai / "embeddings").mkdir(parents=True)
    results.mkdir()
    base_path = ai / "embeddings" / "test.npy"
    if base is not None:
        np.save(base_path, base)
    return {"results": results, "ai": ai, "base_path": base_path, "graph_dir": results / "size70k" / "graph"}


# Run main in this process on a world; returns the exit code.
def run_main(world: dict, *extra: str, size: str = "70k") -> int:
    return mod.main(["--size", size, "--results", str(world["results"]), "--ai", str(world["ai"]), *extra])


# Make build_graph fail the test, so a refused input never loads faiss here.
@pytest.fixture
def no_build(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("build_graph should not run")

    monkeypatch.setattr(mod, "build_graph", refuse)


# One real faiss build of 500 rows, run as a child process with 2 threads.
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if importlib.util.find_spec("faiss") is None:
        pytest.skip("faiss is not installed")
    world = write_world(tmp_path_factory.mktemp("built"), unit_rows(np.random.default_rng(11), 500, vc.DIM))
    proc = subprocess.run(
        [sys.executable, str(HNSW_GRAPH), "--size", "70k", "--threads", "2",
         "--results", str(world["results"]), "--ai", str(world["ai"])],
        capture_output=True, text=True, timeout=120)
    world["proc"] = proc
    world["base"] = np.load(world["base_path"])
    return world


# Loading the module pulls in neither lightgbm nor faiss.
def test_import_does_not_load_lightgbm():
    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('vc_hnsw_graph', {str(HNSW_GRAPH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['vc_hnsw_graph'] = m\n"
        "spec.loader.exec_module(m)\n"
        "print(json.dumps({name: name in sys.modules for name in ('lightgbm', 'faiss')}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"lightgbm": False, "faiss": False}


# The real build at ef 500 finds the exact top 10 for 20 queries.
def test_export_round_trip_exact_at_large_ef(built):
    proc = built["proc"]
    assert proc.returncode == 0, proc.stderr
    # A stray library warning is fine; a traceback or error isn't.
    assert "Traceback" not in proc.stderr and "Error" not in proc.stderr
    base = built["base"]
    searcher = hs.Searcher(hs.load_graph(built["graph_dir"], base))
    for q in unit_rows(np.random.default_rng(12), 20, vc.DIM):
        r = searcher.search(q, ef=500, k=10)
        want_ids, want_d = vc.topk(vc.cosine_dists(base, q), 10)
        assert r.ids.tolist() == want_ids.tolist()
        np.testing.assert_allclose(r.dists, want_d, rtol=0, atol=1e-6)


# graph.json holds every field, in order, matching the saved arrays.
def test_graph_json_fields(built):
    proc = built["proc"]
    assert proc.returncode == 0, proc.stderr
    out = built["graph_dir"]
    info = json.loads((out / "graph.json").read_text())
    assert list(info) == INFO_KEYS
    assert (info["n"], info["dim"], info["m"], info["ef_construction"], info["threads"]) == (500, vc.DIM, 16, 64, 2)
    assert isinstance(info["build_s"], float) and info["build_s"] >= 0

    arrays = [np.load(out / name) for name in ARRAY_FILES]
    levels = arrays[0]
    assert info["sha256"] == hs.graph_sha256(*arrays)
    assert info["max_level"] == int(levels.max()) - 1 == int(levels[info["entry_point"]]) - 1
    # I keep faiss's own dtypes.
    assert [a.dtype for a in arrays] == [np.int32, np.uint64, np.int32, np.int32]
    assert sorted(p.name for p in out.iterdir()) == sorted([*ARRAY_FILES, "graph.json"])
    assert f"n {500:,}" in proc.stdout and f"entry point {info['entry_point']}" in proc.stdout
    assert f"max level {info['max_level']}" in proc.stdout and "built in" in proc.stdout


# Export then load gives back the same arrays, dtypes, entry and max level.
def test_export_then_load_round_trip(tmp_path):
    rng = np.random.default_rng(5)
    n = 9
    vectors = unit_rows(rng, n, vc.DIM)
    graph = ring_graph(n)
    out = tmp_path / "graph"
    mod.export(graph, out, sample_info())

    info = json.loads((out / "graph.json").read_text())
    assert list(info) == INFO_KEYS
    assert info == {"n": n, "dim": vc.DIM, "m": 16, "ef_construction": 64, "entry_point": 0, "max_level": 1,
                    "threads": 3, "build_s": 1.25,
                    "sha256": hs.graph_sha256(graph["levels"], graph["offsets"], graph["neighbors"], graph["cum"])}

    g = hs.load_graph(out, vectors)
    assert (g.entry, g.max_level) == (0, 1)
    for name in ("levels", "offsets", "neighbors", "cum"):
        loaded = np.asarray(getattr(g, name))
        assert loaded.dtype == graph[name].dtype and np.array_equal(loaded, graph[name])

    # A ring is connected, so ef >= n finds the exact top k.
    q = unit_rows(rng, 1, vc.DIM)[0]
    r = hs.Searcher(g).search(q, ef=n, k=3)
    assert r.ids.tolist() == vc.topk(vc.cosine_dists(vectors, q), 3)[0].tolist()


# Old graph.json goes first and the new one is written last.
def test_export_writes_json_last(tmp_path, monkeypatch):
    out = tmp_path / "graph"
    mod.export(ring_graph(6), out, sample_info())
    written = []
    real = mod._write_atomic

    # I record each file written, then stop partway, like a Ctrl-C.
    def spy(path, write):
        written.append(path.name)
        if path.name == "neighbors.npy":
            raise KeyboardInterrupt
        real(path, write)

    monkeypatch.setattr(mod, "_write_atomic", spy)
    with pytest.raises(KeyboardInterrupt):
        mod.export(ring_graph(7), out, sample_info())
    assert written == ["levels.npy", "offsets.npy", "neighbors.npy"]
    assert not (out / "graph.json").exists()
    with pytest.raises((FileNotFoundError, ValueError)):
        hs.load_graph(out, unit_rows(np.random.default_rng(0), 7, vc.DIM))

    written.clear()
    monkeypatch.setattr(mod, "_write_atomic", lambda path, write: (written.append(path.name), real(path, write)))
    mod.export(ring_graph(7), out, sample_info())
    assert written == [*ARRAY_FILES, "graph.json"]


# export refuses missing info fields and arrays that aren't 1-D.
def test_export_checks_its_inputs(tmp_path):
    info = sample_info()
    del info["threads"]
    with pytest.raises(ValueError, match="threads"):
        mod.export(ring_graph(5), tmp_path / "a", info)
    graph = ring_graph(5)
    graph["levels"] = graph["levels"].reshape(5, 1)
    with pytest.raises(ValueError, match="levels"):
        mod.export(graph, tmp_path / "b", sample_info())
    assert not (tmp_path / "a" / "graph.json").exists() and not (tmp_path / "b" / "graph.json").exists()


# The command line wires build, export and the printout together.
def test_main_writes_graph_with_fake_build(tmp_path, monkeypatch, capsys):
    n = 8
    world = write_world(tmp_path, unit_rows(np.random.default_rng(2), n, vc.DIM))
    calls = []

    # I stand in a build that returns a hand-made graph and its timing.
    def fake_build(vectors, m=16, ef_construction=64, threads=None):
        calls.append((vectors.shape, vectors.dtype, m, ef_construction, threads))
        return {**ring_graph(n), "build_s": 0.5, "threads": 4}

    monkeypatch.setattr(mod, "build_graph", fake_build)
    assert run_main(world, "--threads", "4") == 0
    assert calls == [((n, vc.DIM), np.float32, 16, 64, 4)]
    info = json.loads((world["graph_dir"] / "graph.json").read_text())
    assert (info["n"], info["threads"], info["build_s"], info["entry_point"], info["max_level"]) == (n, 4, 0.5, 0, 1)
    out = capsys.readouterr()
    assert out.err == ""
    assert "0.5 s" in out.out and f"n {n:,}" in out.out and "max level 1" in out.out and "entry point 0" in out.out


# A build that doesn't cover every base row leaves no graph.json.
def test_main_refuses_graph_that_does_not_load(tmp_path, monkeypatch, capsys):
    world = write_world(tmp_path, unit_rows(np.random.default_rng(2), 8, vc.DIM))
    monkeypatch.setattr(mod, "build_graph", lambda vectors, **kw: {**ring_graph(7), "build_s": 0.5, "threads": 1})
    assert run_main(world) == 1
    assert "error" in capsys.readouterr().err
    assert not (world["graph_dir"] / "graph.json").exists()


# A missing base exits 1 naming the file, for both sizes.
@pytest.mark.parametrize("size, name", [("70k", "test.npy"), ("500k", "slice.npy")])
def test_missing_base_exits_1(tmp_path, capsys, no_build, size, name):
    world = write_world(tmp_path, None)
    assert run_main(world, size=size) == 1
    err = capsys.readouterr().err
    assert name in err and "not found" in err
    assert not (world["results"] / f"size{size}").exists()


# The 500k base comes from the results folder's slice.npy.
def test_500k_reads_slice(tmp_path, monkeypatch, capsys):
    world = write_world(tmp_path, None)
    (world["results"] / "embeddings").mkdir()
    np.save(world["results"] / "embeddings" / "slice.npy", unit_rows(np.random.default_rng(4), 6, vc.DIM))
    monkeypatch.setattr(mod, "build_graph", lambda vectors, **kw: {**ring_graph(len(vectors)), "build_s": 0.1, "threads": 1})
    assert run_main(world, size="500k") == 0
    assert json.loads((world["results"] / "size500k" / "graph" / "graph.json").read_text())["n"] == 6


# A base of the wrong shape, dtype or values exits 1 naming the file.
@pytest.mark.parametrize("broken", ["width", "dtype", "one_d", "nan", "empty", "garbage"])
def test_bad_base_exits_1(tmp_path, capsys, no_build, broken):
    good = unit_rows(np.random.default_rng(1), 5, vc.DIM)
    bases = {
        "width": good[:, :-1],
        "dtype": good.astype(np.float64),
        "one_d": good[0],
        "nan": np.where(np.arange(vc.DIM) == 3, np.nan, good).astype(np.float32),
        "empty": good[:0],
    }
    world = write_world(tmp_path, bases.get(broken))
    if broken == "garbage":
        world["base_path"].write_bytes(b"not a numpy file")
    assert run_main(world) == 1
    assert "test.npy" in capsys.readouterr().err
    assert not world["graph_dir"].exists()


# An output folder inside the ai folder is refused before anything runs.
def test_output_inside_ai_refused(tmp_path, capsys, no_build):
    world = write_world(tmp_path, unit_rows(np.random.default_rng(1), 5, vc.DIM))
    world["results"] = world["ai"] / "inside"
    assert run_main(world) == 1
    assert "ai_queries" in capsys.readouterr().err
    assert not (world["ai"] / "inside").exists()


# --threads must be at least 1.
def test_threads_must_be_positive(tmp_path, capsys, no_build):
    world = write_world(tmp_path, unit_rows(np.random.default_rng(1), 5, vc.DIM))
    with pytest.raises(SystemExit) as e:
        run_main(world, "--threads", "0")
    assert e.value.code == 2
    assert "--threads" in capsys.readouterr().err
