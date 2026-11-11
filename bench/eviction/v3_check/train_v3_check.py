from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRAIN_TRACES: tuple[str, ...] = ("index_stream_s1", "index_stream_s2", "index_lookup_s1", "index_lookup_s2")
REPLAYS = ("lru", "belady")
FEATURE_COUNTS = {"basic": 3, "richer_no_owner": 11, "richer": 29}

ROW_WIDTH = 30  # label, then 29 features
LR = 1e-3
BATCH = 4096
EPOCHS = 4
TORCH_SEED = 0
THREADS = 8
GBDT_PARAMS = dict(objective="binary", num_iterations=32, num_leaves=32, learning_rate=0.1, deterministic=True, force_row_wise=True, seed=0, num_threads=8, verbose=-1)
REF_ROWS = 256
EXPORT_TOLERANCE = 1e-5
LOG_HEADER = ["cell", "pool", "rows", "positive_share", "final_loss", "label_agreement", "seconds"]


@dataclass(frozen=True)
class CellSpec:
    name: str
    kind: str  # "mlp" | "gbdt"
    features: str
    hidden: tuple[int, ...]


CELLS: dict[str, CellSpec] = {c.name: c for c in (
    CellSpec("mlp_small_basic", "mlp", "basic", (32, 24)),
    CellSpec("mlp_small_richer", "mlp", "richer", (24, 12)),
    CellSpec("mlp_medium_basic", "mlp", "basic", (128, 64)),
    CellSpec("mlp_medium_richer", "mlp", "richer", (96, 64)),
    CellSpec("mlp_large_basic", "mlp", "basic", (256, 256, 128)),
    CellSpec("mlp_large_richer", "mlp", "richer", (256, 256, 128)),
    CellSpec("gbdt_basic", "gbdt", "basic", ()),
    CellSpec("gbdt_richer", "gbdt", "richer", ()),
)}


# The richer cell without the page-owner inputs.
def ablation_cell(name: str) -> CellSpec:
    base = CELLS[name]
    if base.features != "richer":
        raise ValueError(f"{name} has no owner inputs to leave out")
    return CellSpec(f"{name}_no_owner", base.kind, "richer_no_owner", base.hidden)


def resolve_cell(name: str) -> CellSpec:
    if name in CELLS:
        return CELLS[name]
    if name.endswith("_no_owner") and name[: -len("_no_owner")] in CELLS:
        return ablation_cell(name[: -len("_no_owner")])
    raise SystemExit(f"unknown cell '{name}'")


def sample_prefix(samples_dir: Path, trace: str, replay: str, pool: int) -> Path:
    return samples_dir / f"{trace}_{replay}_p{pool}"


def model_path(models_dir: Path, cell: str, pool: int) -> Path:
    return models_dir / f"{cell}_p{pool}.txt"


# Reads the 8 training sample files for a pool.
# Refuses anything that isn't exactly that.
def load_samples(samples_dir: Path, sf: str, pool: int) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for trace in TRAIN_TRACES:
        for replay in REPLAYS:
            prefix = sample_prefix(samples_dir, trace, replay, pool)
            js, bn = Path(str(prefix) + ".json"), Path(str(prefix) + ".bin")
            if not js.exists() or not bn.exists():
                raise SystemExit(f"missing sample file {prefix}(.json/.bin)")
            
            meta = json.loads(js.read_text())
            if meta.get("format") != 1:
                raise SystemExit(f"{js}: format {meta.get('format')} is not 1")
            
            if meta.get("trace") not in TRAIN_TRACES:
                raise SystemExit(f"{js}: trace '{meta.get('trace')}' is not a training trace")
            
            if meta.get("trace") != trace or meta.get("replay") != replay:
                raise SystemExit(f"{js}: is for {meta.get('trace')} / {meta.get('replay')}, expected {trace} / {replay}")
            
            if meta.get("pool") != pool:
                raise SystemExit(f"{js}: pool {meta.get('pool')} is not {pool}")
            
            if str(meta.get("sf")) != sf:
                raise SystemExit(f"{js}: sf '{meta.get('sf')}' is not '{sf}'")
            
            rows = meta.get("rows")
            if not isinstance(rows, int) or bn.stat().st_size != rows * ROW_WIDTH * 4:
                raise SystemExit(f"{bn}: size {bn.stat().st_size} bytes is not {rows} rows x {ROW_WIDTH} x 4")
            parts.append(np.fromfile(bn, dtype="<f4").reshape(rows, ROW_WIDTH))
            
    data = np.concatenate(parts).astype(np.float32)
    if len(data) == 0:
        raise SystemExit(f"no sample rows in {samples_dir}")
    return np.ascontiguousarray(data[:, 1:]), np.ascontiguousarray(data[:, 0])


def param_count(spec: CellSpec) -> int:
    widths = [FEATURE_COUNTS[spec.features], *spec.hidden, 1]
    return sum(i * o + o for i, o in zip(widths, widths[1:]))


def train_mlp(X, y, spec: CellSpec) -> dict:
    import torch
    from torch import nn

    n = FEATURE_COUNTS[spec.features]
    x = np.ascontiguousarray(X[:, :n], dtype=np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, np.float32(1.0), std).astype(np.float32)

    torch.manual_seed(TORCH_SEED)
    torch.set_num_threads(THREADS)
    widths = [n, *spec.hidden, 1]
    linears = [nn.Linear(i, o) for i, o in zip(widths, widths[1:])]
    body = []
    for k, lin in enumerate(linears):
        body.append(lin)
        if k + 1 < len(linears):
            body.append(nn.ReLU())
            
    net = nn.Sequential(*body)
    mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)
    inputs = torch.from_numpy((x - mean) / std)
    labels = torch.from_numpy(np.asarray(y, dtype=np.float32))

    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    last_loss = float("nan")
    for epoch in range(EPOCHS):
        order = np.random.default_rng(epoch).permutation(len(x))
        total = 0.0
        
        for start in range(0, len(x), BATCH):
            idx = torch.from_numpy(order[start:start + BATCH])
            opt.zero_grad()
            loss = loss_fn(net(inputs[idx]).squeeze(1), labels[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            
        last_loss = total / len(x)

    layers = [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32)) for l in linears]
    return {
        "mean": mean, "std": std, 
        "layers": layers, "net": net, 
        "mean_t": mean_t, "std_t": std_t,
        "final_loss": last_loss
    }


SCORE_CHUNK = 262_144  # rows scored at once, so a big model never holds every row's activations


# Torch's logits for raw (unstandardized) rows of the first n features, scored in chunks.
def torch_logits(trained: dict, x: np.ndarray, chunk: int = SCORE_CHUNK) -> np.ndarray:
    import torch

    n = len(trained["mean"])
    out = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            raw = torch.from_numpy(np.ascontiguousarray(x[start:start + chunk, :n], dtype=np.float32))
            out[start:start + chunk] = trained["net"]((raw - trained["mean_t"]) / trained["std_t"]).squeeze(1).numpy()
            
    return out


# LightGBM links a different OpenMP runtime from torch's, so mixing them in one process aborts.
# It runs in a child process that never imports torch; this object holds what it hands back.
class GbdtModel:
    def __init__(self, model_text: str, dump: dict, train_scores: np.ndarray | None = None):
        self.model_text = model_text
        self._dump = dump
        self.train_scores = train_scores  # raw scores of the rows it was trained on

    def dump_model(self) -> dict:
        return self._dump

    def predict(self, x, raw_score: bool = True) -> np.ndarray:
        if not raw_score:
            raise ValueError("only raw scores are used")
        
        with tempfile.TemporaryDirectory() as tmp:
            model, rows, out = Path(tmp, "model.txt"), Path(tmp, "x.npy"), Path(tmp, "out.npy")
            model.write_text(self.model_text)
            np.save(rows, np.ascontiguousarray(x, dtype=np.float32))
            run_worker("predict", model, rows, out)
            return np.load(out)


def run_worker(*args) -> None:
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", *map(str, args)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LightGBM worker failed:\n{result.stdout}{result.stderr}")


def train_gbdt(X, y, spec: CellSpec) -> GbdtModel:
    n = FEATURE_COUNTS[spec.features]
    
    with tempfile.TemporaryDirectory() as tmp:
        data, out, scores = Path(tmp, "data.npz"), Path(tmp, "model.json"), Path(tmp, "scores.npy")
        np.savez(data, x=np.ascontiguousarray(X[:, :n], dtype=np.float32), y=np.asarray(y, dtype=np.float32))
        run_worker("fit", data, out, scores)
        saved = json.loads(out.read_text())
        train_scores = np.load(scores)
        
    return GbdtModel(saved["model"], saved["dump"], train_scores)


def worker_fit(data_path: str, out_path: str, scores_path: str) -> None:
    import lightgbm as lgb

    data = np.load(data_path)
    params = dict(GBDT_PARAMS)
    rounds = params.pop("num_iterations")
    booster = lgb.train(params, lgb.Dataset(data["x"], label=data["y"]), num_boost_round=rounds)
    Path(out_path).write_text(json.dumps({"model": booster.model_to_string(), "dump": booster.dump_model()}))
    np.save(scores_path, np.asarray(booster.predict(data["x"], raw_score=True), dtype=np.float64))


def worker_predict(model_path_: str, x_path: str, out_path: str) -> None:
    import lightgbm as lgb

    booster = lgb.Booster(model_str=Path(model_path_).read_text())
    np.save(out_path, np.asarray(booster.predict(np.load(x_path), raw_score=True), dtype=np.float64))


# Flattens one dumped tree into pre-order (feature, threshold, left, right, value) rows.
def tree_nodes(root: dict) -> list[tuple[int, float, int, int, float]]:
    nodes: list[tuple[int, float, int, int, float]] = []

    def walk(node: dict) -> int:
        at = len(nodes)
        nodes.append((-1, 0.0, -1, -1, 0.0))
        if "leaf_value" in node:
            value = float(node["leaf_value"])
            if not math.isfinite(value):
                raise ValueError("leaf value is not finite")
            
            nodes[at] = (-1, 0.0, -1, -1, value)
            return at
        
        if node["decision_type"] != "<=":
            raise ValueError(f"decision type {node['decision_type']} is not <=")

        if node.get("missing_type", "None") != "None":
            raise ValueError(f"missing_type {node.get('missing_type')} is not None")

        threshold = float(node["threshold"])
        if not math.isfinite(threshold):
            raise ValueError(f"threshold {node['threshold']} is not finite")
        
        left = walk(node["left_child"])
        right = walk(node["right_child"])
        nodes[at] = (int(node["split_feature"]), threshold, left, right, 0.0)
        return at

    walk(root)
    return nodes


def fmt(values, spec="%.9g") -> str:
    return " ".join(spec % float(v) for v in np.ravel(values))


def export_model(spec: CellSpec, trained, path: Path, pool: int, sf: str, ref_x: np.ndarray) -> None:
    n = FEATURE_COUNTS[spec.features]
    ref = np.ascontiguousarray(ref_x[:, :n], dtype=np.float32)
    lines = [f"model {spec.kind}", f"features {spec.features}", f"inputs {n}", f"pool {pool}", f"sf {sf}"]
    
    if spec.kind == "mlp":
        lines += [f"mean {n} {fmt(trained['mean'])}", f"std {n} {fmt(trained['std'])}"]
        for w, b in trained["layers"]:
            lines.append(f"layer {w.shape[0]} {w.shape[1]} {fmt(w)} {fmt(b)}")
            
        expected = torch_logits(trained, ref)
    else:
        dump = trained.dump_model()
        trees = [tree_nodes(t["tree_structure"]) for t in dump["tree_info"]]
        lines.append(f"trees {len(trees)}")
        for nodes in trees:
            lines.append(f"tree {len(nodes)}")
            for feature, threshold, left, right, value in nodes:
                lines.append(f"{feature} {threshold:.17g} {left} {right} {value:.9g}")
                
        expected = np.asarray(trained.predict(ref, raw_score=True), dtype=np.float64)
    lines.append(f"ref {len(ref)}")
    
    for row, score in zip(ref, expected):
        lines.append(f"{fmt(row)} {score:.9g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    mine = score_model_file(path, ref)
    gap = np.abs(mine - expected) / np.maximum(1.0, np.abs(expected))
    if not np.all(gap < EXPORT_TOLERANCE):
        raise RuntimeError(f"{path}: the exported model is off from the trained one by {gap.max():.3g}")


# Scores raw rows from a model file the way the C++ scorer does.
def score_model_file(path: Path, x: np.ndarray) -> np.ndarray:
    tok = Path(path).read_text().split()
    pos = 0

    def take(k=1):
        nonlocal pos
        out = tok[pos:pos + k]
        pos += k
        return out

    def expect(word):
        got = take()[0]
        if got != word:
            raise ValueError(f"{path}: expected '{word}', got '{got}'")

    expect("model")
    kind = take()[0]
    expect("features")
    take()
    expect("inputs")
    n = int(take()[0])
    expect("pool")
    take()
    expect("sf")
    take()
    data = np.ascontiguousarray(x[:, :n], dtype=np.float32)
    
    if kind == "mlp":
        expect("mean")
        mean = np.array(take(int(take()[0])), dtype=np.float32)
        expect("std")
        
        std = np.array(take(int(take()[0])), dtype=np.float32)
        h = (data - mean) / std
        layers = []
        
        while pos < len(tok) and tok[pos] == "layer":
            take()
            out, inn = int(take()[0]), int(take()[0])
            w = np.array(take(out * inn), dtype=np.float32).reshape(out, inn)
            layers.append((w, np.array(take(out), dtype=np.float32)))
            
        for k, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if k + 1 < len(layers):
                h = np.maximum(h, np.float32(0))
        return h[:, 0].astype(np.float64)
    
    expect("trees")
    total = np.zeros(len(data), dtype=np.float64)
    for _ in range(int(take()[0])):
        expect("tree")
        count = int(take()[0])
        nodes = []
        for _ in range(count):
            f, t, l, r, v = take(5)
            nodes.append((int(f), float(t), int(l), int(r), float(v)))
            
        for i, row in enumerate(data):
            at = 0
            while nodes[at][2] >= 0:
                f, t, l, r, _ = nodes[at]
                at = l if float(row[f]) <= t else r
            total[i] += nodes[at][4]
            
    return total


def log_stats(spec: CellSpec, trained, X, y, chunk: int = SCORE_CHUNK) -> tuple[float, float]:
    if spec.kind == "mlp":
        logits = torch_logits(trained, X, chunk)
        loss = trained["final_loss"]
    else:
        logits = trained.train_scores
        p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-15, 1 - 1e-15)
        loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        
    return loss, float(np.mean((logits > 0) == (y > 0.5)))


# Adds a row to the training log.
# Replaces any earlier row for the same cell and pool.
def write_log_row(log: Path, row: list) -> None:
    rows = []
    if log.exists():
        with log.open(newline="") as f:
            rows = [r for r in list(csv.reader(f))[1:] if r[:2] != [str(row[0]), str(row[1])]]
            
    rows.append([str(c) for c in row])
    with log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_HEADER)
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--sf", required=True)
    ap.add_argument("--pool", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(CELLS))
    args = ap.parse_args(argv)

    specs = [resolve_cell(c) for c in args.cells.split(",") if c]
    X, y = load_samples(args.samples_dir, args.sf, args.pool)
    ref_x = X[np.random.default_rng(0).choice(len(X), size=min(REF_ROWS, len(X)), replace=False)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "train_log.csv"
    
    for spec in specs:
        start = time.time()
        trained = train_mlp(X, y, spec) if spec.kind == "mlp" else train_gbdt(X, y, spec)
        export_model(spec, trained, model_path(args.out_dir, spec.name, args.pool), args.pool, args.sf, ref_x)
        loss, agree = log_stats(spec, trained, X, y)
        write_log_row(log, [spec.name, args.pool, len(X), f"{float(y.mean()):.6f}", f"{loss:.6f}", f"{agree:.6f}", f"{time.time() - start:.2f}"])
        print(f"{spec.name} pool {args.pool}: loss {loss:.4f}, agreement {agree:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        {"fit": worker_fit, "predict": worker_predict}[sys.argv[2]](*sys.argv[3:])
        sys.exit(0)
    sys.exit(main())
