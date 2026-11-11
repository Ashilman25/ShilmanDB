"""Tests for bench/eviction/headroom/export_v2_weights.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ml/eviction_policy/models/eviction_model_v2.pt"
if not MODEL.exists():
    pytest.skip("no V2 model file", allow_module_level=True)

SPEC_PATH = Path(__file__).resolve().parents[1] / "headroom" / "export_v2_weights.py"
spec = importlib.util.spec_from_file_location("export_v2_weights", SPEC_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["export_v2_weights"] = mod
spec.loader.exec_module(mod)


def read_floats(line: str, skip: int) -> np.ndarray:
    return np.array(line.split()[skip:], dtype=np.float32)


def test_export_shapes_and_forward_pass(tmp_path):
    out = tmp_path / "w.txt"
    mod.export(str(MODEL), str(out))
    lines = out.read_text().splitlines()
    tags = [line.split()[0] for line in lines[:9]]
    assert tags == ["mean", "std", "w1", "b1", "w2", "b2", "w3", "b3", "ref"]

    mean, std = read_floats(lines[0], 2), read_floats(lines[1], 2)
    assert len(mean) == 4 and len(std) == 4
    h1, n_in = map(int, lines[2].split()[1:3])
    h2, n_h1 = map(int, lines[4].split()[1:3])
    assert n_in == 4 and n_h1 == h1
    w1 = read_floats(lines[2], 3).reshape(h1, 4)
    b1 = read_floats(lines[3], 2)
    w2 = read_floats(lines[4], 3).reshape(h2, h1)
    b2 = read_floats(lines[5], 2)
    assert lines[6].split()[1:3] == ["1", str(h2)]
    w3 = read_floats(lines[6], 3)
    b3 = read_floats(lines[7], 2)
    assert len(b1) == h1 and len(b2) == h2 and len(w3) == h2 and len(b3) == 1

    count = int(lines[8].split()[1])
    assert count == 256 and len(lines) == 9 + count
    refs = np.array([read_floats(line, 0) for line in lines[9:]])
    assert refs.shape == (count, 5)

    mean, std, w1, b1, w2, b2, w3, b3 = (a.astype(np.float64) for a in (mean, std, w1, b1, w2, b2, w3, b3))
    x = (refs[:, :4].astype(np.float64) - mean) / std
    hidden1 = np.maximum(x @ w1.T + b1, 0)
    hidden2 = np.maximum(hidden1 @ w2.T + b2, 0)
    scores = hidden2 @ w3 + b3[0]
    # TorchScript works in float32, so allow its rounding (relative to the score, at least 1).
    err = np.abs(scores - refs[:, 4]) / np.maximum(1.0, np.abs(refs[:, 4]))
    assert err.max() < 1e-4
