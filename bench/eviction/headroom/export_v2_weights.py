#!/usr/bin/env python3
#Export the learned eviction V2 model's weights to a text file.
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

KEYS = [
    "mean", "std", 
    "model.net.0.weight", "model.net.0.bias", 
    "model.net.3.weight", "model.net.3.bias", 
    "model.net.5.weight", "model.net.5.bias"
]


def fmt(values) -> str:
    return " ".join("%.9g" % v for v in np.asarray(values, dtype=np.float32).ravel())


def check_inputs() -> np.ndarray:
    r = np.log1p(np.array([0, 1, 10, 1e3, 1e5, 1e6, 1e7, 1e8]))
    n = np.log1p(np.array([1, 10, 1e3, 1e5]))
    rows = [(a, c, b, 0.0) for a in r for c in n for b in r]  # 8 * 4 * 8 = 256
    return np.array(rows, dtype=np.float32)


def export(model_path: str, out_path: str) -> None:
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # torch.jit.load is the only way to read the file
        model = torch.jit.load(model_path)
    state = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    for key in KEYS:
        if key not in state:
            sys.exit(f"missing weight '{key}' in {model_path}")
            
    w1, b1 = state["model.net.0.weight"], state["model.net.0.bias"]
    w2, b2 = state["model.net.3.weight"], state["model.net.3.bias"]
    w3, b3 = state["model.net.5.weight"], state["model.net.5.bias"]
    h1, h2 = w1.shape[0], w2.shape[0]
    
    shapes_ok = (state["mean"].shape == (4,) and state["std"].shape == (4,) and w1.shape == (h1, 4)
                 and b1.shape == (h1,) and w2.shape == (h2, h1) and b2.shape == (h2,)
                 and w3.shape == (1, h2) and b3.shape == (1,))
    
    if not shapes_ok:
        sys.exit("the model is not a 4 -> h1 -> h2 -> 1 network")

    x = check_inputs()
    with torch.no_grad():
        scores = model(torch.from_numpy(x)).flatten().numpy()

    lines = [
        "mean 4 " + fmt(state["mean"]),
        "std 4 " + fmt(state["std"]),
        f"w1 {h1} 4 " + fmt(w1),
        f"b1 {h1} " + fmt(b1),
        f"w2 {h2} {h1} " + fmt(w2),
        f"b2 {h2} " + fmt(b2),
        f"w3 1 {h2} " + fmt(w3),
        "b3 1 " + fmt(b3),
        f"ref {len(x)}",
    ]
    lines += [fmt(row) + " " + fmt(score) for row, score in zip(x, scores)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({h1} x {h2} network, {len(x)} check inputs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
