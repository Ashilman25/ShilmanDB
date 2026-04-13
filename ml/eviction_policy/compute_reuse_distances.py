#!/usr/bin/env python3

import argparse
import sys
import numpy as np


def load_trace(trace_path: str) -> np.ndarray:
    print(f"Loading trace from {trace_path} ...")
    data = np.loadtxt(trace_path, delimiter=",", skiprows=1, dtype=np.int64)

    # Handle single-row edge case
    if data.ndim == 1:
        page_ids = np.array([data[1]], dtype=np.int32)
    else:
        page_ids = data[:, 1].astype(np.int32)

    print(f"  {len(page_ids)} access events, {len(np.unique(page_ids))} unique pages")
    return page_ids


def compute_next_occurrence(page_ids: np.ndarray) -> np.ndarray:
    n = len(page_ids)
    next_occ = np.full(n, -1, dtype=np.int64)
    last_seen: dict[int, int] = {}

    for t in range(n - 1, -1, -1):
        pid = int(page_ids[t])
        if pid in last_seen:
            next_occ[t] = last_seen[pid]
        last_seen[pid] = t

    return next_occ


def compute_reuse_distances(page_ids: np.ndarray, next_occ: np.ndarray, sentinel: int = 1_000_000) -> np.ndarray:
    n = len(page_ids)
    reuse_dist = np.full(n, sentinel, dtype=np.int64)

    for t in range(n):
        t_next = next_occ[t]
        if t_next == -1:
            continue  # sentinel already set

        if t + 1 >= t_next:
            reuse_dist[t] = 0
        else:
            reuse_dist[t] = len(set(page_ids[t + 1 : t_next].tolist()))

    return reuse_dist


def build_training_data(page_ids: np.ndarray, reuse_dist: np.ndarray, window_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    n = len(page_ids)
    num_samples = n - window_size + 1

    if num_samples <= 0:
        print(
            f"ERROR: trace length ({n}) < window_size ({window_size})",
            file=sys.stderr,
        )
        sys.exit(1)

    windows = np.zeros((num_samples, window_size), dtype=np.int32)
    labels = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        t = window_size - 1 + i
        windows[i] = page_ids[t - window_size + 1 : t + 1]
        labels[i] = float(np.log(reuse_dist[t] + 1))

    return windows, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reuse distances and build eviction training data"
    )
    parser.add_argument(
        "--trace", required=True,
        help="Input access trace CSV (from generate_access_trace.py)",
    )
    parser.add_argument(
        "--output", default="bench/tpch_data/eviction_training_data.npz",
        help="Output .npz file path",
    )
    parser.add_argument(
        "--window-size", type=int, default=64,
        help="Sliding window size (default: 64)",
    )
    parser.add_argument(
        "--sentinel", type=int, default=1_000_000,
        help="Sentinel reuse distance for never-reaccessed pages (default: 1000000)",
    )
    args = parser.parse_args()

    print("=== Reuse Distance Computation ===\n")

    # Phase 1: Load trace
    page_ids = load_trace(args.trace)

    # Phase 2: Precompute next occurrences
    print("\nComputing next occurrences ...")
    next_occ = compute_next_occurrence(page_ids)
    never_again = int(np.sum(next_occ == -1))
    print(f"  Pages never re-accessed: {never_again} ({100 * never_again / len(page_ids):.1f}%)")

    # Phase 3: Compute reuse distances
    print("\nComputing reuse distances ...")
    reuse_dist = compute_reuse_distances(page_ids, next_occ, args.sentinel)
    non_sentinel = reuse_dist[reuse_dist < args.sentinel]
    if len(non_sentinel) > 0:
        print(
            f"  Non-sentinel stats: mean={non_sentinel.mean():.1f}, "
            f"median={float(np.median(non_sentinel)):.0f}, max={non_sentinel.max()}"
        )
    print(f"  Sentinel count: {int(np.sum(reuse_dist >= args.sentinel))}")

    # Phase 4: Build training data
    print(f"\nBuilding training samples (window_size={args.window_size}) ...")
    windows, labels = build_training_data(page_ids, reuse_dist, args.window_size)
    print(f"  Samples: {len(windows)}")
    print(f"  Label range: [{labels.min():.2f}, {labels.max():.2f}]")
    print(f"  Unique pages in windows: {len(np.unique(windows))}")

    # Save
    np.savez_compressed(args.output, windows=windows, labels=labels)
    print(f"\nSaved to {args.output}")
    print(f"  windows: {windows.shape} ({windows.dtype})")
    print(f"  labels:  {labels.shape} ({labels.dtype})")

    # Gate check
    min_samples = 50_000
    if len(windows) >= min_samples:
        print(f"\nGate: PASSED ({len(windows)} >= {min_samples} samples)")
    else:
        print(f"\nGate: WARNING — {len(windows)} samples (target: >= {min_samples})")


if __name__ == "__main__":
    main()
