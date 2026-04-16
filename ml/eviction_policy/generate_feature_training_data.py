#!/usr/bin/env python3

import argparse
import math
import os
import random
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract feature-based training data for V2 eviction model"
    )
    parser.add_argument(
        "--traces-dir", default="bench/tpch_data/traces/",
        help="Directory containing SF trace CSVs (default: bench/tpch_data/traces/)",
    )
    parser.add_argument(
        "--output", default="bench/tpch_data/eviction_features_training_data.npz",
        help="Output .npz file path (default: bench/tpch_data/eviction_features_training_data.npz)",
    )
    parser.add_argument(
        "--pool-sizes", default="128,512,2048",
        help="Comma-separated pool sizes, one per SF in SF order (default: 128,512,2048)",
    )
    parser.add_argument(
        "--scale-factors", default="0.01,0.1,1.0",
        help="Comma-separated scale factors to process (default: 0.01,0.1,1.0)",
    )
    parser.add_argument(
        "--max-samples-per-sf", type=int, default=500_000,
        help="Maximum training samples per scale factor (default: 500000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling (default: 42)")
    return parser.parse_args()


def load_trace(trace_path: str) -> np.ndarray:
    data = np.loadtxt(trace_path, delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1:
        return np.array([data[1]], dtype=np.int32)
    return data[:, 1].astype(np.int32)


def compute_next_occurrence(page_ids: np.ndarray) -> np.ndarray:
    n = len(page_ids)
    next_occ = np.full(n, n, dtype=np.int64)
    last_seen: dict[int, int] = {}

    for t in range(n - 1, -1, -1):
        pid = int(page_ids[t])
        if pid in last_seen:
            next_occ[t] = last_seen[pid]
        last_seen[pid] = t

    return next_occ


def simulate_buffer_pool(page_ids: np.ndarray, next_occ: np.ndarray, pool_size: int, max_samples: int, rng: random.Random) -> tuple[list[list[float]], list[float], int, int, float]:
    n = len(page_ids)
    log = math.log
    max_events = max(1, max_samples // pool_size)

    # Per-page tracking state
    resident: set[int] = set()
    last_access_ts: dict[int, int] = {}
    second_last_access_ts: dict[int, int] = {}
    access_count: dict[int, int] = {}
    last_trace_pos: dict[int, int] = {}

    global_clock = 0
    # Reservoir of sampled eviction events: (event_features, event_labels)
    reservoir: list[tuple[list[list[float]], list[float]]] = []
    hits = 0
    eviction_count = 0

    for t in range(n):
        page = int(page_ids[t])
        global_clock += 1

        if page in resident:
            # Hit — update tracking state
            hits += 1
            second_last_access_ts[page] = last_access_ts[page]
            last_access_ts[page] = global_clock
            access_count[page] = access_count.get(page, 0) + 1
            last_trace_pos[page] = t
            continue

        # Miss — check if eviction needed
        if len(resident) >= pool_size:
            eviction_count += 1

            # Reservoir sampling: decide whether to collect features for this event
            sample_this = False
            reservoir_idx = -1
            if len(reservoir) < max_events:
                sample_this = True
            else:
                j = rng.randint(0, eviction_count - 1)
                if j < max_events:
                    sample_this = True
                    reservoir_idx = j

            event_features: list[list[float]] = []
            event_labels: list[float] = []
            best_victim = -1
            best_forward_dist = -1

            for cand in resident:
                cand_trace_pos = last_trace_pos[cand]
                cand_next = int(next_occ[cand_trace_pos])

                if sample_this:
                    cand_last_ts = last_access_ts[cand]
                    cand_second_ts = second_last_access_ts.get(cand, 0)
                    cand_count = access_count[cand]

                    log_recency = log(1 + (global_clock - cand_last_ts))
                    log_frequency = log(1 + cand_count)
                    log_recency_2 = log(1 + (global_clock - cand_second_ts))
                    is_dirty = 0.0

                    event_features.append([log_recency, log_frequency, log_recency_2, is_dirty])

                    if cand_next >= n:
                        forward_dist = 1_000_000
                    else:
                        forward_dist = max(1, cand_next - t)
                    event_labels.append(log(forward_dist + 1))

                # Track Belady victim (always needed for simulation correctness)
                if cand_next > best_forward_dist:
                    best_forward_dist = cand_next
                    best_victim = cand

            if sample_this:
                if reservoir_idx >= 0:
                    reservoir[reservoir_idx] = (event_features, event_labels)
                else:
                    reservoir.append((event_features, event_labels))

            # Evict Belady victim — clean up tracking state
            resident.remove(best_victim)
            del last_access_ts[best_victim]
            if best_victim in second_last_access_ts:
                del second_last_access_ts[best_victim]
            del access_count[best_victim]
            del last_trace_pos[best_victim]

        # Insert new page
        resident.add(page)
        last_access_ts[page] = global_clock
        access_count[page] = 1
        last_trace_pos[page] = t

    # Flatten reservoir into feature/label lists
    features_list: list[list[float]] = []
    labels_list: list[float] = []
    for ef, el in reservoir:
        features_list.extend(ef)
        labels_list.extend(el)

    hit_rate = hits / n if n > 0 else 0.0
    return features_list, labels_list, hits, eviction_count, hit_rate


def print_sf_stats(sf: str, pool_size: int, accesses: int, hits: int, evictions: int, hit_rate: float, samples_before: int, samples_after: int, features_arr: np.ndarray, labels_arr: np.ndarray) -> dict:
    print(f"  Accesses: {accesses:,}, Hits: {hits:,}, Hit rate: {hit_rate:.1%}")
    print(f"  Eviction events: {evictions:,}")
    print(f"  Samples: {samples_before:,} -> {samples_after:,} (after cap)")

    feat_names = ["log_recency", "log_frequency", "log_recency_2", "is_dirty"]
    for i, name in enumerate(feat_names):
        col = features_arr[:, i]
        print(f"    {name}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")

    print(f"    labels: min={labels_arr.min():.3f}, max={labels_arr.max():.3f}, mean={labels_arr.mean():.3f}")

    return {
        "sf": sf, "pool_size": pool_size, "accesses": accesses,
        "hit_rate": f"{hit_rate:.1%}", "evictions": evictions,
        "samples": samples_after,
    }


def print_summary_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY ACROSS SCALE FACTORS")
    print("=" * 78)
    header = f"{'SF':>6}  {'Pool':>6}  {'Accesses':>10}  {'Hit Rate':>9}  {'Evictions':>10}  {'Samples':>10}"
    print(header)
    print("-" * 78)
    for s in summaries:
        row = (
            f"{s['sf']:>6}  {s['pool_size']:>6}  {s['accesses']:>10,}  "
            f"{s['hit_rate']:>9}  {s['evictions']:>10,}  {s['samples']:>10,}"
        )
        print(row)
    print("=" * 78)


def main() -> None:
    args = parse_args()

    scale_factors = [sf.strip() for sf in args.scale_factors.split(",")]
    pool_sizes = [int(p.strip()) for p in args.pool_sizes.split(",")]

    if len(scale_factors) != len(pool_sizes):
        sys.exit(f"ERROR: {len(scale_factors)} scale factors but {len(pool_sizes)} pool sizes")

    rng = random.Random(args.seed)

    print("=== V2 Feature Training Data Extraction ===\n")
    print(f"Scale factors: {scale_factors}")
    print(f"Pool sizes: {pool_sizes}")
    print(f"Max samples per SF: {args.max_samples_per_sf:,}")
    print(f"Seed: {args.seed}\n")

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sf_labels: list[np.ndarray] = []
    summaries: list[dict] = []

    for sf, pool_size in zip(scale_factors, pool_sizes):
        trace_path = os.path.join(args.traces_dir, f"sf{sf}_trace.csv")
        if not os.path.exists(trace_path):
            sys.exit(f"ERROR: Trace file not found: {trace_path}")

        print(f"--- SF={sf} (pool_size={pool_size}) ---")
        print(f"  Loading {trace_path} ...")

        page_ids = load_trace(trace_path)
        print(f"  {len(page_ids):,} accesses, {len(np.unique(page_ids)):,} unique pages")

        print("  Computing next-occurrence array ...")
        next_occ = compute_next_occurrence(page_ids)

        print("  Simulating buffer pool (reservoir sampling) ...")
        features, labels, hits, evictions, hit_rate = simulate_buffer_pool(
            page_ids, next_occ, pool_size, args.max_samples_per_sf, rng,
        )

        samples_before = evictions * pool_size
        samples_after = len(features)

        features_arr = np.array(features, dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.float32)
        sf_labels_arr = np.full(samples_after, float(sf), dtype=np.float32)

        stats = print_sf_stats(
            sf, pool_size, len(page_ids), hits, evictions, hit_rate,
            samples_before, samples_after, features_arr, labels_arr,
        )
        summaries.append(stats)

        all_features.append(features_arr)
        all_labels.append(labels_arr)
        all_sf_labels.append(sf_labels_arr)
        print()

    # Combine all SFs
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_sf_labels = np.concatenate(all_sf_labels, axis=0)

    print_summary_table(summaries)

    total = len(combined_features)
    print(f"\nTotal combined samples: {total:,}")
    print(f"  features: {combined_features.shape} ({combined_features.dtype})")
    print(f"  labels:   {combined_labels.shape} ({combined_labels.dtype})")
    print(f"  sf_labels: {combined_sf_labels.shape} ({combined_sf_labels.dtype})")

    # Gate check
    min_samples = 100_000
    if total >= min_samples:
        print(f"\nGate: PASSED ({total:,} >= {min_samples:,} combined samples)")
    else:
        print(f"\nGate: WARNING — {total:,} combined samples (target: >= {min_samples:,})")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        features=combined_features,
        labels=combined_labels,
        sf_labels=combined_sf_labels,
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
