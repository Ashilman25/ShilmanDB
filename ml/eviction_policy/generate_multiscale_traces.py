#!/usr/bin/env python3

import argparse
import os
import random
import sys
from collections import Counter

import numpy as np

# Scale factor -> (num_pages, num_accesses)
SF_CONFIGS: dict[str, tuple[int, int]] = {
    "0.01": (1_200, 100_000),
    "0.1":  (12_000, 500_000),
    "1.0":  (120_000, 2_000_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-scale Zipfian access traces for V2 eviction model training"
    )
    parser.add_argument(
        "--output-dir", default="bench/tpch_data/traces/",
        help="Output directory for trace CSVs (default: bench/tpch_data/traces/)",
    )
    parser.add_argument("--alpha", type=float, default=0.99, help="Zipfian skew parameter (default: 0.99)")
    parser.add_argument("--burst-prob", type=float, default=0.15, help="Probability of burst access (default: 0.15)")
    parser.add_argument("--burst-len-min", type=int, default=5, help="Minimum burst length (default: 5)")
    parser.add_argument("--burst-len-max", type=int, default=20, help="Maximum burst length (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Root random seed (default: 42)")
    parser.add_argument(
        "--scale-factors", default="0.01,0.1,1.0",
        help="Comma-separated scale factors to generate (default: 0.01,0.1,1.0)",
    )
    return parser.parse_args()


def build_zipfian_weights(num_pages: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, num_pages + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    return weights


def generate_trace(num_pages: int, num_accesses: int, alpha: float, burst_prob: float, burst_len_min: int, burst_len_max: int, rng: random.Random) -> list[int]:

    weights = build_zipfian_weights(num_pages, alpha)
    pages = np.arange(num_pages)

    np_rng = np.random.RandomState(rng.randint(0, 2**31))
    zipfian_draws = np_rng.choice(pages, size=num_accesses, p=weights)
    draw_idx = 0

    trace: list[int] = []
    while len(trace) < num_accesses and draw_idx < num_accesses:
        if rng.random() < burst_prob:
            page = int(zipfian_draws[draw_idx])
            draw_idx += 1
            burst_len = rng.randint(burst_len_min, burst_len_max)
            for _ in range(burst_len):
                if len(trace) >= num_accesses:
                    break
                trace.append(page)
        else:
            trace.append(int(zipfian_draws[draw_idx]))
            draw_idx += 1

    return trace[:num_accesses]


def write_trace(trace: list[int], output_path: str) -> None:

    with open(output_path, "w") as f:
        f.write("timestamp,page_id\n")
        for t, page_id in enumerate(trace):
            f.write(f"{t},{page_id}\n")


def validate_trace(trace: list[int], num_pages: int, num_accesses: int, sf: str) -> dict:

    assert len(trace) == num_accesses, f"SF={sf}: expected {num_accesses} accesses, got {len(trace)}"

    unique_pages = len(set(trace))
    max_page_id = max(trace)
    min_page_id = min(trace)
    assert min_page_id >= 0, f"SF={sf}: negative page ID {min_page_id}"
    assert max_page_id < num_pages, f"SF={sf}: page ID {max_page_id} >= num_pages {num_pages}"

    counts = Counter(trace)
    top_10 = counts.most_common(10)
    top_10_pct = sum(c for _, c in top_10) / len(trace) * 100

    return {
        "sf": sf,
        "num_pages": num_pages,
        "target_accesses": num_accesses,
        "actual_accesses": len(trace),
        "unique_pages": unique_pages,
        "page_coverage": f"{100 * unique_pages / num_pages:.1f}%",
        "max_page_id": max_page_id,
        "top_10_pct": f"{top_10_pct:.1f}%",
    }


def print_summary_table(summaries: list[dict]) -> None:

    print("\n" + "=" * 72)
    print("SUMMARY ACROSS SCALE FACTORS")
    print("=" * 72)
    header = f"{'SF':>6}  {'Pages':>8}  {'Accesses':>10}  {'Unique':>8}  {'Coverage':>9}  {'Top-10':>7}"
    print(header)
    print("-" * 72)
    for s in summaries:
        row = (
            f"{s['sf']:>6}  {s['num_pages']:>8,}  {s['actual_accesses']:>10,}  "
            f"{s['unique_pages']:>8,}  {s['page_coverage']:>9}  {s['top_10_pct']:>7}"
        )
        print(row)
    print("=" * 72)


def main() -> None:
    args = parse_args()

    requested_sfs = [sf.strip() for sf in args.scale_factors.split(",")]
    for sf in requested_sfs:
        if sf not in SF_CONFIGS:
            sys.exit(f"ERROR: Unknown scale factor '{sf}'. Valid: {list(SF_CONFIGS.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Multi-Scale Zipfian Trace Generator ===\n")
    print(f"Scale factors: {requested_sfs}")
    print(f"Alpha: {args.alpha}, Burst: prob={args.burst_prob}, len=[{args.burst_len_min}, {args.burst_len_max}]")
    print(f"Root seed: {args.seed}")
    print(f"Output dir: {args.output_dir}\n")

    summaries: list[dict] = []

    for sf_idx, sf in enumerate(requested_sfs):
        num_pages, num_accesses = SF_CONFIGS[sf]
        sf_seed = args.seed + sf_idx  # order-dependent: same SF gets different seed if --scale-factors order changes
        rng = random.Random(sf_seed)

        print(f"--- SF={sf} (pages={num_pages:,}, accesses={num_accesses:,}, seed={sf_seed}) ---")

        trace = generate_trace(
            num_pages, num_accesses, args.alpha,
            args.burst_prob, args.burst_len_min, args.burst_len_max, rng,
        )

        stats = validate_trace(trace, num_pages, num_accesses, sf)

        output_path = os.path.join(args.output_dir, f"sf{sf}_trace.csv")
        write_trace(trace, output_path)

        file_size = os.path.getsize(output_path)
        size_label = f"{file_size / 1_048_576:.1f} MB" if file_size >= 1_048_576 else f"{file_size / 1024:.0f} KB"

        print(f"  Unique pages: {stats['unique_pages']:,} / {num_pages:,} ({stats['page_coverage']})")
        print(f"  Top-10 concentration: {stats['top_10_pct']}")
        print(f"  Output: {output_path} ({size_label})\n")

        summaries.append(stats)

    print_summary_table(summaries)
    print(f"\nAll {len(summaries)} traces generated successfully.")


if __name__ == "__main__":
    main()
