#!/usr/bin/env python3


import argparse
import random

import numpy as np


def build_zipfian_weights(num_pages: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, num_pages + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    
    return weights


def generate_trace(num_pages: int, alpha: float, num_accesses: int, burst_prob: float, burst_len_min: int, burst_len_max: int, rng: random.Random) -> list[int]:
    weights = build_zipfian_weights(num_pages, alpha)
    pages = np.arange(num_pages)

    # Pre-draw all Zipfian samples for efficiency
    np_rng = np.random.RandomState(rng.randint(0, 2**31))
    zipfian_draws = np_rng.choice(pages, size=num_accesses, p=weights)
    draw_idx = 0

    trace: list[int] = []

    while len(trace) < num_accesses and draw_idx < num_accesses:
        if rng.random() < burst_prob:
            # Burst: access the same page multiple times consecutively
            page = int(zipfian_draws[draw_idx])
            draw_idx += 1
            burst_len = rng.randint(burst_len_min, burst_len_max)
            for _ in range(burst_len):
                if len(trace) >= num_accesses:
                    break
                trace.append(page)
        else:
            # Single Zipfian draw
            trace.append(int(zipfian_draws[draw_idx]))
            draw_idx += 1

    return trace[:num_accesses]


def write_trace(trace: list[int], output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write("timestamp,page_id\n")
        for t, page_id in enumerate(trace):
            f.write(f"{t},{page_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Zipfian page access trace"
    )
    parser.add_argument(
        "--output", default="bench/tpch_data/zipfian_access_trace.csv",
        help="Output CSV path (default: bench/tpch_data/zipfian_access_trace.csv)",
    )
    parser.add_argument("--num-pages", type=int, default=4096, help="Number of pages (default: 4096)")
    parser.add_argument("--alpha", type=float, default=0.99, help="Zipfian skew parameter (default: 0.99)")
    parser.add_argument("--num-accesses", type=int, default=500_000, help="Total accesses to generate (default: 500000)")
    parser.add_argument("--burst-prob", type=float, default=0.15, help="Probability of burst (default: 0.15)")
    parser.add_argument("--burst-len-min", type=int, default=5, help="Minimum burst length (default: 5)")
    parser.add_argument("--burst-len-max", type=int, default=20, help="Maximum burst length (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=== Zipfian Trace Generator ===\n")
    print(f"Pages: {args.num_pages}, alpha: {args.alpha}")
    print(f"Target accesses: {args.num_accesses}")
    print(f"Burst: prob={args.burst_prob}, len=[{args.burst_len_min}, {args.burst_len_max}]")

    trace = generate_trace(
        args.num_pages, args.alpha, args.num_accesses,
        args.burst_prob, args.burst_len_min, args.burst_len_max, rng,
    )

    write_trace(trace, args.output)

    # Stats
    unique = len(set(trace))
    print(f"\nGenerated {len(trace)} accesses, {unique} unique pages")
    print(f"Output: {args.output}")

    # Quick distribution check
    from collections import Counter
    counts = Counter(trace)
    top_10 = counts.most_common(10)
    total = len(trace)
    top_10_pct = sum(c for _, c in top_10) / total * 100
    print(f"Top 10 pages account for {top_10_pct:.1f}% of accesses")


if __name__ == "__main__":
    main()
