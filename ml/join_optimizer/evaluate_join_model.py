#!/usr/bin/env python3

import argparse
import csv
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_join_model import create_model


def estimate_cost(order: list[int], features: list[float], num_tables: int) -> float:
    rows = [math.exp(features[t * 3]) for t in range(num_tables)]

    selectivities: dict[tuple[int, int], float] = {}
    pair_idx = 0
    for i in range(6):
        for j in range(i + 1, 6):
            offset = 18 + pair_idx * 2
            if features[offset] > 0.5:
                selectivities[(i, j)] = features[offset + 1]
            pair_idx += 1

    total_cost = rows[order[0]]
    current_size = rows[order[0]]
    joined = {order[0]}
    
    for k in range(1, num_tables):
        t = order[k]
        sel = 1.0
        for prev in joined:
            pair = (min(prev, t), max(prev, t))
            if pair in selectivities:
                sel *= selectivities[pair]
                
        current_size = current_size * rows[t] * sel
        total_cost += current_size
        joined.add(t)
        
    return total_cost


def load_val_data(csv_path: str, model_dir: str, val_split: float, seed: int):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)

    n = len(rows)
    features = torch.zeros(n, 48)
    num_tables = torch.zeros(n, dtype=torch.long)
    orders = torch.zeros(n, 6, dtype=torch.long)

    for i, row in enumerate(rows):
        for j in range(48):
            features[i, j] = float(row[j])
        num_tables[i] = int(row[48])
        for j in range(6):
            orders[i, j] = int(row[49 + j])

    # Reproduce identical stratified split from training
    rng = random.Random(seed)
    val_idx = []
    for nt in sorted(set(num_tables.tolist())):
        group = [i for i in range(n) if num_tables[i].item() == nt]
        rng.shuffle(group)
        split = int(len(group) * (1 - val_split))
        val_idx.extend(group[split:])
    val_idx = sorted(val_idx)

    # Keep raw features for cost estimation before standardizing
    raw_features = features[val_idx].clone()

    # Standardize with saved training stats
    stats = torch.load(os.path.join(model_dir, "feature_stats.pt"), weights_only=True)
    features = (features - stats["mean"]) / stats["std"]
    variant = stats.get("variant", "baseline")

    return features[val_idx], num_tables[val_idx], orders[val_idx], raw_features, variant


def evaluate(model_dir: str, features: torch.Tensor, num_tables: torch.Tensor, orders: torch.Tensor, raw_features: torch.Tensor, variant: str) -> dict:
    model = create_model(variant)
    model.load_state_dict(torch.load(os.path.join(model_dir, "join_order_model_best.pt"), weights_only=True))
    model.eval()

    n_val = features.size(0)

    # Per-num_tables accumulators
    per_nt: dict[int, dict] = {}

    # Global accumulators
    total_exact = 0
    total_prefix2 = 0
    n_prefix2 = 0
    regrets: list[float] = []

    with torch.inference_mode():
        scores = model(features)

    for i in range(n_val):
        nt = num_tables[i].item()
        pred = torch.argsort(scores[i, :nt], descending=True)
        true = orders[i, :nt]

        # Initialize per-nt bucket
        if nt not in per_nt:
            per_nt[nt] = {"exact": 0, "prefix2": 0, "n_prefix2": 0, "regrets": [], "total": 0}
            
        per_nt[nt]["total"] += 1

        exact = torch.equal(pred, true)
        if exact:
            total_exact += 1
            per_nt[nt]["exact"] += 1

        if nt >= 2:
            raw_feat = raw_features[i].tolist()
            pred_order = pred.tolist()
            true_order = true.tolist()
            cost_pred = estimate_cost(pred_order, raw_feat, nt)
            cost_opt = estimate_cost(true_order, raw_feat, nt)
            regret = cost_pred / max(cost_opt, 1e-12) - 1.0
            regrets.append(regret)
            per_nt[nt]["regrets"].append(regret)

        if nt >= 2:
            n_prefix2 += 1
            per_nt[nt]["n_prefix2"] += 1
            if pred[0] == true[0] and pred[1] == true[1]:
                total_prefix2 += 1
                per_nt[nt]["prefix2"] += 1

    # summary statistics
    exact_acc = total_exact / n_val
    prefix2_acc = total_prefix2 / n_prefix2 if n_prefix2 > 0 else 0.0

    regrets_sorted = sorted(regrets)
    n_regrets = len(regrets_sorted)
    if n_regrets > 0:
        mean_regret = sum(regrets_sorted) / n_regrets
        mid = n_regrets // 2
        median_regret = regrets_sorted[mid] if n_regrets % 2 == 1 else (regrets_sorted[mid - 1] + regrets_sorted[mid]) / 2
        p95_regret = regrets_sorted[min(int(n_regrets * 0.95), n_regrets - 1)]
        max_regret = regrets_sorted[-1]
    else:
        mean_regret = median_regret = p95_regret = max_regret = 0.0

    return {
        "n_val": n_val,
        "exact_acc": exact_acc,
        "prefix2_acc": prefix2_acc,
        "mean_regret": mean_regret,
        "median_regret": median_regret,
        "p95_regret": p95_regret,
        "max_regret": max_regret,
        "per_nt": per_nt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate join order prediction model (enhanced)")
    parser.add_argument("--data", default="bench/tpch_data/join_training_data.csv", help="Training CSV")
    parser.add_argument("--model-dir", default="ml/join_optimizer/models", help="Model directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    features, num_tables, orders, raw_features, variant = load_val_data(args.data, args.model_dir, args.val_split, args.seed)
    results = evaluate(args.model_dir, features, num_tables, orders, raw_features, variant)

    print("=== Join Order Model Evaluation (Enhanced) ===")
    print(f"Model variant: {variant}")
    print(f"Validation samples: {results['n_val']}")

    all_passed = True

    # Metric 1
    exact_pass = results["exact_acc"] >= 0.85
    if not exact_pass:
        all_passed = False
    print(f"\nMetric 1: Exact-match accuracy")
    print(f"  Overall: {results['exact_acc']:.3f} (target: >= 0.850) {'PASS' if exact_pass else 'FAIL'}")

    # Metric 2
    regret_pass = results["mean_regret"] < 0.10
    if not regret_pass:
        all_passed = False
    print(f"\nMetric 2: Mean cost regret")
    print(f"  Mean: {results['mean_regret']:.3f} | Median: {results['median_regret']:.3f} "
          f"| P95: {results['p95_regret']:.3f} | Max: {results['max_regret']:.3f}")
    print(f"  Gate: < 0.10  {'PASS' if regret_pass else 'FAIL'}")

    # Metric 3
    prefix2_pass = results["prefix2_acc"] >= 0.92
    if not prefix2_pass:
        all_passed = False
    print(f"\nMetric 3: Prefix-2 accuracy")
    print(f"  Overall: {results['prefix2_acc']:.3f} (target: >= 0.920) {'PASS' if prefix2_pass else 'FAIL'}")

    # Per-num_tables breakdown
    print(f"\nPer num_tables breakdown:")
    for nt in sorted(results["per_nt"].keys()):
        bucket = results["per_nt"][nt]
        total = bucket["total"]
        exact = bucket["exact"] / total if total > 0 else 0.0
        p2 = bucket["prefix2"] / bucket["n_prefix2"] if bucket["n_prefix2"] > 0 else float("nan")
        reg = sum(bucket["regrets"]) / len(bucket["regrets"]) if bucket["regrets"] else float("nan")
        print(f"  {nt} tables: exact={exact:.2f}  prefix2={p2:.2f}  regret={reg:.2f}  (N={total})")

    # Summary
    print(f"\n=== Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'} ===")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
