#!/usr/bin/env python3

import argparse
import csv
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_join_model import JoinOrderModel


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

    # Standardize with saved training stats
    stats = torch.load(os.path.join(model_dir, "feature_stats.pt"), weights_only=True)
    features = (features - stats["mean"]) / stats["std"]

    return features[val_idx], num_tables[val_idx], orders[val_idx]


def evaluate(model_dir: str, features: torch.Tensor, num_tables: torch.Tensor, orders: torch.Tensor):
    model = JoinOrderModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, "join_order_model_best.pt"), weights_only = True))
    model.eval()

    n_val = features.size(0)
    overall_correct = 0
    per_nt_correct: dict[int, int] = {}
    per_nt_total: dict[int, int] = {}

    with torch.no_grad():
        scores = model(features)

    for i in range(n_val):
        nt = num_tables[i].item()
        pred = torch.argsort(scores[i, :nt], descending=True)
        true = orders[i, :nt]
        correct = torch.equal(pred, true)

        if correct:
            overall_correct += 1
            
        per_nt_total[nt] = per_nt_total.get(nt, 0) + 1
        per_nt_correct[nt] = per_nt_correct.get(nt, 0) + (1 if correct else 0)

    overall_acc = overall_correct / n_val
    return overall_acc, per_nt_correct, per_nt_total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate join order prediction model")
    parser.add_argument("--data", default="bench/tpch_data/join_training_data.csv", help="Training CSV")
    parser.add_argument("--model-dir", default="ml/join_optimizer/models", help="Model directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    features, num_tables, orders = load_val_data(args.data, args.model_dir, args.val_split, args.seed)
    overall_acc, per_nt_correct, per_nt_total = evaluate(args.model_dir, features, num_tables, orders)

    print("=== Join Order Model Evaluation ===")
    print(f"Validation samples: {features.size(0)}")

    passed = overall_acc >= 0.80
    status = "PASS" if passed else "FAIL"
    print(f"\nOverall exact-match accuracy: {overall_acc:.3f} (target: >= 0.800) {status}")

    print("\nPer num_tables breakdown:")
    for nt in sorted(per_nt_total.keys()):
        c = per_nt_correct[nt]
        t = per_nt_total[nt]
        print(f"  {nt} tables: {c / t:.3f} ({c}/{t})")

    if not passed:
        print("\nWARNING: Accuracy below 80% target. Do not proceed to TorchScript export.")
        sys.exit(1)


if __name__ == "__main__":
    main()
