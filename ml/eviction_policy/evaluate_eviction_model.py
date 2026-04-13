#!/usr/bin/env python3

import argparse
import os
import random
import sys
from bisect import bisect_right
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_eviction_model import ReusePredictionModel


def load_val_data(data_path: str, model_dir: str, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    data = np.load(data_path)
    windows = torch.from_numpy(data["windows"]).long()

    n = len(windows)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * (1 - val_split))
    val_idx = sorted(indices[split:])

    config = torch.load(os.path.join(model_dir, "model_config.pt"), weights_only=True)
    max_pages = config["max_pages"]
    page_means = config.get("page_means", torch.zeros(max_pages))

    val_windows = windows[val_idx]

    return val_windows, page_means, max_pages


def load_model(model_dir: str, max_pages: int) -> ReusePredictionModel:
    model = ReusePredictionModel(max_pages=max_pages)
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "eviction_model_best.pt"), weights_only=True
        )
    )
    model.eval()
    return model


def load_trace(trace_path: str) -> np.ndarray:
    data = np.loadtxt(trace_path, delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1:
        return np.array([data[1]], dtype=np.int32)
    
    return data[:, 1].astype(np.int32)


def build_page_positions(page_ids: np.ndarray) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for t in range(len(page_ids)):
        pid = int(page_ids[t])
        if pid not in positions:
            positions[pid] = []
        positions[pid].append(t)
    return positions





def evaluate_val_mse(model: ReusePredictionModel, val_windows: torch.Tensor, page_means: torch.Tensor, max_pages: int) -> float:
    with torch.inference_mode():
        preds = model(val_windows)

    candidates = val_windows[:, -1]

    # Per-page mean predictions via scatter
    pred_sums = torch.zeros(max_pages, dtype=torch.float64)
    pred_counts = torch.zeros(max_pages, dtype=torch.float64)
    pred_sums.scatter_add_(0, candidates, preds.double())
    pred_counts.scatter_add_(0, candidates, torch.ones(len(candidates), dtype=torch.float64))

    seen = pred_counts > 0
    pred_means = (pred_sums[seen] / pred_counts[seen]).float()
    target_means = page_means[seen]

    return torch.nn.functional.mse_loss(pred_means, target_means).item()





def evaluate_ranking_accuracy(model: ReusePredictionModel,page_ids: np.ndarray, page_positions: dict[int, list[int]], pool_size: int, window_size: int = 64, max_evictions: int = 10_000) -> float:
    n = len(page_ids)
    resident: set[int] = set()
    access_history: list[int] = []
    matches = 0
    total_evictions = 0

    for t in range(n):
        pid = int(page_ids[t])
        access_history.append(pid)

        if pid in resident:
            continue  # hit, no eviction needed

        if len(resident) < pool_size:
            resident.add(pid)
            continue  # pool not full, no eviction

        # Pool full — eviction needed
        if total_evictions >= max_evictions:
            break

        # Belady's choice: resident page with farthest next access (O(log n) binary search)
        belady_victim = max(
            resident,
            key=lambda p: next_occ_bisect(page_positions, p, t, n),
        )

        # Model's choice: resident page with highest predicted reuse distance
        if len(access_history) >= window_size:
            base_window = access_history[-window_size:]
            model_victim = score_and_pick_victim(model, resident, base_window, window_size)
        else:
            # Not enough history — fall back to random
            model_victim = next(iter(resident))

        if model_victim == belady_victim:
            matches += 1

        total_evictions += 1
        if total_evictions % 500 == 0:
            print(f"    Ranking: {total_evictions}/{max_evictions} evictions evaluated...", flush=True)

        # Actually evict Belady's choice (simulate optimally for fair comparison)
        resident.discard(belady_victim)
        resident.add(pid)

    if total_evictions == 0:
        return 0.0
    return matches / total_evictions


def next_occ_bisect(page_positions: dict[int, list[int]], page: int, current_t: int, n: int) -> int:
    positions = page_positions.get(page)
    if positions is None:
        return n
    
    idx = bisect_right(positions, current_t)
    if idx < len(positions):
        return positions[idx]
    
    return n  # never accessed again


def score_and_pick_victim(model: ReusePredictionModel, resident: set[int], base_window: list[int], window_size: int) -> int:
    pages = list(resident)
    # Build prefix once, reuse for all candidates
    prefix = base_window[-(window_size - 1):]
    windows_batch = torch.zeros(len(pages), window_size, dtype=torch.long)
    prefix_tensor = torch.tensor(prefix, dtype=torch.long)
    windows_batch[:, :len(prefix)] = prefix_tensor
    for i, p in enumerate(pages):
        windows_batch[i, -1] = p

    with torch.inference_mode():
        scores = model(windows_batch)

    best_idx = int(scores.argmax())
    return pages[best_idx]





def simulate_lru(page_ids: np.ndarray, pool_size: int) -> float:
    pool: OrderedDict[int, None] = OrderedDict()
    hits = 0

    for t in range(len(page_ids)):
        pid = int(page_ids[t])
        if pid in pool:
            hits += 1
            pool.move_to_end(pid)
        else:
            if len(pool) >= pool_size:
                pool.popitem(last=False)  # evict LRU
            pool[pid] = None

    return hits / len(page_ids)


def simulate_belady(page_ids: np.ndarray, page_positions: dict[int, list[int]], pool_size: int) -> float:

    n = len(page_ids)
    resident: set[int] = set()
    hits = 0

    for t in range(n):
        pid = int(page_ids[t])
        if pid in resident:
            hits += 1
        else:
            if len(resident) >= pool_size:
                victim = max(
                    resident,
                    key=lambda p: next_occ_bisect(page_positions, p, t, n),
                )
                resident.discard(victim)
            resident.add(pid)

        if (t + 1) % 10000 == 0:
            print(f"    Belady sim: {t + 1}/{n} accesses...", flush=True)

    return hits / n


def simulate_learned(model: ReusePredictionModel, page_ids: np.ndarray, pool_size: int, window_size: int = 64) -> float:

    resident: OrderedDict[int, None] = OrderedDict()
    access_history: list[int] = []
    hits = 0

    for t in range(len(page_ids)):
        pid = int(page_ids[t])
        access_history.append(pid)

        if pid in resident:
            hits += 1
            resident.move_to_end(pid)
        else:
            if len(resident) >= pool_size:
                if len(access_history) >= window_size:
                    base_window = access_history[-window_size:]
                    victim = score_and_pick_victim(
                        model, set(resident.keys()), base_window, window_size
                    )
                else:
                    # LRU fallback before window is full
                    victim = next(iter(resident))
                del resident[victim]
            resident[pid] = None

        if (t + 1) % 5000 == 0:
            print(f"    Learned sim: {t + 1}/{len(page_ids)} accesses...", flush=True)

    return hits / len(page_ids)





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate eviction model: MSE, ranking accuracy, hit rate"
    )
    parser.add_argument(
        "--data", default="bench/tpch_data/eviction_training_data.npz",
        help="Training data .npz (for validation split)",
    )
    parser.add_argument(
        "--trace", default="bench/tpch_data/zipfian_access_trace.csv",
        help="Raw access trace CSV (for hit rate simulation)",
    )
    parser.add_argument(
        "--model-dir", default="ml/eviction_policy/models",
        help="Directory containing model artifacts",
    )
    parser.add_argument("--pool-size", type=int, default=256, help="Buffer pool size for simulation")
    parser.add_argument("--max-evictions", type=int, default=10_000, help="Max eviction events for ranking accuracy")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sim-trace-limit", type=int, default=100_000,
        help="Limit trace length for hit rate simulation (0 = full trace)",
    )
    args = parser.parse_args()

    print("=== Eviction Model Evaluation ===\n")

    # Load model
    val_windows, page_means, max_pages = load_val_data(
        args.data, args.model_dir, args.val_split, args.seed
    )
    model = load_model(args.model_dir, max_pages)
    print(f"Model loaded (max_pages={max_pages})")
    print(f"Validation samples: {len(val_windows)}\n")

    all_passed = True

    # ── Metric 1: Per-page Mean MSE ──
    print("--- Metric 1: Per-page Mean MSE ---")
    val_mse = evaluate_val_mse(model, val_windows, page_means, max_pages)
    mse_pass = val_mse < 2.0
    print(f"  Per-page mean MSE: {val_mse:.4f} (target: < 2.0) {'PASS' if mse_pass else 'FAIL'}")
    if not mse_pass:
        all_passed = False

    # ── Metric 2: Ranking Accuracy ──
    print("\n--- Metric 2: Ranking Accuracy vs Belady ---")
    page_ids = load_trace(args.trace)
    page_positions = build_page_positions(page_ids)
    rank_acc = evaluate_ranking_accuracy(
        model, page_ids, page_positions, args.pool_size,
        max_evictions=args.max_evictions,
    )
    rank_pass = rank_acc >= 0.02
    print(f"  Ranking accuracy: {rank_acc:.3f} (target: >= 0.020) {'PASS' if rank_pass else 'FAIL'}")
    if not rank_pass:
        all_passed = False

    # ── Metric 3: Hit Rate Simulation ──
    print("\n--- Metric 3: Hit Rate Simulation ---")
    sim_trace = page_ids
    if args.sim_trace_limit > 0 and len(page_ids) > args.sim_trace_limit:
        sim_trace = page_ids[:args.sim_trace_limit]
        print(f"  (Using first {args.sim_trace_limit} trace entries for simulation)")

    lru_hit_rate = simulate_lru(sim_trace, args.pool_size)
    print(f"  LRU hit rate: {lru_hit_rate:.4f} ({lru_hit_rate * 100:.2f}%)")

    sim_positions = build_page_positions(sim_trace)
    belady_hit_rate = simulate_belady(sim_trace, sim_positions, args.pool_size)
    print(f"  Belady hit rate: {belady_hit_rate:.4f} ({belady_hit_rate * 100:.2f}%)")
    if lru_hit_rate > 0:
        max_improvement = (belady_hit_rate - lru_hit_rate) / lru_hit_rate * 100
        print(f"  Theoretical ceiling: {max_improvement:+.2f}% over LRU")

    learned_hit_rate = simulate_learned(model, sim_trace, args.pool_size)
    print(f"  Learned hit rate: {learned_hit_rate:.4f} ({learned_hit_rate * 100:.2f}%)")

    if lru_hit_rate > 0:
        improvement = (learned_hit_rate - lru_hit_rate) / lru_hit_rate * 100
    else:
        improvement = 100.0 if learned_hit_rate > 0 else 0.0

    hit_pass = improvement >= 3.0
    print(f"  Relative improvement: {improvement:+.2f}% (target: >= +3.00%) {'PASS' if hit_pass else 'FAIL'}")
    if not hit_pass:
        all_passed = False

    # ── Summary ──
    print(f"\n=== Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'} ===")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
