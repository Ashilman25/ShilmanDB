#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ReusePredictionModelV2(nn.Module):

    def __init__(self, num_features: int = 4):
        super().__init__()
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V2 feature-based eviction reuse distance prediction model"
    )
    parser.add_argument(
        "--data", default="bench/tpch_data/eviction_features_training_data.npz",
        help="Training data .npz file (default: bench/tpch_data/eviction_features_training_data.npz)",
    )
    parser.add_argument(
        "--output-dir", default="ml/eviction_policy/models",
        help="Output directory for model artifacts (default: ml/eviction_policy/models)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (default: 0)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    parser.add_argument(
        "--lr-schedule", choices=["cosine", "constant"], default="cosine",
        help="LR schedule: cosine or constant (default: cosine)",
    )
    parser.add_argument(
        "--loss", choices=["mse", "huber"], default="huber",
        help="Loss function: mse or huber (default: huber)",
    )
    parser.add_argument(
        "--label-cap", type=float, default=9.21,
        help="Cap labels at this value, 0 = no cap (default: 9.21 ~ log(10001))",
    )
    return parser.parse_args()


def load_and_split(data_path: str, val_split: float, seed: int, label_cap: float = 0.0) -> dict:
    print(f"Loading data from {data_path} ...")
    data = np.load(data_path)
    features = data["features"]   # (N, 4) float32
    labels = data["labels"]       # (N,) float32
    sf_labels = data["sf_labels"] # (N,) float32

    unique_sfs = np.unique(sf_labels)
    n_total = len(features)
    print(f"  {n_total:,} samples, {len(unique_sfs)} scale factors")

    # Stratified split: 80/20 per scale factor
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    rng = random.Random(seed)

    for sf_val in unique_sfs:
        sf_mask = sf_labels == sf_val
        sf_idx = np.where(sf_mask)[0]
        idx_list = sf_idx.tolist()
        rng.shuffle(idx_list)
        split_point = int(len(idx_list) * (1 - val_split))
        train_indices.append(np.array(sorted(idx_list[:split_point]), dtype=np.int64))
        val_indices.append(np.array(sorted(idx_list[split_point:]), dtype=np.int64))

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)

    train_features = torch.from_numpy(features[train_idx]).float()
    train_labels = torch.from_numpy(labels[train_idx]).float()
    val_features = torch.from_numpy(features[val_idx]).float()
    val_labels = torch.from_numpy(labels[val_idx]).float()

    if label_cap > 0:
        n_capped_train = int((train_labels > label_cap).sum())
        n_capped_val = int((val_labels > label_cap).sum())
        train_labels = torch.clamp(train_labels, max=label_cap)
        val_labels = torch.clamp(val_labels, max=label_cap)
        print(f"  Label cap={label_cap:.2f}: capped {n_capped_train:,} train, {n_capped_val:,} val samples")

    print(f"  Stratified split: {len(train_idx):,} train, {len(val_idx):,} val")

    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
    }


def standardize_features(data: dict, output_dir: str) -> dict:
    train_features = data["train_features"]
    val_features = data["val_features"]

    train_mean = train_features.mean(dim=0)  # (4,)
    train_std = train_features.std(dim=0)    # (4,)
    train_std[train_std == 0] = 1.0          # is_dirty is constant 0.0

    feat_names = ["log_recency", "log_frequency", "log_recency_2", "is_dirty"]
    print("\nFeature standardization (train set):")
    for i, name in enumerate(feat_names):
        replaced = " (replaced zero-std)" if data["train_features"][:, i].std() == 0 else ""
        print(f"  {name:15s}: mean={train_mean[i]:.3f}, std={train_std[i]:.3f}{replaced}")

    data["train_features"] = (train_features - train_mean) / train_std
    data["val_features"] = (val_features - train_mean) / train_std
    data["feature_mean"] = train_mean
    data["feature_std"] = train_std

    # Save feature stats
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "feature_stats_v2.pt")
    torch.save({"mean": train_mean, "std": train_std}, stats_path)

    return data


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data = load_and_split(args.data, args.val_split, args.seed, args.label_cap)
    data = standardize_features(data, args.output_dir)

    model = ReusePredictionModelV2()
    n_params = sum(p.numel() for p in model.parameters())
    n_train = len(data["train_features"])
    n_val = len(data["val_features"])

    print(f"\n=== V2 Eviction Model Training ===")
    print(f"Model: MLP(4->128->64->1) -- {n_params:,} params")
    print(
        f"Hyperparams: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, schedule={args.lr_schedule}, patience={args.patience}, "
        f"loss={args.loss}, label_cap={args.label_cap}\n"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    train_dataset = TensorDataset(data["train_features"], data["train_labels"])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )

    val_features = data["val_features"]
    val_labels = data["val_labels"]

    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_features, batch_labels in train_loader:
            preds = model(batch_features)
            loss = criterion(preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(val_features)
            val_mse = criterion(val_preds, val_labels).item()

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3d}/{args.epochs}  "
                f"train_mse={avg_train_loss:.4f}  "
                f"val_mse={val_mse:.4f}  "
                f"lr={current_lr:.2e}"
            )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "eviction_model_v2_best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}, best epoch={best_epoch})")
                break

    print(f"\nBest validation MSE: {best_val_mse:.4f} (epoch {best_epoch})")
    print(f"Model saved to {os.path.join(args.output_dir, 'eviction_model_v2_best.pt')}")
    print(f"Stats saved to {os.path.join(args.output_dir, 'feature_stats_v2.pt')}")

    if best_val_mse < 2.0:
        print(f"\nGate: PASSED (val MSE {best_val_mse:.4f} < 2.0)")
    else:
        print(f"\nGate: WARNING (val MSE {best_val_mse:.4f} >= 2.0)")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
