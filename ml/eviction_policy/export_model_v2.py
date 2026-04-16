#!/usr/bin/env python3

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from train_eviction_model_v2 import ReusePredictionModelV2


class StandardizedModelV2(nn.Module):
    
    """Wraps V2 model with baked feature standardization.

    C++ passes raw log1p(...) features — this module normalizes
    before feeding to the prediction network.
    """

    def __init__(self, model: ReusePredictionModelV2, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export V2 eviction model to TorchScript with baked standardization"
    )
    parser.add_argument(
        "--model-dir", default="ml/eviction_policy/models",
        help="Directory containing V2 model artifacts (default: ml/eviction_policy/models)",
    )
    args = parser.parse_args()

    # Load trained model
    model = ReusePredictionModelV2()
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "eviction_model_v2_best.pt"),
            weights_only=True,
        )
    )
    model.eval()

    # Load feature stats
    stats = torch.load(
        os.path.join(args.model_dir, "feature_stats_v2.pt"),
        weights_only=True,
    )
    feat_mean = stats["mean"]  # (4,)
    feat_std = stats["std"]    # (4,)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Feature stats: mean={feat_mean.tolist()}, std={feat_std.tolist()}")

    # Wrap with baked standardization
    wrapper = StandardizedModelV2(model, feat_mean, feat_std)
    wrapper.eval()

    # Export via TorchScript
    scripted = torch.jit.script(wrapper)
    output_path = os.path.join(args.model_dir, "eviction_model_v2.pt")
    scripted.save(output_path)
    print(f"\nExported TorchScript model to {output_path}")

    # ── Verification ──────────────────────────────────────────────────

    print("\n=== Post-Export Verification ===\n")
    checks_passed = 0

    # Check 1: Reload
    print("Check 1: Reload exported model ... ", end="", flush=True)
    try:
        loaded = torch.jit.load(output_path)
        print("PASS")
        checks_passed += 1
    except Exception as e:
        print(f"FAIL — {e}")
        sys.exit(1)

    # Check 2: Output shape
    print("Check 2: Output shape ... ", end="", flush=True)
    test_input = torch.randn(5, 4)
    with torch.inference_mode():
        test_output = loaded(test_input)
    if test_output.shape == (5,):
        print(f"PASS — input (5, 4) → output {tuple(test_output.shape)}")
        checks_passed += 1
    else:
        print(f"FAIL — expected (5,), got {tuple(test_output.shape)}")
        sys.exit(1)

    # Check 3: Baked standardization cross-check
    print("Check 3: Baked standardization cross-check ... ", end="", flush=True)
    raw_input = torch.randn(3, 4)
    with torch.inference_mode():
        baked_output = loaded(raw_input)
        manual_normalized = (raw_input - feat_mean) / feat_std
        manual_output = model(manual_normalized)
    if torch.allclose(baked_output, manual_output, atol=1e-5):
        max_diff = (baked_output - manual_output).abs().max().item()
        print(f"PASS — max diff {max_diff:.2e}")
        checks_passed += 1
    else:
        max_diff = (baked_output - manual_output).abs().max().item()
        print(f"FAIL — max diff {max_diff:.2e} (threshold 1e-5)")
        sys.exit(1)

    print(f"\nAll {checks_passed}/3 checks passed. Export successful.")


if __name__ == "__main__":
    main()
