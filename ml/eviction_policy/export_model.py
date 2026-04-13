#!/usr/bin/env python3


import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_eviction_model import ReusePredictionModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export eviction model to TorchScript"
    )
    parser.add_argument(
        "--model-dir", default="ml/eviction_policy/models",
        help="Directory containing model artifacts",
    )
    args = parser.parse_args()

    # Load config
    config = torch.load(
        os.path.join(args.model_dir, "model_config.pt"), weights_only=True
    )
    max_pages = config["max_pages"]

    # Load model
    model = ReusePredictionModel(max_pages=max_pages)
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_dir, "eviction_model_best.pt"),
            weights_only=True,
        )
    )
    model.eval()

    # Script and save
    scripted = torch.jit.script(model)
    output_path = os.path.join(args.model_dir, "eviction_model.pt")
    scripted.save(output_path)
    print(f"Exported TorchScript model to {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    test_input = torch.randint(0, min(max_pages, 100), (1, 64))
    with torch.inference_mode():
        test_output = loaded(test_input)
    print(f"Verification: input shape {tuple(test_input.shape)} → output {test_output[0].item():.4f}")
    print("Export successful.")


if __name__ == "__main__":
    main()
