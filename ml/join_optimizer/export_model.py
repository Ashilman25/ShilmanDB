#!/usr/bin/env python3

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from train_join_model import JoinOrderModel


class JoinOrderModelWithNorm(nn.Module):

    def __init__(self, model: JoinOrderModel, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export join order model to TorchScript")
    parser.add_argument("--model-dir", default="ml/join_optimizer/models", help="Model directory")
    args = parser.parse_args()

    model = JoinOrderModel()
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "join_order_model_best.pt"), weights_only=True))
    model.eval()

    stats = torch.load(os.path.join(args.model_dir, "feature_stats.pt"), weights_only=True)

    wrapper = JoinOrderModelWithNorm(model, stats["mean"], stats["std"])
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    output_path = os.path.join(args.model_dir, "join_order_model.pt")
    scripted.save(output_path)
    print(f"Exported TorchScript model (with baked standardization) to {output_path}")


if __name__ == "__main__":
    main()
