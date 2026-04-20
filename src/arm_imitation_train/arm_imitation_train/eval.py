from __future__ import annotations

import argparse

import torch
from torch import nn

from .config import load_config
from .dataset import build_dataloaders
from .model import build_model
from .utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a minimal BC policy on the ARM dataset.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output", default="eval_metrics.json", help="Where to save evaluation metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])

    _, val_dataset, _, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.SmoothL1Loss(reduction="sum")
    mse_criterion = nn.MSELoss(reduction="sum")
    total_l1 = 0.0
    total_mse = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            states = batch["state"].to(device)
            targets = batch["action"].to(device)
            predictions = model(images, states)

            total_l1 += criterion(predictions, targets).item()
            total_mse += mse_criterion(predictions, targets).item()
            sample_count += images.size(0)

    metrics = {
        "checkpoint": args.checkpoint,
        "num_samples": sample_count,
        "mean_l1_loss": total_l1 / max(1, sample_count),
        "mean_mse_loss": total_mse / max(1, sample_count),
    }
    save_json(args.output, metrics)
    print(metrics, flush=True)

    val_dataset.close()


if __name__ == "__main__":
    main()
