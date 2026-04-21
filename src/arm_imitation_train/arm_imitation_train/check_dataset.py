from __future__ import annotations

import argparse

from .config import load_config
from .dataset import build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the ARM imitation dataset loads correctly.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config)

    print(f"train_samples={len(train_dataset)}")
    print(f"val_samples={len(val_dataset)}")

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("train_batch:")
    print(f"  image.shape={tuple(train_batch['image'].shape)}")
    print(f"  state.shape={tuple(train_batch['state'].shape)}")
    print(f"  action.shape={tuple(train_batch['action'].shape)}")

    print("val_batch:")
    print(f"  image.shape={tuple(val_batch['image'].shape)}")
    print(f"  state.shape={tuple(val_batch['state'].shape)}")
    print(f"  action.shape={tuple(val_batch['action'].shape)}")

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
