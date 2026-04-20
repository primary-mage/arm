from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .dataset import build_dataloaders
from .model import build_model
from .utils import ensure_output_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal BC policy on the ARM dataset.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def run_epoch(model, data_loader, criterion, optimizer, device, log_interval, is_train):
    model.train(is_train)
    total_loss = 0.0
    sample_count = 0

    for batch_index, batch in enumerate(data_loader):
        images = batch["image"].to(device)
        states = batch["state"].to(device)
        targets = batch["action"].to(device)

        with torch.set_grad_enabled(is_train):
            predictions = model(images, states)
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

        if is_train and batch_index % log_interval == 0:
            print(
                f"[train] batch={batch_index:04d} loss={loss.item():.6f}",
                flush=True,
            )

    return total_loss / max(1, sample_count)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    output_dir = ensure_output_dir(config["training"]["output_dir"])

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    summary = {
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "config_path": str(Path(args.config).resolve()),
    }
    save_json(output_dir / "run_summary.json", summary)

    best_val_loss = float("inf")
    history = []
    epochs = int(config["training"]["epochs"])
    log_interval = int(config["training"]["log_interval"])

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=log_interval,
            is_train=True,
        )
        val_loss = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=log_interval,
            is_train=False,
        )

        duration_sec = time.time() - start_time
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "duration_sec": duration_sec,
            }
        )

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} duration={duration_sec:.2f}s",
            flush=True,
        )

        latest_checkpoint = output_dir / "latest.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            latest_checkpoint,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                output_dir / "best.pt",
            )

        save_json(output_dir / "history.json", {"epochs": history, "best_val_loss": best_val_loss})

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
