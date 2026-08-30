"""
src/train.py
------------
Training entry-point for the CIFAR-10 image classifier.

Usage (local):
    python src/train.py                           # uses configs/training_config.yaml
    CONFIG_PATH=/custom/path.yaml python src/train.py

Usage (Docker / Kubernetes):
    The default config path is /app/configs/training_config.yaml.
    Override with the CONFIG_PATH environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Allow running directly from the repo root *or* from inside a container where
# PYTHONPATH=/app/src is not guaranteed.
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _log(record: dict) -> None:
    """Print a JSON-lines log record to stdout (flushed immediately)."""
    print(json.dumps(record), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run evaluation over the validation set.

    Returns:
        (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Resolve config path ────────────────────────────────────────────────
    default_paths = [
        Path(os.environ.get("CONFIG_PATH", "")),
        Path("/app/configs/training_config.yaml"),
        Path(__file__).parent.parent / "configs" / "training_config.yaml",
    ]
    config_path: Path | None = None
    for p in default_paths:
        if p and p.exists():
            config_path = p
            break

    if config_path is None:
        _log({"event": "error", "message": "training_config.yaml not found"})
        sys.exit(1)

    _log({"event": "config_loaded", "path": str(config_path)})
    config = load_config(str(config_path))

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log({"event": "device", "device": str(device)})

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=True,
    ).to(device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
    )

    # ── Optimiser & criterion ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    patience: int = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        _log(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "lr": round(scheduler.get_last_lr()[0], 6),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": config,
                },
                save_path,
            )
            _log({"event": "checkpoint_saved", "path": str(save_path), "val_loss": round(val_loss, 4)})
        else:
            patience_counter += 1
            _log({"event": "no_improvement", "patience_counter": patience_counter, "patience": patience})

        if patience_counter >= patience:
            _log({"event": "early_stopping", "epoch": epoch + 1})
            break

    _log(
        {
            "event": "training_complete",
            "best_val_loss": round(best_val_loss, 4),
            "checkpoint": str(save_path),
        }
    )


if __name__ == "__main__":
    main()
