from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.prediction import evaluate_loader


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler(device=device.type, enabled=enabled and device.type == "cuda")


def make_optimizer(model: nn.Module, config_lr: float, weight_decay: float, name: str):
    if name.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(model.parameters(), lr=config_lr, weight_decay=weight_decay)


def make_scheduler(optimizer, name: str, factor: float, patience: int):
    if name in {"none", "None", ""}:
        return None
    if name != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler: {name}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=factor, patience=patience
    )


def assert_finite(name: str, tensor: torch.Tensor, extra: dict[str, Any] | None = None) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} is NaN/Inf. extra={extra}")


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    gradient_clip: float,
    use_amp: bool,
    optimizer_name: str,
    scheduler_name: str,
    scheduler_factor: float,
    scheduler_patience: int,
    checkpoint_path: Path,
    log_path: Path,
    stage_name: str,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    optimizer = make_optimizer(model, learning_rate, weight_decay, optimizer_name)
    scheduler = make_scheduler(optimizer, scheduler_name, scheduler_factor, scheduler_patience)
    scaler = make_scaler(device, use_amp)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_auc = -math.inf
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                _, logits, _ = model(x)
                loss = criterion(logits, y)
            assert_finite("supervised_loss", loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val_metrics, _ = evaluate_loader(model, val_loader, device, model_stage=-1, test_stage=-1)
        val_auc = float(val_metrics["auroc"])
        score = val_auc if math.isfinite(val_auc) else -math.inf
        row = {
            "epoch": epoch,
            "stage_name": stage_name,
            "train_sup_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            "val_auroc": val_auc,
            "val_auprc": float(val_metrics["auprc"]),
            "val_f1": float(val_metrics["f1"]),
        }
        rows.append(row)
        print(
            f"[{stage_name}] epoch={epoch:03d} "
            f"train_sup={row['train_sup_loss']:.4f} val_auc={val_auc:.4f}"
        )
        if scheduler is not None:
            scheduler.step(score)
        if score > best_auc + 1e-6:
            best_auc = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model": best_state, "val_auroc": best_auc}, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    pd.DataFrame(rows).to_csv(log_path, index=False)
    model.load_state_dict(best_state)
    return best_state, pd.DataFrame(rows)
