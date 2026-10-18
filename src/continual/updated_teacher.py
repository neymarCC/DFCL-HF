from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .memory_teacher import freeze_module
from .supervised import train_supervised


def initialize_updated_teacher(memory_teacher: nn.Module) -> nn.Module:
    """U_t^(0) = deepcopy(M_t). Not random initialization."""
    updated = copy.deepcopy(memory_teacher)
    for parameter in updated.parameters():
        parameter.requires_grad_(True)
    updated.train()
    return updated


def train_updated_teacher(
    memory_teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
) -> tuple[nn.Module, pd.DataFrame]:
    updated = initialize_updated_teacher(memory_teacher)
    best_state, history = train_supervised(
        model=updated,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=int(config["epochs_updated_teacher"]),
        learning_rate=float(config["learning_rate_teacher"]),
        weight_decay=float(config["weight_decay"]),
        patience=int(config["early_stopping_patience"]),
        gradient_clip=float(config["gradient_clip"]),
        use_amp=bool(config["use_amp"]),
        optimizer_name=str(config["optimizer"]),
        scheduler_name=str(config["scheduler"]),
        scheduler_factor=float(config["scheduler_factor"]),
        scheduler_patience=int(config["scheduler_patience"]),
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        stage_name="updated_teacher",
    )
    updated.load_state_dict(best_state)
    freeze_module(updated)
    return updated, history
