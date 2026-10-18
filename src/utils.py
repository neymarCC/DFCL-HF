from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .models.backbone import CNNBiLSTMAttention

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("manifest_root", "output_root", "future_test_manifest")


def resolve_user_path(value: str | Path, base: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in PATH_KEYS:
        if config.get(key):
            config[key] = resolve_user_path(config[key], REPO_ROOT)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: dict[str, Any]) -> CNNBiLSTMAttention:
    return CNNBiLSTMAttention(
        hidden_dim=int(config["lstm_hidden"]),
        num_heads=int(config["attention_heads"]),
        dropout=float(config["dropout"]),
        num_classes=int(config["num_classes"]),
    )


def class_weights(manifest: pd.DataFrame, device: torch.device, focal_alpha: Any) -> torch.Tensor:
    counts = manifest["label"].value_counts().to_dict()
    n0 = max(int(counts.get(0, 1)), 1)
    n1 = max(int(counts.get(1, 1)), 1)
    if focal_alpha == "auto" or focal_alpha is None:
        total = n0 + n1
        weights = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)
    else:
        weights = torch.tensor(list(focal_alpha), dtype=torch.float32)
    return weights.to(device)
