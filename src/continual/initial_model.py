"""Stage-0 Initial Model snapshot used as the PSR parameter reference.

No forward pass is required for PSR.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def parameter_inventory(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    return rows


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clones of every named parameter. Detached; never updated in later stages."""
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in model.named_parameters()
    }


def save_initial_model(model: nn.Module, checkpoint_path: Path, inventory_path: Path) -> dict[str, torch.Tensor]:
    snapshot = snapshot_parameters(model)
    inventory = parameter_inventory(model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "role": "psr_initial_model",
            "stage": 0,
            "model": {name: tensor for name, tensor in snapshot.items()},
            "parameter_inventory": inventory,
        },
        checkpoint_path,
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", encoding="utf-8") as file:
        json.dump({"n_tensors": len(inventory), "parameters": inventory}, file, indent=2)
    return snapshot


def load_initial_model(checkpoint_path: Path | str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    else:
        state = payload
    snapshot = {}
    for name, tensor in state.items():
        if name.startswith("running_") or name.endswith("num_batches_tracked"):
            continue
        if not torch.is_tensor(tensor):
            continue
        snapshot[name] = tensor.detach().cpu().contiguous().clone()
    if not snapshot:
        raise ValueError(f"no parameter tensors in {checkpoint_path}")
    return snapshot


def tensors_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], atol: float = 0.0) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        if left[name].shape != right[name].shape:
            return False
        if not torch.allclose(left[name].float(), right[name].float(), atol=atol, rtol=0.0):
            return False
    return True
