from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import metrics_from_predictions


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_stage: int,
    test_stage: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        _, logits, probs = model(x)
        probability = probs[:, 1].detach().cpu().numpy()
        labels = batch["y"].cpu().numpy().astype(np.int64)
        predicted = (probability >= 0.5).astype(np.int64)
        logits_np = logits.detach().cpu().numpy()
        years = batch["year"].cpu().numpy().astype(np.int64) if "year" in batch else None
        for i in range(len(labels)):
            row = {
                "sample_id": batch["sample_id"][i],
                "patient_id": batch["patient_id"][i],
                "true_label": int(labels[i]),
                "predicted_probability": float(probability[i]),
                "predicted_label": int(predicted[i]),
                "logit_normal": float(logits_np[i, 0]),
                "logit_hf": float(logits_np[i, 1]),
                "model_stage": int(model_stage),
                "test_stage": int(test_stage),
            }
            if years is not None:
                row["year"] = int(years[i])
            rows.append(row)
    y_true = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    y_prob = np.asarray([row["predicted_probability"] for row in rows], dtype=np.float64)
    return metrics_from_predictions(y_true, y_prob), rows


def save_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
