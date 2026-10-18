"""Zero-shot EchoNext evaluation of a frozen MIMIC DFCL-HF checkpoint.

Does not train, fine-tune, or change DFCL-HF weights / architecture.
EchoNext waveforms are mapped through the same resize + per-lead z-score
used for MIMIC training (target_length=2000, 12 leads).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import normalize_per_lead, resize_signal
from src.evaluation.metrics import metrics_from_predictions
from src.utils import build_model, load_config, select_device, set_seed


class EchoNextNpyDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, target_length: int, normalize: bool) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.target_length = target_length
        self.normalize = normalize
        self._arrays: dict[str, np.ndarray] = {}
        for path in sorted(self.manifest["file_path"].astype(str).unique()):
            self._arrays[path] = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.manifest.iloc[index]
        path = str(row["file_path"])
        sample_index = int(row["sample_index"])
        waveform = np.asarray(self._arrays[path][sample_index], dtype=np.float32)
        waveform = np.squeeze(waveform)
        if waveform.ndim != 2:
            raise ValueError(f"{path}[{sample_index}] expected 2-D ECG, got {waveform.shape}")
        if waveform.shape[0] == 12:
            array = waveform
        elif waveform.shape[1] == 12:
            array = waveform.T
        else:
            raise ValueError(f"{path}[{sample_index}] no 12-lead axis: {waveform.shape}")
        if not np.isfinite(array).all():
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = resize_signal(array, self.target_length)
        if self.normalize:
            array = normalize_per_lead(array)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
            "y": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "dfcl_hf.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/mimic/echonext_zeroshot")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(args.device or str(config["device"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    required = {"sample_id", "patient_id", "label", "file_path", "sample_index"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    loader = DataLoader(
        EchoNextNpyDataset(
            manifest,
            target_length=int(config["target_length"]),
            normalize=bool(config["normalize"]),
        ),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows: list[dict[str, object]] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        _, logits, probs = model(x)
        probability = probs[:, 1].detach().cpu().numpy()
        labels = batch["y"].cpu().numpy().astype(np.int64)
        predicted = (probability >= 0.5).astype(np.int64)
        logits_np = logits.detach().cpu().numpy()
        for i in range(len(labels)):
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "true_label": int(labels[i]),
                    "predicted_probability": float(probability[i]),
                    "predicted_label": int(predicted[i]),
                    "logit_normal": float(logits_np[i, 0]),
                    "logit_hf": float(logits_np[i, 1]),
                    "model_stage": 10,
                    "test_set": "echonext_external_zeroshot",
                }
            )

    y_true = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    y_prob = np.asarray([row["predicted_probability"] for row in rows], dtype=np.float64)
    metrics = metrics_from_predictions(y_true, y_prob)
    metrics["sensitivity"] = metrics["recall"]
    metrics["n_total"] = int(len(y_true))
    metrics["n_case"] = int((y_true == 1).sum())
    metrics["n_control"] = int((y_true == 0).sum())
    metrics["threshold"] = 0.5
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["manifest"] = str(Path(args.manifest).resolve())
    metrics["evaluation_type"] = "zero-shot external validation"
    metrics["echonext_used_for_train_finetune_or_selection"] = False
    metrics["device"] = str(device)
    metrics["target_length"] = int(config["target_length"])
    metrics["normalize"] = bool(config["normalize"])

    pred_path = output_dir / "echonext_s10_zeroshot_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    cm = pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["true_control_0", "true_case_1"],
        columns=["pred_control_0", "pred_case_1"],
    )
    cm_path = output_dir / "echonext_s10_zeroshot_confusion_matrix.csv"
    cm.to_csv(cm_path)

    metrics_path = output_dir / "echonext_s10_zeroshot_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"[predictions] {pred_path}")
    print(f"[confusion] {cm_path}")
    print(f"[metrics] {metrics_path}")


if __name__ == "__main__":
    main()
