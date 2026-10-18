"""EchoNext in-domain temporal validation: train 2008-2020, val 2021, test 2022.

Random-init backbone (no MIMIC S10, no DFCL-HF). 2022 is evaluated once after
the best 2021-val-AUROC checkpoint is restored.
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

from src.continual.supervised import train_supervised
from src.data.dataset import normalize_per_lead, resize_signal
from src.evaluation.metrics import metrics_from_predictions
from src.losses.focal_loss import WeightedFocalLoss
from src.utils import build_model, class_weights, load_config, select_device, set_seed


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
            "year": int(row["year"]),
        }


def count_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for year, part in frame.groupby("year"):
        n_case = int((part["label"] == 1).sum())
        n_ctrl = int((part["label"] == 0).sum())
        rows.append(
            {
                "split": split,
                "year": int(year),
                "n_total": int(len(part)),
                "n_case": n_case,
                "n_control": n_ctrl,
            }
        )
    n_case = int((frame["label"] == 1).sum())
    n_ctrl = int((frame["label"] == 0).sum())
    rows.append(
        {
            "split": split,
            "year": "ALL",
            "n_total": int(len(frame)),
            "n_case": n_case,
            "n_control": n_ctrl,
        }
    )
    return pd.DataFrame(rows)


def make_loader(manifest: pd.DataFrame, config: dict, device: torch.device, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EchoNextNpyDataset(manifest, int(config["target_length"]), bool(config["normalize"])),
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    threshold = 0.5
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        _, logits, probs = model(x)
        probability = probs[:, 1].detach().cpu().numpy()
        labels = batch["y"].cpu().numpy().astype(np.int64)
        predicted = (probability >= threshold).astype(np.int64)
        years = batch["year"].cpu().numpy().astype(np.int64)
        for i in range(len(labels)):
            rows.append(
                {
                    "patient_id": batch["patient_id"][i],
                    "sample_id": batch["sample_id"][i],
                    "year": int(years[i]),
                    "true_label": int(labels[i]),
                    "predicted_probability": float(probability[i]),
                    "predicted_label": int(predicted[i]),
                }
            )
    y_true = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    y_prob = np.asarray([row["predicted_probability"] for row in rows], dtype=np.float64)
    metrics = metrics_from_predictions(y_true, y_prob, threshold=threshold)
    metrics["sensitivity"] = metrics["recall"]
    return metrics, rows


def write_split_manifests(source: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    train_years = set(int(y) for y in config["train_years"])
    val_years = set(int(y) for y in config["val_years"])
    test_years = set(int(y) for y in config["test_years"])
    splits = {
        "train": source.loc[source["year"].isin(train_years)].copy(),
        "val": source.loc[source["year"].isin(val_years)].copy(),
        "test": source.loc[source["year"].isin(test_years)].copy(),
    }
    if len(splits["train"]) + len(splits["val"]) + len(splits["test"]) != len(source):
        leftover = source.loc[
            ~source["year"].isin(train_years | val_years | test_years)
        ]
        raise RuntimeError(f"Unassigned years: {sorted(leftover['year'].unique())}")
    root = Path(config["manifest_root"])
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame = frame.reset_index(drop=True)
        splits[name] = frame
        frame.to_csv(root / f"{name}.csv", index=False)
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "echonext_temporal.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if bool(config.get("load_mimic_weights", False)) or bool(config.get("use_dfcl", False)):
        raise RuntimeError("This experiment must not load MIMIC weights or run DFCL-HF.")
    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    output_root = Path(config["output_root"])
    ckpt_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(config["source_manifest"])
    splits = write_split_manifests(source, config)
    counts = pd.concat(
        [count_table(splits[name], name) for name in ("train", "val", "test")],
        ignore_index=True,
    )
    counts.to_csv(output_root / "split_year_counts.csv", index=False)
    print(counts.to_string(index=False))

    train_loader = make_loader(splits["train"], config, device, True, int(config["seed"]))
    val_loader = make_loader(splits["val"], config, device, False, int(config["seed"]) + 1)
    test_loader = make_loader(splits["test"], config, device, False, int(config["seed"]) + 2)

    model = build_model(config).to(device)
    criterion = WeightedFocalLoss(
        gamma=float(config["focal_gamma"]),
        class_weight=class_weights(splits["train"], device, config["focal_alpha"]),
        eps=float(config["focal_eps"]),
    )
    checkpoint_path = ckpt_dir / "echonext_temporal_best.pt"
    best_state, history = train_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        patience=int(config["early_stopping_patience"]),
        gradient_clip=float(config["gradient_clip"]),
        use_amp=bool(config["use_amp"]),
        optimizer_name=str(config["optimizer"]),
        scheduler_name=str(config["scheduler"]),
        scheduler_factor=float(config["scheduler_factor"]),
        scheduler_patience=int(config["scheduler_patience"]),
        checkpoint_path=checkpoint_path,
        log_path=log_dir / "training_log.csv",
        stage_name="echonext_temporal",
    )
    model.load_state_dict(best_state)
    model.eval()
    best_epoch = int(history.loc[history["val_auroc"].idxmax(), "epoch"])
    best_val_auroc = float(history["val_auroc"].max())

    print("===== 2022 independent test (single evaluation after best checkpoint) =====")
    test_metrics, test_rows = predict(model, test_loader, device)
    pd.DataFrame(test_rows).to_csv(output_root / "echonext_temporal_2022_predictions.csv", index=False)
    cm = pd.DataFrame(
        [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]],
        index=["true_control_0", "true_case_1"],
        columns=["pred_control_0", "pred_case_1"],
    )
    cm.to_csv(output_root / "echonext_temporal_2022_confusion_matrix.csv")
    test_metrics["sensitivity"] = test_metrics["recall"]
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_auroc_2021"] = best_val_auroc
    test_metrics["checkpoint"] = str(checkpoint_path.resolve())
    test_metrics["evaluation_type"] = "EchoNext in-domain temporal validation"
    test_metrics["train_years"] = "2008-2020"
    test_metrics["val_year"] = 2021
    test_metrics["test_year"] = 2022
    test_metrics["loaded_mimic_s10"] = False
    test_metrics["used_dfcl_hf"] = False
    test_metrics["threshold"] = 0.5
    test_metrics["seed"] = int(config["seed"])
    with (output_root / "echonext_temporal_2022_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)
    with (output_root / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    print(json.dumps(test_metrics, indent=2))
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
