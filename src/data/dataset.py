from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def extract_array(payload: Any, path: str) -> np.ndarray:
    if isinstance(payload, dict):
        for key in ("waveform", "ecg", "signal", "data"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"{path}: dict does not contain waveform/ecg/signal/data.")
    array = np.asarray(payload, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D ECG, got shape {array.shape}.")
    if array.shape[0] == 12:
        pass
    elif array.shape[1] == 12:
        array = array.T
    else:
        raise ValueError(f"{path}: neither dimension contains 12 leads; shape={array.shape}.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def resize_signal(array: np.ndarray, target_length: int) -> np.ndarray:
    current_length = array.shape[1]
    if current_length == target_length:
        return array
    old_grid = np.linspace(0.0, 1.0, current_length, dtype=np.float64)
    new_grid = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    return np.stack(
        [np.interp(new_grid, old_grid, lead).astype(np.float32) for lead in array],
        axis=0,
    )


def normalize_per_lead(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    return (array - mean) / np.clip(std, 1e-6, None)


class ManifestECGDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: pd.DataFrame, target_length: int, normalize: bool) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.target_length = target_length
        self.normalize = normalize
        self._npy: dict[str, np.ndarray] = {}
        if "file_path" in self.manifest.columns:
            for path in sorted(self.manifest["file_path"].astype(str).unique()):
                if path.lower().endswith(".npy"):
                    self._npy[path] = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_array(self, row: pd.Series) -> np.ndarray:
        path = str(row["file_path"])
        if path.lower().endswith(".npy"):
            sample_index = int(row["sample_index"])
            payload = self._npy[path][sample_index]
        else:
            with Path(path).open("rb") as file:
                payload = pickle.load(file)
        return extract_array(payload, path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        array = self._load_array(row)
        array = resize_signal(array, self.target_length)
        if self.normalize:
            array = normalize_per_lead(array)
        item: dict[str, Any] = {
            "x": torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
            "y": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
        }
        if "year" in self.manifest.columns and pd.notna(row["year"]):
            item["year"] = int(row["year"])
        return item


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_loader(
    manifest: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    target_length: int,
    normalize: bool,
    device: torch.device,
) -> DataLoader:
    dataset = ManifestECGDataset(manifest, target_length, normalize)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
