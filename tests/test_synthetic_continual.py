"""Small-scale 3-stage training on synthetic tensors (no real ECG)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.continual.initial_model import load_initial_model, save_initial_model, snapshot_parameters, tensors_equal
from src.continual.memory_teacher import build_memory_teacher
from src.continual.student_trainer import train_student
from src.continual.supervised import train_supervised
from src.continual.updated_teacher import train_updated_teacher
from src.losses.focal_loss import WeightedFocalLoss
from src.losses.psr import psr_loss
from src.models.backbone import CNNBiLSTMAttention
from src.utils import set_seed


class SynthECG(Dataset):
    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 12, 2000, generator=g)
        self.y = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return {
            "x": self.x[index],
            "y": self.y[index],
            "sample_id": str(index),
            "patient_id": str(index),
        }


def _loader(n: int, seed: int) -> DataLoader:
    return DataLoader(SynthECG(n, seed), batch_size=4, shuffle=True)


def _config() -> dict:
    return {
        "epochs_student": 2,
        "epochs_updated_teacher": 1,
        "learning_rate_student": 2e-4,
        "learning_rate_teacher": 2e-4,
        "weight_decay": 0.0,
        "early_stopping_patience": 5,
        "gradient_clip": 5.0,
        "use_amp": False,
        "optimizer": "Adam",
        "scheduler": "none",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "temperature_tau": 2.0,
        "confidence_threshold_delta": 0.3,
        "lambda_m": 0.5,
        "lambda_u": 0.5,
        "beta": 1.0,
        "eta": 1.5,
        "distill_eps": 1e-8,
        "psr_include_bias": True,
        "psr_layers": "all",
        "psr_eps": 1e-8,
        "debug_gradient_check": True,
        "focal_gamma": 2.0,
    }


def test_three_stage_synthetic(tmp_path: Path):
    set_seed(42)
    device = torch.device("cpu")
    config = _config()
    criterion = WeightedFocalLoss(gamma=2.0, class_weight=torch.tensor([1.0, 1.0]))
    student = CNNBiLSTMAttention().to(device)
    train0, val0 = _loader(16, 1), _loader(8, 2)
    ckpt0 = tmp_path / "s0.pt"
    train_supervised(
        student, train0, val0, criterion, device,
        epochs=2, learning_rate=2e-4, weight_decay=0.0, patience=5,
        gradient_clip=5.0, use_amp=False, optimizer_name="Adam",
        scheduler_name="none", scheduler_factor=0.5, scheduler_patience=2,
        checkpoint_path=ckpt0, log_path=tmp_path / "s0.csv", stage_name="s0",
    )
    initial = save_initial_model(student, tmp_path / "initial.pt", tmp_path / "inv.json")
    memory = build_memory_teacher(student)
    assert tensors_equal(initial, snapshot_parameters(memory))

    stage1_train, stage1_val = _loader(16, 11), _loader(8, 12)
    updated, _ = train_updated_teacher(
        memory, stage1_train, stage1_val, criterion, config, device,
        tmp_path / "u1.pt", tmp_path / "u1.csv",
    )
    student, hist1 = train_student(
        memory, updated, initial, stage1_train, stage1_val, criterion, config, device,
        tmp_path / "s1.pt", tmp_path / "s1.csv", tmp_path / "s1v.csv", stage=1,
    )
    assert not any(math_nan(hist1["train_psr_loss"]))
    memory_s1 = build_memory_teacher(student)
    assert not tensors_equal(initial, snapshot_parameters(memory_s1))
    assert tensors_equal(initial, load_initial_model(tmp_path / "initial.pt"))

    stage2_train, stage2_val = _loader(16, 21), _loader(8, 22)
    updated2, _ = train_updated_teacher(
        memory_s1, stage2_train, stage2_val, criterion, config, device,
        tmp_path / "u2.pt", tmp_path / "u2.csv",
    )
    student, hist2 = train_student(
        memory_s1, updated2, load_initial_model(tmp_path / "initial.pt"),
        stage2_train, stage2_val, criterion, config, device,
        tmp_path / "s2.pt", tmp_path / "s2.csv", tmp_path / "s2v.csv", stage=2,
    )
    assert tensors_equal(initial, load_initial_model(tmp_path / "initial.pt"))
    psr_now = float(psr_loss(student, initial))
    assert psr_now >= 0.0
    assert all(torch.isfinite(p).all() for p in student.parameters())
    Path(tmp_path / "ok.txt").write_text(f"psr_after_s2={psr_now}\n{hist1.to_csv()}\n{hist2.to_csv()}", encoding="utf-8")


def math_nan(series) -> list[bool]:
    return [not (abs(float(v)) < 1e10) for v in series]
