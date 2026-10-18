from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.prediction import evaluate_loader
from ..losses.total_loss import student_total_loss
from .memory_teacher import freeze_module
from .supervised import assert_finite, autocast_context, make_optimizer, make_scaler, make_scheduler


def initialize_student(memory_teacher: nn.Module) -> nn.Module:
    """S_t^(0) = deepcopy(M_t). Not random initialization."""
    student = copy.deepcopy(memory_teacher)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    return student


def check_teacher_grads(memory_teacher: nn.Module, updated_teacher: nn.Module, student: nn.Module) -> None:
    for name, parameter in memory_teacher.named_parameters():
        if parameter.grad is not None:
            raise AssertionError(f"Memory Teacher has gradient at {name}")
    for name, parameter in updated_teacher.named_parameters():
        if parameter.grad is not None:
            raise AssertionError(f"Updated Teacher has gradient at {name}")
    if not any(parameter.grad is not None for parameter in student.parameters()):
        raise AssertionError("Student has no gradient after backward.")


def train_student(
    memory_teacher: nn.Module,
    updated_teacher: nn.Module,
    initial_parameters: dict[str, torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    val_log_path: Path,
    stage: int,
) -> tuple[nn.Module, pd.DataFrame]:
    freeze_module(memory_teacher)
    freeze_module(updated_teacher)
    student = initialize_student(memory_teacher).to(device)
    optimizer = make_optimizer(
        student,
        float(config["learning_rate_student"]),
        float(config["weight_decay"]),
        str(config["optimizer"]),
    )
    scheduler = make_scheduler(
        optimizer,
        str(config["scheduler"]),
        float(config["scheduler_factor"]),
        int(config["scheduler_patience"]),
    )
    scaler = make_scaler(device, bool(config["use_amp"]))
    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    best_auc = -math.inf
    bad_epochs = 0
    rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    checked_grads = False
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["epochs_student"]) + 1):
        student.train()
        totals = {
            "train_total_loss": [],
            "train_sup_loss": [],
            "train_cagd_loss": [],
            "train_psr_loss": [],
            "memory_distill_loss": [],
            "updated_distill_loss": [],
            "average_memory_confidence": [],
            "average_updated_confidence": [],
            "high_confidence_ratio_memory": [],
            "high_confidence_ratio_updated": [],
        }
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch.no_grad():
                _, logits_m, _ = memory_teacher(x)
                _, logits_u, _ = updated_teacher(x)
            with autocast_context(device, bool(config["use_amp"])):
                _, logits_s, _ = student(x)
                losses = student_total_loss(
                    student=student,
                    initial_parameters=initial_parameters,
                    student_logits=logits_s,
                    memory_logits=logits_m,
                    updated_logits=logits_u,
                    labels=y,
                    criterion=criterion,
                    tau=float(config["temperature_tau"]),
                    delta=float(config["confidence_threshold_delta"]),
                    lambda_m=float(config["lambda_m"]),
                    lambda_u=float(config["lambda_u"]),
                    beta=float(config["beta"]),
                    eta=float(config["eta"]),
                    distill_eps=float(config["distill_eps"]),
                    psr_include_bias=bool(config["psr_include_bias"]),
                    psr_layers=config["psr_layers"],
                    psr_eps=float(config["psr_eps"]),
                    disable_confidence_gating=bool(config.get("disable_confidence_gating", False)),
                )
            extra = {
                "stage": stage,
                "epoch": epoch,
                "sample_ids": list(batch["sample_id"]),
            }
            for key in ("loss_total", "loss_sup", "loss_cagd", "loss_psr", "loss_m", "loss_u"):
                assert_finite(key, losses[key], extra)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            if bool(config["debug_gradient_check"]) and not checked_grads:
                check_teacher_grads(memory_teacher, updated_teacher, student)
                checked_grads = True
            nn.utils.clip_grad_norm_(student.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()

            delta = float(config["confidence_threshold_delta"])
            totals["train_total_loss"].append(float(losses["loss_total"].item()))
            totals["train_sup_loss"].append(float(losses["loss_sup"].item()))
            totals["train_cagd_loss"].append(float(losses["loss_cagd"].item()))
            totals["train_psr_loss"].append(float(losses["loss_psr"].item()))
            totals["memory_distill_loss"].append(float(losses["loss_m"].item()))
            totals["updated_distill_loss"].append(float(losses["loss_u"].item()))
            totals["average_memory_confidence"].append(float(losses["q_m"].mean().item()))
            totals["average_updated_confidence"].append(float(losses["q_u"].mean().item()))
            totals["high_confidence_ratio_memory"].append(
                float((losses["q_m"] >= delta).float().mean().item())
            )
            totals["high_confidence_ratio_updated"].append(
                float((losses["q_u"] >= delta).float().mean().item())
            )

        val_metrics, _ = evaluate_loader(student, val_loader, device, model_stage=stage, test_stage=-1)
        val_auc = float(val_metrics["auroc"])
        row = {"epoch": epoch, "stage": stage}
        for key, values in totals.items():
            row[key] = float(sum(values) / max(len(values), 1))
        rows.append(row)
        val_rows.append({"epoch": epoch, "stage": stage, **val_metrics})
        print(
            f"[student stage {stage:02d}] epoch={epoch:03d} "
            f"total={row['train_total_loss']:.4f} sup={row['train_sup_loss']:.4f} "
            f"cagd={row['train_cagd_loss']:.4f} psr={row['train_psr_loss']:.4f} "
            f"val_auc={val_auc:.4f}"
        )
        score = val_auc if math.isfinite(val_auc) else -math.inf
        if scheduler is not None:
            scheduler.step(score)
        if score > best_auc + 1e-6:
            best_auc = score
            best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            torch.save({"model": best_state, "val_auroc": best_auc, "stage": stage}, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["early_stopping_patience"]):
                break

    pd.DataFrame(rows).to_csv(log_path, index=False)
    pd.DataFrame(val_rows).to_csv(val_log_path, index=False)
    student.load_state_dict(best_state)
    student.eval()
    return student, pd.DataFrame(rows)
