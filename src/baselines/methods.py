"""Canonical continual-learning regularizers. No raw-ECG replay."""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..continual.memory_teacher import freeze_module
from ..continual.supervised import autocast_context


def clone_trainable(source: nn.Module) -> nn.Module:
    student = copy.deepcopy(source)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    student.train()
    return student


def empirical_fisher(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    """Diagonal empirical Fisher on the completed stage's train loader. Not replay."""
    was_training = model.training
    # LSTM/cuDNN requires train() for backward; this is not a training step.
    model.train()
    accum = {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    n_batches = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            _, logits, _ = model(x)
            log_probs = F.log_softmax(logits.float(), dim=1)
            loss = F.nll_loss(log_probs, y)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in accum and parameter.grad is not None:
                accum[name] += parameter.grad.detach().float().cpu().pow(2)
        n_batches += 1
    denom = max(n_batches, 1)
    if not was_training:
        model.eval()
    return {name: value / denom for name, value in accum.items()}


def ewc_penalty(
    model: nn.Module,
    ewc_states: list[dict[str, Any]],
    ewc_lambda: float,
) -> torch.Tensor:
    if not ewc_states:
        return next(model.parameters()).new_zeros(())
    total = next(model.parameters()).new_zeros(())
    params = dict(model.named_parameters())
    for state in ewc_states:
        fisher: dict[str, torch.Tensor] = state["fisher"]
        anchor: dict[str, torch.Tensor] = state["params"]
        for name, parameter in params.items():
            if name not in fisher:
                continue
            f = fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            theta_star = anchor[name].to(device=parameter.device, dtype=parameter.dtype)
            total = total + (f * (parameter - theta_star).pow(2)).sum()
    return 0.5 * float(ewc_lambda) * total


def lwf_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    alpha: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    tau = float(temperature)
    teacher = F.softmax(teacher_logits.detach() / tau, dim=1)
    student = F.log_softmax(student_logits / tau, dim=1)
    kd = F.kl_div(student, teacher, reduction="batchmean")
    return float(alpha) * (tau ** 2) * kd


def _get_gt_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(logits)
    mask.scatter_(1, target.unsqueeze(1), 1.0)
    return mask.bool()


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    temperature: float,
) -> torch.Tensor:
    """Decoupled KD (Zhao et al., CVPR 2022). Official TCKD + NCKD."""
    tau = float(temperature)
    gt_mask = _get_gt_mask(student_logits, target)
    other_mask = ~gt_mask
    pred_student = F.softmax(student_logits / tau, dim=1)
    pred_teacher = F.softmax(teacher_logits.detach() / tau, dim=1)
    pred_student_tckd = torch.cat(
        [
            (pred_student * gt_mask).sum(dim=1, keepdim=True),
            (pred_student * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    pred_teacher_tckd = torch.cat(
        [
            (pred_teacher * gt_mask).sum(dim=1, keepdim=True),
            (pred_teacher * other_mask).sum(dim=1, keepdim=True),
        ],
        dim=1,
    )
    tckd = F.kl_div(torch.log(pred_student_tckd.clamp_min(1e-8)), pred_teacher_tckd, reduction="batchmean")
    pred_teacher_nckd = F.softmax(teacher_logits.detach() / tau - 1000.0 * gt_mask.float(), dim=1)
    pred_student_nckd = F.softmax(student_logits / tau - 1000.0 * gt_mask.float(), dim=1)
    nckd = F.kl_div(torch.log(pred_student_nckd.clamp_min(1e-8)), pred_teacher_nckd, reduction="batchmean")
    return (float(alpha) * tckd + float(beta) * nckd) * (tau ** 2)


def lwht_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    temperature: float,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Hierarchical teachers: logit KD + representation MSE (FitNet-style hint)."""
    kd = lwf_loss(student_logits, teacher_logits, temperature, alpha=1.0)
    hint = F.mse_loss(student_features, teacher_features.detach())
    return float(alpha) * kd + float(gamma) * hint


def isolate_by_fisher(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    freeze_fraction: float,
    always_trainable_prefix: str = "classifier.6",
) -> list[str]:
    """Freeze the highest-importance still-trainable parameters (API isolation)."""
    scores: list[tuple[float, str]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(always_trainable_prefix):
            continue
        if name not in fisher:
            continue
        scores.append((float(fisher[name].mean().item()), name))
    scores.sort(reverse=True)
    n_freeze = max(1, int(round(len(scores) * float(freeze_fraction))))
    newly_frozen: list[str] = []
    lookup = dict(model.named_parameters())
    for _, name in scores[:n_freeze]:
        lookup[name].requires_grad_(False)
        newly_frozen.append(name)
    return newly_frozen
