from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def temperature_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return F.softmax(logits / tau, dim=1)


def teacher_confidence(
    teacher_probs: torch.Tensor,
    delta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entropy = -(teacher_probs * torch.log(teacher_probs + eps)).sum(dim=1)
    n_classes = teacher_probs.shape[1]
    confidence = 1.0 - entropy / torch.log(torch.tensor(float(n_classes), device=teacher_probs.device))
    weights = confidence * (confidence >= delta).to(confidence.dtype)
    return entropy, confidence, weights


def weighted_kl(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    eps: float,
) -> torch.Tensor:
    teacher = teacher_probs.detach()
    # PyTorch kl_div(log_student, teacher) = KL(teacher || student) per sample.
    kl = F.kl_div(torch.log(student_probs.clamp_min(eps)), teacher, reduction="none").sum(dim=1)
    weight_mass = weights.sum()
    if float(weight_mass.detach()) <= 0.0:
        return student_probs.new_zeros(())
    # Implemented denominator: (sum_i w_i + eps), not sum_i (w_i + eps).
    return (tau ** 2) * (weights * kl).sum() / (weight_mass + eps)


def cagd_loss(
    memory_logits: torch.Tensor,
    updated_logits: torch.Tensor,
    student_logits: torch.Tensor,
    tau: float,
    delta: float,
    lambda_m: float,
    lambda_u: float,
    eps: float,
    disable_confidence_gating: bool = False,
) -> dict[str, Any]:
    p_m = temperature_softmax(memory_logits, tau)
    p_u = temperature_softmax(updated_logits, tau)
    p_s = temperature_softmax(student_logits, tau)
    _, q_m, w_m = teacher_confidence(p_m, delta, eps)
    _, q_u, w_u = teacher_confidence(p_u, delta, eps)
    if disable_confidence_gating:
        w_m = torch.ones_like(w_m)
        w_u = torch.ones_like(w_u)
    loss_m = weighted_kl(p_m, p_s, w_m, tau, eps)
    loss_u = weighted_kl(p_u, p_s, w_u, tau, eps)
    loss_cagd = lambda_m * loss_m + lambda_u * loss_u
    return {
        "loss_cagd": loss_cagd,
        "loss_m": loss_m,
        "loss_u": loss_u,
        "q_m": q_m,
        "q_u": q_u,
        "w_m": w_m,
        "w_u": w_u,
    }
