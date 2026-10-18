from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .confidence_distillation import cagd_loss
from .psr import psr_loss


def student_total_loss(
    student: nn.Module,
    initial_parameters: dict[str, torch.Tensor],
    student_logits: torch.Tensor,
    memory_logits: torch.Tensor,
    updated_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    tau: float,
    delta: float,
    lambda_m: float,
    lambda_u: float,
    beta: float,
    eta: float,
    distill_eps: float,
    psr_include_bias: bool,
    psr_layers: str | list[str],
    psr_eps: float,
    disable_confidence_gating: bool = False,
) -> dict[str, Any]:
    loss_sup = criterion(student_logits, labels)
    distill = cagd_loss(
        memory_logits,
        updated_logits,
        student_logits,
        tau=tau,
        delta=delta,
        lambda_m=lambda_m,
        lambda_u=lambda_u,
        eps=distill_eps,
        disable_confidence_gating=disable_confidence_gating,
    )
    loss_psr = psr_loss(
        student,
        initial_parameters,
        eps=psr_eps,
        include_bias=psr_include_bias,
        layers=psr_layers,
    )
    loss_total = loss_sup + beta * distill["loss_cagd"] + eta * loss_psr
    return {
        "loss_total": loss_total,
        "loss_sup": loss_sup,
        "loss_cagd": distill["loss_cagd"],
        "loss_psr": loss_psr,
        "loss_m": distill["loss_m"],
        "loss_u": distill["loss_u"],
        "q_m": distill["q_m"],
        "q_u": distill["q_u"],
        "w_m": distill["w_m"],
        "w_u": distill["w_u"],
    }
