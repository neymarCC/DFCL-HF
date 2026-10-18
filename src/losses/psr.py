"""Parameter Stability Regularization (Eqs. 23-24).

Reference: frozen Stage-0 Student parameters (Initial Model).
Loss: sum of (1 - cosine) over every named parameter tensor, including bias.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def cosine_similarity_eq23(current: torch.Tensor, reference: torch.Tensor, eps: float) -> torch.Tensor:
    """Eq. (23): <a,b> / (||a|| ||b|| + eps) on flattened tensors."""
    if current.shape != reference.shape:
        raise ValueError(f"PSR shape mismatch: student {tuple(current.shape)} vs initial {tuple(reference.shape)}")
    a = current.reshape(-1).to(dtype=torch.float32)
    b = reference.reshape(-1).to(dtype=torch.float32).to(device=a.device)
    numerator = torch.dot(a, b)
    denominator = a.norm(p=2) * b.norm(p=2) + float(eps)
    return numerator / denominator


def psr_loss(
    student: nn.Module,
    initial_parameters: dict[str, torch.Tensor],
    eps: float = 1e-8,
    include_bias: bool = True,
    layers: str | list[str] = "all",
) -> torch.Tensor:
    """L_PSR = sum_k (1 - cos(theta_k(S_t, S0*)))."""
    del include_bias, layers
    student_params = dict(student.named_parameters())
    missing = [name for name in initial_parameters if name not in student_params]
    extra = [name for name in student_params if name not in initial_parameters]
    if missing or extra:
        raise KeyError(f"PSR parameter name mismatch. missing={missing[:8]} extra={extra[:8]}")

    terms: list[torch.Tensor] = []
    for name, parameter in student.named_parameters():
        reference = initial_parameters[name]
        if tuple(parameter.shape) != tuple(reference.shape):
            raise ValueError(f"PSR shape mismatch at {name}: {tuple(parameter.shape)} vs {tuple(reference.shape)}")
        similarity = cosine_similarity_eq23(parameter, reference.detach(), eps=eps)
        terms.append(1.0 - similarity)

    if not terms:
        return next(student.parameters()).new_zeros(())
    stacked = torch.stack(terms)
    if not torch.isfinite(stacked).all():
        raise RuntimeError("PSR produced NaN/Inf cosine terms")
    return stacked.sum()
