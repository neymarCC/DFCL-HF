from __future__ import annotations

import copy

import torch.nn as nn


def freeze_module(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def build_memory_teacher(previous_student: nn.Module) -> nn.Module:
    """M_t = deepcopy(S_(t-1)*), then freeze."""
    memory = copy.deepcopy(previous_student)
    return freeze_module(memory)
