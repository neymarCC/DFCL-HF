from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float,
        class_weight: torch.Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        if class_weight is None:
            self.class_weight = None
        else:
            self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=1)
        p_true = probabilities.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        log_pt = torch.log(p_true)
        if self.class_weight is None:
            alpha = torch.ones_like(p_true)
        else:
            alpha = self.class_weight[target]
        loss = -alpha * (1.0 - p_true).pow(self.gamma) * log_pt
        return loss.mean()
