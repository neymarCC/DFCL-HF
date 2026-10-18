"""Dual complementary prompts on CNN-BiLSTM pooled features (idea-based DualCP)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbone import CNNBiLSTMAttention


class DualCPModel(nn.Module):
    """G-prompt (shared) + E-prompts (experts) with key matching; no ViT prefix-tuning."""

    def __init__(self, backbone: CNNBiLSTMAttention, prompt_temperature: float = 1.0) -> None:
        super().__init__()
        self.backbone = backbone
        dim = int(backbone.classifier[0].in_features)
        self.g_prompt = nn.Parameter(torch.zeros(dim))
        self.e_prompts = nn.ParameterList([nn.Parameter(torch.zeros(dim))])
        self.e_keys = nn.ParameterList([nn.Parameter(0.02 * torch.randn(dim))])
        self.prompt_temperature = float(prompt_temperature)

    def add_expert(self) -> None:
        dim = int(self.g_prompt.numel())
        device = self.g_prompt.device
        self.e_prompts.append(nn.Parameter(torch.zeros(dim, device=device)))
        self.e_keys.append(nn.Parameter((0.02 * torch.randn(dim, device=device))))

    def freeze_extractor(self) -> None:
        for name, parameter in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                parameter.requires_grad_(False)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.backbone.cnn(x).transpose(1, 2)
        self.backbone.bilstm.flatten_parameters()
        temporal, _ = self.backbone.bilstm(cnn_out)
        attended, _ = self.backbone.attention(temporal, temporal, temporal, need_weights=False)
        return self.backbone.layer_norm(attended).mean(dim=1)

    def selected_e(self, features: torch.Tensor) -> torch.Tensor:
        keys = torch.stack(list(self.e_keys), dim=0)
        prompts = torch.stack(list(self.e_prompts), dim=0)
        feat_n = F.normalize(features, dim=1)
        key_n = F.normalize(keys, dim=1)
        weights = torch.softmax(feat_n @ key_n.t() / self.prompt_temperature, dim=1)
        return weights @ prompts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.extract(x)
        features = raw + self.g_prompt + self.selected_e(raw)
        logits = self.backbone.classifier(self.backbone.dropout(features))
        return features, logits, F.softmax(logits, dim=1)
