from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMAttention(nn.Module):
    """Existing 12-lead ECG backbone, adapted to return features, logits, probs."""

    def __init__(
        self,
        input_channels: int = 12,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.5,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        embedding_dim = hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(min(dropout, 0.3)),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn_out = self.cnn(x).transpose(1, 2)
        self.bilstm.flatten_parameters()
        temporal, _ = self.bilstm(cnn_out)
        attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
        features = self.layer_norm(attended).mean(dim=1)
        logits = self.classifier(self.dropout(features))
        probs = F.softmax(logits, dim=1)
        return features, logits, probs
