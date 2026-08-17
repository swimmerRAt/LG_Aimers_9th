"""Compact PatchTST implementation adapted for the monthly baseball panel.

Algorithm source:
    Yuqi Nie et al., "A Time Series is Worth 64 Words" (ICLR 2023)
    https://github.com/yuqinie98/PatchTST
    inspected commit: 204c21efe0b39603ad6e2ca640ef5896646ab1a9

This file is a project-specific reimplementation of the supervised backbone
under the upstream Apache-2.0 license. It preserves RevIN, overlapping
patches, channel-independent shared weights, learnable positional encoding,
residual attention, and the flatten forecasting head. Names and layout were
changed to fit this repository and modern PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization used by the official PatchTST."""

    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        self._mean = values.mean(dim=-1, keepdim=True).detach()
        self._stdev = torch.sqrt(
            values.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        result = (values - self._mean) / self._stdev
        if self.affine:
            result = result * self.weight + self.bias
        return result

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.affine:
            values = (values - self.bias) / (self.weight + self.eps * self.eps)
        return values * self._stdev + self._mean


class ResidualMultiheadAttention(nn.Module):
    """Official-style multi-head attention with cross-layer score residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_dropout: float,
        projection_dropout: float,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(projection_dropout)
        )

    def forward(
        self, values: torch.Tensor, previous_scores: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, width = values.shape
        query = self.query(values).view(batch, tokens, self.n_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = self.key(values).view(batch, tokens, self.n_heads, self.head_dim)
        key = key.permute(0, 2, 3, 1)
        value = self.value(values).view(batch, tokens, self.n_heads, self.head_dim)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key) * self.scale
        if previous_scores is not None:
            scores = scores + previous_scores
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        result = torch.matmul(attention, value)
        result = result.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.output(result), scores


class PatchTSTEncoderLayer(nn.Module):
    def __init__(
        self,
        patch_count: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention_dropout: float,
    ):
        super().__init__()
        self.attention = ResidualMultiheadAttention(
            d_model, n_heads, attention_dropout, dropout
        )
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.BatchNorm1d(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feed_forward_residual_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.BatchNorm1d(d_model)
        self.patch_count = patch_count

    @staticmethod
    def _batch_norm(values: torch.Tensor, layer: nn.BatchNorm1d) -> torch.Tensor:
        return layer(values.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, values: torch.Tensor, previous_scores: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(values, previous_scores)
        values = self._batch_norm(
            values + self.attention_residual_dropout(attended), self.attention_norm
        )
        fed = self.feed_forward(values)
        values = self._batch_norm(
            values + self.feed_forward_residual_dropout(fed), self.feed_forward_norm
        )
        return values, scores


class PatchTST(nn.Module):
    """Supervised PatchTST forecasting backbone with a shared channel head."""

    def __init__(
        self,
        channels: int,
        context_length: int,
        prediction_length: int,
        patch_length: int = 4,
        stride: int = 2,
        n_layers: int = 2,
        d_model: int = 16,
        n_heads: int = 4,
        d_ff: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_end: bool = True,
        revin: bool = True,
    ):
        super().__init__()
        if context_length < patch_length:
            raise ValueError("context_length must be at least patch_length")
        if patch_length <= 0 or stride <= 0:
            raise ValueError("patch_length and stride must be positive")
        self.channels = channels
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        self.padding_end = padding_end
        self.use_revin = revin
        self.patch_count = (context_length - patch_length) // stride + 1
        if padding_end:
            self.patch_count += 1
            self.end_padding = nn.ReplicationPad1d((0, stride))
        self.revin = RevIN(channels) if revin else None
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.empty(self.patch_count, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                PatchTSTEncoderLayer(
                    self.patch_count,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * self.patch_count, prediction_length),
            nn.Dropout(head_dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Forecast from ``[batch, channels, context_length]`` input."""
        if values.ndim != 3 or values.shape[1:] != (
            self.channels,
            self.context_length,
        ):
            raise ValueError(
                "PatchTST input must have shape "
                f"[batch, {self.channels}, {self.context_length}]"
            )
        if self.revin is not None:
            values = self.revin.normalize(values)
        if self.padding_end:
            values = self.end_padding(values)
        patches = values.unfold(-1, self.patch_length, self.stride)
        embedded = self.patch_projection(patches)
        batch, channels, patch_count, width = embedded.shape
        if patch_count != self.patch_count:
            raise RuntimeError("unexpected patch count")
        encoded = embedded.reshape(batch * channels, patch_count, width)
        encoded = self.embedding_dropout(encoded + self.position)
        scores = None
        for layer in self.encoder:
            encoded, scores = layer(encoded, scores)
        encoded = encoded.reshape(batch, channels, patch_count, width)
        encoded = encoded.permute(0, 1, 3, 2)
        forecast = self.head(encoded)
        if self.revin is not None:
            forecast = self.revin.denormalize(forecast)
        return forecast


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
