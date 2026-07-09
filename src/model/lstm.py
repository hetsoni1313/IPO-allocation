"""
LSTM Model Architecture for Time-Series Price Forecasting.

Architecture choices:
- Bidirectional LSTM for capturing both past and future-looking patterns.
- Multi-head self-attention layer for learning which timesteps matter most.
- Layer normalisation for training stability on financial data.
- Residual connections between LSTM layers.
- Dropout for regularisation.

Optimised for Apple Silicon MPS acceleration.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from config.settings import config


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.
    Learns which timesteps in the sequence are most informative
    for the prediction task.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        B, T, C = x.shape
        residual = x

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, T, C)
        )
        attn_output = self.out_proj(attn_output)

        # Residual + LayerNorm
        output = self.layer_norm(residual + attn_output)

        return output, attn_weights


class LSTMForecaster(nn.Module):
    """
    Production-grade LSTM with attention for financial time-series forecasting.

    Architecture:
        Input → Linear Projection → [Bidirectional LSTM × N layers] →
        Temporal Attention → FC Head → Output (predicted price)
    """

    def __init__(
        self,
        input_features: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        bidirectional: Optional[bool] = None,
        use_attention: Optional[bool] = None,
        forecast_horizon: Optional[int] = None,
    ) -> None:
        super().__init__()

        cfg = config.model
        self.input_features = input_features or cfg.input_features
        self.hidden_size = hidden_size or cfg.hidden_size
        self.num_layers = num_layers or cfg.num_layers
        self.dropout_rate = dropout or cfg.dropout
        self.bidirectional = bidirectional if bidirectional is not None else cfg.bidirectional
        self.use_attention = use_attention if use_attention is not None else cfg.use_attention
        self.forecast_horizon = forecast_horizon or cfg.forecast_horizon

        self.num_directions = 2 if self.bidirectional else 1
        lstm_output_size = self.hidden_size * self.num_directions

        # ── Input projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # ── LSTM stack ──
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        # ── Layer norm after LSTM ──
        self.lstm_norm = nn.LayerNorm(lstm_output_size)

        # ── Temporal attention ──
        if self.use_attention:
            self.attention = TemporalAttention(
                hidden_size=lstm_output_size, num_heads=4
            )

        # ── Output head ──
        self.output_head = nn.Sequential(
            nn.Linear(lstm_output_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.forecast_horizon),
        )

        # Initialise weights
        self._init_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            f"LSTMForecaster initialised: {total_params:,} total params "
            f"({trainable_params:,} trainable)"
        )

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for better gradient flow."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better long-term memory
                if "bias_ih" in name or "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_features)
            return_attention: If True, also return attention weights

        Returns:
            predictions: (batch, forecast_horizon)
            attention_weights: (batch, num_heads, seq_len, seq_len) — optional
        """
        # Input projection
        x = self.input_proj(x)  # (B, T, hidden)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, hidden*dirs)
        lstm_out = self.lstm_norm(lstm_out)

        # Attention
        attn_weights = None
        if self.use_attention:
            lstm_out, attn_weights = self.attention(lstm_out)

        # Use the last timestep's output for prediction
        last_output = lstm_out[:, -1, :]  # (B, hidden*dirs)

        # Output head
        predictions = self.output_head(last_output)  # (B, forecast_horizon)

        if return_attention and attn_weights is not None:
            return predictions, attn_weights
        return predictions

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-mode forward pass with no gradient computation.

        Note: Caller is responsible for setting model to eval mode.
        InferenceEngine does this once after loading the checkpoint.
        """
        with torch.no_grad():
            return self.forward(x)


def build_model(device: Optional[str] = None) -> LSTMForecaster:
    """Factory function to create and move model to the appropriate device."""
    model = LSTMForecaster()
    device = device or config.model.device
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    return model
