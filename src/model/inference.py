"""
Real-Time Inference Engine for the LSTM Forecaster.

Design decisions:
- Maintains a rolling buffer of recent ticks per symbol for sequence construction.
- Computes prediction intervals using calibrated conformal prediction.
- Tracks prediction residuals for dynamic threshold adjustment.
- Sub-millisecond inference on Apple MPS / CPU.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config.settings import config, MODEL_DIR
from src.model.lstm import LSTMForecaster
from src.model.preprocessor import DataPreprocessor


def _normalize_timestamp(ts: Any) -> Any:
    """Strip timezone info from a datetime/Timestamp to prevent mixed-tz comparisons.

    ClickHouse returns tz-aware datetimes (Asia/Kolkata), yfinance returns UTC,
    and the simulated stream uses naive datetime.now(). Mixing these in a single
    pandas DataFrame triggers 'can't compare offset-naive and offset-aware
    datetimes'. This helper standardises everything to naive.
    """
    if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
        # Convert to UTC then strip tz
        try:
            return ts.tz_convert('UTC').tz_localize(None)
        except TypeError:
            # Already localized — just remove tzinfo
            return ts.replace(tzinfo=None)
    return ts


class InferenceEngine:
    """
    Real-time inference engine.

    Maintains per-symbol rolling buffers and produces predictions
    with confidence intervals on each new tick.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.cfg = config.model
        self.device = self.cfg.device
        self.seq_length = self.cfg.sequence_length

        # Load model and preprocessor from checkpoint
        self.model: Optional[LSTMForecaster] = None
        self.preprocessor = DataPreprocessor()
        self._load_model(checkpoint_path or self.cfg.checkpoint_path)

        # Per-symbol rolling buffers
        # Each buffer stores raw tick dicts for sequence construction
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = {}
        # Buffer size: seq_length + extra for feature computation warm-up
        self._buffer_size = self.seq_length + 50  # Extra for rolling features

        # Residual tracking for dynamic confidence intervals
        self._residuals: Dict[str, Deque[float]] = {}
        self._residual_window = 200

        # Performance metrics
        self._total_inferences = 0
        self._total_latency_us = 0

    def _load_model(self, checkpoint_path: str) -> None:
        """Load the trained LSTM model from checkpoint."""
        path = Path(checkpoint_path)

        if not path.exists():
            logger.warning(
                f"No checkpoint at {path} — inference will return None "
                f"until a model is trained."
            )
            return

        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )

        # Rebuild model from saved config
        model_cfg = checkpoint.get("model_config", {})
        self.model = LSTMForecaster(
            input_features=model_cfg.get("input_features", self.cfg.input_features),
            hidden_size=model_cfg.get("hidden_size", self.cfg.hidden_size),
            num_layers=model_cfg.get("num_layers", self.cfg.num_layers),
            dropout=model_cfg.get("dropout", self.cfg.dropout),
            bidirectional=model_cfg.get("bidirectional", self.cfg.bidirectional),
            use_attention=model_cfg.get("use_attention", self.cfg.use_attention),
            forecast_horizon=model_cfg.get(
                "forecast_horizon", self.cfg.forecast_horizon
            ),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Restore preprocessor
        if "preprocessor_state" in checkpoint:
            self.preprocessor.load_state(checkpoint["preprocessor_state"])

        logger.success(
            f"Inference engine loaded from {path} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None and self.preprocessor._is_fitted

    def ingest_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a new tick and return a prediction if the buffer is full.

        Args:
            tick: Dict with keys: symbol, timestamp, open, high, low, close, volume

        Returns:
            Prediction dict or None if insufficient data.
            {
                "symbol": str,
                "timestamp": datetime,
                "actual_close": float,
                "predicted_close": float,
                "prediction_lower": float,
                "prediction_upper": float,
                "residual": float,
                "z_score": float,
                "inference_latency_us": int,
            }
        """
        if not self.is_ready:
            return None

        symbol = tick["symbol"]

        # Initialise buffer for new symbols
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self._buffer_size)
            self._residuals[symbol] = deque(maxlen=self._residual_window)

        # Add tick to buffer (normalise timestamp at ingestion boundary)
        tick = dict(tick)  # defensive copy — don't mutate caller's dict
        tick["timestamp"] = _normalize_timestamp(tick["timestamp"])
        self._buffers[symbol].append(tick)

        # Need enough data for feature computation + sequence
        if len(self._buffers[symbol]) < self.seq_length + 30:
            return None

        # Run inference
        return self._predict(symbol, tick)

    def _predict(
        self, symbol: str, current_tick: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a prediction for the given symbol."""
        start_us = time.perf_counter_ns() // 1000

        try:
            # Build DataFrame from buffer
            buffer_data = list(self._buffers[symbol])
            df = pd.DataFrame(buffer_data)

            # Ensure required columns
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                return None

            # Normalise timestamps to tz-naive (defensive — should already
            # be clean from ingest_tick/warm_up, but protects against any
            # edge case where tz-aware data leaks through)
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                if df["timestamp"].dt.tz is not None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            else:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"]
                ).dt.tz_localize(None)

            # Create sequence
            sequence = self.preprocessor.create_single_sequence(df)
            if sequence is None:
                return None

            # Inference
            x = torch.FloatTensor(sequence).to(self.device)

            with torch.no_grad():
                raw_prediction = self.model.predict(x)

            # Convert prediction back to price space (per-symbol scaler)
            pred_norm = raw_prediction.cpu().numpy().flatten()
            predicted_price = self.preprocessor.inverse_transform_price(
                pred_norm, symbol=symbol
            )[0]

            actual_price = float(current_tick["close"])

            # Compute residual
            residual = actual_price - predicted_price

            # Track residual for dynamic thresholds
            self._residuals[symbol].append(residual)

            # Compute z-score
            z_score = self._compute_z_score(symbol, residual)

            # Compute confidence interval
            lower, upper = self._compute_confidence_interval(
                symbol, predicted_price
            )

            # Latency
            end_us = time.perf_counter_ns() // 1000
            latency_us = end_us - start_us
            self._total_inferences += 1
            self._total_latency_us += latency_us

            return {
                "symbol": symbol,
                "timestamp": current_tick["timestamp"].isoformat() if hasattr(current_tick["timestamp"], 'isoformat') else str(current_tick["timestamp"]),
                "actual_close": float(actual_price),
                "predicted_close": float(round(predicted_price, 2)),
                "prediction_lower": float(round(lower, 2)),
                "prediction_upper": float(round(upper, 2)),
                "residual": float(round(residual, 4)),
                "z_score": float(round(z_score, 4)),
                "inference_latency_us": int(latency_us),
            }

        except Exception as e:
            logger.error(f"Inference error for {symbol}: {e}")
            return None

    def _compute_z_score(self, symbol: str, residual: float) -> float:
        """Compute z-score of current residual vs historical residuals."""
        residuals = self._residuals[symbol]

        if len(residuals) < 10:
            return 0.0

        residual_arr = np.array(residuals)
        mean = np.mean(residual_arr)
        std = np.std(residual_arr)

        if std < 1e-10:
            return 0.0

        return float((residual - mean) / std)

    def _compute_confidence_interval(
        self, symbol: str, predicted_price: float
    ) -> Tuple[float, float]:
        """
        Compute prediction interval using calibrated residuals.
        Uses the empirical distribution of past residuals (conformal prediction).
        """
        residuals = self._residuals[symbol]

        if len(residuals) < 20:
            # Default: ±2% until we have enough residual history
            margin = predicted_price * 0.02
            return predicted_price - margin, predicted_price + margin

        residual_arr = np.abs(np.array(residuals))

        # 95% prediction interval via quantile of absolute residuals
        q95 = float(np.percentile(residual_arr, 95))

        return predicted_price - q95, predicted_price + q95

    def get_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics."""
        avg_latency = (
            self._total_latency_us / max(self._total_inferences, 1)
        )
        return {
            "total_inferences": self._total_inferences,
            "avg_latency_us": float(round(avg_latency, 1)),
            "symbols_tracked": len(self._buffers),
            "buffer_sizes": {
                sym: len(buf) for sym, buf in self._buffers.items()
            },
        }

    def warm_up(self, historical_df: pd.DataFrame) -> None:
        """
        Pre-fill inference buffers with recent historical data.
        Call this before starting the live stream.

        The data may come from ClickHouse (DESC order) or the cached
        training DataFrame (ASC order), so we always sort ascending.
        """
        for symbol in historical_df["symbol"].unique():
            symbol_df = (
                historical_df[historical_df["symbol"] == symbol]
                .sort_values("timestamp", ascending=True)
                .tail(self._buffer_size)
            )

            for _, row in symbol_df.iterrows():
                tick = {
                    "symbol": symbol,
                    "timestamp": _normalize_timestamp(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                # Add to buffer without running inference
                if symbol not in self._buffers:
                    self._buffers[symbol] = deque(maxlen=self._buffer_size)
                    self._residuals[symbol] = deque(
                        maxlen=self._residual_window
                    )
                self._buffers[symbol].append(tick)

        logger.info(
            f"Inference engine warmed up: "
            + ", ".join(
                f"{sym}={len(buf)} rows"
                for sym, buf in self._buffers.items()
            )
        )

        # Dynamically fit scalers for symbols that weren't in training
        for sym, buf in self._buffers.items():
            if sym not in self.preprocessor._symbol_price_scalers:
                close_vals = np.array([t["close"] for t in buf])
                self.preprocessor.register_live_scaler(sym, close_vals)
