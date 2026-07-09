"""
Anomaly Detection Module.

Implements a multi-signal anomaly detector that:
1. Uses LSTM prediction residuals (z-scores) as the primary signal.
2. Applies adaptive thresholds based on rolling volatility regime.
3. Classifies anomalies by severity and type (flash crash, spike, volume surge).
4. Enforces cooldown periods to prevent alert fatigue.
5. Logs all anomalies to ClickHouse for Tableau dashboards.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import config
from src.data.clickhouse_client import get_clickhouse


class AnomalyDetector:
    """
    Real-time anomaly detection engine.

    Processes prediction results from the inference engine and flags
    statistically significant deviations as anomalies.
    """

    ANOMALY_COLUMNS = [
        "anomaly_id", "symbol", "timestamp", "anomaly_type", "severity",
        "z_score", "actual_price", "predicted_price", "deviation_pct", "context",
    ]

    def __init__(self, risk_manager=None) -> None:
        self.cfg = config.anomaly
        self._ch = None  # Lazy ClickHouse connection
        self._risk_manager = risk_manager  # Optional AdaptiveRiskManager

        # Per-symbol state
        self._price_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}
        self._last_anomaly_tick: Dict[str, int] = {}
        self._tick_counter: Dict[str, int] = {}
        self._anomaly_count: Dict[str, int] = {}

        # V9 FIX: Per-symbol volatility tracking for adaptive thresholds
        self._return_history: Dict[str, Deque[float]] = {}
        self._volatility_history: Dict[str, Deque[float]] = {}
        self._VOLATILITY_WINDOW = 100
        self._HIGH_VOL_MULTIPLIER = 1.5  # Raise threshold by 50% in high-vol regime
        self._HIGH_VOL_PERCENTILE = 90   # Top 10% = high volatility regime

        # Global anomaly log
        self._recent_anomalies: Deque[Dict[str, Any]] = deque(maxlen=1000)

    def check(
        self,
        prediction: Dict[str, Any],
        tick: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Check a prediction result for anomalies.

        Args:
            prediction: Dict from InferenceEngine.ingest_tick() with keys:
                symbol, timestamp, actual_close, predicted_close,
                prediction_lower, prediction_upper, residual, z_score
            tick: Optional raw tick for volume analysis.

        Returns:
            Anomaly dict if detected, None otherwise.
        """
        symbol = prediction["symbol"]
        z_score = prediction["z_score"]
        actual = prediction["actual_close"]
        predicted = prediction["predicted_close"]

        # Initialise per-symbol tracking
        if symbol not in self._tick_counter:
            self._tick_counter[symbol] = 0
            self._last_anomaly_tick[symbol] = -self.cfg.cooldown_ticks
            self._price_history[symbol] = deque(maxlen=self.cfg.rolling_window)
            self._volume_history[symbol] = deque(maxlen=self.cfg.rolling_window)
            self._anomaly_count[symbol] = 0
            # V9 FIX: Initialize volatility tracking
            self._return_history[symbol] = deque(maxlen=self._VOLATILITY_WINDOW)
            self._volatility_history[symbol] = deque(maxlen=self._VOLATILITY_WINDOW)

        self._tick_counter[symbol] += 1
        self._price_history[symbol].append(actual)

        # V9 FIX: Track returns for volatility regime detection
        if len(self._price_history[symbol]) >= 2:
            prices = list(self._price_history[symbol])
            ret = abs((prices[-1] - prices[-2]) / max(prices[-2], 1e-10))
            self._return_history[symbol].append(ret)
            if len(self._return_history[symbol]) >= 20:
                recent_vol = np.std(list(self._return_history[symbol])[-20:])
                self._volatility_history[symbol].append(recent_vol)

        if tick and "volume" in tick:
            self._volume_history[symbol].append(float(tick["volume"]))

        # ── Check minimum samples ──
        if len(self._price_history[symbol]) < self.cfg.min_samples_for_detection:
            return None

        # ── Check cooldown ──
        ticks_since_last = (
            self._tick_counter[symbol] - self._last_anomaly_tick[symbol]
        )
        if ticks_since_last < self.cfg.cooldown_ticks:
            return None

        # ── Multi-signal anomaly detection ──
        anomaly = self._detect_anomaly(
            symbol=symbol,
            z_score=z_score,
            actual=actual,
            predicted=predicted,
            tick=tick,
            timestamp=prediction["timestamp"],
        )

        if anomaly:
            self._last_anomaly_tick[symbol] = self._tick_counter[symbol]
            self._anomaly_count[symbol] += 1
            self._recent_anomalies.append(anomaly)

            # Write to ClickHouse
            self._log_anomaly(anomaly)

            logger.warning(
                f"🚨 ANOMALY [{anomaly['severity']}] {symbol} @ "
                f"{anomaly['timestamp']} | Type: {anomaly['anomaly_type']} | "
                f"Z-score: {z_score:.2f} | "
                f"Actual: {actual:.2f} vs Predicted: {predicted:.2f}"
            )

        return anomaly

    def _detect_anomaly(
        self,
        symbol: str,
        z_score: float,
        actual: float,
        predicted: float,
        tick: Optional[Dict],
        timestamp: Any,
    ) -> Optional[Dict[str, Any]]:
        """Run all anomaly detection signals."""
        abs_z = abs(z_score)

        # V9 FIX: Adaptive threshold based on volatility regime
        effective_threshold = self._get_adaptive_threshold(symbol)

        # ── Signal 1: Price z-score threshold ──
        if abs_z >= effective_threshold:
            anomaly_type = self._classify_type(z_score, tick)
            severity = self._classify_severity(abs_z)
            deviation_pct = (
                (actual - predicted) / predicted * 100
                if predicted != 0
                else 0
            )

            context = {
                "z_score_abs": float(round(abs_z, 4)),
                "rolling_mean": float(round(
                    np.mean(list(self._price_history[symbol])), 2
                )),
                "rolling_std": float(round(
                    np.std(list(self._price_history[symbol])), 4
                )),
                "tick_number": int(self._tick_counter[symbol]),
                "total_anomalies": int(self._anomaly_count[symbol] + 1),
            }

            # Add volume context if available
            if tick and "volume" in tick and self._volume_history[symbol]:
                vol_arr = np.array(list(self._volume_history[symbol]))
                vol_mean = float(np.mean(vol_arr))
                vol_ratio = float(tick["volume"]) / max(vol_mean, 1.0)
                context["volume_ratio"] = float(round(vol_ratio, 2))

                # Volume surge detection
                if vol_ratio > 3.0:
                    anomaly_type = "VOLUME_SURGE"
                    if severity == "LOW":
                        severity = "MEDIUM"

            return {
                "anomaly_id": str(uuid.uuid4()),
                "symbol": symbol,
                "timestamp": timestamp,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "z_score": float(round(z_score, 4)),
                "actual_price": float(round(actual, 2)),
                "predicted_price": float(round(predicted, 2)),
                "deviation_pct": float(round(deviation_pct, 4)),
                "context": json.dumps(context),
            }

        return None

    def _get_adaptive_threshold(self, symbol: str) -> float:
        """Compute adaptive Z-score threshold based on:

        Layer 1 — Volatility regime: During high-volatility periods,
                  the threshold increases to prevent false positive floods.
        Layer 2 — Trading performance: The AdaptiveRiskManager's Z-score
                  multiplier adjusts based on recent win/loss streaks.
        """
        base_threshold = self.cfg.z_score_threshold
        vol_history = self._volatility_history.get(symbol)

        # Layer 1: Volatility-based adjustment
        vol_multiplier = 1.0
        if vol_history and len(vol_history) >= 30:
            vol_arr = np.array(list(vol_history))
            current_vol = vol_arr[-1]
            vol_percentile = np.percentile(vol_arr, self._HIGH_VOL_PERCENTILE)

            if current_vol > vol_percentile:
                vol_multiplier = self._HIGH_VOL_MULTIPLIER
                logger.debug(
                    f"{symbol} in high-vol regime (vol={current_vol:.6f} > "
                    f"p{self._HIGH_VOL_PERCENTILE}={vol_percentile:.6f}). "
                    f"Vol multiplier: {vol_multiplier:.2f}x"
                )

        # Layer 2: Trading performance adjustment
        risk_multiplier = 1.0
        if self._risk_manager is not None:
            risk_multiplier = self._risk_manager.z_score_multiplier

        effective_threshold = base_threshold * vol_multiplier * risk_multiplier

        if risk_multiplier != 1.0 or vol_multiplier != 1.0:
            logger.debug(
                f"{symbol} adaptive threshold: {base_threshold:.2f} × "
                f"vol({vol_multiplier:.2f}) × risk({risk_multiplier:.2f}) "
                f"= {effective_threshold:.2f}"
            )

        return effective_threshold

    def _classify_type(
        self, z_score: float, tick: Optional[Dict]
    ) -> str:
        """Classify anomaly type based on direction and context."""
        if z_score < -self.cfg.z_score_threshold:
            return "FLASH_CRASH"
        elif z_score > self.cfg.z_score_threshold:
            return "SPIKE"
        return "DEVIATION"

    def _classify_severity(self, abs_z_score: float) -> str:
        """Classify severity based on z-score magnitude."""
        for threshold, severity in self.cfg.severity_levels:
            if abs_z_score < threshold:
                return severity
        return "CRITICAL"

    @property
    def ch(self):
        """Lazy ClickHouse connection — only connects when needed."""
        if self._ch is None:
            try:
                self._ch = get_clickhouse()
            except Exception:
                logger.debug("ClickHouse not available — anomaly logging disabled")
                return None
        return self._ch

    def _log_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """Write anomaly to ClickHouse."""
        try:
            if self.ch is None:
                return
            row = [anomaly[col] for col in self.ANOMALY_COLUMNS]
            self.ch.buffer_row(
                table="anomalies",
                columns=self.ANOMALY_COLUMNS,
                row=row,
            )
        except Exception as e:
            logger.error(f"Failed to log anomaly: {e}")

    # ── Public API ────────────────────────────────────────────
    def get_recent_anomalies(
        self, n: int = 50, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent anomalies from memory."""
        anomalies = list(self._recent_anomalies)
        if symbol:
            anomalies = [a for a in anomalies if a["symbol"] == symbol]
        return anomalies[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        return {
            "total_anomalies": sum(self._anomaly_count.values()),
            "per_symbol": dict(self._anomaly_count),
            "symbols_tracked": len(self._tick_counter),
            "recent_count": len(self._recent_anomalies),
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset state for a symbol or all symbols."""
        if symbol:
            for store in (
                self._price_history,
                self._volume_history,
                self._last_anomaly_tick,
                self._tick_counter,
                self._anomaly_count,
            ):
                store.pop(symbol, None)
        else:
            self._price_history.clear()
            self._volume_history.clear()
            self._last_anomaly_tick.clear()
            self._tick_counter.clear()
            self._anomaly_count.clear()
            self._recent_anomalies.clear()
