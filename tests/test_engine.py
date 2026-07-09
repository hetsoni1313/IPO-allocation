"""
Tests for the Engine layer (Anomaly Detection + P&L).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAnomalyDetector:
    """Tests for anomaly detection logic."""

    def test_no_anomaly_normal_data(self):
        """Normal predictions should not trigger anomalies."""
        from src.engine.anomaly import AnomalyDetector

        detector = AnomalyDetector()

        # Feed enough normal data to pass min_samples check
        # Use deterministic z-scores well below threshold (2.5)
        for i in range(50):
            prediction = {
                "symbol": "TCS.NS",
                "timestamp": datetime(2024, 1, 1) + timedelta(minutes=i),
                "actual_close": 3800.0 + (i % 5 - 2) * 0.5,
                "predicted_close": 3800.0,
                "z_score": 0.1 * (i % 3 - 1),  # Values: -0.1, 0, 0.1
            }
            result = detector.check(prediction)

        # With z-scores well below threshold (2.5), no anomaly should be flagged
        assert detector._anomaly_count.get("TCS.NS", 0) == 0

    def test_anomaly_detected_high_z_score(self):
        """High z-scores should trigger anomalies after warm-up."""
        from src.engine.anomaly import AnomalyDetector

        detector = AnomalyDetector()

        # Warm up with normal data
        for i in range(35):
            prediction = {
                "symbol": "TCS.NS",
                "timestamp": datetime(2024, 1, 1) + timedelta(minutes=i),
                "actual_close": 3800.0,
                "predicted_close": 3800.0,
                "z_score": 0.1,
            }
            detector.check(prediction)

        # Now inject a high z-score
        anomaly_prediction = {
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 1, 1) + timedelta(minutes=100),
            "actual_close": 3600.0,  # Big drop
            "predicted_close": 3800.0,
            "z_score": -4.0,  # Well beyond threshold
        }
        result = detector.check(anomaly_prediction)

        assert result is not None
        assert result["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert result["anomaly_type"] == "FLASH_CRASH"

    def test_cooldown_enforcement(self):
        """Anomalies should respect cooldown period."""
        from src.engine.anomaly import AnomalyDetector

        detector = AnomalyDetector()

        # Warm up
        for i in range(35):
            detector.check({
                "symbol": "TCS.NS",
                "timestamp": datetime(2024, 1, 1) + timedelta(minutes=i),
                "actual_close": 3800.0,
                "predicted_close": 3800.0,
                "z_score": 0.1,
            })

        # First anomaly — should trigger
        result1 = detector.check({
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 1, 1) + timedelta(minutes=36),
            "actual_close": 3600.0,
            "predicted_close": 3800.0,
            "z_score": -5.0,
        })
        assert result1 is not None

        # Immediate second anomaly — should be blocked by cooldown
        result2 = detector.check({
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 1, 1) + timedelta(minutes=37),
            "actual_close": 3550.0,
            "predicted_close": 3800.0,
            "z_score": -6.0,
        })
        assert result2 is None  # Cooldown enforced

    def test_severity_classification(self):
        """Test severity levels map correctly."""
        from src.engine.anomaly import AnomalyDetector

        detector = AnomalyDetector()

        assert detector._classify_severity(2.0) == "LOW"
        assert detector._classify_severity(2.7) == "MEDIUM"
        assert detector._classify_severity(3.5) == "HIGH"
        assert detector._classify_severity(5.5) == "CRITICAL"


class TestPnLSimulator:
    """Tests for the P&L simulation engine."""

    def test_initial_state(self):
        """Test that simulator initialises with correct state."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        assert sim.capital == 1_000_000.0
        assert sim.total_trades == 0
        assert sim.gross_pnl == 0.0
        assert len(sim.open_positions) == 0

    def test_trade_on_flash_crash(self):
        """Test that a LONG trade opens on flash crash anomaly."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        anomaly = {
            "anomaly_id": "test-1",
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 6, 1, 10, 30),
            "anomaly_type": "FLASH_CRASH",
            "severity": "HIGH",
            "z_score": -4.0,
            "actual_price": 3600.0,
            "predicted_price": 3800.0,
            "deviation_pct": -5.26,
        }

        result = sim.on_anomaly(anomaly, 3600.0)

        assert result is not None
        assert result["direction"] == "LONG"
        assert result["symbol"] == "TCS.NS"
        assert len(sim.open_positions["TCS.NS"]) == 1

    def test_trade_on_spike(self):
        """Test that a SHORT trade opens on spike anomaly."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        anomaly = {
            "anomaly_id": "test-2",
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 6, 1, 10, 30),
            "anomaly_type": "SPIKE",
            "severity": "HIGH",
            "z_score": 4.0,
            "actual_price": 4100.0,
            "predicted_price": 3800.0,
            "deviation_pct": 7.89,
        }

        result = sim.on_anomaly(anomaly, 4100.0)

        assert result is not None
        assert result["direction"] == "SHORT"

    def test_stop_loss(self):
        """Test that stop loss closes positions."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        # Open LONG position
        anomaly = {
            "anomaly_id": "test-3",
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 6, 1, 10, 30),
            "anomaly_type": "FLASH_CRASH",
            "severity": "HIGH",
            "z_score": -4.0,
            "actual_price": 3600.0,
            "predicted_price": 3800.0,
            "deviation_pct": -5.26,
        }
        sim.on_anomaly(anomaly, 3600.0)
        assert len(sim.open_positions["TCS.NS"]) == 1

        # Price drops further — should trigger stop loss
        stop_price = sim.open_positions["TCS.NS"][0].stop_loss - 10
        closed = sim.on_tick("TCS.NS", stop_price, datetime(2024, 6, 1, 11, 0))

        assert len(closed) == 1
        assert closed[0]["status"] == "STOPPED_OUT"
        assert closed[0]["pnl"] < 0  # Should be a loss
        assert len(sim.open_positions["TCS.NS"]) == 0

    def test_take_profit(self):
        """Test that take profit closes positions."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        # Open LONG position
        anomaly = {
            "anomaly_id": "test-4",
            "symbol": "TCS.NS",
            "timestamp": datetime(2024, 6, 1, 10, 30),
            "anomaly_type": "FLASH_CRASH",
            "severity": "HIGH",
            "z_score": -4.0,
            "actual_price": 3600.0,
            "predicted_price": 3800.0,
            "deviation_pct": -5.26,
        }
        sim.on_anomaly(anomaly, 3600.0)

        # Price rises to take profit
        tp_price = sim.open_positions["TCS.NS"][0].take_profit + 10
        closed = sim.on_tick("TCS.NS", tp_price, datetime(2024, 6, 1, 14, 0))

        assert len(closed) == 1
        assert closed[0]["status"] == "TAKE_PROFIT"
        assert closed[0]["pnl"] > 0  # Should be a profit

    def test_metrics_calculation(self):
        """Test P&L metrics are calculated correctly."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()
        metrics = sim.get_metrics()

        assert metrics["capital"] == 1_000_000.0
        assert metrics["total_trades"] == 0
        assert metrics["win_rate_pct"] == 0
        assert metrics["net_pnl"] == 0

    def test_max_positions_limit(self):
        """Test that max open positions is enforced."""
        from src.engine.pnl import PnLSimulator

        sim = PnLSimulator()

        # Open max positions
        for i in range(sim.cfg.max_open_positions):
            sim.on_anomaly({
                "anomaly_id": f"test-{i}",
                "symbol": f"SYM{i}",
                "timestamp": datetime(2024, 6, 1, 10, 30),
                "anomaly_type": "FLASH_CRASH",
                "severity": "HIGH",
                "z_score": -4.0,
                "actual_price": 1000.0,
                "predicted_price": 1200.0,
                "deviation_pct": -16.67,
            }, 1000.0)

        # This should be rejected
        result = sim.on_anomaly({
            "anomaly_id": "test-overflow",
            "symbol": "OVERFLOW",
            "timestamp": datetime(2024, 6, 1, 10, 35),
            "anomaly_type": "FLASH_CRASH",
            "severity": "CRITICAL",
            "z_score": -10.0,
            "actual_price": 500.0,
            "predicted_price": 1000.0,
            "deviation_pct": -50.0,
        }, 500.0)

        assert result is None  # Rejected due to max positions
