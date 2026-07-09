"""
Tests for the Data Ingestion layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestHistoricalDataFetcher:
    """Tests for yfinance historical data ingestion."""

    def test_yfinance_fetch_single_symbol(self):
        """Verify yfinance can fetch data for a single Indian stock."""
        import yfinance as yf

        ticker = yf.Ticker("TCS.NS")
        df = ticker.history(period="1mo", interval="1d")

        assert not df.empty, "yfinance returned empty data for TCS.NS"
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) >= 10, "Expected at least 10 trading days in 1 month"

    def test_yfinance_fetch_index(self):
        """Verify yfinance can fetch Nifty 50 index data."""
        import yfinance as yf

        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period="1mo", interval="1d")

        assert not df.empty, "yfinance returned empty data for Nifty 50"

    def test_data_preparation(self):
        """Test that data preparation produces correct columns."""
        from src.data.historical import HistoricalDataFetcher

        fetcher = HistoricalDataFetcher.__new__(HistoricalDataFetcher)
        fetcher.cfg = type("cfg", (), {"symbols": ("TCS.NS",), "index_symbols": ("^NSEI",)})()

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame({
            "Date": dates,
            "Open": np.random.uniform(3500, 4000, 50),
            "High": np.random.uniform(3600, 4100, 50),
            "Low": np.random.uniform(3400, 3900, 50),
            "Close": np.random.uniform(3500, 4000, 50),
            "Volume": np.random.randint(100000, 5000000, 50),
        })

        result = fetcher._prepare_dataframe(df, "TCS.NS", "test")

        assert "symbol" in result.columns
        assert "timestamp" in result.columns
        assert "vwap" in result.columns
        assert (result["symbol"] == "TCS.NS").all()
        assert len(result) == 50


class TestStreamingData:
    """Tests for the simulated data feed."""

    def test_simulated_tick_generation(self):
        """Test that simulated ticks have correct structure."""
        from src.data.streaming import LiveDataStream

        stream = LiveDataStream.__new__(LiveDataStream)
        stream.cfg = type("cfg", (), {
            "simulated_tick_interval_ms": 250,
            "stream_symbols": ("TCS.NS",),
        })()

        state = {
            "TCS.NS": {
                "price": 3800.0,
                "volume_base": 50000,
                "volatility": 38.0,
                "trend": 0.0,
            }
        }

        tick = stream._generate_tick("TCS.NS", state)

        assert tick["symbol"] == "TCS.NS"
        assert "timestamp" in tick
        assert "open" in tick
        assert "high" in tick
        assert "low" in tick
        assert "close" in tick
        assert "volume" in tick
        assert "source" in tick
        assert tick["source"] == "live"
        assert tick["high"] >= tick["low"]

    def test_tick_price_movement(self):
        """Test that simulated prices move within reasonable bounds."""
        from src.data.streaming import LiveDataStream

        stream = LiveDataStream.__new__(LiveDataStream)
        stream.cfg = type("cfg", (), {
            "simulated_tick_interval_ms": 250,
            "stream_symbols": ("TCS.NS",),
        })()

        state = {
            "TCS.NS": {
                "price": 3800.0,
                "volume_base": 50000,
                "volatility": 38.0,
                "trend": 0.0,
            }
        }

        prices = []
        for _ in range(100):
            tick = stream._generate_tick("TCS.NS", state)
            prices.append(tick["close"])

        # Prices should stay within reasonable range
        assert min(prices) > 3000, "Price dropped below reasonable floor"
        assert max(prices) < 5000, "Price exceeded reasonable ceiling"
