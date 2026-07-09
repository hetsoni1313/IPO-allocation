"""
Centralized Configuration for Market Anomaly & IPO Allocation Engine.

Uses Pydantic Settings for type-safe, environment-variable-backed configuration.
All tuneable parameters live here — no magic numbers scattered in code.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── ClickHouse Configuration ──────────────────────────────────
@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    native_port: int = int(os.getenv("CLICKHOUSE_NATIVE_PORT", "9000"))
    username: str = os.getenv("CLICKHOUSE_USER", "default")
    password: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    database: str = os.getenv("CLICKHOUSE_DB", "market_anomaly")
    # Batch insert tuning
    insert_batch_size: int = 10_000
    insert_flush_interval_ms: int = 500


# ── Data Ingestion Configuration ──────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    # Historical data params
    symbols: tuple = ("TCS.NS", "BAJAJ-AUTO.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS")
    index_symbols: tuple = ("^NSEI",)  # Nifty 50
    history_period: str = "5y"
    history_interval: str = "1d"  # yfinance free tier: daily for 5y, 1m for 7d
    minute_lookback_days: int = 5  # For minute-level data (yfinance limit)

    # WebSocket / Live stream params
    websocket_provider: str = os.getenv("WS_PROVIDER", "simulated")  # "twelvedata" | "simulated"
    twelvedata_api_key: str = os.getenv("TWELVEDATA_API_KEY", "")
    stream_symbols: tuple = ("TCS.NS", "RELIANCE.NS", "INFY.NS")
    simulated_tick_interval_ms: int = 250  # 4 ticks/sec in simulation mode


# ── LSTM Model Configuration ─────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    # Architecture
    input_features: int = 7   # OHLCV + returns + volatility
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True
    use_attention: bool = True

    # Sequence params
    sequence_length: int = 60   # Look-back window
    forecast_horizon: int = 1   # Predict t+1

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10          # Early stopping patience
    grad_clip_norm: float = 1.0

    # Scheduler
    scheduler: str = "onecycle"  # "onecycle" | "cosine" | "plateau"
    onecycle_max_lr: float = 3e-3
    onecycle_pct_start: float = 0.3

    # Checkpointing
    checkpoint_path: str = str(MODEL_DIR / "lstm_best.pt")
    save_every_n_epochs: int = 5

    # Device — auto-detect Apple MPS
    @property
    def device(self) -> str:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


# ── Anomaly Detection Configuration ──────────────────────────
@dataclass(frozen=True)
class AnomalyConfig:
    # Dynamic threshold params
    z_score_threshold: float = 2.5       # Standard deviations for anomaly
    rolling_window: int = 120            # Window for rolling stats (ticks)
    min_samples_for_detection: int = 30  # Minimum samples before flagging
    cooldown_ticks: int = 20             # Min ticks between anomaly flags

    # Severity classification
    severity_levels: tuple = (
        (2.5, "LOW"),
        (3.0, "MEDIUM"),
        (4.0, "HIGH"),
        (5.0, "CRITICAL"),
    )


# ── P&L / Trading Simulation Configuration ───────────────────
@dataclass(frozen=True)
class TradingConfig:
    initial_capital: float = 1_000_000.0   # INR 10 Lakh
    position_size_pct: float = 0.05        # 5% of capital per trade
    max_open_positions: int = 5
    stop_loss_pct: float = 0.02            # 2% stop loss
    take_profit_pct: float = 0.05          # 5% take profit
    slippage_bps: float = 5.0              # 5 basis points slippage
    commission_bps: float = 3.0            # 3 basis points commission

    # Strategy
    strategy: str = "mean_reversion"       # "mean_reversion" | "momentum"
    entry_on_anomaly_severity: str = "MEDIUM"  # Minimum severity to enter


# ── Dashboard Configuration ──────────────────────────────────
@dataclass(frozen=True)
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 5050
    debug: bool = True
    sse_interval_ms: int = 1000


# ── Master Configuration ─────────────────────────────────────
@dataclass
class AppConfig:
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


# Singleton instance
config = AppConfig()
