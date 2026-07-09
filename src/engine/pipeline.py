"""
Pipeline Orchestrator — Glues all components into one unified engine.

Manages the lifecycle of:
1. Data ingestion (historical + live)
2. LSTM training and inference
3. Anomaly detection
4. P&L simulation
5. ClickHouse write-through

Provides both batch mode (historical analysis) and live mode
(real-time streaming inference).
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from config.settings import config, PROJECT_ROOT, LOG_DIR
from src.data.clickhouse_client import get_clickhouse
from src.data.historical import HistoricalDataFetcher
from src.data.streaming import LiveDataStream
from src.model.inference import InferenceEngine
from src.model.preprocessor import DataPreprocessor
from src.model.train import Trainer
from src.model.lstm import build_model
from src.engine.anomaly import AnomalyDetector
from src.engine.pnl import PnLSimulator, AdaptiveRiskManager


def _push_dashboard_event(event_type: str, data: Dict[str, Any]) -> None:
    """Safely push an event to the dashboard SSE queue.

    Uses lazy import to avoid circular dependency and silently
    ignores failures so the hot path is never blocked.
    """
    try:
        from src.dashboard.app import push_event
        push_event(event_type, data)
    except Exception:
        pass  # Dashboard not running — ignore silently


class Pipeline:
    """
    Master orchestrator for the Market Anomaly Detection Engine.

    Usage:
        pipeline = Pipeline()
        pipeline.setup()              # Deploy schema, ingest historical data
        pipeline.train()              # Train the LSTM model
        pipeline.run_live()           # Start live inference + anomaly detection
    """

    def __init__(self) -> None:
        # Components (lazy-initialised)
        self.ch = get_clickhouse()
        self.historical_fetcher = HistoricalDataFetcher()
        self.live_stream = LiveDataStream()
        self.inference_engine: Optional[InferenceEngine] = None

        # Shared adaptive risk manager (cognitive feedback loop)
        self.risk_manager = AdaptiveRiskManager(
            base_position_pct=config.trading.position_size_pct,
            base_z_threshold=config.anomaly.z_score_threshold,
        )
        self.anomaly_detector = AnomalyDetector(risk_manager=self.risk_manager)
        self.pnl_simulator = PnLSimulator(risk_manager=self.risk_manager)

        # State
        self._is_setup = False
        self._is_trained = False
        self._running = False
        self._tick_count = 0
        self._prediction_count = 0
        self._start_time: Optional[float] = None

        # Historical data cache — persists across phases so the live stream
        # can be anchored to the exact last close of each symbol.
        self._historical_df: Optional[pd.DataFrame] = None

        # Configure logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure structured logging."""
        log_path = LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_path),
            rotation="100 MB",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} | {message}",
        )

    # ════════════════════════════════════════════════════════════
    # PHASE 1: SETUP
    # ════════════════════════════════════════════════════════════
    def setup(self, skip_historical: bool = False) -> None:
        """
        Deploy ClickHouse schema and ingest historical data.

        Args:
            skip_historical: If True, skip historical data fetch
                             (useful for re-runs).
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Environment & Data Setup")
        logger.info("=" * 60)

        # ── Deploy ClickHouse schema ──
        schema_path = PROJECT_ROOT / "scripts" / "setup_clickhouse.sql"
        if schema_path.exists():
            self.ch.deploy_schema(str(schema_path))
        else:
            logger.warning(f"Schema file not found: {schema_path}")

        # ── Ingest historical data ──
        if not skip_historical:
            logger.info("Fetching historical data from yfinance...")
            stats = self.historical_fetcher.fetch_all()
            logger.success(f"Historical ingestion: {stats}")
        else:
            logger.info("Skipping historical data fetch")

        # ── Verify data ──
        try:
            count = self.ch.get_table_count("market_ticks")
            logger.info(f"market_ticks table has {count:,} rows")
        except Exception as e:
            logger.warning(f"Could not verify table count: {e}")

        self._is_setup = True
        logger.success("Setup complete ✓")

    # ════════════════════════════════════════════════════════════
    # PHASE 2: TRAINING
    # ════════════════════════════════════════════════════════════
    def train(
        self,
        symbols: list = None,
        val_split: float = 0.15,
    ) -> Dict:
        """
        Train the LSTM model on historical data.

        Returns:
            Training metrics dictionary.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: LSTM Model Training")
        logger.info("=" * 60)

        # Fetch training data from ClickHouse
        try:
            train_df = self.historical_fetcher.get_training_data(symbols)
        except Exception as e:
            logger.warning(
                f"Could not fetch from ClickHouse: {e}. "
                "Falling back to direct yfinance fetch."
            )
            # Fallback: fetch directly from yfinance
            train_df = self._fetch_training_data_fallback(symbols)

        if train_df.empty:
            logger.error("No training data available!")
            return {"error": "No training data"}

        logger.info(f"Training data: {len(train_df)} rows")

        # Initialise trainer
        preprocessor = DataPreprocessor()
        model = build_model()
        trainer = Trainer(model=model, preprocessor=preprocessor)

        # Check for existing checkpoint
        checkpoint_path = Path(config.model.checkpoint_path)
        if checkpoint_path.exists():
            logger.info("Found existing checkpoint — resuming training")
            trainer.load_checkpoint()

        # Train
        metrics = trainer.train(train_df, val_split=val_split)

        # Cache for live-stream anchoring and inference warm-up
        self._historical_df = train_df

        self._is_trained = True
        logger.success(f"Training complete: {metrics}")
        return metrics

    def _fetch_training_data_fallback(
        self, symbols: list = None
    ) -> pd.DataFrame:
        """Fetch training data directly from yfinance as fallback."""
        import yfinance as yf

        if symbols is None:
            symbols = list(config.data.symbols)

        all_dfs = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="5y", interval="1d")
                if not df.empty:
                    df = df.reset_index()
                    col_map = {
                        "Date": "timestamp", "Datetime": "timestamp",
                        "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume",
                    }
                    df = df.rename(columns=col_map)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"]
                        ).dt.tz_localize(None)
                    df["symbol"] = symbol
                    all_dfs.append(df)
                    logger.info(f"Fetched {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    # ════════════════════════════════════════════════════════════
    # PHASE 3: LIVE INFERENCE
    # ════════════════════════════════════════════════════════════
    def run_live(self, duration_seconds: Optional[int] = None) -> None:
        """
        Start the live inference pipeline.

        Args:
            duration_seconds: Run for this many seconds, then stop.
                              None = run indefinitely.
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Live Inference & Anomaly Detection")
        logger.info("=" * 60)

        # Initialise inference engine
        self.inference_engine = InferenceEngine()

        if not self.inference_engine.is_ready:
            logger.warning(
                "Model not loaded — running in data-collection-only mode. "
                "Train the model first with pipeline.train()"
            )

        # Warm up inference buffers with historical data
        self._warm_up_inference()

        # Anchor the simulated stream to the exact last historical close
        self._anchor_live_stream()

        # Register tick callback
        self.live_stream.register_callback(self._on_live_tick)

        # Start streaming
        self._running = True
        self._start_time = time.time()
        self.live_stream.start()

        logger.info("Live pipeline is running. Press Ctrl+C to stop.")

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run loop
        try:
            while self._running:
                time.sleep(1)

                # Periodic status logging
                if self._tick_count % 100 == 0 and self._tick_count > 0:
                    self._log_status()

                # Duration check
                if duration_seconds:
                    elapsed = time.time() - self._start_time
                    if elapsed >= duration_seconds:
                        logger.info(
                            f"Duration limit reached ({duration_seconds}s)"
                        )
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _on_live_tick(self, tick: Dict[str, Any]) -> None:
        """
        Callback for each live tick. This is the hot path.

        Flow: tick → inference → anomaly check → P&L update → dashboard push
        """
        self._tick_count += 1
        symbol = tick["symbol"]
        current_price = tick["close"]

        # ── Step 1: Run inference ──
        prediction = None
        if self.inference_engine and self.inference_engine.is_ready:
            prediction = self.inference_engine.ingest_tick(tick)

        # ── Step 2: Push tick to dashboard (with prediction overlay) ──
        ts = tick["timestamp"]
        ts_iso = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        tick_event = {
            "symbol": symbol,
            "timestamp": ts_iso,
            "open": float(tick.get("open", current_price)),
            "high": float(tick.get("high", current_price)),
            "low": float(tick.get("low", current_price)),
            "close": float(current_price),
            "volume": int(tick.get("volume", 0)),
        }
        if prediction:
            tick_event["predicted_close"] = prediction.get("predicted_close")
            tick_event["z_score"] = prediction.get("z_score")
        _push_dashboard_event("tick", tick_event)

        # ── Step 3: Log prediction to ClickHouse ──
        if prediction:
            self._prediction_count += 1
            self._log_prediction(prediction)

            # ── Step 4: Check for anomalies ──
            anomaly = self.anomaly_detector.check(prediction, tick)

            # ── Step 5: Execute trading logic ──
            if anomaly:
                self.pnl_simulator.on_anomaly(anomaly, current_price)
                _push_dashboard_event("anomaly", anomaly)

        # ── Step 6: Update open positions ──
        closed_trades = self.pnl_simulator.on_tick(
            symbol, current_price, tick["timestamp"]
        )

        # ── Step 7: Push closed trades + risk state to dashboard ──
        for trade in closed_trades:
            _push_dashboard_event("trade", trade)

        # ── Step 8: Push risk state & metrics after trade events ──
        if closed_trades:
            _push_dashboard_event("risk_state", self.risk_manager.get_state())
        # Push metrics every 10 ticks to keep KPIs updated
        if self._tick_count % 10 == 0:
            _push_dashboard_event("metrics", self.pnl_simulator.get_metrics())

    def _log_prediction(self, prediction: Dict[str, Any]) -> None:
        """Write prediction to ClickHouse."""
        columns = [
            "symbol", "timestamp", "actual_close", "predicted_close",
            "prediction_lower", "prediction_upper", "residual",
            "z_score", "inference_latency_us",
        ]
        row = [prediction.get(col, 0) for col in columns]
        self.ch.buffer_row(table="predictions", columns=columns, row=row)

    def _warm_up_inference(self) -> None:
        """Pre-fill inference buffers with recent historical data.

        Tries ClickHouse first; falls back to the cached _historical_df
        (which is always available after training).
        """
        if not self.inference_engine:
            return

        warm_df: Optional[pd.DataFrame] = None

        # Source 1: ClickHouse
        try:
            symbols = list(config.data.stream_symbols)
            placeholders = ", ".join(f"'{s}'" for s in symbols)
            warm_df = self.ch.query_df(f"""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM market_ticks
                WHERE symbol IN ({placeholders})
                ORDER BY symbol, timestamp DESC
                LIMIT 500
            """)
            if warm_df.empty:
                warm_df = None
        except Exception as e:
            logger.debug(f"ClickHouse warm-up unavailable: {e}")

        # Source 2: Cached historical DataFrame
        if warm_df is None and self._historical_df is not None:
            symbols = list(config.data.stream_symbols)
            frames = []
            for sym in symbols:
                sym_df = self._historical_df[
                    self._historical_df["symbol"] == sym
                ].copy()
                if not sym_df.empty:
                    # Use the most recent rows (tail of the sorted data)
                    sym_df = sym_df.sort_values("timestamp").tail(500)
                    frames.append(sym_df)
            if frames:
                warm_df = pd.concat(frames, ignore_index=True)
                logger.info(
                    f"Warming up inference from cached historical data "
                    f"({len(warm_df)} rows)"
                )

        if warm_df is not None and not warm_df.empty:
            self.inference_engine.warm_up(warm_df)
        else:
            logger.warning("No data available for inference warm-up")

    def _anchor_live_stream(self) -> None:
        """Extract the last historical close per symbol and inject into the
        live stream so the GBM simulation starts from a realistic price."""
        if self._historical_df is None or self._historical_df.empty:
            logger.debug("No historical data to anchor live stream")
            return

        last_prices: Dict[str, float] = {}
        for symbol in self._historical_df["symbol"].unique():
            sym_df = self._historical_df[
                self._historical_df["symbol"] == symbol
            ].sort_values("timestamp")
            if not sym_df.empty:
                last_prices[symbol] = float(sym_df.iloc[-1]["close"])

        if last_prices:
            self.live_stream.start_prices = last_prices
            logger.info(
                f"Anchored live stream to historical closes: "
                f"{{{', '.join(f'{s}: ₹{p:,.2f}' for s, p in last_prices.items())}}}"
            )

    def _log_status(self) -> None:
        """Log pipeline status periodically."""
        elapsed = time.time() - (self._start_time or time.time())
        tps = self._tick_count / max(elapsed, 1)

        inference_stats = {}
        if self.inference_engine:
            inference_stats = self.inference_engine.get_stats()

        anomaly_stats = self.anomaly_detector.get_stats()
        pnl_metrics = self.pnl_simulator.get_metrics()

        logger.info(
            f"📊 STATUS | Ticks: {self._tick_count:,} ({tps:.1f}/s) | "
            f"Predictions: {self._prediction_count:,} | "
            f"Anomalies: {anomaly_stats['total_anomalies']} | "
            f"Trades: {pnl_metrics['total_trades']} | "
            f"P&L: ₹{pnl_metrics['net_pnl']:,.2f} | "
            f"Capital: ₹{pnl_metrics['capital']:,.2f}"
        )

        # Log to ClickHouse pipeline_metrics
        try:
            self.ch.buffer_row(
                table="pipeline_metrics",
                columns=["metric_name", "metric_value", "tags"],
                row=["ticks_per_second", tps, {"component": "pipeline"}],
            )
            if inference_stats.get("avg_latency_us"):
                self.ch.buffer_row(
                    table="pipeline_metrics",
                    columns=["metric_name", "metric_value", "tags"],
                    row=[
                        "inference_latency_us",
                        inference_stats["avg_latency_us"],
                        {"component": "inference"},
                    ],
                )
        except Exception:
            pass

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        logger.info("Stopping pipeline...")
        self._running = False
        self.live_stream.stop()
        self.ch.flush_all()

        # Final metrics
        pnl_metrics = self.pnl_simulator.get_metrics()
        anomaly_stats = self.anomaly_detector.get_stats()

        logger.info("=" * 60)
        logger.info("PIPELINE FINAL REPORT")
        logger.info("=" * 60)
        logger.info(f"Total ticks processed: {self._tick_count:,}")
        logger.info(f"Total predictions: {self._prediction_count:,}")
        logger.info(f"Total anomalies: {anomaly_stats['total_anomalies']}")
        logger.info(f"Total trades: {pnl_metrics['total_trades']}")
        logger.info(f"Net P&L: ₹{pnl_metrics['net_pnl']:,.2f}")
        logger.info(f"Win rate: {pnl_metrics['win_rate_pct']}%")
        logger.info(f"Max drawdown: {pnl_metrics['max_drawdown_pct']}%")
        logger.info(f"Final capital: ₹{pnl_metrics['capital']:,.2f}")
        logger.info("=" * 60)

    # ════════════════════════════════════════════════════════════
    # BATCH MODE (for backtesting on historical data)
    # ════════════════════════════════════════════════════════════
    def run_backtest(
        self,
        symbols: list = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        Run the anomaly detection and trading logic on historical data.

        This is useful for backtesting the strategy without live data.
        """
        logger.info("=" * 60)
        logger.info("BACKTEST MODE")
        logger.info("=" * 60)

        # Initialise inference engine
        self.inference_engine = InferenceEngine()

        if not self.inference_engine.is_ready:
            logger.error("Model not trained — cannot backtest")
            return {"error": "Model not trained"}

        # Fetch historical data
        if symbols is None:
            symbols = list(config.data.symbols)

        try:
            placeholders = ", ".join(f"'{s}'" for s in symbols)
            where_clause = f"symbol IN ({placeholders})"
            if start_date:
                where_clause += f" AND timestamp >= '{start_date}'"
            if end_date:
                where_clause += f" AND timestamp <= '{end_date}'"

            df = self.ch.query_df(f"""
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM market_ticks
                WHERE {where_clause}
                ORDER BY timestamp ASC
            """)
        except Exception as e:
            logger.warning(f"ClickHouse fetch failed: {e}. Using yfinance fallback.")
            df = self._fetch_training_data_fallback(symbols)

        if df.empty:
            return {"error": "No data for backtest"}

        logger.info(f"Backtesting on {len(df)} ticks...")

        # Replay ticks
        for _, row in df.iterrows():
            tick = {
                "symbol": row["symbol"],
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
            self._on_live_tick(tick)

        # Final results
        self.ch.flush_all()
        pnl_metrics = self.pnl_simulator.get_metrics()
        anomaly_stats = self.anomaly_detector.get_stats()

        results = {
            "ticks_processed": len(df),
            "anomalies": anomaly_stats,
            "pnl": pnl_metrics,
        }

        logger.success(f"Backtest complete: {results}")
        return results
