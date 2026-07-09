#!/usr/bin/env python3
"""
Market Anomaly & IPO Allocation Engine — Main Entry Point.

Usage:
    # Full pipeline (setup → train → live)
    python main.py

    # Individual phases
    python main.py --phase setup          # Deploy schema + ingest historical data
    python main.py --phase train          # Train LSTM model
    python main.py --phase live           # Start live inference pipeline
    python main.py --phase backtest       # Run backtest on historical data
    python main.py --phase dashboard      # Start dashboard only

    # Options
    python main.py --skip-historical      # Skip yfinance data fetch
    python main.py --duration 300         # Run live for 300 seconds
    python main.py --no-dashboard         # Disable web dashboard

    # Demo mode (no ClickHouse required)
    python main.py --demo                 # Run with simulated data + in-memory model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


console = Console()


def print_banner() -> None:
    """Print the startup banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║     ⚡ Market Anomaly & IPO Allocation Engine ⚡             ║
║                                                              ║
║     Real-Time LSTM Inference · Anomaly Detection             ║
║     ClickHouse · PyTorch · Quantitative Analysis             ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner)


def print_config_summary() -> None:
    """Print current configuration."""
    from config.settings import config

    table = Table(title="Configuration Summary", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Setting", style="white")
    table.add_column("Value", style="green")

    # ClickHouse
    table.add_row("ClickHouse", "Host", f"{config.clickhouse.host}:{config.clickhouse.port}")
    table.add_row("ClickHouse", "Database", config.clickhouse.database)

    # Data
    table.add_row("Data", "Symbols", ", ".join(config.data.symbols))
    table.add_row("Data", "Stream Provider", config.data.websocket_provider)
    table.add_row("Data", "History Period", config.data.history_period)

    # Model
    table.add_row("Model", "Device", config.model.device)
    table.add_row("Model", "Hidden Size", str(config.model.hidden_size))
    table.add_row("Model", "Num Layers", str(config.model.num_layers))
    table.add_row("Model", "Seq Length", str(config.model.sequence_length))
    table.add_row("Model", "Bidirectional", str(config.model.bidirectional))
    table.add_row("Model", "Attention", str(config.model.use_attention))

    # Anomaly
    table.add_row("Anomaly", "Z-Score Threshold", str(config.anomaly.z_score_threshold))

    # Trading
    table.add_row("Trading", "Initial Capital", f"₹{config.trading.initial_capital:,.0f}")
    table.add_row("Trading", "Strategy", config.trading.strategy)
    table.add_row("Trading", "Position Size", f"{config.trading.position_size_pct * 100}%")

    console.print(table)


def run_demo_mode() -> None:
    """
    Demo mode: runs without ClickHouse using direct yfinance data + in-memory training.
    Perfect for quick portfolio demonstrations.
    """
    console.print(Panel(
        "[bold yellow]🎮 DEMO MODE[/bold yellow]\n"
        "Running without ClickHouse — using direct yfinance data.",
        title="Demo Mode",
        border_style="yellow",
    ))

    import pandas as pd
    import yfinance as yf
    from config.settings import config
    from src.model.preprocessor import DataPreprocessor
    from src.model.lstm import build_model
    from src.model.train import Trainer
    from src.model.inference import InferenceEngine
    from src.engine.anomaly import AnomalyDetector
    from src.engine.pnl import PnLSimulator

    # ── Step 1: Fetch data ──
    console.print("\n[bold cyan]Step 1:[/bold cyan] Fetching market data...")
    symbols = list(config.data.symbols)[:3]  # Use 3 symbols for demo

    all_dfs = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")
            if not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if "date" in df.columns:
                    df = df.rename(columns={"date": "timestamp"})
                elif "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df["symbol"] = symbol
                all_dfs.append(df)
                console.print(f"  ✓ {symbol}: {len(df)} rows")
        except Exception as e:
            console.print(f"  ✗ {symbol}: {e}", style="red")

    if not all_dfs:
        console.print("[bold red]No data fetched — exiting demo.[/bold red]")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    console.print(f"\n  Total: {len(full_df)} rows across {len(all_dfs)} symbols\n")

    # ── Step 2: Train model ──
    console.print("[bold cyan]Step 2:[/bold cyan] Training LSTM model...")

    preprocessor = DataPreprocessor()
    model = build_model()
    trainer = Trainer(model=model, preprocessor=preprocessor)

    metrics = trainer.train(full_df, val_split=0.15)

    console.print(Panel(
        f"Epochs: {metrics['total_epochs']}\n"
        f"Best Val Loss: {metrics['best_val_loss']:.6f}\n"
        f"Training Time: {metrics['total_time_seconds']:.1f}s",
        title="Training Results",
        border_style="green",
    ))

    # ── Step 3: Run inference on held-out data ──
    console.print("\n[bold cyan]Step 3:[/bold cyan] Running inference & anomaly detection...")

    inference_engine = InferenceEngine()
    anomaly_detector = AnomalyDetector()
    pnl_simulator = PnLSimulator()

    if not inference_engine.is_ready:
        console.print("[yellow]Model checkpoint not loaded — skipping inference demo[/yellow]")
        return

    # Warm up inference buffers with training data (first 80%) so predictions
    # start immediately when the test data begins — no cold-start gap.
    for symbol in symbols:
        symbol_df = full_df[full_df["symbol"] == symbol].sort_values("timestamp")
        split_idx = int(len(symbol_df) * 0.8)
        train_partition = symbol_df.iloc[:split_idx]
        if not train_partition.empty:
            inference_engine.warm_up(train_partition)
            last_close = train_partition.iloc[-1]["close"]
            console.print(
                f"  🔗 {symbol} warm-up: {len(train_partition)} rows, "
                f"last close: ₹{float(last_close):,.2f}"
            )

    # Use last 20% of data as "live" data
    for symbol in symbols:
        symbol_df = full_df[full_df["symbol"] == symbol].sort_values("timestamp")
        split_idx = int(len(symbol_df) * 0.8)
        test_df = symbol_df.iloc[split_idx:]

        predictions_made = 0
        anomalies_found = 0

        for _, row in test_df.iterrows():
            tick = {
                "symbol": row["symbol"],
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }

            prediction = inference_engine.ingest_tick(tick)
            if prediction:
                predictions_made += 1
                anomaly = anomaly_detector.check(prediction, tick)
                if anomaly:
                    anomalies_found += 1
                    pnl_simulator.on_anomaly(anomaly, tick["close"])

            # Update positions
            pnl_simulator.on_tick(symbol, tick["close"], tick["timestamp"])

        console.print(
            f"  {symbol}: {predictions_made} predictions, "
            f"{anomalies_found} anomalies"
        )

    # ── Step 4: Print results ──
    pnl_metrics = pnl_simulator.get_metrics()
    anomaly_stats = anomaly_detector.get_stats()

    results_table = Table(title="Demo Results", show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Total Anomalies", str(anomaly_stats["total_anomalies"]))
    results_table.add_row("Total Trades", str(pnl_metrics["total_trades"]))
    results_table.add_row("Win Rate", f"{pnl_metrics['win_rate_pct']}%")
    results_table.add_row("Net P&L", f"₹{pnl_metrics['net_pnl']:,.2f}")
    results_table.add_row("Return", f"{pnl_metrics['total_return_pct']}%")
    results_table.add_row("Sharpe Ratio", f"{pnl_metrics['sharpe_ratio']:.4f}")
    results_table.add_row("Max Drawdown", f"{pnl_metrics['max_drawdown_pct']}%")
    results_table.add_row("Final Capital", f"₹{pnl_metrics['capital']:,.2f}")

    console.print(results_table)
    console.print("\n[bold green]✓ Demo complete![/bold green]")


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline or a specific phase."""
    from src.engine.pipeline import Pipeline
    from src.dashboard.app import start_dashboard

    pipeline = Pipeline()

    # Determine phases to run
    phases = []
    if args.phase:
        phases = [args.phase]
    else:
        phases = ["setup", "train", "live"]

    # ── Setup Phase ──
    if "setup" in phases:
        console.print(Panel(
            "Deploying ClickHouse schema and ingesting historical data",
            title="Phase 1: Setup",
            border_style="cyan",
        ))
        try:
            pipeline.setup(skip_historical=args.skip_historical)
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if not args.phase:
                console.print(
                    "[yellow]ClickHouse not available — switching to demo mode.[/yellow]"
                )
                run_demo_mode()
                return

    # ── Train Phase ──
    if "train" in phases:
        console.print(Panel(
            "Training LSTM model on historical data",
            title="Phase 2: Training",
            border_style="cyan",
        ))
        try:
            metrics = pipeline.train()
            console.print(f"Training metrics: {metrics}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if args.phase == "train":
                return

    # ── Dashboard ──
    if not args.no_dashboard and ("live" in phases or "dashboard" in phases):
        # No monkey-patching needed: pipeline._on_live_tick already pushes
        # tick/anomaly/trade events via _push_dashboard_event().

        # Start dashboard
        start_dashboard(pipeline)
        console.print(
            f"[bold green]Dashboard running at http://localhost:{args.port or 5050}[/bold green]"
        )

    # ── Live Phase ──
    if "live" in phases:
        console.print(Panel(
            "Starting real-time inference and anomaly detection",
            title="Phase 3: Live Pipeline",
            border_style="cyan",
        ))
        pipeline.run_live(duration_seconds=args.duration)

    # ── Backtest Phase ──
    if "backtest" in phases:
        console.print(Panel(
            "Running backtest on historical data",
            title="Backtest Mode",
            border_style="cyan",
        ))
        results = pipeline.run_backtest()
        console.print(f"Backtest results: {results}")

    # ── Dashboard Only ──
    if "dashboard" in phases and "live" not in phases:
        start_dashboard(pipeline)
        console.print("Dashboard is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Market Anomaly & IPO Allocation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline
  python main.py --demo             # Demo mode (no ClickHouse)
  python main.py --phase train      # Train only
  python main.py --phase live --duration 600  # Live for 10 minutes
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["setup", "train", "live", "backtest", "dashboard"],
        help="Run a specific phase only",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (no ClickHouse required)",
    )
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip historical data fetch during setup",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds for live mode",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the web dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Dashboard port (default: 5050)",
    )

    args = parser.parse_args()

    # Configure loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="INFO",
    )

    print_banner()
    print_config_summary()

    if args.demo:
        run_demo_mode()
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
