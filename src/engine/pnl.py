"""
P&L Simulation Engine — Paper Trading Module.

Implements a mean-reversion trading strategy that:
1. Opens positions when anomalies are detected (mean-reversion on flash crashes).
2. Manages positions with stop-loss and take-profit levels.
3. Tracks slippage and commission costs.
4. Logs all trades to ClickHouse for Tableau dashboards.
5. Computes real-time P&L metrics.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import config
from src.data.clickhouse_client import get_clickhouse


@dataclass
class Position:
    """Represents an open trading position."""

    trade_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trigger_anomaly_id: Optional[str] = None
    unrealised_pnl: float = 0.0

    def update_pnl(self, current_price: float) -> float:
        """Update unrealised P&L."""
        if self.direction == "LONG":
            self.unrealised_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealised_pnl = (self.entry_price - current_price) * self.quantity
        return self.unrealised_pnl

    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if self.direction == "LONG":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit has been hit."""
        if self.direction == "LONG":
            return current_price >= self.take_profit
        return current_price <= self.take_profit


# ── Adaptive Risk Manager (Cognitive Feedback Loop) ──────────
class AdaptiveRiskManager:
    """Self-learning risk manager that adjusts trading parameters based on
    rolling performance.

    Implements:
    - Rolling win rate and Sharpe ratio tracking
    - Kelly Criterion proxy for position sizing
    - Dynamic Z-score threshold adjustment
    - Risk regime classification with console logging

    The system "thinks" after every trade close, adjusting its
    confidence and capital allocation in real-time.
    """

    # Risk regimes
    REGIME_AGGRESSIVE = "AGGRESSIVE"
    REGIME_NEUTRAL = "NEUTRAL"
    REGIME_DEFENSIVE = "DEFENSIVE"
    REGIME_CONSERVATIVE = "CONSERVATIVE"

    def __init__(
        self,
        base_position_pct: float = 0.05,
        base_z_threshold: float = 2.5,
        lookback_window: int = 10,
    ) -> None:
        self._base_position_pct = base_position_pct
        self._base_z_threshold = base_z_threshold
        self._lookback = lookback_window

        # Rolling trade results: list of (pnl, pnl_pct) tuples
        self._trade_results: Deque[Dict[str, float]] = deque(maxlen=50)

        # Current adaptive state
        self._current_regime = self.REGIME_NEUTRAL
        self._z_score_multiplier = 1.0
        self._position_size_pct = base_position_pct
        self._kelly_fraction = base_position_pct

        # Streak tracking
        self._consecutive_wins = 0
        self._consecutive_losses = 0

        # Regime change log for dashboard
        self._regime_log: Deque[Dict[str, Any]] = deque(maxlen=100)

        # Adjustment history for sparklines
        self._position_size_history: Deque[float] = deque(maxlen=50)
        self._z_multiplier_history: Deque[float] = deque(maxlen=50)

        logger.info(
            f"🧠 AdaptiveRiskManager initialized | "
            f"Base position: {base_position_pct*100:.1f}% | "
            f"Base Z-threshold: {base_z_threshold:.2f} | "
            f"Lookback: {lookback_window} trades"
        )

    def on_trade_result(self, pnl: float, pnl_pct: float) -> None:
        """Called after every trade close. Triggers the feedback loop."""
        is_win = pnl > 0
        self._trade_results.append({
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "is_win": is_win,
            "timestamp": datetime.now().isoformat(),
        })

        # Update streaks
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        # Recalculate adaptive parameters
        self._recalculate()

    def _recalculate(self) -> None:
        """Core feedback loop: recalculate all adaptive parameters."""
        results = list(self._trade_results)
        if len(results) < 2:
            return

        # Use lookback window
        recent = results[-self._lookback:]

        # ── Rolling Win Rate ──
        wins = sum(1 for r in recent if r["is_win"])
        win_rate = wins / len(recent)

        # ── Rolling Sharpe ──
        pnls = [r["pnl"] for r in recent]
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
        rolling_sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0

        # ── Kelly Criterion Proxy ──
        # f* = W - (1-W)/R  where W=win_rate, R=avg_win/avg_loss
        avg_win = np.mean([r["pnl"] for r in recent if r["is_win"]]) if wins > 0 else 0
        avg_loss = abs(np.mean([r["pnl"] for r in recent if not r["is_win"]])) if (len(recent) - wins) > 0 else 1
        payoff_ratio = avg_win / max(avg_loss, 1e-10)

        kelly_raw = win_rate - (1 - win_rate) / max(payoff_ratio, 1e-10)
        # Half-Kelly for safety, clamped to [1%, 10%]
        self._kelly_fraction = float(np.clip(kelly_raw * 0.5, 0.01, 0.10))

        # ── Determine Regime ──
        old_regime = self._current_regime

        if self._consecutive_losses >= 3:
            new_regime = self.REGIME_DEFENSIVE
            self._z_score_multiplier = 1.30  # Demand 30% more confidence
            self._position_size_pct = max(self._kelly_fraction * 0.5, 0.01)  # Halve Kelly
        elif self._consecutive_losses >= 2 or rolling_sharpe < -1.0:
            new_regime = self.REGIME_CONSERVATIVE
            self._z_score_multiplier = 1.15
            self._position_size_pct = max(self._kelly_fraction * 0.75, 0.01)
        elif self._consecutive_wins >= 3 and rolling_sharpe > 1.0:
            new_regime = self.REGIME_AGGRESSIVE
            self._z_score_multiplier = 0.85  # Slightly relax threshold
            self._position_size_pct = min(self._kelly_fraction * 1.25, 0.10)
        else:
            new_regime = self.REGIME_NEUTRAL
            self._z_score_multiplier = 1.0
            self._position_size_pct = self._kelly_fraction

        self._current_regime = new_regime

        # Track history for sparklines
        self._position_size_history.append(self._position_size_pct)
        self._z_multiplier_history.append(self._z_score_multiplier)

        # ── Log Regime Change ──
        regime_entry = {
            "timestamp": datetime.now().isoformat(),
            "regime": new_regime,
            "win_rate": round(win_rate * 100, 1),
            "rolling_sharpe": round(rolling_sharpe, 4),
            "kelly_fraction": round(self._kelly_fraction * 100, 2),
            "position_size_pct": round(self._position_size_pct * 100, 2),
            "z_multiplier": round(self._z_score_multiplier, 3),
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "total_trades": len(self._trade_results),
        }
        self._regime_log.append(regime_entry)

        if new_regime != old_regime:
            logger.info(
                f"🧠 RISK REGIME CHANGE: {old_regime} → {new_regime} | "
                f"Win Rate: {win_rate*100:.0f}% | "
                f"Sharpe: {rolling_sharpe:.2f} | "
                f"Kelly: {self._kelly_fraction*100:.1f}% | "
                f"Z-mult: {self._z_score_multiplier:.2f}x | "
                f"Pos Size: {self._position_size_pct*100:.1f}%"
            )
        else:
            logger.debug(
                f"🧠 Risk parameters updated | Regime: {new_regime} | "
                f"WR: {win_rate*100:.0f}% | "
                f"Sharpe: {rolling_sharpe:.2f} | "
                f"Kelly: {self._kelly_fraction*100:.1f}% | "
                f"Size: {self._position_size_pct*100:.1f}%"
            )

    @property
    def position_size_pct(self) -> float:
        """Current adaptive position size (fraction of capital)."""
        return self._position_size_pct

    @property
    def z_score_multiplier(self) -> float:
        """Current Z-score threshold multiplier."""
        return self._z_score_multiplier

    @property
    def regime(self) -> str:
        """Current risk regime."""
        return self._current_regime

    def get_state(self) -> Dict[str, Any]:
        """Get full risk manager state for the dashboard."""
        results = list(self._trade_results)
        recent = results[-self._lookback:] if results else []
        wins = sum(1 for r in recent if r["is_win"]) if recent else 0
        win_rate = (wins / len(recent) * 100) if recent else 0

        pnls = [r["pnl"] for r in recent] if recent else []
        avg_pnl = float(np.mean(pnls)) if pnls else 0
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        return {
            "regime": self._current_regime,
            "win_rate": round(win_rate, 1),
            "rolling_sharpe": round(sharpe, 4),
            "kelly_fraction_pct": round(self._kelly_fraction * 100, 2),
            "position_size_pct": round(self._position_size_pct * 100, 2),
            "z_score_multiplier": round(self._z_score_multiplier, 3),
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
            "total_evaluated": len(self._trade_results),
            "position_size_history": list(self._position_size_history),
            "z_multiplier_history": list(self._z_multiplier_history),
        }

    def get_regime_log(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent regime change log entries."""
        return list(self._regime_log)[-n:]


class PnLSimulator:
    """
    Paper trading simulator that executes trades based on anomaly signals
    and tracks P&L with realistic cost modelling.
    """

    TRADE_COLUMNS = [
        "trade_id", "symbol", "direction", "entry_time", "exit_time",
        "entry_price", "exit_price", "quantity", "pnl", "pnl_pct",
        "status", "trigger_anomaly_id", "slippage", "commission",
    ]

    # Severity priority for filtering
    SEVERITY_PRIORITY = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    def __init__(self, risk_manager: Optional[AdaptiveRiskManager] = None) -> None:
        self.cfg = config.trading
        self._ch = None  # Lazy ClickHouse connection

        # Adaptive risk manager (cognitive feedback loop)
        self.risk_manager = risk_manager or AdaptiveRiskManager(
            base_position_pct=self.cfg.position_size_pct,
            base_z_threshold=config.anomaly.z_score_threshold,
        )

        # Portfolio state
        self.capital = self.cfg.initial_capital
        self.initial_capital = self.cfg.initial_capital
        self.open_positions: Dict[str, List[Position]] = {}
        self.closed_trades: Deque[Dict[str, Any]] = deque(maxlen=10000)

        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_pnl = 0.0
        self.total_commission = 0.0
        self.peak_capital = self.capital
        self.max_drawdown = 0.0

        # V10 FIX: Timestamp monotonicity tracking per symbol
        self._last_timestamp: Dict[str, Any] = {}

        # V11 FIX: Per-symbol position limit
        self._max_positions_per_symbol = 2

        # Entry severity threshold
        self._min_severity = self.SEVERITY_PRIORITY.get(
            self.cfg.entry_on_anomaly_severity, 2
        )

    def on_anomaly(
        self,
        anomaly: Dict[str, Any],
        current_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Process an anomaly signal and potentially open a new position.

        Strategy: Mean Reversion
        - On FLASH_CRASH → go LONG (expect bounce back)
        - On SPIKE → go SHORT (expect pullback)

        Returns trade dict if a position was opened, None otherwise.
        """
        symbol = anomaly["symbol"]
        severity = anomaly.get("severity", "LOW")
        anomaly_type = anomaly.get("anomaly_type", "DEVIATION")

        # Check severity threshold
        if self.SEVERITY_PRIORITY.get(severity, 0) < self._min_severity:
            return None

        # Check global position limits
        total_open = sum(
            len(positions)
            for positions in self.open_positions.values()
        )
        if total_open >= self.cfg.max_open_positions:
            logger.debug("Max open positions reached — skipping entry")
            return None

        # V11 FIX: Check per-symbol position limit
        symbol_positions = len(self.open_positions.get(symbol, []))
        if symbol_positions >= self._max_positions_per_symbol:
            logger.debug(
                f"Max positions per symbol ({self._max_positions_per_symbol}) "
                f"reached for {symbol} — skipping entry"
            )
            return None

        # Determine direction
        if self.cfg.strategy == "mean_reversion":
            if anomaly_type == "FLASH_CRASH":
                direction = "LONG"
            elif anomaly_type == "SPIKE":
                direction = "SHORT"
            else:
                return None  # Skip non-directional anomalies
        else:
            # Momentum strategy
            if anomaly_type == "FLASH_CRASH":
                direction = "SHORT"
            elif anomaly_type == "SPIKE":
                direction = "LONG"
            else:
                return None

        # Calculate position size (adaptive via risk manager)
        position_value = self.capital * self.risk_manager.position_size_pct
        slippage = current_price * (self.cfg.slippage_bps / 10000)

        # Adjust entry price for slippage
        if direction == "LONG":
            adjusted_price = current_price + slippage
        else:
            adjusted_price = current_price - slippage

        quantity = position_value / adjusted_price
        commission = position_value * (self.cfg.commission_bps / 10000)

        # Calculate stop loss and take profit
        if direction == "LONG":
            stop_loss = adjusted_price * (1 - self.cfg.stop_loss_pct)
            take_profit = adjusted_price * (1 + self.cfg.take_profit_pct)
        else:
            stop_loss = adjusted_price * (1 + self.cfg.stop_loss_pct)
            take_profit = adjusted_price * (1 - self.cfg.take_profit_pct)

        # Create position
        trade_id = str(uuid.uuid4())
        position = Position(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=anomaly["timestamp"],
            entry_price=adjusted_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trigger_anomaly_id=anomaly.get("anomaly_id"),
        )

        # Add to open positions
        if symbol not in self.open_positions:
            self.open_positions[symbol] = []
        self.open_positions[symbol].append(position)

        # Deduct commission from capital
        self.capital -= commission
        self.total_commission += commission

        trade_info = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_time": anomaly["timestamp"],
            "entry_price": round(adjusted_price, 2),
            "quantity": round(quantity, 4),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "commission": round(commission, 2),
            "slippage": round(slippage, 2),
            "anomaly_severity": severity,
        }

        logger.info(
            f"📈 TRADE OPENED: {direction} {symbol} @ ₹{adjusted_price:.2f} | "
            f"Qty: {quantity:.2f} | SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}"
        )

        return trade_info

    def on_tick(
        self, symbol: str, current_price: float, timestamp: Any
    ) -> List[Dict[str, Any]]:
        """
        Update all open positions for a symbol on a new tick.
        Closes positions that hit stop-loss or take-profit.

        Returns list of closed trade dicts.
        """
        # V10 FIX: Timestamp monotonicity check — prevents look-ahead bias
        if symbol in self._last_timestamp:
            last_ts = self._last_timestamp[symbol]
            try:
                if timestamp < last_ts:
                    logger.warning(
                        f"Non-monotonic timestamp for {symbol}: "
                        f"{timestamp} < {last_ts}. Skipping tick to prevent "
                        f"look-ahead bias."
                    )
                    return []
            except TypeError:
                pass  # Incomparable timestamps (different types) — allow through
        self._last_timestamp[symbol] = timestamp

        closed = []

        if symbol not in self.open_positions:
            return closed

        remaining = []
        for position in self.open_positions[symbol]:
            position.update_pnl(current_price)

            close_reason = None
            if position.should_stop_loss(current_price):
                close_reason = "STOPPED_OUT"
            elif position.should_take_profit(current_price):
                close_reason = "TAKE_PROFIT"

            if close_reason:
                trade = self._close_position(
                    position, current_price, timestamp, close_reason
                )
                closed.append(trade)
            else:
                remaining.append(position)

        self.open_positions[symbol] = remaining
        return closed

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        timestamp: Any,
        status: str,
    ) -> Dict[str, Any]:
        """Close a position and log the trade."""
        # Apply slippage to exit
        slippage = exit_price * (self.cfg.slippage_bps / 10000)
        if position.direction == "LONG":
            adjusted_exit = exit_price - slippage
        else:
            adjusted_exit = exit_price + slippage

        commission = abs(position.quantity * adjusted_exit) * (
            self.cfg.commission_bps / 10000
        )

        # Calculate P&L
        if position.direction == "LONG":
            pnl = (adjusted_exit - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - adjusted_exit) * position.quantity

        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100

        # Update metrics
        self.total_trades += 1
        self.gross_pnl += pnl
        self.capital += pnl - commission
        self.total_commission += commission

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Feed result into adaptive risk manager (cognitive feedback loop)
        self.risk_manager.on_trade_result(pnl, pnl_pct)

        # Track drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        trade = {
            "trade_id": position.trade_id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_time": position.entry_time,
            "exit_time": timestamp,
            "entry_price": round(position.entry_price, 2),
            "exit_price": round(adjusted_exit, 2),
            "quantity": round(position.quantity, 4),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "status": status,
            "trigger_anomaly_id": position.trigger_anomaly_id,
            "slippage": round(slippage, 2),
            "commission": round(commission, 2),
        }

        self.closed_trades.append(trade)
        self._log_trade(trade)

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} TRADE CLOSED ({status}): {position.direction} "
            f"{position.symbol} | P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%) | "
            f"Capital: ₹{self.capital:,.2f}"
        )

        return trade

    @property
    def ch(self):
        """Lazy ClickHouse connection."""
        if self._ch is None:
            try:
                self._ch = get_clickhouse()
            except Exception:
                return None
        return self._ch

    def _log_trade(self, trade: Dict[str, Any]) -> None:
        """Write closed trade to ClickHouse."""
        try:
            if self.ch is None:
                return
            row = [trade[col] for col in self.TRADE_COLUMNS]
            self.ch.buffer_row(
                table="pnl_trades",
                columns=self.TRADE_COLUMNS,
                row=row,
            )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    # ── Public Metrics API ────────────────────────────────────
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive P&L metrics."""
        net_pnl = self.gross_pnl - self.total_commission
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )
        total_return = (
            (self.capital - self.initial_capital) / self.initial_capital * 100
        )

        # Calculate Sharpe-like ratio from closed trades
        if self.closed_trades:
            pnls = [t["pnl"] for t in self.closed_trades]
            avg_pnl = float(np.mean(pnls))
            std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0
            avg_pnl = 0.0

        open_positions_count = sum(
            len(p) for p in self.open_positions.values()
        )

        return {
            "capital": round(self.capital, 2),
            "initial_capital": self.initial_capital,
            "total_return_pct": round(total_return, 2),
            "net_pnl": round(net_pnl, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "total_commission": round(self.total_commission, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(win_rate, 2),
            "avg_trade_pnl": round(avg_pnl, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "open_positions": open_positions_count,
            "peak_capital": round(self.peak_capital, 2),
        }

    def get_recent_trades(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent closed trades."""
        return list(self.closed_trades)[-n:]

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get the equity curve from closed trades."""
        curve = []
        running_capital = self.initial_capital

        for trade in self.closed_trades:
            running_capital += trade["pnl"] - trade["commission"]
            curve.append({
                "timestamp": trade["exit_time"],
                "capital": round(running_capital, 2),
                "trade_pnl": trade["pnl"],
            })

        return curve
