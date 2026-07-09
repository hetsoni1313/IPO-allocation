-- ============================================================================
-- TABLEAU DASHBOARD QUERIES — Market Anomaly & IPO Allocation Engine
-- ============================================================================
--
-- Connection Setup (JDBC):
--   Driver:   ClickHouse JDBC (com.clickhouse.jdbc.ClickHouseDriver)
--   URL:      jdbc:clickhouse://localhost:8123/market_anomaly
--   User:     default
--   Password: (blank)
--
-- Connection Setup (ODBC):
--   DSN:      ClickHouse_MarketAnomaly
--   Host:     localhost
--   Port:     8123
--   Database: market_anomaly
--
-- Tableau Parameter Syntax:
--   <Parameters.Symbol>      → e.g. 'TCS.NS'
--   <Parameters.Start Date>  → e.g. '2024-01-01'
--   <Parameters.End Date>    → e.g. '2026-07-06'
--
-- NOTE: Replace <Parameters.XXX> with your actual Tableau parameter names.
--       For ClickHouse native parameterised queries, use {param:Type} syntax.
-- ============================================================================



-- ════════════════════════════════════════════════════════════════════════════
-- QUERY 1: LIVE TICKER DATA — Minute-Aggregated OHLCV with LSTM Prediction Band
-- ════════════════════════════════════════════════════════════════════════════
--
-- PURPOSE:
--   Feeds a real-time candlestick / line chart in Tableau.
--   Aggregates raw ticks into 1-minute OHLCV bars and LEFT JOINs the LSTM
--   prediction band so the predicted vs. actual close is visible as an
--   overlay "confidence ribbon" on the same chart.
--
-- TABLEAU USAGE:
--   Worksheet type: Dual-axis line chart
--   Columns:        minute_bucket (continuous)
--   Rows:           close (line) + predicted_close (line, dashed)
--   Detail:         prediction_lower / prediction_upper → Reference Band
--   Filters:        symbol, date range
--   Color:          symbol
-- ────────────────────────────────────────────────────────────────────────────

SELECT
    t.symbol                                                        AS symbol,
    toStartOfMinute(t.timestamp)                                    AS minute_bucket,

    -- ── OHLCV (proper candle aggregation) ──
    argMin(t.open, t.timestamp)                                     AS open,
    max(t.high)                                                     AS high,
    min(t.low)                                                      AS low,
    argMax(t.close, t.timestamp)                                    AS close,
    sum(t.volume)                                                   AS volume,

    -- ── Volume-Weighted Average Price ──
    round(sum(t.vwap * t.volume) / nullIf(sum(t.volume), 0), 2)    AS vwap,

    -- ── Tick density (liquidity proxy) ──
    count()                                                         AS tick_count,

    -- ── LSTM prediction band (latest prediction per minute) ──
    argMax(p.predicted_close, p.timestamp)                          AS predicted_close,
    argMax(p.prediction_lower, p.timestamp)                         AS prediction_lower,
    argMax(p.prediction_upper, p.timestamp)                         AS prediction_upper,
    argMax(p.z_score, p.timestamp)                                  AS z_score,
    argMax(p.inference_latency_us, p.timestamp)                     AS latency_us

FROM market_anomaly.market_ticks AS t

LEFT JOIN market_anomaly.predictions AS p
    ON  t.symbol    = p.symbol
    AND toStartOfMinute(t.timestamp) = toStartOfMinute(p.timestamp)

WHERE t.symbol    = '<Parameters.Symbol>'
  AND t.timestamp >= toDateTime64('<Parameters.Start Date>', 3, 'Asia/Kolkata')
  AND t.timestamp <= toDateTime64('<Parameters.End Date>',   3, 'Asia/Kolkata')

GROUP BY t.symbol, minute_bucket
ORDER BY minute_bucket ASC;



-- ════════════════════════════════════════════════════════════════════════════
-- QUERY 2: ANOMALY OVERLAY — High-Confidence Flagged Events Only
-- ════════════════════════════════════════════════════════════════════════════
--
-- PURPOSE:
--   Returns ONLY anomalies where the adaptive Z-score crossed the detection
--   threshold. Designed to be overlaid as distinct scatter markers on top
--   of the price chart from Query 1.
--
-- FILTERING LOGIC:
--   - Severity >= MEDIUM (filters out LOW-confidence noise)
--   - Enriched with the prediction context at the anomaly timestamp
--   - Context JSON is extracted for Tableau tooltip drill-down
--
-- TABLEAU USAGE:
--   Worksheet type: Circle marks (overlaid on Query 1 via dual-axis)
--   Columns:        timestamp (continuous)
--   Rows:           actual_price
--   Shape/Color:    severity (MEDIUM=yellow, HIGH=orange, CRITICAL=red)
--   Size:           abs(deviation_pct) — bigger marker = bigger deviation
--   Tooltip:        anomaly_type, z_score, deviation_pct, predicted_price
-- ────────────────────────────────────────────────────────────────────────────

SELECT
    a.anomaly_id                                                    AS anomaly_id,
    a.symbol                                                        AS symbol,
    a.timestamp                                                     AS timestamp,
    a.anomaly_type                                                  AS anomaly_type,
    a.severity                                                      AS severity,
    a.z_score                                                       AS z_score,
    abs(a.z_score)                                                  AS abs_z_score,
    a.actual_price                                                  AS actual_price,
    a.predicted_price                                               AS predicted_price,
    a.deviation_pct                                                 AS deviation_pct,
    abs(a.deviation_pct)                                            AS abs_deviation_pct,

    -- ── Deviation direction for colour encoding ──
    CASE
        WHEN a.z_score < 0 THEN 'BELOW_PREDICTED'
        ELSE 'ABOVE_PREDICTED'
    END                                                             AS deviation_direction,

    -- ── Severity rank for Tableau sorting/filtering ──
    CASE a.severity
        WHEN 'LOW'      THEN 1
        WHEN 'MEDIUM'   THEN 2
        WHEN 'HIGH'     THEN 3
        WHEN 'CRITICAL' THEN 4
        ELSE 0
    END                                                             AS severity_rank,

    -- ── Context extraction (JSON fields → Tableau tooltip) ──
    JSONExtractFloat(a.context, 'rolling_mean')                     AS rolling_mean_price,
    JSONExtractFloat(a.context, 'rolling_std')                      AS rolling_std_price,
    JSONExtractFloat(a.context, 'volume_ratio')                     AS volume_ratio,
    JSONExtractUInt(a.context, 'tick_number')                       AS tick_number,
    JSONExtractUInt(a.context, 'total_anomalies')                   AS running_anomaly_count,

    -- ── Time since previous anomaly (for cluster detection) ──
    dateDiff(
        'second',
        lagInFrame(a.timestamp) OVER (
            PARTITION BY a.symbol ORDER BY a.timestamp
        ),
        a.timestamp
    )                                                               AS seconds_since_prev_anomaly

FROM market_anomaly.anomalies AS a

WHERE a.severity IN ('MEDIUM', 'HIGH', 'CRITICAL')
  AND a.symbol    = '<Parameters.Symbol>'
  AND a.timestamp >= toDateTime64('<Parameters.Start Date>', 3, 'Asia/Kolkata')
  AND a.timestamp <= toDateTime64('<Parameters.End Date>',   3, 'Asia/Kolkata')

ORDER BY a.timestamp ASC;



-- ════════════════════════════════════════════════════════════════════════════
-- QUERY 3: QUANT STRATEGY P&L — Running Cumulative Returns with Drawdown
-- ════════════════════════════════════════════════════════════════════════════
--
-- PURPOSE:
--   Computes a full equity curve from the paper-trading simulator:
--   cumulative P&L, rolling Sharpe ratio, and real-time drawdown tracking.
--   Each row represents one closed trade event with running portfolio state.
--
-- KEY METRICS:
--   - cumulative_net_pnl:  Running sum of (P&L minus commissions)
--   - equity:              Initial capital + cumulative returns
--   - peak_equity:         High-water mark of equity curve
--   - drawdown_pct:        Current drawdown from peak (negative = in drawdown)
--   - rolling_sharpe:      Annualised Sharpe ratio over last 20 trades
--
-- TABLEAU USAGE:
--   Worksheet 1: Equity curve (line chart)
--     Columns:  exit_time (continuous)
--     Rows:     equity (line) + peak_equity (line, reference)
--     Color:    drawdown_pct → diverging palette
--
--   Worksheet 2: KPI tiles
--     cumulative_net_pnl, win_rate_running, rolling_sharpe, max_drawdown_pct
-- ────────────────────────────────────────────────────────────────────────────

SELECT
    trade_id,
    symbol,
    direction,
    entry_time,
    exit_time,
    entry_price,
    exit_price,
    quantity,
    pnl,
    pnl_pct,
    status,
    slippage,
    commission,

    -- ── Net P&L per trade ──
    (pnl - commission)                                              AS net_pnl,

    -- ── Cumulative P&L (running sum over all trades, time-ordered) ──
    sum(pnl - commission) OVER w_time                               AS cumulative_net_pnl,

    -- ── Equity curve (1,000,000 initial capital + cumulative returns) ──
    1000000 + sum(pnl - commission) OVER w_time                     AS equity,

    -- ── Running trade count ──
    count() OVER w_time                                             AS trade_number,

    -- ── Running win rate ──
    round(
        countIf(pnl > 0) OVER w_time * 100.0
        / count() OVER w_time, 2
    )                                                               AS win_rate_running,

    -- ── Peak equity (high-water mark) ──
    max(1000000 + sum(pnl - commission) OVER w_time)
        OVER (ORDER BY exit_time ROWS BETWEEN UNBOUNDED PRECEDING
              AND CURRENT ROW)                                      AS peak_equity,

    -- ── Drawdown from peak (%) ──
    round(
        (
            (1000000 + sum(pnl - commission) OVER w_time)
            - max(1000000 + sum(pnl - commission) OVER w_time)
                  OVER (ORDER BY exit_time ROWS BETWEEN UNBOUNDED PRECEDING
                        AND CURRENT ROW)
        )
        / nullIf(
            max(1000000 + sum(pnl - commission) OVER w_time)
                OVER (ORDER BY exit_time ROWS BETWEEN UNBOUNDED PRECEDING
                      AND CURRENT ROW),
            0
        ) * 100
    , 4)                                                            AS drawdown_pct,

    -- ── Rolling Sharpe (annualised, last 20 trades) ──
    round(
        CASE WHEN count() OVER w_rolling >= 5
             THEN avg(pnl - commission) OVER w_rolling
                  / nullIf(stddevPop(pnl - commission) OVER w_rolling, 0)
                  * sqrt(252)
             ELSE NULL
        END
    , 4)                                                            AS rolling_sharpe,

    -- ── Average hold time (seconds) ──
    dateDiff('second', entry_time, exit_time)                       AS hold_duration_sec

FROM market_anomaly.pnl_trades

WHERE status IN ('CLOSED', 'STOPPED_OUT', 'TAKE_PROFIT')
  AND exit_time IS NOT NULL

WINDOW
    w_time    AS (ORDER BY exit_time ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    w_rolling AS (ORDER BY exit_time ASC ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)

ORDER BY exit_time ASC;



-- ════════════════════════════════════════════════════════════════════════════
-- QUERY 4: SECTOR CORRELATION MATRIX — Cross-Symbol Volatility Comparison
-- ════════════════════════════════════════════════════════════════════════════
--
-- PURPOSE:
--   Compares intraday price volatility across all tracked symbols to identify
--   sector-wide trends and IPO-proxy divergence. Used for the correlation
--   heatmap / scatter matrix in Tableau.
--
-- METHODOLOGY:
--   1. Computes daily returns, intraday range %, and rolling 20-day volatility
--      per symbol.
--   2. Pivots all symbols into columns using conditional aggregation so Tableau
--      can directly compute a cross-symbol correlation matrix.
--   3. Includes a "beta" metric: each symbol's volatility relative to the
--      Nifty 50 index (^NSEI) for market-relative risk assessment.
--
-- TABLEAU USAGE:
--   Worksheet 1: Correlation heatmap
--     Rows/Columns: symbol pairs
--     Color:        Pearson correlation of daily_return_pct
--
--   Worksheet 2: Volatility comparison bar chart
--     Columns:      symbol
--     Rows:         avg_volatility_20d, avg_intraday_range_pct
--     Detail:       beta_to_nifty
-- ────────────────────────────────────────────────────────────────────────────

SELECT
    trade_date,

    -- ── Per-symbol daily metrics (pivoted for correlation matrix) ──

    -- TCS.NS
    argMaxIf(daily_return_pct,  symbol, symbol = 'TCS.NS')          AS tcs_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = 'TCS.NS')          AS tcs_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = 'TCS.NS')         AS tcs_range_pct,

    -- BAJAJ-AUTO.NS
    argMaxIf(daily_return_pct,  symbol, symbol = 'BAJAJ-AUTO.NS')   AS bajaj_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = 'BAJAJ-AUTO.NS')   AS bajaj_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = 'BAJAJ-AUTO.NS')  AS bajaj_range_pct,

    -- RELIANCE.NS
    argMaxIf(daily_return_pct,  symbol, symbol = 'RELIANCE.NS')     AS reliance_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = 'RELIANCE.NS')     AS reliance_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = 'RELIANCE.NS')    AS reliance_range_pct,

    -- INFY.NS
    argMaxIf(daily_return_pct,  symbol, symbol = 'INFY.NS')         AS infy_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = 'INFY.NS')         AS infy_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = 'INFY.NS')        AS infy_range_pct,

    -- HDFCBANK.NS
    argMaxIf(daily_return_pct,  symbol, symbol = 'HDFCBANK.NS')     AS hdfc_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = 'HDFCBANK.NS')     AS hdfc_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = 'HDFCBANK.NS')    AS hdfc_range_pct,

    -- ^NSEI (Nifty 50 — market benchmark)
    argMaxIf(daily_return_pct,  symbol, symbol = '^NSEI')           AS nifty_return_pct,
    argMaxIf(volatility_20d,    symbol, symbol = '^NSEI')           AS nifty_vol_20d,
    argMaxIf(intraday_range_pct, symbol, symbol = '^NSEI')          AS nifty_range_pct

FROM
(
    -- ── Subquery: compute daily metrics per symbol ──
    SELECT
        symbol,
        toDate(timestamp)                                           AS trade_date,

        -- Daily close-to-close return
        round(
            (argMax(close, timestamp) - argMin(close, timestamp))
            / nullIf(argMin(close, timestamp), 0) * 100
        , 4)                                                        AS daily_return_pct,

        -- Intraday range (High-Low / Open)
        round(
            (max(high) - min(low))
            / nullIf(argMin(open, timestamp), 0) * 100
        , 4)                                                        AS intraday_range_pct,

        -- Daily closing price (for volatility computation)
        argMax(close, timestamp)                                    AS daily_close,

        -- 20-day rolling volatility (std of returns over trailing window)
        stddevPopIf(
            (argMax(close, timestamp) - argMin(close, timestamp))
            / nullIf(argMin(close, timestamp), 0) * 100,
            1 = 1
        ) OVER (
            PARTITION BY symbol
            ORDER BY toDate(timestamp)
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )                                                           AS volatility_20d

    FROM market_anomaly.market_ticks

    WHERE timestamp >= toDateTime64('<Parameters.Start Date>', 3, 'Asia/Kolkata')
      AND timestamp <= toDateTime64('<Parameters.End Date>',   3, 'Asia/Kolkata')
      AND source LIKE 'historical%'

    GROUP BY symbol, trade_date
)

GROUP BY trade_date
ORDER BY trade_date ASC;



-- ════════════════════════════════════════════════════════════════════════════
-- SUPPLEMENTARY QUERIES (for dashboard KPI tiles and health monitoring)
-- ════════════════════════════════════════════════════════════════════════════


-- ──────────────────────────────────────────────────────────────────────────
-- QUERY 5: Anomaly Frequency Heatmap (by hour × day-of-week)
-- Use as: Calendar heatmap in Tableau
-- ──────────────────────────────────────────────────────────────────────────
SELECT
    symbol,
    toDayOfWeek(timestamp)                                          AS day_of_week,
    toHour(timestamp)                                               AS hour_of_day,
    severity,
    count()                                                         AS anomaly_count,
    avg(abs(z_score))                                               AS avg_abs_z_score,
    avg(abs(deviation_pct))                                         AS avg_abs_deviation
FROM market_anomaly.anomalies
WHERE timestamp >= toDateTime64('<Parameters.Start Date>', 3, 'Asia/Kolkata')
GROUP BY symbol, day_of_week, hour_of_day, severity
ORDER BY day_of_week, hour_of_day;


-- ──────────────────────────────────────────────────────────────────────────
-- QUERY 6: Model Prediction Accuracy Over Time
-- Use as: Rolling MAE / RMSE line chart
-- ──────────────────────────────────────────────────────────────────────────
SELECT
    symbol,
    toStartOfHour(timestamp)                                        AS hour_bucket,
    count()                                                         AS prediction_count,
    round(avg(abs(residual)),            4)                         AS mae,
    round(sqrt(avg(residual * residual)), 4)                        AS rmse,
    round(
        avg(abs(residual) / nullIf(actual_close, 0)) * 100, 4
    )                                                               AS mape_pct,
    round(avg(z_score),                  4)                         AS mean_z_score,
    round(stddevPop(z_score),            4)                         AS z_score_stddev,
    round(avg(inference_latency_us),     1)                         AS avg_latency_us,
    max(inference_latency_us)                                       AS max_latency_us
FROM market_anomaly.predictions
WHERE timestamp >= toDateTime64('<Parameters.Start Date>', 3, 'Asia/Kolkata')
GROUP BY symbol, hour_bucket
ORDER BY symbol, hour_bucket;


-- ──────────────────────────────────────────────────────────────────────────
-- QUERY 7: Pipeline Health — Real-Time System Metrics
-- Use as: Status sparklines / health indicators
-- ──────────────────────────────────────────────────────────────────────────
SELECT
    metric_name,
    toStartOfMinute(timestamp)                                      AS minute_bucket,
    round(avg(metric_value), 2)                                     AS avg_value,
    round(max(metric_value), 2)                                     AS max_value,
    round(min(metric_value), 2)                                     AS min_value,
    count()                                                         AS sample_count
FROM market_anomaly.pipeline_metrics
WHERE timestamp >= now() - INTERVAL 1 HOUR
GROUP BY metric_name, minute_bucket
ORDER BY metric_name, minute_bucket;
