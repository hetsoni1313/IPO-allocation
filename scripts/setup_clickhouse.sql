-- ============================================================
-- ClickHouse Schema: Market Anomaly & IPO Allocation Engine
-- Engine: MergeTree (optimised for time-series append workloads)
-- ============================================================

-- Create dedicated database
CREATE DATABASE IF NOT EXISTS market_anomaly;

-- ──────────────────────────────────────────────────────────────
-- 1. RAW MARKET TICKS
-- Stores both historical batch data and live streaming ticks.
-- Partitioned by month for efficient time-range pruning.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.market_ticks
(
    symbol         LowCardinality(String),
    timestamp      DateTime64(3, 'Asia/Kolkata'),   -- millisecond precision
    open           Float64,
    high           Float64,
    low            Float64,
    close          Float64,
    volume         UInt64,
    vwap           Float64        DEFAULT 0,
    num_trades     UInt32         DEFAULT 0,
    source         LowCardinality(String) DEFAULT 'historical',  -- 'historical' | 'live'
    ingested_at    DateTime64(6)  DEFAULT now64(6)                -- microsecond ingest ts
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
TTL toDateTime(timestamp) + INTERVAL 7 YEAR
SETTINGS index_granularity = 8192;

-- Secondary index on source for fast filtering
ALTER TABLE market_anomaly.market_ticks
    ADD INDEX idx_source (source) TYPE set(2) GRANULARITY 4;


-- ──────────────────────────────────────────────────────────────
-- 2. MODEL PREDICTIONS
-- Stores the LSTM t+1 price predictions alongside actuals.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.predictions
(
    symbol              LowCardinality(String),
    timestamp           DateTime64(3, 'Asia/Kolkata'),
    actual_close        Float64,
    predicted_close     Float64,
    prediction_lower    Float64,         -- lower confidence bound
    prediction_upper    Float64,         -- upper confidence bound
    residual            Float64,         -- actual - predicted
    z_score             Float64,         -- standardised residual
    model_version       String DEFAULT '1.0.0',
    inference_latency_us UInt32 DEFAULT 0, -- microseconds
    created_at          DateTime64(6)    DEFAULT now64(6)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;


-- ──────────────────────────────────────────────────────────────
-- 3. ANOMALIES
-- Flagged anomaly events with severity and context.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.anomalies
(
    anomaly_id      UUID DEFAULT generateUUIDv4(),
    symbol          LowCardinality(String),
    timestamp       DateTime64(3, 'Asia/Kolkata'),
    anomaly_type    LowCardinality(String),  -- 'FLASH_CRASH' | 'SPIKE' | 'VOLUME_SURGE'
    severity        LowCardinality(String),  -- 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    z_score         Float64,
    actual_price    Float64,
    predicted_price Float64,
    deviation_pct   Float64,
    context         String DEFAULT '',       -- JSON blob with extra context
    created_at      DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp, severity)
SETTINGS index_granularity = 4096;


-- ──────────────────────────────────────────────────────────────
-- 4. P&L TRADE LOG
-- Every simulated paper trade with entry/exit details.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.pnl_trades
(
    trade_id        UUID DEFAULT generateUUIDv4(),
    symbol          LowCardinality(String),
    direction       LowCardinality(String),  -- 'LONG' | 'SHORT'
    entry_time      DateTime64(3, 'Asia/Kolkata'),
    exit_time       Nullable(DateTime64(3, 'Asia/Kolkata')),
    entry_price     Float64,
    exit_price      Nullable(Float64),
    quantity         Float64,
    pnl             Float64 DEFAULT 0,
    pnl_pct         Float64 DEFAULT 0,
    status          LowCardinality(String) DEFAULT 'OPEN',  -- 'OPEN' | 'CLOSED' | 'STOPPED_OUT'
    trigger_anomaly_id Nullable(UUID),
    slippage        Float64 DEFAULT 0,
    commission      Float64 DEFAULT 0,
    created_at      DateTime64(6) DEFAULT now64(6)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(entry_time)
ORDER BY (symbol, entry_time)
SETTINGS index_granularity = 4096;


-- ──────────────────────────────────────────────────────────────
-- 5. P&L SUMMARY (Materialized View)
-- Aggregated P&L metrics refreshed on every insert to pnl_trades.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.pnl_summary_store
(
    symbol               LowCardinality(String),
    trade_date           Date,
    total_trades         UInt32,
    winning_trades       UInt32,
    losing_trades        UInt32,
    gross_pnl            Float64,
    total_commission     Float64,
    net_pnl              Float64,
    max_drawdown_pct     Float64,
    avg_trade_pnl        Float64
)
ENGINE = SummingMergeTree()
ORDER BY (symbol, trade_date);

CREATE MATERIALIZED VIEW IF NOT EXISTS market_anomaly.pnl_summary_mv
TO market_anomaly.pnl_summary_store
AS SELECT
    symbol,
    toDate(entry_time) AS trade_date,
    count()                                    AS total_trades,
    countIf(pnl > 0)                           AS winning_trades,
    countIf(pnl <= 0)                          AS losing_trades,
    sum(pnl)                                   AS gross_pnl,
    sum(commission)                            AS total_commission,
    sum(pnl) - sum(commission)                AS net_pnl,
    0                                          AS max_drawdown_pct,
    avg(pnl)                                   AS avg_trade_pnl
FROM market_anomaly.pnl_trades
WHERE status = 'CLOSED'
GROUP BY symbol, toDate(entry_time);


-- ──────────────────────────────────────────────────────────────
-- 6. SYSTEM HEALTH / PIPELINE METRICS
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_anomaly.pipeline_metrics
(
    metric_name     LowCardinality(String),
    metric_value    Float64,
    tags            Map(String, String),
    timestamp       DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (metric_name, timestamp)
TTL toDateTime(timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;
