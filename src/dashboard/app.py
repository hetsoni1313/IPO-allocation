"""
Live Dashboard — Flask + Server-Sent Events (SSE).

Provides a real-time web dashboard that:
1. Streams live tick data and predictions via SSE.
2. Shows anomaly alerts in real-time.
3. Displays P&L metrics and equity curve.
4. Integrates TradingView Lightweight Charts for live candlestick charting.
5. Shows adaptive risk regime from the cognitive feedback loop.
6. Auto-refreshes with sub-second latency.

This is a lightweight companion to Tableau —
designed for real-time monitoring during live operation.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, Generator, Optional

from flask import Flask, Response, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger

from config.settings import config

# ── Pub-Sub SSE broadcaster (one queue per client) ─────────────
class SSEBroadcaster:
    """Fan-out broadcaster: each SSE client gets its own queue.

    The old single-queue design had a fatal flaw — only ONE consumer
    could read each event.  If a browser tab reconnected, the dead
    generator could steal events before the new one started reading.
    This class creates a dedicated queue per connected client and
    copies every event into ALL active queues.
    """

    def __init__(self, per_client_max: int = 256) -> None:
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._per_client_max = per_client_max

    def subscribe(self) -> queue.Queue:
        """Create and return a new per-client queue."""
        q: queue.Queue = queue.Queue(maxsize=self._per_client_max)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a client queue (called when the SSE generator exits)."""
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, event: Dict[str, Any]) -> None:
        """Copy *event* into every active client queue."""
        with self._lock:
            dead: list[queue.Queue] = []
            for q in self._clients:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    # Client is too slow — drop oldest event to make room
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except queue.Empty:
                        dead.append(q)
            for q in dead:
                self._clients.remove(q)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


_broadcaster = SSEBroadcaster()


class _NumpyJSONEncoder(json.JSONEncoder):
    """Handle numpy/pandas types that default json.dumps chokes on."""

    def default(self, obj: Any) -> Any:
        import numpy as np
        # numpy scalar → native Python
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # pandas Timestamp / datetime
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to JSON, safely handling numpy/pandas types."""
    return json.dumps(obj, cls=_NumpyJSONEncoder)


def push_event(event_type: str, data: Dict[str, Any]) -> None:
    """Push an event to ALL connected SSE clients."""
    event = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    _broadcaster.publish(event)


# ── Dashboard HTML Template ────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Anomaly Engine — Trader Terminal</title>
    <meta name="description" content="Institutional-grade market anomaly detection with adaptive risk management and live charting">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root {
            --bg-primary: #06080d;
            --bg-secondary: #0c1017;
            --bg-card: #111520;
            --bg-card-hover: #171d2d;
            --border: #1e2740;
            --border-active: #2d3f6a;
            --text-primary: #e8eaed;
            --text-secondary: #8b92a5;
            --text-muted: #4a5268;
            --accent-blue: #4d8bf5;
            --accent-cyan: #00d4aa;
            --accent-green: #00e676;
            --accent-red: #ff3d57;
            --accent-amber: #ffab00;
            --accent-purple: #a78bfa;
            --accent-orange: #ff6d00;
            --gradient-brand: linear-gradient(135deg, #4d8bf5, #00d4aa);
            --gradient-win: linear-gradient(135deg, #00e676, #00d4aa);
            --gradient-loss: linear-gradient(135deg, #ff3d57, #ff6d00);
            --gradient-defensive: linear-gradient(135deg, #ff3d57, #ffab00);
            --gradient-aggressive: linear-gradient(135deg, #00e676, #4d8bf5);
            --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
            --glass: rgba(255,255,255,0.02);
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter',-apple-system,sans-serif; background:var(--bg-primary); color:var(--text-primary); min-height:100vh; overflow-x:hidden; }
        body::before { content:''; position:fixed; inset:0; background: radial-gradient(ellipse at 15% 30%,rgba(77,139,245,0.06) 0%,transparent 50%), radial-gradient(ellipse at 85% 15%,rgba(0,212,170,0.04) 0%,transparent 50%), radial-gradient(ellipse at 50% 85%,rgba(167,139,250,0.03) 0%,transparent 50%); pointer-events:none; z-index:0; }
        .terminal { position:relative; z-index:1; padding:12px 16px; max-width:1800px; margin:0 auto; }

        /* Top Bar */
        .top-bar { display:flex; justify-content:space-between; align-items:center; padding:10px 20px; margin-bottom:12px; background:var(--bg-card); border:1px solid var(--border); border-radius:10px; }
        .top-bar h1 { font-size:1.15rem; font-weight:700; background:var(--gradient-brand); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
        .sys-info { display:flex; align-items:center; gap:16px; font-size:0.75rem; color:var(--text-secondary); }
        .status-dot { width:8px; height:8px; border-radius:50%; background:var(--accent-green); animation:pulse 2s infinite; }
        @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,230,118,0.4)} 50%{box-shadow:0 0 0 6px rgba(0,230,118,0)} }

        /* KPI Strip */
        .kpi-strip { display:grid; grid-template-columns:repeat(7,1fr); gap:10px; margin-bottom:12px; }
        .kpi { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; position:relative; overflow:hidden; transition:all 0.25s ease; }
        .kpi:hover { border-color:var(--border-active); transform:translateY(-1px); box-shadow:var(--shadow-md); }
        .kpi .accent-bar { position:absolute; top:0; left:0; right:0; height:2px; border-radius:10px 10px 0 0; }
        .kpi-label { font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-muted); margin-bottom:6px; }
        .kpi-value { font-size:1.35rem; font-weight:700; font-family:'JetBrains Mono',monospace; line-height:1.2; transition:color 0.3s; }
        .kpi-sub { font-size:0.65rem; color:var(--text-secondary); margin-top:3px; font-family:'JetBrains Mono',monospace; }
        .kpi-spark { height:24px; margin-top:6px; width:100%; }
        .positive { color:var(--accent-green); }
        .negative { color:var(--accent-red); }
        .neutral { color:var(--accent-cyan); }

        /* Regime Badge */
        .regime-badge { display:inline-block; padding:3px 10px; border-radius:6px; font-size:0.7rem; font-weight:700; font-family:'JetBrains Mono',monospace; letter-spacing:0.05em; }
        .regime-NEUTRAL { background:rgba(77,139,245,0.15); color:var(--accent-blue); border:1px solid rgba(77,139,245,0.3); }
        .regime-AGGRESSIVE { background:rgba(0,230,118,0.15); color:var(--accent-green); border:1px solid rgba(0,230,118,0.3); animation:regPulse 2s infinite; }
        .regime-DEFENSIVE { background:rgba(255,61,87,0.15); color:var(--accent-red); border:1px solid rgba(255,61,87,0.3); animation:regPulse 2s infinite; }
        .regime-CONSERVATIVE { background:rgba(255,171,0,0.15); color:var(--accent-amber); border:1px solid rgba(255,171,0,0.3); }
        @keyframes regPulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

        /* Chart */
        .chart-container { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; margin-bottom:12px; overflow:hidden; }
        .chart-toolbar { display:flex; justify-content:space-between; align-items:center; padding:8px 16px; border-bottom:1px solid var(--border); }
        .chart-tabs { display:flex; gap:4px; }
        .chart-tab { padding:5px 14px; border-radius:6px; font-size:0.75rem; font-weight:600; cursor:pointer; border:1px solid transparent; background:transparent; color:var(--text-muted); font-family:'JetBrains Mono',monospace; transition:all 0.2s; }
        .chart-tab:hover { color:var(--text-secondary); background:var(--glass); }
        .chart-tab.active { color:var(--accent-cyan); background:rgba(0,212,170,0.08); border-color:rgba(0,212,170,0.2); }
        .chart-info { font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; }
        #chartArea { height:380px; }

        /* Bottom Panels */
        .panels { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
        .panel { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
        .panel-head { display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border-bottom:1px solid var(--border); }
        .panel-title { font-size:0.8rem; font-weight:600; }
        .panel-badge { font-size:0.65rem; padding:2px 8px; border-radius:12px; font-weight:600; background:rgba(77,139,245,0.12); color:var(--accent-blue); }
        .panel-body { padding:8px 12px; max-height:280px; overflow-y:auto; }
        .panel-body::-webkit-scrollbar { width:4px; }
        .panel-body::-webkit-scrollbar-track { background:transparent; }
        .panel-body::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

        /* Feed Items */
        .anomaly-row { display:flex; gap:8px; padding:8px 10px; margin-bottom:4px; border-radius:6px; background:var(--glass); border-left:3px solid; animation:fadeIn 0.3s ease; font-size:0.75rem; }
        .anomaly-row.severity-MEDIUM { border-left-color:var(--accent-amber); }
        .anomaly-row.severity-HIGH { border-left-color:var(--accent-red); }
        .anomaly-row.severity-CRITICAL { border-left-color:var(--accent-red); animation:critP 1s infinite; }
        .anomaly-row.severity-LOW { border-left-color:var(--accent-blue); }
        @keyframes critP { 0%,100%{box-shadow:0 0 4px rgba(255,61,87,0.3)} 50%{box-shadow:0 0 12px rgba(255,61,87,0.6)} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }
        .anomaly-meta { flex:1; }
        .anomaly-sym { font-weight:600; color:var(--accent-cyan); }
        .anomaly-type { font-size:0.65rem; color:var(--text-muted); margin-left:6px; }
        .anomaly-nums { font-family:'JetBrains Mono',monospace; color:var(--text-secondary); font-size:0.7rem; }

        .trade-row { display:flex; justify-content:space-between; align-items:center; padding:8px 10px; margin-bottom:4px; border-radius:6px; background:var(--glass); font-size:0.75rem; animation:fadeIn 0.3s ease; }
        .trade-row.LONG { border-left:3px solid var(--accent-green); }
        .trade-row.SHORT { border-left:3px solid var(--accent-red); }
        .trade-pnl { font-family:'JetBrains Mono',monospace; font-weight:600; }

        .risk-row { padding:8px 10px; margin-bottom:4px; border-radius:6px; background:var(--glass); font-size:0.7rem; animation:fadeIn 0.3s ease; border-left:3px solid var(--accent-purple); font-family:'JetBrains Mono',monospace; color:var(--text-secondary); }
        .risk-row .regime-tag { font-weight:700; }

        .feed-empty { color:var(--text-muted); text-align:center; padding:30px; font-size:0.8rem; }

        @media (max-width:1100px) { .kpi-strip{grid-template-columns:repeat(4,1fr)} .panels{grid-template-columns:1fr} }
        @media (max-width:700px) { .kpi-strip{grid-template-columns:repeat(2,1fr)} }
    </style>
</head>
<body>
<div class="terminal">
    <div class="top-bar">
        <h1>&#9889; Market Anomaly Engine</h1>
        <div class="sys-info">
            <div style="display:flex;align-items:center;gap:6px">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting&hellip;</span>
            </div>
            <span id="tickRate">0 ticks/s</span>
            <span id="clockDisplay">--:--:--</span>
        </div>
    </div>

    <div class="kpi-strip">
        <div class="kpi">
            <div class="accent-bar" style="background:var(--gradient-brand)"></div>
            <div class="kpi-label">Net P&amp;L</div>
            <div class="kpi-value neutral" id="kpiPnl">&#8377;0</div>
            <div class="kpi-sub" id="kpiReturn">0.00% return</div>
            <canvas class="kpi-spark" id="sparkPnl"></canvas>
        </div>
        <div class="kpi">
            <div class="accent-bar" style="background:var(--gradient-win)"></div>
            <div class="kpi-label">Win Rate</div>
            <div class="kpi-value neutral" id="kpiWinRate">0%</div>
            <div class="kpi-sub" id="kpiTrades">0 trades</div>
            <canvas class="kpi-spark" id="sparkWinRate"></canvas>
        </div>
        <div class="kpi">
            <div class="accent-bar" style="background:linear-gradient(135deg,#a78bfa,#4d8bf5)"></div>
            <div class="kpi-label">Anomalies</div>
            <div class="kpi-value neutral" id="kpiAnomalies">0</div>
            <div class="kpi-sub" id="kpiSeverity">&mdash;</div>
        </div>
        <div class="kpi">
            <div class="accent-bar" style="background:linear-gradient(135deg,#00d4aa,#4d8bf5)"></div>
            <div class="kpi-label">Ticks</div>
            <div class="kpi-value neutral" id="kpiTicks">0</div>
            <div class="kpi-sub" id="kpiLatency">&mdash; &mu;s latency</div>
        </div>
        <div class="kpi">
            <div class="accent-bar" style="background:var(--gradient-brand)"></div>
            <div class="kpi-label">Capital</div>
            <div class="kpi-value neutral" id="kpiCapital">&#8377;10,00,000</div>
            <div class="kpi-sub" id="kpiDrawdown">0% max DD</div>
        </div>
        <div class="kpi">
            <div class="accent-bar" style="background:linear-gradient(135deg,#ffab00,#ff6d00)"></div>
            <div class="kpi-label">Sharpe Ratio</div>
            <div class="kpi-value neutral" id="kpiSharpe">0.00</div>
            <div class="kpi-sub">risk-adjusted</div>
        </div>
        <div class="kpi" id="riskRegimeCard">
            <div class="accent-bar" id="regimeAccent" style="background:var(--gradient-brand)"></div>
            <div class="kpi-label">Risk Regime</div>
            <div id="regimeBadge" class="regime-badge regime-NEUTRAL">NEUTRAL</div>
            <div class="kpi-sub" id="regimeDetail">Kelly: 5.0% | Z&times;1.00</div>
            <canvas class="kpi-spark" id="sparkRisk"></canvas>
        </div>
    </div>

    <div class="chart-container">
        <div class="chart-toolbar">
            <div class="chart-tabs" id="chartTabs"></div>
            <div class="chart-info">
                <span id="chartPrice">&mdash;</span> &nbsp;|&nbsp;
                <span id="chartPredicted" style="color:var(--accent-purple)">Pred: &mdash;</span>
            </div>
        </div>
        <div id="chartArea"></div>
    </div>

    <div class="panels">
        <div class="panel">
            <div class="panel-head">
                <span class="panel-title">&#128680; Anomaly Feed</span>
                <span class="panel-badge" id="anomalyCount">0</span>
            </div>
            <div class="panel-body" id="anomalyFeed">
                <div class="feed-empty">Waiting for anomalies&hellip;</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-head">
                <span class="panel-title">&#128176; Trade Log</span>
                <span class="panel-badge" id="tradeCount">0</span>
            </div>
            <div class="panel-body" id="tradeFeed">
                <div class="feed-empty">No trades yet</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-head">
                <span class="panel-title">&#129504; Risk Adjustments</span>
                <span class="panel-badge" id="riskCount">0</span>
            </div>
            <div class="panel-body" id="riskFeed">
                <div class="feed-empty">Awaiting feedback loop&hellip;</div>
            </div>
        </div>
    </div>
</div>

<script>
// ── State ──
const S = {
    tickCount:0, anomalyCount:0, tradeCount:0, riskCount:0,
    lastTickTime:Date.now(), ticksPerSecond:0,
    seenAnomalies:new Set(), seenTrades:new Set(),
    symbols:[], activeSym:null,
    pnlH:[], wrH:[]
};
const MAX_FEED=40, MAX_CHART=500;

// Clock
setInterval(()=>{document.getElementById('clockDisplay').textContent=new Date().toLocaleTimeString()},1000);

// ── TradingView Chart ──
const chartEl = document.getElementById('chartArea');
const chart = LightweightCharts.createChart(chartEl, {
    width: chartEl.offsetWidth, height: 380,
    layout: { background:{type:'solid',color:'#111520'}, textColor:'#8b92a5', fontSize:11, fontFamily:"'JetBrains Mono',monospace" },
    grid: { vertLines:{color:'rgba(30,39,64,0.5)'}, horzLines:{color:'rgba(30,39,64,0.5)'} },
    crosshair: { mode:LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor:'#1e2740' },
    timeScale: { borderColor:'#1e2740', timeVisible:true, secondsVisible:true, tickMarkFormatter:(t)=>{const d=new Date(t*1000);return d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});} },
    localization: { timeFormatter:(t)=>{const d=new Date(t*1000);return d.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});} },
});
window.addEventListener('resize', ()=>chart.applyOptions({width:chartEl.offsetWidth}));

const seriesMap = {};

function ensureSym(sym) {
    if (seriesMap[sym]) return;
    const cs = chart.addCandlestickSeries({
        upColor:'#00e676', downColor:'#ff3d57',
        borderDownColor:'#ff3d57', borderUpColor:'#00e676',
        wickDownColor:'#ff3d57', wickUpColor:'#00e676',
    });
    const pl = chart.addLineSeries({
        color:'rgba(167,139,250,0.6)', lineWidth:1,
        lineStyle:LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible:false, lastValueVisible:false, priceLineVisible:false,
    });
    seriesMap[sym] = {candle:cs, pred:pl, markers:[], data:[], predData:[]};
    if (!S.symbols.includes(sym)) { S.symbols.push(sym); renderTabs(); }
    if (!S.activeSym) switchSym(sym);
}

function switchSym(sym) {
    S.activeSym = sym;
    Object.entries(seriesMap).forEach(([s,v])=>{
        const vis = s===sym;
        v.candle.applyOptions({visible:vis});
        v.pred.applyOptions({visible:vis});
    });
    renderTabs();
    chart.timeScale().fitContent();
}

function renderTabs() {
    document.getElementById('chartTabs').innerHTML = S.symbols.map(s=>
        '<div class="chart-tab'+(s===S.activeSym?' active':'')+'" onclick="switchSym(\''+s+'\')">'+s.replace('.NS','')+'</div>'
    ).join('');
}

// Time conversion: parse ISO string to UTC epoch seconds for TradingView.
// TradingView Lightweight Charts expects UTC epoch seconds.
// Per-symbol tick counter ensures strictly monotonic time (prevents chart crashes).
const symTickCounter = {};
function toT(ts) {
    // Parse ISO timestamp string to epoch seconds
    const d = new Date(ts);
    if (!isNaN(d.getTime())) {
        return Math.floor(d.getTime() / 1000);
    }
    // Fallback: use current time
    return Math.floor(Date.now() / 1000);
}

// Ensure monotonically increasing time per symbol
function getMonoTime(sym, ts) {
    const t = toT(ts);
    if (!symTickCounter[sym] || t > symTickCounter[sym]) {
        symTickCounter[sym] = t;
    } else {
        symTickCounter[sym] = symTickCounter[sym] + 1;
    }
    return symTickCounter[sym];
}

// Format time for feed display
function fmtTime(ts) {
    try {
        const d = new Date(ts);
        return d.toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
    } catch(e) { return ''; }
}

// ── Sparkline ──
function spark(id, data, color) {
    const c = document.getElementById(id);
    if (!c || data.length<2) return;
    const ctx=c.getContext('2d');
    const w=c.width=c.offsetWidth*2, h=c.height=c.offsetHeight*2;
    ctx.clearRect(0,0,w,h);
    const mn=Math.min(...data), mx=Math.max(...data), rng=mx-mn||1, step=w/(data.length-1);
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=2;
    data.forEach((v,i)=>{
        const x=i*step, y=h-((v-mn)/rng)*(h-4)-2;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.lineTo((data.length-1)*step,h); ctx.lineTo(0,h); ctx.closePath();
    const g=ctx.createLinearGradient(0,0,0,h);
    g.addColorStop(0,color.replace(')',',0.15)').replace('rgb','rgba'));
    g.addColorStop(1,'transparent');
    ctx.fillStyle=g; ctx.fill();
}

// ── SSE ──
const es = new EventSource('/api/stream');
es.onopen=()=>{document.getElementById('statusText').textContent='Connected';document.getElementById('statusDot').style.background='var(--accent-green)';};
es.onerror=()=>{document.getElementById('statusText').textContent='Reconnecting\u2026';document.getElementById('statusDot').style.background='var(--accent-amber)';};
es.onmessage=(e)=>{try{const m=JSON.parse(e.data);if(m.type&&m.type!=='heartbeat')route(m);}catch(err){}};

function route(m){
    switch(m.type){
        case 'tick':onTick(m.data);break;
        case 'anomaly':onAnomaly(m.data);break;
        case 'trade':onTrade(m.data);break;
        case 'metrics':onMetrics(m.data);break;
        case 'risk_state':onRisk(m.data);break;
    }
}

function onTick(d){
    S.tickCount++;
    const now=Date.now();
    if(now-S.lastTickTime>=1000){S.ticksPerSecond=S.tickCount;S.tickCount=0;S.lastTickTime=now;document.getElementById('tickRate').textContent=S.ticksPerSecond+' ticks/s';}
    const sym=d.symbol; if(!sym)return;
    ensureSym(sym);
    const s=seriesMap[sym], t=getMonoTime(sym, d.timestamp);
    const candle={time:t, open:d.open, high:d.high, low:d.low, close:d.close};
    s.data.push(candle); if(s.data.length>MAX_CHART)s.data.shift();
    s.candle.update(candle);
    if(d.predicted_close){
        const p={time:t,value:d.predicted_close};
        s.predData.push(p); if(s.predData.length>MAX_CHART)s.predData.shift();
        s.pred.update(p);
    }
    if(sym===S.activeSym){
        document.getElementById('chartPrice').textContent=sym.replace('.NS','')+' \u20B9'+parseFloat(d.close).toFixed(2);
        if(d.predicted_close) document.getElementById('chartPredicted').textContent='Pred: \u20B9'+parseFloat(d.predicted_close).toFixed(2);
    }
}

function clrEmpty(el){const e=el.querySelector('.feed-empty');if(e)e.remove();}

function onAnomaly(d){
    const id=d.anomaly_id||'';
    if(id&&S.seenAnomalies.has(id))return;
    if(id)S.seenAnomalies.add(id);
    S.anomalyCount++;
    const sym=d.symbol;
    if(sym&&seriesMap[sym]){
        const s=seriesMap[sym], flash=d.anomaly_type==='FLASH_CRASH';
        s.markers.push({time:getMonoTime(sym, d.timestamp),position:flash?'belowBar':'aboveBar',color:flash?'#ff3d57':'#00e676',shape:flash?'arrowUp':'arrowDown',text:(d.severity||'')});
        s.candle.setMarkers(s.markers);
    }
    const feed=document.getElementById('anomalyFeed'); clrEmpty(feed);
    const row=document.createElement('div');
    row.className='anomaly-row severity-'+(d.severity||'LOW');
    const timeStr=fmtTime(d.timestamp);
    row.innerHTML='<div class="anomaly-meta"><span class="anomaly-sym">'+(d.symbol||'\u2014')+'</span><span class="anomaly-type">'+(d.anomaly_type||'')+' \u00B7 '+(d.severity||'')+(timeStr?' \u00B7 '+timeStr:'')+'</span></div><div class="anomaly-nums">Z:'+parseFloat(d.z_score||0).toFixed(2)+' | \u20B9'+parseFloat(d.actual_price||0).toFixed(0)+'\u2192\u20B9'+parseFloat(d.predicted_price||0).toFixed(0)+'</div>';
    feed.insertBefore(row,feed.firstChild);
    while(feed.children.length>MAX_FEED)feed.removeChild(feed.lastChild);
    document.getElementById('anomalyCount').textContent=S.anomalyCount;
}

function onTrade(d){
    const id=d.trade_id||'';
    if(id&&S.seenTrades.has(id))return;
    if(id)S.seenTrades.add(id);
    S.tradeCount++;
    const feed=document.getElementById('tradeFeed'); clrEmpty(feed);
    const pnl=parseFloat(d.pnl||0);
    const row=document.createElement('div');
    row.className='trade-row '+(d.direction||'');
    const timeStr=d.exit_time?fmtTime(d.exit_time):'';
    row.innerHTML='<div><strong>'+(d.direction||'')+'</strong> '+(d.symbol||'').replace('.NS','')+' \u20B9'+parseFloat(d.entry_price||0).toFixed(0)+'\u2192\u20B9'+parseFloat(d.exit_price||0).toFixed(0)+' <span style="color:var(--text-muted);font-size:0.65rem">'+(d.status||'')+(timeStr?' '+timeStr:'')+'</span></div><div class="trade-pnl '+(pnl>=0?'positive':'negative')+'">'+(pnl>=0?'+':'')+'\u20B9'+pnl.toFixed(0)+'</div>';
    feed.insertBefore(row,feed.firstChild);
    while(feed.children.length>MAX_FEED)feed.removeChild(feed.lastChild);
    document.getElementById('tradeCount').textContent=S.tradeCount;
}

function onMetrics(d){
    const pnl=parseFloat(d.net_pnl||0);
    const pe=document.getElementById('kpiPnl');
    pe.textContent='\u20B9'+pnl.toLocaleString('en-IN',{maximumFractionDigits:0});
    pe.className='kpi-value '+(pnl>0?'positive':pnl<0?'negative':'neutral');
    document.getElementById('kpiReturn').textContent=(d.total_return_pct||0).toFixed(2)+'% return';
    const wr=parseFloat(d.win_rate_pct||0);
    const we=document.getElementById('kpiWinRate');
    we.textContent=wr.toFixed(0)+'%';
    we.className='kpi-value '+(wr>=50?'positive':wr>0?'negative':'neutral');
    document.getElementById('kpiTrades').textContent=(d.total_trades||0)+' trades';
    document.getElementById('kpiCapital').textContent='\u20B9'+parseFloat(d.capital||0).toLocaleString('en-IN',{maximumFractionDigits:0});
    document.getElementById('kpiDrawdown').textContent=(d.max_drawdown_pct||0).toFixed(2)+'% max DD';
    const sh=parseFloat(d.sharpe_ratio||0);
    const se=document.getElementById('kpiSharpe');
    se.textContent=sh.toFixed(2);
    se.className='kpi-value '+(sh>1?'positive':sh<0?'negative':'neutral');
    S.pnlH.push(pnl); if(S.pnlH.length>30)S.pnlH.shift();
    spark('sparkPnl',S.pnlH,'rgb(0,212,170)');
    S.wrH.push(wr); if(S.wrH.length>30)S.wrH.shift();
    spark('sparkWinRate',S.wrH,'rgb(0,230,118)');
}

function onRisk(d){
    S.riskCount++;
    const b=document.getElementById('regimeBadge');
    b.textContent=d.regime||'NEUTRAL';
    b.className='regime-badge regime-'+(d.regime||'NEUTRAL');
    document.getElementById('regimeDetail').textContent='Kelly: '+(d.kelly_fraction_pct||5).toFixed(1)+'% | Z\u00D7'+(d.z_score_multiplier||1).toFixed(2);
    const a=document.getElementById('regimeAccent');
    if(d.regime==='DEFENSIVE')a.style.background='var(--gradient-defensive)';
    else if(d.regime==='AGGRESSIVE')a.style.background='var(--gradient-aggressive)';
    else a.style.background='var(--gradient-brand)';
    if(d.position_size_history&&d.position_size_history.length>1) spark('sparkRisk',d.position_size_history.map(v=>v*100),'rgb(167,139,250)');
    const feed=document.getElementById('riskFeed'); clrEmpty(feed);
    const row=document.createElement('div'); row.className='risk-row';
    const rc=d.regime==='DEFENSIVE'?'var(--accent-red)':d.regime==='AGGRESSIVE'?'var(--accent-green)':'var(--accent-blue)';
    row.innerHTML='<span class="regime-tag" style="color:'+rc+'">'+d.regime+'</span> WR:'+(d.win_rate||0).toFixed(0)+'% Sh:'+(d.rolling_sharpe||0).toFixed(2)+' K:'+(d.kelly_fraction_pct||0).toFixed(1)+'% Sz:'+(d.position_size_pct||0).toFixed(1)+'% Z\u00D7'+(d.z_score_multiplier||1).toFixed(2);
    feed.insertBefore(row,feed.firstChild);
    while(feed.children.length>MAX_FEED)feed.removeChild(feed.lastChild);
    document.getElementById('riskCount').textContent=S.riskCount;
}

// Polling fallback
setInterval(async()=>{
    try{const r=await fetch('/api/metrics');const d=await r.json();onMetrics(d.pnl||{});
        document.getElementById('kpiTicks').textContent=(d.pipeline?.tick_count||0).toLocaleString();
        if(d.inference?.avg_latency_us)document.getElementById('kpiLatency').textContent=d.inference.avg_latency_us.toFixed(0)+' \u03BCs latency';
        document.getElementById('kpiAnomalies').textContent=d.anomalies?.total_anomalies||S.anomalyCount;
    }catch(e){}
    try{const r=await fetch('/api/anomalies');const a=await r.json();if(Array.isArray(a))a.forEach(x=>onAnomaly(x));}catch(e){}
    try{const r=await fetch('/api/trades');const t=await r.json();if(Array.isArray(t))t.forEach(x=>onTrade(x));}catch(e){}
    try{const r=await fetch('/api/risk_state');const d=await r.json();if(d&&d.regime)onRisk(d);}catch(e){}
},2500);
</script>
</body>
</html>
"""


def create_dashboard_app(pipeline=None) -> Flask:
    """Create the Flask dashboard application."""
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/stream")
    def stream():
        """SSE endpoint for real-time event streaming.

        Each connected client gets its own queue via the broadcaster.
        When the client disconnects, its queue is removed automatically.
        """
        client_queue = _broadcaster.subscribe()

        def generate() -> Generator:
            try:
                while True:
                    try:
                        event = client_queue.get(timeout=1)
                        yield f"data: {_safe_json(event)}\n\n"
                    except queue.Empty:
                        # Send heartbeat to keep connection alive
                        yield f"data: {_safe_json({'type': 'heartbeat'})}\n\n"
            except GeneratorExit:
                pass
            finally:
                _broadcaster.unsubscribe(client_queue)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/metrics")
    def metrics():
        """Get current metrics snapshot."""
        if pipeline is None:
            return jsonify({"error": "Pipeline not connected"})

        result = {
            "pnl": pipeline.pnl_simulator.get_metrics(),
            "anomalies": pipeline.anomaly_detector.get_stats(),
            "pipeline": {
                "tick_count": pipeline._tick_count,
                "prediction_count": pipeline._prediction_count,
            },
        }

        if pipeline.inference_engine:
            result["inference"] = pipeline.inference_engine.get_stats()

        return jsonify(result)

    @app.route("/api/anomalies")
    def anomalies():
        """Get recent anomalies."""
        if pipeline is None:
            return jsonify([])
        return jsonify(pipeline.anomaly_detector.get_recent_anomalies())

    @app.route("/api/trades")
    def trades():
        """Get recent trades."""
        if pipeline is None:
            return jsonify([])
        return jsonify(pipeline.pnl_simulator.get_recent_trades())

    @app.route("/api/equity")
    def equity():
        """Get equity curve."""
        if pipeline is None:
            return jsonify([])
        return jsonify(pipeline.pnl_simulator.get_equity_curve())

    @app.route("/api/risk_state")
    def risk_state():
        """Get current adaptive risk manager state."""
        if pipeline is None:
            return jsonify({"regime": "NEUTRAL"})
        return jsonify(pipeline.risk_manager.get_state())

    return app


def start_dashboard(
    pipeline=None, host: str = None, port: int = None
) -> threading.Thread:
    """Start the dashboard in a background thread."""
    cfg = config.dashboard
    app = create_dashboard_app(pipeline)

    thread = threading.Thread(
        target=lambda: app.run(
            host=host or cfg.host,
            port=port or cfg.port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    thread.start()
    logger.info(
        f"Dashboard started at http://{host or cfg.host}:{port or cfg.port}"
    )
    return thread
