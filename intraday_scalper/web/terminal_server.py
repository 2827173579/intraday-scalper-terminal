"""Local web terminal for the SOXL / SOXS intraday signal engine.

Architecture
------------
``terminal_server.py`` owns **one** production :class:`intraday_scalper.main.Runtime`
(Alpaca IEX data -> quality gate -> 1m/5m aggregation -> strategies -> risk
manager -> signal engine), exactly like ``main live``. The only difference is
that instead of the Rich dashboard it:

* wraps the provider callbacks to keep a small display-only ``SessionTracker``
  (today's 1m/5m bars, opening-range, SMC liquidity-sweep observations, volume);
* polls ``runtime.snapshot()`` and builds the JSON state consumed by the browser;
* serves ``terminal.html`` and ``/api/state`` on ``http://127.0.0.1:8080``.

It never places orders. ``--demo`` serves a synthetic state so the page can be
verified offline (useful in tests).

Run from the repository root::

    python -m intraday_scalper.web.terminal_server --open
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from datetime import date, datetime, time as dt_time, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from intraday_scalper.core.config import PACKAGE_ROOT, load_config
from intraday_scalper.core.models import Bar, Tick

WEB_DIR = Path(__file__).resolve().parent
HTML_PATH = WEB_DIR / "terminal.html"
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
SYMBOLS = ("SOXL", "SOXS")
FORCED_FLAT_ET = dt_time(15, 55)
REGULAR_OPEN_ET = dt_time(9, 30)
REGULAR_CLOSE_ET = dt_time(16, 0)
ORB_MINUTES = 15
TREND_FAST = 9
TREND_SLOW = 21
SWEEP_LOOKBACK = 20
SIGNALS_DIR = PACKAGE_ROOT / "var" / "live_signals"


# --------------------------------------------------------------------------- #
# Small pure helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def et_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(ET)


def kst_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(KST)


def forced_flat_countdown(moment: datetime) -> Dict[str, Any]:
    """Seconds / progress from 09:30 ET to the 15:55 ET forced-flat deadline."""
    et = moment.astimezone(ET)
    session_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    deadline = et.replace(hour=FORCED_FLAT_ET.hour, minute=FORCED_FLAT_ET.minute, second=0, microsecond=0)
    total = max(1.0, (deadline - session_open).total_seconds())
    remaining = (deadline - et).total_seconds()
    elapsed = 1.0 - remaining / total
    return {
        "deadline_et": "15:55",
        "seconds_left": int(round(remaining)),
        "seconds_total": int(total),
        "fraction_elapsed": max(0.0, min(1.0, elapsed)),
        "is_past": remaining < 0,
    }


def _et_minutes(moment: datetime) -> int:
    et = moment.astimezone(ET)
    return et.hour * 60 + et.minute


def _bar_et_minutes(bar_start_utc: datetime) -> int:
    return _et_minutes(bar_start_utc.astimezone(ET))


def opening_range(bars_1m: Sequence[Dict[str, Any]], session_day: date) -> Dict[str, Any]:
    """Opening range high/low from the first ``ORB_MINUTES`` regular-session 1m bars."""
    window = [
        bar
        for bar in bars_1m
        if bar["et_day"] == session_day and 570 <= _bar_et_minutes(bar["start_utc"]) < 570 + ORB_MINUTES
    ]
    if not window:
        return {"ready": False, "high": None, "low": None, "bars": 0}
    high = max(float(bar["high"]) for bar in window)
    low = min(float(bar["low"]) for bar in window)
    return {
        "ready": len(window) >= ORB_MINUTES - 2,
        "high": round(high, 4),
        "low": round(low, 4),
        "width_pct": round((high / low - 1.0) * 100.0, 4) if low > 0 else None,
        "bars": len(window),
    }


def orb_status(price: Optional[float], orb: Dict[str, Any], last_bar: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not orb.get("ready") or price is None:
        return {"state": "waiting", "label": "等待开盘区间", "detail": "09:30–09:45 ET 数据未齐"}
    high, low = float(orb["high"]), float(orb["low"])
    buffer = max((high - low) * 0.05, 0.0005 * price)
    if price > high + buffer:
        state, label = "break_up", "向上突破"
    elif price < low - buffer:
        state, label = "break_down", "向下突破"
    else:
        state, label = "inside", "区间内震荡"
    return {
        "state": state,
        "label": label,
        "detail": f"ORB 上沿 {high:.3f} / 下沿 {low:.3f}",
        "range_position": round((price - low) / (high - low), 4) if high > low else None,
    }


def detect_liquidity_sweep(bars: Sequence[Dict[str, Any]], lookback: int = SWEEP_LOOKBACK) -> Dict[str, Any]:
    """Display-only SMC sweep observation on **completed** bars (never a trade signal).

    Bullish sweep: latest bar's low pierces the prior ``lookback`` swing low but
    closes back above it (liquidity taken below, then reclaimed). Bearish is the
    mirror. This is a radar indicator only; the production engine keeps its
    deterministic ORB / VWAP / EMA strategies (SMC is not a trading strategy).
    """
    if len(bars) < lookback + 2:
        return {"state": "none", "label": "数据不足"}
    prior = bars[:-1][-lookback:]
    last = bars[-1]
    swing_low = min(float(bar["low"]) for bar in prior)
    swing_high = max(float(bar["high"]) for bar in prior)
    if float(last["low"]) < swing_low and float(last["close"]) > swing_low:
        return {
            "state": "bullish",
            "label": "看涨刺穿反转",
            "detail": f"低点 {float(last['low']):.3f} 刺穿 {swing_low:.3f} 后收回",
        }
    if float(last["high"]) > swing_high and float(last["close"]) < swing_high:
        return {
            "state": "bearish",
            "label": "看跌刺穿反转",
            "detail": f"高点 {float(last['high']):.3f} 刺穿 {swing_high:.3f} 后回落",
        }
    return {"state": "none", "label": "无刺穿", "detail": f"20 根摆动 {swing_low:.3f} ~ {swing_high:.3f}"}


def aggregate_5m(bars_1m: Sequence[Dict[str, Any]], session_day: Optional[date] = None) -> List[Dict[str, Any]]:
    """Aggregate completed 1m bars into 09:30-anchored 5m bars (display only)."""
    buckets: Dict[datetime, Dict[str, Any]] = {}
    for bar in bars_1m:
        if session_day is not None and bar["et_day"] != session_day:
            continue
        start = bar["start_utc"]
        anchor = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
        bucket = buckets.get(anchor)
        if bucket is None:
            buckets[anchor] = {
                "start_utc": anchor,
                "et_day": bar["et_day"],
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            }
        else:
            bucket["high"] = max(bucket["high"], float(bar["high"]))
            bucket["low"] = min(bucket["low"], float(bar["low"]))
            bucket["close"] = float(bar["close"])
            bucket["volume"] += float(bar["volume"])
    result = []
    for anchor in sorted(buckets):
        bar = buckets[anchor]
        bar["end_utc"] = bar["start_utc"] + timedelta(minutes=5)
        # only bars whose 5 minutes are fully covered by data are "completed enough"
        expected = [anchor + timedelta(minutes=i) for i in range(5)]
        present = sum(1 for ts in expected if any(b["start_utc"] == ts for b in bars_1m))
        bar["complete"] = present >= 5
        result.append(bar)
    return result


def aggregate_timeframe(bars_1m: Sequence[Dict[str, Any]], minutes: int) -> List[Dict[str, Any]]:
    """Aggregate completed 1m bars into N-minute bars anchored at 09:30 ET."""
    if minutes <= 1:
        return list(bars_1m)
    buckets: Dict[datetime, Dict[str, Any]] = {}
    for bar in bars_1m:
        start = bar["start_utc"]
        et = start.astimezone(ET)
        anchor_minute = (et.hour * 60 + et.minute - 570) // minutes * minutes + 570
        anchor = et.replace(hour=anchor_minute // 60, minute=anchor_minute % 60, second=0, microsecond=0)
        key = anchor.astimezone(timezone.utc)
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = {
                "start_utc": key,
                "end_utc": key + timedelta(minutes=minutes),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
                "complete": False,
            }
        else:
            bucket["high"] = max(bucket["high"], float(bar["high"]))
            bucket["low"] = min(bucket["low"], float(bar["low"]))
            bucket["close"] = float(bar["close"])
            bucket["volume"] += float(bar["volume"])
    result = []
    for key in sorted(buckets):
        bucket = buckets[key]
        expected = [key + timedelta(minutes=index) for index in range(minutes)]
        present = sum(1 for ts in expected if any(b["start_utc"] == ts for b in bars_1m))
        bucket["complete"] = present >= minutes
        result.append(bucket)
    return result


def ema(values: Sequence[float], span: int) -> List[float]:

    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def trend_label(closes: Sequence[float]) -> str:
    if len(closes) < TREND_SLOW:
        return "flat"
    fast = ema(closes, TREND_FAST)[-1]
    slow = ema(closes, TREND_SLOW)[-1]
    if fast > slow * 1.0002:
        return "bull"
    if fast < slow * 0.9998:
        return "bear"
    return "flat"


# --------------------------------------------------------------------------- #
# Display-only session tracker fed by provider callbacks + startup backfill
# --------------------------------------------------------------------------- #
LOCAL_DATA_DIR = PACKAGE_ROOT / "data"
BACKTEST_DIR = PACKAGE_ROOT / "var" / "backtest"


def load_local_session(symbols: Sequence[str] = SYMBOLS) -> Dict[str, Any]:
    """Load the most recent session from ``data/{symbol}_1m_raw.parquet``.

    This is the fallback that guarantees the terminal can replay the last trading
    day even when the broker REST history is unavailable (weekends/holidays).
    Returns records suitable for :meth:`SessionTracker.backfill` plus chart bars.
    """
    import pandas as pd

    frames = []
    for symbol in symbols:
        path = LOCAL_DATA_DIR / f"{symbol}_1m_raw.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["symbol", "start_utc", "end_utc", "open", "high", "low", "close", "volume", "vwap", "trade_count"])
        except Exception:  # noqa: BLE001 - old schema fallback
            frame = pd.read_parquet(path)
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return {"session_date": None, "records": {}, "bars": {}}
    frame = pd.concat(frames, ignore_index=True)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["et_day"] = frame["start_utc"].dt.tz_convert(ET).dt.date
    session_date = max(frame["et_day"].unique())
    session = frame[frame["et_day"] == session_date].sort_values(["symbol", "start_utc"])
    records: Dict[str, List[Any]] = {}
    bars: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, group in session.groupby("symbol"):
        group = group.sort_values("start_utc")
        records[symbol] = [
            SimpleNamespace(
                timestamp=row.start_utc.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                vwap=float(row.vwap),
                trade_count=int(row.trade_count),
            )
            for row in group.itertuples(index=False)
        ]
        bars[symbol] = [
            {
                "t": int(row.start_utc.timestamp() * 1000),
                "o": round(float(row.open), 4),
                "h": round(float(row.high), 4),
                "l": round(float(row.low), 4),
                "c": round(float(row.close), 4),
                "v": int(round(float(row.volume))),
            }
            for row in group.itertuples(index=False)
        ]
    return {"session_date": str(session_date), "records": records, "bars": bars}


def derive_prev_closes_local(symbols: Sequence[str] = SYMBOLS) -> Dict[str, float]:
    """Fallback previous-session closes from the local raw parquet files."""
    import pandas as pd

    out: Dict[str, float] = {}
    for symbol in symbols:
        path = LOCAL_DATA_DIR / f"{symbol}_1m_raw.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path, columns=["start_utc", "close"])
        except Exception:  # noqa: BLE001
            continue
        frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
        daily = frame.groupby(frame["start_utc"].dt.tz_convert(ET).dt.date)["close"].last().sort_index()
        if len(daily) >= 2:
            out[symbol] = float(daily.iloc[-2])
    return out


def load_local_history_bars(max_sessions: int = 30) -> List[Bar]:
    """Load the last N completed sessions from the local raw parquet files as Bar objects.

    Used for the continuous 5m EMA/ATR warm-up required by ``evaluate_v04_session``.
    """
    import pandas as pd

    frames = []
    for symbol in SYMBOLS:
        path = LOCAL_DATA_DIR / f"{symbol}_1m_raw.parquet"
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(
                path,
                columns=["symbol", "start_utc", "end_utc", "open", "high", "low", "close", "volume", "vwap", "trade_count"],
            )
        except Exception:  # noqa: BLE001
            frame = pd.read_parquet(path)
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return []
    frame = pd.concat(frames, ignore_index=True)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["end_utc"] = pd.to_datetime(frame["end_utc"], utc=True)
    frame["et_day"] = frame["start_utc"].dt.tz_convert(ET).dt.date
    sessions = sorted(frame["et_day"].unique())[-max_sessions:]
    today = et_now().date()
    sessions = [day for day in sessions if day < today]
    frame = frame[frame["et_day"].isin(sessions)].sort_values(["start_utc", "symbol"])
    bars: List[Bar] = []
    for row in frame.itertuples(index=False):
        bars.append(
            Bar(
                symbol=str(row.symbol),
                timeframe_minutes=1,
                start_utc=row.start_utc.to_pydatetime(),
                end_utc=row.end_utc.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                vwap=float(row.vwap),
                trade_count=int(row.trade_count),
            )
        )
    return bars


def replay_payload() -> Dict[str, Any]:
    """Last-session bars + V0.4 recap/markers for the Friday replay UI."""
    local = load_local_session()
    markers: Any = []
    recap: Dict[str, Any] = {}
    marker_path = BACKTEST_DIR / "v04_friday_markers.json"
    recap_path = BACKTEST_DIR / "v04_friday_recap.json"
    try:
        if marker_path.exists():
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            markers = payload.get("markers", payload) if isinstance(payload, dict) else payload
    except Exception:  # noqa: BLE001
        markers = []
    try:
        if recap_path.exists():
            recap = json.loads(recap_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        recap = {}
    # V0.4 default = Step3c-A NoCap (SMC require BOTH; no 8% cap) — best WF net / PF / DD.
    compare: List[Dict[str, Any]] = []
    v04c_recap_path = BACKTEST_DIR / "v04c_friday_recap.json"
    v04c_marker_path = BACKTEST_DIR / "v04c_friday_markers.json"
    v04c_report_path = BACKTEST_DIR / "v04c_report.json"
    try:
        if v04c_recap_path.exists():
            v04c_recap = json.loads(v04c_recap_path.read_text(encoding="utf-8"))
            selected = (v04c_recap.get("versions") or {}).get("NoCap") or {}
            if isinstance(recap, dict) and selected:
                recap["versions"] = {
                    "V0.1": (recap.get("versions") or {}).get("V0.1"),
                    "V0.3": (recap.get("versions") or {}).get("V0.3"),
                    "V0.4": {
                        "version": "V0.4 (Step3c-A NoCap)",
                        "engine": "Step3c-A: window + VWAP trend + SMC AND (EMA21 & VWAP) — no 8% cap",
                        "summary": selected.get("summary", {}),
                        "trades": selected.get("trades", []),
                    },
                }
        if v04c_marker_path.exists():
            payload = json.loads(v04c_marker_path.read_text(encoding="utf-8"))
            by_variant = payload.get("markers_by_variant") if isinstance(payload, dict) else None
            if isinstance(by_variant, dict) and "NoCap" in by_variant:
                markers = by_variant["NoCap"]
            elif isinstance(payload, dict):
                markers = payload.get("markers", payload)
        if v04c_report_path.exists():
            report = json.loads(v04c_report_path.read_text(encoding="utf-8"))
            for name, item in (report.get("main_table") or {}).items():
                full = item.get("full") or {}
                holdout = item.get("holdout") or {}
                walkforward = item.get("walkforward") or {}
                full_metrics = full.get("metrics") or {}
                holdout_metrics = holdout.get("metrics") or {}
                wf_metrics = walkforward.get("aggregate_test") or {}
                wf_summary = walkforward.get("summary") or {}
                compare.append(
                    {
                        "name": name,
                        "trades": full_metrics.get("total_trades"),
                        "pf": full_metrics.get("profit_factor"),
                        "net": full_metrics.get("net_pnl_dollars"),
                        "holdout_net": holdout_metrics.get("net_pnl_dollars"),
                        "holdout_pf": holdout_metrics.get("profit_factor"),
                        "wf_net": wf_metrics.get("net_pnl_dollars"),
                        "wf_pf": wf_metrics.get("profit_factor"),
                        "wf_folds": wf_summary.get("positive_folds"),
                    }
                )
    except Exception:  # noqa: BLE001
        pass
    # B1 true exit structure + 3-month period stats (no watered-down win rate).
    exit_structure: List[Dict[str, Any]] = []
    period_b1: Dict[str, Any] = {}
    try:
        exit_path = BACKTEST_DIR / "exit_structure_10k.json"
        if exit_path.exists():
            exit_data = json.loads(exit_path.read_text(encoding="utf-8"))
            b1 = exit_data.get("B1") or {}
            exit_structure = b1.get("exit_structure") or []
            period_b1 = {
                "label": "B1 · 2026-08-01~11-06 ($10,000)",
                "trades": b1.get("trades"),
                "win_rate_pct": b1.get("win_rate_pct"),
                "target_hit_rate_pct": (exit_data.get("original_68") or {}).get("target_hit_rate_pct"),
                "profit_factor": b1.get("profit_factor"),
                "net": b1.get("net"),
                "net_pct": b1.get("net_pct"),
                "final_equity": b1.get("final_equity"),
                "max_drawdown": b1.get("max_drawdown"),
                "months": b1.get("months"),
            }
    except Exception:  # noqa: BLE001
        pass
    ablation: List[Dict[str, Any]] = []
    warning = ""
    report_path = BACKTEST_DIR / "v04_report.json"
    try:
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for step in report.get("steps", []):
                full = step.get("full", {}).get("metrics", {})
                wf = step.get("walkforward", {}).get("aggregate_test", {})
                ablation.append(
                    {
                        "step": step.get("step"),
                        "name": step.get("name"),
                        "trades": full.get("total_trades"),
                        "pf": full.get("profit_factor"),
                        "net": full.get("net_pnl_dollars"),
                        "wf_trades": wf.get("total_trades"),
                        "wf_pf": wf.get("profit_factor"),
                        "wf_net": wf.get("net_pnl_dollars"),
                        "wf_folds": step.get("walkforward", {}).get("positive_folds"),
                    }
                )
            warning = (
                "当前默认 = V0.4 Step3c-A NoCap（SMC 必须同时站上 EMA21 和 VWAP；已移除 8% 防追高）。"
                "A/B 结果：NoCap 全周期 +$64.57 / PF 1.009 / MaxDD $520、holdout +$388.18 / PF 1.086、"
                "WF OOS -$131.65 / PF 0.978 / 16 折（三者中 WF 最优）；WithCap 全周期 +$13.55、WF -$181.78 / 17 折、"
                "周五 0 笔；Step3b 全周期 +$76.32、holdout +$566.85，但 WF -$210.04 / 15 折。"
                "周五：AND 条件已挡掉 11:25 SOXL SMC 接刀；12:40 SOXS EMA 山顶追高在 NoCap 下恢复（-$21.63）。"
                "仅观察、不实盘。"
            )
    except Exception:  # noqa: BLE001
        pass
    return {
        "session_date": local.get("session_date"),
        "bars": local.get("bars", {}),
        "markers": markers,
        "recap": recap,
        "ablation": ablation,
        "compare": compare,
        "exit_structure": exit_structure,
        "period_b1": period_b1,
        "warning": warning,
    }


@dataclass
class SymbolBook:
    symbol: str
    bars_1m: List[Dict[str, Any]] = field(default_factory=list)
    last_price: Optional[float] = None
    last_bar_utc: Optional[datetime] = None
    day_volume: float = 0.0
    session_open: Optional[float] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None


class SessionTracker:
    """Keeps today's display data. Read paths are lock-protected."""

    def __init__(self, symbols: Sequence[str] = SYMBOLS) -> None:
        self._lock = threading.RLock()
        self.books: Dict[str, SymbolBook] = {symbol: SymbolBook(symbol) for symbol in symbols}
        self.daily_bars: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in symbols}
        self.current_day: date = et_now().date()

    # -- ingestion ---------------------------------------------------- #
    def on_bar(self, bar: Bar) -> None:
        with self._lock:
            book = self.books.get(bar.symbol)
            if book is None:
                return
            et_day = bar.start_utc.astimezone(ET).date()
            if et_day > self.current_day:
                self.current_day = et_day
                for other in self.books.values():
                    other.bars_1m = []
                    other.day_volume = 0.0
                    other.session_open = None
                    other.session_high = None
                    other.session_low = None
            elif et_day < self.current_day:
                return
            if any(existing["start_utc"] == bar.start_utc for existing in book.bars_1m[-3:]):
                return
            record = {
                "start_utc": bar.start_utc,
                "end_utc": bar.end_utc,
                "et_day": et_day,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "vwap": float(bar.vwap or bar.close),
            }
            self._insert(book, record)
            book.last_bar_utc = bar.start_utc
            book.last_price = float(bar.close)
            self._update_day_extremes(book, record)
            if book.session_open is None:
                book.session_open = float(record["open"])

    def on_tick(self, tick: Tick) -> None:
        with self._lock:
            book = self.books.get(tick.symbol)
            if book is None:
                return
            price = float(tick.price)
            if price <= 0:
                return
            book.last_price = price
            if book.session_open is None:
                book.session_open = price
            if book.session_high is None or price > book.session_high:
                book.session_high = price
            if book.session_low is None or price < book.session_low:
                book.session_low = price

    def backfill(self, bars_by_symbol: Dict[str, Iterable[Any]]) -> int:
        """Insert startup REST 1m bars for the **latest available session**.

        On weekends/holidays the latest session is the previous trading day, so the
        terminal still shows the last completed session instead of an empty chart.
        """
        added = 0
        with self._lock:
            latest_day = None
            for symbol, bars in bars_by_symbol.items():
                for bar in bars:
                    start = getattr(bar, "timestamp", None)
                    if start is None:
                        continue
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    day = start.astimezone(ET).date()
                    if latest_day is None or day > latest_day:
                        latest_day = day
            self.current_day = latest_day or self.current_day
            for book in self.books.values():
                book.bars_1m = []
                book.day_volume = 0.0
                book.session_open = None
                book.session_high = None
                book.session_low = None
            for symbol, bars in bars_by_symbol.items():
                book = self.books.get(symbol)
                if book is None:
                    continue
                for bar in bars:
                    start = getattr(bar, "timestamp", None)
                    if start is None:
                        continue
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    if start.astimezone(ET).date() != self.current_day:
                        continue
                    record = {
                        "start_utc": start,
                        "end_utc": start + timedelta(minutes=1),
                        "et_day": start.astimezone(ET).date(),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "vwap": float(getattr(bar, "vwap", None) or bar.close),
                    }
                    if any(existing["start_utc"] == start for existing in book.bars_1m):
                        continue
                    self._insert(book, record)
                    self._update_day_extremes(book, record)
                    book.last_bar_utc = record["start_utc"]
                    book.last_price = float(record["close"])
                    added += 1
        return added

    def backfill_daily(self, bars_by_symbol: Dict[str, Iterable[Any]]) -> int:
        """Store recent daily bars from the startup REST call (for the 日K view)."""
        added = 0
        with self._lock:
            for symbol, bars in bars_by_symbol.items():
                if symbol not in self.daily_bars:
                    continue
                records = []
                for bar in bars:
                    start = getattr(bar, "timestamp", None)
                    if start is None:
                        continue
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    records.append(
                        {
                            "start_utc": start.astimezone(timezone.utc),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        }
                    )
                records.sort(key=lambda item: item["start_utc"])
                self.daily_bars[symbol] = records[-160:]
                added += len(records)
        return added

    def candles(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Return display candles: 1m / 5m / 15m from today's bars, 1d from REST history."""
        with self._lock:
            if timeframe == "1d":
                daily = list(self.daily_bars.get(symbol, []))
                if not daily:
                    return []
                today = et_now().date()
                if daily[-1]["start_utc"].astimezone(ET).date() < today:
                    book = self.books.get(symbol)
                    if book and book.session_open is not None:
                        session_bars = [bar for bar in book.bars_1m if bar["et_day"] == today]
                        if session_bars:
                            daily.append(
                                {
                                    "start_utc": datetime.combine(today, dt_time(9, 30), tzinfo=ET).astimezone(timezone.utc),
                                    "open": float(session_bars[0]["open"]),
                                    "high": max(float(bar["high"]) for bar in session_bars),
                                    "low": min(float(bar["low"]) for bar in session_bars),
                                    "close": float(session_bars[-1]["close"]),
                                    "volume": sum(float(bar["volume"]) for bar in session_bars),
                                }
                            )
                return [
                    {
                        "t": int(bar["start_utc"].timestamp() * 1000),
                        "o": round(bar["open"], 4),
                        "h": round(bar["high"], 4),
                        "l": round(bar["low"], 4),
                        "c": round(bar["close"], 4),
                        "v": int(round(bar["volume"])),
                    }
                    for bar in daily
                ]

            book = self.books.get(symbol)
            if book is None:
                return []
            today = self.current_day
            bars = [bar for bar in book.bars_1m if bar["et_day"] == today]
            minutes = {"1m": 1, "5m": 5, "15m": 15}.get(timeframe, 1)
            frames = bars if minutes == 1 else aggregate_timeframe(bars, minutes)
            frames = [bar for bar in frames if bar.get("complete", True)]
            return [
                {
                    "t": int(bar["start_utc"].timestamp() * 1000),
                    "o": round(bar["open"], 4),
                    "h": round(bar["high"], 4),
                    "l": round(bar["low"], 4),
                    "c": round(bar["close"], 4),
                    "v": int(round(bar["volume"])),
                }
                for bar in frames
            ]

    def _insert(self, book: SymbolBook, record: Dict[str, Any]) -> None:
        book.bars_1m.append(record)
        book.bars_1m.sort(key=lambda item: item["start_utc"])
        if len(book.bars_1m) > 4000:
            book.bars_1m = book.bars_1m[-4000:]

    @staticmethod
    def _update_day_extremes(book: SymbolBook, record: Dict[str, Any]) -> None:
        if book.session_open is None:
            book.session_open = float(record["open"])
        book.session_high = max(book.session_high or float(record["high"]), float(record["high"]))
        book.session_low = min(book.session_low or float(record["low"]), float(record["low"]))
        book.day_volume += float(record["volume"])

    # -- reads -------------------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = et_now()
            day = self.current_day
            data: Dict[str, Any] = {}
            for symbol, book in self.books.items():
                bars = [bar for bar in book.bars_1m if bar["et_day"] == day]
                closes = [bar["close"] for bar in bars]
                bars_5m = [bar for bar in aggregate_5m(bars, day) if bar.get("complete")]
                orb = opening_range(bars, day)
                data[symbol] = {
                    "symbol": symbol,
                    "last_price": round(book.last_price if book.last_price is not None else (bars[-1]["close"] if bars else 0.0), 4) if (book.last_price is not None or bars) else None,
                    "last_bar_utc": book.last_bar_utc.isoformat() if book.last_bar_utc else None,
                    "session_open": round(book.session_open, 4) if book.session_open else None,
                    "session_high": round(book.session_high, 4) if book.session_high else None,
                    "session_low": round(book.session_low, 4) if book.session_low else None,
                    "volume": int(round(book.day_volume)),
                    "bars_1m": len(bars),
                    "bars_5m": len(bars_5m),
                    "orb": orb,
                    "orb_status": orb_status(book.last_price, orb, bars[-1] if bars else None),
                    "smc_1m": detect_liquidity_sweep(bars),
                    "smc_5m": detect_liquidity_sweep(bars_5m),
                    "trend_1m": trend_label(closes),
                    "trend_5m": trend_label([bar["close"] for bar in bars_5m]),
                    "vwap": round(sum(bar["vwap"] * bar["volume"] for bar in bars) / sum(bar["volume"] for bar in bars), 4)
                    if bars and sum(bar["volume"] for bar in bars) > 0
                    else None,
                }
            return {"as_of_utc": datetime.now(timezone.utc).isoformat(), "symbols": data}

    def bars_for(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        with self._lock:
            book = self.books.get(symbol)
            if book is None:
                return []
            day = et_now().date()
            bars = [bar for bar in book.bars_1m if bar["et_day"] == day]
            if timeframe == "5m":
                return [bar for bar in aggregate_5m(bars, day) if bar.get("complete")]
            return bars


# --------------------------------------------------------------------------- #
# State builder shared by live + demo
# --------------------------------------------------------------------------- #
def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def b1_config() -> Any:
    """B1 production risk profile (3-month $10k validation, 2026-11-08).

    $10,000 equity / $60 fixed risk per trade / $10,000 notional cap /
    3 consecutive stops -> 15 min cooldown / -$320 daily breaker.
    This is the default position calculator used by the live terminal and demos.
    """
    config = load_config()
    risk = replace(
        config.risk,
        account_equity=10_000.0,
        max_risk_per_trade=60.0,
        max_notional_per_trade=10_000.0,
        allow_fractional_shares=True,
        fractional_precision=2,
        consecutive_stop_threshold=3,
        consecutive_stop_cooldown_minutes=15,
        max_daily_net_loss=320.0,
    )
    return replace(config, risk=risk)


def _event_dict(event: Any) -> Dict[str, Any]:
    payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    return payload


def build_execution_card(snapshot_dict: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    signal = snapshot_dict.get("latest_signal") or {}
    plan = snapshot_dict.get("latest_plan") or {}
    position = snapshot_dict.get("position") or {}
    candidate = signal if isinstance(signal, dict) else {}
    if not candidate and isinstance(plan, dict):
        candidate = plan.get("candidate") or {}

    entry = _safe_float(candidate.get("entry_ref"))
    stop = _safe_float(candidate.get("stop_loss"))
    target_r = _safe_float(candidate.get("target_r")) or 2.0
    shares = _safe_float(plan.get("shares")) if isinstance(plan, dict) else None
    actual_risk = _safe_float(plan.get("actual_risk")) if isinstance(plan, dict) else None
    notional = _safe_float(plan.get("notional")) if isinstance(plan, dict) else None
    target = round(entry + target_r * (entry - stop), 4) if entry is not None and stop is not None else None
    risk_per_share = round(entry - stop, 4) if entry is not None and stop is not None else None

    be_trigger = round(entry + 1.2 * (entry - stop), 4) if entry is not None and stop is not None else None
    be_stop = round(entry + 0.05 * (entry - stop), 4) if entry is not None and stop is not None else None
    position_r = _safe_float(position.get("unrealized_r")) if isinstance(position, dict) else None
    if position:
        be_state = "active" if (position_r is not None and position_r >= 1.2) else "holding"
    else:
        be_state = "none"

    return {
        "has_signal": bool(entry is not None and shares),
        "signal_id": snapshot_dict.get("latest_signal_id"),
        "symbol": candidate.get("symbol"),
        "strategy": candidate.get("strategy"),
        "rationale": candidate.get("rationale"),
        "entry": entry,
        "stop": stop,
        "target": target,
        "target_r": target_r,
        "risk_per_share": risk_per_share,
        "shares": shares,
        "actual_risk": actual_risk,
        "notional": notional,
        "risk_budget": float(getattr(getattr(config, "risk", None), "max_risk_per_trade", 60.0) or 60.0),
        "notional_cap": float(getattr(getattr(config, "risk", None), "max_notional_per_trade", 10_000.0) or 10_000.0),
        "breakeven_trigger": be_trigger,
        "breakeven_stop": be_stop,
        "breakeven_state": be_state,
        "position_r": position_r,
    }


def serialize_journal(journal: Any) -> Dict[str, Any]:
    """Convert the SQLite journal into the JSON shape used by the UI."""
    try:
        positions = journal.open_positions()
        closed = journal.closed_trades(et_date=et_now().date())
    except Exception:  # noqa: BLE001
        return {"open_positions": [], "closed_trades": [], "today_pnl": 0.0, "today_count": 0}

    position_rows = []
    for position in positions:
        position_rows.append(
            {
                "signal_id": position.signal_id,
                "symbol": position.symbol,
                "strategy": getattr(position.strategy, "value", str(position.strategy)),
                "mode": getattr(position.mode, "value", str(position.mode)),
                "shares": position.shares,
                "entry_price": round(position.entry_price, 4),
                "stop_loss": round(position.stop_loss, 4),
                "take_profit": round(position.take_profit, 4),
                "entry_utc": position.entry_utc.isoformat(),
            }
        )
    trade_rows = []
    total_pnl = 0.0
    for trade in closed:
        exit_fill = float(trade.exit_price_fill or 0.0)
        pnl = (exit_fill - float(trade.entry_price_fill)) * float(trade.shares) - float(trade.fees or 0.0)
        risk = max(float(trade.entry_price_fill) - float(trade.stop_price), 1e-9)
        r_multiple = (exit_fill - float(trade.entry_price_fill)) / risk if exit_fill else 0.0
        total_pnl += pnl
        trade_rows.append(
            {
                "signal_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": getattr(trade.strategy, "value", str(trade.strategy)),
                "shares": trade.shares,
                "entry_price": round(float(trade.entry_price_fill), 4),
                "exit_price": round(exit_fill, 4),
                "entry_utc": trade.entry_time_utc.isoformat(),
                "exit_utc": trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
                "exit_reason": trade.exit_reason,
                "pnl": round(pnl, 2),
                "r_multiple": round(r_multiple, 3),
            }
        )
    return {
        "open_positions": position_rows,
        "closed_trades": trade_rows,
        "today_pnl": round(total_pnl, 2),
        "today_count": len(trade_rows),
    }


def build_ui_state(
    snapshot_dict: Dict[str, Any],
    tracker_state: Dict[str, Any],
    config: Any,
    *,
    prev_closes: Optional[Dict[str, float]] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    journal_state: Optional[Dict[str, Any]] = None,
    provider_name: str = "alpaca",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    prev_closes = prev_closes or {}
    symbols_state: Dict[str, Any] = {}
    for symbol in SYMBOLS:
        display = dict((tracker_state.get("symbols") or {}).get(symbol) or {})
        last = _safe_float(display.get("last_price"))
        prev = _safe_float(prev_closes.get(symbol))
        change_pct = round((last / prev - 1.0) * 100.0, 3) if last and prev else None
        display["change_pct"] = change_pct
        display["change_abs"] = round(last - prev, 4) if last and prev else None
        symbols_state[symbol] = display

    risk_state = snapshot_dict.get("risk") or {}
    pnl = _safe_float(snapshot_dict.get("daily_pnl")) or 0.0
    risk_cfg = getattr(config, "risk", None)
    daily_loss_limit = float(
        getattr(risk_cfg, "max_daily_net_loss", None)
        or getattr(risk_cfg, "daily_loss_limit", None)
        or 320.0
    )
    cooldown_until_raw = risk_state.get("cooldown_until_utc") if isinstance(risk_state, dict) else None
    cooldown_left = 0
    if cooldown_until_raw:
        try:
            cooldown_until = datetime.fromisoformat(str(cooldown_until_raw))
            if cooldown_until.tzinfo is None:
                cooldown_until = cooldown_until.replace(tzinfo=timezone.utc)
            cooldown_left = max(0, int((cooldown_until - now).total_seconds()))
        except ValueError:
            cooldown_left = 0

    execution = build_execution_card(snapshot_dict, config)
    return {
        "server": {
            "provider": provider_name,
            "generated_at_utc": now.isoformat(),
            "et": now.astimezone(ET).isoformat(),
            "kst": now.astimezone(KST).isoformat(),
        },
        "connection": {
            "state": snapshot_dict.get("connection_state", "DISCONNECTED"),
            "data_status": snapshot_dict.get("data_status", "UNAVAILABLE"),
            "mode": snapshot_dict.get("mode", "DUAL_WATCH"),
            "phase": snapshot_dict.get("session_phase", "CLOSED"),
            "strategy_names": list(snapshot_dict.get("strategy_names") or []),
        },
        "countdown": forced_flat_countdown(now),
        "symbols": symbols_state,
        "risk": {
            "entries_today": int(snapshot_dict.get("entries_today") or 0),
            "max_entries": int(snapshot_dict.get("max_entries_per_day") or 15),
            "trades_today": int(snapshot_dict.get("trades_today") or 0),
            "wins_today": int(snapshot_dict.get("wins_today") or 0),
            "stops_today": int(snapshot_dict.get("stops_today") or 0),
            "daily_pnl": round(pnl, 2),
            "daily_loss_limit": daily_loss_limit,
            "loss_progress": max(0.0, min(1.0, abs(min(pnl, 0.0)) / daily_loss_limit)) if daily_loss_limit else 0.0,
            "consecutive_losses": int(risk_state.get("consecutive_losses") or 0) if isinstance(risk_state, dict) else 0,
            "cooldown_seconds_left": cooldown_left,
            "daily_stopped": bool(snapshot_dict.get("daily_stopped") or (risk_state.get("daily_stopped") if isinstance(risk_state, dict) else False)),
            "daily_realized_pnl": _safe_float(risk_state.get("daily_realized_pnl")) if isinstance(risk_state, dict) else None,
            "account_equity": float(getattr(risk_cfg, "account_equity", 10_000.0) or 10_000.0),
            "risk_per_trade": float(getattr(risk_cfg, "max_risk_per_trade", 60.0) or 60.0),
            "notional_cap": float(getattr(risk_cfg, "max_notional_per_trade", 10_000.0) or 10_000.0),
            "cooldown_threshold": int(getattr(risk_cfg, "consecutive_stop_threshold", 3) or 3),
            "cooldown_minutes": int(getattr(risk_cfg, "consecutive_stop_cooldown_minutes", 15) or 15),
            "config_name": "B1",
        },
        "radar": {
            symbol: {
                "orb": symbols_state[symbol].get("orb"),
                "orb_status": symbols_state[symbol].get("orb_status"),
                "smc_1m": symbols_state[symbol].get("smc_1m"),
                "smc_5m": symbols_state[symbol].get("smc_5m"),
            }
            for symbol in SYMBOLS
        },
        "execution": execution,
        "journal": journal_state or {"open_positions": [], "closed_trades": [], "today_pnl": 0.0, "today_count": 0},
        "events": list(events or [])[-8:],
    }


# --------------------------------------------------------------------------- #
# Live runner
# --------------------------------------------------------------------------- #
class LiveTerminal:
    def __init__(self, config: Any, tracker: SessionTracker, prev_closes: Dict[str, float]) -> None:
        from intraday_scalper.main import Runtime

        self.config = config
        self.tracker = tracker
        self.prev_closes = prev_closes
        self.state: Dict[str, Any] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=30)
        self._lock = threading.RLock()
        self.runtime = Runtime(config, token="", provider="alpaca", audio=False)
        self._wrap_callbacks()
        self._started_utc: Optional[datetime] = None
        self._signal_path: Optional[Path] = None
        self._command_lock = threading.Lock()
        self._history_bars: List[Bar] = []
        self._v04_state: Optional[Dict[str, Any]] = None
        self._v04_last_count = -1
        self._v04_marker_keys: set[str] = set()
        self._ext_marker_keys: set[str] = set()

    def _wrap_callbacks(self) -> None:
        provider = self.runtime.alpaca
        if provider is None:
            raise RuntimeError("Alpaca provider not constructed")
        original_tick = provider._on_tick
        original_bar = provider._on_bar

        def on_tick(tick: Tick) -> None:
            self.tracker.on_tick(tick)
            original_tick(tick)

        def on_bar(bar: Bar) -> None:
            self.tracker.on_bar(bar)
            if original_bar is not None:
                original_bar(bar)

        provider._on_tick = on_tick
        provider._on_bar = on_bar

    def start(self) -> None:
        startup = self.runtime.startup_validation()
        print(f"[terminal] startup validation: {json.dumps(startup, ensure_ascii=False)[:400]}", flush=True)
        provider = self.runtime.alpaca
        if provider is None:
            raise RuntimeError("Alpaca provider not constructed")
        provider.start()
        connected = provider.wait_until_connected(timeout=20.0)
        print(f"[terminal] provider connected={connected}", flush=True)
        threading.Thread(target=self.runtime._status_loop, name="terminal-status", daemon=True).start()
        self._started_utc = datetime.now(timezone.utc)
        self._signal_path = SIGNALS_DIR / f"signals_{et_now().date().isoformat()}.jsonl"
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._history_bars = load_local_history_bars(30)
            print(f"[terminal] V0.4 warm-up history bars: {len(self._history_bars)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[terminal] V0.4 history load skipped: {exc}", flush=True)
        threading.Thread(target=self._loop, name="terminal-state-loop", daemon=True).start()
        threading.Thread(target=self._extended_loop, name="terminal-extended-loop", daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                snapshot = self.runtime.snapshot()
                snap_dict = snapshot.to_dict()
                for event in self.runtime.drain_events():
                    # V0.4 is the displayed engine: do not forward V0.1 strategy alerts.
                    if event.kind.value not in {"INFO", "STATE_CHANGE", "DATA_UNAVAILABLE", "EOD_REMINDER"}:
                        continue
                    payload = _event_dict(event)
                    with self._lock:
                        self.events.append(payload)
                    if self._signal_path is not None:
                        with self._signal_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                tracker_state = self.tracker.snapshot()
                journal_state = serialize_journal(self.runtime.journal)
                try:
                    self._maybe_evaluate_v04()
                except Exception as exc:  # noqa: BLE001
                    print(f"[terminal] V0.4 evaluation error: {exc}", flush=True)
                state = build_ui_state(
                    snap_dict,
                    tracker_state,
                    self.config,
                    prev_closes=self.prev_closes,
                    events=list(self.events),
                    journal_state=journal_state,
                )
                state["v04"] = self._v04_state
                with self._lock:
                    self.state = state
            except Exception as exc:  # noqa: BLE001 - keep the terminal alive
                print(f"[terminal] state loop error: {exc}", flush=True)
            time.sleep(0.8)

    def _extended_loop(self) -> None:
        """Refresh yfinance prepost 5m data and emit pre/post limit-order alerts."""
        try:
            from intraday_scalper.providers.extended_hours_provider import ExtendedHoursProvider

            provider = ExtendedHoursProvider(period="5d")
        except Exception as exc:  # noqa: BLE001
            print(f"[terminal] extended provider unavailable: {exc}", flush=True)
            return
        while True:
            try:
                provider.refresh()
                payload = provider.payload()
                today = et_now().date()
                for signal in payload.get("signals", []):
                    try:
                        signal_dt = datetime.fromisoformat(str(signal.get("time_utc"))).astimezone(ET)
                    except Exception:  # noqa: BLE001
                        continue
                    if signal_dt.date() != today:
                        continue
                    key = f"{signal.get('symbol')}|{signal.get('type')}|{signal.get('time_utc')}|{signal.get('window')}"
                    if key in self._ext_marker_keys:
                        continue
                    self._ext_marker_keys.add(key)
                    window = signal.get("window")
                    prefix = "【盘前信号·限价单】" if window == "pre" else ("【盘后信号·限价单】" if window == "post" else "【盘中 PM 锚点】")
                    self.events.append(
                        {
                            "ts_utc": signal.get("time_utc"),
                            "ts_et": signal.get("time_utc"),
                            "kind": "ENTRY_SIGNAL",
                            "level": "CRITICAL",
                            "symbol": signal.get("symbol"),
                            "signal_id": f"EXT-{signal.get('symbol')}-{signal.get('type')}-{signal.get('t')}",
                            "message": (
                                f"{prefix} V0.4 {signal.get('type')} {signal.get('symbol')} "
                                f"entry {signal.get('entry')} / stop {signal.get('stop')} / target {signal.get('target')}"
                            ),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[terminal] extended loop error: {exc}", flush=True)
            time.sleep(300)

    def _maybe_evaluate_v04(self) -> None:
        """Run the V0.4 evaluator on the live session once per new 5m bucket."""
        from intraday_scalper.research import v04

        current = self.tracker.current_day
        if current is None or current != et_now().date():
            return
        session_records = []
        for symbol in SYMBOLS:
            session_records.extend(self.tracker.bars_for(symbol, "1m"))
        if len(session_records) < 10:
            return
        five_count = max(0, (len(session_records) // (5 * len(SYMBOLS))))
        if five_count == self._v04_last_count:
            return
        session_bars: List[Bar] = []
        for record in session_records:
            session_bars.append(
                Bar(
                    symbol=record.get("symbol", ""),
                    timeframe_minutes=1,
                    start_utc=record["start_utc"],
                    end_utc=record["end_utc"],
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(record["volume"]),
                    vwap=float(record["vwap"]),
                    trade_count=0,
                )
            )
        # tracker records do not carry symbol; rebuild per symbol instead
        session_bars = []
        for symbol in SYMBOLS:
            for record in self.tracker.bars_for(symbol, "1m"):
                session_bars.append(
                    Bar(
                        symbol=symbol,
                        timeframe_minutes=1,
                        start_utc=record["start_utc"],
                        end_utc=record["end_utc"],
                        open=float(record["open"]),
                        high=float(record["high"]),
                        low=float(record["low"]),
                        close=float(record["close"]),
                        volume=float(record["volume"]),
                        vwap=float(record["vwap"]),
                        trade_count=0,
                    )
                )
        all_bars = list(self._history_bars) + session_bars
        trades = v04.evaluate_v04_session(all_bars, current, variant=v04.STEP3C_NOCAP_VARIANT)
        markers: List[Dict[str, Any]] = []
        latest_trade: Optional[Dict[str, Any]] = None
        for trade in trades:
            latest_trade = trade
            for marker in trade.get("markers", []) or []:
                markers.append(marker)
        summary = {
            "trades": len(trades),
            "wins": sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0),
            "net": round(sum(float(t.get("pnl", 0.0)) for t in trades), 2),
        }
        self._v04_state = {
            "session_date": current.isoformat(),
            "trades": trades[-20:],
            "markers": markers,
            "summary": summary,
            "latest_trade": latest_trade,
            "variant": "V0.4 Step3c-A NoCap",
        }
        # emit alerts for new V0.4 markers (drives banner + sound)
        kind_map = {
            "ENTRY": "ENTRY_SIGNAL",
            "STOP": "STOP_HIT_HINT",
            "TARGET": "TAKE_PROFIT_HINT",
            "BREAKEVEN": "BREAKEVEN_HINT",
            "EXIT": "INFO",
        }
        for marker in markers:
            key = f"{marker.get('trade_id')}|{marker.get('type')}|{marker.get('time_utc')}"
            if key in self._v04_marker_keys:
                continue
            self._v04_marker_keys.add(key)
            kind = kind_map.get(str(marker.get("type", "")).upper(), "INFO")
            self.events.append(
                {
                    "ts_utc": marker.get("time_utc"),
                    "ts_et": marker.get("time_utc"),
                    "kind": kind,
                    "level": "CRITICAL" if kind == "ENTRY_SIGNAL" else "WARNING",
                    "symbol": marker.get("symbol"),
                    "signal_id": marker.get("trade_id"),
                    "message": (
                        f"V0.4(NoCap) {marker.get('type')} {marker.get('symbol')} "
                        f"@ {marker.get('price')} · {marker.get('strategy', '')}"
                    ),
                }
            )
        self._v04_last_count = five_count

    def candles(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        return {"symbol": symbol, "timeframe": timeframe, "bars": self.tracker.candles(symbol, timeframe)}

    def command(self, line: str) -> str:
        with self._command_lock:
            try:
                return self.runtime.handle_command(line)
            except Exception as exc:  # noqa: BLE001
                return f"命令执行失败：{exc}"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.state

    def stop(self) -> None:
        try:
            self.runtime.shutdown()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Startup helpers: previous close + today's 1m backfill (one-time REST calls)
# --------------------------------------------------------------------------- #
def _historical_client():
    from alpaca.data.historical import StockHistoricalDataClient
    from intraday_scalper.providers.alpaca_provider import load_alpaca_credentials

    credentials = load_alpaca_credentials()
    return StockHistoricalDataClient(credentials.api_key, credentials.secret_key)


def fetch_startup_market_data(symbols: Sequence[str] = SYMBOLS) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
    """One-time REST: previous daily close and today's 1m bars (IEX, adjusted)."""
    prev_closes: Dict[str, float] = {}
    today_bars: Dict[str, Any] = {}
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = _historical_client()
        now = datetime.now(timezone.utc)
        et_today = now.astimezone(ET)
        session_open = et_today.replace(hour=9, minute=30, second=0, microsecond=0)
        day_request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=now - timedelta(days=10),
            end=now,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
            limit=10000,
        )
        day_bars = client.get_stock_bars(day_request).data
        for symbol in symbols:
            bars = day_bars.get(symbol) or []
            if len(bars) >= 2:
                prev_closes[symbol] = float(bars[-2].close)
            elif bars:
                prev_closes[symbol] = float(bars[-1].close)
        minute_request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=now - timedelta(days=7),
            end=now,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
            limit=10000,
        )
        minute_bars = client.get_stock_bars(minute_request).data
        today_bars = {symbol: list(minute_bars.get(symbol) or []) for symbol in symbols}
        print(
            f"[terminal] REST backfill: prev_closes={prev_closes} today_1m_bars="
            f"{ {k: len(v) for k, v in today_bars.items()} }",
            flush=True,
        )
        daily_request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=now - timedelta(days=260),
            end=now,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
            limit=10000,
        )
        daily_data = client.get_stock_bars(daily_request).data
        daily_bars = {symbol: list(daily_data.get(symbol) or [])[-160:] for symbol in symbols}
        print(
            f"[terminal] REST daily history: { {k: len(v) for k, v in daily_bars.items()} }",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - UI still works without REST backfill
        print(f"[terminal] REST backfill failed (UI continues): {exc}", flush=True)
        daily_bars = {}
    return prev_closes, today_bars, daily_bars


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
_EXTENDED_PROVIDER = None
_EXTENDED_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": {}}


def extended_payload(force: bool = False) -> Dict[str, Any]:
    """Extended-hours context with hybrid real-time quote merge.

    Base yfinance 5m prepost is refreshed at most every 5 minutes; during
    08:00-17:00 ET the Alpaca IEX snapshot is polled at most every 20 seconds and
    merged into the current 5m bucket.  ``data_age_seconds``/``bar_ages_seconds``
    are returned for the UI latency badge.
    """
    global _EXTENDED_PROVIDER
    now = time.time()
    try:
        from intraday_scalper.providers.extended_hours_provider import ExtendedHoursProvider

        if _EXTENDED_PROVIDER is None:
            _EXTENDED_PROVIDER = ExtendedHoursProvider(period="5d")
        provider = _EXTENDED_PROVIDER
        now_utc = datetime.now(timezone.utc)
        full_age = None if provider._last_full_refresh is None else (now_utc - provider._last_full_refresh).total_seconds()
        if force or full_age is None or full_age > 300:
            provider.refresh()
        et = et_now()
        in_quote_window = dt_time(8, 0) <= et.time() < dt_time(17, 0)
        poll_age = None if provider._last_quote_poll is None else (now_utc - provider._last_quote_poll).total_seconds()
        if in_quote_window and (poll_age is None or poll_age > 20):
            provider.refresh_quotes()
        payload = provider.payload()
    except Exception as exc:  # noqa: BLE001
        payload = {
            "session_date": None,
            "symbols": {},
            "pm": {},
            "signals": [],
            "error": f"extended-hours data unavailable: {exc}",
        }
    _EXTENDED_CACHE["ts"] = now
    _EXTENDED_CACHE["payload"] = payload
    return payload


class TerminalHandler(BaseHTTPRequestHandler):
    server_version = "IntradayTerminal/1.0"

    def _auth_ok(self) -> bool:
        password = os.environ.get("TERMINAL_PASSWORD", "").strip()
        if not password:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return False
        _, _, supplied = decoded.partition(":")
        return supplied == password

    def _require_auth(self) -> bool:
        if self._auth_ok():
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Intraday Terminal"')
        body = b"authentication required"
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        route = urlparse(self.path).path
        if route != "/healthz" and not self._require_auth():
            return
        if route in ("/", "/index.html", "/terminal.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/api/extended"):
            payload = extended_payload()
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/api/replay"):
            provider = getattr(self.server, "replay_provider", None)
            payload = provider() if provider is not None else {"session_date": None, "bars": {}, "markers": [], "recap": {}}
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/api/candles"):
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            symbol = (query.get("symbol") or ["SOXL"])[0].upper()
            timeframe = (query.get("tf") or ["1m"])[0]
            payload = self.server.candles_provider(symbol, timeframe)  # type: ignore[attr-defined]
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route.startswith("/api/state"):
            state = self.server.state_provider()  # type: ignore[attr-defined]
            body = json.dumps(state, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/healthz":
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        if self.path != "/api/command":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            line = str(payload.get("line") or "").strip()
            handler = getattr(self.server, "command_handler", None)
            message = handler(line) if handler is not None and line else "空命令"
            body = json.dumps({"ok": True, "message": message}, ensure_ascii=False).encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"ok": False, "message": str(exc)}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet default HTTP logs
        return


def serve(
    state_provider: Callable[[], Dict[str, Any]],
    port: int = 8080,
    *,
    host: str = "127.0.0.1",
    candles_provider: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    command_handler: Optional[Callable[[str], str]] = None,
    replay_provider: Optional[Callable[[], Dict[str, Any]]] = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), TerminalHandler)
    server.state_provider = state_provider  # type: ignore[attr-defined]
    server.candles_provider = candles_provider or (lambda symbol, timeframe: {"symbol": symbol, "timeframe": timeframe, "bars": []})  # type: ignore[attr-defined]
    server.command_handler = command_handler  # type: ignore[attr-defined]
    server.replay_provider = replay_provider or replay_payload  # type: ignore[attr-defined]
    return server


# --------------------------------------------------------------------------- #
# Demo state (offline UI verification)
# --------------------------------------------------------------------------- #
_DEMO_TRACKER: Optional[SessionTracker] = None


def _demo_tracker() -> SessionTracker:
    global _DEMO_TRACKER
    if _DEMO_TRACKER is None:
        _DEMO_TRACKER = SessionTracker()
        now = datetime.now(timezone.utc)
        session_open = now.astimezone(ET).replace(hour=9, minute=30, second=0, microsecond=0)
        closes = [44.0, 44.1, 44.05, 44.2, 44.3, 44.25, 44.35, 44.5, 44.42, 44.55,
                  44.6, 44.52, 44.68, 44.72, 44.65, 44.8, 44.9, 44.86, 44.95, 45.0,
                  45.1, 45.05, 45.2, 45.15, 45.3, 45.25, 45.4, 45.35]
        for index, close in enumerate(closes):
            start = session_open + timedelta(minutes=index)
            for symbol, base in (("SOXL", 1.0), ("SOXS", 1.66)):
                price = close * base
                bar = Bar(
                    symbol=symbol,
                    timeframe_minutes=1,
                    start_utc=start.astimezone(timezone.utc),
                    end_utc=(start + timedelta(minutes=1)).astimezone(timezone.utc),
                    open=price,
                    high=price * 1.002,
                    low=price * 0.998,
                    close=price,
                    volume=1200.0 if symbol == "SOXL" else 800.0,
                    vwap=price,
                    trade_count=12,
                )
                _DEMO_TRACKER.on_bar(bar)
    return _DEMO_TRACKER


def demo_candles(symbol: str, timeframe: str) -> Dict[str, Any]:
    tracker = _demo_tracker()
    if timeframe == "1d":
        today = et_now().date()
        bars = tracker.candles(symbol, "1m")
        if not bars:
            return {"symbol": symbol, "timeframe": timeframe, "bars": []}
        day_bars = [{
            "t": bars[0]["t"] - 86400000,
            "o": round(bars[0]["o"] * 0.995, 4),
            "h": round(max(bar["h"] for bar in bars) * 0.998, 4),
            "l": round(min(bar["l"] for bar in bars) * 0.997, 4),
            "c": round(bars[0]["o"] * 0.999, 4),
            "v": 900000,
        }]
        return {"symbol": symbol, "timeframe": timeframe, "bars": day_bars}
    return {"symbol": symbol, "timeframe": timeframe, "bars": tracker.candles(symbol, timeframe)}


def demo_state() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    tracker = _demo_tracker()
    day = now.astimezone(ET).date()
    session_open = now.astimezone(ET).replace(hour=9, minute=30, second=0, microsecond=0)
    rng = [1.0, 0.998, 1.004, 1.008, 1.002, 0.996, 0.992, 1.001, 1.012, 1.018,
           1.024, 1.02, 1.014, 1.022, 1.03, 1.026, 1.033, 1.041, 1.037, 1.045]
    for index, factor in enumerate(rng):
        start = session_open + timedelta(minutes=index)
        for symbol, base in (("SOXL", 44.0), ("SOXS", 74.0)):
            price = base * factor
            bar = Bar(
                symbol=symbol,
                timeframe_minutes=1,
                start_utc=start.astimezone(timezone.utc),
                end_utc=(start + timedelta(minutes=1)).astimezone(timezone.utc),
                open=price,
                high=price * 1.002,
                low=price * 0.998,
                close=price * 1.001,
                volume=1200.0 if symbol == "SOXL" else 800.0,
                vwap=price,
                trade_count=12 if symbol == "SOXL" else 8,
            )
            tracker.on_bar(bar)
    tracker_state = tracker.snapshot()
    snapshot = {
        "mode": "DUAL_WATCH",
        "session_phase": "REGULAR",
        "data_status": "OK",
        "connection_state": "CONNECTED",
        "strategy_names": ["ORB_5M", "VWAP_RECLAIM", "EMA_PULLBACK"],
        "entries_today": 3,
        "max_entries_per_day": 15,
        "trades_today": 3,
        "wins_today": 2,
        "stops_today": 1,
        "daily_pnl": 17.4,
        "daily_stopped": False,
        "risk": {
            "consecutive_losses": 1,
            "cooldown_until_utc": None,
            "daily_realized_pnl": 17.4,
            "daily_stopped": False,
        },
        "latest_signal": {
            "symbol": "SOXL",
            "strategy": "ORB_5M",
            "entry_ref": 44.12,
            "stop_loss": 43.72,
            "target_r": 2.0,
            "rationale": "开盘区间突破 + 相对活跃度确认",
            "created_utc": now.isoformat(),
        },
        "latest_plan": {
            "shares": 50.0,
            "notional": 2206.0,
            "actual_risk": 20.0,
            "theoretical_risk": 20.0,
            "valid": True,
        },
        "position": None,
    }
    journal_state = {
        "open_positions": [
            {
                "signal_id": "DEMO-0001",
                "symbol": "SOXL",
                "strategy": "orb_5m",
                "mode": "long_soxl",
                "shares": 20.0,
                "entry_price": 44.12,
                "stop_loss": 43.72,
                "take_profit": 44.92,
                "entry_utc": now.isoformat(),
            }
        ],
        "closed_trades": [
            {
                "signal_id": "DEMO-0000",
                "symbol": "SOXS",
                "strategy": "vwap_reclaim",
                "shares": 10.0,
                "entry_price": 75.10,
                "exit_price": 75.52,
                "entry_utc": now.isoformat(),
                "exit_utc": now.isoformat(),
                "exit_reason": "TARGET",
                "pnl": 4.2,
                "r_multiple": 1.0,
            }
        ],
        "today_pnl": 4.2,
        "today_count": 1,
    }
    return build_ui_state(
        snapshot,
        tracker_state,
        b1_config(),
        prev_closes={"SOXL": 43.6, "SOXS": 74.9},
        provider_name="demo",
        journal_state=journal_state,
        events=[{"kind": "ENTRY_SIGNAL", "level": "CRITICAL", "message": "SOXL ORB_5M 入场信号", "ts_utc": now.isoformat()}],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="local web terminal for the SOXL/SOXS intraday engine")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--host",
        default=os.environ.get("TERMINAL_HOST", "127.0.0.1"),
        help="bind host (use 0.0.0.0 on Render/Railway)",
    )
    parser.add_argument("--demo", action="store_true", help="serve synthetic demo state, no broker connection")
    parser.add_argument("--open", action="store_true", help="open the browser after the server starts")
    parser.add_argument("--no-broker", action="store_true", help="alias of --demo")
    args = parser.parse_args(argv)
    demo = args.demo or args.no_broker

    if demo:
        state_provider = demo_state
        server = serve(state_provider, port=args.port, host=args.host, candles_provider=demo_candles, replay_provider=replay_payload)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"[terminal] demo server on {url}", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return 0

    config = b1_config()
    tracker = SessionTracker()
    local = load_local_session()
    local_session = None
    local_added = 0
    if local.get("records"):
        local_added = tracker.backfill(local["records"])
        local_session = tracker.current_day
    prev_closes, today_bars, daily_bars = fetch_startup_market_data()
    rest_dates = [
        bar.timestamp.astimezone(ET).date()
        for bars in (today_bars or {}).values()
        for bar in bars
        if getattr(bar, "timestamp", None) is not None
    ]
    rest_latest = max(rest_dates) if rest_dates else None
    added = 0
    if rest_latest is not None and (local_session is None or rest_latest > local_session):
        added = tracker.backfill(today_bars)
    if not prev_closes:
        prev_closes = derive_prev_closes_local()
    daily_added = tracker.backfill_daily(daily_bars)
    print(
        f"[terminal] tracker backfilled local={local_added} rest={added} 1m bars / "
        f"{daily_added} daily bars (session {tracker.current_day})",
        flush=True,
    )
    terminal = LiveTerminal(config, tracker, prev_closes)
    terminal.start()
    server = serve(
        terminal.snapshot,
        port=args.port,
        host=args.host,
        candles_provider=terminal.candles,
        command_handler=terminal.command,
        replay_provider=replay_payload,
    )
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[terminal] live server on {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        terminal.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
