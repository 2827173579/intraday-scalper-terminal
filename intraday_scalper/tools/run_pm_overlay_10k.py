"""D variant: V0.4 Step3c-A NoCap under plan-B risk + PM_High/PM_Low overlay.

The regular V0.4 backtest uses raw IEX 1m bars (data/*_1m_raw.parquet).
PM-anchor signals are computed from Alpaca IEX **extended-hours 1m bars**
(04:00-20:00 ET, fetched via REST and cached) aggregated to 5m:
  - SMC_PM_LOW_SWEEP: sweep PM_Low + reclaim + close >= EMA21 & session VWAP
  - PM_HIGH_BREAKOUT: break PM_High + pullback holds EMA9/VWAP
They are simulated as an **overlay** on top of plan B (risk $80, notional $4,500,
fractional 0.01), one position at a time within the overlay, 2R target / stop,
1.5 bps slippage.  This is an additive estimate: PM trades do NOT share the V0.4
risk manager state (cooldown / daily breaker / max one position) with the regular
engine — documented explicitly in the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_scalper.core import Bar, et_date, load_config
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.providers import extended_hours_provider as ext
from intraday_scalper.providers.alpaca_provider import load_alpaca_credentials
from intraday_scalper.research import v02, v04
from intraday_scalper.tools.run_risk_compare_10k import (
    WINDOW_END,
    WINDOW_START,
    load_raw_bars,
    risk_config,
)

ET = ZoneInfo("America/New_York")
PACKAGE = PACKAGE_ROOT
DATA_DIR = PACKAGE / "data"
EXT_DIR = PACKAGE / "var" / "extended"
REPORT_DIR = PACKAGE / "var" / "backtest"
COST_BPS = 1.5


def load_extended_1m() -> pd.DataFrame:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EXT_DIR / "alpaca_ext_1m_2026_08_11.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    credentials = load_alpaca_credentials()
    client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = datetime(2026, 11, 7, tzinfo=timezone.utc)
    rows = []
    for symbol in ("SOXL", "SOXS"):
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
            adjustment=Adjustment.RAW,
            feed=DataFeed.IEX,
            limit=100_000,
        )
        data = client.get_stock_bars(request).data
        for bar in list(data.get(symbol) or []):
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            rows.append(
                {
                    "symbol": symbol,
                    "start_utc": ts,
                    "end_utc": ts + timedelta(minutes=1),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume or 0.0),
                    "vwap": float(getattr(bar, "vwap", None) or bar.close),
                    "trade_count": int(getattr(bar, "trade_count", 0) or 0),
                }
            )
    frame = pd.DataFrame(rows)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["end_utc"] = pd.to_datetime(frame["end_utc"], utc=True)
    frame.to_parquet(cache, index=False)
    return frame


def aggregate_extended_5m(frame: pd.DataFrame) -> Dict[str, List[Bar]]:
    result: Dict[str, List[Bar]] = {}
    for symbol, group in frame.groupby("symbol"):
        group = group.sort_values("start_utc").copy()
        group["bucket"] = group["start_utc"].dt.floor("5min")
        bars: List[Bar] = []
        for bucket, chunk in group.groupby("bucket"):
            volume = float(chunk["volume"].sum())
            vwap = (
                float((chunk["vwap"] * chunk["volume"]).sum() / volume)
                if volume > 0
                else float(((chunk["high"] + chunk["low"] + chunk["close"]) / 3.0).mean())
            )
            high = float(chunk["high"].max())
            low = float(chunk["low"].min())
            vwap = min(max(float(vwap), low), high)
            bars.append(
                Bar(
                    symbol=str(symbol),
                    timeframe_minutes=5,
                    start_utc=bucket.to_pydatetime(),
                    end_utc=(bucket + pd.Timedelta(minutes=5)).to_pydatetime(),
                    open=float(chunk["open"].iloc[0]),
                    high=high,
                    low=low,
                    close=float(chunk["close"].iloc[-1]),
                    volume=volume,
                    vwap=vwap,
                    trade_count=int(chunk["trade_count"].sum()),
                )
            )
        result[str(symbol)] = sorted(bars, key=lambda bar: bar.start_utc)
    return result


def simulate_pm_trades(
    signals: Sequence[Dict[str, Any]],
    bars_5m: Sequence[Bar],
    *,
    risk_per_trade: float,
    max_notional: float,
    precision: int = 2,
) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    ordered = sorted(bars_5m, key=lambda bar: bar.start_utc)
    for signal in signals:
        signal_ts = datetime.fromisoformat(str(signal["time_utc"]))
        future = [bar for bar in ordered if bar.start_utc > signal_ts]
        if not future:
            continue
        entry_bar = future[0]
        entry = float(entry_bar.open) * (1.0 + COST_BPS / 10_000.0)
        signal_close = float(signal["price"])
        stop = float(signal["stop"])
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 1e-6:
            continue
        target = entry + 2.0 * risk_per_share
        qty_risk = risk_per_trade / risk_per_share
        qty_notional = max_notional / entry
        shares = math.floor(min(qty_risk, qty_notional) / (10 ** -precision)) * (10 ** -precision)
        if shares <= 0:
            continue
        exit_price = None
        exit_time = None
        reason = "EOD"
        entry_day = entry_bar.start_utc.astimezone(ET).date()
        same_day_bars = []
        for bar in ordered:
            if bar.start_utc <= entry_bar.start_utc:
                continue
            bar_et = bar.start_utc.astimezone(ET)
            if bar_et.date() != entry_day or bar_et.time() >= time(16, 0):
                break
            same_day_bars.append(bar)
            hit_stop = float(bar.low) <= stop
            hit_target = float(bar.high) >= target
            if hit_stop and hit_target:
                exit_price, exit_time, reason = stop, bar.start_utc, "STOP"
                break
            if hit_stop:
                exit_price, exit_time, reason = stop, bar.start_utc, "STOP"
                break
            if hit_target:
                exit_price, exit_time, reason = target, bar.start_utc, "TARGET"
                break
        if exit_price is None:
            last = same_day_bars or [bar for bar in ordered if bar.start_utc > entry_bar.start_utc][:1]
            if not last:
                continue
            exit_price = float(last[-1].close)
            exit_time = last[-1].start_utc
        exit_fill = exit_price * (1.0 - COST_BPS / 10_000.0)
        pnl = (exit_fill - entry) * shares
        if pnl == 0.0 and reason == "EOD":
            continue
        trades.append(
            {
                "symbol": signal.get("symbol"),
                "strategy": signal.get("type"),
                "entry_time_utc": entry_bar.start_utc.isoformat(),
                "exit_time_utc": exit_time.isoformat() if exit_time else None,
                "entry_price": round(entry, 4),
                "exit_price": round(exit_fill, 4),
                "shares": round(shares, precision),
                "pnl": round(pnl, 2),
                "exit_reason": reason,
                "source": "PM_OVERLAY",
            }
        )
    return trades


def simulate_pm_overlay(
    signals: Sequence[Dict[str, Any]],
    bars_by_symbol: Dict[str, List[Bar]],
    *,
    risk_per_trade: float,
    max_notional: float,
    precision: int = 2,
) -> List[Dict[str, Any]]:
    """Simulate PM-anchor signals with **one active position at a time**.

    This is closer to the live rule (max one position); signals arriving while a
    PM trade is open are skipped.  The regular V0.4 engine still runs separately
    (overlay caveat unchanged).
    """
    trades: List[Dict[str, Any]] = []
    ordered_signals = sorted(
        signals,
        key=lambda item: datetime.fromisoformat(str(item["time_utc"])),
    )
    busy_until: Optional[datetime] = None
    for signal in ordered_signals:
        signal_ts = datetime.fromisoformat(str(signal["time_utc"]))
        if busy_until is not None and signal_ts < busy_until:
            continue
        symbol = signal.get("symbol")
        ordered = sorted(bars_by_symbol.get(symbol, []), key=lambda bar: bar.start_utc)
        future = [bar for bar in ordered if bar.start_utc > signal_ts]
        if not future:
            continue
        entry_bar = future[0]
        entry = float(entry_bar.open) * (1.0 + COST_BPS / 10_000.0)
        stop = float(signal["stop"])
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 1e-6:
            continue
        target = entry + 2.0 * risk_per_share
        qty_risk = risk_per_trade / risk_per_share
        qty_notional = max_notional / entry
        shares = math.floor(min(qty_risk, qty_notional) / (10 ** -precision)) * (10 ** -precision)
        if shares <= 0:
            continue
        entry_day = entry_bar.start_utc.astimezone(ET).date()
        exit_price: Optional[float] = None
        exit_time: Optional[datetime] = None
        reason = "EOD"
        same_day_bars: List[Bar] = []
        for bar in ordered:
            if bar.start_utc <= entry_bar.start_utc:
                continue
            bar_et = bar.start_utc.astimezone(ET)
            if bar_et.date() != entry_day or bar_et.time() >= time(16, 0):
                break
            same_day_bars.append(bar)
            hit_stop = float(bar.low) <= stop
            hit_target = float(bar.high) >= target
            if hit_stop:
                exit_price, exit_time, reason = stop, bar.start_utc, "STOP"
                break
            if hit_target:
                exit_price, exit_time, reason = target, bar.start_utc, "TARGET"
                break
        if exit_price is None:
            if not same_day_bars:
                continue
            exit_price = float(same_day_bars[-1].close)
            exit_time = same_day_bars[-1].start_utc
        exit_fill = exit_price * (1.0 - COST_BPS / 10_000.0)
        pnl = (exit_fill - entry) * shares
        busy_until = exit_time
        trades.append(
            {
                "symbol": symbol,
                "strategy": signal.get("type"),
                "entry_time_utc": entry_bar.start_utc.isoformat(),
                "exit_time_utc": exit_time.isoformat() if exit_time else None,
                "entry_price": round(entry, 4),
                "exit_price": round(exit_fill, 4),
                "shares": round(shares, precision),
                "pnl": round(pnl, 2),
                "exit_reason": reason,
                "source": "PM_OVERLAY",
            }
        )
    return trades


def main() -> Dict[str, Any]:
    # 1) B-risk regular V0.4 trades
    bars = load_raw_bars()
    prepared = v02.prepare_v02_data(bars)
    dates = [day for day in prepared.dates if WINDOW_START <= day <= WINDOW_END]
    config, risk = risk_config("B")
    runtime_config = replace(config, risk=risk)
    base = v04.run_v04_backtest(
        prepared,
        runtime_config,
        variant=v04.STEP3C_NOCAP_VARIANT,
        session_dates=dates,
        initial_equity=10_000.0,
    )
    base_trades = []
    for trade in base.trades:
        if not trade.is_closed:
            continue
        base_trades.append(
            {
                "symbol": trade.symbol,
                "strategy": getattr(trade.strategy, "value", str(trade.strategy)),
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "exit_time_utc": trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
                "entry_price": round(float(trade.entry_price_fill), 4),
                "exit_price": round(float(trade.exit_price_fill or 0.0), 4),
                "shares": round(float(trade.shares), 4),
                "pnl": round(float(trade.pnl), 2),
                "exit_reason": trade.exit_reason,
                "source": "V0.4",
            }
        )

    # 2) extended 5m bars + PM anchor signals per session
    frame = load_extended_1m()
    bars5 = aggregate_extended_5m(frame)
    pm_signals: List[Dict[str, Any]] = []
    for symbol, symbol_bars in bars5.items():
        sessions = sorted({bar.start_utc.astimezone(ET).date() for bar in symbol_bars})
        for session in sessions:
            if not (WINDOW_START <= session <= WINDOW_END):
                continue
            signals = ext.evaluate_extended_signals(symbol, symbol_bars, session_date=session)
            pm_signals.extend([s for s in signals if s["type"] in {"SMC_PM_LOW_SWEEP", "PM_HIGH_BREAKOUT"}])
    pm_trades: List[Dict[str, Any]] = simulate_pm_overlay(
        pm_signals,
        bars5,
        risk_per_trade=float(risk.max_risk_per_trade),
        max_notional=float(risk.max_notional_per_trade),
        precision=int(risk.fractional_precision),
    )

    combined = base_trades + pm_trades
    combined.sort(key=lambda row: row["entry_time_utc"])
    initial = 10_000.0
    wins = sum(1 for row in combined if row["pnl"] > 0)
    gross_profit = sum(row["pnl"] for row in combined if row["pnl"] > 0)
    gross_loss = -sum(row["pnl"] for row in combined if row["pnl"] < 0)
    net = sum(row["pnl"] for row in combined)
    equity = initial
    peak = initial
    max_dd = 0.0
    for row in sorted(combined, key=lambda item: item["exit_time_utc"] or item["entry_time_utc"]):
        equity += row["pnl"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    months: Dict[str, Dict[str, Any]] = {}
    for row in combined:
        month = datetime.fromisoformat(row["entry_time_utc"]).astimezone(ET).strftime("%Y-%m")
        info = months.setdefault(month, {"trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net": 0.0})
        info["trades"] += 1
        info["net"] += row["pnl"]
        if row["pnl"] > 0:
            info["wins"] += 1
            info["gross_profit"] += row["pnl"]
        else:
            info["gross_loss"] += -row["pnl"]
    month_rows = {}
    for month, info in sorted(months.items()):
        month_rows[month] = {
            "trades": info["trades"],
            "win_rate_pct": round(info["wins"] / info["trades"] * 100.0, 2) if info["trades"] else 0.0,
            "profit_factor": round(info["gross_profit"] / info["gross_loss"], 4) if info["gross_loss"] > 0 else None,
            "net": round(info["net"], 2),
        }
    payload = {
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "initial_equity": initial,
        "engine": "V0.4 Step3c-A NoCap + PM anchors overlay",
        "caveat": (
            "D = B (regular V0.4) + PM-anchor overlay estimated independently; PM trades do not "
            "share V0.4 RiskManager state/positions with the regular engine. Use as directional "
            "evidence, not as a joint risk-managed portfolio."
        ),
        "summary": {
            "trades": len(combined),
            "v04_trades": len(base_trades),
            "pm_trades": len(pm_trades),
            "win_rate_pct": round(wins / len(combined) * 100.0, 2) if combined else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
            "final_equity": round(initial + net, 2),
            "net": round(net, 2),
            "net_pct": round(net / initial * 100.0, 2),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd / initial * 100.0, 2),
            "breakers": int(base.breaker_days),
        },
        "months": month_rows,
        "pm_signals": pm_signals,
        "pm_trades": pm_trades,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "pm_overlay_10k.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("months:", json.dumps(month_rows, ensure_ascii=False, indent=2))
    print("PM signals:", len(pm_signals), "PM trades:", len(pm_trades))
    return payload


if __name__ == "__main__":
    main()
