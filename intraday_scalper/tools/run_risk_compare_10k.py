"""3-month ($10,000) risk-config comparison on raw IEX 1m data.

Window: 2026-08-01 .. 2026-11-06, engine V0.4 Step3c-A NoCap.
Variants A/B/C only change risk config (equity, risk/trade, notional cap,
consecutive-stop cooldown, daily net-loss breaker); D adds PM-anchor signals
(separate overlay, documented in the output).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from intraday_scalper.core import Bar, et_date, load_config
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v04, v02

PACKAGE = PACKAGE_ROOT
DATA_DIR = PACKAGE / "data"
REPORT_DIR = PACKAGE / "var" / "backtest"
WINDOW_START = date(2026, 8, 1)
WINDOW_END = date(2026, 11, 6)


def load_raw_bars() -> List[Bar]:
    frames = []
    for symbol in ("SOXL", "SOXS"):
        path = DATA_DIR / f"{symbol}_1m_raw.parquet"
        frame = pd.read_parquet(path)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["end_utc"] = pd.to_datetime(frame["end_utc"], utc=True)
    frame = frame[(frame["start_utc"] >= "2024-01-01")].sort_values(["start_utc", "symbol"])
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


def risk_config(name: str):
    config = load_config()
    base = config.risk
    common = dict(
        account_equity=10_000.0,
        allow_fractional_shares=True,
        fractional_precision=2,
        max_notional_per_trade=4_500.0,
        max_entries_per_day=15,
    )
    if name == "A":
        return config, replace(
            base,
            max_risk_per_trade=60.0,
            consecutive_stop_threshold=2,
            consecutive_stop_cooldown_minutes=20,
            max_daily_net_loss=180.0,
            **common,
        )
    if name == "B":
        return config, replace(
            base,
            max_risk_per_trade=80.0,
            consecutive_stop_threshold=3,
            consecutive_stop_cooldown_minutes=15,
            max_daily_net_loss=320.0,
            **common,
        )
    if name == "C":
        return config, replace(
            base,
            max_risk_per_trade=100.0,
            consecutive_stop_threshold=10**6,
            consecutive_stop_cooldown_minutes=0,
            max_daily_net_loss=10**9,
            **common,
        )
    raise ValueError(name)


def summarize_result(result: Any, label: str) -> Dict[str, Any]:
    metrics = result.metrics
    trades = [trade for trade in result.trades if trade.is_closed]
    months: Dict[str, Dict[str, Any]] = {}
    for trade in trades:
        month = et_date(trade.entry_time_utc).strftime("%Y-%m")
        row = months.setdefault(month, {"trades": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net": 0.0})
        row["trades"] += 1
        row["net"] += trade.pnl
        if trade.pnl > 0:
            row["wins"] += 1
            row["gross_profit"] += trade.pnl
        else:
            row["gross_loss"] += -trade.pnl
    month_rows = {}
    for month, row in sorted(months.items()):
        month_rows[month] = {
            "trades": row["trades"],
            "win_rate_pct": round(row["wins"] / row["trades"] * 100.0, 2) if row["trades"] else 0.0,
            "profit_factor": round(row["gross_profit"] / row["gross_loss"], 4) if row["gross_loss"] > 0 else None,
            "net": round(row["net"], 2),
        }
    initial = float(result.initial_equity)
    final = float(result.final_equity)
    dd_pct = getattr(metrics, "max_drawdown_pct", None)
    if dd_pct is None:
        dd_pct = (metrics.max_drawdown_dollars / initial * 100.0) if initial else 0.0
    return {
        "name": label,
        "trades": int(metrics.total_trades),
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": None if metrics.profit_factor is None else round(metrics.profit_factor, 4),
        "initial_equity": initial,
        "final_equity": round(final, 2),
        "net": round(final - initial, 2),
        "net_pct": round((final / initial - 1.0) * 100.0, 2),
        "max_drawdown": round(metrics.max_drawdown_dollars, 2),
        "max_drawdown_pct": round(float(dd_pct), 2),
        "breaker_days": int(result.breaker_days),
        "risk_blocks": dict(result.risk_block_counts),
        "months": month_rows,
    }


def run() -> Dict[str, Any]:
    bars = load_raw_bars()
    prepared = v02.prepare_v02_data(bars)
    dates = [day for day in prepared.dates if WINDOW_START <= day <= WINDOW_END]
    print(f"prepared bars={len(bars)} sessions={len(prepared.dates)} window_sessions={len(dates)}")
    rows = []
    for name in ("A", "B", "C"):
        config, risk = risk_config(name)
        runtime_config = replace(config, risk=risk)
        result = v04.run_v04_backtest(
            prepared,
            runtime_config,
            variant=v04.STEP3C_NOCAP_VARIANT,
            session_dates=dates,
            initial_equity=10_000.0,
        )
        row = summarize_result(result, name)
        rows.append(row)
        print(f"{name}: trades={row['trades']} win={row['win_rate_pct']} PF={row['profit_factor']} "
              f"final=${row['final_equity']:,.2f} net={row['net']:+.2f} ({row['net_pct']:+.2f}%) "
              f"DD=${row['max_drawdown']:,.2f} breaker_days={row['breaker_days']}")
        for month, info in row["months"].items():
            print(f"   {month}: {info['trades']} trades, win {info['win_rate_pct']}%, net {info['net']:+.2f}")
    payload = {
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "initial_equity": 10_000.0,
        "engine": "V0.4 Step3c-A NoCap",
        "data": "raw IEX 1m bars from data/{SOXL,SOXS}_1m_raw.parquet",
        "variants": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "risk_compare_10k_3m.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# $10,000 · 2026-08-01~2026-11-06 风控方案对比（V0.4 Step3c-A NoCap，raw IEX 1m）",
        "",
        "| 方案 | 笔数 | 胜率 | PF | 期末资产 | 净盈亏 | 净% | MaxDD$ | 熔断天数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pf_text = "—" if row["profit_factor"] is None else f"{row['profit_factor']:.4f}"
        lines.append(
            f"| {row['name']} | {row['trades']} | {row['win_rate_pct']:.2f}% | {pf_text} | "
            f"${row['final_equity']:,.2f} | {row['net']:+.2f} | {row['net_pct']:+.2f}% | "
            f"${row['max_drawdown']:,.2f} | {row['breaker_days']} |"
        )
    lines += ["", "## 分月拆解", ""]
    for row in rows:
        lines.append(f"### 方案 {row['name']}")
        lines.append("")
        lines.append("| 月份 | 笔数 | 胜率 | PF | 净盈亏 |")
        lines.append("|---|---:|---:|---:|---:|")
        for month, info in row["months"].items():
            pf = "—" if info["profit_factor"] is None else f"{info['profit_factor']:.4f}"
            lines.append(f"| {month} | {info['trades']} | {info['win_rate_pct']:.2f}% | {pf} | {info['net']:+.2f} |")
        lines.append("")
    (REPORT_DIR / "risk_compare_10k_3m.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote risk_compare_10k_3m.json/md")
    return payload


if __name__ == "__main__":
    run()
