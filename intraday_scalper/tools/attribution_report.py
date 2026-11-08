"""V0.1 real-data backtest attribution report (4 tables + segment metrics).

What this tool does
-------------------
Read-only analysis on top of the *completed* V0.1 real-data backtest:

1. loads the local Alpaca IEX 1m parquet cache via
   :func:`intraday_scalper.tools.run_real_backtest.load_bars`;
2. reproduces the exact V0.1 full-period backtest
   (``load_config()`` + ``build_strategies()`` + ``backtest.engine.run_backtest``);
3. merges ``result.trades`` (TradeRecord) with ``result.trade_details``
   (signal_id -> MAE/MFE/slippage/ambiguous annotations) into a per-trade ledger;
4. writes three artefacts under ``intraday_scalper/var/backtest/``:

   * ``attribution.md``   — markdown tables ready to paste into
     ``docs/verification_report.md`` §9;
   * ``attribution.json`` — the same data structured for machine use;
   * ``real_iex_trades.csv`` — per-trade ledger (price/R/MAE-MFE/stop width/exit).

Tables
------
A. US/Eastern time-of-day buckets (09:30-10:00 / 10:00-11:00 / 11:00-13:30 /
   13:30-15:30): trades, wins, losses, win rate, PF, net P&L, average R.
B. strategy x symbol cross table (+ per-strategy and per-symbol totals) and a
   written explanation of why SOXL lost far more than SOXS.
C. exit-reason decomposition: STOP (non-collision), TARGET (2.0R), EOD_EXIT,
   END_OF_DATA, same-bar TP/SL collision (``ambiguous_bar=True``), and the
   1.2R break-even exit which **V0.1 never implemented as an automatic exit**
   (0 trades; break-even is a *hint only*).
D. stop-width / MAE-MFE diagnostics on STOP trades: stop-distance percentage
   distribution and buckets, MAE/MFE in R, share of stopped trades that had
   MFE >= 0.5R / 1.0R / 1.2R before being stopped, plus a quantitative
   "are 1m stops too tight / noise-swept?" conclusion.

Also: segment metrics for 2024 (in-sample year), 2025-01-01..2026-11-05, and the
full cycle, plus an explicit self-consistency check against the frozen V0.1
baseline (``var/backtest/real_iex_report.json``: 2,687 trades / win rate 34.72% /
PF 0.91 / net -$1,734.06 / max drawdown $2,265.07 / 22 breaker days / slippage
$823.84).

Boundaries / rules
------------------
* Only this new file and the three ``var/backtest/`` artefacts above are written.
* No network, no config/strategy changes, no parameter tuning, no commits.
* Data caveat: **Alpaca IEX single-venue 1m bars** — volume/VWAP are IEX-scoped,
  NOT consolidated tape; results are not evidence of real-world profitability.

Usage::

    python -m intraday_scalper.tools.attribution_report
    python -m intraday_scalper.tools.attribution_report --from-csv   # re-render only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from intraday_scalper.backtest.engine import run_backtest
from intraday_scalper.core import et_date, load_config, session_open_utc, to_et, to_utc
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.strategies.registry import build_strategies
from intraday_scalper.tools.run_real_backtest import (
    CAVEAT,
    DATA_DIR,
    REPORT_DIR,
    _fmt_metrics,
    circuit_breaker_stats,
    daily_pnl,
    equity_stats,
    load_bars,
)

UTC = timezone.utc

ATTRIBUTION_MD = "attribution.md"
ATTRIBUTION_JSON = "attribution.json"
TRADES_CSV = "real_iex_trades.csv"
BASELINE_JSON = "real_iex_report.json"

STRATEGY_ORDER: Tuple[str, ...] = ("ORB_5M", "ORB_15M", "VWAP_RECLAIM", "EMA_PULLBACK")
SYMBOL_ORDER: Tuple[str, ...] = ("SOXL", "SOXS")

#: (label, start inclusive, end exclusive) in minutes after the 09:30 ET open.
TIME_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("09:30-10:00", 0.0, 30.0),
    ("10:00-11:00", 30.0, 90.0),
    ("11:00-13:30", 90.0, 240.0),
    ("13:30-15:30", 240.0, 360.0),
    ("15:30+（V0.1 应无新仓）", 360.0, math.inf),
)

#: Segment boundaries (inclusive) keyed on the ET *entry* date.
SEGMENTS: Tuple[Tuple[str, Optional[date], Optional[date]], ...] = (
    ("训练集 2024-01-01..2024-12-31", date(2024, 1, 1), date(2024, 12, 31)),
    ("后段 2025-01-01..2026-11-05", date(2025, 1, 1), date(2026, 11, 5)),
    ("全周期 2024-01-02..2026-11-05", None, None),
)

STOP_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("<0.25%", 0.0, 0.25),
    ("0.25-0.50%", 0.25, 0.50),
    ("0.50-0.75%", 0.50, 0.75),
    ("0.75-1.00%", 0.75, 1.00),
    ("1.00-1.50%", 1.00, 1.50),
    ("1.50-2.00%", 1.50, 2.00),
    (">2.00%", 2.00, math.inf),
)

CSV_COLUMNS: Tuple[str, ...] = (
    "signal_id",
    "symbol",
    "strategy",
    "mode",
    "shares",
    "entry_time_utc",
    "entry_time_et",
    "entry_price_signal",
    "entry_price_fill",
    "stop_price",
    "target_price",
    "exit_time_utc",
    "exit_time_et",
    "exit_price_fill",
    "pnl",
    "r_multiple",
    "mae_r",
    "mfe_r",
    "stop_distance_pct",
    "exit_reason",
    "ambiguous",
    "holding_minutes",
    "entry_slippage_bps",
    "mae_dollars",
    "mfe_dollars",
    "risk_dollars",
    "fees",
    "entry_slippage_dollars",
    "exit_slippage_dollars",
    "trade_id",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _json_safe(value: Any) -> Any:
    """Recursively replace inf/nan (JSON has no representation for them)."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return value


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / len(values) if values else 0.0


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (same convention as numpy/pandas default)."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    if gross_profit > 0.0:
        return math.inf
    return 0.0


# --------------------------------------------------------------------------- #
# Ledger construction (engine result -> per-trade dicts)
# --------------------------------------------------------------------------- #
def build_ledger(result: Any) -> List[Dict[str, Any]]:
    """Merge ``result.trades`` with ``result.trade_details`` into plain dicts."""

    ledger: List[Dict[str, Any]] = []
    details = getattr(result, "trade_details", {}) or {}
    for trade in result.trades:
        if not trade.is_closed or trade.exit_time_utc is None:
            continue
        detail = details.get(trade.signal_id, {}) or {}
        entry_time_utc = to_utc(trade.entry_time_utc)
        exit_time_utc = to_utc(trade.exit_time_utc)
        entry_et = to_et(entry_time_utc)
        exit_et = to_et(exit_time_utc)
        session_open = session_open_utc(entry_time_utc)
        risk_dollars = float(trade.risk_dollars)
        mae_dollars = float(detail.get("mae_dollars", 0.0) or 0.0)
        mfe_dollars = float(detail.get("mfe_dollars", 0.0) or 0.0)
        entry_slip_dollars = float(detail.get("entry_slippage_dollars", 0.0) or 0.0)
        exit_slip_dollars = float(detail.get("exit_slippage_dollars", 0.0) or 0.0)
        entry_fill = float(trade.entry_price_fill or 0.0)
        ledger.append(
            {
                "signal_id": str(trade.signal_id),
                "trade_id": int(trade.trade_id or 0),
                "symbol": str(trade.symbol),
                "strategy": str(trade.strategy.value),
                "mode": str(trade.mode.value),
                "shares": float(trade.shares),
                "entry_time_utc": entry_time_utc,
                "entry_time_et": entry_et,
                "entry_et_date": entry_et.date(),
                "session_minutes": (entry_time_utc - session_open).total_seconds() / 60.0,
                "exit_time_utc": exit_time_utc,
                "exit_time_et": exit_et,
                "entry_price_signal": float(trade.entry_price_signal),
                "entry_price_fill": entry_fill,
                "stop_price": float(trade.stop_price),
                "target_price": float(trade.target_price),
                "exit_price_fill": float(trade.exit_price_fill or 0.0),
                "exit_reason": str(trade.exit_reason or "OTHER"),
                "ambiguous": bool(detail.get("ambiguous_bar", False)),
                "holding_minutes": float(trade.holding_minutes),
                "pnl": float(trade.pnl),
                "r_multiple": float(trade.r_multiple),
                "risk_dollars": risk_dollars,
                "mae_dollars": mae_dollars,
                "mfe_dollars": mfe_dollars,
                "mae_r": (mae_dollars / risk_dollars) if risk_dollars > 0 else 0.0,
                "mfe_r": (mfe_dollars / risk_dollars) if risk_dollars > 0 else 0.0,
                "stop_distance_pct": (
                    (entry_fill - float(trade.stop_price)) / entry_fill * 100.0 if entry_fill > 0 else 0.0
                ),
                "entry_slippage_bps": float(trade.entry_slippage_bps),
                "entry_slippage_dollars": entry_slip_dollars,
                "exit_slippage_dollars": exit_slip_dollars,
                "fees": float(trade.fees),
            }
        )
    return ledger


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def stats(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Win/loss/PF/expectancy summary for a list of ledger rows."""

    pnls = [row["pnl"] for row in trades]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl < 0.0)
    breakeven = len(pnls) - wins - losses
    gross_profit = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0.0)
    r_values = [row["r_multiple"] for row in trades]
    holding = [row["holding_minutes"] for row in trades]
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": (wins / len(trades)) if trades else 0.0,
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": sum(pnls),
        "avg_r": _mean(r_values),
        "avg_holding_minutes": _mean(holding),
    }


# --------------------------------------------------------------------------- #
# Formatting helpers (markdown + terminal)
# --------------------------------------------------------------------------- #
def _fmt_money(value: float) -> str:
    return f"{value:+,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _fmt_pf(value: float) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    return f"{value:.2f}"


def _fmt_r(value: float) -> str:
    return f"{value:+.3f}"


def _fmt_hold(value: float) -> str:
    return f"{value:.1f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a markdown table."""

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _num_row(label: str, summary: Dict[str, Any]) -> List[str]:
    """Standard stats row used by tables A/B."""

    return [
        label,
        str(summary["trades"]),
        _fmt_pct(summary["win_rate"]),
        _fmt_pf(summary["profit_factor"]),
        _fmt_money(summary["net_pnl"]),
        _fmt_r(summary["avg_r"]),
        _fmt_hold(summary["avg_holding_minutes"]),
    ]


# --------------------------------------------------------------------------- #
# Tables A / B / C / D + segments
# --------------------------------------------------------------------------- #
def build_table_a(ledger: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, low, high in TIME_BUCKETS:
        subset = [row for row in ledger if low <= row["session_minutes"] < high]
        rows.append({"bucket": label, **stats(subset)})
    return rows


def build_table_b(ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cross: List[Dict[str, Any]] = []
    for strategy in STRATEGY_ORDER:
        for symbol in SYMBOL_ORDER:
            subset = [
                row for row in ledger if row["strategy"] == strategy and row["symbol"] == symbol
            ]
            cross.append({"strategy": strategy, "symbol": symbol, **stats(subset)})

    per_strategy = [
        {"strategy": strategy, **stats([row for row in ledger if row["strategy"] == strategy])}
        for strategy in STRATEGY_ORDER
    ]
    per_symbol = [
        {"symbol": symbol, **stats([row for row in ledger if row["symbol"] == symbol])}
        for symbol in SYMBOL_ORDER
    ]
    return {
        "cross": cross,
        "per_strategy": per_strategy,
        "per_symbol": per_symbol,
        "total": stats(ledger),
    }


def build_table_c(ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Exit-reason partition (rows sum to the total trade count)."""

    categories: List[Tuple[str, str]] = [
        ("STOP（非碰撞，止损）", "stop"),
        ("TARGET（2.0R 止盈）", "target"),
        ("EOD_EXIT（尾盘强平）", "eod"),
        ("END_OF_DATA（数据结束强平）", "end_of_data"),
        ("同 bar TP/SL 碰撞（ambiguous，保守按止损）", "ambiguous"),
        ("1.2R 保本出局（V0.1 未实现自动出场，仅提示）", "breakeven"),
        ("其他/未知", "other"),
    ]
    buckets: Dict[str, List[Dict[str, Any]]] = {key: [] for _, key in categories}
    for row in ledger:
        reason = row["exit_reason"]
        if row["ambiguous"] and reason == "STOP":
            buckets["ambiguous"].append(row)
        elif reason == "STOP":
            buckets["stop"].append(row)
        elif reason == "TARGET":
            buckets["target"].append(row)
        elif reason == "EOD_EXIT":
            buckets["eod"].append(row)
        elif reason == "END_OF_DATA":
            buckets["end_of_data"].append(row)
        else:
            buckets["other"].append(row)

    rows: List[Dict[str, Any]] = []
    for label, key in categories:
        summary = stats(buckets[key])
        rows.append({"category": label, "key": key, **summary})
    rows.append({"category": "合计", "key": "total", **stats(ledger)})

    covered = sum(row["trades"] for row in rows[:-1])
    return {
        "rows": rows,
        "ambiguous_stop_count": len(buckets["ambiguous"]),
        "breakeven_exit_count": len(buckets["breakeven"]),
        "covered_trades": covered,
        "total_trades": len(ledger),
        "partition_ok": covered == len(ledger),
    }


def build_table_d(ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stop-width + MAE/MFE diagnostics for STOP (non-EOD) trades."""

    stops = [row for row in ledger if row["exit_reason"] == "STOP"]
    n = len(stops)
    stop_pcts = sorted(float(row["stop_distance_pct"]) for row in stops)
    mean_sd = _mean(stop_pcts)
    median_sd = _percentile(stop_pcts, 0.5)
    p25_sd = _percentile(stop_pcts, 0.25)
    p75_sd = _percentile(stop_pcts, 0.75)

    mae_rs = [float(row["mae_r"]) for row in stops]
    mfe_rs = [float(row["mfe_r"]) for row in stops]

    stop_thresholds = {
        "mfe_ge_0.5r": sum(1 for value in mfe_rs if value >= 0.5) / n if n else 0.0,
        "mfe_ge_1.0r": sum(1 for value in mfe_rs if value >= 1.0) / n if n else 0.0,
        "mfe_ge_1.2r": sum(1 for value in mfe_rs if value >= 1.2) / n if n else 0.0,
    }
    mfe_counts = {
        "mfe_ge_0.5r": sum(1 for value in mfe_rs if value >= 0.5),
        "mfe_ge_1.0r": sum(1 for value in mfe_rs if value >= 1.0),
        "mfe_ge_1.2r": sum(1 for value in mfe_rs if value >= 1.2),
    }

    bucket_rows: List[Dict[str, Any]] = []
    for label, low, high in STOP_BUCKETS:
        subset = [row for row in stops if low <= row["stop_distance_pct"] < high]
        summary = stats(subset)
        bucket_mfe = [float(row["mfe_r"]) for row in subset]
        bucket_rows.append(
            {
                "bucket": label,
                "trades": len(subset),
                "share_pct": (len(subset) / n * 100.0) if n else 0.0,
                "net_pnl": summary["net_pnl"],
                "avg_r": summary["avg_r"],
                "avg_mae_r": _mean([float(row["mae_r"]) for row in subset]),
                "avg_mfe_r": _mean(bucket_mfe),
                "share_mfe_ge_0_5r": (
                    sum(1 for value in bucket_mfe if value >= 0.5) / len(subset) * 100.0
                    if subset
                    else 0.0
                ),
            }
        )

    narrow = [row for row in stops if row["stop_distance_pct"] < 0.5]
    wide = [row for row in stops if row["stop_distance_pct"] >= 1.0]
    narrow_summary = stats(narrow)
    wide_summary = stats(wide)
    narrow_share = len(narrow) / n if n else 0.0
    wide_share = len(wide) / n if n else 0.0
    narrow_mfe_share = (
        sum(1 for row in narrow if row["mfe_r"] >= 0.5) / len(narrow) if narrow else 0.0
    )
    wide_mfe_share = sum(1 for row in wide if row["mfe_r"] >= 0.5) / len(wide) if wide else 0.0
    if abs(narrow_mfe_share - wide_mfe_share) <= 0.10:
        mfe_compare = "接近"
    elif narrow_mfe_share > wide_mfe_share:
        mfe_compare = "窄止损桶更高"
    else:
        mfe_compare = "宽止损桶更高"
    r_quality = "更差" if narrow_summary["avg_r"] < wide_summary["avg_r"] else "更好"

    if n == 0:  # pragma: no cover - V0.1 always has stops
        judgement = "无可诊断的止损单。"
    elif narrow_share >= 0.40 and stop_thresholds["mfe_ge_0.5r"] >= 0.40:
        judgement = (
            "1m 止损明显偏窄、噪音扫损特征强：窄止损占比与「先浮盈后被扫」占比同时偏高。"
        )
    elif narrow_share >= 0.30 or stop_thresholds["mfe_ge_0.5r"] >= 0.30:
        judgement = "1m 止损偏窄/存在一定噪音扫损，但证据强度中等。"
    else:
        judgement = "止损宽度与噪音扫损特征不明显，亏损更可能来自方向性判断而非止损过窄。"

    conclusion = (
        f"被止损单共 {n} 笔（占全部 {len(ledger)} 笔的 {n / len(ledger) * 100.0:.1f}%），"
        f"止损距离中位数 {median_sd:.2f}%（均值 {mean_sd:.2f}%，P25 {p25_sd:.2f}% / "
        f"P75 {p75_sd:.2f}%）。窄止损（<0.50%）占 {narrow_share * 100.0:.1f}%"
        f"（{len(narrow)} 笔），该组平均 R {narrow_summary['avg_r']:+.3f}、"
        f"净盈亏 ${narrow_summary['net_pnl']:+,.2f}；"
        f"宽止损（≥1.00%）占 {wide_share * 100.0:.1f}%（{len(wide)} 笔），"
        f"该组平均 R {wide_summary['avg_r']:+.3f}、净盈亏 ${wide_summary['net_pnl']:+,.2f}。"
        f"全部止损单中 MFE≥0.5R 占 {stop_thresholds['mfe_ge_0.5r'] * 100.0:.1f}%、"
        f"≥1.0R 占 {stop_thresholds['mfe_ge_1.0r'] * 100.0:.1f}%、"
        f"≥1.2R 占 {stop_thresholds['mfe_ge_1.2r'] * 100.0:.1f}%"
        f"（{mfe_counts['mfe_ge_1.2r']} 笔曾达到 1.2R 保本提示线后仍被止损）。"
        f"宽/窄止损桶的 MFE≥0.5R 占比{mfe_compare}"
        f"（窄 {narrow_mfe_share * 100.0:.1f}% vs 宽 {wide_mfe_share * 100.0:.1f}%），"
        f"窄止损桶平均 R {r_quality}（{narrow_summary['avg_r']:+.3f} vs {wide_summary['avg_r']:+.3f}）。"
        f"其中 {sum(1 for row in stops if row['ambiguous'])} 笔为同 bar TP/SL 碰撞的保守止损。"
        f"判断：{judgement}"
    )

    return {
        "stops": n,
        "stops_share_pct": (n / len(ledger) * 100.0) if ledger else 0.0,
        "ambiguous_stops": sum(1 for row in stops if row["ambiguous"]),
        "stop_distance_pct": {
            "mean": mean_sd,
            "median": median_sd,
            "p25": p25_sd,
            "p75": p75_sd,
            "min": min(stop_pcts) if stop_pcts else 0.0,
            "max": max(stop_pcts) if stop_pcts else 0.0,
        },
        "mae_r": {"mean": _mean(mae_rs), "median": _percentile(sorted(mae_rs), 0.5)},
        "mfe_r": {"mean": _mean(mfe_rs), "median": _percentile(sorted(mfe_rs), 0.5)},
        "mfe_thresholds": stop_thresholds,
        "mfe_counts": mfe_counts,
        "buckets": bucket_rows,
        "narrow_lt_0_5_pct": {"trades": len(narrow), "share_pct": narrow_share * 100.0, **narrow_summary},
        "wide_ge_1_0_pct": {"trades": len(wide), "share_pct": wide_share * 100.0, **wide_summary},
        "conclusion": conclusion,
        "judgement": judgement,
    }


def build_segments(ledger: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, start, end in SEGMENTS:
        subset = []
        for row in ledger:
            day = row["entry_et_date"]
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            subset.append(row)
        rows.append({"segment": label, "start": start, "end": end, **stats(subset)})
    return rows


# --------------------------------------------------------------------------- #
# Equity / circuit-breaker / consistency (baseline definitions)
# --------------------------------------------------------------------------- #
def trade_derived_equity(ledger: Sequence[Dict[str, Any]], initial_equity: float) -> Dict[str, Any]:
    """Reproduce ``run_real_backtest.equity_stats`` from the ledger alone."""

    ordered = sorted(ledger, key=lambda row: row["exit_time_utc"])
    equity = initial_equity
    curve: List[Tuple[datetime, float]] = []
    for row in ordered:
        equity += row["pnl"]
        curve.append((row["exit_time_utc"], equity))
    if not curve:
        return {"points": 0, "final_equity": initial_equity, "max_drawdown_dollars": 0.0,
                "max_drawdown_pct": 0.0}
    peak = curve[0][1]
    max_dd = 0.0
    for _, value in curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return {
        "points": len(curve),
        "final_equity": round(curve[-1][1], 2),
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / initial_equity * 100.0, 2),
    }


def trade_derived_circuit_breaker(
    ledger: Sequence[Dict[str, Any]], max_daily_loss: float
) -> Dict[str, Any]:
    """Reproduce ``run_real_backtest.circuit_breaker_stats`` from the ledger."""

    by_day: Dict[str, float] = {}
    for row in ledger:
        day = to_et(row["exit_time_utc"]).date().isoformat()
        by_day[day] = by_day.get(day, 0.0) + row["pnl"]
    if not by_day:
        return {"breaker_days": 0, "traded_days": 0, "worst_day_pnl": None,
                "best_day_pnl": None, "avg_day_pnl": None}
    daily = sorted(by_day.values())
    return {
        "breaker_days": int(sum(1 for pnl in daily if pnl <= -abs(max_daily_loss))),
        "traded_days": len(daily),
        "worst_day_pnl": round(min(daily), 2),
        "best_day_pnl": round(max(daily), 2),
        "avg_day_pnl": round(sum(daily) / len(daily), 2),
    }


def find_files_newer_than(baseline_path: Path) -> List[str]:
    """Files that changed after the baseline report (drift diagnosis helper)."""

    if not baseline_path.exists():
        return []
    cutoff = baseline_path.stat().st_mtime
    changed: List[str] = []
    for pattern in ("*.py", "*.yaml", "*.yml"):
        for path in PACKAGE_ROOT.rglob(pattern):
            text = str(path)
            if any(part in text for part in ("\\.venv\\", "__pycache__", "\\var\\")):
                continue
            if path.name == "attribution_report.py":
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    changed.append(str(path.relative_to(PACKAGE_ROOT)))
            except OSError:  # pragma: no cover - transient file
                continue
    return sorted(changed)


def _check(name: str, baseline: Any, reproduced: Any, tolerance: float, unit: str = "") -> Dict[str, Any]:
    passed: bool
    delta: Optional[float]
    if isinstance(baseline, (int, float)) and isinstance(reproduced, (int, float)):
        if isinstance(baseline, float) and not math.isfinite(baseline):
            passed = baseline == reproduced
            delta = None
        else:
            delta = float(reproduced) - float(baseline)
            passed = abs(delta) <= tolerance
    else:
        delta = None
        passed = baseline == reproduced
    return {
        "metric": name,
        "unit": unit,
        "baseline": _json_safe(baseline),
        "reproduced": _json_safe(reproduced),
        "delta": _json_safe(delta),
        "passed": bool(passed),
    }


def build_consistency(
    ledger: Sequence[Dict[str, Any]],
    full_stats: Dict[str, Any],
    table_b: Dict[str, Any],
    equity_info: Dict[str, Any],
    breaker_info: Dict[str, Any],
    baseline: Dict[str, Any],
    baseline_path: Path,
    reproduction: Dict[str, Any],
) -> Dict[str, Any]:
    full = baseline.get("full_period", {})
    bm = full.get("metrics", {})
    checks: List[Dict[str, Any]] = [
        _check("总笔数", bm.get("total_trades"), full_stats["trades"], 0, "笔"),
        _check("胜", bm.get("wins"), full_stats["wins"], 0, "笔"),
        _check("负", bm.get("losses"), full_stats["losses"], 0, "笔"),
        _check("胜率", bm.get("win_rate_pct"), round(full_stats["win_rate"] * 100.0, 2), 0.01, "%"),
        _check("ProfitFactor", bm.get("profit_factor"), round(full_stats["profit_factor"], 3), 0.005, ""),
        _check("净盈亏", bm.get("net_pnl_dollars"), round(full_stats["net_pnl"], 2), 0.02, "$"),
        _check("平均R", bm.get("avg_r"), round(full_stats["avg_r"], 3), 0.001, "R"),
        _check(
            "滑点合计",
            bm.get("total_slippage_dollars"),
            round(
                sum(row["entry_slippage_dollars"] + row["exit_slippage_dollars"] for row in ledger), 2
            ),
            0.02,
            "$",
        ),
        _check(
            "最大回撤",
            full.get("equity", {}).get("max_drawdown_dollars"),
            equity_info.get("max_drawdown_dollars"),
            0.02,
            "$",
        ),
        _check(
            "期末权益",
            full.get("equity", {}).get("final_equity"),
            equity_info.get("final_equity"),
            0.02,
            "$",
        ),
        _check(
            "熔断天数（单日 -$60 熔断）",
            full.get("circuit_breaker", {}).get("breaker_days"),
            breaker_info.get("breaker_days"),
            0,
            "",
        ),
        _check(
            "交易天数",
            full.get("circuit_breaker", {}).get("traded_days"),
            breaker_info.get("traded_days"),
            0,
            "天",
        ),
        _check(
            "最差单日",
            full.get("circuit_breaker", {}).get("worst_day_pnl"),
            breaker_info.get("worst_day_pnl"),
            0.02,
            "$",
        ),
    ]

    if "ambiguous_bars" in reproduction:
        checks.append(_check("ambiguous_bars", full.get("ambiguous_bars"), reproduction["ambiguous_bars"], 0, "根"))
    if "bars_processed" in reproduction:
        checks.append(_check("bars_processed", full.get("bars_processed"), reproduction["bars_processed"], 0, "根"))

    # per-strategy and per-symbol cross-checks against the baseline report.
    per_strategy_checks: List[Dict[str, Any]] = []
    baseline_per_strategy = bm.get("per_strategy", {})
    reproduced_per_strategy = {row["strategy"]: row for row in table_b["per_strategy"]}
    for strategy, base_stats in baseline_per_strategy.items():
        ours = reproduced_per_strategy.get(strategy, {})
        per_strategy_checks.extend(
            [
                _check(f"{strategy}.trades", base_stats.get("trades"), ours.get("trades"), 0, "笔"),
                _check(
                    f"{strategy}.win_rate",
                    base_stats.get("win_rate"),
                    round(ours.get("win_rate", 0.0), 3),
                    0.0006,
                    "",
                ),
                _check(
                    f"{strategy}.net_pnl",
                    base_stats.get("net_pnl"),
                    round(ours.get("net_pnl", 0.0), 2),
                    0.02,
                    "$",
                ),
                _check(
                    f"{strategy}.avg_r",
                    base_stats.get("avg_r"),
                    round(ours.get("avg_r", 0.0), 3),
                    0.001,
                    "R",
                ),
            ]
        )

    per_symbol_checks: List[Dict[str, Any]] = []
    baseline_per_symbol = bm.get("per_symbol", {})
    reproduced_per_symbol = {row["symbol"]: row for row in table_b["per_symbol"]}
    for symbol, base_stats in baseline_per_symbol.items():
        ours = reproduced_per_symbol.get(symbol, {})
        per_symbol_checks.extend(
            [
                _check(f"{symbol}.trades", base_stats.get("trades"), ours.get("trades"), 0, "笔"),
                _check(
                    f"{symbol}.win_rate",
                    base_stats.get("win_rate"),
                    round(ours.get("win_rate", 0.0), 3),
                    0.0006,
                    "",
                ),
                _check(
                    f"{symbol}.net_pnl",
                    base_stats.get("net_pnl"),
                    round(ours.get("net_pnl", 0.0), 2),
                    0.02,
                    "$",
                ),
                _check(
                    f"{symbol}.avg_r",
                    base_stats.get("avg_r"),
                    round(ours.get("avg_r", 0.0), 3),
                    0.001,
                    "R",
                ),
            ]
        )

    all_checks = checks + per_strategy_checks + per_symbol_checks
    failed = [check for check in all_checks if not check["passed"]]
    drift = find_files_newer_than(baseline_path)
    verdict = (
        "✅ 与 V0.1 基线完全自洽：全周期笔数/胜率/PF/净盈亏/回撤/熔断/滑点及分策略分标的指标一致。"
        if not failed
        else f"⚠️ 与 V0.1 基线存在 {len(failed)} 项差异，需排查（见 checks 中 passed=false 项）。"
    )
    return {
        "baseline_report": str(baseline_path),
        "baseline_generated_at_utc": baseline.get("generated_at_utc"),
        "mode": reproduction.get("mode", "full_engine"),
        "checks": checks,
        "per_strategy_checks": per_strategy_checks,
        "per_symbol_checks": per_symbol_checks,
        "all_passed": not failed,
        "failed_count": len(failed),
        "verdict": verdict,
        "files_newer_than_baseline": drift,
    }


# --------------------------------------------------------------------------- #
# Markdown rendering (§9-ready)
# --------------------------------------------------------------------------- #
def render_markdown(
    *,
    config: Any,
    bars_meta: Dict[str, Any],
    ledger: Sequence[Dict[str, Any]],
    table_a: Sequence[Dict[str, Any]],
    table_b: Dict[str, Any],
    table_c: Dict[str, Any],
    table_d: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    consistency: Dict[str, Any],
    full_stats: Dict[str, Any],
    equity_info: Dict[str, Any],
    breaker_info: Dict[str, Any],
    reproduction: Dict[str, Any],
    top_row: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("## 9. V0.1 真实回测归因分析（IEX 单交易所口径）")
    lines.append("")
    lines.append(
        "> **数据口径**：" + CAVEAT + " "
        "本表由 `tools/attribution_report.py` 基于 `result.trades` + `result.trade_details` "
        "逐笔归因生成，不含任何策略/参数修改。"
    )
    lines.append("")
    lines.append(
        f"> 复现脚本：`intraday_scalper/tools/attribution_report.py` ｜ "
        f"基线：`var/backtest/{BASELINE_JSON}` ｜ "
        f"产物：`attribution.md` / `attribution.json` / `real_iex_trades.csv`"
    )
    lines.append("")

    # -- 9.1 reproduction & baseline consistency ------------------------- #
    lines.append("### 9.1 全周期复现与基线自洽性")
    lines.append("")
    if consistency.get("skipped"):
        lines.append(
            "> ⚠️ 本次运行未做基线自洽性检查（调试模式/缺少基线），表数据仅用于预览。"
        )
        lines.append("")
    else:
        check_rows = []
        for check in consistency["checks"]:
            unit = check["unit"]

            def _fmt_check_value(value: Any, unit_: str) -> str:
                if value is None:
                    return "-"
                if isinstance(value, (int, float)) and unit_ in ("$", "笔", "根", "天"):
                    return f"{value:,}"
                if isinstance(value, float):
                    return f"{value:g}"
                return str(value)

            delta_text = "-"
            if check["delta"] is not None:
                delta = float(check["delta"])
                delta_text = f"{int(delta):+d}" if delta.is_integer() else f"{delta:+.4f}"
            check_rows.append(
                [
                    check["metric"] + (f"（{unit}）" if unit else ""),
                    _fmt_check_value(check["baseline"], unit),
                    _fmt_check_value(check["reproduced"], unit),
                    delta_text,
                    "✅" if check["passed"] else "❌",
                ]
            )
        lines.append(md_table(["指标", "V0.1 基线", "本次复现", "差异", "结果"], check_rows))
        lines.append("")
        lines.append(
            f"**自洽结论**：{consistency['verdict']}"
            + (
                ""
                if consistency["all_passed"]
                else f"（差异项 {consistency['failed_count']} 个；基线后变更文件："
                f"{consistency['files_newer_than_baseline'] or '无'}）"
            )
        )
        lines.append("")
    lines.append(
        f"全周期复现详情：{full_stats['trades']:,} 笔（胜 {full_stats['wins']:,} / "
        f"负 {full_stats['losses']:,}），胜率 {_fmt_pct(full_stats['win_rate'])}，"
        f"PF {_fmt_pf(full_stats['profit_factor'])}，净盈亏 ${full_stats['net_pnl']:+,.2f}，"
        f"平均 R {_fmt_r(full_stats['avg_r'])}，平均持仓 {_fmt_hold(full_stats['avg_holding_minutes'])} 分钟；"
        f"最大回撤 ${equity_info['max_drawdown_dollars']:,.2f}"
        f"（{equity_info['max_drawdown_pct']:.2f}%），期末权益 ${equity_info['final_equity']:,.2f}；"
        f"单日 -${config.risk.max_daily_net_loss:,.0f} 熔断 {breaker_info['breaker_days']} 天，"
        f"交易天数 {breaker_info['traded_days']:,}。"
    )
    lines.append("")

    # -- 9.2 segments ---------------------------------------------------- #
    lines.append("### 9.2 全周期与分段指标（按入场 ET 日期）")
    lines.append("")
    segment_rows = [
        [
            row["segment"],
            str(row["trades"]),
            _fmt_pct(row["win_rate"]),
            _fmt_pf(row["profit_factor"]),
            _fmt_money(row["net_pnl"]),
            _fmt_r(row["avg_r"]),
        ]
        for row in segments
    ]
    lines.append(md_table(["分段", "总笔数", "胜率", "PF", "净盈亏($)", "平均R"], segment_rows))
    lines.append("")

    # -- 9.3 table A ----------------------------------------------------- #
    lines.append("### 9.3 表 A：美东时段归因（按入场时刻）")
    lines.append("")
    a_rows = [
        _num_row(row["bucket"], row) for row in table_a
    ]
    lines.append(md_table(["时段(ET)", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"], a_rows))
    lines.append("")
    traded_buckets = [row for row in table_a if row["trades"] > 0]
    best = max(traded_buckets, key=lambda row: row["net_pnl"]) if traded_buckets else table_a[0]
    worst = min(traded_buckets, key=lambda row: row["net_pnl"]) if traded_buckets else table_a[0]
    first_30 = table_a[0]
    lines.append(
        f"- 最佳时段：**{best['bucket']}**，{best['trades']} 笔，净盈亏 ${best['net_pnl']:+,.2f}"
        f"（胜率 {_fmt_pct(best['win_rate'])}、PF {_fmt_pf(best['profit_factor'])}）。"
    )
    lines.append(
        f"- 最差时段：**{worst['bucket']}**，{worst['trades']} 笔，净盈亏 ${worst['net_pnl']:+,.2f}"
        f"（胜率 {_fmt_pct(worst['win_rate'])}、PF {_fmt_pf(worst['profit_factor'])}）。"
    )
    lines.append(
        f"- 09:30-10:00 开盘 30 分钟：{first_30['trades']} 笔，净盈亏 ${first_30['net_pnl']:+,.2f}，"
        f"平均 R {_fmt_r(first_30['avg_r'])}。"
    )
    lines.append("")

    # -- 9.4 table B ----------------------------------------------------- #
    lines.append("### 9.4 表 B：策略 × 标的交叉归因")
    lines.append("")
    lines.append("**交叉表（策略 × 标的）**")
    lines.append("")
    cross_rows = [
        _num_row(f"{row['strategy']} × {row['symbol']}", row) for row in table_b["cross"]
    ]
    lines.append(md_table(["策略 × 标的", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"], cross_rows))
    lines.append("")
    lines.append("**按策略合计**")
    lines.append("")
    lines.append(
        md_table(
            ["策略", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"],
            [_num_row(row["strategy"], row) for row in table_b["per_strategy"]],
        )
    )
    lines.append("")
    lines.append("**按标的合计**")
    lines.append("")
    lines.append(
        md_table(
            ["标的", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"],
            [_num_row(row["symbol"], row) for row in table_b["per_symbol"]],
        )
    )
    lines.append("")

    soxl = next(row for row in table_b["per_symbol"] if row["symbol"] == "SOXL")
    soxs = next(row for row in table_b["per_symbol"] if row["symbol"] == "SOXS")
    soxl_share = soxl["trades"] / full_stats["trades"] * 100.0 if full_stats["trades"] else 0.0
    soxl_loss_share = (
        soxl["net_pnl"] / full_stats["net_pnl"] * 100.0 if full_stats["net_pnl"] != 0.0 else 0.0
    )
    soxl_expectancy = soxl["net_pnl"] / soxl["trades"] if soxl["trades"] else 0.0
    soxs_expectancy = soxs["net_pnl"] / soxs["trades"] if soxs["trades"] else 0.0
    lines.append(
        f"**SOXL 为何远亏于 SOXS**：SOXL {soxl['trades']:,} 笔（占 {soxl_share:.1f}%）净盈亏 "
        f"${soxl['net_pnl']:+,.2f}（占全部净亏损的 {soxl_loss_share:.1f}%），平均 R {_fmt_r(soxl['avg_r'])}；"
        f"SOXS {soxs['trades']:,} 笔净盈亏 ${soxs['net_pnl']:+,.2f}，平均 R {_fmt_r(soxs['avg_r'])}。"
        f"SOXL 单笔期望 ${soxl_expectancy:+,.2f} vs SOXS ${soxs_expectancy:+,.2f}，"
        f"即 SOXL 平均每笔多亏 ${abs(soxl_expectancy - soxs_expectancy):.2f}；"
        f"最差组合为 **{top_row['strategy']} × {top_row['symbol']}**（{top_row['trades']} 笔，"
        f"净 ${top_row['net_pnl']:+,.2f}，平均 R {_fmt_r(top_row['avg_r'])}）。"
        "结论：亏损并非仅由 SOXL 笔数多造成，SOXL 的单笔质量显著更差，需在 V0.2 中单独审视 SOXL 方向性/波动结构。"
    )
    lines.append("")

    # -- 9.5 table C ----------------------------------------------------- #
    lines.append("### 9.5 表 C：出场原因归因")
    lines.append("")
    c_rows = [
        [
            row["category"],
            str(row["trades"]),
            _fmt_money(row["net_pnl"]),
            _fmt_r(row["avg_r"]),
            _fmt_hold(row["avg_holding_minutes"]),
        ]
        for row in table_c["rows"]
    ]
    lines.append(md_table(["出场类型", "笔数", "净盈亏($)", "平均R", "平均持仓(分)"], c_rows))
    lines.append("")
    lines.append(
        "- **1.2R 保本出局 = 0 笔**：V0.1 只在 1.2R 发出 `BREAKEVEN_HINT` 声音/看板提示，"
        "**没有自动保本出场**，仓位只有在 STOP / TARGET / EOD_EXIT / END_OF_DATA 时了结。"
    )
    lines.append(
        f"- 同 bar TP/SL 碰撞（`ambiguous_bar=True`）{table_c['ambiguous_stop_count']} 笔，"
        "按保守规则以止损成交（`pessimism=stop_first`），在上表中单独成行，不重复计入 STOP（非碰撞）。"
    )
    eod = next((row for row in table_c["rows"] if row["key"] == "eod"), None)
    target = next((row for row in table_c["rows"] if row["key"] == "target"), None)
    stop_row = next(row for row in table_c["rows"] if row["key"] == "stop")
    stop_share = stop_row["trades"] / table_c["total_trades"] * 100.0 if table_c["total_trades"] else 0.0
    target_share = target["trades"] / table_c["total_trades"] * 100.0 if target and table_c["total_trades"] else 0.0
    lines.append(
        f"- 止损（非碰撞）{stop_row['trades']:,} 笔（{stop_share:.1f}%），净 ${stop_row['net_pnl']:+,.2f}，"
        f"平均 R {_fmt_r(stop_row['avg_r'])}；止盈（2.0R）{target['trades']:,} 笔（{target_share:.1f}%），"
        f"净 ${target['net_pnl']:+,.2f}。"
    )
    if eod:
        lines.append(
            f"- 尾盘 EOD_EXIT {eod['trades']:,} 笔，净 ${eod['net_pnl']:+,.2f}，"
            f"平均 R {_fmt_r(eod['avg_r'])}，平均持仓 {_fmt_hold(eod['avg_holding_minutes'])} 分钟。"
        )
    lines.append("")

    # -- 9.6 table D ----------------------------------------------------- #
    lines.append("### 9.6 表 D：MAE/MFE 与止损宽度诊断（仅 STOP 止损单）")
    lines.append("")
    sd = table_d["stop_distance_pct"]
    lines.append(
        f"止损单 **{table_d['stops']:,}** 笔（占全部 {table_d['stops_share_pct']:.1f}%）。"
        f"止损距离（占入场价）：均值 **{sd['mean']:.2f}%** / 中位数 **{sd['median']:.2f}%** / "
        f"P25 **{sd['p25']:.2f}%** / P75 **{sd['p75']:.2f}%**。"
    )
    lines.append("")
    d_rows = [
        [
            row["bucket"],
            str(row["trades"]),
            f"{row['share_pct']:.1f}%",
            _fmt_money(row["net_pnl"]),
            _fmt_r(row["avg_r"]),
            _fmt_r(row["avg_mae_r"]),
            _fmt_r(row["avg_mfe_r"]),
            f"{row['share_mfe_ge_0_5r']:.1f}%",
        ]
        for row in table_d["buckets"]
    ]
    lines.append(
        md_table(
            [
                "止损距离桶",
                "笔数",
                "占止损比",
                "净盈亏($)",
                "平均R",
                "平均MAE(R)",
                "平均MFE(R)",
                "MFE≥0.5R占比",
            ],
            d_rows,
        )
    )
    lines.append("")
    mae = table_d["mae_r"]
    mfe = table_d["mfe_r"]
    thresholds = table_d["mfe_thresholds"]
    lines.append(
        f"- MAE（R 口径）：均值 **{mae['mean']:.3f}R** / 中位数 **{mae['median']:.3f}R**；"
        f"MFE（R 口径）：均值 **{mfe['mean']:.3f}R** / 中位数 **{mfe['median']:.3f}R**。"
    )
    lines.append(
        f"- 被止损单中 MFE≥0.5R 占 **{thresholds['mfe_ge_0.5r'] * 100.0:.1f}%**"
        f"（{table_d['mfe_counts']['mfe_ge_0.5r']:,} 笔）、≥1.0R 占 **{thresholds['mfe_ge_1.0r'] * 100.0:.1f}%**"
        f"（{table_d['mfe_counts']['mfe_ge_1.0r']:,} 笔）、≥1.2R 占 **{thresholds['mfe_ge_1.2r'] * 100.0:.1f}%**"
        f"（{table_d['mfe_counts']['mfe_ge_1.2r']:,} 笔）—— 即这些仓位在被打掉之前曾达到 1.2R 保本提示线，"
        "但 V0.1 没有自动保本出场。"
    )
    lines.append(
        f"- 窄止损（<0.50%）{table_d['narrow_lt_0_5_pct']['trades']:,} 笔"
        f"（{table_d['narrow_lt_0_5_pct']['share_pct']:.1f}%），平均 R "
        f"{_fmt_r(table_d['narrow_lt_0_5_pct']['avg_r'])}、净 "
        f"${table_d['narrow_lt_0_5_pct']['net_pnl']:+,.2f}；宽止损（≥1.00%）"
        f"{table_d['wide_ge_1_0_pct']['trades']:,} 笔，平均 R "
        f"{_fmt_r(table_d['wide_ge_1_0_pct']['avg_r'])}、净 ${table_d['wide_ge_1_0_pct']['net_pnl']:+,.2f}。"
    )
    lines.append("")
    lines.append(f"**表 D 结论（量化）**：{table_d['conclusion']}")
    lines.append("")

    # -- 9.7 conclusions for V0.2 --------------------------------------- #
    lines.append("### 9.7 归因结论（供 V0.2 结构重构，不含调参结论）")
    lines.append("")
    avg_win = full_stats["gross_profit"] / full_stats["wins"] if full_stats["wins"] else 0.0
    avg_loss = full_stats["gross_loss"] / full_stats["losses"] if full_stats["losses"] else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    bullets: List[str] = []
    bullets.append(
        f"1. **亏损高度集中**：SOXL {soxl['trades']:,} 笔（{soxl_share:.1f}% 的笔数）净 "
        f"${soxl['net_pnl']:+,.2f}（占全部净亏损 {soxl_loss_share:.1f}%），"
        f"其中 {top_row['strategy']} × SOXL 组合净 ${top_row['net_pnl']:+,.2f}、平均 R {_fmt_r(top_row['avg_r'])}，"
        "是首要结构性亏损来源。"
    )
    bullets.append(
        f"2. **时段结构**：开盘 09:30-10:00 净 ${first_30['net_pnl']:+,.2f}；"
        f"10:00-11:00 与 11:00-13:30 分别净 "
        f"${next(r['net_pnl'] for r in table_a if r['bucket'] == '10:00-11:00'):+,.2f} / "
        f"${next(r['net_pnl'] for r in table_a if r['bucket'] == '11:00-13:30'):+,.2f}；"
        f"最差时段为 {worst['bucket']}（净 ${worst['net_pnl']:+,.2f}）。"
    )
    bullets.append(
        f"3. **出场结构**：止损（非碰撞）{stop_row['trades']:,} 笔（{stop_share:.1f}%）对 2.0R 止盈 "
        f"{target['trades']:,} 笔（{target_share:.1f}%），止损笔数约为止盈的 "
        f"{(stop_row['trades'] / target['trades']) if target and target['trades'] else 0.0:.2f} 倍；"
        f"平均盈利 ${avg_win:,.2f} / 平均亏损 ${avg_loss:,.2f} = {payoff:.2f} 倍，"
        f"但胜率仅 {_fmt_pct(full_stats['win_rate'])}，盈亏比不足以抵消低胜率，故 PF "
        f"{_fmt_pf(full_stats['profit_factor'])} < 1。"
    )
    bullets.append(
        f"4. **止损宽度/噪音扫损**：止损距离中位数 {sd['median']:.2f}%；"
        f"被止损单中 MFE≥0.5R 占 {thresholds['mfe_ge_0.5r'] * 100.0:.1f}%、"
        f"≥1.2R 占 {thresholds['mfe_ge_1.2r'] * 100.0:.1f}% —— {table_d['judgement']}"
    )
    bullets.append(
        f"5. **保本提示与出场脱节**：1.2R 保本在 V0.1 只有提示、0 笔自动出场；"
        f"{table_d['mfe_counts']['mfe_ge_1.2r']:,} 笔曾达到 1.2R 的仓位最终仍以止损/尾盘了结。"
        "V0.2 若引入规则化保本/移动止损，属结构性改动，需重新做样本外验证。"
    )
    bullets.append(
        "6. **数据口径限制**：以上全部为 Alpaca IEX 单一交易所 1m 口径，成交量/VWAP 非全市场；"
        "结论只能用于 V0.2 结构方向的假设，不能作为策略有效性或未来收益的证据。"
    )
    lines.extend(bullets)
    lines.append("")
    lines.append(
        f"---\n*复现：`python -m intraday_scalper.tools.attribution_report`；"
        f"耗时 {reproduction.get('elapsed_seconds', 0.0):.1f}s（回测部分）；"
        f"数据区间 {bars_meta['data_span']}，{bars_meta['total_bars']:,} 根 1m bar；"
        f"生成于 {reproduction['generated_at_utc']}。*"
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# JSON payload
# --------------------------------------------------------------------------- #
def build_payload(
    *,
    config: Any,
    bars_meta: Dict[str, Any],
    ledger: Sequence[Dict[str, Any]],
    table_a: Sequence[Dict[str, Any]],
    table_b: Dict[str, Any],
    table_c: Dict[str, Any],
    table_d: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    consistency: Dict[str, Any],
    full_stats: Dict[str, Any],
    equity_info: Dict[str, Any],
    breaker_info: Dict[str, Any],
    reproduction: Dict[str, Any],
    top_row: Dict[str, Any],
) -> Dict[str, Any]:
    return _json_safe(
        {
            "report": "V0.1 real-data backtest attribution (task-9, 4 tables)",
            "data_source": "Alpaca IEX 1m bars (single venue; NOT consolidated tape)",
            "data_span": bars_meta["data_span"],
            "total_bars": bars_meta["total_bars"],
            "caveat": CAVEAT,
            "generated_at_utc": reproduction.get("generated_at_utc"),
            "script": "intraday_scalper/tools/attribution_report.py",
            "config_snapshot": {
                "mode": config.mode.value,
                "initial_equity": config.risk.account_equity,
                "max_risk_per_trade": config.risk.max_risk_per_trade,
                "max_notional_per_trade": config.risk.max_notional_per_trade,
                "max_daily_net_loss": config.risk.max_daily_net_loss,
                "max_entries_per_day": config.risk.max_entries_per_day,
                "consecutive_stop_threshold": config.risk.consecutive_stop_threshold,
                "consecutive_stop_cooldown_minutes": config.risk.consecutive_stop_cooldown_minutes,
                "switch_cooldown_minutes": config.risk.switch_cooldown_minutes,
                "est_slippage_bps": config.risk.est_slippage_bps,
                "commission_per_share": config.risk.commission_per_share,
                "min_reward_risk": config.risk.min_reward_risk,
                "no_new_entries_et": config.session.no_new_entries_et.strftime("%H:%M"),
            },
            "reproduction": reproduction,
            "full_period": {
                "stats": full_stats,
                "equity": equity_info,
                "circuit_breaker": breaker_info,
            },
            "segments": segments,
            "table_a_time_of_day": list(table_a),
            "table_b_strategy_symbol": table_b,
            "table_b_soxl_vs_soxs": {
                "soxl": next(row for row in table_b["per_symbol"] if row["symbol"] == "SOXL"),
                "soxs": next(row for row in table_b["per_symbol"] if row["symbol"] == "SOXS"),
                "worst_cross_cell": top_row,
            },
            "table_c_exit_reason": table_c,
            "table_d_stop_diagnostics": table_d,
            "consistency_vs_v01_baseline": consistency,
            "ledger_rows": len(ledger),
            "artifacts": {
                "markdown": f"var/backtest/{ATTRIBUTION_MD}",
                "json": f"var/backtest/{ATTRIBUTION_JSON}",
                "trades_csv": f"var/backtest/{TRADES_CSV}",
            },
        }
    )


# --------------------------------------------------------------------------- #
# CSV read / write
# --------------------------------------------------------------------------- #
def write_trades_csv(path: Path, ledger: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in ledger:
            writer.writerow(
                {
                    "signal_id": row["signal_id"],
                    "symbol": row["symbol"],
                    "strategy": row["strategy"],
                    "mode": row["mode"],
                    "shares": row["shares"],
                    "entry_time_utc": row["entry_time_utc"].isoformat(),
                    "entry_time_et": row["entry_time_et"].isoformat(),
                    "entry_price_signal": round(row["entry_price_signal"], 6),
                    "entry_price_fill": round(row["entry_price_fill"], 6),
                    "stop_price": round(row["stop_price"], 6),
                    "target_price": round(row["target_price"], 6),
                    "exit_time_utc": row["exit_time_utc"].isoformat(),
                    "exit_time_et": row["exit_time_et"].isoformat(),
                    "exit_price_fill": round(row["exit_price_fill"], 6),
                    "pnl": round(row["pnl"], 4),
                    "r_multiple": round(row["r_multiple"], 4),
                    "mae_r": round(row["mae_r"], 4),
                    "mfe_r": round(row["mfe_r"], 4),
                    "stop_distance_pct": round(row["stop_distance_pct"], 4),
                    "exit_reason": row["exit_reason"],
                    "ambiguous": int(row["ambiguous"]),
                    "holding_minutes": round(row["holding_minutes"], 2),
                    "entry_slippage_bps": round(row["entry_slippage_bps"], 3),
                    "mae_dollars": round(row["mae_dollars"], 4),
                    "mfe_dollars": round(row["mfe_dollars"], 4),
                    "risk_dollars": round(row["risk_dollars"], 4),
                    "fees": round(row["fees"], 4),
                    "entry_slippage_dollars": round(row["entry_slippage_dollars"], 4),
                    "exit_slippage_dollars": round(row["exit_slippage_dollars"], 4),
                    "trade_id": row["trade_id"],
                }
            )


def load_ledger_csv(path: Path, initial_equity: float) -> List[Dict[str, Any]]:
    """Re-render support: rebuild the ledger from the enriched trades CSV."""

    ledger: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            entry_utc = datetime.fromisoformat(raw["entry_time_utc"])
            exit_utc = datetime.fromisoformat(raw["exit_time_utc"])
            session_open = session_open_utc(entry_utc)
            ledger.append(
                {
                    "signal_id": raw["signal_id"],
                    "trade_id": int(raw["trade_id"]),
                    "symbol": raw["symbol"],
                    "strategy": raw["strategy"],
                    "mode": raw["mode"],
                    "shares": float(raw["shares"]),
                    "entry_time_utc": entry_utc,
                    "entry_time_et": to_et(entry_utc),
                    "entry_et_date": et_date(entry_utc),
                    "session_minutes": (entry_utc - session_open).total_seconds() / 60.0,
                    "exit_time_utc": exit_utc,
                    "exit_time_et": to_et(exit_utc),
                    "entry_price_signal": float(raw["entry_price_signal"]),
                    "entry_price_fill": float(raw["entry_price_fill"]),
                    "stop_price": float(raw["stop_price"]),
                    "target_price": float(raw["target_price"]),
                    "exit_price_fill": float(raw["exit_price_fill"]),
                    "exit_reason": raw["exit_reason"],
                    "ambiguous": raw["ambiguous"] in ("1", "True", "true"),
                    "holding_minutes": float(raw["holding_minutes"]),
                    "pnl": float(raw["pnl"]),
                    "r_multiple": float(raw["r_multiple"]),
                    "risk_dollars": float(raw["risk_dollars"]),
                    "mae_dollars": float(raw["mae_dollars"]),
                    "mfe_dollars": float(raw["mfe_dollars"]),
                    "mae_r": float(raw["mae_r"]),
                    "mfe_r": float(raw["mfe_r"]),
                    "stop_distance_pct": float(raw["stop_distance_pct"]),
                    "entry_slippage_bps": float(raw["entry_slippage_bps"]),
                    "entry_slippage_dollars": float(raw["entry_slippage_dollars"]),
                    "exit_slippage_dollars": float(raw["exit_slippage_dollars"]),
                    "fees": float(raw["fees"]),
                }
            )
    return ledger


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _run_engine(config: Any, bars: Sequence[Any]) -> Any:
    strategies = build_strategies(config.strategies)
    started = time.perf_counter()
    stop = threading.Event()

    def ticker() -> None:  # pragma: no cover - console progress only
        while not stop.wait(20.0):
            _log(f"回测进行中 … {time.perf_counter() - started:.0f}s")

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        return run_backtest(bars, config, strategies), time.perf_counter() - started
    finally:
        stop.set()
        thread.join(timeout=1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="V0.1 real-data backtest attribution report (tables A-D)."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--baseline", type=Path, default=REPORT_DIR / BASELINE_JSON)
    parser.add_argument("--from-csv", type=Path, default=None, help="re-render from saved trades CSV")
    parser.add_argument("--limit-bars", type=int, default=None, help="debug: only run first N bars")
    parser.add_argument("--dry-run", action="store_true", help="print only; do not write artefacts")
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    baseline_path: Path = args.baseline
    if not baseline_path.exists():
        baseline_path = (output_dir / BASELINE_JSON).resolve()
    baseline: Dict[str, Any] = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        _log(f"⚠️ 未找到基线 {baseline_path}，自洽性检查将跳过。")

    config = load_config()
    initial_equity = float(config.risk.account_equity)
    generated_at = datetime.now(timezone.utc).isoformat()

    _log("=" * 78)
    _log("V0.1 真实回测归因分析（IEX 单交易所口径；只读分析，不改策略/参数）")
    _log("=" * 78)

    if args.from_csv is not None:
        csv_path = args.from_csv
        if not csv_path.exists():
            _log(f"❌ 找不到 {csv_path}")
            return 2
        _log(f"从 {csv_path} 读取逐笔流水（跳过 5 分钟回测复现）…")
        started = time.perf_counter()
        ledger = load_ledger_csv(csv_path, initial_equity)
        _log(f"读取 {len(ledger):,} 笔（{time.perf_counter() - started:.1f}s）")
        result = None
        elapsed = 0.0
        bars_meta = {
            "data_span": baseline.get("data_span", "unknown"),
            "total_bars": baseline.get("total_bars", 0),
        }
        reproduction: Dict[str, Any] = {
            "mode": "ledger_csv",
            "source_csv": str(csv_path),
            "elapsed_seconds": 0.0,
            "generated_at_utc": generated_at,
            "ledger_rows": len(ledger),
        }
    else:
        started = time.perf_counter()
        _log(f"加载 parquet（{args.data_dir}）…")
        bars = load_bars(args.data_dir)
        if args.limit_bars is not None:
            bars = bars[: args.limit_bars]
            _log(f"⚠️ --limit-bars 调试模式：仅使用前 {len(bars):,} 根 bar（自洽性检查将跳过）")
        loading_seconds = time.perf_counter() - started
        data_span = (
            f"{min(bar.start_utc for bar in bars):%Y-%m-%d} .. {max(bar.start_utc for bar in bars):%Y-%m-%d}"
        )
        bars_meta = {"data_span": data_span, "total_bars": len(bars)}
        _log(f"数据：{len(bars):,} 根 1m bar，区间 {data_span}（加载 {loading_seconds:.1f}s）")
        _log("运行 V0.1 全周期回测（固定配置/固定策略，预计约 5 分钟）…")
        result, elapsed = _run_engine(config, bars)
        _log(
            f"回测完成：{result.bars_processed:,} 根 bar、{len(result.trades):,} 笔、"
            f"{elapsed:.1f}s（{(elapsed / 60.0):.2f} 分钟）"
        )

        started_merge = time.perf_counter()
        ledger = build_ledger(result)
        _log(f"合并逐笔流水（trades + trade_details）：{len(ledger):,} 笔（{time.perf_counter() - started_merge:.1f}s）")
        equity_info_engine = equity_stats(result, initial_equity)
        daily = daily_pnl(result.trades, initial_equity)
        reproduction = {
            "mode": "full_engine",
            "elapsed_seconds": round(elapsed, 2),
            "generated_at_utc": generated_at,
            "bars_processed": int(result.bars_processed),
            "skipped_bars": int(result.skipped_bars),
            "ambiguous_bars": int(getattr(result, "ambiguous_bars", 0) or 0),
            "metrics": _fmt_metrics(result.metrics),
            "equity_engine": equity_info_engine,
            "risk_block_counts": dict(result.risk_block_counts or {}),
            "notes_sample": list(result.notes)[:6],
        }
        if args.limit_bars is not None:
            reproduction["limit_bars"] = int(args.limit_bars)

    # -- aggregations ---------------------------------------------------- #
    started_tables = time.perf_counter()
    full_stats = stats(ledger)
    table_a = build_table_a(ledger)
    table_b = build_table_b(ledger)
    table_c = build_table_c(ledger)
    table_d = build_table_d(ledger)
    segments = build_segments(ledger)
    equity_info = trade_derived_equity(ledger, initial_equity)
    breaker_info = trade_derived_circuit_breaker(ledger, config.risk.max_daily_net_loss)
    top_row = min(table_b["cross"], key=lambda row: row["net_pnl"])
    _log(f"四表 + 分段统计完成（{time.perf_counter() - started_tables:.2f}s）")

    # Internal reconciliation: every table must add up to the same trade count.
    partial_segments_total = sum(row["trades"] for row in segments[:-1])
    full_segment_total = segments[-1]["trades"] if segments else 0
    internal_checks = {
        "table_a_total": sum(row["trades"] for row in table_a),
        "table_c_total": table_c["total_trades"],
        "table_c_partition_ok": table_c["partition_ok"],
        "partial_segments_total": partial_segments_total,
        "full_segment_total": full_segment_total,
        "ledger_total": full_stats["trades"],
    }
    internal_checks["all_passed"] = (
        internal_checks["table_a_total"] == full_stats["trades"]
        and internal_checks["table_c_total"] == full_stats["trades"]
        and internal_checks["table_c_partition_ok"]
        and partial_segments_total == full_stats["trades"]
        and full_segment_total == full_stats["trades"]
    )
    if internal_checks["all_passed"]:
        _log(f"内部对账通过：表 A/表 C/分段 合计均 = {full_stats['trades']:,} 笔")
    else:
        _log(f"⚠️ 内部对账失败：{internal_checks}")
    reproduction["internal_checks"] = internal_checks

    if result is not None and "equity_engine" in reproduction:
        reproduction["equity_trade_derived"] = equity_info
        reproduction["metrics_trade_derived"] = {
            "trades": full_stats["trades"],
            "win_rate": round(full_stats["win_rate"], 4),
            "profit_factor": None if math.isinf(full_stats["profit_factor"]) else round(full_stats["profit_factor"], 4),
            "net_pnl": round(full_stats["net_pnl"], 2),
            "avg_r": round(full_stats["avg_r"], 4),
        }

    consistency: Dict[str, Any] = {}
    if baseline and args.limit_bars is None:
        consistency = build_consistency(
            ledger,
            full_stats,
            table_b,
            equity_info,
            breaker_info,
            baseline,
            baseline_path,
            reproduction,
        )
        _log(f"基线自洽性：{consistency['verdict']}")
        failed = [check for check in consistency["checks"] + consistency["per_strategy_checks"] + consistency["per_symbol_checks"] if not check["passed"]]
        for check in failed[:12]:
            _log(
                f"  ❌ {check['metric']}: baseline={check['baseline']} reproduced={check['reproduced']}"
            )
    else:
        consistency = {"skipped": True, "reason": "no baseline or debug limit-bars run", "all_passed": None}
        _log("跳过基线自洽性检查（无基线或 --limit-bars 调试运行）")

    # -- terminal summary ------------------------------------------------- #
    print()
    print("=" * 78)
    print("表 A：美东时段")
    print(md_table(["时段(ET)", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"],
                   [_num_row(row["bucket"], row) for row in table_a]))
    print()
    print("表 B：策略 × 标的（交叉）")
    print(md_table(["策略 × 标的", "笔数", "胜率", "PF", "净盈亏($)", "平均R", "平均持仓(分)"],
                   [_num_row(f"{row['strategy']} × {row['symbol']}", row) for row in table_b["cross"]]))
    print()
    print("表 C：出场原因")
    print(md_table(["出场类型", "笔数", "净盈亏($)", "平均R", "平均持仓(分)"],
                   [[row["category"], str(row["trades"]), _fmt_money(row["net_pnl"]),
                     _fmt_r(row["avg_r"]), _fmt_hold(row["avg_holding_minutes"])] for row in table_c["rows"]]))
    print()
    print("表 D：止损宽度 / MAE-MFE（仅 STOP 单）")
    sd = table_d["stop_distance_pct"]
    print(f"  止损单 {table_d['stops']:,} 笔；止损距离 均值 {sd['mean']:.2f}% / 中位 {sd['median']:.2f}% / P25 {sd['p25']:.2f}% / P75 {sd['p75']:.2f}%")
    print(md_table(["止损距离桶", "笔数", "占止损比", "净盈亏($)", "平均R", "平均MAE(R)", "平均MFE(R)", "MFE≥0.5R占比"],
                   [[row["bucket"], str(row["trades"]), f"{row['share_pct']:.1f}%", _fmt_money(row["net_pnl"]),
                     _fmt_r(row["avg_r"]), _fmt_r(row["avg_mae_r"]), _fmt_r(row["avg_mfe_r"]),
                     f"{row['share_mfe_ge_0_5r']:.1f}%"] for row in table_d["buckets"]]))
    print()
    print(f"表 D 结论：{table_d['conclusion']}")
    print("=" * 78)

    # -- artefacts ------------------------------------------------------- #
    markdown = render_markdown(
        config=config,
        bars_meta=bars_meta,
        ledger=ledger,
        table_a=table_a,
        table_b=table_b,
        table_c=table_c,
        table_d=table_d,
        segments=segments,
        consistency=consistency,
        full_stats=full_stats,
        equity_info=equity_info,
        breaker_info=breaker_info,
        reproduction=reproduction,
        top_row=top_row,
    )
    payload = build_payload(
        config=config,
        bars_meta=bars_meta,
        ledger=ledger,
        table_a=table_a,
        table_b=table_b,
        table_c=table_c,
        table_d=table_d,
        segments=segments,
        consistency=consistency,
        full_stats=full_stats,
        equity_info=equity_info,
        breaker_info=breaker_info,
        reproduction=reproduction,
        top_row=top_row,
    )

    if args.dry_run:
        _log("--dry-run：不写产物。Markdown 预览前 60 行：")
        print("\n".join(markdown.splitlines()[:60]))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / ATTRIBUTION_MD
    json_path = output_dir / ATTRIBUTION_JSON
    csv_path = output_dir / TRADES_CSV
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_trades_csv(csv_path, ledger)
    _log(f"已写入：{md_path}")
    _log(f"已写入：{json_path}")
    _log(f"已写入：{csv_path}（{len(ledger):,} 行）")
    _log("完成 ✅")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
