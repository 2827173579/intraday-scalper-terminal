"""V0.2 structural backtest battle report: V0.1 (1m driven) vs V0.2 (5m driven).

Runs, on the local Alpaca **IEX-only** 1m parquet cache:

1. **Full-period V0.2 backtest** (2024-01-02 .. 2026-11-05) with the production
   ``RiskManager`` ($20 risk / $1,200 notional, -$60 daily circuit breaker,
   15 entries/day, 20-minute consecutive-loss cooldown, SOXL/SOXS mutual
   exclusion, no new entries after 15:30 ET).
2. **Segments** for structural sanity: 2024 (in-sample structure check only,
   never used to tune a parameter) and 2025-01-01..2026-11-05 (holdout).
3. **29-fold walk-forward** with the exact V0.1 grid
   (``walk_forward_splits(dates, 120, 20, 20)``); every train/test run gets a
   fresh risk manager and fresh engine state, no compounding across folds.

Artifacts written under ``intraday_scalper/var/backtest/``::

    v02_report.json    # full-period + segments + walk-forward + V0.1 comparison
    v02_trades.csv     # every V0.2 full-period trade (signal price vs fill price)
    v02_oos.json       # per-fold out-of-sample metrics

The V0.1 baseline is read from ``var/backtest/real_iex_report.json`` when it
exists (it does in this workspace); the comparison is printed side by side.

Data caveat: IEX is a single venue with a small share of US consolidated
volume.  Volume/VWAP/relative-activity are IEX-scoped, so these numbers are a
structural comparison against V0.1 on identical data — not consolidated-tape
performance and not proof of profitability.

Usage::

    python -m intraday_scalper.tools.run_v02_backtest
    python -m intraday_scalper.tools.run_v02_backtest --no-walkforward
    python -m intraday_scalper.tools.run_v02_backtest --train-days 120 --test-days 20 --step-days 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research.v02 import (
    ATR_PERIOD,
    DATA_CAVEAT,
    EMA_FAST,
    EMA_SLOW,
    ENUM_TO_LABEL,
    STRATEGY_LABELS,
    TARGET_R,
    V02Data,
    V02Result,
    V02WalkForwardResult,
    prepare_v02_data,
    run_v02_backtest,
    run_v02_walkforward,
)
from intraday_scalper.tools.run_real_backtest import load_bars

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

TRAIN_2024_START = date(2024, 1, 1)
TRAIN_2024_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 11, 5)

V02_STRUCTURE_NOTE = (
    "V0.2 结构：仅 5m 闭合 bar 驱动（ORB_15M_5M / VWAP_RECLAIM_5M / EMA_PULLBACK_5M）；"
    "禁用 ORB_5M 与所有 1m 触发；镜像共振过滤（做多 SOXL 要求 SOXS close < SOXS session VWAP；"
    "做多 SOXS 要求 SOXL close < SOXL session VWAP，已修正原任务文本笔误）；"
    "IEX 口径下以 ATR range_expansion + 相对成交笔数活跃度替代绝对成交量过滤；"
    "下一 5m bar 开盘成交、1m 级止损/止盈保守碰撞、15:59 ET 强制平仓。"
    f"阈值一次性先验（a-priori, not tuned）：EMA{EMA_FAST}/{EMA_SLOW}、ATR{ATR_PERIOD}、"
    f"target={TARGET_R}R、止损 clamp 0.25%-2.0%。"
)

V02_MIRROR_RULE = (
    "long SOXL requires SOXS close < SOXS session VWAP; "
    "long SOXS requires SOXL close < SOXL session VWAP "
    "(the original V0.2 task text had '>' on the SOXS side; corrected as a spec fix, "
    "not a parameter change)"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _number(value: Any, digits: int = 2) -> float | None:
    """Round and make non-finite floats JSON-safe (``None``)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _pct(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    return _number(float(value) * 100.0, digits)


def _strategy_label(key: str) -> str:
    """Map a production enum value (ORB_15M/...) onto its V0.2 report label."""

    for enum_value, label in ENUM_TO_LABEL.items():
        if enum_value.value == key:
            return label
    return key


def _exit_reason_counts(result: V02Result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in result.trades:
        reason = trade.exit_reason or "UNKNOWN"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def format_result(result: V02Result) -> dict[str, Any]:
    """JSON-safe summary of one V0.2 run, comparable with the V0.1 report format."""
    metrics = result.metrics
    per_strategy = {
        _strategy_label(key): {
            "trades": group.trades,
            "win_rate_pct": _pct(group.win_rate),
            "profit_factor": _number(group.profit_factor, 3),
            "net_pnl_dollars": _number(group.net_pnl_dollars),
            "avg_r": _number(group.avg_r, 3),
            "trades_share_pct": _pct(group.trades / metrics.total_trades)
            if metrics.total_trades
            else 0.0,
        }
        for key, group in metrics.per_strategy.items()
    }
    per_symbol = {
        symbol: {
            "trades": group.trades,
            "win_rate_pct": _pct(group.win_rate),
            "profit_factor": _number(group.profit_factor, 3),
            "net_pnl_dollars": _number(group.net_pnl_dollars),
            "avg_r": _number(group.avg_r, 3),
        }
        for symbol, group in metrics.per_symbol.items()
    }
    return {
        "metrics": {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate_pct": _pct(metrics.win_rate),
            "profit_factor": _number(metrics.profit_factor, 3),
            "net_pnl_dollars": _number(metrics.net_pnl_dollars),
            "expectancy_dollars": _number(metrics.expectancy_dollars),
            "expectancy_r": _number(metrics.expectancy_r, 3),
            "avg_r": _number(metrics.avg_r, 3),
            "avg_holding_minutes": _number(metrics.avg_holding_minutes, 1),
            "max_drawdown_dollars": _number(metrics.max_drawdown_dollars),
            "max_drawdown_pct": _pct(metrics.max_drawdown_pct),
            "trades_per_day": _number(metrics.trades_per_day, 2),
            "trading_days": metrics.trading_days,
            "total_slippage_dollars": _number(metrics.total_slippage_dollars),
            "total_fees_dollars": _number(metrics.total_fees_dollars),
            "initial_equity": _number(metrics.initial_equity),
            "final_equity": _number(metrics.final_equity),
        },
        "extra": result.extra_metrics(),
        "exit_reasons": _exit_reason_counts(result),
        "per_strategy": per_strategy,
        "per_symbol": per_symbol,
        "time_of_day": {
            bucket: {
                "trades": value.get("trades", 0),
                "win_rate_pct": _pct(value.get("win_rate", 0.0)),
                "net_pnl_dollars": _number(value.get("net_pnl_dollars", 0.0)),
            }
            for bucket, value in metrics.time_of_day_performance.items()
        },
        "notes": list(result.notes)[:12],
    }


def format_walkforward(wf: V02WalkForwardResult) -> dict[str, Any]:
    """JSON-safe walk-forward payload with per-fold and aggregate OOS metrics."""
    fold_rows = []
    for fold_result in wf.folds:
        test_metrics = fold_result.test_result.metrics
        fold_rows.append(
            {
                "fold": fold_result.fold.index,
                "train_start": fold_result.fold.train_start.isoformat()
                if fold_result.fold.train_start
                else None,
                "train_end": fold_result.fold.train_end.isoformat()
                if fold_result.fold.train_end
                else None,
                "test_start": fold_result.fold.test_start.isoformat()
                if fold_result.fold.test_start
                else None,
                "test_end": fold_result.fold.test_end.isoformat()
                if fold_result.fold.test_end
                else None,
                "train_trades": fold_result.train_result.total_trades,
                "test_trades": test_metrics.total_trades,
                "test_win_rate_pct": _pct(test_metrics.win_rate),
                "test_profit_factor": _number(test_metrics.profit_factor, 3),
                "test_net_pnl_dollars": _number(test_metrics.net_pnl_dollars),
                "test_avg_r": _number(test_metrics.avg_r, 3),
                "test_max_drawdown_dollars": _number(test_metrics.max_drawdown_dollars),
                "test_breaker_days": fold_result.test_result.breaker_days,
                "leakage_checked": fold_result.leakage_checked,
            }
        )
    aggregate = wf.aggregate_test_metrics
    summary = dict(wf.summary)
    summary["test_win_rate_pct"] = _pct(summary.get("test_win_rate", 0.0))
    summary["test_profit_factor"] = _number(summary.get("test_profit_factor"), 3)
    summary["test_net_pnl_dollars"] = _number(summary.get("test_net_pnl_dollars"))
    summary["test_avg_r"] = _number(summary.get("test_avg_r"), 3)
    summary["test_max_drawdown_dollars_max_fold"] = _number(
        summary.get("test_max_drawdown_dollars_max_fold")
    )
    summary["test_max_drawdown_pct_max_fold"] = _pct(
        summary.get("test_max_drawdown_pct_max_fold")
    )
    summary["test_trades_per_session"] = _number(
        summary.get("test_trades_per_session"), 2
    )
    return {
        "fold_rows": fold_rows,
        "summary": summary,
        "aggregate_test": {
            "total_trades": aggregate.total_trades,
            "wins": aggregate.wins,
            "losses": aggregate.losses,
            "win_rate_pct": _pct(aggregate.win_rate),
            "profit_factor": _number(aggregate.profit_factor, 3),
            "net_pnl_dollars": _number(aggregate.net_pnl_dollars),
            "avg_r": _number(aggregate.avg_r, 3),
            "max_drawdown_dollars": _number(aggregate.max_drawdown_dollars),
            "trades_per_day": _number(aggregate.trades_per_day, 2),
            "trading_days": aggregate.trading_days,
            "total_slippage_dollars": _number(aggregate.total_slippage_dollars),
            "max_drawdown_note": (
                "aggregate max_drawdown is computed on the concatenated (non-compounded) "
                "test-trade equity path; the V0.1-vs-V0.2 comparison uses the per-fold "
                "max drawdown maximum instead (see summary.test_max_drawdown_dollars_max_fold)"
            ),
        },
        "notes": list(wf.notes),
    }


# --------------------------------------------------------------------------- #
# V0.1 baseline
# --------------------------------------------------------------------------- #
def load_v01_baseline(report_dir: Path = REPORT_DIR) -> dict[str, Any] | None:
    """Read the V0.1 battle report for the side-by-side comparison (if present)."""
    path = report_dir / "real_iex_report.json"
    if not path.exists():
        print(f"[WARN] 未找到 V0.1 基线报告 {path}；对比列将显示 —")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = payload.get("full_period", {})
    wf = payload.get("walkforward") or {}
    fold_rows = wf.get("fold_results") or []
    per_fold_dd = [
        row.get("test", {}).get("max_drawdown_dollars")
        for row in fold_rows
        if row.get("test", {}).get("max_drawdown_dollars") is not None
    ]
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "data_span": payload.get("data_span"),
        "total_bars": payload.get("total_bars"),
        "full_period": {
            "metrics": full.get("metrics", {}),
            "circuit_breaker": full.get("circuit_breaker", {}),
            "ambiguous_bars": full.get("ambiguous_bars"),
            "elapsed_seconds": full.get("elapsed_seconds"),
        },
        "walkforward": {
            "folds": wf.get("folds"),
            "positive_folds": wf.get("positive_folds"),
            "best_fold_pnl": wf.get("best_fold_pnl"),
            "worst_fold_pnl": wf.get("worst_fold_pnl"),
            "aggregate_test": wf.get("aggregate_test", {}),
            "max_fold_drawdown_dollars": max(
                (value for value in per_fold_dd if value is not None), default=None
            ),
            "elapsed_seconds": wf.get("elapsed_seconds"),
        },
    }


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #
_COMPARISON_KEYS: tuple[str, ...] = (
    "total_trades",
    "win_rate_pct",
    "profit_factor",
    "net_pnl_dollars",
    "avg_r",
    "max_drawdown_dollars",
    "trades_per_day",
)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return number if math.isfinite(number) else None


def _delta(v02: Any, v01: Any) -> float | None:
    a = _numeric(v01)
    b = _numeric(v02)
    if a is None or b is None:
        return None
    return round(b - a, 6)


def build_comparison(
    v01: dict[str, Any] | None,
    full_payload: dict[str, Any],
    wf_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Explicit V0.1-vs-V0.2 delta block (same metric keys for both sides)."""

    v01_full = ((v01 or {}).get("full_period") or {}).get("metrics") or None
    v02_full = full_payload["metrics"]
    comparison: dict[str, Any] = {
        "max_drawdown_note": (
            "WF 最大回撤使用「单折最大回撤的最大值」口径（V0.1 也由其逐折 test 指标重算）；"
            "全周期最大回撤为整段已实现权益曲线口径。"
        ),
        "full_period": {
            "v01": {key: (v01_full or {}).get(key) for key in _COMPARISON_KEYS},
            "v02": {key: v02_full.get(key) for key in _COMPARISON_KEYS},
            "delta_v02_minus_v01": {
                key: _delta(v02_full.get(key), (v01_full or {}).get(key))
                for key in _COMPARISON_KEYS
            },
        },
    }
    if wf_payload:
        v01_wf = ((v01 or {}).get("walkforward") or {})
        v01_wf_agg = dict(v01_wf.get("aggregate_test") or {})
        if v01_wf.get("max_fold_drawdown_dollars") is not None:
            v01_wf_agg["max_drawdown_dollars"] = v01_wf["max_fold_drawdown_dollars"]
        v02_wf_agg = dict(wf_payload["aggregate_test"])
        if wf_payload["summary"].get("test_max_drawdown_dollars_max_fold") is not None:
            v02_wf_agg["max_drawdown_dollars"] = wf_payload["summary"][
                "test_max_drawdown_dollars_max_fold"
            ]
        comparison["walkforward"] = {
            "v01": {key: v01_wf_agg.get(key) for key in _COMPARISON_KEYS},
            "v02": {key: v02_wf_agg.get(key) for key in _COMPARISON_KEYS},
            "delta_v02_minus_v01": {
                key: _delta(v02_wf_agg.get(key), v01_wf_agg.get(key))
                for key in _COMPARISON_KEYS
            },
            "folds": {
                "v01": v01_wf.get("folds"),
                "v02": wf_payload["summary"].get("folds"),
            },
            "positive_folds": {
                "v01": v01_wf.get("positive_folds"),
                "v02": wf_payload["summary"].get("positive_folds"),
            },
            "best_fold_pnl": {
                "v01": v01_wf.get("best_fold_pnl"),
                "v02": wf_payload["summary"].get("best_fold_pnl"),
            },
            "worst_fold_pnl": {
                "v01": v01_wf.get("worst_fold_pnl"),
                "v02": wf_payload["summary"].get("worst_fold_pnl"),
            },
        }
    comparison["v02_full_extra"] = full_payload.get("extra")
    return comparison


def _fmt_span(dates: Sequence[date]) -> str:
    if not dates:
        return "—"
    return f"{min(dates):%Y-%m-%d} .. {max(dates):%Y-%m-%d}"


def run_full(
    data: V02Data,
    config,
    *,
    initial_equity: float,
) -> tuple[V02Result, dict[str, Any]]:
    started = time.perf_counter()
    result = run_v02_backtest(data, config, initial_equity=initial_equity)
    elapsed = time.perf_counter() - started
    result.elapsed_seconds = elapsed
    formatted = format_result(result)
    formatted["elapsed_seconds"] = round(elapsed, 2)
    formatted["session_span"] = _fmt_span(result.session_dates)
    return result, formatted


def run_segments(
    data: V02Data,
    config,
    *,
    initial_equity: float,
) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    plans = {
        "train_2024": (TRAIN_2024_START, TRAIN_2024_END),
        "holdout_2025_2026": (HOLDOUT_START, HOLDOUT_END),
    }
    for name, (start, end) in plans.items():
        dates = [value for value in data.dates if start <= value <= end]
        if not dates:
            segments[name] = {"metrics": {"total_trades": 0}, "session_dates": 0}
            continue
        started = time.perf_counter()
        result = run_v02_backtest(
            data, config, initial_equity=initial_equity, session_dates=dates
        )
        elapsed = time.perf_counter() - started
        payload = format_result(result)
        payload["elapsed_seconds"] = round(elapsed, 2)
        payload["session_span"] = _fmt_span(dates)
        payload["session_dates"] = len(dates)
        segments[name] = payload
    return segments


def run_walkforward_report(
    data: V02Data,
    config,
    *,
    initial_equity: float,
    train_days: int,
    test_days: int,
    step_days: int,
) -> tuple[V02WalkForwardResult, dict[str, Any]]:
    started = time.perf_counter()
    wf = run_v02_walkforward(
        data,
        config,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_equity=initial_equity,
        progress=lambda index, total: print(
            f"  walk-forward 折 {index}/{total} 完成", flush=True
        ),
    )
    elapsed = time.perf_counter() - started
    payload = format_walkforward(wf)
    payload["elapsed_seconds"] = round(elapsed, 2)
    return wf, payload


# --------------------------------------------------------------------------- #
# Trade ledger
# --------------------------------------------------------------------------- #
def _write_trades_csv(result: V02Result, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for trade in result.trades:
        detail = result.trade_details.get(trade.signal_id, {})
        rows.append(
            {
                "signal_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": trade.strategy.value,
                "strategy_label": detail.get("strategy_label", trade.strategy.value),
                "mode": trade.mode.value,
                "shares": trade.shares,
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat()
                if trade.exit_time_utc
                else "",
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason or "",
                "fees": trade.fees,
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                "mae_dollars": round(float(detail.get("mae_dollars", 0.0) or 0.0), 4),
                "mfe_dollars": round(float(detail.get("mfe_dollars", 0.0) or 0.0), 4),
                "ambiguous_bar": bool(detail.get("ambiguous_bar", False)),
                "signal_bar_start_utc": detail.get("signal_bar_start_utc", ""),
                "entry_bucket_start_utc": detail.get("entry_bucket_start_utc", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Report printing
# --------------------------------------------------------------------------- #
_COMPARISON_ROWS: tuple[tuple[str, str], ...] = (
    ("交易笔数", "total_trades"),
    ("胜率", "win_rate_pct"),
    ("ProfitFactor", "profit_factor"),
    ("净盈亏($)", "net_pnl_dollars"),
    ("平均R", "avg_r"),
    ("最大回撤($)", "max_drawdown_dollars"),
    ("日均笔数", "trades_per_day"),
)


def _fmt_cell(row: str, value: Any) -> str:
    if value is None:
        return "—"
    if row == "胜率":
        return f"{value:.2f}%"
    if row == "ProfitFactor":
        return f"{value:.3f}"
    if row == "净盈亏($)":
        return f"{value:+,.2f}"
    if row == "最大回撤($)":
        return f"{value:,.2f}"
    if row == "平均R":
        return f"{value:+.3f}"
    if row == "日均笔数":
        return f"{value:.2f}"
    return f"{int(value)}"


def print_comparison_table(
    title: str,
    v01_metrics: dict[str, Any] | None,
    v02_metrics: dict[str, Any],
    *,
    v01_max_dd: Any = None,
    v02_max_dd: Any = None,
) -> None:
    print(f"\n{title}")
    print(f"  {'指标':<14}{'V0.1 (1m 驱动)':>18}{'V0.2 (5m 驱动)':>18}{'差异':>18}")
    print(f"  {'-' * 66}")
    for row, key in _COMPARISON_ROWS:
        v01_value = None
        if v01_metrics:
            if row == "最大回撤($)":
                v01_value = v01_max_dd
            else:
                v01_value = v01_metrics.get(key)
        v02_value = v02_max_dd if row == "最大回撤($)" else v02_metrics.get(key)
        if v01_value is None or v02_value is None:
            diff = "—"
        elif row == "胜率":
            diff = f"{v02_value - v01_value:+.2f}pp"
        elif row == "ProfitFactor":
            diff = f"{v02_value - v01_value:+.3f}"
        elif row in {"净盈亏($)", "最大回撤($)"}:
            diff = f"{v02_value - v01_value:+,.2f}"
        elif row == "平均R":
            diff = f"{v02_value - v01_value:+.3f}"
        elif row == "日均笔数":
            diff = f"{v02_value - v01_value:+.2f}"
        else:
            diff = f"{int(v02_value) - int(v01_value):+d}"
        print(
            f"  {row:<14}{_fmt_cell(row, v01_value):>18}"
            f"{_fmt_cell(row, v02_value):>18}{diff:>18}"
        )


def print_report(
    *,
    payload: dict[str, Any],
    data: V02Data,
    total_1m_bars: int,
    elapsed: dict[str, float],
    v01: dict[str, Any] | None,
) -> None:
    full = payload["v02_full_period"]
    metrics = full["metrics"]
    v01_full = (v01 or {}).get("full_period", {}).get("metrics")
    print("\n" + "=" * 78)
    print("V0.2 结构重构战报 · 5m 驱动 + 镜像共振过滤 + IEX 口径适配")
    print("=" * 78)
    print(f"数据区间    : {full.get('session_span', '—')}")
    print(f"数据量      : {total_1m_bars:,} 根 1m bar → {data.total_5m_bars:,} 根 5m bar；"
          f"session {len(data.sessions)}；空 5m bucket {data.skipped_buckets:,}")
    print(f"结构与先验  : {V02_STRUCTURE_NOTE}")
    print(f"口径警告    : {DATA_CAVEAT}")

    print_comparison_table(
        "[一、全周期对比 2024-01-02 .. 2026-11-05（V0.1 基线来自 real_iex_report.json）]",
        v01_full,
        metrics,
        v01_max_dd=(v01 or {}).get("full_period", {}).get("metrics", {}).get("max_drawdown_dollars"),
        v02_max_dd=metrics.get("max_drawdown_dollars"),
    )
    print(
        f"  平均持仓 {metrics.get('avg_holding_minutes', 0.0):.1f} 分钟；"
        f"总滑点 ${metrics.get('total_slippage_dollars') or 0.0:,.2f}；"
        f"V0.2 每 session {full['extra'].get('trades_per_session', 0.0):.2f} 笔"
    )

    print("\n[二、V0.2 分段结构验证]")
    print(f"  {'区间':<26}{'笔数':>8}{'胜率':>10}{'PF':>8}{'净盈亏($)':>14}{'平均R':>10}{'最大回撤($)':>14}")
    print(f"  {'-' * 88}")
    for name, label in (
        ("train_2024", "2024 训练段 (in-sample)"),
        ("holdout_2025_2026", "2025-01-01..2026-11-05 holdout"),
    ):
        segment = payload["v02_segments"].get(name)
        if not segment:
            continue
        seg_metrics = segment["metrics"]
        print(
            f"  {label:<26}{int(seg_metrics.get('total_trades') or 0):>8}"
            f"{_fmt_cell('胜率', seg_metrics.get('win_rate_pct')):>10}"
            f"{_fmt_cell('ProfitFactor', seg_metrics.get('profit_factor')):>8}"
            f"{_fmt_cell('净盈亏($)', seg_metrics.get('net_pnl_dollars')):>14}"
            f"{_fmt_cell('平均R', seg_metrics.get('avg_r')):>10}"
            f"{_fmt_cell('最大回撤($)', seg_metrics.get('max_drawdown_dollars')):>14}"
        )
    print("  说明：2024 段只做结构验证，未用于任何调参；holdout 才是后段样本外近似。")

    print("\n[三、V0.2 全周期分策略]")
    for label in STRATEGY_LABELS:
        stats = full["per_strategy"].get(label)
        if not stats:
            print(f"  {label:<16} 0 笔")
            continue
        print(
            f"  {label:<16} 笔数 {stats['trades']:>4}  胜率 {stats['win_rate_pct']:>6.2f}%  "
            f"PF {_fmt_cell('ProfitFactor', stats['profit_factor']):>6}  "
            f"净盈亏 ${_fmt_cell('净盈亏($)', stats['net_pnl_dollars'])}  平均R "
            f"{_fmt_cell('平均R', stats['avg_r'])}"
        )

    print("\n[四、V0.2 全周期分标的]")
    for symbol, stats in full["per_symbol"].items():
        print(
            f"  {symbol:<6} 笔数 {stats['trades']:>4}  胜率 {stats['win_rate_pct']:>6.2f}%  "
            f"净盈亏 ${_fmt_cell('净盈亏($)', stats['net_pnl_dollars'])}  平均R "
            f"{_fmt_cell('平均R', stats['avg_r'])}"
        )

    print("\n[五、V0.2 风控 / 执行统计]")
    extra = full["extra"]
    print(f"  单日 -$60 熔断触发天数   : {extra.get('breaker_days', 0)}")
    print(f"  同 bar TP/SL 碰撞(按止损) : {extra.get('ambiguous_bars', 0)}")
    print(f"  产生候选信号数           : {extra.get('signals', 0)}")
    print(f"  pending 取消数           : {extra.get('pending_cancelled', 0)}")
    reasons = full.get("exit_reasons", {})
    print("  出场原因分布             : " + "  ".join(f"{k}={v}" for k, v in reasons.items()))
    blocks = extra.get("risk_block_counts") or {}
    print("  风控拒单分布             : " + "  ".join(f"{k}={v}" for k, v in sorted(blocks.items())))

    wf_payload = payload.get("walkforward")
    if wf_payload:
        v01_wf = (v01 or {}).get("walkforward", {}).get("aggregate_test")
        v02_wf = wf_payload["aggregate_test"]
        print_comparison_table(
            f"[六、Walk-Forward 样本外对比（{wf_payload['summary'].get('folds', 0)} 折，"
            f"train {wf_payload['summary'].get('train_days')}d / test {wf_payload['summary'].get('test_days')}d，"
            "固定参数、不复利）]",
            v01_wf,
            v02_wf,
            v01_max_dd=(v01 or {}).get("walkforward", {}).get("max_fold_drawdown_dollars"),
            v02_max_dd=wf_payload["summary"].get("test_max_drawdown_dollars_max_fold"),
        )
        print(
            f"  盈利折数 V0.1 {int((v01 or {}).get('walkforward', {}).get('positive_folds') or 0)}"
            f"/{int((v01 or {}).get('walkforward', {}).get('folds') or 0)}  vs  "
            f"V0.2 {int(wf_payload['summary'].get('positive_folds') or 0)}"
            f"/{int(wf_payload['summary'].get('folds') or 0)}"
            f"；V0.2 最好折 ${_fmt_cell('净盈亏($)', wf_payload['summary'].get('best_fold_pnl'))}"
            f"，最差折 ${_fmt_cell('净盈亏($)', wf_payload['summary'].get('worst_fold_pnl'))}"
        )
        print("  说明：最大回撤行 WF 采用「单折最大回撤的最大值」口径。")

    print("\n[七、结论摘要]")
    if v01_full:
        delta_trades = metrics["total_trades"] - int(v01_full.get("total_trades") or 0)
        delta_pf = (metrics.get("profit_factor") or 0.0) - float(
            v01_full.get("profit_factor") or 0.0
        )
        delta_net = (metrics.get("net_pnl_dollars") or 0.0) - float(
            v01_full.get("net_pnl_dollars") or 0.0
        )
        print(
            f"  全周期：笔数 {delta_trades:+d}；PF {delta_pf:+.3f}；净盈亏 {delta_net:+,.2f}"
            f"（V0.2 仍{'未' if (metrics.get('net_pnl_dollars') or 0.0) < 0 else ''}盈利）"
        )
    print(
        f"  运行时长：全周期 {elapsed.get('full', 0.0):.1f}s；"
        f"WF {elapsed.get('walkforward', 0.0):.1f}s（目标 ≤180s / ≤900s）"
    )
    print("  结论必须与 IEX 单一交易所口径一起阅读；V0.2 是结构性对比研究，不是实盘有效性证明。")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_and_run(
    *,
    data_dir: Path = DATA_DIR,
    report_dir: Path = REPORT_DIR,
    initial_equity: Optional[float] = None,
    with_walkforward: bool = True,
    with_segments: bool = True,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
) -> dict[str, Any]:
    """Run the full V0.2 program and write all artifacts.  Returns the payload."""
    from intraday_scalper.core import load_config

    config = load_config()
    equity = float(config.risk.account_equity if initial_equity is None else initial_equity)
    started = time.perf_counter()

    print(f"加载 1m bar：{data_dir}")
    bars = load_bars(data_dir)
    total_1m_bars = len(bars)
    prepared_started = time.perf_counter()
    data = prepare_v02_data(bars)
    prepare_elapsed = time.perf_counter() - prepared_started
    print(
        f"5m 重采样完成：{data.total_5m_bars:,} 根 5m bar，"
        f"{len(data.sessions)} 个 session，{data.skipped_buckets:,} 个空 bucket "
        f"({prepare_elapsed:.1f}s)"
    )

    print("\n开始 V0.2 全周期回测…", flush=True)
    full_result, full_payload = run_full(data, config, initial_equity=equity)
    print(
        f"全周期完成：{full_payload['metrics']['total_trades']} 笔，"
        f"{full_payload['elapsed_seconds']:.1f}s",
        flush=True,
    )

    segments: dict[str, Any] = {}
    if with_segments:
        print("开始分段（2024 训练段 / 2025-2026 holdout）…", flush=True)
        segments = run_segments(data, config, initial_equity=equity)
        for name, segment in segments.items():
            print(
                f"  {name}: {segment['metrics'].get('total_trades', 0)} 笔，"
                f"{segment['metrics'].get('net_pnl_dollars')} 美元",
                flush=True,
            )

    wf: dict[str, Any] | None = None
    wf_elapsed = 0.0
    if with_walkforward:
        print(
            f"\n开始 walk-forward（train {train_days}d / test {test_days}d / "
            f"step {step_days}d，共 29 折量级）…",
            flush=True,
        )
        _, wf = run_walkforward_report(
            data,
            config,
            initial_equity=equity,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )
        wf_elapsed = float(wf["elapsed_seconds"])
        print(
            f"walk-forward 完成：{wf['summary'].get('folds')} 折，"
            f"样本外 {wf['aggregate_test'].get('total_trades')} 笔，{wf_elapsed:.1f}s",
            flush=True,
        )

    v01 = load_v01_baseline(report_dir)
    elapsed = {
        "full": float(full_payload["elapsed_seconds"]),
        "walkforward": wf_elapsed,
        "total": time.perf_counter() - started,
    }
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.2",
        "data_source": "Alpaca IEX 1m bars (single venue; NOT consolidated tape)",
        "data_span": f"{min(data.dates):%Y-%m-%d} .. {max(data.dates):%Y-%m-%d}",
        "total_1m_bars": total_1m_bars,
        "total_sessions": len(data.sessions),
        "prepared": {
            "total_5m_bars": data.total_5m_bars,
            "skipped_5m_buckets": data.skipped_buckets,
            "symbols": list(data.symbols),
        },
        "structure": {
            "strategies": list(STRATEGY_LABELS),
            "ema_fast": EMA_FAST,
            "ema_slow": EMA_SLOW,
            "atr_period": ATR_PERIOD,
            "target_r": TARGET_R,
            "mirror_rule": V02_MIRROR_RULE,
            "note": V02_STRUCTURE_NOTE,
        },
        "caveat": DATA_CAVEAT,
        "v02_full_period": full_payload,
        "v02_segments": segments,
        "walkforward": wf,
        "v01_baseline": v01,
        "comparison": build_comparison(v01, full_payload, wf),
        "elapsed_seconds": elapsed,
        "v02_trades": len(full_result.trades),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v02_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    trades_path = report_dir / "v02_trades.csv"
    _write_trades_csv(full_result, trades_path)
    oos_path = report_dir / "v02_oos.json"
    oos_payload = {
        "generated_at_utc": payload["generated_at_utc"],
        "version": "V0.2",
        "data_source": payload["data_source"],
        "caveat": DATA_CAVEAT,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "walkforward": wf,
        "v01_walkforward": (v01 or {}).get("walkforward"),
        "comparison": payload["comparison"].get("walkforward"),
    }
    oos_path.write_text(
        json.dumps(oos_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print_report(
        payload=payload,
        data=data,
        total_1m_bars=total_1m_bars,
        elapsed=elapsed,
        v01=v01,
    )
    print(f"\n完整报告 : {report_path}")
    print(f"逐笔流水 : {trades_path}")
    print(f"样本外   : {oos_path}")
    print(f"总运行时长: {elapsed['total']:.1f}s")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass

    parser = argparse.ArgumentParser(
        description="V0.2 structural backtest (5m driven) vs V0.1 (1m driven) on real IEX 1m bars."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--no-walkforward", action="store_true")
    parser.add_argument("--no-segments", action="store_true")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    args = parser.parse_args(argv)

    build_and_run(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        initial_equity=args.initial_equity,
        with_walkforward=not args.no_walkforward,
        with_segments=not args.no_segments,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
