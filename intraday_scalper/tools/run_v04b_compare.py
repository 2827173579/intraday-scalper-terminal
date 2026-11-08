"""V0.4b comparison: Step 1 vs Step 3 vs Step 3b + Friday 2026-11-06 recap.

Step 3b adds two **fixed, a-priori** filters on top of Step 3 (10:00-14:00 ET
entry window, fractional shares, VWAP_RECLAIM_5M trend filter, EMA_PULLBACK_5M,
SMC_SWEEP_5M, no 1.2R scale-out):

1. **EMA_PULLBACK daily-exhaustion guard**: block the EMA_PULLBACK long signal
   when ``(close_5m - session_low_so_far) / session_low_so_far >= 8.0%``
   (session low = min completed 5m low since 09:30 ET, causal).
2. **SMC right-side confirmation**: the SMC sweep is only bought when
   ``close_5m >= EMA21_5m`` **or** ``close_5m >= session VWAP`` (rejects
   underwater knife-catching where the close is below both).

This tool runs the three variants on the raw (unadjusted) IEX backtest dataset
(2024-01-02 .. 2026-11-05, 483,751 bars, 714 sessions) — full period, 2024
in-sample, 2025-2026 holdout and the same 29-fold walk-forward grid
(120/20/20, fresh ``RiskManager`` per fold, no compounding) — and then replays
Friday 2026-11-06 for all three variants with per-trade detail and per-variant
markers for the Web chart.

Step 3 must reproduce the task-12 record exactly (804 trades / +$9.28 /
PF 1.0013 / WF 676 / -$250.63 / 15 folds); the tool self-checks this and aborts
the report with a loud failure marker if it does not.

Artifacts (``intraday_scalper/var/backtest/``)::

    v04b_report.json          # 3-variant compare + Step3 self-check + Friday summary
    v04b_compare.md           # markdown compare table + Friday detail (REVIEW_PACK)
    v04b_trades.csv           # Step 3b full-period aggregated trade ledger
    v04b_friday_recap.json    # Friday recap (versions: Step1/Step3/Step3b)
    v04b_friday_recap.md      # Friday markdown recap
    v04b_friday_markers.json  # {"markers_by_variant": {"Step1": [...], ...}}

No parameter tuning: every threshold is fixed by the task.  Data caveat: Alpaca
IEX-only single venue, not consolidated tape; research output only.

Usage::

    python -m intraday_scalper.tools.run_v04b_compare
    python -m intraday_scalper.tools.run_v04b_compare --no-walkforward
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from intraday_scalper.core import Bar, et_date, load_config
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v02, v04
from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    EMA_PULLBACK_5M,
    VWAP_RECLAIM_5M,
    build_strategy_view,
    prepare_v02_data,
)
from intraday_scalper.research.v04 import (
    EMA_EXHAUSTION_GAIN_LIMIT,
    SMC_SWEEP_5M,
    build_eval_context,
    ema_pullback_exhaustion_blocked,
    smc_trend_confirm_ok,
)
from intraday_scalper.tools.run_v03_raw_crosscheck import load_raw_bars
from intraday_scalper.tools.run_v04_backtest import (
    FRIDAY_DATE,
    _dt_et,
    _fmt,
    _format_v04_result,
    _format_walkforward,
    _market_snapshot,
    _metrics_row,
    _number,
    _pct,
    _summary_from_metrics,
    load_friday_raw_bars,
    write_trades_csv,
)

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

TRAIN_2024_START = date(2024, 1, 1)
TRAIN_2024_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 11, 5)

#: The three compared variants (order = report/terminal order).
COMPARE_VARIANTS: tuple[tuple[str, v04.V04Variant], ...] = (
    ("Step1", v04.STEP1_VARIANT),
    ("Step3", v04.STEP3_VARIANT),
    ("Step3b", v04.STEP3B_VARIANT),
)

#: Structure description for this comparison (Step 4 scale-out is deprecated).
V04B_STRUCTURE_NOTE = (
    "V0.4b 在 V0.4 结构基础上对比三个变体："
    "Step1 = 10:00-14:00 ET 入场窗 + fractional + VWAP_RECLAIM_5M/EMA_PULLBACK_5M + "
    "V0.3 1.2R 单止损保本盾；Step3 = Step1 + VWAP_RECLAIM 趋势过滤 + SMC_SWEEP_5M（3 策略）；"
    "Step3b = Step3 + 两项固定先验过滤：① EMA_PULLBACK 日内涨幅 >= 8.0%（相对 09:30 起已完成 "
    "5m low）禁止追高；② SMC 要求 close_5m >= EMA21 或 close_5m >= session VWAP（禁止水下接刀）。"
    "Step4（1.2R 分批止盈）在 task-12 中为负贡献，仅存档、不参与本对比。"
    "29 折 WF 纪律（120/20/20、每折全新 RiskManager、不复利）与 V0.2/V0.3/V0.4 完全一致。"
)

#: Task-12 recorded Step 3 numbers (must be reproduced exactly by this tool).
STEP3_RECORDED: Dict[str, Any] = {
    "full": {
        "total_trades": 804,
        "win_rate_pct": 43.53,
        "profit_factor": 1.0013,
        "net_pnl_dollars": 9.28,
        "avg_r": 0.0015,
        "max_drawdown_dollars": 620.85,
    },
    "train_2024": {"total_trades": 265, "profit_factor": 0.8468, "net_pnl_dollars": -376.25},
    "holdout": {"total_trades": 539, "profit_factor": 1.0816, "net_pnl_dollars": 385.53},
    "walkforward": {
        "total_trades": 676,
        "win_rate_pct": 43.05,
        "profit_factor": 0.9594,
        "net_pnl_dollars": -250.63,
        "positive_folds": 15,
        "folds": 29,
    },
}

#: The two Friday "traps" the Lead asked to verify.  Signals are matched by the
#: Step-3 trade key (symbol, strategy, entry_time_utc).
FRIDAY_PITFALLS: tuple[tuple[str, Dict[str, str]], ...] = (
    ("11:25 SOXL SMC 接刀", {"symbol": "SOXL", "strategy": SMC_SWEEP_5M, "entry_time_utc": "2026-11-06T16:25:00+00:00"}),
    ("12:40 SOXS EMA 山顶追高", {"symbol": "SOXS", "strategy": EMA_PULLBACK_5M, "entry_time_utc": "2026-11-06T17:40:00+00:00"}),
)


# --------------------------------------------------------------------------- #
# Backtest comparison
# --------------------------------------------------------------------------- #
def _fmt_span(dates: Sequence[date]) -> str:
    return f"{min(dates):%Y-%m-%d} .. {max(dates):%Y-%m-%d}" if dates else "—"


def run_compare(
    data: v02.V02Data,
    config: Any,
    *,
    with_segments: bool = True,
    with_walkforward: bool = True,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
    initial_equity: float = 3000.0,
) -> Dict[str, Dict[str, Any]]:
    """Run Step1/Step3/Step3b full period + segments + walk-forward."""

    dates = list(data.dates)
    dates_2024 = [value for value in dates if TRAIN_2024_START <= value <= TRAIN_2024_END]
    dates_holdout = [value for value in dates if HOLDOUT_START <= value <= HOLDOUT_END]
    results: Dict[str, Dict[str, Any]] = {}
    for label, variant in COMPARE_VARIANTS:
        print(f"\n[{label}] {variant.name} …", flush=True)
        started = time.perf_counter()
        full = v04.run_v04_backtest(
            data, config, variant=variant, initial_equity=initial_equity
        )
        seg_2024 = (
            v04.run_v04_backtest(
                data,
                config,
                variant=variant,
                initial_equity=initial_equity,
                session_dates=dates_2024,
            )
            if with_segments
            else None
        )
        holdout = (
            v04.run_v04_backtest(
                data,
                config,
                variant=variant,
                initial_equity=initial_equity,
                session_dates=dates_holdout,
            )
            if with_segments
            else None
        )
        walkforward = (
            v04.run_v04_walkforward(
                data,
                config,
                variant=variant,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
                initial_equity=initial_equity,
            )
            if with_walkforward
            else None
        )
        elapsed = time.perf_counter() - started
        results[label] = {
            "label": label,
            "variant": variant.flags(),
            "full": _format_v04_result(full),
            "train_2024": {
                "metrics": _metrics_row(seg_2024.metrics) if seg_2024 else None,
                "session_dates": len(seg_2024.session_dates) if seg_2024 else 0,
            },
            "holdout": {
                "metrics": _metrics_row(holdout.metrics) if holdout else None,
                "session_dates": len(holdout.session_dates) if holdout else 0,
            },
            "walkforward": _format_walkforward(walkforward) if walkforward else None,
            "elapsed_seconds": round(elapsed, 2),
            "result": full,  # kept in memory for the trade ledger / recaps
        }
        row = results[label]["full"]["metrics"]
        print(
            f"  full {row['total_trades']} 笔 / {row['net_pnl_dollars']:+.2f} / "
            f"PF {row['profit_factor']:.4f}"
            + (
                f" | WF {results[label]['walkforward']['aggregate_test']['total_trades']} 笔 / "
                f"{results[label]['walkforward']['aggregate_test']['net_pnl_dollars']:+.2f} / "
                f"{results[label]['walkforward']['summary']['positive_folds']}/29 盈利折"
                if walkforward is not None
                else ""
            )
            + f"（{elapsed:.1f}s）",
            flush=True,
        )
    return results


def _step3_self_check(compare: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    step3 = compare["Step3"]
    sources = {
        "full": step3["full"]["metrics"],
        "train_2024": (step3.get("train_2024") or {}).get("metrics") or {},
        "holdout": (step3.get("holdout") or {}).get("metrics") or {},
        "walkforward": {
            **((step3.get("walkforward") or {}).get("aggregate_test") or {}),
            "positive_folds": (step3.get("walkforward") or {}).get("summary", {}).get(
                "positive_folds"
            ),
            "folds": (step3.get("walkforward") or {}).get("summary", {}).get("folds"),
        },
    }
    tolerances = {
        "total_trades": 0.0,
        "win_rate_pct": 0.01,
        "profit_factor": 0.001,
        "net_pnl_dollars": 0.01,
        "avg_r": 0.001,
        "max_drawdown_dollars": 0.01,
        "positive_folds": 0.0,
        "folds": 0.0,
    }
    checks: List[Dict[str, Any]] = []
    passed = True
    for section, expected_fields in STEP3_RECORDED.items():
        actual_metrics = sources[section]
        for field, expected in expected_fields.items():
            actual = actual_metrics.get(field)
            tolerance = tolerances.get(field, 0.01)
            ok = (
                actual is not None
                and abs(float(actual) - float(expected)) <= tolerance + 1e-9
            )
            passed = passed and ok
            checks.append(
                {
                    "section": section,
                    "field": field,
                    "recorded_task12": expected,
                    "recomputed": actual,
                    "delta": _number(float(actual) - float(expected), 6)
                    if actual is not None
                    else None,
                    "tolerance": tolerance,
                    "ok": bool(ok),
                }
            )
    return {"passed": bool(passed), "checks": checks}


# --------------------------------------------------------------------------- #
# Friday recap + filter diagnostics
# --------------------------------------------------------------------------- #
def _trade_key(trade: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(trade.get("symbol")),
        str(trade.get("strategy")),
        str(trade.get("entry_time_utc")),
    )


def _signal_diagnostics(
    prepared: v02.V02Data,
    trade: Mapping[str, Any],
    flat_buckets: Mapping[str, Sequence[Any]],
) -> Dict[str, Any]:
    """Reconstruct the Step-3b filter values at the trade's signal 5m bar."""

    entry_utc = datetime.fromisoformat(str(trade["entry_time_utc"]))
    # The signal 5m bar ends at the fill instant (next 5m bar open), so its
    # start is the fill time minus one 5m bucket.
    signal_bar_start = entry_utc - timedelta(minutes=5)
    session_date = et_date(signal_bar_start)
    session = next(
        (item for item in prepared.sessions if item.et_date == session_date), None
    )
    diagnostics: Dict[str, Any] = {
        "signal_bar_start_utc": signal_bar_start.isoformat(),
        "strategy": trade.get("strategy"),
        "symbol": trade.get("symbol"),
    }
    if session is None:
        diagnostics["error"] = "session not found"
        return diagnostics
    symbol = str(trade["symbol"])
    view = build_strategy_view(prepared, session, symbol, signal_bar_start)
    ctx = build_eval_context(prepared, session, symbol, signal_bar_start, flat_buckets)
    if view is None:
        diagnostics["error"] = "bucket not found"
        return diagnostics
    exhaustion_gain = None
    if ctx.session_low_so_far:
        exhaustion_gain = (
            (view.bar.close - ctx.session_low_so_far) / ctx.session_low_so_far * 100.0
        )
    diagnostics.update(
        {
            "close_5m": round(view.bar.close, 4),
            "ema21_5m": round(view.ema_slow, 4) if view.ema_slow is not None else None,
            "session_vwap": round(view.session_vwap, 4)
            if view.session_vwap is not None
            else None,
            "session_low_so_far": round(ctx.session_low_so_far, 4)
            if ctx.session_low_so_far is not None
            else None,
            "exhaustion_gain_pct": _number(exhaustion_gain),
            "exhaustion_blocked": ema_pullback_exhaustion_blocked(
                view.bar.close, ctx.session_low_so_far
            ),
            "close_ge_ema21": (
                view.ema_slow is not None
                and view.bar.close >= float(view.ema_slow) - 1e-12
            ),
            "close_ge_vwap": (
                view.session_vwap is not None
                and view.bar.close >= float(view.session_vwap) - 1e-12
            ),
            "smc_trend_confirm_ok": smc_trend_confirm_ok(
                view.bar.close, view.ema_slow, view.session_vwap
            ),
        }
    )
    return diagnostics


def _blocked_reason(trade: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    strategy = str(trade.get("strategy"))
    if strategy == EMA_PULLBACK_5M:
        if diagnostics.get("exhaustion_blocked"):
            return (
                "EMA 衰竭防追高：close 较当日低点已涨 "
                f"{diagnostics.get('exhaustion_gain_pct'):.2f}% >= "
                f"{EMA_EXHAUSTION_GAIN_LIMIT * 100:.1f}%"
            )
        return "EMA_PULLBACK 被其它条件过滤"
    if strategy == SMC_SWEEP_5M:
        if not diagnostics.get("smc_trend_confirm_ok"):
            return (
                "SMC 水下接刀：close 同时低于 EMA21 与 session VWAP "
                f"(close={diagnostics.get('close_5m')}, "
                f"ema21={diagnostics.get('ema21_5m')}, vwap={diagnostics.get('session_vwap')})"
            )
        return "SMC 被其它条件过滤"
    return "未知原因（理论上 Step3b 不改变 VWAP_RECLAIM）"


def build_friday_recap(
    friday_bars: Sequence[Bar],
    *,
    variants: Sequence[tuple[str, v04.V04Variant]] = COMPARE_VARIANTS,
    initial_equity: float = 3000.0,
) -> Dict[str, Any]:
    """Run the three variants on Friday 2026-11-06 and build the recap payload."""

    config = load_config()
    print("[周五复盘] 准备 5m 数据（含历史预热）…", flush=True)
    started = time.perf_counter()
    prepared = prepare_v02_data(friday_bars)
    print(
        f"  prepared {len(prepared.sessions)} sessions，最后 {prepared.dates[-1]}"
        f"（{time.perf_counter() - started:.1f}s）",
        flush=True,
    )
    flat_buckets = v04._flat_buckets_by_symbol(prepared)
    market = _market_snapshot(
        [bar for bar in friday_bars if et_date(bar.start_utc) == FRIDAY_DATE]
    )

    versions: Dict[str, Any] = {}
    markers_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for label, variant in variants:
        print(f"[周五复盘] {label} …", flush=True)
        result = v04.run_v04_backtest(
            prepared,
            config,
            variant=variant,
            initial_equity=initial_equity,
            session_dates=[FRIDAY_DATE],
        )
        trade_dicts = v04.result_to_trade_dicts(result)
        for trade in trade_dicts:
            trade["diagnostics"] = _signal_diagnostics(
                prepared, trade, flat_buckets
            )
        ordered_markers = sorted(
            result.markers, key=lambda item: (item.get("t") or 0, item["type"])
        )
        markers = [
            {
                "type": marker["type"],
                "time_utc": marker["time_utc"],
                "price": marker["price"],
                "symbol": marker["symbol"],
                "strategy": marker["strategy"],
                "trade_id": marker["trade_id"],
                "t": marker.get("t"),
                "detail": marker.get("detail", ""),
                "variant": label,
                **{
                    key: marker[key]
                    for key in ("stop", "target", "qty", "limit_price")
                    if key in marker
                },
            }
            for marker in ordered_markers
        ]
        versions[label] = {
            "version": label,
            "variant": variant.flags(),
            "engine": (
                "v0.4 Step1 (window, 2 strategies, V0.3 shield)"
                if variant.step == 1
                else (
                    "v0.4 Step3 (window + VWAP trend + SMC, V0.3 shield)"
                    if variant.step == 3
                    else "v0.4 Step3b (Step3 + exhaustion guard + SMC right-side confirm)"
                )
            ),
            "summary": _summary_from_metrics(result.metrics),
            "trades": trade_dicts,
            "markers": markers,
            "diagnostics": {
                "signals": result.signals,
                "pending_cancelled": result.pending_cancelled,
                "risk_block_counts": dict(result.risk_block_counts),
                "window_skipped_evaluations": result.window_skipped_evaluations,
                "breakeven_armed_trades": result.breakeven_armed_trades,
            },
        }
        markers_by_variant[label] = markers
        if not trade_dicts:
            versions[label]["zero_trade_reason"] = (
                "当日 0 笔：10:00-14:00 入场窗内无 5m 策略触发，或触发后被趋势/防追高/防接刀/风控过滤"
            )

    # -- Step3 vs Step3b filter effect --------------------------------------
    step3_trades = {_trade_key(t): t for t in versions["Step3"]["trades"]}
    step3b_trades = {_trade_key(t): t for t in versions["Step3b"]["trades"]}
    filtered_out: List[Dict[str, Any]] = []
    for key, trade in step3_trades.items():
        if key in step3b_trades:
            continue
        diagnostics = trade.get("diagnostics") or {}
        filtered_out.append(
            {
                "trade": trade,
                "diagnostics": diagnostics,
                "blocked_reason": _blocked_reason(trade, diagnostics),
            }
        )
    appeared_in_step3b: List[Dict[str, Any]] = [
        trade for key, trade in step3b_trades.items() if key not in step3_trades
    ]

    # -- The two Friday pitfalls the Lead asked to verify --------------------
    pitfalls: Dict[str, Any] = {}
    for pit_name, pit in FRIDAY_PITFALLS:
        key = (pit["symbol"], pit["strategy"], pit["entry_time_utc"])
        pitfalls[pit_name] = {
            "symbol": pit["symbol"],
            "strategy": pit["strategy"],
            "entry_time_utc": pit["entry_time_utc"],
            "present_in": {
                label: key in {_trade_key(t) for t in versions[label]["trades"]}
                for label in versions
            },
            "step3b_avoided": key not in step3b_trades,
        }

    filters_effect = {
        "step3_trade_count": len(step3_trades),
        "step3b_trade_count": len(step3b_trades),
        "filtered_out_count": len(filtered_out),
        "filtered_out": filtered_out,
        "appeared_in_step3b_count": len(appeared_in_step3b),
        "appeared_in_step3b": appeared_in_step3b,
        "pitfalls": pitfalls,
    }
    versions["Step3b"]["filter_effect_vs_step3"] = filters_effect

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4b",
        "session_date": FRIDAY_DATE.isoformat(),
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw, 2026-11-06 regular session "
            "(single venue; NOT consolidated tape)"
        ),
        "caveat": DATA_CAVEAT,
        "market": market,
        "versions": versions,
        "filters_effect": filters_effect,
        "markers_by_variant": markers_by_variant,
        "markers_by_variant_count": {
            label: len(markers) for label, markers in markers_by_variant.items()
        },
        "notes": [
            "V0.4b: Step1 = 10:00-14:00 window + 2 strategies; Step3 adds VWAP trend filter + SMC; "
            "Step3b adds EMA exhaustion guard (>=8.0% above session low) + SMC right-side confirm "
            "(close >= EMA21 or session VWAP).",
            "Both Step3b filters only affect NEW entries; open positions keep their stops/targets/EOD logic.",
            "Step3/Step3b use the V0.3 1.2R single-stop breakeven shield (not the deprecated Step4 scale-out).",
            "The 16:00 ET post-close bar in the Friday file is excluded from replay (regular session only).",
            "IEX single-venue data; research output, not proof of profitability.",
        ],
    }


def build_markers_payload(recap: Mapping[str, Any]) -> Dict[str, Any]:
    """Web-friendly per-variant marker payload."""

    markers_by_variant = recap.get("markers_by_variant") or {}
    flat_markers: List[Dict[str, Any]] = []
    for label in ("Step1", "Step3", "Step3b"):
        for marker in markers_by_variant.get(label, []):
            flat_markers.append({**marker, "variant": label})
    flat_markers.sort(key=lambda item: (item.get("t") or 0, item["type"]))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": recap.get("session_date"),
        "data_source": recap.get("data_source"),
        "variants": list(markers_by_variant.keys()),
        "markers_by_variant": markers_by_variant,
        "markers": flat_markers,
        "trades_by_variant": {
            label: [
                {
                    "trade_id": trade["trade_id"],
                    "symbol": trade["symbol"],
                    "strategy": trade["strategy"],
                    "entry_time_utc": trade["entry_time_utc"],
                    "exit_time_utc": trade["exit_time_utc"],
                    "exit_reason": trade["exit_reason"],
                    "pnl": trade["pnl"],
                }
                for trade in (recap.get("versions", {}).get(label, {}) or {}).get(
                    "trades", []
                )
            ]
            for label in markers_by_variant
        },
    }


# --------------------------------------------------------------------------- #
# Markdown / terminal rendering
# --------------------------------------------------------------------------- #
_COMPARE_HEADER = (
    "| 变体 | 全周期笔数 | 胜率 | PF | 净盈亏($) | 平均R | 最大回撤($) | "
    "2024笔数 | 2024 PF | 2024净($) | Holdout笔数 | Holdout PF | Holdout净($) | "
    "WF笔数 | WF胜率 | WF PF | WF净($) | WF盈利折 |"
)
_COMPARE_SEPARATOR = "|" + "---|" * 18


def _compare_row(label: str, payload: Mapping[str, Any]) -> str:
    full = payload["full"]["metrics"]
    train = (payload.get("train_2024") or {}).get("metrics") or {}
    holdout = (payload.get("holdout") or {}).get("metrics") or {}
    wf = payload.get("walkforward") or {}
    wf_agg = wf.get("aggregate_test") or {}
    wf_summary = wf.get("summary") or {}
    variant_info = payload.get("variant") or {}
    change = variant_info.get("name", "")
    return (
        f"| {label} | {full.get('total_trades')} | {_fmt(full.get('win_rate_pct'), 'pct')} | "
        f"{_fmt(full.get('profit_factor'), 'pf')} | {_fmt(full.get('net_pnl_dollars'), 'money')} | "
        f"{_fmt(full.get('avg_r'), 'r')} | {_fmt(full.get('max_drawdown_dollars'), 'dd')} | "
        f"{train.get('total_trades')} | {_fmt(train.get('profit_factor'), 'pf')} | "
        f"{_fmt(train.get('net_pnl_dollars'), 'money')} | "
        f"{holdout.get('total_trades')} | {_fmt(holdout.get('profit_factor'), 'pf')} | "
        f"{_fmt(holdout.get('net_pnl_dollars'), 'money')} | "
        f"{wf_agg.get('total_trades', '—')} | {_fmt(wf_agg.get('win_rate_pct'), 'pct')} | "
        f"{_fmt(wf_agg.get('profit_factor'), 'pf')} | {_fmt(wf_agg.get('net_pnl_dollars'), 'money')} | "
        f"{wf_summary.get('positive_folds', 0)}/{wf_summary.get('folds', 0)} |"
    )


def _friday_summary_row(label: str, payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    extra = ""
    if summary.get("total_trades", 0) == 0 and payload.get("zero_trade_reason"):
        extra = payload["zero_trade_reason"]
    return (
        f"| {label} | {payload.get('engine', '')} | {summary.get('total_trades', 0)} | "
        f"{_fmt(summary.get('win_rate_pct'), 'pct')} | {_fmt(summary.get('profit_factor'), 'pf')} | "
        f"{_fmt(summary.get('net_pnl_dollars'), 'money')} | {_fmt(summary.get('avg_r'), 'r')} | {extra or '—'} |"
    )


def _friday_trade_rows(payload: Mapping[str, Any]) -> List[str]:
    rows: List[str] = []
    for index, trade in enumerate(payload.get("trades", []), start=1):
        diagnostics = trade.get("diagnostics") or {}
        bits = []
        if trade.get("strategy") == EMA_PULLBACK_5M:
            bits.append(
                f"日内涨幅 {diagnostics.get('exhaustion_gain_pct')}%"
                + ("（应被防追高拦截）" if diagnostics.get("exhaustion_blocked") else "（未触发防追高）")
            )
        if trade.get("strategy") == SMC_SWEEP_5M:
            bits.append(
                f"confirm={'OK' if diagnostics.get('smc_trend_confirm_ok') else '水下'}"
                f" (close={diagnostics.get('close_5m')}, ema21={diagnostics.get('ema21_5m')}, "
                f"vwap={diagnostics.get('session_vwap')})"
            )
        rows.append(
            "| {index} | {entry} | {symbol} | {strategy} | {entry_price} | {stop} | {target} | "
            "{exit_time} | {exit_price} | {reason} | {pnl} | {r} | {bits} |".format(
                index=index,
                entry=trade.get("entry_time_et") or trade.get("entry_time_utc"),
                symbol=trade.get("symbol"),
                strategy=trade.get("strategy"),
                entry_price=_fmt(trade.get("entry_price_fill"), "num"),
                stop=_fmt(trade.get("stop_price"), "num"),
                target=_fmt(trade.get("target_price"), "num"),
                exit_time=trade.get("exit_time_et") or trade.get("exit_time_utc"),
                exit_price=_fmt(trade.get("exit_price_fill"), "num"),
                reason=trade.get("exit_reason"),
                pnl=_fmt(trade.get("pnl"), "money"),
                r=_fmt(trade.get("r_multiple"), "r"),
                bits="；".join(bits) or "—",
            )
        )
    return rows


def build_compare_markdown(
    *,
    compare: Mapping[str, Mapping[str, Any]],
    self_check: Mapping[str, Any],
    friday: Mapping[str, Any],
    data_span: str,
    total_1m_bars: int,
    total_sessions: int,
    initial_equity: float,
    elapsed: Mapping[str, float],
) -> str:
    lines = [
        "# V0.4b Step1 vs Step3 vs Step3b · raw IEX 1m 对比 + 周五复盘",
        "",
        f"- 数据：Alpaca IEX raw 1m，{data_span}，{total_1m_bars:,} 根 1m bar / "
        f"{total_sessions} 个 session（单一交易所口径，非 consolidated tape）",
        f"- 结构：{V04B_STRUCTURE_NOTE}",
        "- Step3b 新增（固定阈值，不调参）：EMA_PULLBACK 日内涨幅 ≥ "
        f"{EMA_EXHAUSTION_GAIN_LIMIT * 100:.1f}% 禁止（相对 09:30 起已完成 5m low）；"
        "SMC 要求 close ≥ EMA21 或 close ≥ session VWAP（禁止水下接刀）。",
        f"- 初始权益 ${initial_equity:,.2f}；29 折 WF（120/20/20、每折 fresh RiskManager、不复利）。",
        "",
        "## 主表：Step1 / Step3 / Step3b（全周期 + 2024 + holdout + 29 折 WF）",
        "",
        _COMPARE_HEADER,
        _COMPARE_SEPARATOR,
        *[_compare_row(label, compare[label]) for label in ("Step1", "Step3", "Step3b")],
        "",
        "## Step 3 与 task-12 记录自洽校验（804 / +$9.28 / WF -$250.63 / 15 折）",
        "",
        f"结果：{'通过 ✅' if self_check.get('passed') else '未通过 ❌'}",
        "",
        "| 区段 | 指标 | task-12 记录 | 本次重算 | 偏差 | 容差 | 通过 |",
        "|---|---|---|---|---|---|---|",
    ]
    for check in self_check.get("checks", []):
        lines.append(
            f"| {check['section']} | {check['field']} | {check['recorded_task12']} | "
            f"{check['recomputed']} | {check['delta']} | ±{check['tolerance']} | "
            f"{'✅' if check['ok'] else '❌'} |"
        )
    lines.extend(
        [
            "",
            f"## 周五 {friday.get('session_date')} 三版本复盘（raw IEX 1m）",
            "",
            "### 当日行情（常规时段 09:30-16:00 ET）",
            "",
            "| 标的 | 1m bars | 开盘 | 最高 | 最低 | 收盘 | 振幅% | 成交量 | VWAP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol, market in friday.get("market", {}).items():
        lines.append(
            f"| {symbol} | {market.get('regular_bars', 0)} | {_fmt(market.get('open'), 'num')} | "
            f"{_fmt(market.get('high'), 'num')} | {_fmt(market.get('low'), 'num')} | "
            f"{_fmt(market.get('close'), 'num')} | {_fmt(market.get('range_pct'), 'num2')} | "
            f"{market.get('volume', 0):,.0f} | {_fmt(market.get('vwap'), 'num')} |"
        )
    lines.extend(
        [
            "",
            "### 三版本汇总",
            "",
            "| 变体 | 引擎 | 笔数 | 胜率 | PF | 净盈亏($) | 平均R | 说明 |",
            "|---|---|---|---|---|---|---|---|",
            *[_friday_summary_row(label, friday["versions"][label]) for label in ("Step1", "Step3", "Step3b")],
            "",
            "### 逐笔信号明细 & 过滤器诊断",
            "",
        ]
    )
    for label in ("Step1", "Step3", "Step3b"):
        payload = friday["versions"][label]
        lines.append(f"#### {label}（{len(payload.get('trades', []))} 笔）")
        lines.append("")
        if not payload.get("trades"):
            lines.append(f"当日无信号。原因：{payload.get('zero_trade_reason', '—')}")
            lines.append("")
            continue
        lines.append(
            "| # | 入场时间(ET) | 标的 | 策略 | 入场价 | 止损 | 目标 | 出场时间(ET) | 出场价 | 原因 | PnL($) | R | 过滤诊断 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---|")
        lines.extend(_friday_trade_rows(payload))
        lines.append("")
    effect = friday.get("filters_effect") or {}
    pitfalls = effect.get("pitfalls") or {}
    lines.extend(
        [
            "### Step3 → Step3b 过滤效果（重点：是否避开两坑）",
            "",
            f"- Step3 当日 {effect.get('step3_trade_count', 0)} 笔 → Step3b 当日 "
            f"{effect.get('step3b_trade_count', 0)} 笔；被拦截 {effect.get('filtered_out_count', 0)} 笔，"
            f"新增 {effect.get('appeared_in_step3b_count', 0)} 笔。",
            "",
            "| 坑 | 标的/策略/入场(UTC) | Step1 | Step3 | Step3b | Step3b 是否避开 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, info in pitfalls.items():
        present = info.get("present_in", {})
        lines.append(
            f"| {name} | {info.get('symbol')} / {info.get('strategy')} / {info.get('entry_time_utc')} | "
            f"{'有' if present.get('Step1') else '无'} | {'有' if present.get('Step3') else '无'} | "
            f"{'有' if present.get('Step3b') else '无'} | "
            f"{'✅ 避开' if info.get('step3b_avoided') else '❌ 未避开'} |"
        )
    lines.append("")
    if effect.get("filtered_out"):
        lines.append("被 Step3b 拦截的 Step3 信号：")
        lines.append("")
        lines.append("| 时间(UTC) | 标的 | 策略 | 拦截原因 |")
        lines.append("|---|---|---|---|")
        for item in effect["filtered_out"]:
            trade = item["trade"]
            lines.append(
                f"| {trade.get('entry_time_utc')} | {trade.get('symbol')} | "
                f"{trade.get('strategy')} | {item.get('blocked_reason')} |"
            )
        lines.append("")
    lines.extend(
        [
            f"### 图表标记（markers_by_variant）",
            "",
            "- "
            + "；".join(
                f"{label}={count}"
                for label, count in (friday.get("markers_by_variant_count") or {}).items()
            ),
            "- 字段：type（ENTRY/STOP/BREAKEVEN/TARGET/EXIT）/time_utc/price/symbol/strategy/trade_id/t；",
            "- 文件：`var/backtest/v04b_friday_markers.json`（`markers_by_variant` 按变体分组）。",
            "",
            "## 说明",
            "",
            "- 所有阈值 a-priori、未调参；Step1/Step3/Step3b 只做结构对比，不择优。",
            "- Step3b 两个过滤器只影响新开仓；已持仓的止损/止盈/EOD 逻辑不变。",
            "- PRODUCTION_VARIANT 当前为 Step3b（Lead 可按 OOS 结果一行切换）。",
            f"- 运行时长：{elapsed.get('total', 0.0):.1f}s。",
            "",
            "> 数据口径：IEX 单一交易所；研究结果不代表实盘有效性或盈利预期。",
            "",
        ]
    )
    return "\n".join(lines)


def print_compare_table(compare: Mapping[str, Mapping[str, Any]]) -> None:
    print("\n" + "=" * 150)
    print("V0.4b Step1 vs Step3 vs Step3b（raw 未复权 IEX 1m，29 折 WF 样本外）")
    print("=" * 150)
    print(
        f"{'变体':<8}{'全周期笔数':>10}{'胜率':>9}{'PF':>8}{'净盈亏($)':>13}{'平均R':>9}{'MaxDD($)':>12}"
        f"{'2024笔':>8}{'2024PF':>8}{'2024净($)':>12}{'Hold笔':>8}{'HoldPF':>8}{'Hold净($)':>12}"
        f"{'WF笔':>7}{'WF胜率':>8}{'WF PF':>7}{'WF净($)':>12}{'盈利折':>9}"
    )
    print("-" * 150)
    for label in ("Step1", "Step3", "Step3b"):
        payload = compare[label]
        full = payload["full"]["metrics"]
        train = (payload.get("train_2024") or {}).get("metrics") or {}
        holdout = (payload.get("holdout") or {}).get("metrics") or {}
        wf = payload.get("walkforward") or {}
        wf_agg = wf.get("aggregate_test") or {}
        wf_summary = wf.get("summary") or {}
        print(
            f"{label:<8}{full.get('total_trades', '—'):>10}"
            f"{_fmt(full.get('win_rate_pct'), 'pct'):>9}"
            f"{_fmt(full.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(full.get('net_pnl_dollars'), 'money'):>13}"
            f"{_fmt(full.get('avg_r'), 'r'):>9}"
            f"{_fmt(full.get('max_drawdown_dollars'), 'dd'):>12}"
            f"{train.get('total_trades', '—'):>8}"
            f"{_fmt(train.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(train.get('net_pnl_dollars'), 'money'):>12}"
            f"{holdout.get('total_trades', '—'):>8}"
            f"{_fmt(holdout.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(holdout.get('net_pnl_dollars'), 'money'):>12}"
            f"{wf_agg.get('total_trades', '—'):>7}"
            f"{_fmt(wf_agg.get('win_rate_pct'), 'pct'):>8}"
            f"{_fmt(wf_agg.get('profit_factor'), 'pf'):>7}"
            f"{_fmt(wf_agg.get('net_pnl_dollars'), 'money'):>12}"
            f"{str(wf_summary.get('positive_folds', 0)) + '/' + str(wf_summary.get('folds', 0)):>9}"
        )
    print("-" * 150)


def print_friday_recap(friday: Mapping[str, Any]) -> None:
    print("\n" + "=" * 120)
    print(f"周五 {friday.get('session_date')} 三版本复盘（raw IEX 1m，IEX 单一交易所口径）")
    print("=" * 120)
    print(f"  {'标的':<6}{'1m bars':>9}{'开盘':>11}{'最高':>11}{'最低':>11}{'收盘':>11}{'振幅%':>9}{'成交量':>14}")
    print("-" * 120)
    for symbol, market in friday.get("market", {}).items():
        print(
            f"  {symbol:<6}{market.get('regular_bars', 0):>9}"
            f"{_fmt(market.get('open'), 'num'):>11}{_fmt(market.get('high'), 'num'):>11}"
            f"{_fmt(market.get('low'), 'num'):>11}{_fmt(market.get('close'), 'num'):>11}"
            f"{_fmt(market.get('range_pct'), 'num2'):>9}{market.get('volume', 0):>14,.0f}"
        )
    print("\n  " + f"{'变体':<8}{'笔数':>6}{'胜率':>9}{'PF':>8}{'净盈亏($)':>13}{'平均R':>9}")
    print("  " + "-" * 80)
    for label in ("Step1", "Step3", "Step3b"):
        summary = friday["versions"][label]["summary"]
        print(
            f"  {label:<8}{summary.get('total_trades', 0):>6}"
            f"{_fmt(summary.get('win_rate_pct'), 'pct'):>9}"
            f"{_fmt(summary.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(summary.get('net_pnl_dollars'), 'money'):>13}"
            f"{_fmt(summary.get('avg_r'), 'r'):>9}"
        )
    print("  " + "-" * 80)
    for label in ("Step1", "Step3", "Step3b"):
        payload = friday["versions"][label]
        trades = payload.get("trades", [])
        print(f"\n  {label} 当日信号（{len(trades)} 笔）：")
        if not trades:
            print(f"    0 笔：{payload.get('zero_trade_reason', '—')}")
            continue
        for trade in trades:
            diagnostics = trade.get("diagnostics") or {}
            note = ""
            if trade.get("strategy") == EMA_PULLBACK_5M:
                note = (
                    f" | 日内涨幅 {diagnostics.get('exhaustion_gain_pct')}%"
                    f"{' → Step3b 应拦截' if diagnostics.get('exhaustion_blocked') else ''}"
                )
            elif trade.get("strategy") == SMC_SWEEP_5M:
                note = (
                    f" | confirm={'OK' if diagnostics.get('smc_trend_confirm_ok') else '水下'}"
                    f" close={diagnostics.get('close_5m')} ema21={diagnostics.get('ema21_5m')}"
                    f" vwap={diagnostics.get('session_vwap')}"
                )
            print(
                f"    {trade.get('entry_time_et')} ET {trade.get('symbol')} {trade.get('strategy'):<16}"
                f" → {trade.get('exit_time_et')} ET {trade.get('exit_reason'):<12}"
                f" PnL {_fmt(trade.get('pnl'), 'money')} R {_fmt(trade.get('r_multiple'), 'r')}{note}"
            )
    pitfalls = friday.get("filters_effect", {}).get("pitfalls", {})
    if pitfalls:
        print("\n  两坑验证：")
        for name, info in pitfalls.items():
            print(
                f"    {name}: Step3b {'✅ 避开' if info.get('step3b_avoided') else '❌ 未避开'}"
                f"（Step1={info['present_in'].get('Step1')}, Step3={info['present_in'].get('Step3')}, "
                f"Step3b={info['present_in'].get('Step3b')}）"
            )
    print(
        "\n  markers_by_variant: "
        + ", ".join(
            f"{label}={count}"
            for label, count in (friday.get("markers_by_variant_count") or {}).items()
        )
    )
    print("=" * 120)


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
) -> Dict[str, Any]:
    config = load_config()
    equity = float(config.risk.account_equity if initial_equity is None else initial_equity)
    started = time.perf_counter()

    print(f"加载 raw 回测数据：{data_dir}")
    raw_bars = load_raw_bars(data_dir)
    print("准备 5m 数据 …")
    prepare_started = time.perf_counter()
    data = prepare_v02_data(raw_bars)
    print(
        f"5m 数据就绪：{data.total_5m_bars:,} 根 5m bar，{len(data.sessions)} 个 session"
        f"（{time.perf_counter() - prepare_started:.1f}s）"
    )
    data_span = _fmt_span(list(data.dates))
    compare = run_compare(
        data,
        config,
        with_segments=with_segments,
        with_walkforward=with_walkforward,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_equity=equity,
    )
    self_check = _step3_self_check(compare)
    print(
        f"\nStep3 与 task-12 记录自洽校验：{'通过 ✅' if self_check['passed'] else '未通过 ❌'}"
    )
    for check in self_check["checks"]:
        print(
            f"  {'✅' if check['ok'] else '❌'} {check['section']:<12} {check['field']:<24} "
            f"记录={check['recorded_task12']} 重算={check['recomputed']} Δ={check['delta']}"
        )
    if not self_check["passed"]:  # investigate before publishing any numbers
        raise RuntimeError(
            "Step3 does not reproduce the task-12 record; refusing to publish V0.4b numbers"
        )

    print("\n加载周五数据（含 2026-11-06，raw）…")
    friday_bars = load_friday_raw_bars(data_dir)
    friday_started = time.perf_counter()
    friday = build_friday_recap(friday_bars, initial_equity=equity)
    friday_elapsed = time.perf_counter() - friday_started
    total_elapsed = time.perf_counter() - started

    step3b_result = compare["Step3b"]["result"]
    compare_markdown = build_compare_markdown(
        compare=compare,
        self_check=self_check,
        friday=friday,
        data_span=data_span,
        total_1m_bars=len(raw_bars),
        total_sessions=len(data.sessions),
        initial_equity=equity,
        elapsed={"total": total_elapsed},
    )
    markers_payload = build_markers_payload(friday)

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4b",
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw (single venue; NOT consolidated tape)"
        ),
        "backtest_data_span": data_span,
        "total_1m_bars": len(raw_bars),
        "total_sessions": len(data.sessions),
        "structure": V04B_STRUCTURE_NOTE,
        "step3b_filters": {
            "ema_exhaustion_gain_limit": EMA_EXHAUSTION_GAIN_LIMIT,
            "ema_exhaustion_rule": "(close_5m - session_low_so_far) / session_low_so_far >= 8.0% blocks new EMA_PULLBACK signals",
            "smc_trend_confirm_rule": "require close_5m >= EMA21_5m OR close_5m >= session_vwap at SMC trigger",
            "production_variant": v04.PRODUCTION_VARIANT.flags(),
        },
        "caveat": DATA_CAVEAT,
        "initial_equity": equity,
        "walkforward_grid": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
        },
        "compare": {
            label: {
                "variant": compare[label]["variant"],
                "full": compare[label]["full"],
                "train_2024": compare[label]["train_2024"],
                "holdout": compare[label]["holdout"],
                "walkforward": compare[label]["walkforward"],
                "elapsed_seconds": compare[label]["elapsed_seconds"],
            }
            for label in ("Step1", "Step3", "Step3b")
        },
        "step3_self_check": self_check,
        "friday": {
            "session_date": friday["session_date"],
            "market": friday["market"],
            "versions": {
                label: {
                    "summary": friday["versions"][label]["summary"],
                    "zero_trade_reason": friday["versions"][label].get(
                        "zero_trade_reason"
                    ),
                    "diagnostics": friday["versions"][label]["diagnostics"],
                }
                for label in ("Step1", "Step3", "Step3b")
            },
            "filters_effect": friday["filters_effect"],
            "markers_by_variant_count": friday["markers_by_variant_count"],
        },
        "elapsed_seconds": {
            "compare": round(total_elapsed - friday_elapsed, 2),
            "friday_recap": round(friday_elapsed, 2),
            "total": round(total_elapsed, 2),
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v04b_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    compare_md_path = report_dir / "v04b_compare.md"
    compare_md_path.write_text(compare_markdown, encoding="utf-8")
    trades_path = report_dir / "v04b_trades.csv"
    write_trades_csv(step3b_result, trades_path)
    friday_recap_path = report_dir / "v04b_friday_recap.json"
    friday_recap_path.write_text(
        json.dumps(friday, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    friday_md_path = report_dir / "v04b_friday_recap.md"
    friday_md_path.write_text(build_compare_markdown_section_md(friday), encoding="utf-8")
    markers_path = report_dir / "v04b_friday_markers.json"
    markers_path.write_text(
        json.dumps(markers_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_compare_table(compare)
    print_friday_recap(friday)
    print("\n产物：")
    print(f"  对比报告 : {report_path}")
    print(f"  对比 MD  : {compare_md_path}")
    print(f"  逐笔流水 : {trades_path}（Step3b 全周期 {len(step3b_result.trades)} 笔）")
    print(f"  周五复盘 : {friday_recap_path}")
    print(f"  周五 MD  : {friday_md_path}")
    print(
        f"  图表标记 : {markers_path}"
        f"（markers_by_variant: {friday['markers_by_variant_count']}）"
    )
    print(
        f"\n总运行时长：{total_elapsed:.1f}s"
        f"（对比 {total_elapsed - friday_elapsed:.1f}s + 周五复盘 {friday_elapsed:.1f}s）"
    )
    return report_payload


def build_compare_markdown_section_md(friday: Mapping[str, Any]) -> str:
    """Standalone markdown recap for the Friday session (reuses compare helpers)."""

    lines = [
        f"# V0.4b 周五单日复盘 · {friday.get('session_date')}（raw IEX 1m）",
        "",
        f"- 数据：{friday.get('data_source')}",
        f"- 口径警告：{DATA_CAVEAT}",
        "",
        "## 当日行情（常规时段 09:30-16:00 ET）",
        "",
        "| 标的 | 1m bars | 开盘 | 最高 | 最低 | 收盘 | 振幅% | 成交量 | VWAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, market in friday.get("market", {}).items():
        lines.append(
            f"| {symbol} | {market.get('regular_bars', 0)} | {_fmt(market.get('open'), 'num')} | "
            f"{_fmt(market.get('high'), 'num')} | {_fmt(market.get('low'), 'num')} | "
            f"{_fmt(market.get('close'), 'num')} | {_fmt(market.get('range_pct'), 'num2')} | "
            f"{market.get('volume', 0):,.0f} | {_fmt(market.get('vwap'), 'num')} |"
        )
    lines.extend(
        [
            "",
            "## 三版本汇总",
            "",
            "| 变体 | 引擎 | 笔数 | 胜率 | PF | 净盈亏($) | 平均R | 说明 |",
            "|---|---|---|---|---|---|---|---|",
            *[_friday_summary_row(label, friday["versions"][label]) for label in ("Step1", "Step3", "Step3b")],
            "",
            "## 逐笔信号明细 & 过滤器诊断",
            "",
        ]
    )
    for label in ("Step1", "Step3", "Step3b"):
        payload = friday["versions"][label]
        lines.append(f"### {label}（{len(payload.get('trades', []))} 笔）")
        lines.append("")
        if not payload.get("trades"):
            lines.append(f"当日无信号。原因：{payload.get('zero_trade_reason', '—')}")
            lines.append("")
            continue
        lines.append(
            "| # | 入场时间(ET) | 标的 | 策略 | 入场价 | 止损 | 目标 | 出场时间(ET) | 出场价 | 原因 | PnL($) | R | 过滤诊断 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---|")
        lines.extend(_friday_trade_rows(payload))
        lines.append("")
    effect = friday.get("filters_effect") or {}
    pitfalls = effect.get("pitfalls") or {}
    lines.extend(
        [
            "## Step3 → Step3b 过滤效果（重点：是否避开两坑）",
            "",
            f"- Step3 {effect.get('step3_trade_count', 0)} 笔 → Step3b "
            f"{effect.get('step3b_trade_count', 0)} 笔；拦截 {effect.get('filtered_out_count', 0)} 笔，"
            f"新增 {effect.get('appeared_in_step3b_count', 0)} 笔。",
            "",
            "| 坑 | 标的/策略/入场(UTC) | Step1 | Step3 | Step3b | Step3b 是否避开 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, info in pitfalls.items():
        present = info.get("present_in", {})
        lines.append(
            f"| {name} | {info.get('symbol')} / {info.get('strategy')} / {info.get('entry_time_utc')} | "
            f"{'有' if present.get('Step1') else '无'} | {'有' if present.get('Step3') else '无'} | "
            f"{'有' if present.get('Step3b') else '无'} | "
            f"{'✅ 避开' if info.get('step3b_avoided') else '❌ 未避开'} |"
        )
    lines.append("")
    if effect.get("filtered_out"):
        lines.extend(
            [
                "被拦截信号：",
                "",
                "| 时间(UTC) | 标的 | 策略 | 拦截原因 |",
                "|---|---|---|---|",
            ]
        )
        for item in effect["filtered_out"]:
            trade = item["trade"]
            lines.append(
                f"| {trade.get('entry_time_utc')} | {trade.get('symbol')} | "
                f"{trade.get('strategy')} | {item.get('blocked_reason')} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 图表标记（markers_by_variant）",
            "",
            "- "
            + "；".join(
                f"{label}={count}"
                for label, count in (friday.get("markers_by_variant_count") or {}).items()
            ),
            "- 字段：type / time_utc / price / symbol / strategy / trade_id / t；",
            "- 文件：`var/backtest/v04b_friday_markers.json`。",
            "",
            "## 说明",
            "",
            *[f"- {note}" for note in friday.get("notes", [])],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.4b Step1/Step3/Step3b comparison + Friday recap on raw IEX 1m bars."
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
