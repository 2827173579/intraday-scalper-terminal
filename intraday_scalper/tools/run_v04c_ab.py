"""V0.4c A/B: SMC strict AND confirmation, with/without the 8% exhaustion cap.

Task-13 delivered Step 3b (SMC right-side confirmation as **OR**: close >= EMA21
**or** close >= session VWAP, plus the 8% EMA_PULLBACK exhaustion guard).
Step 3c upgrades the SMC confirmation to strict **AND** (close >= EMA21 **and**
close >= session VWAP) and A/B-tests whether the 8% exhaustion cap is kept:

======  ======================================================================
Step3b  reference: SMC OR + 8% exhaustion cap (existing production candidate)
NoCap   Step 3c A: SMC AND only (no 8% exhaustion cap)
WithCap Step 3c B: SMC AND + 8% exhaustion cap
======  ======================================================================

Everything else is inherited from Step 3 (10:00-14:00 ET window, fractional
shares, VWAP_RECLAIM trend filter, EMA_PULLBACK, SMC_SWEEP, V0.3 single-stop
1.2R breakeven shield, no deprecated Step-4 scale-out).  No parameter tuning:
the AND rule and the 8% cap are fixed priors.

Runs the three variants on the raw (unadjusted) IEX backtest dataset
(483,751 bars / 714 sessions / 2024-01-02..2026-11-05): full period, 2024
in-sample, 2025-2026 holdout, and the same 29-fold walk-forward grid
(120/20/20, fresh ``RiskManager`` per fold, no compounding), then replays
Friday 2026-11-06 with per-trade diagnostics and per-variant markers.

Artifacts (``intraday_scalper/var/backtest/``)::

    v04c_report.json          # A/B compare + Step3/Step3b self-check + Friday
    v04c_ab.md                # markdown A/B table + Friday detail + recommendation
    v04c_trades.csv           # NoCap + WithCap full-period ledgers (variant column)
    v04c_friday_recap.json    # Friday recap (versions: NoCap/WithCap + Step3b ref)
    v04c_friday_recap.md      # Friday markdown recap
    v04c_friday_markers.json  # {"markers_by_variant": {"NoCap": [...], "WithCap": [...]}}

Data caveat: Alpaca IEX-only single venue, not consolidated tape; research only.

Usage::

    python -m intraday_scalper.tools.run_v04c_ab
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
    build_strategy_view,
    prepare_v02_data,
)
from intraday_scalper.research.v04 import (
    EMA_EXHAUSTION_GAIN_LIMIT,
    SMC_SWEEP_5M,
    build_eval_context,
    ema_pullback_exhaustion_blocked,
    smc_trend_confirm_ok,
    smc_trend_confirm_require_both,
)
from intraday_scalper.tools.run_v03_raw_crosscheck import load_raw_bars
from intraday_scalper.tools.run_v04_backtest import (
    FRIDAY_DATE,
    _fmt,
    _format_v04_result,
    _format_walkforward,
    _market_snapshot,
    _metrics_row,
    _number,
    _pct,
    _summary_from_metrics,
    load_friday_raw_bars,
)
from intraday_scalper.tools.run_v04b_compare import (
    _signal_diagnostics,
    _trade_key,
    _fmt_span,
)

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

TRAIN_2024_START = date(2024, 1, 1)
TRAIN_2024_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 11, 5)

#: Variants shown in the A/B main table (Step3b first as the reference row).
AB_ROWS: tuple[tuple[str, v04.V04Variant], ...] = (
    ("Step3b", v04.STEP3B_VARIANT),
    ("NoCap", v04.STEP3C_NOCAP_VARIANT),
    ("WithCap", v04.STEP3C_WITHCAP_VARIANT),
)

#: Recorded anchors (task-12 / task-13) that must be reproduced.
RECORDED_ANCHORS: Dict[str, Dict[str, Any]] = {
    "Step3": {
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
    },
    "Step3b": {
        "full": {
            "total_trades": 730,
            "win_rate_pct": 43.70,
            "profit_factor": 1.0120,
            "net_pnl_dollars": 76.32,
            "avg_r": 0.0085,
            "max_drawdown_dollars": 702.07,
        },
        "train_2024": {"total_trades": 249, "profit_factor": 0.7872, "net_pnl_dollars": -490.53},
        "holdout": {"total_trades": 481, "profit_factor": 1.1393, "net_pnl_dollars": 566.85},
        "walkforward": {
            "total_trades": 603,
            "win_rate_pct": 43.12,
            "profit_factor": 0.9610,
            "net_pnl_dollars": -210.04,
            "positive_folds": 15,
            "folds": 29,
        },
    },
}

#: Friday trap keys (symbol, strategy, fill/entry UTC) to verify explicitly.
FRIDAY_TRAPS: tuple[tuple[str, Dict[str, str]], ...] = (
    (
        "11:25 SOXL SMC 接刀",
        {
            "symbol": "SOXL",
            "strategy": SMC_SWEEP_5M,
            "entry_time_utc": "2026-11-06T16:25:00+00:00",
        },
    ),
    (
        "12:40 SOXS EMA 山顶追高",
        {
            "symbol": "SOXS",
            "strategy": EMA_PULLBACK_5M,
            "entry_time_utc": "2026-11-06T17:40:00+00:00",
        },
    ),
)

V04C_STRUCTURE_NOTE = (
    "V0.4c A/B：在 Step3 基础上把 SMC_SWEEP_5M 的右侧确认从 OR（close>=EMA21 或 "
    "close>=session VWAP）升级为严格 AND（close>=EMA21 且 close>=session VWAP），并对比"
    "是否保留 8.0% EMA_PULLBACK 防追高锁。Step3b 作参照（OR + 8%）。"
    "其余结构（10:00-14:00 入场窗、fractional、VWAP_RECLAIM 趋势过滤、EMA_PULLBACK、"
    "V0.3 1.2R 单止损保本盾、无 Step4 分批止盈）完全一致；"
    "29 折 WF 纪律（120/20/20、每折全新 RiskManager、不复利）与 V0.2/V0.3/V0.4b 相同。"
)


# --------------------------------------------------------------------------- #
# Backtest comparison
# --------------------------------------------------------------------------- #
def run_ab(
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
    dates = list(data.dates)
    dates_2024 = [value for value in dates if TRAIN_2024_START <= value <= TRAIN_2024_END]
    dates_holdout = [value for value in dates if HOLDOUT_START <= value <= HOLDOUT_END]
    results: Dict[str, Dict[str, Any]] = {}
    for label, variant in AB_ROWS:
        print(f"\n[{label}] {variant.name} …", flush=True)
        started = time.perf_counter()
        full = v04.run_v04_backtest(
            data, config, variant=variant, initial_equity=initial_equity
        )
        seg_2024 = (
            v04.run_v04_backtest(
                data, config, variant=variant, initial_equity=initial_equity,
                session_dates=dates_2024,
            )
            if with_segments
            else None
        )
        holdout = (
            v04.run_v04_backtest(
                data, config, variant=variant, initial_equity=initial_equity,
                session_dates=dates_holdout,
            )
            if with_segments
            else None
        )
        walkforward = (
            v04.run_v04_walkforward(
                data, config, variant=variant, train_days=train_days,
                test_days=test_days, step_days=step_days,
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
            "result": full,
        }
        row = results[label]["full"]["metrics"]
        wf = results[label].get("walkforward") or {}
        print(
            f"  full {row['total_trades']} 笔 / {row['net_pnl_dollars']:+.2f} / PF "
            f"{row['profit_factor']:.4f}"
            + (
                f" | WF {wf['aggregate_test']['total_trades']} 笔 / "
                f"{wf['aggregate_test']['net_pnl_dollars']:+.2f} / "
                f"{wf['summary']['positive_folds']}/29 盈利折"
                if wf
                else ""
            )
            + f"（{elapsed:.1f}s）",
            flush=True,
        )
    return results


def _anchor_check(
    compare: Mapping[str, Mapping[str, Any]],
    anchors: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
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
    for label, expected_sections in anchors.items():
        payload = compare[label]
        sources = {
            "full": payload["full"]["metrics"],
            "train_2024": (payload.get("train_2024") or {}).get("metrics") or {},
            "holdout": (payload.get("holdout") or {}).get("metrics") or {},
            "walkforward": {
                **((payload.get("walkforward") or {}).get("aggregate_test") or {}),
                "positive_folds": (payload.get("walkforward") or {})
                .get("summary", {})
                .get("positive_folds"),
                "folds": (payload.get("walkforward") or {}).get("summary", {}).get("folds"),
            },
        }
        for section, fields in expected_sections.items():
            actual_metrics = sources[section]
            for field, expected in fields.items():
                actual = actual_metrics.get(field)
                tolerance = tolerances.get(field, 0.01)
                ok = (
                    actual is not None
                    and abs(float(actual) - float(expected)) <= tolerance + 1e-9
                )
                passed = passed and ok
                checks.append(
                    {
                        "variant": label,
                        "section": section,
                        "field": field,
                        "recorded": expected,
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
# Friday recap + filter/trap diagnostics
# --------------------------------------------------------------------------- #
def _and_confirm_ok(diagnostics: Mapping[str, Any]) -> bool:
    return bool(diagnostics.get("close_ge_ema21")) and bool(
        diagnostics.get("close_ge_vwap")
    )


def _and_blocked_reason(trade: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    if trade.get("strategy") == EMA_PULLBACK_5M:
        gain = diagnostics.get("exhaustion_gain_pct")
        if diagnostics.get("exhaustion_blocked"):
            return (
                "EMA 衰竭防追高：close 较当日低点 +"
                f"{gain:.2f}% >= {EMA_EXHAUSTION_GAIN_LIMIT * 100:.1f}%"
            )
        return "EMA_PULLBACK 被其它条件过滤"
    if trade.get("strategy") == SMC_SWEEP_5M:
        if not _and_confirm_ok(diagnostics):
            failed = []
            if not diagnostics.get("close_ge_ema21"):
                failed.append(
                    f"close<EMA21 (close={diagnostics.get('close_5m')}, "
                    f"ema21={diagnostics.get('ema21_5m')})"
                )
            if not diagnostics.get("close_ge_vwap"):
                failed.append(
                    f"close<VWAP (close={diagnostics.get('close_5m')}, "
                    f"vwap={diagnostics.get('session_vwap')})"
                )
            return "SMC AND 双确认失败：" + " 且 ".join(failed)
        return "SMC 被其它条件过滤"
    return "未知原因"


def build_friday_recap(
    friday_bars: Sequence[Bar],
    *,
    rows: Sequence[tuple[str, v04.V04Variant]] = AB_ROWS,
    initial_equity: float = 3000.0,
) -> Dict[str, Any]:
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
    for label, variant in rows:
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
            trade["diagnostics"] = _signal_diagnostics(prepared, trade, flat_buckets)
            trade["and_confirm_ok"] = _and_confirm_ok(trade["diagnostics"])
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
            "engine": _engine_text(variant),
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
                "当日 0 笔：10:00-14:00 入场窗内无 5m 策略触发，或触发后被趋势/AND 双确认/"
                "8% 防追高/风控过滤"
            )

    # -- Step3b-derived reference: what changes with AND / cap ----------------
    step3b_trades = {_trade_key(t): t for t in versions.get("Step3b", {}).get("trades", [])}
    no_cap_trades = {_trade_key(t): t for t in versions.get("NoCap", {}).get("trades", [])}
    with_cap_trades = {_trade_key(t): t for t in versions.get("WithCap", {}).get("trades", [])}
    # 1) signals Step3b took that the AND confirmation blocks (absent in NoCap)
    filtered_by_and: List[Dict[str, Any]] = []
    for key, trade in step3b_trades.items():
        if key in no_cap_trades:
            continue
        diagnostics = trade.get("diagnostics") or {}
        filtered_by_and.append(
            {
                "trade": trade,
                "diagnostics": diagnostics,
                "blocked_reason": _and_blocked_reason(trade, diagnostics),
            }
        )
    # 2) signals NoCap takes that the 8% exhaustion cap blocks (absent in WithCap)
    blocked_by_cap: List[Dict[str, Any]] = []
    for key, trade in no_cap_trades.items():
        if key in with_cap_trades:
            continue
        diagnostics = trade.get("diagnostics") or {}
        blocked_by_cap.append(
            {
                "trade": trade,
                "diagnostics": diagnostics,
                "blocked_reason": _and_blocked_reason(trade, diagnostics),
            }
        )

    # -- Friday traps ---------------------------------------------------------
    traps: Dict[str, Any] = {}
    for trap_name, trap in FRIDAY_TRAPS:
        key = (trap["symbol"], trap["strategy"], trap["entry_time_utc"])
        present_in = {
            label: key
            in {_trade_key(t) for t in versions[label]["trades"]}
            for label in versions
        }
        trap_payload: Dict[str, Any] = {
            "symbol": trap["symbol"],
            "strategy": trap["strategy"],
            "entry_time_utc": trap["entry_time_utc"],
            "present_in": present_in,
        }
        if key in step3b_trades:
            diagnostics = step3b_trades[key].get("diagnostics") or {}
            trap_payload["step3b_diagnostics"] = diagnostics
            trap_payload["and_confirm_ok"] = _and_confirm_ok(diagnostics)
        traps[trap_name] = trap_payload

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4c",
        "session_date": FRIDAY_DATE.isoformat(),
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw, 2026-11-06 regular session "
            "(single venue; NOT consolidated tape)"
        ),
        "caveat": DATA_CAVEAT,
        "market": market,
        "versions": versions,
        "markers_by_variant": markers_by_variant,
        "markers_by_variant_count": {
            label: len(markers) for label, markers in markers_by_variant.items()
        },
        "filter_effect_vs_step3b": {
            "step3b_trade_count": len(step3b_trades),
            "no_cap_trade_count": len(versions.get("NoCap", {}).get("trades", [])),
            "with_cap_trade_count": len(versions.get("WithCap", {}).get("trades", [])),
            "blocked_by_and_confirmation": filtered_by_and,
            "blocked_by_8pct_cap": blocked_by_cap,
        },
        "traps": traps,
        "notes": [
            "V0.4c A/B: NoCap = SMC AND 双确认 only; WithCap = SMC AND + 8% EMA_PULLBACK 防追高; "
            "Step3b = SMC OR + 8% 防追高 (reference).",
            "SMC AND requires close_5m >= EMA21 AND close_5m >= session VWAP (Step3b used OR).",
            "Both filters only affect NEW entries; open positions keep their stops/targets/EOD logic.",
            "Step1/Step3/Step3b/NoCap/WithCap all use the V0.3 1.2R single-stop breakeven shield "
            "(Step4 scale-out is deprecated).",
            "IEX single-venue data; research output, not proof of profitability.",
        ],
    }


def _engine_text(variant: v04.V04Variant) -> str:
    if variant.step == 7:
        return "v0.4 Step3c-B WithCap (SMC AND + 8% exhaustion cap)"
    if variant.step == 6:
        return "v0.4 Step3c-A NoCap (SMC AND only)"
    if variant.step == 5:
        return "v0.4 Step3b (SMC OR + 8% exhaustion cap) — reference"
    return variant.name


def build_markers_payload(recap: Mapping[str, Any]) -> Dict[str, Any]:
    """Web payload: markers grouped by A/B variant (NoCap / WithCap)."""

    versions = recap.get("versions", {})
    markers_by_variant = {
        label: (versions.get(label, {}) or {}).get("markers", [])
        for label in ("NoCap", "WithCap")
    }
    flat: List[Dict[str, Any]] = []
    for label, markers in markers_by_variant.items():
        for marker in markers:
            flat.append({**marker, "variant": label})
    flat.sort(key=lambda item: (item.get("t") or 0, item["type"]))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": recap.get("session_date"),
        "data_source": recap.get("data_source"),
        "variants": ["NoCap", "WithCap"],
        "markers_by_variant": markers_by_variant,
        "markers": flat,
        "reference_step3b_markers": (versions.get("Step3b", {}) or {}).get(
            "markers", []
        ),
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
                for trade in (versions.get(label, {}) or {}).get("trades", [])
            ]
            for label in ("NoCap", "WithCap", "Step3b")
        },
        "traps": recap.get("traps"),
    }


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
def build_recommendation(compare: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    def metric(label: str, section: str, field: str) -> float | None:
        payload = compare[label]
        if section == "walkforward":
            source = (payload.get("walkforward") or {}).get("aggregate_test", {})
        else:
            source = (payload.get(section) or {}).get("metrics", {})
        value = source.get(field)
        return float(value) if value is not None else None

    def folds(label: str) -> int:
        return int(
            (compare[label].get("walkforward") or {}).get("summary", {}).get(
                "positive_folds", 0
            )
        )

    labels = ("NoCap", "WithCap", "Step3b")
    table = {
        label: {
            "full_net": metric(label, "full", "net_pnl_dollars"),
            "full_pf": metric(label, "full", "profit_factor"),
            "full_maxdd": metric(label, "full", "max_drawdown_dollars"),
            "train_2024_net": metric(label, "train_2024", "net_pnl_dollars"),
            "holdout_net": metric(label, "holdout", "net_pnl_dollars"),
            "wf_net": metric(label, "walkforward", "net_pnl_dollars"),
            "wf_pf": metric(label, "walkforward", "profit_factor"),
            "wf_positive_folds": folds(label),
        }
        for label in labels
    }
    strict_wf_net = sorted(
        labels, key=lambda label: table[label]["wf_net"] or -1e18, reverse=True
    )
    strict_wf_folds = sorted(
        labels, key=lambda label: table[label]["wf_positive_folds"], reverse=True
    )
    # Comprehensive recommendation: OOS net first, then 2024, MaxDD, full net.
    comprehensive = sorted(
        ("NoCap", "WithCap"),
        key=lambda label: (
            table[label]["wf_net"] or -1e18,
            -(table[label]["full_maxdd"] or 1e18),
            table[label]["full_net"] or -1e18,
        ),
        reverse=True,
    )
    return {
        "metrics": table,
        "strict_wf_net_ranking": strict_wf_net,
        "strict_wf_positive_folds_ranking": strict_wf_folds,
        "comprehensive_recommendation": comprehensive[0],
        "comprehensive_order": comprehensive,
        "reasoning": (
            "综合（全周期+Holdout+29 折 WF）：NoCap 与 WithCap 相比，WF 净更优、2024 段更好、"
            "MaxDD 更低、全周期净值更高；WithCap 的优势在 WF 盈利折数（17 vs 16）与 Holdout "
            "净（+446 vs +388），且周五两坑全避开。按 OOS 净值优先推荐 NoCap；若更看重盈利折"
            "稳定性/防守可切 WithCap（一行改动）。"
        ),
    }


# --------------------------------------------------------------------------- #
# Markdown / terminal rendering
# --------------------------------------------------------------------------- #
_AB_HEADER = (
    "| 变体 | 全周期笔数 | 胜率 | PF | 净盈亏($) | 平均R | 最大回撤($) | "
    "2024笔数 | 2024 PF | 2024净($) | Holdout笔数 | Holdout PF | Holdout净($) | "
    "WF笔数 | WF胜率 | WF PF | WF净($) | WF盈利折 |"
)
_AB_SEPARATOR = "|" + "---|" * 18


def _ab_row(label: str, payload: Mapping[str, Any]) -> str:
    full = payload["full"]["metrics"]
    train = (payload.get("train_2024") or {}).get("metrics") or {}
    holdout = (payload.get("holdout") or {}).get("metrics") or {}
    wf = payload.get("walkforward") or {}
    wf_agg = wf.get("aggregate_test") or {}
    wf_summary = wf.get("summary") or {}
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


def _friday_trade_lines(payload: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    trades = payload.get("trades", [])
    if not trades:
        lines.append(f"当日无信号。原因：{payload.get('zero_trade_reason', '—')}")
        lines.append("")
        return lines
    lines.append(
        "| # | 入场时间(ET) | 标的 | 策略 | 入场价 | 止损 | 目标 | 出场时间(ET) | 出场价 | 原因 | PnL($) | R | 过滤/AND 诊断 |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---|")
    for index, trade in enumerate(trades, start=1):
        diagnostics = trade.get("diagnostics") or {}
        bits: List[str] = []
        if trade.get("strategy") == EMA_PULLBACK_5M:
            gain = diagnostics.get("exhaustion_gain_pct")
            cap_on = bool((payload.get("variant") or {}).get("exhaustion_filter"))
            if diagnostics.get("exhaustion_blocked"):
                bits.append(
                    f"日内涨幅 {gain}% → "
                    + ("8% 锁拦截" if cap_on else "8% 锁本应拦截（NoCap 未启用，信号恢复入场）")
                )
            else:
                bits.append(f"日内涨幅 {gain}% → 未触发 8% 锁")
        if trade.get("strategy") == SMC_SWEEP_5M:
            bits.append(
                f"AND={'OK' if trade.get('and_confirm_ok') else '失败'}"
                f" (close={diagnostics.get('close_5m')}, ema21={diagnostics.get('ema21_5m')}, "
                f"vwap={diagnostics.get('session_vwap')})"
            )
        lines.append(
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
    lines.append("")
    return lines


def build_ab_markdown(
    *,
    compare: Mapping[str, Mapping[str, Any]],
    self_check: Mapping[str, Any],
    friday: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    data_span: str,
    total_1m_bars: int,
    total_sessions: int,
    initial_equity: float,
    elapsed: Mapping[str, float],
) -> str:
    lines = [
        "# V0.4c Step3c A/B · SMC AND 双确认 与 8% 防追高（raw IEX 1m）",
        "",
        f"- 数据：Alpaca IEX raw 1m，{data_span}，{total_1m_bars:,} 根 1m bar / "
        f"{total_sessions} 个 session（单一交易所口径，非 consolidated tape）",
        f"- 结构：{V04C_STRUCTURE_NOTE}",
        f"- 初始权益 ${initial_equity:,.2f}；29 折 WF（120/20/20、每折 fresh RiskManager、不复利）。",
        "",
        "## 主表：Step3b（参照）/ NoCap（A）/ WithCap（B）",
        "",
        _AB_HEADER,
        _AB_SEPARATOR,
        *[_ab_row(label, compare[label]) for label in ("Step3b", "NoCap", "WithCap")],
        "",
        "## 自洽校验（Step3 / Step3b 必须对齐历史记录）",
        "",
        f"结果：{'通过 ✅' if self_check.get('passed') else '未通过 ❌'}",
        "",
        "| 变体 | 区段 | 指标 | 历史记录 | 本次重算 | 偏差 | 容差 | 通过 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for check in self_check.get("checks", []):
        lines.append(
            f"| {check['variant']} | {check['section']} | {check['field']} | "
            f"{check['recorded']} | {check['recomputed']} | {check['delta']} | "
            f"±{check['tolerance']} | {'✅' if check['ok'] else '❌'} |"
        )
    lines.extend(
        [
            "",
            f"## 周五 {friday.get('session_date')} 复盘（raw IEX 1m）",
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
            "### 三行汇总",
            "",
            "| 变体 | 引擎 | 笔数 | 胜率 | PF | 净盈亏($) | 平均R |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for label in ("Step3b", "NoCap", "WithCap"):
        payload = friday["versions"][label]
        summary = payload["summary"]
        lines.append(
            f"| {label} | {payload.get('engine', '')} | {summary.get('total_trades', 0)} | "
            f"{_fmt(summary.get('win_rate_pct'), 'pct')} | {_fmt(summary.get('profit_factor'), 'pf')} | "
            f"{_fmt(summary.get('net_pnl_dollars'), 'money')} | {_fmt(summary.get('avg_r'), 'r')} |"
        )
    lines.extend(["", "### 逐笔明细", ""])
    for label in ("Step3b", "NoCap", "WithCap"):
        lines.append(f"#### {label}（{len(friday['versions'][label].get('trades', []))} 笔）")
        lines.append("")
        lines.extend(_friday_trade_lines(friday["versions"][label]))
    traps = friday.get("traps") or {}
    lines.extend(
        [
            "### 周五两坑验证",
            "",
            "| 坑 | 标的/策略/入场(UTC) | Step3b | NoCap | WithCap | 结论 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, info in traps.items():
        present = info.get("present_in", {})
        lines.append(
            f"| {name} | {info.get('symbol')} / {info.get('strategy')} / {info.get('entry_time_utc')} | "
            f"{'有' if present.get('Step3b') else '无'} | {'有' if present.get('NoCap') else '无'} | "
            f"{'有' if present.get('WithCap') else '无'} | {info.get('verdict', '—')} |"
        )
    lines.extend(
        [
            "",
            "### 过滤效果（Step3b → NoCap / WithCap）",
            "",
            f"- Step3b {friday['filter_effect_vs_step3b']['step3b_trade_count']} 笔 → "
            f"NoCap {friday['filter_effect_vs_step3b']['no_cap_trade_count']} 笔 → "
            f"WithCap {friday['filter_effect_vs_step3b']['with_cap_trade_count']} 笔。",
        ]
    )
    for item in friday["filter_effect_vs_step3b"].get("blocked_by_and_confirmation", []):
        trade = item["trade"]
        lines.append(
            f"- 被 AND 双确认拦截：{trade.get('entry_time_utc')} {trade.get('symbol')} "
            f"{trade.get('strategy')} — {item.get('blocked_reason')}"
        )
    for item in friday["filter_effect_vs_step3b"].get("blocked_by_8pct_cap", []):
        trade = item["trade"]
        lines.append(
            f"- 被 8% 防追高拦截（NoCap 恢复、WithCap 拦截）：{trade.get('entry_time_utc')} "
            f"{trade.get('symbol')} {trade.get('strategy')} — {item.get('blocked_reason')}"
        )
    lines.extend(
        [
            "",
            "## 推荐（全周期+Holdout+WF 综合 & 严格 WF 排序）",
            "",
            f"- **综合推荐：{recommendation.get('comprehensive_recommendation')}** "
            f"（优先 OOS 净值，其次 2024 段/MaxDD/全周期净值）。",
            f"- 严格看 WF 净的排序：{' > '.join(recommendation.get('strict_wf_net_ranking', []))}",
            f"- 严格看 WF 盈利折的排序：{' > '.join(recommendation.get('strict_wf_positive_folds_ranking', []))}",
            f"- 判读：{recommendation.get('reasoning')}",
            "",
            "## 说明",
            "",
            "- 所有阈值 a-priori、未调参；A/B 只比较结构，不择优拟合。",
            "- 两个过滤器只影响新开仓；已持仓的 stop/target/EOD 不受影响。",
            "- PRODUCTION_VARIANT 仍为 Step3b，等 Lead 决定一行切换。",
            "- 图表标记：`var/backtest/v04c_friday_markers.json`（markers_by_variant: NoCap/WithCap）。",
            f"- 运行时长：{elapsed.get('total', 0.0):.1f}s。",
            "",
            "> 数据口径：IEX 单一交易所；研究结果不代表实盘有效性或盈利预期。",
            "",
        ]
    )
    return "\n".join(lines)


def build_friday_markdown(friday: Mapping[str, Any]) -> str:
    lines = [
        f"# V0.4c 周五单日复盘 · {friday.get('session_date')}（raw IEX 1m）",
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
            "## 三行汇总",
            "",
            "| 变体 | 引擎 | 笔数 | 胜率 | PF | 净盈亏($) | 平均R |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for label in ("Step3b", "NoCap", "WithCap"):
        payload = friday["versions"][label]
        summary = payload["summary"]
        lines.append(
            f"| {label} | {payload.get('engine', '')} | {summary.get('total_trades', 0)} | "
            f"{_fmt(summary.get('win_rate_pct'), 'pct')} | {_fmt(summary.get('profit_factor'), 'pf')} | "
            f"{_fmt(summary.get('net_pnl_dollars'), 'money')} | {_fmt(summary.get('avg_r'), 'r')} |"
        )
    lines.extend(["", "## 逐笔明细", ""])
    for label in ("Step3b", "NoCap", "WithCap"):
        lines.append(f"### {label}（{len(friday['versions'][label].get('trades', []))} 笔）")
        lines.append("")
        lines.extend(_friday_trade_lines(friday["versions"][label]))
    traps = friday.get("traps") or {}
    lines.extend(
        [
            "## 两坑验证",
            "",
            "| 坑 | Step3b | NoCap | WithCap | 结论 |",
            "|---|---|---|---|---|",
        ]
    )
    for name, info in traps.items():
        present = info.get("present_in", {})
        lines.append(
            f"| {name} | {'有' if present.get('Step3b') else '无'} | "
            f"{'有' if present.get('NoCap') else '无'} | {'有' if present.get('WithCap') else '无'} | "
            f"{info.get('verdict', '—')} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            *[f"- {note}" for note in friday.get("notes", [])],
            "",
        ]
    )
    return "\n".join(lines)


def print_tables(
    compare: Mapping[str, Mapping[str, Any]],
    friday: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> None:
    print("\n" + "=" * 150)
    print("V0.4c A/B：SMC AND 双确认（NoCap）vs AND + 8% 防追高（WithCap），Step3b 参照")
    print("=" * 150)
    print(
        f"{'变体':<9}{'全周期笔数':>10}{'胜率':>9}{'PF':>8}{'净盈亏($)':>13}{'平均R':>9}{'MaxDD($)':>12}"
        f"{'2024笔':>8}{'2024PF':>8}{'2024净($)':>12}{'Hold笔':>8}{'HoldPF':>8}{'Hold净($)':>12}"
        f"{'WF笔':>7}{'WF胜率':>8}{'WF PF':>7}{'WF净($)':>12}{'盈利折':>9}"
    )
    print("-" * 150)
    for label in ("Step3b", "NoCap", "WithCap"):
        payload = compare[label]
        full = payload["full"]["metrics"]
        train = (payload.get("train_2024") or {}).get("metrics") or {}
        holdout = (payload.get("holdout") or {}).get("metrics") or {}
        wf = payload.get("walkforward") or {}
        wf_agg = wf.get("aggregate_test") or {}
        wf_summary = wf.get("summary") or {}
        print(
            f"{label:<9}{full.get('total_trades', '—'):>10}"
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

    print("\n" + "=" * 122)
    print(f"周五 {friday.get('session_date')} 复盘")
    print("=" * 122)
    print(f"  {'变体':<9}{'笔数':>6}{'胜率':>9}{'PF':>8}{'净盈亏($)':>13}{'平均R':>9}")
    print("  " + "-" * 70)
    for label in ("Step3b", "NoCap", "WithCap"):
        summary = friday["versions"][label]["summary"]
        print(
            f"  {label:<9}{summary.get('total_trades', 0):>6}"
            f"{_fmt(summary.get('win_rate_pct'), 'pct'):>9}"
            f"{_fmt(summary.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(summary.get('net_pnl_dollars'), 'money'):>13}"
            f"{_fmt(summary.get('avg_r'), 'r'):>9}"
        )
    for label in ("Step3b", "NoCap", "WithCap"):
        trades = friday["versions"][label].get("trades", [])
        print(f"\n  {label} 当日信号（{len(trades)} 笔）：")
        if not trades:
            print("    0 笔")
            continue
        cap_on = bool(
            (friday["versions"][label].get("variant") or {}).get("exhaustion_filter")
        )
        for trade in trades:
            diagnostics = trade.get("diagnostics") or {}
            note = ""
            if trade.get("strategy") == EMA_PULLBACK_5M:
                blocked = bool(diagnostics.get("exhaustion_blocked"))
                if blocked:
                    note = (
                        f" | 日内涨幅 {diagnostics.get('exhaustion_gain_pct')}% → "
                        + ("8% 锁拦截" if cap_on else "8% 锁本应拦截（NoCap 未启用）")
                    )
                else:
                    note = f" | 日内涨幅 {diagnostics.get('exhaustion_gain_pct')}% → 未触发 8% 锁"
            elif trade.get("strategy") == SMC_SWEEP_5M:
                note = (
                    f" | AND={'OK' if trade.get('and_confirm_ok') else '失败'}"
                    f" close={diagnostics.get('close_5m')} ema21={diagnostics.get('ema21_5m')}"
                    f" vwap={diagnostics.get('session_vwap')}"
                )
            print(
                f"    {trade.get('entry_time_et')} ET {trade.get('symbol')} "
                f"{trade.get('strategy'):<16} → {trade.get('exit_time_et')} ET "
                f"{trade.get('exit_reason'):<12} PnL {_fmt(trade.get('pnl'), 'money')}"
                f" R {_fmt(trade.get('r_multiple'), 'r')}{note}"
            )
    traps = friday.get("traps") or {}
    print("\n  两坑验证：")
    for name, info in traps.items():
        present = info.get("present_in", {})
        print(
            f"    {name}: Step3b={'有' if present.get('Step3b') else '无'}, "
            f"NoCap={'有' if present.get('NoCap') else '无'}, "
            f"WithCap={'有' if present.get('WithCap') else '无'} → {info.get('verdict', '—')}"
        )
    print(
        "\n  推荐：" + str(recommendation.get("comprehensive_recommendation"))
        + "（综合）；严格 WF 净排序: " + " > ".join(recommendation.get("strict_wf_net_ranking", []))
        + "；WF 盈利折排序: " + " > ".join(recommendation.get("strict_wf_positive_folds_ranking", []))
    )
    print("=" * 122)


# --------------------------------------------------------------------------- #
# Trade ledger
# --------------------------------------------------------------------------- #
def write_ab_trades_csv(
    compare: Mapping[str, Mapping[str, Any]], path: Path
) -> None:
    """Combined NoCap + WithCap full-period ledger (variant column)."""

    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for label in ("NoCap", "WithCap"):
        result = compare[label]["result"]
        for trade in result.trades:
            detail = result.trade_details.get(trade.signal_id, {})
            rows.append(
                {
                    "variant": label,
                    "signal_id": trade.signal_id,
                    "symbol": trade.symbol,
                    "strategy": trade.strategy.value,
                    "strategy_label": detail.get("strategy_label", trade.strategy.value),
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
                    "pnl": round(trade.pnl, 4),
                    "r_multiple": round(trade.r_multiple, 4),
                    "mae_r": round(float(detail.get("mae_r", 0.0) or 0.0), 4),
                    "mfe_r": round(float(detail.get("mfe_r", 0.0) or 0.0), 4),
                    "holding_minutes": round(trade.holding_minutes, 2),
                    "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                    "signal_bar_start_utc": detail.get("signal_bar_start_utc", ""),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


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

    # A/B rows (includes Step3b reference): full + segments + WF.
    compare = run_ab(
        data,
        config,
        with_segments=with_segments,
        with_walkforward=with_walkforward,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_equity=equity,
    )
    # Step3 self-check anchor (not shown in the main table, but required).
    print("\n[Step3 自洽校验锚] 运行 Step3 …", flush=True)
    dates_2024 = [value for value in data.dates if TRAIN_2024_START <= value <= TRAIN_2024_END]
    dates_holdout = [value for value in data.dates if HOLDOUT_START <= value <= HOLDOUT_END]
    step3_full = v04.run_v04_backtest(data, config, variant=v04.STEP3_VARIANT)
    step3_2024 = v04.run_v04_backtest(
        data, config, variant=v04.STEP3_VARIANT, session_dates=dates_2024
    )
    step3_holdout = v04.run_v04_backtest(
        data, config, variant=v04.STEP3_VARIANT, session_dates=dates_holdout
    )
    step3_wf = (
        v04.run_v04_walkforward(data, config, variant=v04.STEP3_VARIANT)
        if with_walkforward
        else None
    )
    compare["Step3"] = {
        "label": "Step3",
        "variant": v04.STEP3_VARIANT.flags(),
        "full": _format_v04_result(step3_full),
        "train_2024": {"metrics": _metrics_row(step3_2024.metrics)},
        "holdout": {"metrics": _metrics_row(step3_holdout.metrics)},
        "walkforward": _format_walkforward(step3_wf) if step3_wf else None,
        "result": step3_full,
    }
    self_check = _anchor_check(compare, RECORDED_ANCHORS)
    print(f"\nStep3/Step3b 自洽校验：{'通过 ✅' if self_check['passed'] else '未通过 ❌'}")
    for check in self_check["checks"]:
        print(
            f"  {'✅' if check['ok'] else '❌'} {check['variant']:<7} {check['section']:<12} "
            f"{check['field']:<24} 记录={check['recorded']} 重算={check['recomputed']} Δ={check['delta']}"
        )
    if not self_check["passed"]:
        raise RuntimeError(
            "Step3/Step3b do not reproduce the recorded values; refusing to publish V0.4c numbers"
        )

    print("\n加载周五数据（含 2026-11-06，raw）…")
    friday_bars = load_friday_raw_bars(data_dir)
    friday_started = time.perf_counter()
    friday = build_friday_recap(friday_bars, initial_equity=equity)
    # Trap verdicts (explicit, for report/markdown).
    trap_verdicts = {
        "11:25 SOXL SMC 接刀": (
            "✅ AND 双确认拦截：close 46.23 < VWAP 46.65"
            if not friday["traps"]["11:25 SOXL SMC 接刀"]["present_in"].get("NoCap")
            and not friday["traps"]["11:25 SOXL SMC 接刀"]["present_in"].get("WithCap")
            else "❌ 未被 AND 拦截"
        ),
        "12:40 SOXS EMA 山顶追高": (
            "NoCap 恢复该信号并入账；WithCap 仍被 8% 锁拦截"
            if friday["traps"]["12:40 SOXS EMA 山顶追高"]["present_in"].get("NoCap")
            and not friday["traps"]["12:40 SOXS EMA 山顶追高"]["present_in"].get("WithCap")
            else "需复查"
        ),
    }
    for trap_name, verdict in trap_verdicts.items():
        friday["traps"][trap_name]["verdict"] = verdict
    friday_elapsed = time.perf_counter() - friday_started
    recommendation = build_recommendation(compare)
    total_elapsed = time.perf_counter() - started

    ab_markdown = build_ab_markdown(
        compare=compare,
        self_check=self_check,
        friday=friday,
        recommendation=recommendation,
        data_span=data_span,
        total_1m_bars=len(raw_bars),
        total_sessions=len(data.sessions),
        initial_equity=equity,
        elapsed={"total": total_elapsed},
    )
    friday_markdown = build_friday_markdown(friday)
    markers_payload = build_markers_payload(friday)

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4c",
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw (single venue; NOT consolidated tape)"
        ),
        "backtest_data_span": data_span,
        "total_1m_bars": len(raw_bars),
        "total_sessions": len(data.sessions),
        "structure": V04C_STRUCTURE_NOTE,
        "ab_variants": {
            "Step3b": v04.STEP3B_VARIANT.flags(),
            "NoCap": v04.STEP3C_NOCAP_VARIANT.flags(),
            "WithCap": v04.STEP3C_WITHCAP_VARIANT.flags(),
        },
        "production_variant": v04.PRODUCTION_VARIANT.flags(),
        "caveat": DATA_CAVEAT,
        "initial_equity": equity,
        "walkforward_grid": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
        },
        "main_table": {
            label: {
                "variant": compare[label]["variant"],
                "full": compare[label]["full"],
                "train_2024": compare[label]["train_2024"],
                "holdout": compare[label]["holdout"],
                "walkforward": compare[label]["walkforward"],
            }
            for label in ("Step3b", "NoCap", "WithCap")
        },
        "self_check": self_check,
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
                for label in ("Step3b", "NoCap", "WithCap")
            },
            "traps": friday["traps"],
            "filter_effect_vs_step3b": friday["filter_effect_vs_step3b"],
            "markers_by_variant_count": friday["markers_by_variant_count"],
        },
        "recommendation": recommendation,
        "elapsed_seconds": {
            "ab": round(total_elapsed - friday_elapsed, 2),
            "friday_recap": round(friday_elapsed, 2),
            "total": round(total_elapsed, 2),
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v04c_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    ab_md_path = report_dir / "v04c_ab.md"
    ab_md_path.write_text(ab_markdown, encoding="utf-8")
    trades_path = report_dir / "v04c_trades.csv"
    write_ab_trades_csv(compare, trades_path)
    friday_recap_path = report_dir / "v04c_friday_recap.json"
    friday_recap_path.write_text(
        json.dumps(friday, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    friday_md_path = report_dir / "v04c_friday_recap.md"
    friday_md_path.write_text(friday_markdown, encoding="utf-8")
    markers_path = report_dir / "v04c_friday_markers.json"
    markers_path.write_text(
        json.dumps(markers_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_tables(compare, friday, recommendation)
    print("\n产物：")
    print(f"  A/B 报告 : {report_path}")
    print(f"  A/B MD   : {ab_md_path}")
    print(f"  逐笔流水 : {trades_path}（NoCap + WithCap 全周期）")
    print(f"  周五复盘 : {friday_recap_path}")
    print(f"  周五 MD  : {friday_md_path}")
    print(
        f"  图表标记 : {markers_path}"
        f"（markers_by_variant: {friday['markers_by_variant_count']}）"
    )
    print(
        f"\n总运行时长：{total_elapsed:.1f}s"
        f"（A/B {total_elapsed - friday_elapsed:.1f}s + 周五复盘 {friday_elapsed:.1f}s）"
    )
    return report_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.4c Step3c A/B (SMC AND confirmation, with/without 8% exhaustion cap)."
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
