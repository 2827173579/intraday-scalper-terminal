"""V0.4 structural ablation + Friday single-session recap (raw IEX 1m bars).

Runs two programs on the unadjusted (``adjustment=raw``) Alpaca IEX cache:

1. **Five cumulative ablation steps** on the full raw backtest dataset
   (2024-01-02 .. 2026-11-05, 483,751 raw 1m bars, 714 sessions):
   Step 0 = V0.3 raw STEP4 baseline anchor (970 trades / -$214.76 /
   WF 814 trades / -$234.92 / 16 folds), then + 10:00-14:00 ET entry window,
   + VWAP_RECLAIM trend filter, + SMC_SWEEP_5M, + 1.2R scale-out.
   Each step also runs the 2024 in-sample segment, the 2025-2026 holdout and the
   same 29-fold walk-forward grid (120/20/20, fresh risk manager per fold).
2. **Friday 2026-11-06 single-session recap** comparing V0.1 (production 1m
   engine), V0.3 (v0.3 STEP4) and V0.4 (final) with per-trade detail plus
   bar-by-bar markers for the Web replay chart.

Artifacts (``intraday_scalper/var/backtest/``)::

    v04_report.json        # 5-step ablation detail + Step0 self-check
    v04_ablation.md        # markdown ablation table (REVIEW_PACK ready)
    v04_trades.csv         # final step full-period aggregated trade ledger
    v04_friday_recap.json  # Friday V0.1/V0.3/V0.4 recap (Web consumes .versions)
    v04_friday_recap.md    # Friday markdown recap
    v04_friday_markers.json# Friday V0.4 markers (Web replays on the chart)

Data caveat: Alpaca IEX-only single venue (small share of consolidated volume);
this is research, not proof of profitability.

Usage::

    python -m intraday_scalper.tools.run_v04_backtest
    python -m intraday_scalper.tools.run_v04_backtest --no-walkforward
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from intraday_scalper.backtest.engine import run_backtest
from intraday_scalper.backtest.metrics import compute_metrics
from intraday_scalper.core import Bar, et_date, is_regular_session, load_config, to_et
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v02, v03, v04
from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    EMA_PULLBACK_5M,
    ENUM_TO_LABEL,
    VWAP_RECLAIM_5M,
    prepare_v02_data,
)
from intraday_scalper.research.v04 import SMC_SWEEP_5M
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.strategies.registry import build_strategies
from intraday_scalper.tools.run_v03_raw_crosscheck import (
    _frames_to_bars,
    load_raw_bars,
)

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

RAW_FULL_GLOB = "*_1m_2024_2026_raw.parquet"
FRIDAY_RAW_GLOB = "*_1m_raw.parquet"
FRIDAY_DATE = date(2026, 11, 6)

TRAIN_2024_START = date(2024, 1, 1)
TRAIN_2024_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 11, 5)

#: Recorded V0.3 raw_full STEP4 anchor (``v03_raw_crosscheck.json``).
V03_RAW_STEP4_ANCHOR: Dict[str, Any] = {
    "full": {
        "total_trades": 970,
        "win_rate_pct": 43.51,
        "profit_factor": 0.9772,
        "net_pnl_dollars": -214.76,
        "avg_r": -0.0143,
        "max_drawdown_dollars": 628.85,
    },
    "train_2024": {"total_trades": 313, "profit_factor": 1.0074, "net_pnl_dollars": 21.80},
    "holdout": {"total_trades": 657, "profit_factor": 0.9634, "net_pnl_dollars": -236.56},
    "walkforward": {
        "total_trades": 814,
        "win_rate_pct": 43.86,
        "profit_factor": 0.9704,
        "net_pnl_dollars": -234.92,
        "positive_folds": 16,
        "folds": 29,
    },
}

V04_STRUCTURE_NOTE = (
    "V0.4 在 V0.3 之上做四项一次性先验结构改动（a-priori, not tuned），全部跑在 "
    "adjustment=raw 未复权 IEX 1m 数据上：① 10:00-14:00 ET 入场窗（09:30-10:00 只用于 "
    "OR_30M / Session VWAP / 指标）；② VWAP_RECLAIM_5M 趋势过滤（session 振幅 >=1.2×ATR14，"
    "SOXL 要求 EMA9>EMA21，SOXS 用镜像 SOXL EMA9<EMA21）；③ 新增 SMC_SWEEP_5M（下影线刺穿"
    "前20根5m低点/OR30低点后收回，波幅 >=1.2×前20均幅，镜像共振；止损=近端支撑-0.25×ATR，"
    "clamp 0.25%-2%，目标 2R）；④ +1.2R 分批止盈（50% 在 1.2R 限价、剩余止损移至 entry+0.10R、"
    "剩余目标 2R、5m 收盘跌破 EMA9 下一根 1m 开盘平剩余）。Step0 = V0.3 raw STEP4 基线。"
    "29 折 WF 纪律（120/20/20、每折全新 RiskManager、不复利）与 V0.2/V0.3 完全一致。"
)

MARKER_TYPES = ("ENTRY", "STOP", "BREAKEVEN", "TARGET", "EXIT")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _number(value: Any, digits: int = 2) -> float | None:
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


def _dt_et(ts: datetime | None) -> str:
    return to_et(ts).strftime("%H:%M:%S") if ts is not None else "—"


def _metrics_row(metrics: Any) -> Dict[str, Any]:
    return {
        "total_trades": int(metrics.total_trades),
        "wins": int(metrics.wins),
        "losses": int(metrics.losses),
        "win_rate_pct": _pct(metrics.win_rate),
        "profit_factor": _number(metrics.profit_factor, 4),
        "net_pnl_dollars": _number(metrics.net_pnl_dollars),
        "avg_r": _number(metrics.avg_r, 4),
        "avg_holding_minutes": _number(metrics.avg_holding_minutes, 1),
        "max_drawdown_dollars": _number(metrics.max_drawdown_dollars),
        "max_drawdown_pct": _pct(metrics.max_drawdown_pct),
        "trades_per_day": _number(metrics.trades_per_day, 2),
        "trading_days": int(metrics.trading_days),
        "total_slippage_dollars": _number(metrics.total_slippage_dollars),
    }


def _format_v04_result(result: v04.V04Result) -> Dict[str, Any]:
    metrics = result.metrics
    details = result.trade_details
    per_strategy: Dict[str, Dict[str, Any]] = {}
    for trade in result.trades:
        label = details.get(trade.signal_id, {}).get(
            "strategy_label", trade.strategy.value
        )
        bucket = per_strategy.setdefault(
            label, {"trades": 0, "wins": 0, "pnl": 0.0, "r_sum": 0.0}
        )
        bucket["trades"] += 1
        bucket["wins"] += 1 if trade.pnl > 0 else 0
        bucket["pnl"] += trade.pnl
        bucket["r_sum"] += trade.r_multiple
    per_strategy_out = {
        label: {
            "trades": bucket["trades"],
            "win_rate_pct": _pct(bucket["wins"] / bucket["trades"])
            if bucket["trades"]
            else 0.0,
            "net_pnl_dollars": _number(bucket["pnl"]),
            "avg_r": _number(bucket["r_sum"] / bucket["trades"], 4)
            if bucket["trades"]
            else 0.0,
        }
        for label, bucket in sorted(per_strategy.items())
    }
    exit_reasons: Dict[str, int] = {}
    for trade in result.trades:
        reason = trade.exit_reason or "UNKNOWN"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    return {
        "variant": result.variant.flags(),
        "metrics": _metrics_row(metrics),
        "extra": result.extra_metrics(),
        "exit_reasons": dict(sorted(exit_reasons.items())),
        "per_strategy": per_strategy_out,
        "per_symbol": {
            symbol: {
                "trades": group.trades,
                "win_rate_pct": _pct(group.win_rate),
                "profit_factor": _number(group.profit_factor, 4),
                "net_pnl_dollars": _number(group.net_pnl_dollars),
                "avg_r": _number(group.avg_r, 4),
            }
            for symbol, group in metrics.per_symbol.items()
        },
        "session_dates": len(result.session_dates),
    }


def _format_walkforward(wf: v04.V04WalkForwardResult) -> Dict[str, Any]:
    fold_rows: List[Dict[str, Any]] = []
    for fold_result in wf.folds:
        test_metrics = fold_result.test_result.metrics
        fold_rows.append(
            {
                "fold": fold_result.fold.index,
                "train_start": fold_result.fold.train_start.isoformat()
                if fold_result.fold.train_start
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
                "test_profit_factor": _number(test_metrics.profit_factor, 4),
                "test_net_pnl_dollars": _number(test_metrics.net_pnl_dollars),
                "test_max_drawdown_dollars": _number(
                    test_metrics.max_drawdown_dollars
                ),
                "leakage_checked": fold_result.leakage_checked,
            }
        )
    summary = dict(wf.summary)
    for key, digits in (
        ("test_win_rate", 6),
        ("test_profit_factor", 4),
        ("test_net_pnl_dollars", 2),
        ("test_avg_r", 4),
        ("test_max_drawdown_dollars_max_fold", 2),
        ("test_max_drawdown_pct_max_fold", 6),
        ("test_trades_per_session", 2),
    ):
        if key in summary:
            summary[key] = _number(summary[key], digits)
    return {
        "variant": wf.variant.flags(),
        "aggregate_test": _metrics_row(wf.aggregate_test_metrics),
        "summary": summary,
        "fold_rows": fold_rows,
    }


# --------------------------------------------------------------------------- #
# Step-0 anchor self-check
# --------------------------------------------------------------------------- #
def _anchor_checks(
    v03_result: v03.V03Result,
    v03_wf: v03.V03WalkForwardResult,
    v04_step0: v04.V04Result,
    v04_step0_wf: v04.V04WalkForwardResult,
    seg_2024: v03.V03Result,
    holdout: v03.V03Result,
) -> Dict[str, Any]:
    tolerances = {
        "total_trades": 0.0,
        "win_rate_pct": 0.01,
        "profit_factor": 0.001,
        "net_pnl_dollars": 0.01,
        "avg_r": 0.001,
        "max_drawdown_dollars": 0.01,
        "positive_folds": 0.0,
    }
    sources = {
        "full": (_metrics_row(v03_result.metrics), _format_v04_result(v04_step0)["metrics"]),
        "train_2024": (_metrics_row(seg_2024.metrics), None),
        "holdout": (_metrics_row(holdout.metrics), None),
        "walkforward": (
            {
                **_metrics_row(v03_wf.aggregate_test_metrics),
                "positive_folds": int(v03_wf.summary["positive_folds"]),
            },
            {
                **_format_walkforward(v04_step0_wf)["aggregate_test"],
                "positive_folds": int(v04_step0_wf.summary["positive_folds"]),
            },
        ),
    }
    checks: List[Dict[str, Any]] = []
    passed = True
    for section, expected_fields in V03_RAW_STEP4_ANCHOR.items():
        if section == "walkforward":
            v03_recomputed = {
                **sources["walkforward"][0],
                "folds": len(v03_wf.folds),
            }
            v04_recomputed = {
                **sources["walkforward"][1],
                "folds": len(v04_step0_wf.folds),
            }
        else:
            v03_recomputed = sources[section][0]
            v04_recomputed = None
        for field, expected in expected_fields.items():
            actual = v03_recomputed.get(field)
            tolerance = tolerances.get(field, 0.01)
            ok = (
                actual is not None
                and abs(float(actual) - float(expected)) <= tolerance + 1e-9
            )
            v04_actual = (
                v04_recomputed.get(field) if v04_recomputed is not None else None
            )
            v04_ok = (
                v04_actual is not None
                and abs(float(v04_actual) - float(actual)) <= tolerance + 1e-9
            )
            passed = passed and ok and (v04_ok if v04_actual is not None else True)
            checks.append(
                {
                    "section": section,
                    "field": field,
                    "recorded_v03_raw_step4": expected,
                    "v03_recomputed": actual,
                    "v04_step0_recomputed": v04_actual,
                    "delta_vs_recorded": _number(float(actual) - float(expected), 6)
                    if actual is not None
                    else None,
                    "tolerance": tolerance,
                    "ok": bool(ok),
                    "v04_step0_ok": bool(v04_ok) if v04_actual is not None else None,
                }
            )
    return {"passed": bool(passed), "checks": checks}


# --------------------------------------------------------------------------- #
# Ablation runner
# --------------------------------------------------------------------------- #
def _fmt_span(dates: Sequence[date]) -> str:
    return f"{min(dates):%Y-%m-%d} .. {max(dates):%Y-%m-%d}" if dates else "—"


def _run_ablation(
    data: v02.V02Data,
    config: Any,
    *,
    with_segments: bool,
    with_walkforward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
    initial_equity: float,
) -> Dict[str, Any]:
    dates = list(data.dates)
    dates_2024 = [value for value in dates if TRAIN_2024_START <= value <= TRAIN_2024_END]
    dates_holdout = [value for value in dates if HOLDOUT_START <= value <= HOLDOUT_END]
    print(
        f"分段：2024 {len(dates_2024)} sessions / holdout {len(dates_holdout)} sessions / "
        f"total {len(dates)} sessions"
    )

    # -- Step 0 anchor: the authoritative V0.3 raw STEP4 engine path ----------
    print("\n[Step 0 锚] 运行 V0.3 raw STEP4 基线（v03 engine，权威锚）…", flush=True)
    started = time.perf_counter()
    v03_full = v03.run_v03_backtest(data, config, variant=v03.STEP4_VARIANT)
    v03_2024 = (
        v03.run_v03_backtest(
            data, config, variant=v03.STEP4_VARIANT, session_dates=dates_2024
        )
        if with_segments
        else None
    )
    v03_holdout = (
        v03.run_v03_backtest(
            data, config, variant=v03.STEP4_VARIANT, session_dates=dates_holdout
        )
        if with_segments
        else None
    )
    v03_wf = (
        v03.run_v03_walkforward(data, config, variant=v03.STEP4_VARIANT)
        if with_walkforward
        else None
    )
    v03_elapsed = time.perf_counter() - started
    print(
        f"  V0.3 raw STEP4: {v03_full.metrics.total_trades} 笔 / "
        f"{v03_full.metrics.net_pnl_dollars:+.2f} / PF "
        f"{v03_full.metrics.profit_factor:.4f}（{v03_elapsed:.1f}s）",
        flush=True,
    )

    # -- V0.4 Step 0 through the V0.4 engine (equivalence check) --------------
    print("[Step 0 锚] 运行 V0.4 Step0（应逐笔等价 V0.3 STEP4）…", flush=True)
    step0 = v04.run_v04_backtest(data, config, variant=v04.STEP0_VARIANT)
    step0_wf = (
        v04.run_v04_walkforward(data, config, variant=v04.STEP0_VARIANT)
        if with_walkforward
        else None
    )
    print(
        f"  V0.4 Step0: {step0.metrics.total_trades} 笔 / "
        f"{step0.metrics.net_pnl_dollars:+.2f} / PF {step0.metrics.profit_factor:.4f}",
        flush=True,
    )

    steps: List[Dict[str, Any]] = []
    step0_payload = {
        "step": 0,
        "name": v04.STEP0_VARIANT.name,
        "flags": v04.STEP0_VARIANT.flags(),
        "engine": "v0.3 STEP4 (authoritative anchor)",
        "full": {
            "metrics": _metrics_row(v03_full.metrics),
            "extra": v03_full.extra_metrics(),
            "per_strategy": {},
            "per_symbol": {},
            "session_dates": len(v03_full.session_dates),
        },
        "train_2024": {
            "metrics": _metrics_row(v03_2024.metrics) if v03_2024 else None,
            "session_dates": len(v03_2024.session_dates) if v03_2024 else 0,
        },
        "holdout": {
            "metrics": _metrics_row(v03_holdout.metrics) if v03_holdout else None,
            "session_dates": len(v03_holdout.session_dates) if v03_holdout else 0,
        },
        "walkforward": _format_walkforward(v03_wf) if v03_wf else None,
        "v04_step0_check": {
            "full_metrics": _format_v04_result(step0)["metrics"],
            "walkforward": _format_walkforward(step0_wf) if step0_wf else None,
            "trade_count_equal": len(step0.trades) == len(v03_full.trades),
            "net_equal_within_cent": abs(
                step0.metrics.net_pnl_dollars - v03_full.metrics.net_pnl_dollars
            )
            <= 0.01,
        },
        "elapsed_seconds": round(v03_elapsed, 2),
    }
    steps.append(step0_payload)

    # -- V0.4 steps 1-4 -------------------------------------------------------
    for variant in v04.ABLATION_STEPS[1:]:
        print(f"\n[{variant.name}] …", flush=True)
        started = time.perf_counter()
        full = v04.run_v04_backtest(data, config, variant=variant)
        seg_2024 = (
            v04.run_v04_backtest(
                data, config, variant=variant, session_dates=dates_2024
            )
            if with_segments
            else None
        )
        holdout = (
            v04.run_v04_backtest(
                data, config, variant=variant, session_dates=dates_holdout
            )
            if with_segments
            else None
        )
        walkforward = (
            v04.run_v04_walkforward(data, config, variant=variant)
            if with_walkforward
            else None
        )
        elapsed = time.perf_counter() - started
        payload = {
            "step": variant.step,
            "name": variant.name,
            "flags": variant.flags(),
            "engine": "v0.4",
            "full": _format_v04_result(full),
            "train_2024": {
                "metrics": _metrics_row(seg_2024.metrics) if seg_2024 else None,
                "extra": seg_2024.extra_metrics() if seg_2024 else None,
                "session_dates": len(seg_2024.session_dates) if seg_2024 else 0,
            },
            "holdout": {
                "metrics": _metrics_row(holdout.metrics) if holdout else None,
                "extra": holdout.extra_metrics() if holdout else None,
                "session_dates": len(holdout.session_dates) if holdout else 0,
            },
            "walkforward": _format_walkforward(walkforward) if walkforward else None,
            "elapsed_seconds": round(elapsed, 2),
        }
        steps.append(payload)
        print(
            f"  full {payload['full']['metrics']['total_trades']} 笔 / "
            f"{payload['full']['metrics']['net_pnl_dollars']:+.2f} / PF "
            f"{payload['full']['metrics']['profit_factor']:.4f}"
            + (
                f" | WF {payload['walkforward']['aggregate_test']['total_trades']} 笔 / "
                f"{payload['walkforward']['aggregate_test']['net_pnl_dollars']:+.2f} / "
                f"{payload['walkforward']['summary']['positive_folds']}/{payload['walkforward']['summary']['folds']} 盈利折"
                if walkforward is not None
                else ""
            )
            + f"（{elapsed:.1f}s）",
            flush=True,
        )

    self_check = _anchor_checks(
        v03_full,
        v03_wf,
        step0,
        step0_wf,
        v03_2024,
        v03_holdout,
    ) if (v03_wf is not None and step0_wf is not None and v03_2024 is not None and v03_holdout is not None) else {
        "passed": False,
        "checks": [],
        "note": "walk-forward or segments were skipped; anchor not fully verified",
    }
    return {"steps": steps, "self_check": self_check}


# --------------------------------------------------------------------------- #
# Friday recap
# --------------------------------------------------------------------------- #
def load_friday_raw_bars(data_dir: Path = DATA_DIR) -> List[Bar]:
    files = sorted(data_dir.glob(FRIDAY_RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 {FRIDAY_RAW_GLOB}")
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        print(f"加载 {path.name}: {len(frame):,} 行")
    return _frames_to_bars(frames)


def _market_snapshot(friday_bars: Sequence[Bar]) -> Dict[str, Any]:
    market: Dict[str, Any] = {}
    for symbol in ("SOXL", "SOXS"):
        symbol_bars = sorted(
            (
                bar
                for bar in friday_bars
                if bar.symbol == symbol and et_date(bar.start_utc) == FRIDAY_DATE
            ),
            key=lambda bar: bar.start_utc,
        )
        regular = [bar for bar in symbol_bars if is_regular_session(bar.start_utc)]
        if not regular:
            market[symbol] = {"bars": len(symbol_bars), "regular_bars": 0}
            continue
        volume = sum(bar.volume for bar in regular)
        vwap = (
            sum(bar.vwap * bar.volume for bar in regular) / volume
            if volume > 0
            else None
        )
        first, last = regular[0], regular[-1]
        high = max(bar.high for bar in regular)
        low = min(bar.low for bar in regular)
        market[symbol] = {
            "bars": len(symbol_bars),
            "regular_bars": len(regular),
            "post_close_bars": len(symbol_bars) - len(regular),
            "session_open_utc": first.start_utc.isoformat(),
            "session_close_utc": last.end_utc.isoformat(),
            "open": round(first.open, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(last.close, 4),
            "range": round(high - low, 4),
            "range_pct": round((high - low) / first.open * 100.0, 2),
            "volume": round(volume, 0),
            "vwap": round(vwap, 4) if vwap is not None else None,
            "last_price_ok": True,
        }
    return market


def _summary_from_metrics(metrics: Any) -> Dict[str, Any]:
    return {
        "total_trades": int(metrics.total_trades),
        "wins": int(metrics.wins),
        "losses": int(metrics.losses),
        "win_rate_pct": _pct(metrics.win_rate),
        "profit_factor": _number(metrics.profit_factor, 4),
        "net_pnl_dollars": _number(metrics.net_pnl_dollars),
        "avg_r": _number(metrics.avg_r, 4),
        "max_drawdown_dollars": _number(metrics.max_drawdown_dollars),
        "total_slippage_dollars": _number(metrics.total_slippage_dollars),
    }


def _trade_record_to_dict(trade: Any, *, strategy_label: str | None = None) -> Dict[str, Any]:
    label = strategy_label or ENUM_TO_LABEL.get(trade.strategy, trade.strategy.value)
    return {
        "trade_id": trade.signal_id,
        "symbol": trade.symbol,
        "strategy": label,
        "direction": "LONG",
        "shares": trade.shares,
        "entry_time_utc": trade.entry_time_utc.isoformat(),
        "entry_time_et": _dt_et(trade.entry_time_utc),
        "entry_price_signal": trade.entry_price_signal,
        "entry_price_fill": trade.entry_price_fill,
        "stop_price": trade.stop_price,
        "target_price": trade.target_price,
        "exit_time_utc": trade.exit_time_utc.isoformat()
        if trade.exit_time_utc
        else None,
        "exit_time_et": _dt_et(trade.exit_time_utc),
        "exit_price_fill": trade.exit_price_fill,
        "exit_reason": trade.exit_reason,
        "exit_reason_detail": trade.exit_reason,
        "pnl": round(trade.pnl, 4),
        "r_multiple": round(trade.r_multiple, 4),
        "holding_minutes": round(trade.holding_minutes, 2),
        "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
    }


def _zero_trade_reason(version: str, extra: Mapping[str, Any] | None) -> str:
    extra = dict(extra or {})
    if version == "V0.1":
        return (
            "生产 1m 引擎当日无入场：1m 级 ORB/VWAP/EMA 触发条件未满足或触发后被风控拒绝"
        )
    if version == "V0.3":
        return (
            "V0.3 5m 引擎当日无入场：无 5m 策略触发，或触发后被 14:00 ET 截止/风控拒绝"
        )
    blocks = extra.get("risk_block_counts") or {}
    signals = extra.get("signals")
    skipped = extra.get("window_skipped_evaluations")
    detail = []
    if signals is not None:
        detail.append(f"候选信号 {signals} 个")
    if skipped is not None:
        detail.append(f"入场窗/截止跳过评估 {skipped} 次")
    if blocks:
        detail.append(
            "风控拒单 "
            + ", ".join(f"{key}={value}" for key, value in sorted(blocks.items()))
        )
    if not detail:
        detail.append("无 5m 触发")
    return (
        "V0.4 当日无入场：10:00-14:00 入场窗 + 趋势过滤/SMC 未触发（"
        + "；".join(detail)
        + "）"
    )


def run_friday_recap(
    friday_bars: Sequence[Bar],
    *,
    initial_equity: float,
) -> Dict[str, Any]:
    config = load_config()
    print("\n[周五复盘] 准备 5m 数据（含历史，供 EMA/ATR 预热）…", flush=True)
    started = time.perf_counter()
    prepared = v02.prepare_v02_data(friday_bars)
    print(
        f"  prepared {len(prepared.sessions)} sessions，最后 {prepared.dates[-1]}"
        f"（{time.perf_counter() - started:.1f}s）",
        flush=True,
    )
    market = _market_snapshot(friday_bars)
    versions: Dict[str, Any] = {}

    # -- V0.1: production 1m engine, fresh risk manager -----------------------
    print("[周五复盘] V0.1 生产 1m 引擎 …", flush=True)
    friday_regular = sorted(
        (
            bar
            for bar in friday_bars
            if et_date(bar.start_utc) == FRIDAY_DATE and is_regular_session(bar.start_utc)
        ),
        key=lambda bar: bar.start_utc,
    )
    v01_result = run_backtest(
        friday_regular,
        config,
        build_strategies(config.strategies),
        risk=RiskManager(config),
    )
    versions["V0.1"] = {
        "version": "V0.1",
        "engine": "production 1m engine (backtest.engine.run_backtest + build_strategies)",
        "summary": _summary_from_metrics(v01_result.metrics),
        "trades": [_trade_record_to_dict(trade) for trade in v01_result.trades],
        "diagnostics": {
            "bars_processed": v01_result.bars_processed,
            "risk_block_counts": dict(v01_result.risk_block_counts),
            "ambiguous_bars": v01_result.ambiguous_bars,
        },
    }

    # -- V0.3: STEP4 on the Friday session -----------------------------------
    print("[周五复盘] V0.3 STEP4 5m 引擎 …", flush=True)
    v03_result = v03.run_v03_backtest(
        prepared,
        config,
        variant=v03.STEP4_VARIANT,
        session_dates=[FRIDAY_DATE],
    )
    versions["V0.3"] = {
        "version": "V0.3",
        "engine": "v0.3 STEP4 (fractional + 2 strategies + 14:00 cutoff + 1.2R shield)",
        "summary": _summary_from_metrics(v03_result.metrics),
        "trades": [_trade_record_to_dict(trade) for trade in v03_result.trades],
        "diagnostics": {
            "signals": v03_result.signals,
            "pending_cancelled": v03_result.pending_cancelled,
            "risk_block_counts": dict(v03_result.risk_block_counts),
            "cutoff_skipped_evaluations": v03_result.cutoff_skipped_evaluations,
            "breakeven_armed_trades": v03_result.breakeven_armed_trades,
        },
    }

    # -- V0.4: final variant --------------------------------------------------
    print("[周五复盘] V0.4 最终引擎 …", flush=True)
    v04_result = v04.run_v04_backtest(
        prepared,
        config,
        variant=v04.V04_FINAL_VARIANT,
        session_dates=[FRIDAY_DATE],
    )
    v04_trade_dicts = v04.result_to_trade_dicts(v04_result)
    versions["V0.4"] = {
        "version": "V0.4",
        "engine": "v0.4 final (10:00-14:00 window + VWAP trend + SMC + 1.2R scale-out)",
        "summary": _summary_from_metrics(v04_result.metrics),
        "trades": v04_trade_dicts,
        "diagnostics": {
            "signals": v04_result.signals,
            "pending_cancelled": v04_result.pending_cancelled,
            "risk_block_counts": dict(v04_result.risk_block_counts),
            "window_skipped_evaluations": v04_result.window_skipped_evaluations,
            "scale_target_trades": v04_result.scale_target_trades,
            "trail_exit_trades": v04_result.trail_exit_trades,
            "markers": len(v04_result.markers),
        },
    }

    for version, payload in versions.items():
        if payload["summary"]["total_trades"] == 0:
            payload["zero_trade_reason"] = _zero_trade_reason(
                version, payload.get("diagnostics")
            )

    ordered_markers = sorted(
        v04_result.markers, key=lambda item: (item.get("t") or 0, item["type"])
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
            **{
                key: marker[key]
                for key in ("stop", "target", "qty", "limit_price")
                if key in marker
            },
        }
        for marker in ordered_markers
    ]

    recap = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4",
        "session_date": FRIDAY_DATE.isoformat(),
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw, 2026-11-06 regular session "
            "(single venue; NOT consolidated tape)"
        ),
        "caveat": DATA_CAVEAT,
        "market": market,
        "versions": versions,
        "markers_count": len(markers),
        "markers_by_type": {
            marker_type: sum(1 for marker in markers if marker["type"] == marker_type)
            for marker_type in MARKER_TYPES
        },
        "notes": [
            "V0.4 markers are generated from the final variant only; Web replays them on the IEX chart.",
            "收盘后（16:00 ET）的额外 1m bar 不计入策略回放，仅在行情统计中注明 post_close_bars。",
            "四舍五入口径：PnL 保留 4 位小数，行情保留 4 位小数。",
        ],
    }
    markers_payload = {
        "generated_at_utc": recap["generated_at_utc"],
        "session_date": FRIDAY_DATE.isoformat(),
        "data_source": recap["data_source"],
        "variant": v04.V04_FINAL_VARIANT.flags(),
        "markers": markers,
        "trades": [
            {
                "trade_id": trade["trade_id"],
                "symbol": trade["symbol"],
                "strategy": trade["strategy"],
                "entry_time_utc": trade["entry_time_utc"],
                "exit_time_utc": trade["exit_time_utc"],
            }
            for trade in v04_trade_dicts
        ],
    }
    return {"recap": recap, "markers": markers_payload}


# --------------------------------------------------------------------------- #
# Markdown / terminal rendering
# --------------------------------------------------------------------------- #
_ABLATION_HEADER = (
    "| Step | 结构改动 | 入场窗 | 趋势过滤 | SMC | 分批止盈 | 全周期笔数 | 胜率 | PF | "
    "净盈亏($) | 平均R | 最大回撤($) | 2024笔数 | 2024 PF | 2024净($) | "
    "Holdout笔数 | Holdout PF | Holdout净($) | WF笔数 | WF胜率 | WF PF | WF净($) | WF盈利折 |"
)
_ABLATION_SEPARATOR = "|" + "---|" * 23


def _strategy_change(row: Mapping[str, Any]) -> str:
    return str(row.get("name", "")).split("·", 1)[-1].strip()


def _ablation_row(row: Mapping[str, Any]) -> str:
    flags = row["flags"]
    full = row["full"]["metrics"]
    train = (row.get("train_2024") or {}).get("metrics") or {}
    holdout = (row.get("holdout") or {}).get("metrics") or {}
    wf = row.get("walkforward") or {}
    wf_agg = wf.get("aggregate_test") or {}
    wf_summary = wf.get("summary") or {}
    window_start = flags.get("entry_window_start_et")
    window_end = flags.get("entry_window_end_et")
    window_text = (
        f"{window_start}-{window_end}"
        if window_start
        else (f"{window_end} cutoff" if window_end else "—")
    )
    return (
        f"| {row['step']} | {_strategy_change(row)} | "
        f"{window_text} | "
        f"{'✅' if flags.get('vwap_trend_filter') else '—'} | "
        f"{'✅' if flags.get('smc_sweep') else '—'} | "
        f"{'✅ 1.2R' if flags.get('scale_out_1_2r') else '—'} | "
        f"{full.get('total_trades')} | {_fmt(full.get('win_rate_pct'), 'pct')} | "
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


def _fmt(value: Any, kind: str = "") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{float(value):.2f}%"
    if kind == "pf":
        return f"{float(value):.3f}"
    if kind == "money":
        return f"{float(value):+,.2f}"
    if kind == "r":
        return f"{float(value):+.3f}"
    if kind == "num":
        return f"{float(value):,.4f}"
    if kind == "num2":
        return f"{float(value):,.2f}"
    if kind == "dd":
        return f"{float(value):,.2f}"
    return str(value)


def build_ablation_markdown(
    *,
    steps: Sequence[Mapping[str, Any]],
    self_check: Mapping[str, Any],
    data_span: str,
    total_1m_bars: int,
    total_sessions: int,
    initial_equity: float,
    elapsed: Mapping[str, float],
) -> str:
    lines = [
        "# V0.4 SOXL/SOXS 结构升级 · 逐步增量消融（raw 未复权 IEX 1m）",
        "",
        f"- 数据：Alpaca IEX raw 1m，{data_span}，{total_1m_bars:,} 根 1m bar / "
        f"{total_sessions} 个 session（单一交易所口径，非 consolidated tape）",
        f"- 结构：{V04_STRUCTURE_NOTE}",
        f"- 初始权益 ${initial_equity:,.2f}；风控/止损参数沿用 config/default.yaml，未调参",
        "",
        "## 主表：5 行累计消融",
        "",
        _ABLATION_HEADER,
        _ABLATION_SEPARATOR,
        *[_ablation_row(row) for row in steps],
        "",
        "## Step 0 自洽校验（V0.3 raw STEP4 锚：970 笔 / -$214.76 / WF -$234.92 / 16 折）",
        "",
        f"结果：{'通过 ✅' if self_check.get('passed') else '未通过 ❌'}",
        "",
        "| 区段 | 指标 | 记录值(V0.3 raw STEP4) | v03 重算 | v04 Step0 | 偏差 | 容差 | 通过 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for check in self_check.get("checks", []):
        lines.append(
            f"| {check['section']} | {check['field']} | {check['recorded_v03_raw_step4']} | "
            f"{check['v03_recomputed']} | {check['v04_step0_recomputed']} | "
            f"{check['delta_vs_recorded']} | ±{check['tolerance']} | "
            f"{'✅' if check['ok'] else '❌'} |"
        )
    final = steps[-1]
    final_extra = final["full"].get("extra") or {}
    lines.extend(
        [
            "",
            "## V0.4 最终（Step 4）结构统计",
            "",
            f"- 全周期：{final['full']['metrics']['total_trades']} 笔，"
            f"胜率 {_fmt(final['full']['metrics']['win_rate_pct'], 'pct')}，"
            f"PF {_fmt(final['full']['metrics']['profit_factor'], 'pf')}，"
            f"净盈亏 {_fmt(final['full']['metrics']['net_pnl_dollars'], 'money')}，"
            f"最大回撤 {_fmt(final['full']['metrics']['max_drawdown_dollars'], 'dd')}",
            f"- 分批止盈：触发 1.2R 分批 {final_extra.get('scale_target_trades', 0)} 笔；"
            f"TRAIL_EMA9 出场 {final_extra.get('trail_exit_trades', 0)} 笔；"
            f"入场窗/截止跳过评估 {final_extra.get('window_skipped_evaluations', 0)} 次",
            f"- 风控：单日熔断 {final_extra.get('breaker_days', 0)} 天；"
            f"同 bar TP/SL 碰撞按止损 {final_extra.get('ambiguous_bars', 0)} 次；"
            f"候选 {final_extra.get('signals', 0)} 笔 / 成交 "
            f"{final['full']['metrics']['total_trades']} 笔",
            f"- 拒单分布：{json.dumps(final_extra.get('risk_block_counts', {}), ensure_ascii=False)}",
            "",
            f"运行时长：5 步 ×（全周期+分段+29 折 WF）合计 {elapsed.get('total', 0.0):.1f}s。",
            "",
            "> 数据口径：IEX 单一交易所；本表是结构性消融研究，不代表实盘有效性或盈利预期。",
            "",
        ]
    )
    return "\n".join(lines)


def print_ablation_table(steps: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 138)
    print("V0.4 结构升级 · 逐步增量消融表（raw 未复权 IEX 1m，29 折 WF 样本外）")
    print("=" * 138)
    print(
        f"{'Step':<6}{'结构改动':<30}{'全周期笔数':>10}{'胜率':>8}{'PF':>8}{'净盈亏($)':>13}"
        f"{'平均R':>9}{'MaxDD($)':>12}{'2024笔':>8}{'2024PF':>8}{'2024净($)':>12}"
        f"{'Hold笔':>8}{'HoldPF':>8}{'Hold净($)':>12}{'WF笔':>7}{'WF胜率':>8}{'WF PF':>7}{'WF净($)':>12}{'盈利折':>9}"
    )
    print("-" * 138)
    for row in steps:
        flags = row["flags"]
        full = row["full"]["metrics"]
        train = (row.get("train_2024") or {}).get("metrics") or {}
        holdout = (row.get("holdout") or {}).get("metrics") or {}
        wf = row.get("walkforward") or {}
        wf_agg = wf.get("aggregate_test") or {}
        wf_summary = wf.get("summary") or {}
        change = _strategy_change(row)[:14]
        print(
            f"{row['step']:<6}{change:<30}"
            f"{full.get('total_trades', '—'):>10}"
            f"{_fmt(full.get('win_rate_pct'), 'pct'):>8}"
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
    print("-" * 138)
    print("  说明：每行是累计结构（Step k 包含 Step 0..k 的改动）；Step0 用 v0.3 STEP4 引擎作权威锚。")


def print_friday_recap(recap: Mapping[str, Any]) -> None:
    print("\n" + "=" * 110)
    print(f"周五 2026-11-06 单日复盘（raw IEX 1m，IEX 单一交易所口径）")
    print("=" * 110)
    print(f"  {'标的':<6}{'1m bars':>9}{'开放':>10}{'最高':>10}{'最低':>10}{'收':>10}{'振幅%':>9}{'成交量':>14}")
    print("-" * 110)
    for symbol, market in recap.get("market", {}).items():
        print(
            f"  {symbol:<6}{market.get('regular_bars', 0):>9}"
            f"{_fmt(market.get('open'), 'num'):>10}{_fmt(market.get('high'), 'num'):>10}"
            f"{_fmt(market.get('low'), 'num'):>10}{_fmt(market.get('close'), 'num'):>10}"
            f"{_fmt(market.get('range_pct'), 'num2'):>9}"
            f"{market.get('volume', 0):>14,.0f}"
        )
    print("\n  " + f"{'版本':<7}{'引擎':<54}{'笔数':>6}{'胜率':>9}{'PF':>8}{'净盈亏($)':>13}{'平均R':>9}")
    print("  " + "-" * 106)
    for version, payload in recap.get("versions", {}).items():
        summary = payload["summary"]
        print(
            f"  {version:<7}{str(payload.get('engine', ''))[:52]:<54}"
            f"{summary.get('total_trades', 0):>6}"
            f"{_fmt(summary.get('win_rate_pct'), 'pct'):>9}"
            f"{_fmt(summary.get('profit_factor'), 'pf'):>8}"
            f"{_fmt(summary.get('net_pnl_dollars'), 'money'):>13}"
            f"{_fmt(summary.get('avg_r'), 'r'):>9}"
        )
        if summary.get("total_trades", 0) == 0 and payload.get("zero_trade_reason"):
            print(f"          ↳ {payload['zero_trade_reason']}")
    print("-" * 110)
    v04_payload = recap.get("versions", {}).get("V0.4", {})
    if v04_payload.get("trades"):
        print("\n  V0.4 当日信号明细：")
        for trade in v04_payload["trades"]:
            print(
                f"    {trade['entry_time_et']} ET {trade['symbol']} {trade['strategy']:<16}"
                f" 入场 {_fmt(trade['entry_price_fill'], 'num')} 止损 {_fmt(trade['stop_price'], 'num')}"
                f" 目标 {_fmt(trade['target_price'], 'num')} → {trade['exit_time_et']} ET"
                f" {trade['exit_reason']:<12} PnL {_fmt(trade['pnl'], 'money')} R {_fmt(trade['r_multiple'], 'r')}"
            )
    print(
        f"\n  V0.4 markers: {recap.get('markers_count', 0)} 个 "
        f"({', '.join(f'{k}={v}' for k, v in recap.get('markers_by_type', {}).items() if v)})"
    )
    print("=" * 110)


def write_trades_csv(result: v04.V04Result, path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for trade in result.trades:
        detail = result.trade_details.get(trade.signal_id, {})
        rows.append(
            {
                "signal_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": trade.strategy.value,
                "strategy_label": detail.get("strategy_label", trade.strategy.value),
                "shares": trade.shares,
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "entry_time_et": _dt_et(trade.entry_time_utc),
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat()
                if trade.exit_time_utc
                else "",
                "exit_time_et": _dt_et(trade.exit_time_utc),
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason or "",
                "exit_reason_detail": detail.get("exit_reason_detail", ""),
                "exit_stop_kind": detail.get("exit_stop_kind") or "",
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "mae_r": round(float(detail.get("mae_r", 0.0) or 0.0), 4),
                "mfe_r": round(float(detail.get("mfe_r", 0.0) or 0.0), 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                "scale_target_hit": bool(detail.get("scale_target_hit", False)),
                "breakeven_armed": bool(detail.get("breakeven_armed", False)),
                "leg_count": len(detail.get("partial_legs") or []),
                "legs_json": json.dumps(
                    detail.get("partial_legs") or [], ensure_ascii=False
                ),
                "signal_bar_start_utc": detail.get("signal_bar_start_utc", ""),
                "entry_bucket_start_utc": detail.get("entry_bucket_start_utc", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def build_friday_markdown(recap: Mapping[str, Any]) -> str:
    lines = [
        f"# V0.4 周五单日复盘 · {recap.get('session_date')}（raw IEX 1m）",
        "",
        f"- 数据：{recap.get('data_source')}",
        f"- 口径警告：{DATA_CAVEAT}",
        "",
        "## 当日行情（常规时段 09:30-16:00 ET）",
        "",
        "| 标的 | 1m bars | 开盘 | 最高 | 最低 | 收盘 | 振幅% | 成交量 | VWAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, market in recap.get("market", {}).items():
        lines.append(
            f"| {symbol} | {market.get('regular_bars', 0)} | "
            f"{_fmt(market.get('open'), 'num')} | {_fmt(market.get('high'), 'num')} | "
            f"{_fmt(market.get('low'), 'num')} | {_fmt(market.get('close'), 'num')} | "
            f"{_fmt(market.get('range_pct'), 'num2')} | {market.get('volume', 0):,.0f} | "
            f"{_fmt(market.get('vwap'), 'num')} |"
        )
    post_close = [
        f"- {symbol} 另有 {market.get('post_close_bars', 0)} 根 16:00 ET 收盘后 bar 仅计入行情文件、未参与策略回放"
        for symbol, market in recap.get("market", {}).items()
        if market.get("post_close_bars")
    ]
    if post_close:
        lines.extend(["", *post_close])
    lines.extend(
        [
            "",
            "## 三版本汇总",
            "",
            "| 版本 | 引擎 | 笔数 | 胜率 | PF | 净盈亏($) | 平均R | 说明 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for version, payload in recap.get("versions", {}).items():
        summary = payload["summary"]
        note = payload.get("zero_trade_reason") or "—"
        lines.append(
            f"| {version} | {payload.get('engine', '')} | {summary.get('total_trades', 0)} | "
            f"{_fmt(summary.get('win_rate_pct'), 'pct')} | "
            f"{_fmt(summary.get('profit_factor'), 'pf')} | "
            f"{_fmt(summary.get('net_pnl_dollars'), 'money')} | "
            f"{_fmt(summary.get('avg_r'), 'r')} | {note} |"
        )
    lines.extend(
        [
            "",
            "## 逐笔信号明细",
            "",
        ]
    )
    for version, payload in recap.get("versions", {}).items():
        trades = payload.get("trades", [])
        lines.append(f"### {version}（{len(trades)} 笔）")
        lines.append("")
        if not trades:
            lines.append(f"当日无信号。原因：{payload.get('zero_trade_reason', '—')}")
            lines.append("")
            continue
        lines.append(
            "| # | 入场时间(ET) | 标的 | 策略 | 股数 | 入场价 | 止损 | 目标 | 出场时间(ET) | 出场价 | 出场原因 | PnL($) | R | 分批/备注 |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|")
        for index, trade in enumerate(trades, start=1):
            extra_bits = []
            if trade.get("scale_target_hit"):
                extra_bits.append("1.2R 分批")
            exit_detail = trade.get("exit_reason_detail") or ""
            if exit_detail and exit_detail != trade.get("exit_reason"):
                extra_bits.append(exit_detail)
            lines.append(
                "| {index} | {entry} | {symbol} | {strategy} | {shares} | {entry_price} | "
                "{stop} | {target} | {exit_time} | {exit_price} | {reason} | {pnl} | {r} | {extra} |".format(
                    index=index,
                    entry=trade.get("entry_time_et") or trade.get("entry_time_utc"),
                    symbol=trade.get("symbol"),
                    strategy=trade.get("strategy"),
                    shares=f"{float(trade.get('shares', 0)):g}",
                    entry_price=_fmt(trade.get("entry_price_fill"), "num"),
                    stop=_fmt(trade.get("stop_price"), "num"),
                    target=_fmt(trade.get("target_price"), "num"),
                    exit_time=trade.get("exit_time_et") or trade.get("exit_time_utc"),
                    exit_price=_fmt(trade.get("exit_price_fill"), "num"),
                    reason=trade.get("exit_reason"),
                    pnl=_fmt(trade.get("pnl"), "money"),
                    r=_fmt(trade.get("r_multiple"), "r"),
                    extra="；".join(extra_bits) or "—",
                )
            )
        lines.append("")
    lines.extend(
        [
            "## 图表标记（V0.4，供 Web 回放）",
            "",
            f"- 标记数：{recap.get('markers_count', 0)}（"
            + "，".join(
                f"{key}={value}"
                for key, value in recap.get("markers_by_type", {}).items()
                if value
            )
            + "）",
            "- 类型：ENTRY / STOP / BREAKEVEN / TARGET / EXIT；字段：type, time_utc, price, symbol, strategy, trade_id。",
            "- JSON：`var/backtest/v04_friday_markers.json`。",
            "",
            "## 说明",
            "",
            *[f"- {note}" for note in recap.get("notes", [])],
            "",
        ]
    )
    return "\n".join(lines)


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

    # -- backtest dataset (raw, excludes 2026-11-06) -------------------------
    print(f"加载 raw 回测数据：{data_dir}")
    raw_bars = load_raw_bars(data_dir)
    prepare_started = time.perf_counter()
    data = prepare_v02_data(raw_bars)
    print(
        f"5m 数据就绪：{data.total_5m_bars:,} 根 5m bar，{len(data.sessions)} 个 session"
        f"（{time.perf_counter() - prepare_started:.1f}s）"
    )
    data_span = _fmt_span(list(data.dates))
    ablation = _run_ablation(
        data,
        config,
        with_segments=with_segments,
        with_walkforward=with_walkforward,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_equity=equity,
    )
    steps = ablation["steps"]
    self_check = ablation["self_check"]
    print(
        f"\nStep 0 自洽校验（V0.3 raw STEP4 锚）：{'通过 ✅' if self_check.get('passed') else '未通过 ❌'}"
    )
    for check in self_check.get("checks", []):
        status = "✅" if check["ok"] else "❌"
        print(
            f"  {status} {check['section']:<12} {check['field']:<28} "
            f"记录={check['recorded_v03_raw_step4']} v03重算={check['v03_recomputed']} "
            f"v04Step0={check['v04_step0_recomputed']} Δ={check['delta_vs_recorded']}"
        )

    # -- final full-period trade ledger (re-run final step once) -------------
    print("\n生成 V0.4 最终全周期逐笔流水 …", flush=True)
    final_result = v04.run_v04_backtest(
        data, config, variant=v04.V04_FINAL_VARIANT
    )

    # -- Friday recap --------------------------------------------------------
    print("\n加载周五数据（含 2026-11-06，raw）…")
    friday_bars = load_friday_raw_bars(data_dir)
    friday_started = time.perf_counter()
    friday = run_friday_recap(friday_bars, initial_equity=equity)
    friday_elapsed = time.perf_counter() - friday_started

    total_elapsed = time.perf_counter() - started
    ablation_markdown = build_ablation_markdown(
        steps=steps,
        self_check=self_check,
        data_span=data_span,
        total_1m_bars=len(raw_bars),
        total_sessions=len(data.sessions),
        initial_equity=equity,
        elapsed={"total": total_elapsed},
    )
    friday_markdown = build_friday_markdown(friday["recap"])

    final_step = steps[-1]
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.4",
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw (single venue; NOT consolidated tape)"
        ),
        "backtest_data_span": data_span,
        "total_1m_bars": len(raw_bars),
        "total_sessions": len(data.sessions),
        "prepared": {
            "total_5m_bars": data.total_5m_bars,
            "skipped_5m_buckets": data.skipped_buckets,
            "symbols": list(data.symbols),
        },
        "structure": V04_STRUCTURE_NOTE,
        "constants": {
            "entry_window_start_et": str(v04.ENTRY_WINDOW_START_ET),
            "entry_window_end_et": str(v04.ENTRY_WINDOW_END_ET),
            "or30_bars": v04.OR30_BARS,
            "smc_lookback_bars": v04.SMC_LOOKBACK_BARS,
            "smc_range_multiple": v04.SMC_RANGE_MULTIPLE,
            "smc_stop_atr_buffer": v04.SMC_STOP_ATR_BUFFER,
            "vwap_trend_range_atr_multiple": v04.VWAP_TREND_RANGE_ATR_MULTIPLE,
            "scale_trigger_r": v04.SCALE_TRIGGER_R,
            "scale_portion": v04.SCALE_PORTION,
            "breakeven_offset_r": v04.BREAKEVEN_OFFSET_R,
        },
        "caveat": DATA_CAVEAT,
        "initial_equity": equity,
        "walkforward_grid": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
        },
        "steps": steps,
        "step0_self_check": self_check,
        "v03_raw_step4_anchor": V03_RAW_STEP4_ANCHOR,
        "comparison": {
            "step0_full": steps[0]["full"]["metrics"],
            "step4_full": final_step["full"]["metrics"],
            "step0_walkforward": (steps[0].get("walkforward") or {}).get(
                "aggregate_test"
            ),
            "step4_walkforward": (final_step.get("walkforward") or {}).get(
                "aggregate_test"
            ),
        },
        "friday_recap_summary": {
            version: payload_data["summary"]
            for version, payload_data in friday["recap"]["versions"].items()
        },
        "elapsed_seconds": {
            "ablation_total": round(total_elapsed - friday_elapsed, 2),
            "friday_recap": round(friday_elapsed, 2),
            "total": round(total_elapsed, 2),
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v04_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    ablation_path = report_dir / "v04_ablation.md"
    ablation_path.write_text(ablation_markdown, encoding="utf-8")
    trades_path = report_dir / "v04_trades.csv"
    write_trades_csv(final_result, trades_path)
    recap_path = report_dir / "v04_friday_recap.json"
    recap_path.write_text(
        json.dumps(friday["recap"], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    recap_md_path = report_dir / "v04_friday_recap.md"
    recap_md_path.write_text(friday_markdown, encoding="utf-8")
    markers_path = report_dir / "v04_friday_markers.json"
    markers_path.write_text(
        json.dumps(friday["markers"], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_ablation_table(steps)
    print_friday_recap(friday["recap"])
    print("\n产物：")
    print(f"  消融报告 : {report_path}")
    print(f"  消融 MD  : {ablation_path}")
    print(f"  逐笔流水 : {trades_path}（Step4 全周期 {len(final_result.trades)} 笔）")
    print(f"  周五复盘 : {recap_path}")
    print(f"  周五 MD  : {recap_md_path}")
    print(f"  图表标记 : {markers_path}（{len(friday['markers']['markers'])} 个）")
    print(
        f"\n总运行时长：{total_elapsed:.1f}s"
        f"（消融 {total_elapsed - friday_elapsed:.1f}s + 周五复盘 {friday_elapsed:.1f}s）"
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.4 structural ablation + Friday recap on raw IEX 1m bars."
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
