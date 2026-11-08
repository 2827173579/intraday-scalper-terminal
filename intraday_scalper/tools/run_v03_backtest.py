"""V0.3 structural ablation CLI: Step 0 (V0.2 baseline) -> Step 4 (V0.3 final).

Runs, on the local Alpaca **IEX-only** 1m parquet cache, the five cumulative
ablation steps described in :mod:`intraday_scalper.research.v03`:

* Step 0  V0.2 Baseline (sanity anchor, must reproduce the recorded V0.2 numbers)
* Step 1  + 碎股 (fractional shares)
* Step 2  + 移除 ORB_15M_5M (VWAP_RECLAIM_5M / EMA_PULLBACK_5M only)
* Step 3  + 14:00 ET entry cutoff
* Step 4  + 1.2R breakeven shield  == V0.3 final

For every step it runs the full period, the 2024 in-sample segment, the
2025-01-01..2026-11-05 holdout, and the **same 29-fold walk-forward grid**
(120/20/20, fresh risk manager and engine per fold, no compounding).

Artifacts (``intraday_scalper/var/backtest/``)::

    v03_report.json    # every step: full/segments/walk-forward + Step0 self-check
    v03_ablation.md    # markdown ablation table for REVIEW_PACK
    v03_trades.csv     # final Step 4 full-period trade ledger
    v03_oos.json       # final Step 4 per-fold out-of-sample metrics

Data caveat: IEX-only bars (single venue, small share of consolidated volume);
this is a structural ablation study, not proof of profitability.

Usage::

    python -m intraday_scalper.tools.run_v03_backtest
    python -m intraday_scalper.tools.run_v03_backtest --no-walkforward   # quick
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
from intraday_scalper.research import v03
from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    EMA_PULLBACK_5M,
    ORB_15M_5M,
    STRATEGY_LABELS,
    VWAP_RECLAIM_5M,
    prepare_v02_data,
)
from intraday_scalper.tools.run_real_backtest import load_bars

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

TRAIN_2024_START = date(2024, 1, 1)
TRAIN_2024_END = date(2024, 12, 31)
HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2026, 11, 5)

#: Recorded V0.2 baseline (from task-11 spec / v02_report.json) used as the
#: self-consistency anchor for Step 0.
V02_EXPECTED: dict[str, dict[str, float | int]] = {
    "full": {
        "total_trades": 1206,
        "win_rate_pct": 38.89,
        "profit_factor": 0.967,
        "net_pnl_dollars": -385.40,
        "avg_r": -0.011,
        "max_drawdown_dollars": 1417.16,
    },
    "train_2024": {
        "total_trades": 287,
        "profit_factor": 1.159,
        "net_pnl_dollars": 408.89,
    },
    "holdout": {
        "total_trades": 919,
        "profit_factor": 0.914,
        "net_pnl_dollars": -794.30,
    },
    "walkforward": {
        "total_trades": 1035,
        "win_rate_pct": 37.58,
        "profit_factor": 0.918,
        "net_pnl_dollars": -852.52,
        "positive_folds": 11,
    },
}

V03_STRUCTURE_NOTE = (
    "V0.3 在冻结的 V0.2 实现之上做四项一次性先验结构改动（a-priori, not tuned）："
    "① 碎股（allow_fractional_shares=True, precision=2）修复 SOXS 高价 0 股问题；"
    "② 只保留 VWAP_RECLAIM_5M 与 EMA_PULLBACK_5M（移除 ORB_15M_5M）；"
    "③ 14:00 ET 后不再新开仓（已完成 5m bar 的 end_utc 或 pending 成交时刻 >= 14:00 即取消）；"
    "④ +1.2R 保本盾（最高浮盈达 +1.2R 后，自下一根 1m bar 起止损上移至 entry_fill+0.05R）。"
    "29 折 WF 纪律与 V0.2 完全一致（120/20/20、每折全新 RiskManager、不复利），无任何逐窗调参。"
)


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


def _strategy_label(key: str) -> str:
    return {
        "ORB_15M": ORB_15M_5M,
        "VWAP_RECLAIM": VWAP_RECLAIM_5M,
        "EMA_PULLBACK": EMA_PULLBACK_5M,
    }.get(key, key)


def _exit_reason_counts(result: v03.V03Result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in result.trades:
        reason = trade.exit_reason or "UNKNOWN"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def format_result(result: v03.V03Result) -> dict[str, Any]:
    """JSON-safe summary for one V0.3 run (same shape across steps/segments)."""
    metrics = result.metrics
    return {
        "variant": result.variant.flags(),
        "metrics": {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate_pct": _pct(metrics.win_rate),
            "profit_factor": _number(metrics.profit_factor, 3),
            "net_pnl_dollars": _number(metrics.net_pnl_dollars),
            "expectancy_dollars": _number(metrics.expectancy_dollars),
            "avg_r": _number(metrics.avg_r, 3),
            "avg_holding_minutes": _number(metrics.avg_holding_minutes, 1),
            "max_drawdown_dollars": _number(metrics.max_drawdown_dollars),
            "max_drawdown_pct": _pct(metrics.max_drawdown_pct),
            "trades_per_day": _number(metrics.trades_per_day, 2),
            "trading_days": metrics.trading_days,
            "total_slippage_dollars": _number(metrics.total_slippage_dollars),
            "initial_equity": _number(metrics.initial_equity),
            "final_equity": _number(metrics.final_equity),
        },
        "extra": result.extra_metrics(),
        "exit_reasons": _exit_reason_counts(result),
        "per_strategy": {
            _strategy_label(key): {
                "trades": group.trades,
                "win_rate_pct": _pct(group.win_rate),
                "profit_factor": _number(group.profit_factor, 3),
                "net_pnl_dollars": _number(group.net_pnl_dollars),
                "avg_r": _number(group.avg_r, 3),
            }
            for key, group in metrics.per_strategy.items()
        },
        "per_symbol": {
            symbol: {
                "trades": group.trades,
                "win_rate_pct": _pct(group.win_rate),
                "profit_factor": _number(group.profit_factor, 3),
                "net_pnl_dollars": _number(group.net_pnl_dollars),
                "avg_r": _number(group.avg_r, 3),
            }
            for symbol, group in metrics.per_symbol.items()
        },
        "session_dates": len(result.session_dates),
        "notes": list(result.notes)[:10],
    }


def format_walkforward(wf: v03.V03WalkForwardResult) -> dict[str, Any]:
    fold_rows = []
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
                "test_profit_factor": _number(test_metrics.profit_factor, 3),
                "test_net_pnl_dollars": _number(test_metrics.net_pnl_dollars),
                "test_avg_r": _number(test_metrics.avg_r, 3),
                "test_max_drawdown_dollars": _number(test_metrics.max_drawdown_dollars),
                "test_breaker_days": fold_result.test_result.breaker_days,
                "leakage_checked": fold_result.leakage_checked,
            }
        )
    summary = dict(wf.summary)
    for key, digits in (
        ("test_win_rate", 4),
        ("test_profit_factor", 3),
        ("test_net_pnl_dollars", 2),
        ("test_avg_r", 3),
        ("test_max_drawdown_dollars_max_fold", 2),
        ("test_max_drawdown_pct_max_fold", 4),
        ("test_trades_per_session", 2),
    ):
        if key in summary:
            summary[key] = _number(summary[key], digits)
    aggregate = wf.aggregate_test_metrics
    return {
        "variant": wf.variant.flags(),
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
        },
        "max_fold_drawdown_dollars": _number(
            summary.get("test_max_drawdown_dollars_max_fold")
        ),
        "positive_folds": int(summary.get("positive_folds") or 0),
        "folds": int(summary.get("folds") or 0),
        "elapsed_seconds": None,
        "notes": list(wf.notes),
    }


# --------------------------------------------------------------------------- #
# Step0 self-consistency check
# --------------------------------------------------------------------------- #
def self_check_step0(step0: dict[str, Any]) -> dict[str, Any]:
    """Compare Step 0 against the recorded V0.2 baseline (tolerance stated)."""

    tolerances: dict[str, float] = {
        "total_trades": 0.0,
        "win_rate_pct": 0.01,
        "profit_factor": 0.001,
        "net_pnl_dollars": 0.01,
        "avg_r": 0.001,
        "max_drawdown_dollars": 0.01,
        "positive_folds": 0.0,
    }
    sections: dict[str, dict[str, Any]] = {
        "full": step0["full"]["metrics"],
        "train_2024": step0.get("train_2024", {}).get("metrics", {}),
        "holdout": step0.get("holdout", {}).get("metrics", {}),
    }
    if step0.get("walkforward"):
        sections["walkforward"] = {
            **step0["walkforward"]["aggregate_test"],
            "positive_folds": step0["walkforward"]["positive_folds"],
        }
    checks: list[dict[str, Any]] = []
    passed = True
    for section, expected_fields in V02_EXPECTED.items():
        actual_metrics = sections.get(section)
        if actual_metrics is None:
            checks.append(
                {
                    "section": section,
                    "field": "*",
                    "expected": "present",
                    "actual": None,
                    "delta": None,
                    "tolerance": 0.0,
                    "ok": False,
                    "note": f"section '{section}' was not computed in this run",
                }
            )
            passed = False
            continue
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
                    "expected": expected,
                    "actual": actual,
                    "delta": _number(float(actual) - float(expected), 4)
                    if actual is not None
                    else None,
                    "tolerance": tolerance,
                    "ok": bool(ok),
                }
            )
    return {"passed": passed, "checks": checks}


# --------------------------------------------------------------------------- #
# Run orchestration
# --------------------------------------------------------------------------- #
def _fmt_span(dates: Sequence[date]) -> str:
    return f"{min(dates):%Y-%m-%d} .. {max(dates):%Y-%m-%d}" if dates else "—"


def _run_one(
    data,
    config,
    variant: v03.V03Variant,
    *,
    initial_equity: float,
    with_segments: bool,
    with_walkforward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    full_result = v03.run_v03_backtest(
        data, config, variant=variant, initial_equity=initial_equity
    )
    payload: dict[str, Any] = {
        "step": variant.step,
        "name": variant.name,
        "flags": variant.flags(),
        "full": format_result(full_result),
    }
    print(
        f"    full: {full_result.metrics.total_trades} 笔，"
        f"净 {full_result.metrics.net_pnl_dollars:+.2f}，"
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )

    if with_segments:
        for name, (start, end) in (
            ("train_2024", (TRAIN_2024_START, TRAIN_2024_END)),
            ("holdout", (HOLDOUT_START, HOLDOUT_END)),
        ):
            dates = [value for value in data.dates if start <= value <= end]
            segment_result = v03.run_v03_backtest(
                data,
                config,
                variant=variant,
                initial_equity=initial_equity,
                session_dates=dates,
            )
            segment_payload = format_result(segment_result)
            segment_payload["session_span"] = _fmt_span(dates)
            payload[name] = segment_payload
            print(
                f"    {name}: {segment_result.metrics.total_trades} 笔，"
                f"净 {segment_result.metrics.net_pnl_dollars:+.2f}",
                flush=True,
            )

    if with_walkforward:
        wf = v03.run_v03_walkforward(
            data,
            config,
            variant=variant,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            initial_equity=initial_equity,
            progress=lambda index, total: print(
                f"    walk-forward 折 {index}/{total}", flush=True
            ),
        )
        wf_payload = format_walkforward(wf)
        wf_payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        payload["walkforward"] = wf_payload
        print(
            f"    WF OOS: {wf.summary['total_test_trades']} 笔，"
            f"净 {wf.summary['test_net_pnl_dollars']:+.2f}，"
            f"{wf.summary['positive_folds']}/{wf.summary['folds']} 盈利折",
            flush=True,
        )

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return payload


def _run_all_steps(
    data,
    config,
    *,
    initial_equity: float,
    with_segments: bool,
    with_walkforward: bool,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for variant in v03.ABLATION_VARIANTS:
        print(f"\n开始 {variant.name} …", flush=True)
        steps.append(
            _run_one(
                data,
                config,
                variant,
                initial_equity=initial_equity,
                with_segments=with_segments,
                with_walkforward=with_walkforward,
                train_days=train_days,
                test_days=test_days,
                step_days=step_days,
            )
        )
    return steps


# --------------------------------------------------------------------------- #
# Markdown / terminal rendering
# --------------------------------------------------------------------------- #
_ABLATION_HEADER = (
    "| Step | 结构改动 | 碎股 | 14:00截止 | 保本盾 | 全周期笔数 | 胜率 | PF | "
    "净盈亏($) | 平均R | 最大回撤($) | 2024笔数 | 2024 PF | 2024净($) | "
    "Holdout笔数 | Holdout PF | Holdout净($) | WF笔数 | WF胜率 | WF PF | WF净($) | WF盈利折 |"
)
_ABLATION_SEPARATOR = (
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def _row_cell(value: Any, kind: str = "") -> str:
    if value is None:
        return "—"
    if kind == "int":
        return str(int(value))
    if kind == "pct":
        return f"{float(value):.2f}%"
    if kind == "pf":
        return f"{float(value):.3f}"
    if kind == "money":
        return f"{float(value):+,.2f}"
    if kind == "r":
        return f"{float(value):+.3f}"
    return str(value)


def ablation_markdown_rows(steps: Sequence[dict[str, Any]]) -> list[str]:
    rows = []
    for step in steps:
        flags = step["flags"]
        full = step["full"]["metrics"]
        train = step.get("train_2024", {}).get("metrics", {})
        holdout = step.get("holdout", {}).get("metrics", {})
        wf = step.get("walkforward", {})
        wf_agg = wf.get("aggregate_test", {})
        change = step["name"].split("·", 1)[-1].strip()
        rows.append(
            "| {step} | {change} | {frac} | {cutoff} | {shield} | {full_trades} | "
            "{win} | {pf} | {net} | {avg_r} | {dd} | {tr_trades} | {tr_pf} | {tr_net} | "
            "{ho_trades} | {ho_pf} | {ho_net} | {wf_trades} | {wf_win} | {wf_pf} | "
            "{wf_net} | {wf_pos} |".format(
                step=step["step"],
                change=change,
                frac="✅" if flags["allow_fractional_shares"] else "—",
                cutoff=flags["entry_cutoff_et"] or "—",
                shield="✅ 1.2R" if flags["breakeven_trigger_r"] else "—",
                full_trades=_row_cell(full.get("total_trades"), "int"),
                win=_row_cell(full.get("win_rate_pct"), "pct"),
                pf=_row_cell(full.get("profit_factor"), "pf"),
                net=_row_cell(full.get("net_pnl_dollars"), "money"),
                avg_r=_row_cell(full.get("avg_r"), "r"),
                dd=_row_cell(full.get("max_drawdown_dollars"), "money"),
                tr_trades=_row_cell(train.get("total_trades"), "int"),
                tr_pf=_row_cell(train.get("profit_factor"), "pf"),
                tr_net=_row_cell(train.get("net_pnl_dollars"), "money"),
                ho_trades=_row_cell(holdout.get("total_trades"), "int"),
                ho_pf=_row_cell(holdout.get("profit_factor"), "pf"),
                ho_net=_row_cell(holdout.get("net_pnl_dollars"), "money"),
                wf_trades=_row_cell(wf_agg.get("total_trades"), "int"),
                wf_win=_row_cell(wf_agg.get("win_rate_pct"), "pct"),
                wf_pf=_row_cell(wf_agg.get("profit_factor"), "pf"),
                wf_net=_row_cell(wf_agg.get("net_pnl_dollars"), "money"),
                wf_pos=f"{wf.get('positive_folds', 0)}/{wf.get('folds', 0)}",
            )
        )
    return rows


def _delta(step: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if previous is None:
        return {}

    def _float(value: Any, default: float = 0.0) -> float:
        number = _number(value, 6)
        return default if number is None else float(number)

    full = step["full"]["metrics"]
    prev_full = previous["full"]["metrics"]
    wf = step.get("walkforward") or {}
    prev_wf = previous.get("walkforward") or {}
    wf_agg = wf.get("aggregate_test", {})
    prev_wf_agg = prev_wf.get("aggregate_test", {})
    return {
        "trades": int(full["total_trades"]) - int(prev_full["total_trades"]),
        "net": _number(_float(full["net_pnl_dollars"]) - _float(prev_full["net_pnl_dollars"])),
        "pf": _number(_float(full["profit_factor"]) - _float(prev_full["profit_factor"]), 3),
        "win": _number(_float(full["win_rate_pct"]) - _float(prev_full["win_rate_pct"])),
        "dd": _number(
            _float(full["max_drawdown_dollars"]) - _float(prev_full["max_drawdown_dollars"])
        ),
        "wf_trades": int(wf_agg.get("total_trades", 0) or 0)
        - int(prev_wf_agg.get("total_trades", 0) or 0),
        "wf_net": _number(
            _float(wf_agg.get("net_pnl_dollars")) - _float(prev_wf_agg.get("net_pnl_dollars"))
        ),
        "wf_pf": _number(
            _float(wf_agg.get("profit_factor")) - _float(prev_wf_agg.get("profit_factor")), 3
        ),
        "wf_positive": int(wf.get("positive_folds", 0) or 0)
        - int(prev_wf.get("positive_folds", 0) or 0),
    }


def build_ablation_markdown(
    steps: Sequence[dict[str, Any]],
    *,
    self_check: dict[str, Any],
    data_span: str,
    total_1m_bars: int,
    total_sessions: int,
    initial_equity: float,
    elapsed: dict[str, float],
) -> str:
    lines = [
        "# V0.3 结构重构 · 逐步增量消融实验（Step 0 → Step 4）",
        "",
        f"- 数据：Alpaca IEX 1m bars，{data_span}，{total_1m_bars:,} 根 1m bar / "
        f"{total_sessions} 个 session（单一交易所口径，非 consolidated tape）",
        f"- 结构：{V03_STRUCTURE_NOTE}",
        f"- 初始权益：${initial_equity:,.2f}；止损/止盈/风控参数沿用 config/default.yaml，未调参",
        f"- 数据口径警告：{DATA_CAVEAT}",
        "",
        "## 主表：5 行累计消融（每行均含全周期 / 2024 / holdout / 29 折 WF OOS）",
        "",
        _ABLATION_HEADER,
        _ABLATION_SEPARATOR,
        *ablation_markdown_rows(steps),
        "",
        "## 增量变化（相对上一 Step）",
        "",
        "| Step | Δ全周期笔数 | Δ净盈亏($) | ΔPF | Δ胜率(pp) | Δ最大回撤($) | "
        "ΔWF笔数 | ΔWF净($) | ΔWF PF | ΔWF盈利折 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    previous: dict[str, Any] | None = None
    for step in steps:
        delta = _delta(step, previous)
        if delta:
            lines.append(
                f"| {step['step']} | {delta['trades']:+d} | {delta['net']:+,.2f} | "
                f"{delta['pf']:+.3f} | {delta['win']:+.2f} | {delta['dd']:+,.2f} | "
                f"{delta['wf_trades']:+d} | {delta['wf_net']:+,.2f} | "
                f"{delta['wf_pf']:+.3f} | {delta['wf_positive']:+d} |"
            )
        previous = step
    lines.extend(
        [
            "",
            "## Step 0 自洽校验（V0.2 基线复现）",
            "",
            f"结果：{'通过 ✅' if self_check['passed'] else '未通过 ❌'} "
            "（容差：笔数/盈利折必须精确，金额 ±$0.01，胜率 ±0.01pp，PF ±0.001，平均R ±0.001）",
            "",
            "| 区段 | 指标 | V0.2 基线 | V0.3 Step0 实测 | 偏差 | 容差 | 通过 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for check in self_check["checks"]:
        lines.append(
            f"| {check['section']} | {check['field']} | {check['expected']} | "
            f"{check['actual']} | {check['delta']} | ±{check['tolerance']} | "
            f"{'✅' if check['ok'] else '❌'} |"
        )
    step4 = steps[-1]
    extra = step4["full"]["extra"]
    lines.extend(
        [
            "",
            "## V0.3 最终（Step 4）结构统计",
            "",
            f"- 全周期：{step4['full']['metrics']['total_trades']} 笔，"
            f"胜率 {step4['full']['metrics']['win_rate_pct']:.2f}%，"
            f"PF {step4['full']['metrics']['profit_factor']:.3f}，"
            f"净盈亏 {step4['full']['metrics']['net_pnl_dollars']:+.2f}，"
            f"平均R {step4['full']['metrics']['avg_r']:+.3f}，"
            f"最大回撤 ${step4['full']['metrics']['max_drawdown_dollars']:,.2f}",
            f"- 保本盾：arm 笔数 {extra['breakeven_armed_trades']}，"
            f"保本盾止损出场 {extra['shield_stop_exits']} 笔，"
            f"arm 且净盈利 {extra['shield_saved_trades']} 笔",
            f"- 风控：单日熔断 {extra['breaker_days']} 天，"
            f"同 bar TP/SL 碰撞按止损 {extra['ambiguous_bars']} 次，"
            f"14:00 截止跳过评估 {extra['cutoff_skipped_evaluations']} 次，"
            f"候选 {extra['signals']} 笔 / 成交 {step4['full']['metrics']['total_trades']} 笔",
            f"- 拒单分布：{json.dumps(extra['risk_block_counts'], ensure_ascii=False)}",
            "",
            f"运行时长：每步（全周期+分段+29 折 WF）平均 {elapsed.get('per_step', 0.0):.1f}s，"
            f"5 步合计 {elapsed.get('total', 0.0):.1f}s。",
            "",
            "> 数据口径：IEX 单一交易所；本表是结构性消融研究，不代表实盘有效性或盈利预期。",
            "",
        ]
    )
    return "\n".join(lines)


def print_ablation_table(steps: Sequence[dict[str, Any]]) -> None:
    def cell(value: Any, kind: str = "") -> str:
        return _row_cell(value, kind)

    print("\n" + "=" * 132)
    print("V0.3 结构重构 · 逐步增量消融表（全周期 + 分段 + 29 折 WF 样本外）")
    print("=" * 132)
    header = (
        f"{'Step':<7}{'结构改动':<24}{'碎股':<5}{'14:00截止':<9}{'保本盾':<8}"
        f"{'全周期笔数':>10}{'胜率':>8}{'PF':>7}{'净盈亏($)':>12}{'平均R':>8}"
        f"{'MaxDD($)':>11}{'2024笔':>7}{'2024PF':>7}{'2024净($)':>11}"
        f"{'Hold笔':>7}{'HoldPF':>7}{'Hold净($)':>11}"
    )
    print(header)
    print("-" * 132)
    for step in steps:
        flags = step["flags"]
        full = step["full"]["metrics"]
        train = step.get("train_2024", {}).get("metrics", {})
        holdout = step.get("holdout", {}).get("metrics", {})
        change = step["name"].split("·", 1)[-1].strip().replace("+ ", "")[:12]
        print(
            f"{step['step']:<7}{change:<24}"
            f"{('✅' if flags['allow_fractional_shares'] else '—'):<5}"
            f"{(flags['entry_cutoff_et'] or '—'):<9}"
            f"{('✅1.2R' if flags['breakeven_trigger_r'] else '—'):<8}"
            f"{cell(full.get('total_trades'), 'int'):>10}"
            f"{cell(full.get('win_rate_pct'), 'pct'):>8}"
            f"{cell(full.get('profit_factor'), 'pf'):>7}"
            f"{cell(full.get('net_pnl_dollars'), 'money'):>12}"
            f"{cell(full.get('avg_r'), 'r'):>8}"
            f"{cell(full.get('max_drawdown_dollars'), 'money'):>11}"
            f"{cell(train.get('total_trades'), 'int'):>7}"
            f"{cell(train.get('profit_factor'), 'pf'):>7}"
            f"{cell(train.get('net_pnl_dollars'), 'money'):>11}"
            f"{cell(holdout.get('total_trades'), 'int'):>7}"
            f"{cell(holdout.get('profit_factor'), 'pf'):>7}"
            f"{cell(holdout.get('net_pnl_dollars'), 'money'):>11}"
        )
    print("-" * 132)
    print(
        "  说明：每行是累计结构（Step k 包含 Step 0..k 的全部改动）；"
        "2024 段仅作 in-sample 结构验证，未用于调参。"
    )

    print("\n" + "=" * 132)
    print("V0.3 29 折 Walk-Forward 样本外（120/20/20，每折全新 RiskManager，不复利）")
    print("=" * 132)
    print(
        f"{'Step':<7}{'结构改动':<24}"
        f"{'WF笔数':>9}{'WF胜率':>9}{'WF PF':>8}{'WF净($)':>13}"
        f"{'盈利折':>10}{'单折MaxDD($)':>16}{'WF avgR':>9}"
    )
    print("-" * 132)
    for step in steps:
        wf = step.get("walkforward")
        if not wf:
            continue
        agg = wf.get("aggregate_test", {})
        change = step["name"].split("·", 1)[-1].strip().replace("+ ", "")[:12]
        print(
            f"{step['step']:<7}{change:<24}"
            f"{cell(agg.get('total_trades'), 'int'):>9}"
            f"{cell(agg.get('win_rate_pct'), 'pct'):>9}"
            f"{cell(agg.get('profit_factor'), 'pf'):>8}"
            f"{cell(agg.get('net_pnl_dollars'), 'money'):>13}"
            f"{(str(wf.get('positive_folds', 0)) + '/' + str(wf.get('folds', 0))):>10}"
            f"{cell(wf.get('max_fold_drawdown_dollars'), 'money'):>16}"
            f"{cell(agg.get('avg_r'), 'r'):>9}"
        )
    print("-" * 132)


# --------------------------------------------------------------------------- #
# Trade ledger
# --------------------------------------------------------------------------- #
def write_trades_csv(result: v03.V03Result, path: Path) -> None:
    rows: list[dict[str, Any]] = []
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
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat()
                if trade.exit_time_utc
                else "",
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason or "",
                "exit_stop_kind": detail.get("exit_stop_kind") or "",
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "mae_r": round(float(detail.get("mae_r", 0.0) or 0.0), 4),
                "mfe_r": round(float(detail.get("mfe_r", 0.0) or 0.0), 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                "breakeven_armed": bool(detail.get("breakeven_armed", False)),
                "shield_stop": detail.get("shield_stop"),
                "shield_armed_bar_start_utc": detail.get("shield_armed_bar_start_utc") or "",
                "signal_bar_start_utc": detail.get("signal_bar_start_utc", ""),
                "entry_bucket_start_utc": detail.get("entry_bucket_start_utc", ""),
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
) -> dict[str, Any]:
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
    data_span = _fmt_span(list(data.dates))
    print(
        f"5m 数据就绪：{data.total_5m_bars:,} 根 5m bar，{len(data.sessions)} 个 session，"
        f"{data.skipped_buckets:,} 个空 bucket（{prepare_elapsed:.1f}s）"
    )
    print(f"\n开始 5 步累计消融：Step 0 → Step 4，共 5 × (全周期 + 分段 + 29 折 WF) …")

    steps = _run_all_steps(
        data,
        config,
        initial_equity=equity,
        with_segments=with_segments,
        with_walkforward=with_walkforward,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    total_elapsed = time.perf_counter() - started
    self_check = self_check_step0(steps[0])

    print("\n" + "=" * 78)
    print(f"Step 0 自洽校验（V0.2 基线复现）：{'通过 ✅' if self_check['passed'] else '未通过 ❌'}")
    print("=" * 78)
    for check in self_check["checks"]:
        status = "✅" if check["ok"] else "❌"
        print(
            f"  {status} {check['section']:<12} {check['field']:<28} "
            f"V0.2={check['expected']}  Step0={check['actual']}  Δ={check['delta']}"
        )

    elapsed = {
        "full_and_segments_and_wf_per_step": round(total_elapsed / len(steps), 2),
        "total": round(total_elapsed, 2),
    }
    markdown = build_ablation_markdown(
        steps,
        self_check=self_check,
        data_span=data_span,
        total_1m_bars=total_1m_bars,
        total_sessions=len(data.sessions),
        initial_equity=equity,
        elapsed={
            "per_step": round(total_elapsed / len(steps), 2),
            "total": total_elapsed,
        },
    )

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.3",
        "data_source": "Alpaca IEX 1m bars (single venue; NOT consolidated tape)",
        "data_span": data_span,
        "total_1m_bars": total_1m_bars,
        "total_sessions": len(data.sessions),
        "prepared": {
            "total_5m_bars": data.total_5m_bars,
            "skipped_5m_buckets": data.skipped_buckets,
            "symbols": list(data.symbols),
        },
        "structure": V03_STRUCTURE_NOTE,
        "constants": {
            "ENTRY_CUTOFF_ET": v03.ENTRY_CUTOFF_ET.strftime("%H:%M"),
            "BREAKEVEN_TRIGGER_R": v03.BREAKEVEN_TRIGGER_R,
            "BREAKEVEN_OFFSET_R": v03.BREAKEVEN_OFFSET_R,
            "FRACTIONAL_PRECISION": v03.FRACTIONAL_PRECISION,
            "strategies_all": list(STRATEGY_LABELS),
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
        "v02_expected_baseline": V02_EXPECTED,
        "comparison_to_v02": {
            "step0_full": steps[0]["full"]["metrics"],
            "step4_full": steps[-1]["full"]["metrics"],
            "step0_walkforward": steps[0].get("walkforward", {}).get("aggregate_test"),
            "step4_walkforward": steps[-1].get("walkforward", {}).get("aggregate_test"),
        },
        "elapsed_seconds": elapsed,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v03_report.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    markdown_path = report_dir / "v03_ablation.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    final_step = steps[-1]
    final_trades_path = report_dir / "v03_trades.csv"
    # Re-run the final step once to keep the trade ledger in memory for CSV (the
    # step payload carries no TradeRecord objects, only formatted metrics).
    final_result = v03.run_v03_backtest(
        data, config, variant=v03.STEP4_VARIANT, initial_equity=equity
    )
    write_trades_csv(final_result, final_trades_path)

    oos_payload = {
        "generated_at_utc": payload["generated_at_utc"],
        "version": "V0.3",
        "data_source": payload["data_source"],
        "caveat": DATA_CAVEAT,
        "walkforward_grid": payload["walkforward_grid"],
        "final_variant": v03.STEP4_VARIANT.flags(),
        "walkforward": final_step.get("walkforward"),
        "step0_walkforward": steps[0].get("walkforward"),
    }
    oos_path = report_dir / "v03_oos.json"
    oos_path.write_text(
        json.dumps(oos_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_ablation_table(steps)
    print(f"\n产物：")
    print(f"  完整报告 : {report_path}")
    print(f"  Markdown : {markdown_path}")
    print(f"  逐笔流水 : {final_trades_path}（Step4 全周期 {len(final_result.trades)} 笔）")
    print(f"  样本外   : {oos_path}")
    print(f"总运行时长: {total_elapsed:.1f}s")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.3 structural ablation (5 cumulative steps) vs V0.2 on real IEX 1m bars."
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
