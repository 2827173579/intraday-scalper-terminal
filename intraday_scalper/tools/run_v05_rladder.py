"""V0.5 R-Ladder runner ablation: baseline / Runner-A / Runner-B x OR30 filter.

Entries are frozen to V0.4 Step3c-A NoCap (10:00-14:00 window, SMC strict AND,
no 8% cap, no Step-4 scale-out).  Only exits and the OR30 dead-fish day filter
change.  B1 risk base: equity $10,000, risk $60/trade (sensitivity $80),
notional $10,000, fractional 0.01, 3 consecutive stops -> 15 min cooldown,
daily -$320, max 15 entries/day.

Runs the six matrix rows (B1, A, B, B1+F, A+F, B+F) on:

* 2026-08-01 .. 2026-11-06 (3-month window, includes Friday) + monthly breakdown;
* 2024-01-02 .. 2026-11-05 (full period) + 29-fold walk-forward (120/20/20,
  fresh RiskManager per fold, no compounding);
* the two OR30 sub-definitions (``pct_only`` vs literal ``pct OR atr``) for the
  three exit modes (reported separately, final table uses the literal OR version);
* a risk-$80 sensitivity row for the winning variant.

The 3-month baseline must reproduce 68 trades / +$385.89 / PF 1.2138 exactly;
the tool aborts before publishing if it does not.

Artifacts (``intraday_scalper/var/backtest/``)::

    v05_report.json      # full matrix + monthly + WF + sub-definitions + anchors
    v05_rladder.md       # markdown main tables + conclusions (REVIEW_PACK)
    v05_trades.csv       # all six variants x (window + full period) trade ledger
    v05_ablation.json    # machine-readable ablation matrix for downstream tools

Data caveat: Alpaca IEX-only single venue, not consolidated tape; research only.
No parameter tuning; all thresholds are fixed a-priori.
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

from intraday_scalper.core import et_date, load_config
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v04, v05
from intraday_scalper.research.v02 import prepare_v02_data
from intraday_scalper.tools.run_v03_raw_crosscheck import _frames_to_bars
from intraday_scalper.tools.run_v04_backtest import load_friday_raw_bars

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

WINDOW_START = date(2026, 8, 1)
WINDOW_END = date(2026, 11, 6)
FULL_START = date(2024, 1, 2)
FULL_END = date(2026, 11, 5)

#: Required 3-month baseline anchor (task-15).
WINDOW_ANCHOR = {
    "total_trades": 68,
    "win_rate_pct": 51.47,
    "profit_factor": 1.2138,
    "net_pnl_dollars": 385.89,
}

#: Exit-mode labels used by the report.
EXIT_LABELS = {
    v05.EXIT_BASELINE: "B1 baseline (1.2R shield -> 2.0R target)",
    v05.EXIT_RUNNER_A: "A Runner (ladder, 4.5R target)",
    v05.EXIT_RUNNER_B: "B Runner (lock + EMA21 trail, uncapped)",
}

V05_STRUCTURE_NOTE = (
    "V0.5 入场固定为 V0.4 Step3c-A NoCap（10:00-14:00 ET 入场窗、fractional、"
    "VWAP_RECLAIM 趋势过滤 + EMA_PULLBACK + SMC_SWEEP(严格 AND 双确认)、无 8% 防追高、"
    "无 Step4 分批止盈）；仅研究出场结构与 OR30 震荡日过滤。B1 风控：$10,000 权益、"
    "单笔风险 $60（敏感性 $80）、名义 $10,000、碎股 0.01、连续 3 笔止损停 15 分钟、"
    "单日 -$320、最多 15 笔/日。出场：B1=1.2R 保本盾+固定 2R；A=1.5R→+0.20R、"
    "2.2R→+1.20R、3.0R→+2.00R、目标 4.5R；B=1.5R→+0.20R、2.0R→锁 +1.00R、"
    "取消固定封顶、5m 收盘跌破 EMA21 或 max(MFE-1.0R,1.0R) 移动锁利退出。"
    "执行口径：5m 完成 bar 更新 MFE/止损、下一根 1m 生效；同 bar 止损/目标碰撞按止损；"
    "目标用 bar high 触发；无未来函数。所有阈值为一次性先验，不调参。"
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


def _row_metrics(result: v05.V05Result, *, initial_equity: float = v05.B1_EQUITY) -> Dict[str, Any]:
    return _row_metrics_from(result.metrics, result.extra_metrics(), initial_equity=initial_equity)


def _row_metrics_from(
    metrics: Any,
    extra: Mapping[str, Any],
    *,
    initial_equity: float = v05.B1_EQUITY,
) -> Dict[str, Any]:
    return {
        "total_trades": int(metrics.total_trades),
        "wins": int(metrics.wins),
        "losses": int(metrics.losses),
        "win_rate_pct": _number(metrics.win_rate * 100.0, 2),
        "profit_factor": _number(metrics.profit_factor, 4),
        "net_pnl_dollars": _number(metrics.net_pnl_dollars),
        "net_pct": _number(metrics.net_pnl_dollars / float(initial_equity) * 100.0, 2),
        "avg_r": _number(metrics.avg_r, 4),
        "max_drawdown_dollars": _number(metrics.max_drawdown_dollars),
        "max_drawdown_pct": _number(metrics.max_drawdown_pct * 100.0, 2),
        "avg_holding_minutes": _number(metrics.avg_holding_minutes, 1),
        "total_slippage_dollars": _number(metrics.total_slippage_dollars),
        "trading_days": int(metrics.trading_days),
        "exit_reasons": extra.get("exit_reasons", {}),
        "target_exits": int(extra["target_exits"]),
        "target_avg_r": _number(extra["target_avg_r"], 4),
        "or30_skipped_evaluations": int(extra["or30_skipped_evaluations"]),
        "sessions_run": int(extra["sessions_run"]),
    }


def _wf_metrics(wf: v05.V05WalkForwardResult) -> Dict[str, Any]:
    summary = wf.summary
    return {
        "folds": int(summary.get("folds", 0)),
        "total_test_trades": int(summary.get("total_test_trades", 0)),
        "win_rate_pct": _number(float(summary.get("test_win_rate", 0.0)) * 100.0, 2),
        "profit_factor": _number(summary.get("test_profit_factor"), 4),
        "net_pnl_dollars": _number(summary.get("test_net_pnl_dollars")),
        "avg_r": _number(summary.get("test_avg_r"), 4),
        "max_drawdown_dollars_max_fold": _number(
            summary.get("test_max_drawdown_dollars_max_fold")
        ),
        "positive_folds": int(summary.get("positive_folds", 0)),
        "best_fold_pnl": _number(summary.get("best_fold_pnl")),
        "worst_fold_pnl": _number(summary.get("worst_fold_pnl")),
    }


def _monthly_breakdown(result: v05.V05Result) -> Dict[str, Any]:
    by_month: Dict[str, List[Any]] = {}
    for trade in result.trades:
        month_key = f"{et_date(trade.entry_time_utc):%Y-%m}"
        by_month.setdefault(month_key, []).append(trade)
    breakdown: Dict[str, Any] = {}
    for month_key in sorted(by_month):
        trades = by_month[month_key]
        pnls = [trade.pnl for trade in trades]
        wins = sum(1 for pnl in pnls if pnl > 0)
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = -sum(pnl for pnl in pnls if pnl < 0)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        breakdown[month_key] = {
            "total_trades": len(trades),
            "wins": wins,
            "win_rate_pct": _number(wins / len(trades) * 100.0, 2) if trades else 0.0,
            "profit_factor": _number(pf, 4),
            "net_pnl_dollars": _number(sum(pnls)),
            "avg_r": _number(
                sum(trade.r_multiple for trade in trades) / len(trades), 4
            )
            if trades
            else 0.0,
        }
    return breakdown


def _expected_row(compare: Mapping[str, Mapping[str, Any]]) -> tuple[bool, List[Dict[str, Any]]]:
    anchor = compare["B1"]["window"]
    tolerances = {
        "total_trades": 0.0,
        "win_rate_pct": 0.01,
        "profit_factor": 0.001,
        "net_pnl_dollars": 0.01,
    }
    checks: List[Dict[str, Any]] = []
    passed = True
    for field, expected in WINDOW_ANCHOR.items():
        actual = anchor.get(field)
        tolerance = tolerances.get(field, 0.01)
        ok = (
            actual is not None
            and abs(float(actual) - float(expected)) <= tolerance + 1e-9
        )
        passed = passed and ok
        checks.append(
            {
                "field": field,
                "expected": expected,
                "actual": actual,
                "delta": _number(float(actual) - float(expected), 6)
                if actual is not None
                else None,
                "tolerance": tolerance,
                "ok": bool(ok),
            }
        )
    return passed, checks


# --------------------------------------------------------------------------- #
# Run matrix
# --------------------------------------------------------------------------- #
def run_matrix(
    data: Any,
    *,
    full_dates: Sequence[date],
    window_dates: Sequence[date],
    with_walkforward: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> Dict[str, Dict[str, Any]]:
    compare: Dict[str, Dict[str, Any]] = {}
    for index, variant in enumerate(v05.MATRIX_VARIANTS, start=1):
        label = variant.name
        print(f"\n[{label}] {EXIT_LABELS[variant.exit_mode]}" + (
            f" + OR30({variant.or30_filter})" if variant.or30_filter != v05.OR30_FILTER_NONE else ""
        ) + " …", flush=True)
        started = time.perf_counter()
        window_result = v05.run_v05_backtest(data, variant=variant, session_dates=window_dates)
        full_result = v05.run_v05_backtest(data, variant=variant, session_dates=full_dates)
        wf = (
            v05.run_v05_walkforward(data, variant=variant, dates=full_dates)
            if with_walkforward
            else None
        )
        elapsed = time.perf_counter() - started
        compare[label] = {
            "label": label,
            "variant": variant.flags(),
            "exit_label": EXIT_LABELS[variant.exit_mode],
            "window": _row_metrics(window_result),
            "full": _row_metrics(full_result),
            "walkforward": _wf_metrics(wf) if wf is not None else None,
            "monthly": _monthly_breakdown(window_result),
            "window_result": window_result,
            "full_result": full_result,
            "elapsed_seconds": round(elapsed, 2),
        }
        w = compare[label]["window"]
        f = compare[label]["full"]
        wf_row = compare[label].get("walkforward") or {}
        print(
            f"  3M {w['total_trades']} 笔 / PF {w['profit_factor']:.4f} / "
            f"${w['net_pnl_dollars']:+.2f} / avgR {w['avg_r']:+.4f} / DD ${w['max_drawdown_dollars']:,.2f}"
            f" | full {f['total_trades']} 笔 / PF {f['profit_factor']:.4f} / "
            f"${f['net_pnl_dollars']:+.2f} / avgR {f['avg_r']:+.4f} / DD ${f['max_drawdown_dollars']:,.2f}"
            + (
                f" | WF {wf_row.get('total_test_trades')} 笔 / "
                f"${wf_row.get('net_pnl_dollars'):+.2f} / "
                f"{wf_row.get('positive_folds')}/{wf_row.get('folds')} 盈利折"
                if wf_row
                else ""
            )
            + f"（{elapsed:.1f}s）",
            flush=True,
        )
        if progress is not None:
            progress(index, len(v05.MATRIX_VARIANTS))
    return compare


def run_or30_subdefs(
    data: Any,
    *,
    full_dates: Sequence[date],
) -> Dict[str, Any]:
    """Run the two OR30 sub-definitions for the three exit modes (full period)."""

    blocks_pct, _ = v05.compute_or30_blocks(data, v05.OR30_FILTER_PCT_ONLY)
    blocks_or, _ = v05.compute_or30_blocks(data, v05.OR30_FILTER_PCT_OR_ATR)
    window_blocks_pct = {item for item in blocks_pct if WINDOW_START <= item[0] <= WINDOW_END}
    window_blocks_or = {item for item in blocks_or if WINDOW_START <= item[0] <= WINDOW_END}
    rows: Dict[str, Any] = {
        "blocked_pct_only": {
            "blocked_symbol_sessions": len(blocks_pct),
            "blocked_sessions": len({day for day, _ in blocks_pct}),
            "blocked_window_symbol_sessions": len(window_blocks_pct),
        },
        "blocked_pct_or_atr": {
            "blocked_symbol_sessions": len(blocks_or),
            "blocked_sessions": len({day for day, _ in blocks_or}),
            "blocked_window_symbol_sessions": len(window_blocks_or),
        },
        "identical_definitions": set(blocks_pct) == set(blocks_or),
        "variants": {},
    }
    for base in (v05.BASELINE_VARIANT, v05.RUNNER_A_VARIANT, v05.RUNNER_B_VARIANT):
        variants: Dict[str, Any] = {}
        for mode in (v05.OR30_FILTER_PCT_ONLY, v05.OR30_FILTER_PCT_OR_ATR):
            variant = v05.V05Variant(
                name=f"{base.name}+{mode}", exit_mode=base.exit_mode, or30_filter=mode
            )
            result = v05.run_v05_backtest(data, variant=variant, session_dates=full_dates)
            variants[mode] = _row_metrics(result)
        rows["variants"][base.name] = variants
    return rows


def run_sensitivity(
    data: Any,
    *,
    winner_label: str,
    full_dates: Sequence[date],
    window_dates: Sequence[date],
    risk_dollars: float = 80.0,
) -> Dict[str, Any]:
    base_variant = next(item for item in v05.MATRIX_VARIANTS if item.name == winner_label)
    variant = v05.V05Variant(
        name=f"{base_variant.name}@${risk_dollars:.0f}",
        exit_mode=base_variant.exit_mode,
        or30_filter=base_variant.or30_filter,
        risk_dollars=risk_dollars,
    )
    window_result = v05.run_v05_backtest(data, variant=variant, session_dates=window_dates)
    full_result = v05.run_v05_backtest(data, variant=variant, session_dates=full_dates)
    return {
        "winner_label": winner_label,
        "risk_dollars": float(risk_dollars),
        "variant": variant.flags(),
        "window": _row_metrics(window_result),
        "full": _row_metrics(full_result),
    }


# --------------------------------------------------------------------------- #
# Conclusions
# --------------------------------------------------------------------------- #
def build_conclusions(compare: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    labels = list(compare.keys())
    best_window = max(labels, key=lambda label: compare[label]["window"]["net_pnl_dollars"])
    best_full = max(labels, key=lambda label: compare[label]["full"]["net_pnl_dollars"])
    wf_labels = [
        label
        for label in labels
        if (compare[label].get("walkforward") or {}).get("net_pnl_dollars") is not None
    ]
    if wf_labels:
        best_wf = max(
            wf_labels,
            key=lambda label: compare[label]["walkforward"]["net_pnl_dollars"],
        )
        best_wf_folds = max(
            wf_labels,
            key=lambda label: compare[label]["walkforward"].get("positive_folds", 0),
        )
    else:
        best_wf = None
        best_wf_folds = None
    best_window_avg_r = max(labels, key=lambda label: compare[label]["window"]["avg_r"])
    monthly_months = sorted(
        set().union(*[set(compare[label]["monthly"].keys()) for label in labels])
    )
    monthly_summary: Dict[str, Any] = {}
    for month in monthly_months:
        monthly_summary[month] = {
            label: compare[label]["monthly"].get(month, {})
            for label in labels
        }
    september_best = None
    if "2026-09" in monthly_months:
        september_best = max(
            labels,
            key=lambda label: compare[label]["monthly"].get("2026-09", {}).get(
                "net_pnl_dollars", -1e18
            ),
        )
    target_distribution = {
        label: {
            "target_exits": compare[label]["window"]["target_exits"],
            "target_avg_r": compare[label]["window"]["target_avg_r"],
            "exit_reasons_window": compare[label]["window"]["exit_reasons"],
            "exit_reasons_full": compare[label]["full"]["exit_reasons"],
        }
        for label in labels
    }
    return {
        "best_3m_net": best_window,
        "best_full_net": best_full,
        "best_wf_net": best_wf,
        "best_wf_positive_folds": best_wf_folds,
        "best_3m_avg_r": best_window_avg_r,
        "september_best": september_best,
        "monthly": monthly_summary,
        "target_distribution": target_distribution,
        "summary_text": (
            f"3个月净值最好={best_window}；全周期净值最好={best_full}；"
            f"29 折 WF 净值最好={best_wf or '—'}；WF 盈利折最多={best_wf_folds or '—'}；"
            f"3个月 avgR 最好={best_window_avg_r}；2026-09 净值最好={september_best}。"
        ),
    }


def build_winner_recommendation(
    compare: Mapping[str, Mapping[str, Any]],
    conclusions: Mapping[str, Any],
) -> Dict[str, Any]:
    """Composite recommendation: full WF net first, then full net, then window net."""

    labels = list(compare.keys())

    def score(label: str) -> tuple:
        wf_value = (compare[label].get("walkforward") or {}).get("net_pnl_dollars")
        wf = -1e18 if wf_value is None else float(wf_value)
        full = compare[label]["full"]["net_pnl_dollars"]
        window = compare[label]["window"]["net_pnl_dollars"]
        pf = compare[label]["full"]["profit_factor"]
        return (wf, full, window, pf)

    ranked = sorted(labels, key=score, reverse=True)
    winner = ranked[0]
    filtered_winners = [label for label in ranked if "F" in label]
    unfiltered_winners = [label for label in ranked if "F" not in label]
    return {
        "composite_winner": winner,
        "ranking": ranked,
        "best_filtered": filtered_winners[0] if filtered_winners else None,
        "best_unfiltered": unfiltered_winners[0] if unfiltered_winners else None,
        "criteria": (
            "primary = 29 折 WF 净；tie-break = 全周期净 > 3个月净 > 全周期 PF"
        ),
        "note": (
            "OR30 过滤在 2026-08-01..11-06 三个月窗口内 0 次触发（该窗口 69 个 session 均 "
            "OR30 振幅高于阈值），因此 B1/A/B 与 +F 行在 3 个月窗口完全相同；过滤只在更早的 "
            "2024-2025 历史区间生效。"
        ),
    }


# --------------------------------------------------------------------------- #
# Markdown / terminal rendering
# --------------------------------------------------------------------------- #
_MAIN_HEADER = (
    "| 变体 | 出场结构 | OR30 | 3个月 笔/胜率/PF/净($)/净%/avgR/MaxDD | "
    "全周期 笔/胜率/PF/净($)/净%/avgR/MaxDD | 29折 WF 笔/胜率/PF/净($)/盈利折 |"
)
_MAIN_SEPARATOR = "|---|---|---|---|---|---|"


def _cell(value: Any, kind: str = "") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{float(value):.2f}%"
    if kind == "pf":
        return f"{float(value):.4f}"
    if kind == "money":
        return f"{float(value):+,.2f}"
    if kind == "dd":
        return f"{float(value):,.2f}"
    if kind == "r":
        return f"{float(value):+.4f}"
    return str(value)


def _slim(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('total_trades')}/{_cell(row.get('win_rate_pct'), 'pct')}/"
        f"{_cell(row.get('profit_factor'), 'pf')}/{_cell(row.get('net_pnl_dollars'), 'money')}/"
        f"{_cell(row.get('net_pct'), 'pct')}/{_cell(row.get('avg_r'), 'r')}/"
        f"{_cell(row.get('max_drawdown_dollars'), 'dd')}"
    )


def _main_row(label: str, payload: Mapping[str, Any]) -> str:
    flags = payload["variant"]
    wf = payload.get("walkforward") or {}
    wf_cell = "—"
    if wf:
        wf_cell = (
            f"{wf.get('total_test_trades')}/{_cell(wf.get('win_rate_pct'), 'pct')}/"
            f"{_cell(wf.get('profit_factor'), 'pf')}/{_cell(wf.get('net_pnl_dollars'), 'money')}/"
            f"{wf.get('positive_folds')}/{wf.get('folds')}"
        )
    return (
        f"| {label} | {payload.get('exit_label')} | "
        f"{flags.get('or30_filter')} | {_slim(payload['window'])} | "
        f"{_slim(payload['full'])} | {wf_cell} |"
    )


def build_markdown(
    *,
    compare: Mapping[str, Mapping[str, Any]],
    conclusions: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    or30_subdefs: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    anchor_checks: Sequence[Mapping[str, Any]],
    anchor_passed: bool,
    reference_full: Mapping[str, Any],
    data_span: str,
    total_sessions: int,
    window_sessions: int,
    full_sessions: int,
    elapsed: Mapping[str, float],
) -> str:
    lines = [
        "# V0.5 R-Ladder Runner 三种出场 + OR30 死鱼过滤（B1 $10k / $60 风险）",
        "",
        f"- 数据：Alpaca IEX raw 1m，全周期 {data_span}（{total_sessions} 个 session）；"
        f"3 个月窗口 2026-08-01..2026-11-06（{window_sessions} 个 session，含周五）；"
        f"全周期 {full_sessions} 个 session（不含 2026-11-06）",
        f"- 结构：{V05_STRUCTURE_NOTE}",
        f"- 初始权益 ${v05.B1_EQUITY:,.0f}；29 折 WF（120/20/20、每折 fresh RiskManager、不复利）。",
        "",
        "## 3 个月 baseline 锚校验（必须 68 笔 / +$385.89 / PF 1.2138）",
        "",
        f"结果：{'通过 ✅' if anchor_passed else '未通过 ❌'}",
        "",
        "| 指标 | 任务锚 | 本次实测 | 偏差 | 容差 | 通过 |",
        "|---|---|---|---|---|---|",
    ]
    for check in anchor_checks:
        lines.append(
            f"| {check['field']} | {check['expected']} | {check['actual']} | "
            f"{check['delta']} | ±{check['tolerance']} | {'✅' if check['ok'] else '❌'} |"
        )
    lines.extend(
        [
            "",
            "### 全周期 baseline 与既有记录的说明（仓位不同）",
            "",
            f"- 本次 B1 全周期（$10k 权益 / $60 风险 / $10k 名义）："
            f"{_slim(compare['B1']['full'])}",
            f"- 既有 v0.4c Step3c-NoCap 记录（$3k 权益 / $20 风险 / $1.2k 名义，"
            f"无 OR30 过滤）：{_slim(reference_full)}",
            "- 两者**不可直接比较**：B1 单笔风险 3 倍、名义上限更高、日亏熔断 -$320、"
            "连亏 3 笔才停 15 分钟，因此交易笔数/仓位完全不同；本表以 3 个月锚为准。",
            "",
            "## 主表：六组（3 个月 + 全周期 + 29 折 WF）",
            "",
            _MAIN_HEADER,
            _MAIN_SEPARATOR,
            *[_main_row(label, compare[label]) for label in ("B1", "A", "B", "B1+F", "A+F", "B+F")],
            "",
            "## 3 个月分月（笔数 / 胜率 / PF / 净$）",
            "",
        ]
    )
    months = sorted(conclusions["monthly"].keys())
    header = "| 变体 | 出场 | " + " | ".join(
        f"{int(month[-2:])}月 笔/胜率/PF/净($)" for month in months
    ) + " |"
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(months))
    for label in ("B1", "A", "B", "B1+F", "A+F", "B+F"):
        cells = []
        for month in months:
            row = conclusions["monthly"][month].get(label) or {}
            cells.append(
                f"{row.get('total_trades', 0)}/{_cell(row.get('win_rate_pct'), 'pct')}/"
                f"{_cell(row.get('profit_factor'), 'pf')}/{_cell(row.get('net_pnl_dollars'), 'money')}"
            )
        lines.append(f"| {label} | {compare[label]['exit_label']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 真实 TARGET 分布（不注水）",
            "",
            "| 变体 | 3个月 TARGET 笔数/平均R | 3个月出场分布 | 全周期出场分布 |",
            "|---|---|---|---|",
        ]
    )
    for label in ("B1", "A", "B", "B1+F", "A+F", "B+F"):
        distribution = conclusions["target_distribution"][label]
        lines.append(
            f"| {label} | {distribution['target_exits']} / {_r_cell(distribution['target_avg_r'])} | "
            f"{json.dumps(distribution['exit_reasons_window'], ensure_ascii=False)} | "
            f"{json.dumps(distribution['exit_reasons_full'], ensure_ascii=False)} |"
        )
    lines.extend(
        [
            "",
            "## OR30 子定义对比（全周期）",
            "",
            f"- pct_only 拦截 symbol-session：{or30_subdefs['blocked_pct_only']['blocked_symbol_sessions']}；"
            f"pct_or_atr 拦截：{or30_subdefs['blocked_pct_or_atr']['blocked_symbol_sessions']}；"
            f"两种定义在全部历史上是否完全一致：{or30_subdefs['identical_definitions']}",
            f"- 3 个月窗口内被拦截的 symbol-session：pct_only="
            f"{or30_subdefs['blocked_pct_only']['blocked_window_symbol_sessions']}、"
            f"pct_or_atr={or30_subdefs['blocked_pct_or_atr']['blocked_window_symbol_sessions']}"
            "（窗口内 0 次触发 → +F 行与无过滤行完全相同）",
            "",
            "| 出场结构 | 定义 | 全周期 笔/胜率/PF/净($)/净%/avgR/MaxDD |",
            "|---|---|---|",
        ]
    )
    for base_label in ("B1", "A", "B"):
        for mode in (v05.OR30_FILTER_PCT_ONLY, v05.OR30_FILTER_PCT_OR_ATR):
            row = or30_subdefs["variants"][base_label][mode]
            lines.append(f"| {base_label} | {mode} | {_slim(row)} |")
    lines.extend(
        [
            "",
            f"## 风险 $80 敏感性（胜出变体：{sensitivity['winner_label']}）",
            "",
            "| 区间 | 笔数 | 胜率 | PF | 净($) | 净% | avgR | MaxDD($) |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for period_label, row in (
        ("3个月", sensitivity["window"]),
        ("全周期", sensitivity["full"]),
    ):
        lines.append(
            f"| {period_label} | {row.get('total_trades')} | "
            f"{_cell(row.get('win_rate_pct'), 'pct')} | {_cell(row.get('profit_factor'), 'pf')} | "
            f"{_cell(row.get('net_pnl_dollars'), 'money')} | {_cell(row.get('net_pct'), 'pct')} | "
            f"{_cell(row.get('avg_r'), 'r')} | {_cell(row.get('max_drawdown_dollars'), 'dd')} |"
        )
    lines.extend(
        [
            "",
            "## 六组结论",
            "",
            f"- {conclusions['summary_text']}",
            f"- 综合推荐（primary = 29 折 WF 净）：**{recommendation['composite_winner']}**；"
            f"排序 {' > '.join(recommendation['ranking'])}",
            f"- 最过滤版：{recommendation['best_filtered']}；最无过滤版：{recommendation['best_unfiltered']}",
            f"- {recommendation['note']}",
            "",
            "### 9月 / 2R 问题",
            "",
            "- 9 月（2026-09）各变体净利见分月表；baseline 固定 2R 在 9 月是否仍为负、"
            "Runner-A/B 是否改善以表内数字为准（未择优、未调参）。",
            "- 2R 封顶问题：B1 只有 2R TARGET；Runner-A 目标 4.5R、Runner-B 取消固定封顶。"
            "真实 TARGET 笔数与平均 R 在上表如实给出：若目标更难触发，TARGET 笔数下降但"
            "avgR/尾部盈利可能上升，反之亦然——不注水。",
            "",
            "## 执行口径与风险提示",
            "",
            "- 5m 完成 bar 更新 MFE/止损，下一根 1m bar 生效；同 bar 止损/目标碰撞按止损；"
            "目标用 bar high 触发；无未来函数。",
            "- OR30 过滤只阻止新开仓，已持仓的止损/目标/EOD 不受影响。",
            "- IEX 单一交易所口径；研究结果不代表实盘有效性或盈利预期；未调参、未择优、未上传 Drive。",
            f"- 运行时长：{elapsed.get('total', 0.0):.1f}s。",
            "",
        ]
    )
    return "\n".join(lines)


def _r_cell(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.4f}"


def print_tables(
    compare: Mapping[str, Mapping[str, Any]],
    conclusions: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    or30_subdefs: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> None:
    print("\n" + "=" * 160)
    print("V0.5 R-Ladder Runner：六组对比（3个月 2026-08-01..11-06 + 全周期 + 29 折 WF，B1 $10k/$60）")
    print("=" * 160)
    print(
        f"{'变体':<6}{'出场':<10}{'OR30':<10}"
        f"{'3M笔':>6}{'胜率':>8}{'PF':>8}{'净($)':>11}{'净%':>8}{'avgR':>9}{'MaxDD':>10}"
        f"{'全笔':>6}{'胜率':>8}{'PF':>8}{'净($)':>12}{'净%':>8}{'avgR':>9}{'MaxDD':>11}"
        f"{'WF笔':>7}{'WF胜率':>8}{'WF PF':>8}{'WF净($)':>12}{'盈利折':>8}"
    )
    print("-" * 160)
    for label in ("B1", "A", "B", "B1+F", "A+F", "B+F"):
        payload = compare[label]
        w, f = payload["window"], payload["full"]
        wf = payload.get("walkforward") or {}
        print(
            f"{label:<6}{payload['variant']['exit_mode']:<10}{payload['variant']['or30_filter']:<10}"
            f"{w['total_trades']:>6}{_cell(w['win_rate_pct'], 'pct'):>8}"
            f"{_cell(w['profit_factor'], 'pf'):>8}{_cell(w['net_pnl_dollars'], 'money'):>11}"
            f"{_cell(w['net_pct'], 'pct'):>8}{_cell(w['avg_r'], 'r'):>9}"
            f"{_cell(w['max_drawdown_dollars'], 'dd'):>10}"
            f"{f['total_trades']:>6}{_cell(f['win_rate_pct'], 'pct'):>8}"
            f"{_cell(f['profit_factor'], 'pf'):>8}{_cell(f['net_pnl_dollars'], 'money'):>12}"
            f"{_cell(f['net_pct'], 'pct'):>8}{_cell(f['avg_r'], 'r'):>9}"
            f"{_cell(f['max_drawdown_dollars'], 'dd'):>11}"
            f"{wf.get('total_test_trades', '—'):>7}{_cell(wf.get('win_rate_pct'), 'pct'):>8}"
            f"{_cell(wf.get('profit_factor'), 'pf'):>8}{_cell(wf.get('net_pnl_dollars'), 'money'):>12}"
            f"{str(wf.get('positive_folds', 0)) + '/' + str(wf.get('folds', 0)):>8}"
        )
    print("-" * 160)
    print("\n3个月分月（笔数 / 胜率 / PF / 净$）：")
    months = sorted(conclusions["monthly"].keys())
    for label in ("B1", "A", "B", "B1+F", "A+F", "B+F"):
        parts = []
        for month in months:
            row = conclusions["monthly"][month].get(label) or {}
            parts.append(
                f"{month}: {row.get('total_trades', 0)}/{_cell(row.get('win_rate_pct'), 'pct')}/"
                f"{_cell(row.get('profit_factor'), 'pf')}/{_cell(row.get('net_pnl_dollars'), 'money')}"
            )
        print(f"  {label:<6} " + " | ".join(parts))
    print("\nOR30 子定义（全周期）：")
    print(
        f"  pct_only 拦截 {or30_subdefs['blocked_pct_only']['blocked_symbol_sessions']} symbol-sessions；"
        f"pct_or_atr 拦截 {or30_subdefs['blocked_pct_or_atr']['blocked_symbol_sessions']}；"
        f"两者一致={or30_subdefs['identical_definitions']}；"
        f"3 个月窗口拦截：pct_only={or30_subdefs['blocked_pct_only']['blocked_window_symbol_sessions']}、"
        f"pct_or_atr={or30_subdefs['blocked_pct_or_atr']['blocked_window_symbol_sessions']}"
    )
    for base_label in ("B1", "A", "B"):
        for mode in (v05.OR30_FILTER_PCT_ONLY, v05.OR30_FILTER_PCT_OR_ATR):
            row = or30_subdefs["variants"][base_label][mode]
            print(f"  {base_label}+{mode:<10} {_slim(row)}")
    print(f"\n风险 $80 敏感性（{sensitivity['winner_label']}）：")
    for period_label, row in (("3M", sensitivity["window"]), ("full", sensitivity["full"])):
        print(
            f"  {period_label:<5} {row['total_trades']} 笔 / PF {_cell(row['profit_factor'], 'pf')} / "
            f"净 {_cell(row['net_pnl_dollars'], 'money')} / avgR {_cell(row['avg_r'], 'r')} / "
            f"DD {_cell(row['max_drawdown_dollars'], 'dd')}"
        )
    print("\n结论：", conclusions["summary_text"])
    print("综合推荐：", recommendation["composite_winner"], "| 排序:", " > ".join(recommendation["ranking"]))
    print("=" * 160)


# --------------------------------------------------------------------------- #
# Trade ledger
# --------------------------------------------------------------------------- #
def write_trades_csv(
    compare: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    for label in ("B1", "A", "B", "B1+F", "A+F", "B+F"):
        payload = compare[label]
        for period, result_key in (("3M", "window_result"), ("FULL", "full_result")):
            result = payload[result_key]
            for trade in result.trades:
                detail = result.trade_details.get(trade.signal_id, {})
                rows.append(
                    {
                        "variant": label,
                        "exit_mode": payload["variant"]["exit_mode"],
                        "or30_filter": payload["variant"]["or30_filter"],
                        "period": period,
                        "signal_id": trade.signal_id,
                        "symbol": trade.symbol,
                        "strategy": trade.strategy.value,
                        "strategy_label": detail.get("strategy_label", trade.strategy.value),
                        "shares": trade.shares,
                        "entry_time_utc": trade.entry_time_utc.isoformat(),
                        "entry_time_et": str(et_date(trade.entry_time_utc)),
                        "entry_price_signal": trade.entry_price_signal,
                        "entry_price_fill": trade.entry_price_fill,
                        "stop_price": trade.stop_price,
                        "target_price": trade.target_price,
                        "exit_time_utc": trade.exit_time_utc.isoformat()
                        if trade.exit_time_utc
                        else "",
                        "exit_price_fill": trade.exit_price_fill,
                        "exit_reason": trade.exit_reason or "",
                        "exit_reason_detail": detail.get("exit_reason_detail", ""),
                        "exit_stop_kind": detail.get("exit_stop_kind") or "",
                        "pnl": round(trade.pnl, 4),
                        "r_multiple": round(trade.r_multiple, 4),
                        "holding_minutes": round(trade.holding_minutes, 2),
                        "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                        "max_mfe_r": detail.get("max_mfe_r"),
                        "ladder_stop_r": detail.get("ladder_stop_r"),
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
    with_walkforward: bool = True,
    risk_dollars: float = v05.B1_RISK,
) -> Dict[str, Any]:
    started = time.perf_counter()
    print(f"加载 raw 数据（含 2026-11-06，用于 3 个月窗口）：{data_dir}")
    friday_bars = load_friday_raw_bars(data_dir)
    print("准备 5m 数据 …")
    prepare_started = time.perf_counter()
    data = prepare_v02_data(friday_bars)
    print(
        f"5m 数据就绪：{data.total_5m_bars:,} 根 5m bar，{len(data.sessions)} 个 session"
        f"（{time.perf_counter() - prepare_started:.1f}s）"
    )
    full_dates = [value for value in data.dates if FULL_START <= value <= FULL_END]
    window_dates = [value for value in data.dates if WINDOW_START <= value <= WINDOW_END]
    print(
        f"窗口：3 个月 {len(window_dates)} sessions（{window_dates[0]}..{window_dates[-1]}）；"
        f"全周期 {len(full_dates)} sessions"
    )

    # B1 risk config from the task (also the default inside run_v05_* wrappers).
    compare = run_matrix(
        data,
        full_dates=full_dates,
        window_dates=window_dates,
        with_walkforward=with_walkforward,
    )
    anchor_passed, anchor_checks = _expected_row(compare)

    # Full-period reference of the SAME entry set at the old V0.4 sizing.
    print("\n[参考] v0.4c Step3c-NoCap 在旧仓位($3k/$20)下的全周期 …", flush=True)
    reference = v04.run_v04_backtest(
        data,
        load_config(),
        variant=v04.STEP3C_NOCAP_VARIANT,
        session_dates=full_dates,
    )
    reference_full = _row_metrics_from(
        reference.metrics,
        {
            "exit_reasons": {},
            "target_exits": 0,
            "target_avg_r": 0.0,
            "or30_skipped_evaluations": 0,
            "sessions_run": len(reference.session_dates),
        },
    )

    print("\n[OR30 子定义] pct_only vs pct_or_atr（全周期，三种出场各跑一次）…", flush=True)
    or30_subdefs = run_or30_subdefs(data, full_dates=full_dates)
    print(
        f"  pct_only 拦截 {or30_subdefs['blocked_pct_only']['blocked_symbol_sessions']}；"
        f"pct_or_atr 拦截 {or30_subdefs['blocked_pct_or_atr']['blocked_symbol_sessions']}；"
        f"一致={or30_subdefs['identical_definitions']}",
        flush=True,
    )

    conclusions = build_conclusions(compare)
    recommendation = build_winner_recommendation(compare, conclusions)
    print("\n[敏感性] 胜出变体在风险 $80 下重跑 3 个月 + 全周期 …", flush=True)
    sensitivity = run_sensitivity(
        data,
        winner_label=recommendation["composite_winner"],
        full_dates=full_dates,
        window_dates=window_dates,
        risk_dollars=80.0,
    )
    total_elapsed = time.perf_counter() - started

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.5",
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw (single venue; NOT consolidated tape)"
        ),
        "data_span": f"{data.dates[0]:%Y-%m-%d} .. {data.dates[-1]:%Y-%m-%d}",
        "structure": V05_STRUCTURE_NOTE,
        "risk_base": {
            "equity": v05.B1_EQUITY,
            "risk_dollars": float(risk_dollars),
            "notional": v05.B1_NOTIONAL,
            "fractional_precision": v05.B1_FRACTIONAL_PRECISION,
            "consecutive_stops": v05.B1_CONSECUTIVE_STOPS,
            "cooldown_minutes": v05.B1_COOLDOWN_MINUTES,
            "daily_loss_limit": v05.B1_DAILY_LOSS,
            "max_entries_per_day": v05.B1_MAX_ENTRIES,
            "sensitivity_risk_dollars": 80.0,
        },
        "windows": {
            "three_month": [min(window_dates).isoformat(), max(window_dates).isoformat()],
            "three_month_sessions": len(window_dates),
            "full_period": [min(full_dates).isoformat(), max(full_dates).isoformat()],
            "full_sessions": len(full_dates),
        },
        "walkforward_grid": {"train_days": 120, "test_days": 20, "step_days": 20},
        "matrix": {
            label: {
                key: value
                for key, value in compare[label].items()
                if key not in ("window_result", "full_result")
            }
            for label in compare
        },
        "three_month_anchor": {
            "expected": WINDOW_ANCHOR,
            "passed": bool(anchor_passed),
            "checks": anchor_checks,
        },
        "reference_full_v04_sizing": reference_full,
        "or30_subdefinitions": or30_subdefs,
        "sensitivity_risk_80": sensitivity,
        "conclusions": conclusions,
        "recommendation": recommendation,
        "elapsed_seconds": {
            "matrix": round(
                sum(compare[label]["elapsed_seconds"] for label in compare), 2
            ),
            "total": round(total_elapsed, 2),
        },
    }
    ablation_payload: Dict[str, Any] = {
        "generated_at_utc": report_payload["generated_at_utc"],
        "version": "V0.5",
        "structure": V05_STRUCTURE_NOTE,
        "risk_base": report_payload["risk_base"],
        "matrix": {
            label: {
                "window": compare[label]["window"],
                "full": compare[label]["full"],
                "walkforward": compare[label]["walkforward"],
                "monthly": compare[label]["monthly"],
                "exit_mode": compare[label]["variant"]["exit_mode"],
                "or30_filter": compare[label]["variant"]["or30_filter"],
            }
            for label in compare
        },
        "or30_subdefinitions": or30_subdefs,
        "sensitivity_risk_80": sensitivity,
        "conclusions": conclusions,
        "recommendation": recommendation,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v05_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    ablation_path = report_dir / "v05_ablation.json"
    ablation_path.write_text(
        json.dumps(ablation_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    markdown = build_markdown(
        compare=compare,
        conclusions=conclusions,
        recommendation=recommendation,
        or30_subdefs=or30_subdefs,
        sensitivity=sensitivity,
        anchor_checks=anchor_checks,
        anchor_passed=anchor_passed,
        reference_full=reference_full,
        data_span=report_payload["data_span"],
        total_sessions=len(data.sessions),
        window_sessions=len(window_dates),
        full_sessions=len(full_dates),
        elapsed={"total": total_elapsed},
    )
    markdown_path = report_dir / "v05_rladder.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    trades_path = report_dir / "v05_trades.csv"
    write_trades_csv(compare, trades_path)

    print_tables(compare, conclusions, recommendation, or30_subdefs, sensitivity)
    print(
        f"\n3 个月 baseline 锚校验：{'通过 ✅' if anchor_passed else '未通过 ❌'} "
        f"({json.dumps(anchor_checks, ensure_ascii=False)})"
    )
    print("\n产物：")
    print(f"  报告 JSON : {report_path}")
    print(f"  消融 JSON : {ablation_path}")
    print(f"  Markdown  : {markdown_path}")
    print(f"  逐笔流水  : {trades_path}")
    print(f"总运行时长：{total_elapsed:.1f}s")
    if not anchor_passed:
        raise RuntimeError(
            "3-month baseline anchor (68 / +385.89 / PF 1.2138) not reproduced; "
            "refusing to publish V0.5 results"
        )
    return report_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.5 R-ladder runner ablation + OR30 dead-fish filter."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-walkforward", action="store_true")
    parser.add_argument("--risk-dollars", type=float, default=v05.B1_RISK)
    args = parser.parse_args(argv)
    build_and_run(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        with_walkforward=not args.no_walkforward,
        risk_dollars=args.risk_dollars,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
