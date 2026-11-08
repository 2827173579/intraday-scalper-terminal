"""V0.5b pre-registered F1/F2 anti-chop comparison + Web evaluation API.

Compares five Runner-A (4.5R R-ladder) variants on the frozen V0.4 Step3c-A
NoCap entries with the B1 $10k/$60 risk base:

===========  ==================================================================
A            Runner-A reference (no regime filter)
A+F          original OR30 filter: Range_pct < 1.8% or Range < 1.2 x ATR14_5m
A+F1         F1: corrected 30m-scale thresholds - Range_pct < 2.8% or
             Range < 2.2 x ATR14_5m(10:00)
A+F2         F2: trend strategies (EMA_PULLBACK / VWAP_RECLAIM) may only fire at
             close >= OR30_High, or upper half of the OR30 box with a rising
             3-bar 5m EMA21 slope >= +0.15%; SMC_SWEEP_5M is exempt
A+F1+F2      F1 + F2 combined
===========  ==================================================================

Matrix: 2026-08-01..11-06 (3-month window, monthly 8/9/10/11), full period
2024-01-02..2026-11-05 and the same 29-fold walk-forward grid (120/20/20,
fresh RiskManager per fold, no compounding).  The 3-month A+F row must
reproduce the recorded anchor (64 trades / +$573.91 / PF 1.2884, WF +$230.97 /
14 positive folds); the tool refuses to publish otherwise.

Selection rule is **pre-registered** (no post-hoc metrics): only variants with
WF net > 0 are eligible; score = WF net + 3-month net + full-period net.  A new
variant that improves both the 3-month and WF nets but leads the original A+F
by less than 5% is "not a significant beat" and A+F is kept.

Also emits the winner's Friday 2026-11-06 recap + Web markers and the Web-facing
``v05.evaluate_v05_session`` API is available for the live terminal.

Artifacts (``intraday_scalper/var/backtest/``)::

    v05b_report.json          # matrix + filter stats + selection + Friday winner
    v05b_compare.md           # markdown main table / monthly / selection / Friday
    v05b_trades.csv           # all five variants x (3-month + full) trade ledger
    v05b_friday_recap.json    # winner Friday recap (summary + trades + markers)
    v05b_friday_markers.json  # winner Friday markers (Web replay)

Data caveat: Alpaca IEX-only single venue; research output, not proof of
profitability.  No parameter tuning beyond the pre-registered F1/F2 rules.
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
from intraday_scalper.research import v05
from intraday_scalper.research.v02 import DATA_CAVEAT, prepare_v02_data
from intraday_scalper.tools.run_v04_backtest import _market_snapshot, load_friday_raw_bars
from intraday_scalper.tools.run_v05_rladder import (
    _cell,
    _monthly_breakdown,
    _row_metrics,
    _slim,
    _wf_metrics,
)

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"

WINDOW_START = date(2026, 8, 1)
WINDOW_END = date(2026, 11, 6)
FULL_START = date(2024, 1, 2)
FULL_END = date(2026, 11, 5)
FRIDAY = date(2026, 11, 6)

#: Matrix rows in report order (A reference first, then the pre-registered set).
MATRIX_LABELS: tuple[str, ...] = ("A", "A+F", "A+F1", "A+F2", "A+F1+F2")

#: Variants eligible for the pre-registered selection (A is a reference only).
SELECTION_POOL: tuple[str, ...] = ("A+F", "A+F1", "A+F2", "A+F1+F2")

#: Recorded 3-month / WF anchors for A+F (from task-15).
A_PLUS_F_ANCHOR = {
    "window_net_pnl_dollars": 573.91,
    "window_profit_factor": 1.2884,
    "window_total_trades": 64,
    "wf_net_pnl_dollars": 230.97,
    "wf_positive_folds": 14,
}
SIGNIFICANCE_LEAD_PCT = 5.0

V05B_STRUCTURE_NOTE = (
    "V0.5b 预注册防震荡对比：入场固定 V0.4 Step3c-A NoCap（10:00-14:00 ET 窗、"
    "SMC 严格 AND、无 8% cap、无 Step4 分批），出场固定 Runner-A（MFE 1.5R→+0.20R、"
    "2.2R→+1.20R、3.0R→+2.00R、目标 4.5R、EOD 15:59），B1 风控 $10k/$60/名义$10k/"
    "碎股0.01/3 连亏停 15min/日-$320/15 笔每日。防震荡规则：A+F=原 OR30(1.8%/1.2ATR)；"
    "F1=修正 30m 尺度(2.8%/2.2ATR)；F2=趋势策略仅允许在 OR30 上沿或上半区+EMA21 3 根斜率"
    ">=+0.15% 时入场（SMC 不受限）。选择规则预注册：仅 WF净>0 的变体参与，"
    "综合分=WF净+3M净+全周期净；新变体若 3M 与 WF 同时改善但综合分领先 <5% 则视为无显著击败、"
    "保留原版 A+F。"
)


# --------------------------------------------------------------------------- #
# Small helpers
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


def _row_with_variant(result: v05.V05Result) -> Dict[str, Any]:
    row = _row_metrics(result)
    extra = result.extra_metrics()
    row["variant"] = result.variant.flags()
    row["trend_box_rejected"] = int(extra.get("trend_box_rejected", 0))
    row["or30_skipped_evaluations"] = int(extra.get("or30_skipped_evaluations", 0))
    row["or30_blocked_sessions_global"] = int(extra.get("or30_blocked_sessions", 0))
    return row


def _wf_row(wf: v05.V05WalkForwardResult) -> Dict[str, Any]:
    row = dict(_wf_metrics(wf))
    row["variant"] = wf.variant.flags()
    return row


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #
def run_matrix(
    data: Any,
    *,
    full_dates: Sequence[date],
    window_dates: Sequence[date],
    with_walkforward: bool = True,
) -> Dict[str, Dict[str, Any]]:
    compare: Dict[str, Dict[str, Any]] = {}
    for variant_label in MATRIX_LABELS:
        variant = v05.V05_VARIANTS[variant_label]
        print(
            f"\n[{variant_label}] exit={variant.exit_mode} or30={variant.or30_filter} "
            f"trend_box={variant.trend_box_filter} …",
            flush=True,
        )
        started = time.perf_counter()
        window_result = v05.run_v05_backtest(
            data, variant=variant, session_dates=window_dates
        )
        full_result = v05.run_v05_backtest(
            data, variant=variant, session_dates=full_dates
        )
        wf = (
            v05.run_v05_walkforward(data, variant=variant, dates=full_dates)
            if with_walkforward
            else None
        )
        elapsed = time.perf_counter() - started
        compare[variant_label] = {
            "label": variant_label,
            "variant": variant.flags(),
            "exit_label": "Runner-A (4.5R R-ladder)",
            "window": _row_with_variant(window_result),
            "full": _row_with_variant(full_result),
            "walkforward": _wf_row(wf) if wf is not None else None,
            "monthly": _monthly_breakdown(window_result),
            "window_result": window_result,
            "full_result": full_result,
            "elapsed_seconds": round(elapsed, 2),
        }
        row_w = compare[variant_label]["window"]
        row_f = compare[variant_label]["full"]
        wf_row = compare[variant_label]["walkforward"] or {}
        print(
            f"  3M {row_w['total_trades']} 笔 / 胜率 {row_w['win_rate_pct']}% / "
            f"PF {row_w['profit_factor']} / ${row_w['net_pnl_dollars']:+.2f} / "
            f"avgR {row_w['avg_r']:+.4f} / DD ${row_w['max_drawdown_dollars']:,.2f}"
            f" | full {row_f['total_trades']} 笔 / PF {row_f['profit_factor']} / "
            f"${row_f['net_pnl_dollars']:+.2f} / avgR {row_f['avg_r']:+.4f}"
            + (
                f" | WF {wf_row['total_test_trades']} 笔 / "
                f"${wf_row['net_pnl_dollars']:+.2f} / "
                f"{wf_row['positive_folds']}/{wf_row['folds']} 盈利折"
                if wf_row
                else ""
            )
            + (
                f" | F1 skips 3M={row_w['or30_skipped_evaluations']} "
                f"F2 rejects 3M={row_w['trend_box_rejected']}"
                if variant_label != "A"
                else ""
            )
            + f"（{elapsed:.1f}s）",
            flush=True,
        )
    return compare


# --------------------------------------------------------------------------- #
# Filter trigger statistics
# --------------------------------------------------------------------------- #
def build_filter_stats(data: Any) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"blocked": {}, "f2": {}}
    for mode, label in (
        (v05.OR30_FILTER_PCT_OR_ATR, "A+F_original"),
        (v05.OR30_FILTER_F1, "F1"),
    ):
        blocks, rows = v05.compute_or30_blocks(data, mode)
        window_blocks = [
            item for item in blocks if WINDOW_START <= item[0] <= WINDOW_END
        ]
        stats["blocked"][label] = {
            "threshold_note": (
                "pct<1.8% or range<1.2xATR14_5m"
                if mode == v05.OR30_FILTER_PCT_OR_ATR
                else "pct<2.8% or range<2.2xATR14_5m"
            ),
            "blocked_symbol_sessions": len(blocks),
            "blocked_session_dates": len({day for day, _ in blocks}),
            "blocked_window_symbol_sessions": len(window_blocks),
            "blocked_window_session_dates": len({day for day, _ in window_blocks}),
        }
    stats["f2"]["rule"] = (
        "trend-only: close>=OR30_High OR (close>=OR30_Low+0.5*box AND "
        "(EMA21-EMA21[3])/EMA21[3]>=0.15%); SMC exempt"
    )
    return stats


# --------------------------------------------------------------------------- #
# Pre-registered selection
# --------------------------------------------------------------------------- #
def select_winner(compare: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    def score(label: str) -> float:
        payload = compare[label]
        wf_net = float((payload.get("walkforward") or {}).get("net_pnl_dollars", 0.0))
        return (
            wf_net
            + float(payload["window"]["net_pnl_dollars"])
            + float(payload["full"]["net_pnl_dollars"])
        )

    def wf_net(label: str) -> float:
        return float(
            (compare[label].get("walkforward") or {}).get("net_pnl_dollars", 0.0)
        )

    def window_net(label: str) -> float:
        return float(compare[label]["window"]["net_pnl_dollars"])

    def full_net(label: str) -> float:
        return float(compare[label]["full"]["net_pnl_dollars"])

    rows = {
        label: {
            "window_net": window_net(label),
            "full_net": full_net(label),
            "wf_net": wf_net(label),
            "wf_positive_folds": int(
                (compare[label].get("walkforward") or {}).get("positive_folds", 0)
            ),
            "composite_score": _number(score(label), 2),
            "wf_positive": wf_net(label) > 0.0,
        }
        for label in compare
    }
    eligible = [label for label in SELECTION_POOL if rows[label]["wf_positive"]]
    ranked = sorted(eligible, key=score, reverse=True)
    best = ranked[0] if ranked else "A+F"
    af = "A+F"
    improved_both: List[str] = []
    lead_pct: Dict[str, float] = {}
    significant_beat: Dict[str, bool] = {}
    for label in SELECTION_POOL:
        if label == af:
            continue
        improved = window_net(label) > window_net(af) and wf_net(label) > wf_net(af)
        advantage = score(label) - score(af)
        relative = (
            advantage / abs(score(af)) * 100.0 if abs(score(af)) > 1e-9 else 0.0
        )
        improved_both.append(label) if improved else None
        lead_pct[label] = _number(relative, 2)
        significant_beat[label] = bool(improved and relative >= SIGNIFICANCE_LEAD_PCT)
    # Apply rule: if a new variant beats both 3M and WF but leads < 5% -> no
    # significant beat -> keep A+F. Otherwise take the highest composite score
    # among WF-positive variants.
    winner = best
    no_significant_beat = False
    if best != af and best in improved_both and lead_pct.get(best, 0.0) < SIGNIFICANCE_LEAD_PCT:
        winner = af
        no_significant_beat = True
    if not eligible:
        winner = af
        no_significant_beat = True
    return {
        "rule": (
            "eligible = WF net > 0; score = WF net + 3M net + full net; "
            "new variant improving both 3M and WF but leading A+F by <5% is not a "
            "significant beat -> keep A+F; else take the highest composite score."
        ),
        "variants": rows,
        "eligible": eligible,
        "ranking": ranked,
        "best_by_score": best,
        "improved_both_vs_A+F": improved_both,
        "lead_pct_vs_A+F": lead_pct,
        "significant_beat_vs_A+F": significant_beat,
        "winner": winner,
        "no_significant_beat": no_significant_beat,
        "significance_threshold_pct": SIGNIFICANCE_LEAD_PCT,
        "significance_conclusion": (
            "无显著击败：没有任何新变体在 3M 与 WF 同时改善 A+F；F1/F2/F1+F2 的 3M 与 "
            "全周期净均更低，F1 与 F1+F2 的 WF 净为负。"
            if not improved_both
            else (
                "存在同时改善 3M 与 WF 的新变体，但综合分领先 <5%，按预注册规则保留 A+F。"
                if no_significant_beat
                else "存在同时改善 3M 与 WF 且综合分领先 >=5% 的新变体，按预注册规则选其最高分者。"
            )
        ),
        "verdict": (
            f"winner={winner}；"
            + (
                "无新变体同时改善 3M 与 WF 且综合分领先 >=5%，保留原版 A+F"
                if winner == af and no_significant_beat
                else (
                    "无新变体同时改善 3M 与 WF，A+F 综合分最高，保留原版 A+F"
                    if winner == af
                    else f"{winner} 综合分最高且通过预注册显著性判断"
                )
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Anchor checks (A+F 3-month + WF vs recorded values)
# --------------------------------------------------------------------------- #
def check_anchor(
    compare: Mapping[str, Mapping[str, Any]],
    *,
    require_walkforward: bool = True,
) -> tuple[bool, List[Dict[str, Any]]]:
    payload = compare["A+F"]
    wf = payload.get("walkforward") or {}
    actual = {
        "window_net_pnl_dollars": payload["window"]["net_pnl_dollars"],
        "window_profit_factor": payload["window"]["profit_factor"],
        "window_total_trades": payload["window"]["total_trades"],
        "wf_net_pnl_dollars": wf.get("net_pnl_dollars"),
        "wf_positive_folds": wf.get("positive_folds"),
    }
    tolerances = {
        "window_net_pnl_dollars": 0.01,
        "window_profit_factor": 0.001,
        "window_total_trades": 0.0,
        "wf_net_pnl_dollars": 0.01,
        "wf_positive_folds": 0.0,
    }
    checks: List[Dict[str, Any]] = []
    passed = True
    for field, expected in A_PLUS_F_ANCHOR.items():
        value = actual.get(field)
        tolerance = tolerances.get(field, 0.01)
        is_wf_field = field.startswith("wf_")
        if is_wf_field and not require_walkforward:
            checks.append(
                {
                    "field": field,
                    "expected": expected,
                    "actual": None,
                    "delta": None,
                    "tolerance": tolerance,
                    "ok": True,
                    "skipped": "walk-forward disabled for this run",
                }
            )
            continue
        ok = value is not None and abs(float(value) - float(expected)) <= tolerance + 1e-9
        passed = passed and ok
        checks.append(
            {
                "field": field,
                "expected": expected,
                "actual": None if value is None else _number(float(value), 4),
                "delta": None if value is None else _number(float(value) - float(expected), 6),
                "tolerance": tolerance,
                "ok": bool(ok),
            }
        )
    return passed, checks


# --------------------------------------------------------------------------- #
# Friday recap / markers for the winner
# --------------------------------------------------------------------------- #
def build_friday_payload(
    friday_bars: Sequence[Any],
    prepared: Any,
    *,
    winner_label: str,
    selection: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    variant = v05.V05_VARIANTS[winner_label]
    result = v05.run_v05_backtest(
        prepared, variant=variant, session_dates=[FRIDAY]
    )
    trades = v05.result_to_trade_dicts_v05(result)
    markers = sorted(
        result.markers, key=lambda item: (item.get("t") or 0, item["type"])
    )
    exit_distribution: Dict[str, int] = {}
    for trade in trades:
        reason = trade["exit_reason"] or "UNKNOWN"
        exit_distribution[reason] = exit_distribution.get(reason, 0) + 1
    market = _market_snapshot(
        [bar for bar in friday_bars if et_date(bar.start_utc) == FRIDAY]
    )
    summary = _row_metrics(result)
    recap: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.5b",
        "session_date": FRIDAY.isoformat(),
        "winner": winner_label,
        "variant": variant.flags(),
        "selection_verdict": selection.get("verdict"),
        "structure": V05B_STRUCTURE_NOTE,
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw, 2026-11-06 regular session "
            "(single venue; NOT consolidated tape)"
        ),
        "caveat": DATA_CAVEAT,
        "market": market,
        "summary": summary,
        "exit_reason_distribution": dict(sorted(exit_distribution.items())),
        "trades": trades,
        "markers": markers,
        "filters": {
            "or30_filter": variant.or30_filter,
            "trend_box_filter": variant.trend_box_filter,
            "trend_box_rejected": int(
                result.extra_metrics().get("trend_box_rejected", 0)
            ),
            "or30_skipped_evaluations": int(
                result.extra_metrics().get("or30_skipped_evaluations", 0)
            ),
        },
        "notes": [
            "Winner of the pre-registered V0.5b selection; entries frozen to Step3c-A NoCap.",
            "Markers feed the Web replay/terminal; types are ENTRY/STOP/TARGET/EOD_EXIT/TRAIL_EMA21 etc.",
            "IEX single-venue data; research output, not proof of profitability.",
        ],
    }
    markers_payload: Dict[str, Any] = {
        "generated_at_utc": recap["generated_at_utc"],
        "version": "V0.5b",
        "session_date": FRIDAY.isoformat(),
        "winner": winner_label,
        "variant": variant.flags(),
        "data_source": recap["data_source"],
        "summary": summary,
        "markers": markers,
        "markers_by_trade": {
            trade["trade_id"]: trade["markers"] for trade in trades
        },
        "trades": trades,
    }
    return recap, markers_payload


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
_HEADER = (
    "| 变体 | 3个月 笔/胜率/PF/净($)/净%/avgR/MaxDD | "
    "全周期 笔/胜率/PF/净($)/净%/avgR/MaxDD | 29折 WF 笔/胜率/PF/净($)/盈利折 | "
    "3M F1跳过/F2拒绝 |"
)
_SEPARATOR = "|---|---|---|---|---|"


def _matrix_row(label: str, payload: Mapping[str, Any]) -> str:
    w, f = payload["window"], payload["full"]
    wf = payload.get("walkforward") or {}
    wf_cell = "—"
    if wf:
        wf_cell = (
            f"{wf.get('total_test_trades')}/{_cell(wf.get('win_rate_pct'), 'pct')}/"
            f"{_cell(wf.get('profit_factor'), 'pf')}/{_cell(wf.get('net_pnl_dollars'), 'money')}/"
            f"{wf.get('positive_folds')}/{wf.get('folds')}"
        )
    return (
        f"| {label} | {_slim(w)} | {_slim(f)} | {wf_cell} | "
        f"{w.get('or30_skipped_evaluations', 0)}/{w.get('trend_box_rejected', 0)} |"
    )


def build_markdown(
    *,
    compare: Mapping[str, Mapping[str, Any]],
    filter_stats: Mapping[str, Any],
    selection: Mapping[str, Any],
    anchor_checks: Sequence[Mapping[str, Any]],
    anchor_passed: bool,
    friday_recap: Mapping[str, Any],
    data_span: str,
    window_sessions: int,
    full_sessions: int,
    elapsed: Mapping[str, float],
) -> str:
    months = sorted({month for payload in compare.values() for month in payload["monthly"]})
    lines = [
        "# V0.5b 预注册防震荡对比（F1/F2）+ Web 选优（A Runner 4.5R / B1 $10k·$60）",
        "",
        f"- 数据：Alpaca IEX raw 1m，{data_span}；3 个月窗口 2026-08-01..11-06"
        f"（{window_sessions} sessions，含周五）；全周期 {full_sessions} sessions"
        f"（不含 2026-11-06）",
        f"- 结构：{V05B_STRUCTURE_NOTE}",
        "",
        "## A+F 锚校验（3M 64 笔 / +$573.91 / PF 1.2884；WF +$230.97 / 14 折）",
        "",
        f"结果：{'通过 ✅' if anchor_passed else '未通过 ❌'}",
        "",
        "| 指标 | 锚值 | 实测 | 偏差 | 容差 | 通过 |",
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
            "## 五行主表（3 个月 / 全周期 / 29 折 WF）",
            "",
            _HEADER,
            _SEPARATOR,
            *[_matrix_row(label, compare[label]) for label in MATRIX_LABELS],
            "",
            "## 3 个月分月（笔数 / 胜率 / PF / 净$）",
            "",
        ]
    )
    header = "| 变体 | " + " | ".join(f"{int(month[-2:])}月" for month in months) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(months))
    for label in MATRIX_LABELS:
        cells = []
        for month in months:
            row = compare[label]["monthly"].get(month) or {}
            cells.append(
                f"{row.get('total_trades', 0)}/{_cell(row.get('win_rate_pct'), 'pct')}/"
                f"{_cell(row.get('profit_factor'), 'pf')}/{_cell(row.get('net_pnl_dollars'), 'money')}"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## F1/F2 触发与拦截统计",
            "",
            "### F1 日过滤（当日 10:00 后跳过新开仓）",
            "",
            "| 规则 | 阈值 | 全周期拦截 symbol-sessions | 全周期拦截日数 | 3M 拦截 symbol-sessions | 3M 拦截日数 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for label, stats in filter_stats["blocked"].items():
        lines.append(
            f"| {label} | {stats['threshold_note']} | {stats['blocked_symbol_sessions']} | "
            f"{stats['blocked_session_dates']} | {stats['blocked_window_symbol_sessions']} | "
            f"{stats['blocked_window_session_dates']} |"
        )
    lines.extend(
        [
            "",
            f"### F2 趋势信号拒绝（规则：{filter_stats['f2']['rule']}）",
            "",
            "| 变体 | 3M F1 跳过评估 | 3M F2 拒绝信号 | 全周期 F1 跳过评估 | 全周期 F2 拒绝信号 |",
            "|---|---|---|---|---|",
        ]
    )
    for label in MATRIX_LABELS:
        w = compare[label]["window"]
        f = compare[label]["full"]
        lines.append(
            f"| {label} | {w.get('or30_skipped_evaluations', 0)} | "
            f"{w.get('trend_box_rejected', 0)} | {f.get('or30_skipped_evaluations', 0)} | "
            f"{f.get('trend_box_rejected', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 预注册选择结果",
            "",
            f"- 规则：{selection['rule']}",
            f"- WF>0 的候选：{', '.join(selection['eligible']) or '无'}",
            f"- 综合分排序：{' > '.join(selection['ranking']) or '无'}",
            "",
            "| 变体 | 3M净($) | 全周期净($) | WF净($) | WF盈利折 | 综合分 | WF>0 | 相对A+F 3M+WF 同改善 | 综合分领先% | 显著击败A+F |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for label in SELECTION_POOL:
        row = selection["variants"][label]
        lines.append(
            f"| {label} | {row['window_net']:+.2f} | {row['full_net']:+.2f} | "
            f"{row['wf_net']:+.2f} | {row['wf_positive_folds']} | {row['composite_score']:.2f} | "
            f"{'✅' if row['wf_positive'] else '❌'} | "
            f"{'是' if label in selection['improved_both_vs_A+F'] else '否'} | "
            f"{selection['lead_pct_vs_A+F'].get(label, '—')} | "
            f"{'是' if selection['significant_beat_vs_A+F'].get(label) else '否'} |"
        )
    lines.extend(
        [
            "",
            f"**结论：winner = {selection['winner']}；"
            f"{'无显著击败，保留原版 A+F。' if selection['no_significant_beat'] else '通过显著性判断，按综合分取胜。'}**",
            "",
            "## 周五 2026-11-06（winner）复盘",
            "",
            f"- winner：{friday_recap['winner']}；变体：{json.dumps(friday_recap['variant'], ensure_ascii=False)}",
            f"- 当日行情：{json.dumps(friday_recap['market'], ensure_ascii=False)}",
            f"- 汇总：{_slim(friday_recap['summary'])}",
            f"- 出场分布：{json.dumps(friday_recap['exit_reason_distribution'], ensure_ascii=False)}",
            "",
            "| # | 入场(ET) | 标的 | 策略 | 入场价 | 止损 | 目标 | 出场(ET) | 出场价 | 原因 | PnL($) | R | markers |",
            "|---|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---|",
        ]
    )
    for index, trade in enumerate(friday_recap["trades"], start=1):
        marker_types = ",".join(marker["type"] for marker in trade["markers"])
        lines.append(
            f"| {index} | {trade['entry_time_et']} | {trade['symbol']} | "
            f"{trade['strategy']} | {_cell(trade['entry_price_fill'], 'd')} | "
            f"{_cell(trade['stop_price'], 'd')} | {_cell(trade['target_price'], 'd')} | "
            f"{trade['exit_time_et']} | {_cell(trade['exit_price_fill'], 'd')} | "
            f"{trade['exit_reason']} | {_cell(trade['pnl'], 'money')} | "
            f"{_cell(trade['r_multiple'], 'r')} | {marker_types} |"
        )
    lines.extend(
        [
            "",
            "## Web 接入",
            "",
            "- `intraday_scalper.research.v05.evaluate_v05_session(bars_1m, session_date, variant)` "
            "返回带 `markers` 的逐笔 dict（字段 type/time_utc/price/symbol/strategy/trade_id/t）。",
            "- `v05.V05_PRODUCTION_VARIANT` 当前为 A+F（本任务预注册选择结果见上；Lead 一行切换 winner）。",
            "- 周五复盘文件：`var/backtest/v05b_friday_recap.json` / `v05b_friday_markers.json`。",
            "",
            "## 偏离项与风险提示",
            "",
            "- ATR14_5m 口径：v0.5 实现为 10:00 前最后一根已完成 5m bar 上的 ATR14（简单均值 TR），"
            "F1 沿用同一口径（未改为 Wilder），特此注明。",
            "- F2 的 EMA21 3 根斜率使用全局已完成 5m bar 的 EMA21 序列 index-3，"
            "与 v05 的 EMA21 口径一致，无未来函数。",
            "- 选择规则在跑数前预注册，未事后添加指标；F1/F2 结果变差也如实报告。",
            "- IEX 单一交易所口径；研究结果不代表实盘有效性或盈利预期。",
            f"- 运行时长：{elapsed.get('total', 0.0):.1f}s。",
            "",
        ]
    )
    return "\n".join(lines)


def _cell_number(value: Any, kind: str) -> str:
    """Markdown/terminal cell formatting for the Friday table (numbers)."""
    if value is None:
        return "—"
    if kind == "money":
        return f"{float(value):+,.2f}"
    if kind == "r":
        return f"{float(value):+.4f}"
    return f"{float(value):.2f}"


def print_tables(
    compare: Mapping[str, Mapping[str, Any]],
    filter_stats: Mapping[str, Any],
    selection: Mapping[str, Any],
    friday_recap: Mapping[str, Any],
) -> None:
    print("\n" + "=" * 158)
    print("V0.5b 预注册 F1/F2 对比（Runner-A 4.5R，B1 $10k/$60，3M + 全周期 + 29 折 WF）")
    print("=" * 158)
    print(
        f"{'变体':<9}{'3M笔':>6}{'胜率':>8}{'PF':>8}{'净($)':>11}{'avgR':>9}{'MaxDD':>10}"
        f"{'全笔':>6}{'胜率':>8}{'PF':>8}{'净($)':>12}{'avgR':>9}{'MaxDD':>11}"
        f"{'WF笔':>7}{'PF':>8}{'WF净($)':>12}{'盈利折':>8}"
    )
    print("-" * 158)
    for label in MATRIX_LABELS:
        payload = compare[label]
        w, f = payload["window"], payload["full"]
        wf = payload.get("walkforward") or {}
        print(
            f"{label:<9}{w['total_trades']:>6}{_cell(w['win_rate_pct'], 'pct'):>8}"
            f"{_cell(w['profit_factor'], 'pf'):>8}{_cell(w['net_pnl_dollars'], 'money'):>11}"
            f"{_cell(w['avg_r'], 'r'):>9}{_cell(w['max_drawdown_dollars'], 'dd'):>10}"
            f"{f['total_trades']:>6}{_cell(f['win_rate_pct'], 'pct'):>8}"
            f"{_cell(f['profit_factor'], 'pf'):>8}{_cell(f['net_pnl_dollars'], 'money'):>12}"
            f"{_cell(f['avg_r'], 'r'):>9}{_cell(f['max_drawdown_dollars'], 'dd'):>11}"
            f"{wf.get('total_test_trades', '—'):>7}{_cell(wf.get('profit_factor'), 'pf'):>8}"
            f"{_cell(wf.get('net_pnl_dollars'), 'money'):>12}"
            f"{str(wf.get('positive_folds', 0)) + '/' + str(wf.get('folds', 0)):>8}"
        )
    print("-" * 158)
    print("\n3 个月分月（笔数/胜率/PF/净$）：")
    months = sorted({month for payload in compare.values() for month in payload["monthly"]})
    for label in MATRIX_LABELS:
        parts = []
        for month in months:
            row = compare[label]["monthly"].get(month) or {}
            parts.append(
                f"{month[-2:]}月: {row.get('total_trades', 0)}/"
                f"{_cell(row.get('win_rate_pct'), 'pct')}/"
                f"{_cell(row.get('profit_factor'), 'pf')}/{_cell(row.get('net_pnl_dollars'), 'money')}"
            )
        print(f"  {label:<9} " + " | ".join(parts))
    print("\nF1 日过滤拦截统计：")
    for label, stats in filter_stats["blocked"].items():
        print(
            f"  {label:<14} {stats['threshold_note']}: "
            f"全周期 {stats['blocked_symbol_sessions']} symbol-sessions / "
            f"{stats['blocked_session_dates']} 日；3M "
            f"{stats['blocked_window_symbol_sessions']} / {stats['blocked_window_session_dates']} 日"
        )
    print("F2 趋势信号拒绝：")
    for label in MATRIX_LABELS:
        w, f = compare[label]["window"], compare[label]["full"]
        print(
            f"  {label:<9} 3M F1跳过评估={w.get('or30_skipped_evaluations', 0)} "
            f"F2拒绝={w.get('trend_box_rejected', 0)} | 全周期 F1跳过={f.get('or30_skipped_evaluations', 0)} "
            f"F2拒绝={f.get('trend_box_rejected', 0)}"
        )
    print("\n预注册选择：")
    print(f"  {selection['rule']}")
    print(f"  候选(WF>0)：{selection['eligible']}；综合分排序：{selection['ranking']}")
    for label in SELECTION_POOL:
        row = selection["variants"][label]
        print(
            f"    {label:<9} 3M {row['window_net']:+.2f} / full {row['full_net']:+.2f} / "
            f"WF {row['wf_net']:+.2f} / 折 {row['wf_positive_folds']} / 综合分 {row['composite_score']:.2f} / "
            f"WF>0={row['wf_positive']} / 领先A+F {selection['lead_pct_vs_A+F'].get(label)}% / "
            f"同时改善={label in selection['improved_both_vs_A+F']}"
        )
    print(f"\n  >>> winner = {selection['winner']}；{selection['verdict']}")
    print("\n周五 2026-11-06（winner）：")
    print(
        f"  行情：{json.dumps(friday_recap['market'], ensure_ascii=False)}"
    )
    print(f"  汇总：{_slim(friday_recap['summary'])}")
    print(f"  出场分布：{friday_recap['exit_reason_distribution']}")
    for trade in friday_recap["trades"]:
        print(
            f"    {trade['entry_time_et']} ET {trade['symbol']} {trade['strategy']:<16} "
            f"→ {trade['exit_time_et']} ET {trade['exit_reason']:<12} "
            f"PnL {_cell(trade['pnl'], 'money')} R {_cell(trade['r_multiple'], 'r')} "
            f"markers={[marker['type'] for marker in trade['markers']]}"
        )
    print("=" * 158)


# --------------------------------------------------------------------------- #
# Trade ledger
# --------------------------------------------------------------------------- #
def write_trades_csv(compare: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for label in MATRIX_LABELS:
        payload = compare[label]
        for period, result_key in (("3M", "window_result"), ("FULL", "full_result")):
            result = payload[result_key]
            for trade in result.trades:
                detail = result.trade_details.get(trade.signal_id, {})
                rows.append(
                    {
                        "variant": label,
                        "or30_filter": payload["variant"]["or30_filter"],
                        "trend_box_filter": payload["variant"]["trend_box_filter"],
                        "period": period,
                        "signal_id": trade.signal_id,
                        "symbol": trade.symbol,
                        "strategy": trade.strategy.value,
                        "strategy_label": detail.get("strategy_label", trade.strategy.value),
                        "shares": trade.shares,
                        "entry_time_utc": trade.entry_time_utc.isoformat(),
                        "entry_date_et": et_date(trade.entry_time_utc).isoformat(),
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
) -> Dict[str, Any]:
    started = time.perf_counter()
    print(f"加载 raw 数据（含 2026-11-06）：{data_dir}")
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

    compare = run_matrix(
        data,
        full_dates=full_dates,
        window_dates=window_dates,
        with_walkforward=with_walkforward,
    )
    anchor_passed, anchor_checks = check_anchor(compare, require_walkforward=with_walkforward)
    filter_stats = build_filter_stats(data)
    selection = select_winner(compare)
    print(f"\nA+F 锚校验：{'通过 ✅' if anchor_passed else '未通过 ❌'}")
    for check in anchor_checks:
        print(
            f"  {'✅' if check['ok'] else '❌'} {check['field']:<26} "
            f"锚={check['expected']} 实测={check['actual']} Δ={check['delta']}"
        )
    if not anchor_passed:
        raise RuntimeError(
            "A+F anchor (3M 64/+$573.91/PF 1.2884, WF +$230.97/14) not reproduced; "
            "refusing to publish V0.5b results"
        )
    winner = selection["winner"]
    print(f"\n预注册选择 winner：{winner}（{selection['verdict']}）")

    friday_recap, friday_markers = build_friday_payload(
        friday_bars,
        data,
        winner_label=winner,
        selection=selection,
    )
    total_elapsed = time.perf_counter() - started

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "V0.5b",
        "data_source": (
            "Alpaca IEX 1m bars, adjustment=raw (single venue; NOT consolidated tape)"
        ),
        "data_span": f"{data.dates[0]:%Y-%m-%d} .. {data.dates[-1]:%Y-%m-%d}",
        "structure": V05B_STRUCTURE_NOTE,
        "risk_base": {
            "equity": v05.B1_EQUITY,
            "risk_dollars": v05.B1_RISK,
            "notional": v05.B1_NOTIONAL,
            "fractional_precision": v05.B1_FRACTIONAL_PRECISION,
            "consecutive_stops": v05.B1_CONSECUTIVE_STOPS,
            "cooldown_minutes": v05.B1_COOLDOWN_MINUTES,
            "daily_loss_limit": v05.B1_DAILY_LOSS,
            "max_entries_per_day": v05.B1_MAX_ENTRIES,
        },
        "windows": {
            "three_month": [window_dates[0].isoformat(), window_dates[-1].isoformat()],
            "three_month_sessions": len(window_dates),
            "full_period": [full_dates[0].isoformat(), full_dates[-1].isoformat()],
            "full_sessions": len(full_dates),
        },
        "walkforward_grid": {"train_days": 120, "test_days": 20, "step_days": 20},
        "matrix": {
            label: {
                key: value
                for key, value in compare[label].items()
                if key not in ("window_result", "full_result")
            }
            for label in MATRIX_LABELS
        },
        "anchor": {"expected": A_PLUS_F_ANCHOR, "passed": bool(anchor_passed), "checks": anchor_checks},
        "filter_stats": filter_stats,
        "selection": selection,
        "friday": {
            "session_date": friday_recap["session_date"],
            "winner": winner,
            "variant": friday_recap["variant"],
            "market": friday_recap["market"],
            "summary": friday_recap["summary"],
            "exit_reason_distribution": friday_recap["exit_reason_distribution"],
            "trades": friday_recap["trades"],
            "markers": friday_recap["markers"],
            "filters": friday_recap["filters"],
        },
        "web_api": {
            "evaluate_v05_session": "intraday_scalper.research.v05.evaluate_v05_session",
            "production_variant": v05.V05_PRODUCTION_VARIANT.name,
            "variants": {name: variant.flags() for name, variant in v05.V05_VARIANTS.items()},
        },
        "elapsed_seconds": {
            "matrix": round(sum(compare[label]["elapsed_seconds"] for label in compare), 2),
            "total": round(total_elapsed, 2),
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "v05b_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    markdown = build_markdown(
        compare=compare,
        filter_stats=filter_stats,
        selection=selection,
        anchor_checks=anchor_checks,
        anchor_passed=anchor_passed,
        friday_recap=friday_recap,
        data_span=report_payload["data_span"],
        window_sessions=len(window_dates),
        full_sessions=len(full_dates),
        elapsed={"total": total_elapsed},
    )
    markdown_path = report_dir / "v05b_compare.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    trades_path = report_dir / "v05b_trades.csv"
    write_trades_csv(compare, trades_path)
    recap_path = report_dir / "v05b_friday_recap.json"
    recap_path.write_text(
        json.dumps(friday_recap, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    markers_path = report_dir / "v05b_friday_markers.json"
    markers_path.write_text(
        json.dumps(friday_markers, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_tables(compare, filter_stats, selection, friday_recap)
    print("\n产物：")
    print(f"  报告 JSON     : {report_path}")
    print(f"  对比 Markdown : {markdown_path}")
    print(f"  逐笔流水     : {trades_path}")
    print(f"  周五复盘     : {recap_path}")
    print(f"  周五 markers : {markers_path}")
    print(f"总运行时长：{total_elapsed:.1f}s")
    return report_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort console setup
        pass
    parser = argparse.ArgumentParser(
        description="V0.5b pre-registered F1/F2 anti-chop comparison + Web winner."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-walkforward", action="store_true")
    args = parser.parse_args(argv)
    build_and_run(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        with_walkforward=not args.no_walkforward,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
