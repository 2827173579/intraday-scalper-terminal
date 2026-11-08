"""Canonical raw-data baseline comparison: V0.1 vs V0.2 vs V0.3 on *unadjusted* bars.

Why this exists
---------------
The V0.1/V0.2/V0.3 history was built on Alpaca's ``adjustment="all"`` dataset,
which (a) inflates SOXS prices through reverse splits and (b) divides bar
volume by the split factor, rounding low-volume minutes to ``v=0`` — those
bars were dropped, so the adjusted SOXS sample is 16,631 bars short.  The §12
cross-check showed the adjusted V0.3 "+$90" was partly that coverage artifact.

This tool re-establishes a **single canonical baseline on raw (unadjusted)
data** so all three versions are measured on identical, complete bars:

* V0.1 — production 1m engine (``backtest.engine.run_backtest``) on raw bars;
* V0.2 — the 5m research replay (``research.v02``) on raw bars;
* V0.3 — read from the already-computed raw-full ablation in
  ``var/backtest/v03_raw_crosscheck.json`` (step 4), reused to avoid re-running.

Outputs
-------
* ``var/backtest/raw_canonical_report.json``
* ``var/backtest/raw_canonical_comparison.md``

Usage::

    python -m intraday_scalper.tools.run_raw_canonical
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

from intraday_scalper.backtest.engine import run_backtest
from intraday_scalper.core import TradeRecord, et_date, load_config
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v02
from intraday_scalper.strategies.registry import build_strategies
from intraday_scalper.tools.run_v03_raw_crosscheck import load_raw_bars

REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"
YEAR_2024 = date(2024, 1, 1)
YEAR_2025 = date(2025, 1, 1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pf(gross_profit: float, gross_loss: float) -> float | None:
    if gross_loss == 0:
        return None
    return round(gross_profit / gross_loss, 4)


def segment_stats(trades: Sequence[TradeRecord], initial_equity: float) -> Dict[str, Any]:
    closed = [t for t in trades if t.is_closed]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    net = sum(t.pnl for t in closed)
    equity = initial_equity
    peak = initial_equity
    max_dd = 0.0
    for trade in sorted(closed, key=lambda t: t.exit_time_utc):
        equity += trade.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed) * 100.0, 2) if closed else 0.0,
        "profit_factor": _pf(gross_profit, gross_loss),
        "net_pnl_dollars": round(net, 2),
        "avg_r": round(sum(t.r_multiple for t in closed) / len(closed), 4) if closed else 0.0,
        "max_drawdown_dollars": round(max_dd, 2),
    }


def metrics_summary(metrics: Any) -> Dict[str, Any]:
    return {
        "total_trades": int(metrics.total_trades),
        "wins": int(metrics.wins),
        "losses": int(metrics.losses),
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": None if metrics.profit_factor is None else round(metrics.profit_factor, 4),
        "net_pnl_dollars": round(metrics.net_pnl_dollars, 2),
        "avg_r": round(metrics.avg_r, 4),
        "max_drawdown_dollars": round(metrics.max_drawdown_dollars, 2),
        "trades_per_day": round(metrics.trades_per_day, 2),
        "total_slippage_dollars": round(metrics.total_slippage_dollars, 2),
    }


def split_trades(trades: Sequence[TradeRecord]) -> Dict[str, List[TradeRecord]]:
    groups: Dict[str, List[TradeRecord]] = {"train_2024": [], "holdout": []}
    for trade in trades:
        if not trade.is_closed:
            continue
        day = et_date(trade.entry_time_utc)
        if day.year == 2024:
            groups["train_2024"].append(trade)
        if day >= YEAR_2025:
            groups["holdout"].append(trade)
    return groups


def time_of_day(trades: Sequence[TradeRecord]) -> Dict[str, Any]:
    from intraday_scalper.core import to_et

    buckets = {
        "09:30-10:00": [0, 0.0],
        "10:00-11:00": [0, 0.0],
        "11:00-13:30": [0, 0.0],
        "13:30-15:30": [0, 0.0],
    }
    for trade in trades:
        if not trade.is_closed:
            continue
        et = to_et(trade.entry_time_utc)
        minutes = et.hour * 60 + et.minute
        if minutes < 10 * 60:
            key = "09:30-10:00"
        elif minutes < 11 * 60:
            key = "10:00-11:00"
        elif minutes < 13 * 60 + 30:
            key = "11:00-13:30"
        else:
            key = "13:30-15:30"
        buckets[key][0] += 1
        buckets[key][1] += trade.pnl
    return {
        key: {"trades": int(count), "net_pnl_dollars": round(net, 2)}
        for key, (count, net) in buckets.items()
    }


def exit_reason_counts(result: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for detail in (getattr(result, "trade_details", None) or {}).values():
        reason = str(detail.get("exit_reason", "UNKNOWN"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# V0.1 (production 1m engine) on raw bars
# --------------------------------------------------------------------------- #
def run_v01_raw(bars, config, initial_equity: float) -> Dict[str, Any]:
    print("[V0.1] 运行 production 1m engine on raw bars …（约 5-6 分钟）", flush=True)
    started = time.perf_counter()
    strategies = build_strategies(config.strategies)
    result = run_backtest(bars, config, strategies)
    elapsed = round(time.perf_counter() - started, 2)
    groups = split_trades(result.trades)
    payload = {
        "full": {
            **metrics_summary(result.metrics),
            "risk_block_counts": dict(result.risk_block_counts),
            "ambiguous_bars": int(getattr(result, "ambiguous_bars", 0) or 0),
            "exit_reasons": exit_reason_counts(result),
            "time_of_day": time_of_day(result.trades),
            "per_strategy": {
                key: {
                    "trades": value.trades,
                    "win_rate_pct": round(value.win_rate * 100.0, 2),
                    "profit_factor": None if value.profit_factor is None else round(value.profit_factor, 4),
                    "net_pnl_dollars": round(value.net_pnl_dollars, 2),
                    "avg_r": round(value.avg_r, 4),
                }
                for key, value in result.metrics.per_strategy.items()
            },
            "per_symbol": {
                key: {
                    "trades": value.trades,
                    "win_rate_pct": round(value.win_rate * 100.0, 2),
                    "net_pnl_dollars": round(value.net_pnl_dollars, 2),
                    "avg_r": round(value.avg_r, 4),
                }
                for key, value in result.metrics.per_symbol.items()
            },
        },
        "train_2024": segment_stats(groups["train_2024"], initial_equity),
        "holdout": segment_stats(groups["holdout"], initial_equity),
        "elapsed_seconds": elapsed,
        "bars_processed": int(getattr(result, "bars_processed", 0) or 0),
    }
    print(
        f"[V0.1] done: {payload['full']['total_trades']} 笔 / PF {payload['full']['profit_factor']} / "
        f"净 {payload['full']['net_pnl_dollars']:+.2f}（{elapsed:.0f}s）"
    )
    return payload


# --------------------------------------------------------------------------- #
# V0.2 (5m research replay) on raw bars
# --------------------------------------------------------------------------- #
def run_v02_raw(prepared, session_dates: Sequence[date], initial_equity: float) -> Dict[str, Any]:
    dates_2024 = [d for d in session_dates if d.year == 2024]
    dates_holdout = [d for d in session_dates if d >= YEAR_2025]
    print("[V0.2] 运行 5m replay on raw bars …", flush=True)
    started = time.perf_counter()
    full = v02.run_v02_backtest(prepared)
    seg_2024 = v02.run_v02_backtest(prepared, session_dates=dates_2024)
    holdout = v02.run_v02_backtest(prepared, session_dates=dates_holdout)
    wf = v02.run_v02_walkforward(prepared)
    elapsed = round(time.perf_counter() - started, 2)
    payload = {
        "full": metrics_summary(full.metrics),
        "train_2024": metrics_summary(seg_2024.metrics),
        "holdout": metrics_summary(holdout.metrics),
        "walkforward": {
            **metrics_summary(wf.aggregate_test_metrics),
            "positive_folds": int(wf.summary.get("positive_folds", 0)),
            "folds": int(wf.summary.get("folds", len(wf.folds))),
        },
        "elapsed_seconds": elapsed,
    }
    print(
        f"[V0.2] done: {payload['full']['total_trades']} 笔 / PF {payload['full']['profit_factor']} / "
        f"净 {payload['full']['net_pnl_dollars']:+.2f}（{elapsed:.1f}s）"
    )
    return payload


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _load_v03_raw_step4() -> Dict[str, Any]:
    path = REPORT_DIR / "v03_raw_crosscheck.json"
    if not path.exists():
        raise FileNotFoundError(f"missing raw crosscheck report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    step4 = next(s for s in payload["steps"]["raw_full"] if int(s["step"]) == 4)
    return {
        "full": {**step4["full"], "breakeven_armed_trades": step4["breakeven_armed_trades"],
                 "shield_stop_exits": step4["shield_stop_exits"]},
        "train_2024": step4["train_2024"],
        "holdout": step4["holdout"],
        "walkforward": step4["walkforward"],
        "elapsed_seconds": step4.get("elapsed_seconds"),
    }


def build_markdown(canonical: Dict[str, Any]) -> str:
    rows = []
    for label, section in (
        ("V0.1（production 1m engine）", canonical["v01"]),
        ("V0.2（5m research replay）", canonical["v02"]),
        ("V0.3（5m + 移除ORB + 14:00截止 + 保本盾）", canonical["v03"]),
    ):
        full = section["full"]
        seg24 = section["train_2024"]
        hold = section["holdout"]
        wf = section.get("walkforward")
        wf_text = "—"
        if wf:
            wf_text = (
                f"{wf['total_trades']} / {wf['win_rate_pct']:.2f}% / "
                f"{'—' if wf['profit_factor'] is None else f'{wf['profit_factor']:.3f}'} / "
                f"${wf['net_pnl_dollars']:+.2f} / {wf['positive_folds']}/{wf['folds']} 折"
                if isinstance(wf, dict)
                else "—"
            )
        rows.append(
            "| {label} | {t} | {wr:.2f}% | {pf} | {net:+.2f} | {avg_r:+.3f} | {dd:.2f} | "
            "{t24} / {pf24} / {net24:+.2f} | {hold_t} / {hold_pf} / {hold_net:+.2f} | {wf} |".format(
                label=label,
                t=full["total_trades"],
                wr=full["win_rate_pct"],
                pf="—" if full["profit_factor"] is None else f"{full['profit_factor']:.3f}",
                net=full["net_pnl_dollars"],
                avg_r=full["avg_r"],
                dd=full["max_drawdown_dollars"],
                t24=seg24["total_trades"],
                pf24="—" if seg24["profit_factor"] is None else f"{seg24['profit_factor']:.3f}",
                net24=seg24["net_pnl_dollars"],
                hold_t=hold["total_trades"],
                hold_pf="—" if hold["profit_factor"] is None else f"{hold['profit_factor']:.3f}",
                hold_net=hold["net_pnl_dollars"],
                wf=wf_text,
            )
        )
    lines = [
        "# raw（未复权）canonical 基线：V0.1 / V0.2 / V0.3（2026-11-06）",
        "",
        "- 数据集：`adjustment=raw` IEX 1m bars（483,751 根，2024-01-02..2026-11-05）；"
        "与 adjusted 相比多出 16,631 根 SOXS 低成交 bar（见 §12）。",
        "- 所有版本共用同一批 raw bars、同一 $3,000 初始权益、同一 29 折 WF（120/20/20）纪律；"
        "策略/结构参数均为已冻结的先验常量（未调参）。",
        "- 口径：Alpaca IEX 单一交易所，非 consolidated tape；本表为研究/审计基线，不构成投资建议。",
        "",
        "## 统一基线对比（raw）",
        "",
        "| 版本 | 全周期笔数 | 胜率 | PF | 全周期净($) | 平均R | MaxDD($) | 2024 笔/PF/净($) | Holdout 笔/PF/净($) | WF 笔/胜率/PF/净($)/盈利折 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        *rows,
        "",
        "## 结论",
        "",
        "- raw canonical 下：V0.1 最差、V0.2 结构降噪后改善、V0.3 进一步改善（与 §12 的 raw_full 消融一致）；",
        "- 但 V0.3 在 raw 上仍非稳定正期望，2024 段正收益未在 2025-2026 holdout 复现；",
        "- 此前文档中基于 adjusted 数据的 V0.1/V0.2/V0.3 数字仅作历史审计轨迹保留，**口径以本表为准**。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    started = time.perf_counter()
    config = load_config()
    initial_equity = float(config.risk.account_equity)
    bars = load_raw_bars()
    session_dates = sorted({et_date(bar.start_utc) for bar in bars})

    v01 = run_v01_raw(bars, config, initial_equity)
    prepared = v02.prepare_v02_data(bars)
    v02_payload = run_v02_raw(prepared, session_dates, initial_equity)
    v03_payload = _load_v03_raw_step4()

    canonical = {
        "version": "raw-canonical-baselines",
        "generated_at_utc": _now(),
        "dataset": "Alpaca IEX 1m bars, adjustment=raw, 2024-01-02..2026-11-05",
        "bars": len(bars),
        "sessions": len(session_dates),
        "initial_equity": initial_equity,
        "v01": v01,
        "v02": v02_payload,
        "v03": v03_payload,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "raw_canonical_report.json"
    json_path.write_text(json.dumps(canonical, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path = REPORT_DIR / "raw_canonical_comparison.md"
    md_path.write_text(build_markdown(canonical), encoding="utf-8")
    print("\n=== raw canonical baseline ===")
    for label, section in (("V0.1", v01), ("V0.2", v02_payload), ("V0.3", v03_payload)):
        full = section["full"]
        wf = section.get("walkforward") or {}
        print(
            f"{label}: full {full['total_trades']} 笔 / PF {full['profit_factor']} / "
            f"净 {full['net_pnl_dollars']:+.2f} | WF "
            f"{wf.get('net_pnl_dollars', float('nan')):+.2f} / {wf.get('positive_folds', '—')} 折"
        )
    print(f"\nJSON: {json_path}\nMD  : {md_path}\n总耗时: {canonical['elapsed_seconds']:.1f}s")
    return 0


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))