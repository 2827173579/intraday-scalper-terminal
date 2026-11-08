"""V0.3 raw-price cross-check (adjusted ``all`` vs unadjusted ``raw`` bars).

Motivation
----------
The adjusted (``adjustment="all"``) IEX dataset back-fills SOXS reverse splits,
inflating 2024 prices to ~$10,900/share on the first sessions (raw was ~$6), which
is what caused the 1,048 ``ZERO_SHARES`` rejections before V0.3's fractional-share
fix.  Worse, Alpaca's adjusted feed **also divides the bar volume by the split
factor**, so low-volume IEX minutes round to ``v=0``; our downloader dropped those
bars, which is why the adjusted SOXS dataset has 16,631 fewer bars than raw
(missing mostly 2024 pre-split minutes).

This tool therefore re-runs the **same five cumulative ablation steps** on:

* ``raw_full``   — every ``adjustment="raw"`` 1m bar (the more faithful dataset
                   for an intraday system: real Firstrade prices, no volume
                   rounding loss, no cross-session corporate actions);
* ``raw_common`` — raw bars restricted to the timestamps present in the adjusted
                   dataset (isolates the adjustment effect from the coverage
                   difference);

and compares each step with the adjusted reference in
``var/backtest/v03_report.json``.  Variants and constants are imported unchanged
from :mod:`intraday_scalper.research.v03` (``a-priori, not tuned``), and the same
29-fold walk-forward discipline is used.  This is a data-integrity verification,
not a parameter search.

Outputs
-------
* ``var/backtest/v03_raw_crosscheck.json`` — machine-readable comparison
* ``var/backtest/v03_raw_ablation.md`` — markdown table for REVIEW_PACK

Usage::

    python -m intraday_scalper.tools.run_v03_raw_crosscheck
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from intraday_scalper.core import Bar, et_date
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v02, v03

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"
RAW_GLOB = "*_1m_2024_2026_raw.parquet"
ADJ_GLOB = "*_1m_2024_2026.parquet"
V01_2025 = date(2025, 1, 1)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _frames_to_bars(frames: Sequence[pd.DataFrame]) -> List[Bar]:
    frame = pd.concat(frames, ignore_index=True)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["end_utc"] = pd.to_datetime(frame["end_utc"], utc=True)
    frame = frame.drop_duplicates(subset=["symbol", "start_utc"]).sort_values(
        ["start_utc", "symbol"], kind="mergesort"
    )
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


def load_raw_bars(data_dir: Path = DATA_DIR) -> List[Bar]:
    files = sorted(data_dir.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(
            f"{data_dir} 下没有 {RAW_GLOB}；请先运行 "
            "python -m intraday_scalper.tools.download_history --adjustment raw"
        )
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        print(f"加载 {path.name}: {len(frame):,} 行")
    bars = _frames_to_bars(frames)
    print(f"raw bars 合计: {len(bars):,} 根")
    return bars


def adjusted_coverage_keys(data_dir: Path = DATA_DIR) -> set[tuple[str, pd.Timestamp]]:
    """Timestamps covered by the adjusted dataset (for the common-coverage run)."""
    keys: set[tuple[str, pd.Timestamp]] = set()
    for path in sorted(data_dir.glob(ADJ_GLOB)):
        frame = pd.read_parquet(path, columns=["symbol", "start_utc"])
        frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
        keys.update(zip(frame["symbol"].astype(str), frame["start_utc"]))
    if not keys:
        raise FileNotFoundError(f"no adjusted dataset found in {data_dir}")
    return keys


def filter_bars_by_keys(bars: Sequence[Bar], keys: set[tuple[str, pd.Timestamp]]) -> List[Bar]:
    return [
        bar
        for bar in bars
        if (bar.symbol, pd.Timestamp(bar.start_utc)) in keys
    ]


def data_integrity_summary(raw_bars: Sequence[Bar], adjusted_keys: set[tuple[str, pd.Timestamp]]) -> Dict[str, Any]:
    """Per-year median price + coverage difference raw vs adjusted."""
    frame = pd.DataFrame(
        {
            "symbol": [bar.symbol for bar in raw_bars],
            "year": [et_date(bar.start_utc).year for bar in raw_bars],
            "close": [bar.close for bar in raw_bars],
            "key": [(bar.symbol, pd.Timestamp(bar.start_utc)) for bar in raw_bars],
        }
    )
    raw_keys = set(frame["key"])
    summary: Dict[str, Any] = {
        "raw_bars": len(raw_bars),
        "adjusted_bars": len(adjusted_keys),
        "raw_only_bars": len(raw_keys - adjusted_keys),
        "adjusted_only_bars": len(adjusted_keys - raw_keys),
        "years": {},
    }
    for year, group in frame.groupby("year"):
        summary["years"][int(year)] = {
            "median_close_by_symbol": {
                symbol: round(float(sub["close"].median()), 2)
                for symbol, sub in group.groupby("symbol")
            },
            "raw_bars": int(len(group)),
            "raw_only_bars": int(len(set(group["key"]) - adjusted_keys)),
        }
    return summary


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def _metric_row(metrics: Any) -> Dict[str, Any]:
    return {
        "total_trades": int(metrics.total_trades),
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": None if metrics.profit_factor is None else round(metrics.profit_factor, 4),
        "net_pnl_dollars": round(metrics.net_pnl_dollars, 2),
        "avg_r": round(metrics.avg_r, 4),
        "max_drawdown_dollars": round(metrics.max_drawdown_dollars, 2),
    }


def _wf_row(wf: v03.V03WalkForwardResult) -> Dict[str, Any]:
    row = _metric_row(wf.aggregate_test_metrics)
    row["positive_folds"] = int(wf.summary.get("positive_folds", 0))
    row["folds"] = int(wf.summary.get("folds", len(wf.folds)))
    return row


# --------------------------------------------------------------------------- #
# Ablation runner
# --------------------------------------------------------------------------- #
def _reference_steps() -> Dict[int, Dict[str, Any]]:
    path = REPORT_DIR / "v03_report.json"
    if not path.exists():
        raise FileNotFoundError(f"missing adjusted reference report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {int(step["step"]): step for step in report["steps"]}


def _run_ablation(prepared: v02.V02Data, label: str) -> Dict[int, Dict[str, Any]]:
    session_dates = list(prepared.dates)
    dates_2024 = [d for d in session_dates if d.year == 2024]
    dates_holdout = [d for d in session_dates if d >= V01_2025]
    results: Dict[int, Dict[str, Any]] = {}
    for variant in v03.ABLATION_VARIANTS:
        print(f"[{label}] {variant.name} …", flush=True)
        started = time.perf_counter()
        full = v03.run_v03_backtest(prepared, variant=variant)
        seg_2024 = v03.run_v03_backtest(prepared, variant=variant, session_dates=dates_2024)
        holdout = v03.run_v03_backtest(prepared, variant=variant, session_dates=dates_holdout)
        walkforward = v03.run_v03_walkforward(prepared, variant=variant)
        results[int(variant.step)] = {
            "step": int(variant.step),
            "name": variant.name,
            "flags": variant.flags(),
            "full": _metric_row(full.metrics),
            "train_2024": _metric_row(seg_2024.metrics),
            "holdout": _metric_row(holdout.metrics),
            "walkforward": _wf_row(walkforward),
            "risk_blocks": dict(full.risk_block_counts),
            "breakeven_armed_trades": int(full.breakeven_armed_trades),
            "shield_stop_exits": int(full.shield_stop_exits),
            "sessions": {"total": len(session_dates), "2024": len(dates_2024), "holdout": len(dates_holdout)},
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        row = results[int(variant.step)]["full"]
        print(
            f"    {label} Step{variant.step}: {row['total_trades']} 笔 / PF "
            f"{row['profit_factor']:.3f} / 净 {row['net_pnl_dollars']:+.2f}"
        )
    return results


def build_comparison(
    adjusted_ref: Dict[int, Dict[str, Any]],
    raw_full: Dict[int, Dict[str, Any]],
    raw_common: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in sorted(raw_full):
        adj = adjusted_ref.get(step, {})
        adj_full = adj.get("full", {}).get("metrics", {})
        adj_wf = adj.get("walkforward", {}).get("aggregate_test", {})
        adj_2024 = adj.get("train_2024", {}).get("metrics", {})
        adj_holdout = adj.get("holdout", {}).get("metrics", {})
        full = raw_full[step]["full"]
        wf = raw_full[step]["walkforward"]
        common = raw_common[step]["full"]
        common_wf = raw_common[step]["walkforward"]
        rows.append(
            {
                "step": step,
                "name": raw_full[step]["name"],
                "adjusted": {
                    "full_trades": adj_full.get("total_trades"),
                    "full_pf": adj_full.get("profit_factor"),
                    "full_net": adj_full.get("net_pnl_dollars"),
                    "wf_net": adj_wf.get("net_pnl_dollars"),
                    "wf_pf": adj_wf.get("profit_factor"),
                    "wf_folds": adj.get("walkforward", {}).get("positive_folds"),
                    "2024_net": adj_2024.get("net_pnl_dollars"),
                    "holdout_net": adj_holdout.get("net_pnl_dollars"),
                },
                "raw_full": full,
                "raw_full_wf": wf,
                "raw_common": common,
                "raw_common_wf": common_wf,
                "delta_full_net": round(full["net_pnl_dollars"] - (adj_full.get("net_pnl_dollars") or 0.0), 2),
                "delta_wf_net": round(wf["net_pnl_dollars"] - (adj_wf.get("net_pnl_dollars") or 0.0), 2),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_crosscheck() -> Dict[str, Any]:
    started = time.perf_counter()
    raw_bars = load_raw_bars()
    adjusted_keys = adjusted_coverage_keys()
    raw_common_bars = filter_bars_by_keys(raw_bars, adjusted_keys)
    print(
        f"coverage: raw {len(raw_bars):,} / adjusted {len(adjusted_keys):,} / "
        f"raw∩adjusted {len(raw_common_bars):,} / raw-only {len(raw_bars)-len(raw_common_bars):,}"
    )

    raw_full_prepared = v02.prepare_v02_data(raw_bars)
    raw_common_prepared = v02.prepare_v02_data(raw_common_bars)
    adjusted_ref = _reference_steps()

    raw_full = _run_ablation(raw_full_prepared, "raw_full")
    raw_common = _run_ablation(raw_common_prepared, "raw_common")
    comparison = build_comparison(adjusted_ref, raw_full, raw_common)

    elapsed = round(time.perf_counter() - started, 2)
    payload = {
        "version": "V0.3-raw-crosscheck",
        "generated_at_utc": datetime_now_utc(),
        "data_source": (
            "Alpaca IEX 1m bars: adjustment=raw (full coverage and common-coverage subset) "
            "vs adjustment=all (adjusted reference in v03_report.json)"
        ),
        "root_cause_note": (
            "Alpaca adjusted bars divide volume by the split factor, so low-volume IEX minutes "
            "round to v=0 and were dropped by the downloader (>0 volume filter): the adjusted "
            "SOXS dataset is missing 16,631 bars vs raw, mostly 2024 pre-split minutes. "
            "Adjusted prices in 2024 are also inflated (~$10.9k vs raw ~$6)."
        ),
        "integrity": data_integrity_summary(raw_bars, adjusted_keys),
        "steps": {"raw_full": list(raw_full.values()), "raw_common": list(raw_common.values())},
        "comparison": comparison,
        "elapsed_seconds": elapsed,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "v03_raw_crosscheck.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path = REPORT_DIR / "v03_raw_ablation.md"
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(f"\nJSON : {json_path}")
    print(f"MD   : {md_path}")
    print(f"总耗时: {elapsed:.1f}s")
    return payload


def datetime_now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# V0.3 raw（未复权）价格交叉验证")
    lines.append("")
    lines.append(
        "- **目的**：验证 V0.3 结论是否受 `adjustment=\"all\"` 后向复权影响。SOXS 经历多次反向合股，"
        "复权后 2024 年价格被放大到 ~$10,907（raw ~$6），且 Alpaca 复权还**按合股比例缩小成交量**，"
        "低成交分钟被 round 成 `v=0` 并被下载器丢弃 → adjusted SOXS 比 raw 少 16,631 根 bar。"
    )
    lines.append(
        "- **方法**：用 `--adjustment raw` 重新下载同一区间，复用**完全相同**的 V0.3 五个变体、常量"
        "（a-priori, not tuned）与同一 29 折 WF 纪律，跑 `raw_full`（全量 raw）与 `raw_common`"
        "（raw 限定在 adjusted 覆盖的时间戳，隔离复权影响与覆盖差异），再与 adjusted 参考对比。"
    )
    lines.append("- **口径**：Alpaca IEX 单交易所，非 consolidated tape；本表仅数据完整性验证，不构成投资建议。")
    lines.append("")
    lines.append("## 数据覆盖差异（raw vs adjusted）")
    lines.append("")
    integrity = payload["integrity"]
    lines.append(
        f"- raw bars **{integrity['raw_bars']:,}** / adjusted bars **{integrity['adjusted_bars']:,}** / "
        f"raw-only **{integrity['raw_only_bars']:,}** / adjusted-only {integrity['adjusted_only_bars']:,}"
    )
    lines.append("")
    lines.append("| 年份 | raw bars | raw-only bars | SOXL 中位价(raw) | SOXS 中位价(raw) |")
    lines.append("|---|---:|---:|---:|---:|")
    for year, info in integrity["years"].items():
        med = info["median_close_by_symbol"]
        lines.append(
            f"| {year} | {info['raw_bars']:,} | {info['raw_only_bars']:,} | "
            f"${med.get('SOXL', 0):,.2f} | ${med.get('SOXS', 0):,.2f} |"
        )
    lines.append("")
    lines.append("## 五步消融：adjusted 参考 vs raw_full vs raw_common")
    lines.append("")
    lines.append(
        "| Step | 结构改动 | adj 全周期 笔/PF/净($) | raw_full 全周期 笔/PF/净($) | ΔvsAdj($) | raw_common 全周期 笔/PF/净($) | "
        "adj WF 净/PF/折 | raw_full WF 净/PF/折 | raw_common WF 净/PF/折 |"
    )
    lines.append("|---|---|---|---|---:|---|---|---|---|")
    for row in payload["comparison"]:
        adj = row["adjusted"]
        rf = row["raw_full"]
        rc = row["raw_common"]
        rfw = row["raw_full_wf"]
        rcw = row["raw_common_wf"]
        lines.append(
            "| {step} | {name} | {at}/{apf}/{anet:+.2f} | {rt}/{rpf}/{rnet:+.2f} | {d:+.2f} | "
            "{ct}/{cpf}/{cnet:+.2f} | {awf:+.2f}/{awpf}/{af} | {rwf:+.2f}/{rwpf}/{rf} | {cwf:+.2f}/{cwpf}/{cf} |".format(
                step=row["step"],
                name=row["name"],
                at=adj["full_trades"],
                apf="—" if adj["full_pf"] is None else f"{adj['full_pf']:.3f}",
                anet=adj["full_net"] or 0.0,
                rt=rf["total_trades"],
                rpf="—" if rf["profit_factor"] is None else f"{rf['profit_factor']:.3f}",
                rnet=rf["net_pnl_dollars"],
                d=row["delta_full_net"],
                ct=rc["total_trades"],
                cpf="—" if rc["profit_factor"] is None else f"{rc['profit_factor']:.3f}",
                cnet=rc["net_pnl_dollars"],
                awf=adj["wf_net"] or 0.0,
                awpf="—" if adj["wf_pf"] is None else f"{adj['wf_pf']:.3f}",
                af=adj["wf_folds"],
                rwf=rfw["net_pnl_dollars"],
                rwpf="—" if rfw["profit_factor"] is None else f"{rfw['profit_factor']:.3f}",
                rf=rfw["positive_folds"],
                cwf=rcw["net_pnl_dollars"],
                cwpf="—" if rcw["profit_factor"] is None else f"{rcw['profit_factor']:.3f}",
                cf=rcw["positive_folds"],
            )
        )
    lines.append("")
    lines.append("## 分段对照（2024 / holdout）")
    lines.append("")
    lines.append("| Step | adj 2024 净($) | raw_full 2024 净($) | raw_common 2024 净($) | adj holdout 净($) | raw_full holdout 净($) | raw_common holdout 净($) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["comparison"]:
        adj = row["adjusted"]
        rf = row["raw_full"]
        rc = row["raw_common"]
        rf_steps = next(s for s in payload["steps"]["raw_full"] if s["step"] == row["step"])
        rc_steps = next(s for s in payload["steps"]["raw_common"] if s["step"] == row["step"])
        lines.append(
            f"| {row['step']} | {adj['2024_net']:+.2f} | {rf_steps['train_2024']['net_pnl_dollars']:+.2f} | "
            f"{rc_steps['train_2024']['net_pnl_dollars']:+.2f} | {adj['holdout_net']:+.2f} | "
            f"{rf_steps['holdout']['net_pnl_dollars']:+.2f} | {rc_steps['holdout']['net_pnl_dollars']:+.2f} |"
        )
    lines.append("")
    raw_step1 = next((s for s in payload["steps"]["raw_full"] if s["step"] == 1), None)
    raw_step4 = next((s for s in payload["steps"]["raw_full"] if s["step"] == 4), None)
    if raw_step1 is not None:
        lines.append(
            f"- raw_full Step1 `ZERO_SHARES` 拒单：**{raw_step1['risk_blocks'].get('ZERO_SHARES', 0)}**；"
            f"raw_full Step4 保本盾 arm **{raw_step4['breakeven_armed_trades'] if raw_step4 else '—'}** 笔、"
            f"shield stop exits **{raw_step4['shield_stop_exits'] if raw_step4 else '—'}** 笔。"
        )
    lines.append(
        "- 判读：若 raw_full 与 adjusted 在方向上一致（V0.3 全周期接近盈亏平衡 / OOS 仍略负），"
        "则 V0.3 结论对复权口径稳健；若差异显著，应以 raw 数据为准并重述结论。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    run_crosscheck()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))