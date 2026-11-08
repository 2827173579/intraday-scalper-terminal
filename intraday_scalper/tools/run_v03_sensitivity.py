"""Pre-registered structural sensitivity grid for V0.3 on raw (unadjusted) data.

This is **not** a parameter optimisation: the grid cells are declared up front
(entry cutoff 13:30 / 14:00 / 14:30 ET × breakeven-stop offset 0.02R / 0.05R /
0.10R), every cell is reported, and nothing is selected based on the test set.
The purpose is to check whether the V0.3 improvements are *stable* across the
structural choices or whether they only exist at one lucky combination.

All other V0.3 switches stay at their frozen values (fractional shares, two
strategies only — ORB_15M_5M removed, breakeven shield enabled, 1.2R trigger).
The default (14:00, 0.05R) cell must reproduce the raw-full Step 4 result from
``var/backtest/v03_raw_crosscheck.json`` (970 trades / -$214.76 / WF -$234.92).

Outputs: ``var/backtest/v03_sensitivity.json`` and ``v03_sensitivity.md``.

Usage::

    python -m intraday_scalper.tools.run_v03_sensitivity
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.research import v02, v03
from intraday_scalper.tools.run_v03_raw_crosscheck import load_raw_bars

REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"
CUTOFFS: tuple[dt_time, ...] = (dt_time(13, 30), dt_time(14, 0), dt_time(14, 30))
OFFSETS: tuple[float, ...] = (0.02, 0.05, 0.10)
DEFAULT_CUTOFF = dt_time(14, 0)
DEFAULT_OFFSET = 0.05


def _fmt_pf(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _run_cell(prepared, cutoff: dt_time, offset: float) -> Dict[str, Any]:
    variant = replace(
        v03.STEP4_VARIANT,
        name=f"cutoff={cutoff.strftime('%H:%M')}, shield_offset={offset:.2f}R",
        entry_cutoff_et=cutoff,
    )
    v03.BREAKEVEN_OFFSET_R = offset
    try:
        full = v03.run_v03_backtest(prepared, variant=variant)
        wf = v03.run_v03_walkforward(prepared, variant=variant)
    finally:
        v03.BREAKEVEN_OFFSET_R = DEFAULT_OFFSET
    return {
        "cutoff_et": cutoff.strftime("%H:%M"),
        "shield_offset_r": offset,
        "full": {
            "total_trades": int(full.metrics.total_trades),
            "win_rate_pct": round(full.metrics.win_rate * 100.0, 2),
            "profit_factor": None if full.metrics.profit_factor is None else round(full.metrics.profit_factor, 4),
            "net_pnl_dollars": round(full.metrics.net_pnl_dollars, 2),
            "avg_r": round(full.metrics.avg_r, 4),
            "max_drawdown_dollars": round(full.metrics.max_drawdown_dollars, 2),
        },
        "walkforward": {
            "total_trades": int(wf.aggregate_test_metrics.total_trades),
            "win_rate_pct": round(wf.aggregate_test_metrics.win_rate * 100.0, 2),
            "profit_factor": (
                None
                if wf.aggregate_test_metrics.profit_factor is None
                else round(wf.aggregate_test_metrics.profit_factor, 4)
            ),
            "net_pnl_dollars": round(wf.aggregate_test_metrics.net_pnl_dollars, 2),
            "positive_folds": int(wf.summary.get("positive_folds", 0)),
            "folds": int(wf.summary.get("folds", len(wf.folds))),
        },
    }


def _load_reference_step4() -> Dict[str, Any]:
    path = REPORT_DIR / "v03_raw_crosscheck.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(s for s in payload["steps"]["raw_full"] if int(s["step"]) == 4)


def build_markdown(payload: Dict[str, Any]) -> str:
    cells = {(cell["cutoff_et"], cell["shield_offset_r"]): cell for cell in payload["cells"]}
    lines: List[str] = [
        "# V0.3 结构敏感性（raw 未复权数据，预先声明的网格，不做择优）",
        "",
        "- 目的：检验 V0.3 的改善是否只存在于单一「幸运组合」。"
        "网格事先声明：截止时间 {13:30, 14:00, 14:30} ET × 保本盾止损偏移 {0.02, 0.05, 0.10}R；"
        "其余开关冻结（碎股、两策略、1.2R 触发、29 折 WF）。",
        "- 全部单元格都报告；**不依据 test 指标选择参数**。若结论只在某个单元格成立，则说明结构假设不稳定。",
        "- 口径：Alpaca IEX 单交易所 raw 1m bars；非投资建议。",
        "",
        "## 全周期净盈亏($) / PF",
        "",
        "| 截止 ET \\ 偏移 | " + " | ".join(f"{offset:.2f}R" for offset in OFFSETS) + " |",
        "|---|---:|---:|---:|",
    ]
    for cutoff in CUTOFFS:
        label = cutoff.strftime("%H:%M")
        values = []
        for offset in OFFSETS:
            cell = cells[(label, offset)]
            values.append(f"{cell['full']['net_pnl_dollars']:+.2f} / {_fmt_pf(cell['full']['profit_factor'])}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## WF 样本外净盈亏($) / PF / 盈利折",
        "",
        "| 截止 ET \\ 偏移 | " + " | ".join(f"{offset:.2f}R" for offset in OFFSETS) + " |",
        "|---|---:|---:|---:|",
    ]
    for cutoff in CUTOFFS:
        label = cutoff.strftime("%H:%M")
        values = []
        for offset in OFFSETS:
            cell = cells[(label, offset)]["walkforward"]
            values.append(
                f"{cell['net_pnl_dollars']:+.2f} / {_fmt_pf(cell['profit_factor'])} / "
                f"{cell['positive_folds']}/{cell['folds']}"
            )
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 自检与判读",
        "",
        f"- 默认组合（14:00, 0.05R）应复现 raw-full Step4：{payload['self_check']['status']}",
        f"- 全周期净盈亏范围：{payload['summary']['full_net_min']:+.2f} .. {payload['summary']['full_net_max']:+.2f}；"
        f"WF 净盈亏范围：{payload['summary']['wf_net_min']:+.2f} .. {payload['summary']['wf_net_max']:+.2f}。",
        f"- 正 PF 单元格（全周期）：{payload['summary']['full_pf_positive_cells']}/{payload['summary']['cells']}；"
        f"正 PF 单元格（WF 样本外）：{payload['summary']['wf_pf_positive_cells']}/{payload['summary']['cells']}。",
        "- 判读：若 9 个单元格的符号方向一致（例如全周期多数为负、WF 全部为负），"
        "则说明 V0.3 的结论对结构常数不敏感，不能靠「调这两个常数」翻正；"
        "若只有个别单元格转正，则视为噪声，不做择优。",
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
    bars = load_raw_bars()
    prepared = v02.prepare_v02_data(bars)
    reference = _load_reference_step4()

    cells: List[Dict[str, Any]] = []
    for cutoff in CUTOFFS:
        for offset in OFFSETS:
            label = f"cutoff={cutoff.strftime('%H:%M')}, offset={offset:.2f}R"
            print(f"[sensitivity] {label} …", flush=True)
            cell = _run_cell(prepared, cutoff, offset)
            cells.append(cell)
            print(
                f"    full {cell['full']['net_pnl_dollars']:+.2f} / PF {_fmt_pf(cell['full']['profit_factor'])} | "
                f"WF {cell['walkforward']['net_pnl_dollars']:+.2f} / PF {_fmt_pf(cell['walkforward']['profit_factor'])} / "
                f"{cell['walkforward']['positive_folds']} 折"
            )

    # self-check: default cell vs raw-full Step4 reference
    default_cell = next(
        c for c in cells if c["cutoff_et"] == DEFAULT_CUTOFF.strftime("%H:%M") and c["shield_offset_r"] == DEFAULT_OFFSET
    )
    ref_full = reference["full"]
    ref_wf = reference["walkforward"]
    check_ok = (
        default_cell["full"]["total_trades"] == ref_full["total_trades"]
        and abs(default_cell["full"]["net_pnl_dollars"] - ref_full["net_pnl_dollars"]) <= 0.01
        and default_cell["walkforward"]["total_trades"] == ref_wf["total_trades"]
        and abs(default_cell["walkforward"]["net_pnl_dollars"] - ref_wf["net_pnl_dollars"]) <= 0.01
    )
    full_nets = [c["full"]["net_pnl_dollars"] for c in cells]
    wf_nets = [c["walkforward"]["net_pnl_dollars"] for c in cells]
    payload = {
        "version": "V0.3-sensitivity",
        "dataset": "Alpaca IEX 1m bars, adjustment=raw (483,751 bars)",
        "note": "Pre-registered structural grid; all cells reported; no test-set selection.",
        "grid": {
            "cutoffs_et": [c.strftime("%H:%M") for c in CUTOFFS],
            "shield_offsets_r": list(OFFSETS),
            "fixed": {
                "strategies": list(v03.STEP4_VARIANT.enabled_strategies),
                "allow_fractional_shares": True,
                "breakeven_shield": True,
                "breakeven_trigger_r": v03.BREAKEVEN_TRIGGER_R,
                "walkforward": "120/20/20, 29 folds, fresh RiskManager per fold, no compounding",
            },
        },
        "cells": cells,
        "self_check": {
            "status": "PASS ✅ default cell reproduces raw-full Step4" if check_ok else "FAIL ❌ default cell mismatch",
            "reference_step4_full": ref_full,
            "reference_step4_walkforward": ref_wf,
            "default_cell": default_cell,
        },
        "summary": {
            "cells": len(cells),
            "full_net_min": min(full_nets),
            "full_net_max": max(full_nets),
            "wf_net_min": min(wf_nets),
            "wf_net_max": max(wf_nets),
            "full_pf_positive_cells": sum(
                1 for c in cells if (c["full"]["profit_factor"] or 0.0) > 1.0
            ),
            "wf_pf_positive_cells": sum(
                1 for c in cells if (c["walkforward"]["profit_factor"] or 0.0) > 1.0
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "v03_sensitivity.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path = REPORT_DIR / "v03_sensitivity.md"
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print("\n=== sensitivity matrix (full net / PF) ===")
    for cutoff in CUTOFFS:
        row = []
        for offset in OFFSETS:
            cell = next(
                c for c in cells
                if c["cutoff_et"] == cutoff.strftime("%H:%M") and c["shield_offset_r"] == offset
            )
            row.append(
                f"{cell['full']['net_pnl_dollars']:+.2f}/{_fmt_pf(cell['full']['profit_factor'])}"
            )
        print(f"  cutoff {cutoff.strftime('%H:%M')}: " + "  ".join(row))
    print(f"\nself-check: {payload['self_check']['status']}")
    print(f"JSON: {json_path}\nMD  : {md_path}\n总耗时: {payload['summary']['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))