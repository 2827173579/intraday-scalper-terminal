"""Remove watered-down win-rate + fix notional-cap/risk mismatch (B1/B2).

Window: 2026-08-01 .. 2026-11-06, $10,000 initial, V0.4 Step3c-A NoCap.

Part 1: classify the original 68 trades (risk B: $80 risk, $4,500 cap) into
        TARGET / SHIELD_STOP / EOD (positive or negative) / INITIAL STOP.
Part 2: run corrected sizing:
        B1: $60 risk, $10,000 notional cap, B controls
        B2: $80 risk, $10,000 notional cap, reject stop_pct > 1.20%, B controls
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

from intraday_scalper.core import et_date
from intraday_scalper.research import v02, v04
from intraday_scalper.tools.run_risk_compare_10k import (
    REPORT_DIR,
    WINDOW_END,
    WINDOW_START,
    load_raw_bars,
    risk_config,
    summarize_result,
)

RISK_PER_TRADE_B1 = 60.0
RISK_PER_TRADE_B2 = 80.0
NOTIONAL_CAP = 10_000.0
MAX_STOP_PCT_B2 = 0.012
CONTROLS = dict(
    consecutive_stop_threshold=2,
    consecutive_stop_cooldown_minutes=20,
    max_daily_net_loss=180.0,
)


def risk_for(risk_per_trade: float):
    """Plan-B style controls (3 stop / 15 min, -$320) are kept for B1/B2."""
    config, base = risk_config("B")
    risk = replace(
        base,
        account_equity=10_000.0,
        max_risk_per_trade=risk_per_trade,
        max_notional_per_trade=NOTIONAL_CAP,
        allow_fractional_shares=True,
        fractional_precision=2,
        consecutive_stop_threshold=3,
        consecutive_stop_cooldown_minutes=15,
        max_daily_net_loss=320.0,
    )
    return config, risk


def classify_trades(result: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trade in result.trades:
        if not trade.is_closed:
            continue
        detail = result.trade_details.get(trade.signal_id or "", {})
        exit_reason = str(trade.exit_reason)
        stop_kind = detail.get("exit_stop_kind")
        if exit_reason == "TARGET":
            category = "TARGET"
        elif exit_reason == "STOP" and stop_kind == "SHIELD":
            category = "SHIELD_STOP"
        elif exit_reason == "EOD_EXIT":
            category = "EOD_POSITIVE" if trade.pnl > 0 else "EOD_NEGATIVE"
        elif exit_reason == "STOP":
            category = "INITIAL_STOP"
        else:
            category = exit_reason
        rows.append(
            {
                "signal_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": getattr(trade.strategy, "value", str(trade.strategy)),
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "month": et_date(trade.entry_time_utc).strftime("%Y-%m"),
                "category": category,
                "exit_reason": exit_reason,
                "exit_stop_kind": stop_kind,
                "r_multiple": round(float(trade.r_multiple or 0.0), 4),
                "pnl": round(float(trade.pnl), 2),
                "risk_dollars": round(float(trade.risk_dollars or 0.0), 2),
                "shares": round(float(trade.shares), 4),
                "entry_price": round(float(trade.entry_price_fill), 4),
                "exit_price": round(float(trade.exit_price_fill or 0.0), 4),
                "stop_price": round(float(trade.stop_price), 4),
                "target_price": round(float(trade.target_price or 0.0), 4),
                "breakeven_armed": bool(detail.get("breakeven_armed")),
            }
        )
    return rows


def category_table(rows: Sequence[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
    categories = ["TARGET", "SHIELD_STOP", "EOD_POSITIVE", "EOD_NEGATIVE", "INITIAL_STOP"]
    out = []
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        count = len(subset)
        avg_pnl = sum(row["pnl"] for row in subset) / count if count else 0.0
        out.append(
            {
                "category": category,
                "count": count,
                "pct": round(count / total * 100.0, 2) if total else 0.0,
                "avg_pnl": round(avg_pnl, 2),
                "total_pnl": round(sum(row["pnl"] for row in subset), 2),
                "avg_r": round(sum(row["r_multiple"] for row in subset) / count, 4) if count else 0.0,
            }
        )
    return out


def month_category_matrix(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    months = sorted({row["month"] for row in rows})
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for month in months:
        month_rows = [row for row in rows if row["month"] == month]
        matrix[month] = {}
        for category in ["TARGET", "SHIELD_STOP", "EOD_POSITIVE", "EOD_NEGATIVE", "INITIAL_STOP"]:
            subset = [row for row in month_rows if row["category"] == category]
            matrix[month][category] = {
                "count": len(subset),
                "avg_pnl": round(sum(row["pnl"] for row in subset) / len(subset), 2) if subset else 0.0,
                "total_pnl": round(sum(row["pnl"] for row in subset), 2),
            }
    return matrix


def main() -> Dict[str, Any]:
    bars = load_raw_bars()
    prepared = v02.prepare_v02_data(bars)
    dates = [day for day in prepared.dates if WINDOW_START <= day <= WINDOW_END]

    # original 68-trade set (B: $80 risk / $4,500 cap)
    config_b, risk_b = risk_config("B")
    original = v04.run_v04_backtest(
        prepared,
        replace(config_b, risk=risk_b),
        variant=v04.STEP3C_NOCAP_VARIANT,
        session_dates=dates,
        initial_equity=10_000.0,
    )
    rows = classify_trades(original)
    total = len(rows)
    overall_table = category_table(rows, total)
    monthly_matrix = month_category_matrix(rows)

    # B1 / B2 corrected sizing
    variants: Dict[str, Any] = {}
    for name, risk_per_trade, variant, cap in (
        ("B1", RISK_PER_TRADE_B1, v04.STEP3C_NOCAP_VARIANT, NOTIONAL_CAP),
        (
            "B2",
            RISK_PER_TRADE_B2,
            replace(
                v04.STEP3C_NOCAP_VARIANT,
                step=8,
                name="Step3c-A NoCap + max_stop_pct<=1.20% ($10k cap)",
                max_stop_pct=MAX_STOP_PCT_B2,
            ),
            NOTIONAL_CAP,
        ),
    ):
        config_c, risk_c = risk_for(risk_per_trade)
        config_c = replace(config_c, risk=replace(risk_c, max_notional_per_trade=cap))
        result = v04.run_v04_backtest(
            prepared,
            config_c,
            variant=variant,
            session_dates=dates,
            initial_equity=10_000.0,
        )
        summary = summarize_result(result, name)
        classified = classify_trades(result)
        summary["exit_structure"] = category_table(classified, len(classified))
        summary["monthly_matrix"] = month_category_matrix(classified)
        variants[name] = summary
        print(f"{name}: trades={summary['trades']} win={summary['win_rate_pct']} PF={summary['profit_factor']} "
              f"final=${summary['final_equity']:,.2f} net={summary['net']:+.2f} DD=${summary['max_drawdown']:,.2f}")
        for item in summary["exit_structure"]:
            print(f"   {item['category']}: {item['count']} ({item['pct']}%) avg ${item['avg_pnl']:+.2f} avgR {item['avg_r']:+.4f}")
        for month, info in summary["months"].items():
            print(f"   {month}: {info['trades']} trades win {info['win_rate_pct']}% net {info['net']:+.2f}")

    payload = {
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "initial_equity": 10_000.0,
        "engine": "V0.4 Step3c-A NoCap",
        "original_68": {
            "trades": total,
            "nominal_win_rate_pct": round(sum(1 for row in rows if row["pnl"] > 0) / total * 100.0, 2),
            "target_hit_rate_pct": round(sum(1 for row in rows if row["category"] == "TARGET") / total * 100.0, 2),
            "exit_structure": overall_table,
            "monthly_matrix": monthly_matrix,
            "rows": rows,
        },
        "B1": variants["B1"],
        "B2": variants["B2"],
        "definitions": {
            "B1": "risk $60, notional cap $10,000, B controls (3 stop/15 min, -$320 daily)",
            "B2": "risk $80, notional cap $10,000, reject stop_pct > 1.20%, B controls",
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "exit_structure_10k.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote exit_structure_10k.json")
    return payload


if __name__ == "__main__":
    main()
