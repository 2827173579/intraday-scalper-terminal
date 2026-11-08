"""Build the compact V0.5b web payload for terminal replay/recap cards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\wendang\deepseek-harness\default-workspace\intraday_scalper")
REPORT = ROOT / "var" / "backtest" / "v05b_report.json"
TRADES = ROOT / "var" / "backtest" / "v05b_trades.csv"
OUT = ROOT / "var" / "backtest" / "v05b_web.json"

report = json.loads(REPORT.read_text(encoding="utf-8"))
frame = pd.read_csv(TRADES)
winner = report["selection"]["winner"]
a = frame[(frame["variant"] == winner) & (frame["period"] == "3M")].copy()
a["entry_date_et"] = pd.to_datetime(a["entry_date_et"])

categories = []
for category, subset in [
    ("TARGET", a[a["exit_reason"] == "TARGET"]),
    ("LADDER_STOP", a[a["exit_stop_kind"].fillna("").str.startswith("LADDER")]),
    ("EOD_POSITIVE", a[(a["exit_reason"] == "EOD_EXIT") & (a["pnl"] > 0)]),
    ("EOD_NEGATIVE", a[(a["exit_reason"] == "EOD_EXIT") & (a["pnl"] <= 0)]),
    ("INITIAL_STOP", a[(a["exit_reason"] == "STOP") & (a["exit_stop_kind"] == "INITIAL")]),
    ("TRAIL_EXIT", a[a["exit_reason"] == "TRAIL_EMA21"]),
]:
    if category == "TRAIL_EXIT" and subset.empty:
        continue
    count = len(subset)
    categories.append(
        {
            "category": category,
            "count": count,
            "pct": round(count / len(a) * 100.0, 2) if len(a) else 0.0,
            "avg_pnl": round(float(subset["pnl"].mean()), 2) if count else 0.0,
            "total_pnl": round(float(subset["pnl"].sum()), 2) if count else 0.0,
            "avg_r": round(float(subset["r_multiple"].mean()), 4) if count else 0.0,
        }
    )

months = {}
for month, subset in a.groupby(a["entry_date_et"].dt.strftime("%Y-%m")):
    wins = int((subset["pnl"] > 0).sum())
    gross_profit = float(subset.loc[subset["pnl"] > 0, "pnl"].sum())
    gross_loss = -float(subset.loc[subset["pnl"] <= 0, "pnl"].sum())
    months[month] = {
        "trades": int(len(subset)),
        "win_rate_pct": round(wins / len(subset) * 100.0, 2) if len(subset) else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "net": round(float(subset["pnl"].sum()), 2),
    }

window = report["matrix"][winner]["window"]
compare = []
for name, item in report["matrix"].items():
    wf = item["walkforward"]
    compare.append(
        {
            "name": name,
            "trades": item["window"]["total_trades"],
            "pf": item["window"]["profit_factor"],
            "net": item["window"]["net_pnl_dollars"],
            "full_net": item["full"]["net_pnl_dollars"],
            "full_pf": item["full"]["profit_factor"],
            "wf_net": wf["net_pnl_dollars"],
            "wf_pf": wf["profit_factor"],
            "wf_folds": wf["positive_folds"],
        }
    )

payload = {
    "version": "V0.5b",
    "winner": winner,
    "period": {
        "label": f"{winner} · 2026-08-01~11-06 ($10,000)",
        "trades": window["total_trades"],
        "win_rate_pct": window["win_rate_pct"],
        "target_hit_rate_pct": round(window["target_exits"] / window["total_trades"] * 100.0, 2) if window["total_trades"] else 0.0,
        "target_exits": window["target_exits"],
        "target_avg_r": window["target_avg_r"],
        "profit_factor": window["profit_factor"],
        "net": window["net_pnl_dollars"],
        "net_pct": window["net_pct"],
        "final_equity": 10_000.0 + float(window["net_pnl_dollars"]),
        "max_drawdown": window["max_drawdown_dollars"],
        "avg_r": window["avg_r"],
        "months": months,
    },
    "exit_structure": categories,
    "compare": compare,
    "friday": report["friday"],
    "selection": report["selection"],
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("wrote", OUT)
print("winner:", winner, "3M net:", window["net_pnl_dollars"], "trades:", window["total_trades"])
print("exit structure:", [(item["category"], item["count"], item["avg_pnl"]) for item in categories])
print("months:", months)
