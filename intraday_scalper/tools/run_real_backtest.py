"""Real-data (Alpaca IEX 1m) battle report + walk-forward evaluation.

Loads the local parquet cache produced by ``tools/download_history.py`` and runs:

1. **Full-period event-driven backtest** on every downloaded bar, with the real
   ``RiskManager`` (position sizing $20 risk / $1,200 notional, daily -$60
   circuit breaker, 15-entry cap, 20-minute consecutive-loss cooldown,
   SOXL/SOXS mutual exclusion, 15:30 ET entry cutoff).
2. **Chronological walk-forward**: rolling train/test windows, fresh engine and
   risk manager per fold, no parameter fitting (the strategy parameters are
   fixed config values, so "train" metrics are in-sample diagnostics only and
   every "test" window is a genuine out-of-sample evaluation).

Outputs a concise Chinese battle report to the terminal and writes the full
machine-readable report + equity curve under ``intraday_scalper/var/backtest/``.

Data caveat: the feed is **IEX only** (single venue, a small share of US
consolidated volume). Volume-based filters and VWAP are therefore IEX-scoped;
results are NOT a consolidated-tape backtest and must not be presented as proof
of profitability.

Usage::

    python -m intraday_scalper.tools.run_real_backtest
    python -m intraday_scalper.tools.run_real_backtest --no-walkforward
    python -m intraday_scalper.tools.run_real_backtest --train-days 120 --test-days 20 --step-days 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from intraday_scalper.backtest.engine import BacktestResult, run_backtest
from intraday_scalper.backtest.metrics import BacktestMetrics
from intraday_scalper.backtest.walkforward import run_walkforward
from intraday_scalper.core import Bar, et_date, load_config, to_utc
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.strategies.registry import build_strategies

DATA_DIR = PACKAGE_ROOT / "data"
REPORT_DIR = PACKAGE_ROOT / "var" / "backtest"
CAVEAT = (
    "IEX 单一交易所口径：成交量仅含 IEX 份额，VWAP 为 IEX-trade VWAP，"
    "不是全市场 consolidated tape；样本外结果不代表未来收益，也不构成投资建议。"
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_bars(data_dir: Path = DATA_DIR) -> List[Bar]:
    files = sorted(data_dir.glob("*_1m_2024_2026.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{data_dir} 下没有 *_1m_2024_2026.parquet；请先运行 "
            "python -m intraday_scalper.tools.download_history"
        )
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        print(f"加载 {path.name}: {len(frame):,} 行")
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
    print(f"合计 {len(bars):,} 根 1m bar（{frame['symbol'].nunique()} 个标的）")
    return bars


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def daily_pnl(trades, initial_equity: float) -> pd.DataFrame:
    rows = []
    for trade in trades:
        if not trade.is_closed or trade.exit_time_utc is None:
            continue
        rows.append(
            {
                "et_date": et_date(trade.exit_time_utc).isoformat(),
                "pnl": trade.pnl,
                "r": trade.r_multiple,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["et_date", "pnl", "r", "cum_pnl", "equity"])
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("et_date", as_index=False).agg(pnl=("pnl", "sum"), trades=("pnl", "size"))
    grouped = grouped.sort_values("et_date")
    grouped["cum_pnl"] = grouped["pnl"].cumsum()
    grouped["equity"] = initial_equity + grouped["cum_pnl"]
    return grouped


def equity_stats(result: BacktestResult, initial_equity: float) -> Dict:
    curve = [(ts, float(eq)) for ts, eq in result.equity_curve]
    if not curve:
        return {"points": 0}
    peak = curve[0][1]
    max_dd = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return {
        "points": len(curve),
        "final_equity": round(curve[-1][1], 2),
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd / initial_equity * 100.0, 2),
    }


def _fmt_metrics(metrics: BacktestMetrics) -> Dict:
    return {
        "total_trades": metrics.total_trades,
        "wins": metrics.wins,
        "losses": metrics.losses,
        "win_rate_pct": round(metrics.win_rate * 100.0, 2),
        "profit_factor": None if metrics.profit_factor is None else round(metrics.profit_factor, 3),
        "net_pnl_dollars": round(metrics.net_pnl_dollars, 2),
        "expectancy_dollars": round(metrics.expectancy_dollars, 2),
        "expectancy_r": round(metrics.expectancy_r, 3),
        "avg_r": round(metrics.avg_r, 3),
        "avg_holding_minutes": round(metrics.avg_holding_minutes, 1),
        "max_drawdown_dollars": round(metrics.max_drawdown_dollars, 2),
        "max_drawdown_pct": round(metrics.max_drawdown_pct * 100.0, 2),
        "trades_per_day": round(metrics.trades_per_day, 2),
        "trading_days": metrics.trading_days,
        "total_slippage_dollars": round(metrics.total_slippage_dollars, 2),
        "time_of_day": {
            key: {
                "trades": value.get("trades", 0),
                "win_rate": round(value.get("win_rate", 0.0), 3),
                "net_pnl": round(value.get("net_pnl_dollars", 0.0), 2),
            }
            for key, value in metrics.time_of_day_performance.items()
        },
        "per_strategy": {
            key: {
                "trades": value.trades,
                "win_rate": round(value.win_rate, 3),
                "profit_factor": None if value.profit_factor is None else round(value.profit_factor, 3),
                "net_pnl": round(value.net_pnl_dollars, 2),
                "avg_r": round(value.avg_r, 3),
            }
            for key, value in metrics.per_strategy.items()
        },
        "per_symbol": {
            key: {
                "trades": value.trades,
                "win_rate": round(value.win_rate, 3),
                "net_pnl": round(value.net_pnl_dollars, 2),
                "avg_r": round(value.avg_r, 3),
            }
            for key, value in metrics.per_symbol.items()
        },
    }


def circuit_breaker_stats(
    result: BacktestResult, daily_frame: pd.DataFrame, max_daily_loss: float
) -> Dict:
    blocked = {}
    for key, value in (result.risk_block_counts or {}).items():
        blocked[str(key)] = int(value)
    stats = {
        "risk_block_counts": blocked,
        "daily_loss_limit_blocked_evals": blocked.get("DAILY_LOSS_LIMIT", 0),
        "max_entries_blocked_evals": blocked.get("MAX_ENTRIES_REACHED", 0),
        "cooldown_blocked_evals": blocked.get("CONSECUTIVE_LOSS_COOLDOWN", 0),
        "breaker_days": 0,
        "worst_day_pnl": None,
        "best_day_pnl": None,
        "avg_day_pnl": None,
        "traded_days": 0,
    }
    if daily_frame.empty:
        return stats
    stats["traded_days"] = int(len(daily_frame))
    stats["breaker_days"] = int((daily_frame["pnl"] <= -abs(max_daily_loss)).sum())
    stats["worst_day_pnl"] = round(float(daily_frame["pnl"].min()), 2)
    stats["best_day_pnl"] = round(float(daily_frame["pnl"].max()), 2)
    stats["avg_day_pnl"] = round(float(daily_frame["pnl"].mean()), 2)
    return stats


# --------------------------------------------------------------------------- #
# Report printing
# --------------------------------------------------------------------------- #
def print_report(full: Dict, wf: Optional[Dict], bars: int, span: str) -> None:
    def pct(value):
        return "—" if value is None else f"{value:.2f}%"

    def num(value, digits=2):
        return "—" if value is None else f"{value:+,.{digits}f}"

    def pf(value):
        return "—" if value is None else f"{value:.2f}"

    print("\n" + "=" * 78)
    print("真实数据战报 · Alpaca IEX 1m bars（SOXL / SOXS，常规时段 09:30–16:00 ET）")
    print("=" * 78)
    print(f"数据区间 : {span}")
    print(f"总 bar 数: {bars:,}")
    print(f"口径警告 : {CAVEAT}")

    print("\n[一、全周期逐笔回测（$3,000 账户 / 单笔风险 $20 / 单笔名义 ≤$1,200）]")
    m = full["metrics"]
    print(f"  交易笔数   : {m['total_trades']}  （胜 {m['wins']} / 负 {m['losses']}）")
    print(f"  胜率       : {pct(m['win_rate_pct'])}")
    print(f"  ProfitFactor: {pf(m['profit_factor'])}")
    print(f"  净盈亏     : ${num(m['net_pnl_dollars'])}  期望/笔 ${num(m['expectancy_dollars'])}  ({m['expectancy_r']:+.3f}R)")
    print(f"  平均 R     : {m['avg_r']:+.3f}R   平均持仓 {m['avg_holding_minutes']:.1f} 分钟")
    print(f"  最大回撤   : ${m['max_drawdown_dollars']:,.2f}（{m['max_drawdown_pct']:.2f}%）")
    print(f"  期内交易天 : {m['trading_days']}  日均 {m['trades_per_day']:.2f} 笔")
    print(f"  滑点成本   : ${m['total_slippage_dollars']:,.2f}")

    cb = full["circuit_breaker"]
    print("\n[二、风控 / 熔断统计]")
    print(f"  单日 -${abs(full['max_daily_loss']):.0f} 熔断触发天数 : {cb['breaker_days']}")
    print(f"  熔断后拒单评估次数     : {cb['daily_loss_limit_blocked_evals']}")
    print(f"  连亏冷却拒单评估次数   : {cb['cooldown_blocked_evals']}")
    print(f"  单日上限拒单评估次数   : {cb['max_entries_blocked_evals']}")
    if cb["traded_days"]:
        print(
            f"  交易日均日盈亏         : ${num(cb['avg_day_pnl'])}  "
            f"最差日 ${num(cb['worst_day_pnl'])}  最佳日 ${num(cb['best_day_pnl'])}"
        )

    print("\n[三、分策略表现]")
    for name, stats in m["per_strategy"].items():
        print(
            f"  {name:<14} 笔数 {stats['trades']:>3}  胜率 {stats['win_rate']*100:5.1f}%  "
            f"PF {pf(stats['profit_factor']):>5}  净盈亏 ${num(stats['net_pnl'])}  平均R {stats['avg_r']:+.2f}"
        )

    print("\n[四、分标的]")
    for name, stats in m["per_symbol"].items():
        print(
            f"  {name:<6} 笔数 {stats['trades']:>3}  胜率 {stats['win_rate']*100:5.1f}%  "
            f"净盈亏 ${num(stats['net_pnl'])}  平均R {stats['avg_r']:+.2f}"
        )

    if wf is not None:
        print("\n[五、Walk-Forward 样本外（滚动训练→测试，固定参数、不复利）]")
        print(
            f"  折数 {wf['folds']}  train {wf['train_days']}d / test {wf['test_days']}d / "
            f"step {wf['step_days']}d  每折初始权益 $3,000"
        )
        oos = wf["aggregate_test"]
        print(
            f"  样本外合计: {oos['total_trades']} 笔  胜率 {pct(oos['win_rate_pct'])}  "
            f"PF {pf(oos['profit_factor'])}"
        )
        print(
            f"  样本外净盈亏 ${num(oos['net_pnl_dollars'])}  平均R {oos['avg_r']:+.3f}  "
            f"最大回撤 ${oos['max_drawdown_dollars']:,.2f}"
        )
        print(
            f"  未亏损的测试折: {wf['positive_folds']}/{wf['folds']}  "
            f"最好折 ${num(wf['best_fold_pnl'])}  最差折 ${num(wf['worst_fold_pnl'])}"
        )
        print("  说明: 策略参数为固定配置，未经训练集拟合；train 指标仅作样本内参考。")

    print("\n" + "-" * 78)
    print(CAVEAT)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_report(
    bars: List[Bar],
    *,
    initial_equity: float,
    max_daily_loss: float,
) -> Dict:
    config = load_config()
    strategies = build_strategies(config.strategies)
    started = time.perf_counter()
    result = run_backtest(bars, config, strategies)
    full_elapsed = time.perf_counter() - started
    daily = daily_pnl(result.trades, initial_equity)
    full = {
        "metrics": _fmt_metrics(result.metrics),
        "equity": equity_stats(result, initial_equity),
        "circuit_breaker": circuit_breaker_stats(result, daily, max_daily_loss),
        "max_daily_loss": max_daily_loss,
        "initial_equity": initial_equity,
        "ambiguous_bars": int(getattr(result, "ambiguous_bars", 0) or 0),
        "bars_processed": int(getattr(result, "bars_processed", 0) or 0),
        "elapsed_seconds": round(full_elapsed, 2),
        "notes": list(result.notes),
    }
    return {"config": config, "result": result, "daily": daily, "full": full, "strategies": strategies}


def run_walkforward_report(
    bars: List[Bar],
    config,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    initial_equity: float,
) -> Dict:
    started = time.perf_counter()
    wf = run_walkforward(
        bars,
        config,
        lambda: build_strategies(config.strategies),
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        initial_equity=initial_equity,
        risk_factory=lambda: RiskManager(config),
    )
    elapsed = time.perf_counter() - started
    folds = []
    for fold_result in wf.folds:
        test_metrics = fold_result.test_result.metrics
        folds.append(
            {
                "train_dates": list(fold_result.fold.train_dates),
                "test_dates": list(fold_result.fold.test_dates),
                "leakage_checked": bool(fold_result.leakage_checked),
                "test": _fmt_metrics(test_metrics),
                "test_trades": int(test_metrics.total_trades),
                "train_trades": int(fold_result.train_result.metrics.total_trades),
            }
        )
    aggregate = _fmt_metrics(wf.aggregate_test_metrics)
    fold_pnls = [fold["test"]["net_pnl_dollars"] for fold in folds]
    return {
        "folds": len(folds),
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "aggregate_test": aggregate,
        "positive_folds": sum(1 for pnl in fold_pnls if pnl > 0),
        "best_fold_pnl": max(fold_pnls) if fold_pnls else 0.0,
        "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
        "summary": dict(wf.summary),
        "notes": list(wf.notes),
        "fold_results": folds,
        "elapsed_seconds": round(elapsed, 2),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="Real Alpaca IEX data battle report + walk-forward.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--no-walkforward", action="store_true")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=20)
    args = parser.parse_args(argv)

    bars = load_bars(args.data_dir)
    config = load_config()
    initial_equity = float(args.initial_equity or config.risk.account_equity)
    span = f"{min(bar.start_utc for bar in bars):%Y-%m-%d} .. {max(bar.start_utc for bar in bars):%Y-%m-%d}"

    bundle = build_report(
        bars, initial_equity=initial_equity, max_daily_loss=config.risk.max_daily_net_loss
    )
    wf_report = None
    if not args.no_walkforward:
        print(
            f"\n开始 walk-forward（train {args.train_days}d / test {args.test_days}d / "
            f"step {args.step_days}d），可能需要几分钟…"
        )
        wf_report = run_walkforward_report(
            bars,
            bundle["config"],
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            initial_equity=initial_equity,
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Alpaca IEX 1m bars (single venue; NOT consolidated tape)",
        "data_span": span,
        "total_bars": len(bars),
        "caveat": CAVEAT,
        "full_period": bundle["full"],
        "walkforward": wf_report,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "real_iex_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    equity_path = REPORT_DIR / "real_iex_equity_curve.csv"
    bundle["daily"].to_csv(equity_path, index=False)

    # Full trade ledger for auditing (signal price vs fill price, R, MFE/MAE).
    trades_path = REPORT_DIR / "real_iex_trades.csv"
    ledger = []
    for trade in bundle["result"].trades:
        if not trade.is_closed:
            continue
        ledger.append(
            {
                "signal_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": trade.strategy.value,
                "mode": trade.mode.value,
                "shares": trade.shares,
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat() if trade.exit_time_utc else "",
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason or "",
                "fees": trade.fees,
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
            }
        )
    pd.DataFrame(ledger).to_csv(trades_path, index=False)

    print_report(bundle["full"], wf_report, len(bars), span)
    print(f"\n完整报告: {report_path}")
    print(f"逐日权益: {equity_path}")
    print(f"逐笔流水: {trades_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
