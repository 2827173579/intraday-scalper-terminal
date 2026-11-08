"""Performance metrics for the event-driven backtester.

Everything here is pure and deterministic: metrics are a function of the closed
:class:`~intraday_scalper.core.models.TradeRecord` list, an optional equity
curve and optional per-trade annotations supplied by the engine (MAE / MFE /
execution slippage).  No clock, no I/O, no network.

Conventions
-----------
* ``win``  = closed trade with ``pnl > 0`` (after fees);
* ``loss`` = closed trade with ``pnl < 0``;
* ``profit_factor`` = gross profit / |gross loss|.  With zero losses it is
  ``inf`` when there is profit and ``0.0`` when there is none; ``to_dict()``
  converts non-finite floats to ``None`` so the report is JSON-safe.
* ``expectancy_r`` / ``avg_r`` are the mean R multiple over closed trades
  (they intentionally coincide; both names are kept for report compatibility).
* all per-day / per-bucket grouping is done on the *America/New_York* clock.

The engine never claims real-market performance: metrics computed from
synthetic bars are logic checks only.  See :mod:`intraday_scalper.backtest`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Mapping, Sequence

from intraday_scalper.core import TradeRecord
from intraday_scalper.core.timeutil import et_date, session_open_utc, to_et

__all__ = [
    "BacktestMetrics",
    "MetricGroup",
    "compute_metrics",
    "to_jsonable",
]

REGULAR_SESSION_MINUTES = 390
TOD_BUCKET_MINUTES = 30


# --------------------------------------------------------------------------- #
# JSON-safe conversion
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / datetimes into JSON-safe values.

    Non-finite floats (``inf`` profit factor, ``nan``) become ``None`` because
    standard JSON has no representation for them.
    """

    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {item.name: to_jsonable(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(value) for value in obj]
    return obj


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Divide-by-zero safe profit factor (``inf`` = profits with zero losses)."""

    if gross_loss > 0.0:
        return gross_profit / gross_loss
    if gross_profit > 0.0:
        return float("inf")
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / len(values) if values else 0.0


def _max_drawdown(values: Sequence[float], initial_equity: float) -> tuple[float, float]:
    """Return ``(max_drawdown_dollars, max_drawdown_pct)`` from an equity path."""

    path = [float(initial_equity)]
    path.extend(float(value) for value in values)
    peak = path[0]
    worst_dollars = 0.0
    worst_pct = 0.0
    for value in path[1:]:
        peak = max(peak, value)
        drawdown = peak - value
        if drawdown > worst_dollars:
            worst_dollars = drawdown
        if peak > 0.0:
            worst_pct = max(worst_pct, drawdown / peak)
    return worst_dollars, worst_pct


@dataclass(frozen=True, slots=True)
class MetricGroup:
    """A small metric summary for one strategy / symbol / time-of-day bucket."""

    label: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_pnl_dollars: float = 0.0
    gross_profit_dollars: float = 0.0
    gross_loss_dollars: float = 0.0
    expectancy_dollars: float = 0.0
    avg_r: float = 0.0
    total_slippage_dollars: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _group_metrics(
    label: str,
    trades: Sequence[TradeRecord],
    trade_details: Mapping[str, Mapping[str, Any]] | None = None,
) -> MetricGroup:
    pnls = [trade.pnl for trade in trades]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl < 0.0)
    breakeven = len(pnls) - wins - losses
    gross_profit = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0.0)
    r_values = [trade.r_multiple for trade in trades]
    return MetricGroup(
        label=label,
        trades=len(trades),
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=(wins / len(trades)) if trades else 0.0,
        profit_factor=_profit_factor(gross_profit, gross_loss),
        net_pnl_dollars=sum(pnls),
        gross_profit_dollars=gross_profit,
        gross_loss_dollars=gross_loss,
        expectancy_dollars=(sum(pnls) / len(trades)) if trades else 0.0,
        avg_r=_mean(r_values),
        total_slippage_dollars=_slippage_from_details(trades, trade_details),
    )


def _slippage_from_details(
    trades: Sequence[TradeRecord],
    trade_details: Mapping[str, Mapping[str, Any]] | None,
) -> float:
    if not trade_details:
        return 0.0
    total = 0.0
    for trade in trades:
        detail = trade_details.get(trade.signal_id)
        if not detail:
            continue
        total += float(detail.get("entry_slippage_dollars", 0.0) or 0.0)
        total += float(detail.get("exit_slippage_dollars", 0.0) or 0.0)
    return total


def _tod_bucket(entry_utc: datetime) -> tuple[str, int]:
    """Return ``(label, sort_index)`` for the ET 30-minute bucket of *entry_utc*."""

    session_open = session_open_utc(entry_utc)
    minutes = (entry_utc - session_open).total_seconds() / 60.0
    if minutes < 0.0 or minutes >= REGULAR_SESSION_MINUTES:
        return "OUTSIDE", 99
    index = int(minutes // TOD_BUCKET_MINUTES)
    start_et = to_et(session_open) + timedelta(minutes=TOD_BUCKET_MINUTES * index)
    end_et = start_et + timedelta(minutes=TOD_BUCKET_MINUTES)
    return f"{start_et:%H:%M}-{end_et:%H:%M}", index


def _time_of_day_performance(trades: Sequence[TradeRecord]) -> dict[str, dict[str, Any]]:
    buckets: dict[tuple[int, str], list[TradeRecord]] = {}
    for trade in trades:
        label, index = _tod_bucket(trade.entry_time_utc)
        buckets.setdefault((index, label), []).append(trade)
    result: dict[str, dict[str, Any]] = {}
    for (index, label) in sorted(buckets, key=lambda item: (item[0], item[1])):
        group = _group_metrics(label, buckets[(index, label)])
        result[label] = {
            "trades": group.trades,
            "wins": group.wins,
            "losses": group.losses,
            "win_rate": group.win_rate,
            "profit_factor": group.profit_factor,
            "net_pnl_dollars": group.net_pnl_dollars,
            "expectancy_dollars": group.expectancy_dollars,
        }
    return result


def _group_by(trades: Sequence[TradeRecord], key: Any) -> dict[str, list[TradeRecord]]:
    grouped: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        grouped.setdefault(str(key(trade)), []).append(trade)
    return grouped


# --------------------------------------------------------------------------- #
# The metric bundle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate backtest performance plus per-strategy / per-symbol breakdowns."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_profit_dollars: float = 0.0
    gross_loss_dollars: float = 0.0
    net_pnl_dollars: float = 0.0
    expectancy_dollars: float = 0.0
    expectancy_r: float = 0.0
    avg_r: float = 0.0
    avg_win_dollars: float = 0.0
    avg_loss_dollars: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_holding_minutes: float = 0.0
    avg_mae_dollars: float = 0.0
    avg_mfe_dollars: float = 0.0
    total_slippage_dollars: float = 0.0
    total_fees_dollars: float = 0.0
    trades_per_day: float = 0.0
    trading_days: int = 0
    initial_equity: float = 0.0
    final_equity: float = 0.0
    time_of_day_performance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    per_strategy: Mapping[str, MetricGroup] = field(default_factory=dict)
    per_symbol: Mapping[str, MetricGroup] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary (non-finite floats become ``None``)."""

        return to_jsonable(self)


# --------------------------------------------------------------------------- #
# Public computation entry point
# --------------------------------------------------------------------------- #
def compute_metrics(
    trades: Sequence[TradeRecord],
    equity_curve: Sequence[tuple[datetime, float]] | None = None,
    *,
    initial_equity: float,
    trade_details: Mapping[str, Mapping[str, Any]] | None = None,
    include_breakdowns: bool = True,
) -> BacktestMetrics:
    """Compute :class:`BacktestMetrics` from closed trades and an equity curve.

    Parameters
    ----------
    trades:
        Any trade records; only *closed* ones (``exit_price_fill`` and
        ``exit_time_utc`` present) are counted.  An empty input returns all-zero
        metrics (never raises).
    equity_curve:
        Chronological ``(timestamp, equity)`` points produced by the engine.
        When omitted, an equity path is derived from the realised P&L of the
        closed trades so the max drawdown is still meaningful.
    initial_equity:
        Starting equity; the drawdown baseline.  Must be positive.
    trade_details:
        Optional per-``signal_id`` annotations from the engine, notably
        ``entry_slippage_dollars``, ``exit_slippage_dollars``,
        ``mae_dollars`` and ``mfe_dollars``.
    include_breakdowns:
        Set to ``False`` for the recursive per-group summaries.
    """

    initial = float(initial_equity)
    if not math.isfinite(initial) or initial <= 0.0:
        raise ValueError(f"initial_equity must be finite and > 0, got {initial_equity!r}")

    notes: list[str] = []
    all_trades = list(trades)
    closed = [trade for trade in all_trades if trade.is_closed]
    if len(closed) != len(all_trades):
        notes.append(f"ignored {len(all_trades) - len(closed)} open/unclosed trade record(s)")
    if not closed:
        notes.append("no closed trades: metrics are zero by definition")

    pnls = [trade.pnl for trade in closed]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl < 0.0)
    breakeven = len(pnls) - wins - losses
    gross_profit = sum(pnl for pnl in pnls if pnl > 0.0)
    gross_loss = -sum(pnl for pnl in pnls if pnl < 0.0)
    net_pnl = sum(pnls)

    win_pnls = [pnl for pnl in pnls if pnl > 0.0]
    loss_pnls = [pnl for pnl in pnls if pnl < 0.0]
    r_values = [trade.r_multiple for trade in closed]
    holding = [trade.holding_minutes for trade in closed]

    # -- equity path / drawdown ----------------------------------------- #
    curve = [(ts, float(value)) for ts, value in (equity_curve or [])]
    if not curve and closed:
        ordered = sorted(
            closed,
            key=lambda trade: (trade.exit_time_utc or trade.entry_time_utc, trade.signal_id),
        )
        running = initial
        derived: list[tuple[datetime, float]] = []
        for trade in ordered:
            running += trade.pnl
            derived.append((trade.exit_time_utc or trade.entry_time_utc, running))
        curve = derived
        notes.append("equity curve derived from realised P&L (no engine curve supplied)")
    drawdown_dollars, drawdown_pct = _max_drawdown([value for _, value in curve], initial)
    final_equity = curve[-1][1] if curve else initial + net_pnl

    # -- per-trade annotations (MAE / MFE / slippage) -------------------- #
    details = trade_details or {}
    mae_values = [
        float(details[trade.signal_id].get("mae_dollars", 0.0) or 0.0)
        for trade in closed
        if trade.signal_id in details
    ]
    mfe_values = [
        float(details[trade.signal_id].get("mfe_dollars", 0.0) or 0.0)
        for trade in closed
        if trade.signal_id in details
    ]
    total_slippage = _slippage_from_details(closed, details)
    if not details:
        # Fall back to the entry slippage encoded in each TradeRecord.
        total_slippage = sum(
            (trade.entry_price_fill - trade.entry_price_signal) * trade.shares for trade in closed
        )
        notes.append("slippage from entry fills only (no exit annotations supplied)")
    if not mae_values and not mfe_values:
        notes.append("MAE/MFE are 0.0: no per-trade bar annotations were supplied")

    # -- trading days ----------------------------------------------------- #
    trading_dates = {et_date(trade.entry_time_utc) for trade in closed}
    trading_dates.update(et_date(ts) for ts, _ in curve)
    trading_days = len(trading_dates)
    total_trades = len(closed)
    trades_per_day = (total_trades / trading_days) if trading_days else 0.0

    # -- breakdowns ------------------------------------------------------- #
    time_of_day: Mapping[str, Mapping[str, Any]] = {}
    per_strategy: Mapping[str, MetricGroup] = {}
    per_symbol: Mapping[str, MetricGroup] = {}
    if include_breakdowns:
        time_of_day = _time_of_day_performance(closed)
        per_strategy = {
            name: _group_metrics(name, group)
            for name, group in sorted(_group_by(closed, lambda trade: trade.strategy.value).items())
        }
        per_symbol = {
            name: _group_metrics(name, group)
            for name, group in sorted(_group_by(closed, lambda trade: trade.symbol).items())
        }

    return BacktestMetrics(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=(wins / total_trades) if total_trades else 0.0,
        profit_factor=_profit_factor(gross_profit, gross_loss),
        gross_profit_dollars=gross_profit,
        gross_loss_dollars=gross_loss,
        net_pnl_dollars=net_pnl,
        expectancy_dollars=(net_pnl / total_trades) if total_trades else 0.0,
        expectancy_r=_mean(r_values),
        avg_r=_mean(r_values),
        avg_win_dollars=_mean(win_pnls),
        avg_loss_dollars=_mean(loss_pnls),
        max_drawdown_dollars=drawdown_dollars,
        max_drawdown_pct=drawdown_pct,
        avg_holding_minutes=_mean(holding),
        avg_mae_dollars=_mean(mae_values),
        avg_mfe_dollars=_mean(mfe_values),
        total_slippage_dollars=total_slippage,
        total_fees_dollars=sum(trade.fees for trade in closed),
        trades_per_day=trades_per_day,
        trading_days=trading_days,
        initial_equity=initial,
        final_equity=final_equity,
        time_of_day_performance=time_of_day,
        per_strategy=per_strategy,
        per_symbol=per_symbol,
        notes=tuple(notes),
    )
