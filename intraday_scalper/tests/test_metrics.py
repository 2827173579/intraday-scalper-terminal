"""Offline, deterministic tests for :mod:`intraday_scalper.backtest.metrics`."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.backtest.metrics import (
    BacktestMetrics,
    MetricGroup,
    compute_metrics,
    to_jsonable,
)
from intraday_scalper.core import Mode, StrategyName, TradeRecord, TradeStatus

UTC = timezone.utc
# Friday 2026-11-06 in US standard time: 14:30 UTC == 09:30 ET.
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)


def utc(minutes_after_open: int = 5) -> datetime:
    return SESSION_START + timedelta(minutes=minutes_after_open)


def make_trade(
    *,
    signal_id: str = "SIG-1",
    symbol: str = "SOXL",
    strategy: StrategyName = StrategyName.ORB_5M,
    entry: float = 40.0,
    stop: float = 38.0,
    exit_price: float = 44.0,
    shares: float = 5.0,
    fees: float = 0.0,
    entry_utc: datetime | None = None,
    exit_utc: datetime | None = None,
    exit_reason: str = "TARGET",
) -> TradeRecord:
    entry_time = entry_utc if entry_utc is not None else utc(5)
    exit_time = exit_utc if exit_utc is not None else entry_time + timedelta(minutes=10)
    return TradeRecord(
        signal_id=signal_id,
        symbol=symbol,
        strategy=strategy,
        mode=Mode.DUAL_WATCH,
        shares=shares,
        entry_time_utc=entry_time,
        entry_price_signal=entry,
        entry_price_fill=entry,
        stop_price=stop,
        target_price=entry + (entry - stop) * 2.0,
        exit_time_utc=exit_time,
        exit_price_fill=exit_price,
        exit_reason=exit_reason,
        fees=fees,
        status=TradeStatus.CLOSED,
        trade_id=1,
    )


# --------------------------------------------------------------------------- #
# Known-value arithmetic
# --------------------------------------------------------------------------- #
def test_four_known_trades_metrics_exact():
    """entry=40, stop=38, shares=5 -> risk $10/trade; exits chosen for known P&L."""
    trades = [
        make_trade(signal_id="W1", exit_price=44.0),  # +20  -> +2.0R
        make_trade(signal_id="W2", exit_price=43.0),  # +15  -> +1.5R
        make_trade(signal_id="L1", exit_price=37.0, exit_reason="STOP"),  # -15 -> -1.5R
        make_trade(signal_id="L2", exit_price=39.0, exit_reason="STOP"),  # -5  -> -0.5R
    ]
    metrics = compute_metrics(trades, [], initial_equity=3000.0)

    assert metrics.total_trades == 4
    assert metrics.wins == 2
    assert metrics.losses == 2
    assert metrics.breakeven == 0
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.gross_profit_dollars == pytest.approx(35.0)
    assert metrics.gross_loss_dollars == pytest.approx(20.0)
    assert metrics.profit_factor == pytest.approx(1.75)
    assert metrics.net_pnl_dollars == pytest.approx(15.0)
    assert metrics.expectancy_dollars == pytest.approx(3.75)
    assert metrics.avg_r == pytest.approx(0.375)
    assert metrics.expectancy_r == pytest.approx(0.375)
    assert metrics.avg_win_dollars == pytest.approx(17.5)
    assert metrics.avg_loss_dollars == pytest.approx(-10.0)
    assert metrics.avg_holding_minutes == pytest.approx(10.0)
    assert metrics.initial_equity == pytest.approx(3000.0)
    assert metrics.final_equity == pytest.approx(3015.0)


def test_empty_trades_returns_zero_values_not_exception():
    metrics = compute_metrics([], [], initial_equity=3000.0)
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.profit_factor == 0.0
    assert metrics.expectancy_dollars == 0.0
    assert metrics.expectancy_r == 0.0
    assert metrics.max_drawdown_dollars == 0.0
    assert metrics.max_drawdown_pct == 0.0
    assert metrics.trading_days == 0
    assert metrics.trades_per_day == 0.0
    assert metrics.final_equity == pytest.approx(3000.0)
    assert any("no closed trades" in note for note in metrics.notes)
    # to_dict works on the all-zero object
    assert metrics.to_dict()["total_trades"] == 0


def test_fees_are_deducted_from_pnl():
    trade = make_trade(fees=2.5)
    metrics = compute_metrics([trade], [], initial_equity=3000.0)
    assert trade.pnl == pytest.approx(20.0 - 2.5)
    assert metrics.net_pnl_dollars == pytest.approx(17.5)
    assert metrics.total_fees_dollars == pytest.approx(2.5)


def test_profit_factor_infinite_without_losses_and_json_safe():
    metrics = compute_metrics(
        [make_trade(signal_id="W1"), make_trade(signal_id="W2", exit_price=43.0)],
        [],
        initial_equity=3000.0,
    )
    assert math.isinf(metrics.profit_factor)
    payload = metrics.to_dict()
    assert payload["profit_factor"] is None
    assert json.dumps(payload, allow_nan=False)  # must never emit Infinity/NaN


# --------------------------------------------------------------------------- #
# Drawdown / equity curve
# --------------------------------------------------------------------------- #
def test_max_drawdown_from_equity_curve():
    curve = [
        (utc(10), 3040.0),
        (utc(20), 3060.0),
        (utc(30), 3030.0),
        (utc(40), 3020.0),
    ]
    metrics = compute_metrics([], curve, initial_equity=3000.0)
    assert metrics.max_drawdown_dollars == pytest.approx(40.0)
    assert metrics.max_drawdown_pct == pytest.approx(40.0 / 3060.0)
    assert metrics.final_equity == pytest.approx(3020.0)
    assert metrics.trading_days == 1


def test_equity_curve_derived_from_trades_when_missing():
    trades = [
        make_trade(signal_id="A", exit_price=44.0),
        make_trade(signal_id="B", exit_price=37.0, exit_reason="STOP"),
    ]
    metrics = compute_metrics(trades, None, initial_equity=3000.0)
    # +20 then -15 => peak 3020, trough 3005 => max drawdown 15
    assert metrics.max_drawdown_dollars == pytest.approx(15.0)
    assert any("derived" in note for note in metrics.notes)


# --------------------------------------------------------------------------- #
# Per-strategy / per-symbol / time-of-day breakdowns
# --------------------------------------------------------------------------- #
def test_per_strategy_and_per_symbol_breakdowns():
    trades = [
        make_trade(
            signal_id="A1", strategy=StrategyName.ORB_5M, symbol="SOXL", exit_price=44.0
        ),
        make_trade(
            signal_id="A2", strategy=StrategyName.ORB_5M, symbol="SOXL", exit_price=37.0
        ),
        make_trade(
            signal_id="B1",
            strategy=StrategyName.VWAP_RECLAIM,
            symbol="SOXS",
            exit_price=43.0,
        ),
    ]
    metrics = compute_metrics(trades, [], initial_equity=3000.0)
    assert set(metrics.per_strategy) == {"ORB_5M", "VWAP_RECLAIM"}
    assert set(metrics.per_symbol) == {"SOXL", "SOXS"}

    orb = metrics.per_strategy["ORB_5M"]
    assert isinstance(orb, MetricGroup)
    assert orb.trades == 2
    assert orb.wins == 1
    assert orb.losses == 1
    assert orb.win_rate == pytest.approx(0.5)
    assert orb.net_pnl_dollars == pytest.approx(5.0)
    assert orb.profit_factor == pytest.approx(20.0 / 15.0)

    reclaim = metrics.per_strategy["VWAP_RECLAIM"]
    assert reclaim.trades == 1
    assert reclaim.win_rate == pytest.approx(1.0)

    assert metrics.per_symbol["SOXL"].trades == 2
    assert metrics.per_symbol["SOXS"].trades == 1


def test_time_of_day_buckets_use_et_clock():
    trades = [
        make_trade(signal_id="A", entry_utc=utc(5)),  # 09:35 ET -> 09:30-10:00
        make_trade(signal_id="B", entry_utc=utc(35)),  # 10:05 ET -> 10:00-10:30
        make_trade(signal_id="C", entry_utc=utc(34)),  # 10:04 ET -> 10:00-10:30
        make_trade(signal_id="D", entry_utc=utc(5), exit_price=37.0),  # 09:35 ET loss
    ]
    metrics = compute_metrics(trades, [], initial_equity=3000.0)
    buckets = metrics.time_of_day_performance
    assert set(buckets) == {"09:30-10:00", "10:00-10:30"}
    first = buckets["09:30-10:00"]
    assert first["trades"] == 2
    assert first["wins"] == 1
    assert first["win_rate"] == pytest.approx(0.5)
    second = buckets["10:00-10:30"]
    assert second["trades"] == 2
    assert second["wins"] == 2
    assert second["win_rate"] == pytest.approx(1.0)
    # dict order is chronological
    assert list(buckets) == ["09:30-10:00", "10:00-10:30"]


def test_trading_days_and_trades_per_day():
    day2 = SESSION_START + timedelta(days=3)  # Monday 2026-11-09
    trades = [
        make_trade(signal_id="A", entry_utc=utc(5)),
        make_trade(signal_id="B", entry_utc=utc(10)),
        make_trade(signal_id="C", entry_utc=day2 + timedelta(minutes=5)),
    ]
    metrics = compute_metrics(trades, [], initial_equity=3000.0)
    assert metrics.trading_days == 2
    assert metrics.trades_per_day == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# Engine annotations (slippage / MAE / MFE)
# --------------------------------------------------------------------------- #
def test_slippage_and_mae_mfe_from_trade_details():
    trade = make_trade(signal_id="A")
    details = {
        "A": {
            "entry_slippage_dollars": 1.25,
            "exit_slippage_dollars": 0.75,
            "mae_dollars": 4.0,
            "mfe_dollars": 22.0,
        }
    }
    metrics = compute_metrics([trade], [], initial_equity=3000.0, trade_details=details)
    assert metrics.total_slippage_dollars == pytest.approx(2.0)
    assert metrics.avg_mae_dollars == pytest.approx(4.0)
    assert metrics.avg_mfe_dollars == pytest.approx(22.0)
    assert not any("MAE/MFE are 0.0" in note for note in metrics.notes)


def test_slippage_falls_back_to_entry_fill_difference():
    entry_signal = 40.0
    trade = TradeRecord(
        signal_id="A",
        symbol="SOXL",
        strategy=StrategyName.ORB_5M,
        mode=Mode.DUAL_WATCH,
        shares=10.0,
        entry_time_utc=utc(5),
        entry_price_signal=entry_signal,
        entry_price_fill=40.05,
        stop_price=39.0,
        target_price=42.0,
        exit_time_utc=utc(10),
        exit_price_fill=41.0,
        exit_reason="TARGET",
        status=TradeStatus.CLOSED,
    )
    metrics = compute_metrics([trade], [], initial_equity=3000.0)
    assert metrics.total_slippage_dollars == pytest.approx(0.5)
    assert any("entry fills only" in note for note in metrics.notes)


def test_open_trades_are_ignored_with_note():
    closed = make_trade(signal_id="A")
    still_open = make_trade(signal_id="B")
    still_open = TradeRecord(
        signal_id=still_open.signal_id,
        symbol=still_open.symbol,
        strategy=still_open.strategy,
        mode=still_open.mode,
        shares=still_open.shares,
        entry_time_utc=still_open.entry_time_utc,
        entry_price_signal=still_open.entry_price_signal,
        entry_price_fill=still_open.entry_price_fill,
        stop_price=still_open.stop_price,
        target_price=still_open.target_price,
        status=TradeStatus.OPEN,
    )
    metrics = compute_metrics([closed, still_open], [], initial_equity=3000.0)
    assert metrics.total_trades == 1
    assert any("ignored 1 open" in note for note in metrics.notes)


# --------------------------------------------------------------------------- #
# Validation / serialisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_equity", [0.0, -100.0, float("nan"), float("inf")])
def test_initial_equity_must_be_positive(bad_equity: float):
    with pytest.raises(ValueError):
        compute_metrics([], [], initial_equity=bad_equity)


def test_to_jsonable_handles_nested_structures():
    payload = {
        "inf": float("inf"),
        "nan": float("nan"),
        "when": utc(1),
        "enum": StrategyName.ORB_5M,
        "group": MetricGroup(label="x", trades=2, win_rate=0.5),
        "tuple": (1, 2),
    }
    converted = to_jsonable(payload)
    assert converted["inf"] is None
    assert converted["nan"] is None
    assert converted["when"] == utc(1).isoformat()
    assert converted["enum"] == "ORB_5M"
    assert converted["group"]["trades"] == 2
    assert converted["tuple"] == [1, 2]
    json.dumps(converted, allow_nan=False)


def test_metrics_to_dict_round_trip_is_json_safe():
    trades = [make_trade(signal_id="A"), make_trade(signal_id="B", exit_price=37.0)]
    metrics = compute_metrics(
        trades,
        [(utc(5), 3020.0), (utc(15), 3005.0)],
        initial_equity=3000.0,
        trade_details={
            "A": {"entry_slippage_dollars": 0.1, "exit_slippage_dollars": 0.2},
            "B": {"entry_slippage_dollars": 0.1, "exit_slippage_dollars": 0.3},
        },
    )
    payload = metrics.to_dict()
    assert payload["total_trades"] == 2
    assert payload["per_symbol"]["SOXL"]["trades"] == 2
    assert "09:30-10:00" in payload["time_of_day_performance"]
    text = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in text
