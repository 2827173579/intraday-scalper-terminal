"""Offline, deterministic tests for the pure indicator functions.

No network, no clock, no local data: every bar below is synthetic and every
timestamp is anchored to the fixed session ``2026-11-06 14:30 UTC`` (09:30 ET,
US standard time).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import Bar
from intraday_scalper.features import (
    OpeningRange,
    atr,
    ema,
    minutes_into_session,
    opening_range,
    rolling_high,
    rolling_low,
    rolling_mean,
    rsi,
    sma,
    true_range,
    volume_mean,
)

UTC = timezone.utc
SESSION_OPEN = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET (EST)


# --------------------------------------------------------------------------- #
# Synthetic bar helpers
# --------------------------------------------------------------------------- #
def make_bar(
    start_minute: int = 0,
    *,
    open_: float = 40.0,
    high: float = 40.2,
    low: float = 39.8,
    close: float = 40.1,
    volume: float = 1_000.0,
    symbol: str = "SOXL",
    is_complete: bool = True,
    vwap: float | None = None,
) -> Bar:
    if vwap is None:
        vwap = (high + low + close) / 3.0
    start = SESSION_OPEN + timedelta(minutes=start_minute)
    return Bar(
        symbol=symbol,
        timeframe_minutes=1,
        start_utc=start,
        end_utc=start + timedelta(minutes=1),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=10,
        is_complete=is_complete,
    )


def bar_from_hl(start_minute: int, high: float, low: float) -> Bar:
    mid = (high + low) / 2.0
    return make_bar(start_minute, open_=mid, high=high, low=low, close=mid, vwap=mid)


# --------------------------------------------------------------------------- #
# SMA / rolling mean / EMA
# --------------------------------------------------------------------------- #
def test_sma_known_values() -> None:
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_sma_short_input_is_all_none() -> None:
    assert sma([1.0, 2.0], 3) == [None, None]


def test_rolling_mean_equals_sma() -> None:
    values = [10.0, 11.0, 9.5, 12.0, 12.5, 11.0]
    assert rolling_mean(values, 4) == sma(values, 4)


def test_sma_invalid_period_raises() -> None:
    with pytest.raises(ValueError):
        sma([1.0, 2.0, 3.0], 0)
    with pytest.raises(TypeError):
        rolling_mean([1.0, 2.0, 3.0], 1.5)  # type: ignore[arg-type]


def test_ema_known_values() -> None:
    # seed = SMA(1,2,3) = 2 at index 2, alpha = 0.5 afterwards
    assert ema([1.0, 2.0, 3.0, 4.0, 5.0], 3) == [None, None, 2.0, 3.0, 4.0]


def test_ema_period_one_equals_input() -> None:
    values = [3.0, 1.0, 4.0]
    assert ema(values, 1) == values


def test_ema_first_valid_is_sma_seed() -> None:
    values = [2.0, 4.0, 6.0, 8.0]
    result = ema(values, 4)
    assert result[:3] == [None, None, None]
    assert result[3] == pytest.approx(5.0)


def test_ema_has_no_lookahead() -> None:
    values = [10.0, 10.5, 9.8, 11.2, 11.0, 10.4, 12.0, 12.5, 11.8, 13.0]
    period = 4
    full = ema(values, period)
    assert len(full) == len(values)
    for k in range(period, len(values) + 1):
        assert ema(values[:k], period) == full[:k]


# --------------------------------------------------------------------------- #
# True range / ATR
# --------------------------------------------------------------------------- #
def test_true_range_known_values() -> None:
    bars = [
        make_bar(0, open_=10.0, high=10.5, low=9.5, close=10.0),
        make_bar(1, open_=10.2, high=10.8, low=10.1, close=10.5),
        make_bar(2, open_=10.6, high=10.9, low=9.9, close=10.0),
    ]
    assert true_range(bars) == pytest.approx([1.0, 0.8, 1.0])


def test_atr_wilder_smoothing_known_values() -> None:
    bars = [
        make_bar(0, open_=10.0, high=10.5, low=9.5, close=10.0),
        make_bar(1, open_=10.2, high=10.8, low=10.1, close=10.5),
        make_bar(2, open_=10.6, high=10.9, low=9.9, close=10.0),
    ]
    result = atr(bars, period=2)
    # None, (1.0+0.8)/2 = 0.9, (0.9*1 + 1.0)/2 = 0.95
    assert result[0] is None
    assert result[1] == pytest.approx(0.9)
    assert result[2] == pytest.approx(0.95)


def test_atr_flat_bars_all_equal_range() -> None:
    bars = [make_bar(i, open_=10.0, high=10.2, low=9.8, close=10.0, vwap=10.0) for i in range(5)]
    result = atr(bars, period=3)
    assert result[:2] == [None, None]
    assert result[2:] == pytest.approx([0.4, 0.4, 0.4])


def test_atr_period_one_equals_true_range() -> None:
    bars = [
        make_bar(0, open_=10.0, high=10.5, low=9.5, close=10.0),
        make_bar(1, open_=10.2, high=10.8, low=10.1, close=10.5),
    ]
    assert atr(bars, period=1) == pytest.approx(true_range(bars))


def test_atr_has_no_lookahead() -> None:
    bars = [bar_from_hl(i, 40.2 + 0.1 * (i % 3), 39.8 - 0.05 * (i % 2)) for i in range(9)]
    full = atr(bars, period=3)
    assert len(full) == len(bars)
    for k in range(3, len(bars) + 1):
        assert atr(bars[:k], period=3) == full[:k]


# --------------------------------------------------------------------------- #
# RSI
# --------------------------------------------------------------------------- #
def test_rsi_all_gains_is_100() -> None:
    values = [float(i) for i in range(1, 21)]
    result = rsi(values, period=14)
    assert result[:14] == [None] * 14
    assert result[14] == pytest.approx(100.0)
    assert result[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_zero() -> None:
    values = [float(21 - i) for i in range(20)]
    result = rsi(values, period=14)
    assert result[14] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(0.0)


def test_rsi_flat_series_is_50() -> None:
    result = rsi([40.0] * 20, period=14)
    assert result[14] == pytest.approx(50.0)
    assert result[-1] == pytest.approx(50.0)


def test_rsi_wilder_known_values() -> None:
    # period=2 on [1,2,3,2,2]: first avg gain=1/loss=0 -> 100,
    # then (0.5 gain / 0.5 loss) -> 50, then 50 again.
    result = rsi([1.0, 2.0, 3.0, 2.0, 2.0], period=2)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(100.0)
    assert result[3] == pytest.approx(50.0)
    assert result[4] == pytest.approx(50.0)


def test_rsi_short_input_is_all_none() -> None:
    assert rsi([1.0] * 14, period=14) == [None] * 14


def test_rsi_has_no_lookahead() -> None:
    values = [10.0, 10.4, 9.9, 10.6, 11.0, 10.7, 11.3, 11.1, 11.8, 12.0, 11.6, 12.2]
    full = rsi(values, period=3)
    for k in range(4, len(values) + 1):
        assert rsi(values[:k], period=3) == full[:k]


def test_rsi_invalid_period_raises() -> None:
    with pytest.raises(ValueError):
        rsi([1.0, 2.0], 0)


# --------------------------------------------------------------------------- #
# Volume mean / rolling extremes
# --------------------------------------------------------------------------- #
def test_volume_mean_known_values() -> None:
    bars = [make_bar(i, volume=volume) for i, volume in enumerate([100.0, 200.0, 300.0, 400.0])]
    assert volume_mean(bars, 2) == [None, 150.0, 250.0, 350.0]


def test_rolling_high_and_low_known_values() -> None:
    highs = [10.0, 11.0, 9.5, 12.0]
    lows = [9.0, 10.0, 8.5, 11.0]
    bars = [bar_from_hl(i, highs[i], lows[i]) for i in range(4)]
    assert rolling_high(bars, 2) == [None, 11.0, 11.0, 12.0]
    assert rolling_low(bars, 2) == [None, 9.0, 8.5, 8.5]


# --------------------------------------------------------------------------- #
# Opening range
# --------------------------------------------------------------------------- #
def _opening_bars() -> list[Bar]:
    specs = [
        (40.00, 40.20, 39.80),
        (40.00, 40.30, 39.90),
        (39.95, 40.10, 39.70),
        (40.05, 40.40, 39.85),
        (40.10, 40.25, 39.90),
    ]
    return [bar_from_hl(index, high, low) for index, (_, high, low) in enumerate(specs)]


def test_opening_range_complete() -> None:
    result = opening_range(_opening_bars(), SESSION_OPEN, 5)
    assert result is not None
    assert result.complete is True
    assert result.high == pytest.approx(40.40)
    assert result.low == pytest.approx(39.70)
    assert result.start_utc == SESSION_OPEN
    assert result.end_utc == SESSION_OPEN + timedelta(minutes=5)
    assert result.bar_count == 5
    assert result.range == pytest.approx(0.70)
    assert result.contains(40.0) is True


def test_opening_range_incomplete_when_minutes_missing() -> None:
    result = opening_range(_opening_bars()[:3], SESSION_OPEN, 5)
    assert result is not None
    assert result.complete is False
    assert result.bar_count == 3
    assert result.high == pytest.approx(40.30)
    assert result.low == pytest.approx(39.70)


def test_opening_range_excludes_premarket_incomplete_and_post_window_bars() -> None:
    bars = [
        bar_from_hl(-60, 50.0, 49.0),  # pre-market (08:30 ET) - must not count
        make_bar(1, open_=40.0, high=60.0, low=39.9, close=40.0, is_complete=False),
        *_opening_bars(),
        bar_from_hl(5, 70.0, 69.0),  # starts exactly at OR end - must not count
    ]
    result = opening_range(bars, SESSION_OPEN, 5)
    assert result is not None
    assert result.high == pytest.approx(40.40)
    assert result.low == pytest.approx(39.70)
    assert result.bar_count == 5
    assert result.complete is True


def test_opening_range_returns_none_without_qualifying_bars() -> None:
    assert opening_range([], SESSION_OPEN, 5) is None
    assert opening_range([bar_from_hl(-60, 50.0, 49.0)], SESSION_OPEN, 5) is None


def test_opening_range_15m_requires_fifteen_minutes() -> None:
    bars = [bar_from_hl(i, 40.2, 39.8) for i in range(15)]
    result = opening_range(bars, SESSION_OPEN, 15)
    assert result is not None
    assert result.complete is True
    assert result.bar_count == 15
    assert result.end_utc == SESSION_OPEN + timedelta(minutes=15)


def test_opening_range_invalid_minutes_raises() -> None:
    with pytest.raises(ValueError):
        opening_range(_opening_bars(), SESSION_OPEN, 0)


def test_opening_range_dataclass_validates_bounds() -> None:
    with pytest.raises(ValueError):
        OpeningRange(high=1.0, low=2.0, start_utc=SESSION_OPEN, end_utc=SESSION_OPEN)


# --------------------------------------------------------------------------- #
# Session geometry
# --------------------------------------------------------------------------- #
def test_minutes_into_session() -> None:
    assert minutes_into_session(SESSION_OPEN, SESSION_OPEN) == pytest.approx(0.0)
    assert minutes_into_session(SESSION_OPEN + timedelta(minutes=3), SESSION_OPEN) == pytest.approx(3.0)
    assert minutes_into_session(SESSION_OPEN - timedelta(minutes=60), SESSION_OPEN) == pytest.approx(-60.0)
