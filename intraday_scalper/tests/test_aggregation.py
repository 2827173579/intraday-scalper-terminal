"""Offline tests for tick -> 1m, 1m -> 5m and session VWAP aggregation.

Fixed instant: Friday 2026-11-06 14:30 UTC == 09:30 ET (US standard time).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.aggregation import BarAggregator, FiveMinuteAggregator, SessionVWAP
from intraday_scalper.core import Bar, Tick

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SESSION_DAY = SESSION_START.date()
NEXT_SESSION_START = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)  # Monday, EST


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def make_tick(
    price: float = 40.0,
    volume: float = 100.0,
    ts: datetime | None = None,
    symbol: str = "SOXL",
    trade_id: int | None = None,
) -> Tick:
    return Tick(
        symbol=symbol,
        price=price,
        volume=volume,
        ts_utc=ts if ts is not None else SESSION_START,
        trade_id=trade_id,
    )


def at(minutes: float, base: datetime = SESSION_START) -> datetime:
    return base + timedelta(minutes=minutes)


def make_bar(
    start: datetime,
    *,
    open_: float = 40.0,
    high: float = 40.3,
    low: float = 39.8,
    close: float = 40.1,
    volume: float = 1_000.0,
    vwap: float | None = None,
    timeframe_minutes: int = 1,
    symbol: str = "SOXL",
    trade_count: int = 10,
    is_complete: bool = True,
) -> Bar:
    if vwap is None:
        vwap = (open_ + close) / 2.0
    return Bar(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        start_utc=start,
        end_utc=start + timedelta(minutes=timeframe_minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=trade_count,
        is_complete=is_complete,
    )


def five_one_minute_bars(start: datetime = SESSION_START) -> list[Bar]:
    bars: list[Bar] = []
    previous_close = 40.0
    for index in range(5):
        close = 40.0 + 0.1 * (index + 1)
        open_ = previous_close
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        bars.append(
            make_bar(
                start + timedelta(minutes=index),
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=1_000.0 + 100.0 * index,
                vwap=(open_ + close) / 2.0,
                trade_count=10 + index,
            )
        )
        previous_close = close
    return bars


# --------------------------------------------------------------------------- #
# 1-minute aggregator
# --------------------------------------------------------------------------- #
def test_1m_bar_ohlcv_and_true_vwap():
    aggregator = BarAggregator()
    ticks = [
        make_tick(40.00, 100.0, SESSION_START + timedelta(seconds=1)),
        make_tick(40.20, 50.0, SESSION_START + timedelta(seconds=10)),
        make_tick(39.90, 150.0, SESSION_START + timedelta(seconds=20)),
        make_tick(40.15, 100.0, SESSION_START + timedelta(seconds=30)),
    ]
    for tick in ticks:
        assert aggregator.on_tick(tick) == []

    assert aggregator.current_partial("SOXL") is not None

    completed = aggregator.on_tick(make_tick(40.30, 25.0, at(1.0)))
    assert len(completed) == 1
    bar = completed[0]
    assert bar.is_complete is True
    assert bar.timeframe_minutes == 1
    assert bar.start_utc == SESSION_START
    assert bar.end_utc == at(1.0)
    assert bar.open == pytest.approx(40.00)
    assert bar.high == pytest.approx(40.20)
    assert bar.low == pytest.approx(39.90)
    assert bar.close == pytest.approx(40.15)
    assert bar.volume == pytest.approx(400.0)
    assert bar.trade_count == 4
    expected_vwap = (40.00 * 100 + 40.20 * 50 + 39.90 * 150 + 40.15 * 100) / 400.0
    assert bar.vwap == pytest.approx(expected_vwap)
    assert aggregator.last_completed("SOXL") == bar

    partial = aggregator.current_partial("SOXL")
    assert partial is not None
    assert partial.is_complete is False
    assert partial.open == pytest.approx(40.30)
    assert partial.volume == pytest.approx(25.0)


def test_1m_minute_boundary_59_to_00():
    aggregator = BarAggregator()
    late_tick = make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=59))
    assert aggregator.on_tick(late_tick) == []
    completed = aggregator.on_tick(make_tick(40.5, 200.0, at(1.0)))
    assert len(completed) == 1
    assert completed[0].start_utc == SESSION_START
    assert completed[0].open == pytest.approx(40.0)
    assert completed[0].close == pytest.approx(40.0)
    assert completed[0].volume == pytest.approx(100.0)
    assert aggregator.current_partial("SOXL").start_utc == at(1.0)


def test_1m_skipped_minutes_emit_no_empty_bars():
    aggregator = BarAggregator()
    aggregator.on_tick(make_tick(40.0, 100.0, at(1.0)))
    completed = aggregator.on_tick(make_tick(40.4, 100.0, at(5.0)))
    assert len(completed) == 1
    assert completed[0].start_utc == at(1.0)
    assert completed[0].end_utc == at(2.0)
    current = aggregator.current_partial("SOXL")
    assert current is not None and current.start_utc == at(5.0)
    assert aggregator.current_partial("SOXL").is_complete is False


def test_1m_symbols_are_independent():
    aggregator = BarAggregator()
    aggregator.on_tick(make_tick(40.0, 100.0, at(0), symbol="SOXL"))
    aggregator.on_tick(make_tick(6.0, 100.0, at(0), symbol="SOXS"))
    completed = aggregator.on_tick(make_tick(6.1, 100.0, at(1), symbol="SOXS"))
    assert len(completed) == 1
    assert completed[0].symbol == "SOXS"
    assert aggregator.current_partial("SOXL") is not None
    assert aggregator.current_partial("SOXS").start_utc == at(1.0)


def test_1m_zero_volume_ticks_fall_back_to_last_price():
    aggregator = BarAggregator()
    aggregator.on_tick(make_tick(40.0, 0.0, SESSION_START + timedelta(seconds=1)))
    aggregator.on_tick(make_tick(40.6, 0.0, SESSION_START + timedelta(seconds=2)))
    completed = aggregator.on_tick(make_tick(40.7, 0.0, at(1.0)))
    bar = completed[0]
    assert bar.volume == 0.0
    assert bar.vwap == pytest.approx(40.6)
    assert bar.low <= bar.vwap <= bar.high
    assert bar.trade_count == 2


def test_1m_late_tick_is_ignored():
    aggregator = BarAggregator()
    aggregator.on_tick(make_tick(40.0, 100.0, at(1.0)))
    assert aggregator.on_tick(make_tick(39.9, 100.0, at(0.5))) == []
    assert aggregator.late_ticks == 1
    assert aggregator.current_partial("SOXL").start_utc == at(1.0)


def test_1m_flush_completes_open_minute():
    aggregator = BarAggregator()
    aggregator.on_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=5)))
    flushed = aggregator.flush()
    assert len(flushed) == 1
    assert flushed[0].is_complete is True
    assert aggregator.current_partial("SOXL") is None
    assert aggregator.last_completed("SOXL") == flushed[0]


def test_1m_symbol_filter():
    aggregator = BarAggregator(["SOXL"])
    assert aggregator.on_tick(make_tick(6.0, 100.0, at(0), symbol="SOXS")) == []
    assert aggregator.current_partial("SOXS") is None


def test_1m_rejects_non_tick_input():
    aggregator = BarAggregator()
    with pytest.raises(TypeError):
        aggregator.on_tick("not a tick")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 5-minute aggregator
# --------------------------------------------------------------------------- #
def test_5m_aggregates_five_completed_one_minute_bars():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    bars = five_one_minute_bars()
    outputs: list[Bar] = []
    for bar in bars:
        outputs.extend(aggregator.on_bar(bar))
    assert len(outputs) == 1
    bar_5m = outputs[0]
    assert bar_5m.is_complete is True
    assert bar_5m.timeframe_minutes == 5
    assert bar_5m.start_utc == SESSION_START
    assert bar_5m.end_utc == at(5.0)
    assert bar_5m.open == pytest.approx(bars[0].open)
    assert bar_5m.close == pytest.approx(bars[-1].close)
    assert bar_5m.high == pytest.approx(max(bar.high for bar in bars))
    assert bar_5m.low == pytest.approx(min(bar.low for bar in bars))
    assert bar_5m.volume == pytest.approx(sum(bar.volume for bar in bars))
    assert bar_5m.trade_count == sum(bar.trade_count for bar in bars)
    expected_vwap = sum(bar.vwap * bar.volume for bar in bars) / sum(bar.volume for bar in bars)
    assert bar_5m.vwap == pytest.approx(expected_vwap)
    assert aggregator.last_completed("SOXL") == bar_5m
    assert aggregator.anchor_utc == SESSION_START


def test_5m_is_aligned_to_the_0930_et_anchor():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    outputs: list[Bar] = []
    # Bucket 14:30 gets only three minutes (partial), then the stream jumps to
    # bucket 14:35 which becomes complete.
    for minute in [2, 3, 4, 35, 36, 37, 38, 39]:
        outputs.extend(aggregator.on_bar(make_bar(at(minute))))
    assert len(outputs) == 1
    assert outputs[0].start_utc == at(35.0)
    assert outputs[0].end_utc == at(40.0)
    assert aggregator.dropped_buckets == 1
    assert aggregator.anchor_utc == SESSION_START


def test_5m_requires_all_five_minutes():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    outputs: list[Bar] = []
    # 14:30 bucket misses minute 14:34 forever.
    for minute in [30, 31, 32, 33, 35, 36, 37, 38, 39]:
        outputs.extend(aggregator.on_bar(make_bar(at(minute))))
    assert len(outputs) == 1
    assert outputs[0].start_utc == at(35.0)
    assert all(bar.start_utc != SESSION_START for bar in outputs)


def test_5m_ignores_incomplete_one_minute_bars():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    partial = make_bar(SESSION_START, is_complete=False)
    assert aggregator.on_bar(partial) == []
    assert aggregator.pending_minutes("SOXL") == {}


def test_5m_excludes_premarket_by_default():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    premarket = [make_bar(at(-5 + index)) for index in range(5)]  # 09:25..09:29 ET
    outputs: list[Bar] = []
    for bar in premarket:
        outputs.extend(aggregator.on_bar(bar))
    assert outputs == []

    inclusive = FiveMinuteAggregator(session_open_utc=SESSION_START, include_premarket=True)
    outputs = []
    for bar in premarket:
        outputs.extend(inclusive.on_bar(bar))
    assert len(outputs) == 1
    assert outputs[0].start_utc == at(-5.0)  # 09:25 ET
    assert outputs[0].end_utc == SESSION_START


def test_5m_duplicate_minute_does_not_complete_bucket_early():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    outputs: list[Bar] = []
    for minute in [0, 1, 1, 2, 3]:
        outputs.extend(aggregator.on_bar(make_bar(at(minute))))
    assert outputs == []
    assert aggregator.duplicate_bars == 1
    outputs.extend(aggregator.on_bar(make_bar(at(4.0))))
    assert len(outputs) == 1
    assert outputs[0].start_utc == SESSION_START


def test_5m_zero_total_volume_uses_average_vwap():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    bars = [
        make_bar(
            at(index),
            high=41.0,
            low=39.0,
            volume=0.0,
            vwap=40.0 + index * 0.1,
            trade_count=0,
        )
        for index in range(5)
    ]
    outputs: list[Bar] = []
    for bar in bars:
        outputs.extend(aggregator.on_bar(bar))
    assert len(outputs) == 1
    assert outputs[0].volume == 0.0
    assert outputs[0].vwap == pytest.approx(sum(bar.vwap for bar in bars) / 5.0)


def test_5m_rolls_over_to_a_new_et_day():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    for minute in [30, 31, 32]:  # incomplete bucket of the old day
        aggregator.on_bar(make_bar(at(minute)))
    outputs: list[Bar] = []
    for index in range(5):
        outputs.extend(
            aggregator.on_bar(make_bar(NEXT_SESSION_START + timedelta(minutes=index)))
        )
    assert len(outputs) == 1
    assert outputs[0].start_utc == NEXT_SESSION_START
    assert aggregator.dropped_buckets >= 1
    assert aggregator.anchor_utc == NEXT_SESSION_START


def test_5m_rejects_wrong_input():
    aggregator = FiveMinuteAggregator(session_open_utc=SESSION_START)
    with pytest.raises(ValueError):
        aggregator.on_bar(make_bar(SESSION_START, timeframe_minutes=5))
    with pytest.raises(TypeError):
        aggregator.on_bar("not a bar")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Session VWAP
# --------------------------------------------------------------------------- #
def test_session_vwap_accumulates_true_vwap():
    vwap = SessionVWAP(session_open_utc=SESSION_START)
    assert vwap.value is None
    vwap.on_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=1)))
    assert vwap.value == pytest.approx(40.0)
    vwap.on_tick(make_tick(40.4, 100.0, SESSION_START + timedelta(seconds=2)))
    assert vwap.value == pytest.approx(40.2)
    assert vwap.volume == pytest.approx(200.0)
    assert vwap.tick_count == 2
    assert vwap.et_day == SESSION_DAY


def test_session_vwap_ignores_zero_volume():
    vwap = SessionVWAP(session_open_utc=SESSION_START)
    vwap.on_tick(make_tick(40.0, 0.0, SESSION_START + timedelta(seconds=1)))
    assert vwap.value is None
    assert vwap.volume == 0.0
    assert vwap.tick_count == 0
    vwap.on_tick(make_tick(41.0, 10.0, SESSION_START + timedelta(seconds=2)))
    assert vwap.value == pytest.approx(41.0)


def test_session_vwap_resets_on_new_et_day():
    vwap = SessionVWAP(session_open_utc=SESSION_START)
    vwap.on_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=1)))
    vwap.on_tick(make_tick(50.0, 100.0, NEXT_SESSION_START + timedelta(seconds=1)))
    assert vwap.value == pytest.approx(50.0)
    assert vwap.volume == pytest.approx(100.0)
    assert vwap.tick_count == 1
    assert vwap.et_day == NEXT_SESSION_START.date()


def test_session_vwap_reset_methods():
    vwap = SessionVWAP(session_open_utc=SESSION_START)
    vwap.on_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=1)))
    assert vwap.reset_day(SESSION_DAY) is False
    assert vwap.value == pytest.approx(40.0)
    assert vwap.reset_day(NEXT_SESSION_START.date()) is True
    assert vwap.value is None
    vwap.on_tick(make_tick(41.0, 10.0, SESSION_START + timedelta(seconds=2)))
    vwap.reset()
    assert vwap.value is None
    assert vwap.volume == 0.0


def test_session_vwap_symbol_filter():
    vwap = SessionVWAP("SOXL", session_open_utc=SESSION_START)
    vwap.on_tick(make_tick(6.0, 100.0, SESSION_START, symbol="SOXS"))
    assert vwap.value is None
    vwap.on_tick(make_tick(40.0, 100.0, SESSION_START, symbol="SOXL"))
    assert vwap.value == pytest.approx(40.0)


def test_session_vwap_rejects_non_tick():
    vwap = SessionVWAP(session_open_utc=SESSION_START)
    with pytest.raises(TypeError):
        vwap.on_tick(40.0)  # type: ignore[arg-type]
