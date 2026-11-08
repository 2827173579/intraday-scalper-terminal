"""Shared pytest fixtures for the intraday scalper test suite.

The fixtures deliberately build *synthetic* data: no test may depend on the
network, a live Finnhub websocket, or the real system clock. Fixed instants use
Friday 2026-11-06 during US standard time, so 09:30 ET == 14:30 UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence

import pytest

from intraday_scalper.core import Bar, Mode, Tick, load_config

UTC = timezone.utc

# Friday 2026-11-06 (after US DST ended) -> 09:30 ET is 14:30 UTC.
SESSION_DATE = datetime(2026, 11, 6, tzinfo=UTC).date()
SESSION_START_UTC = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SESSION_END_UTC = datetime(2026, 11, 6, 21, 0, tzinfo=UTC)


def build_bar(
    symbol: str = "SOXL",
    start: datetime | None = None,
    open_: float = 40.0,
    high: float = 40.20,
    low: float = 39.85,
    close: float = 40.10,
    volume: float = 10_000.0,
    vwap: float | None = None,
    timeframe_minutes: int = 1,
    trade_count: int = 25,
    is_complete: bool = True,
) -> Bar:
    """Build one valid Bar; defaults describe a mildly bullish minute."""
    start_utc = start if start is not None else SESSION_START_UTC
    if vwap is None:
        vwap = (high + low + close) / 3.0
    return Bar(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        start_utc=start_utc,
        end_utc=start_utc + timedelta(minutes=timeframe_minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=trade_count,
        is_complete=is_complete,
    )


def build_bars(
    closes: Sequence[float],
    symbol: str = "SOXL",
    start: datetime | None = None,
    base_volume: float = 10_000.0,
    spread_pct: float = 0.002,
) -> list[Bar]:
    """Build a valid 1m-close series with sane OHLC and true-ish VWAP."""
    start_utc = start if start is not None else SESSION_START_UTC
    bars: list[Bar] = []
    prev_close = closes[0]
    for index, close in enumerate(closes):
        open_ = prev_close
        high = max(open_, close) * (1 + spread_pct)
        low = min(open_, close) * (1 - spread_pct)
        vwap = (open_ + high + low + close) / 4.0
        bars.append(
            build_bar(
                symbol=symbol,
                start=start_utc + timedelta(minutes=index),
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=base_volume,
                vwap=vwap,
            )
        )
        prev_close = close
    return bars


@pytest.fixture
def config():
    """Default configuration loaded from config/default.yaml."""
    return load_config()


@pytest.fixture
def session_start() -> datetime:
    return SESSION_START_UTC


@pytest.fixture
def session_end() -> datetime:
    return SESSION_END_UTC


@pytest.fixture
def make_tick():
    def _make(
        symbol: str = "SOXL",
        price: float = 40.0,
        volume: float = 100.0,
        ts: datetime | None = None,
        trade_id: int | None = None,
    ) -> Tick:
        return Tick(
            symbol=symbol,
            price=price,
            volume=volume,
            ts_utc=ts if ts is not None else SESSION_START_UTC,
            trade_id=trade_id,
        )

    return _make


@pytest.fixture
def make_bar():
    return build_bar


@pytest.fixture
def make_bars():
    return build_bars


@pytest.fixture
def tick_stream() -> Iterator[Tick]:
    """A deterministic, strictly increasing stream of SOXL prints."""
    price = 40.0
    for index in range(120):
        price += 0.01 * (1 if index % 3 else -1)
        yield Tick(
            symbol="SOXL",
            price=round(price, 4),
            volume=50.0 + (index % 7) * 10.0,
            ts_utc=SESSION_START_UTC + timedelta(seconds=index * 30),
            trade_id=index + 1,
        )


@pytest.fixture
def modes() -> list[Mode]:
    return [Mode.LONG_SOXL, Mode.LONG_SOXS, Mode.DUAL_WATCH]
