"""==========================================================
SYNTHETIC DATA ONLY — this module generates *fake* 1m bars.
==========================================================

The current Finnhub account has **no historical candle permission**
(``/stock/candle`` returns HTTP 403) and **no real-time US trade feed**
(WebSocket connects but no SOXL/SOXS prints arrive).  Real historical data is
therefore unavailable, so the backtester is exercised on deterministic,
seed-reproducible synthetic bars.  Nothing produced from this module is real
market data and no number derived from it may be presented as real backtest
performance.

Contract
--------
* fully deterministic: the same ``(seed, symbol, et_date)`` always produces the
  exact same bars, independent of the system clock, network or platform;
* every bar is a valid :class:`~intraday_scalper.core.models.Bar` (OHLC
  consistency, true-ish VWAP inside ``[low, high]``, exact 1m window);
* only regular-session minutes (09:30 ET .. 15:59 ET) are generated.

Usage::

    bars = generate_session("SOXL", date(2026, 11, 6), seed=7, trend=0.0002)
    bars = generate_sessions(("SOXL", "SOXS"), next_trading_dates(date(2026, 11, 2), 5), seed=7)
"""

from __future__ import annotations

import math
import random
import zlib
from datetime import date, datetime, time, timedelta
from typing import Mapping, Sequence, Union

from intraday_scalper.core import Bar, ET, UTC

__all__ = [
    "SYNTHETIC_DATA_ONLY",
    "generate_session",
    "generate_sessions",
    "make_choppy_session",
    "make_trending_session",
    "next_trading_dates",
]

SYNTHETIC_DATA_ONLY = (
    "SYNTHETIC DATA ONLY: generated bars are not real market data; results are "
    "logic checks, not backtest performance claims."
)

MARKET_OPEN_ET = time(9, 30)
REGULAR_SESSION_MINUTES = 390


def _session_open_utc(et_date: date) -> datetime:
    return datetime.combine(et_date, MARKET_OPEN_ET, tzinfo=ET).astimezone(UTC)


def _seed_for(seed: int, symbol: str, et_date: date) -> int:
    """Stable sub-seed independent of Python's randomised ``hash()``."""

    payload = f"{int(seed)}|{symbol.strip().upper()}|{et_date.isoformat()}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def next_trading_dates(start: date, count: int) -> list[date]:
    """Return the next *count* weekday dates starting at *start* (inclusive)."""

    if not isinstance(start, date) or isinstance(start, datetime):
        raise TypeError("start must be a datetime.date")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be an int >= 0, got {count!r}")
    dates: list[date] = []
    cursor = start
    while len(dates) < count:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def generate_session(
    symbol: str = "SOXL",
    et_date: date | None = None,
    *,
    seed: int = 0,
    trend: float = 0.0,
    vol: float = 0.0015,
    n_minutes: int = 390,
    price0: float = 40.0,
    base_volume: float = 10_000.0,
) -> list[Bar]:
    """Generate one deterministic synthetic regular-session 1m bar series.

    Parameters
    ----------
    symbol:
        Ticker stored on every bar (uppercased by ``Bar``).
    et_date:
        Exchange-local session date; required (no hidden clock read).
    seed:
        Any integer; the same seed + symbol + date reproduces the same bars.
    trend:
        Per-minute fractional drift (``0.0002`` == +0.02%/minute on average).
    vol:
        Per-minute fractional volatility (Gaussian).
    n_minutes:
        Number of 1m bars from 09:30 ET, ``1..390``.
    price0:
        First bar's open price.
    base_volume:
        Mean per-minute volume scaling.
    """

    if et_date is None:
        raise ValueError("et_date is required (the generator never reads the clock)")
    if not isinstance(et_date, date) or isinstance(et_date, datetime):
        raise TypeError("et_date must be a datetime.date")
    if isinstance(n_minutes, bool) or not isinstance(n_minutes, int):
        raise TypeError("n_minutes must be an int")
    if not 1 <= n_minutes <= REGULAR_SESSION_MINUTES:
        raise ValueError(f"n_minutes must be in 1..{REGULAR_SESSION_MINUTES}, got {n_minutes}")
    for name, value in (("trend", trend), ("vol", vol), ("price0", price0), ("base_volume", base_volume)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if float(vol) < 0.0:
        raise ValueError("vol must be >= 0")
    if float(price0) <= 0.0:
        raise ValueError("price0 must be > 0")
    if float(base_volume) <= 0.0:
        raise ValueError("base_volume must be > 0")
    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol must be non-empty")

    rng = random.Random(_seed_for(seed, symbol, et_date))
    open_utc = _session_open_utc(et_date)
    vol_f = float(vol)
    trend_f = float(trend)
    base_volume_f = float(base_volume)

    bars: list[Bar] = []
    previous_close = float(price0)
    for index in range(n_minutes):
        start_utc = open_utc + timedelta(minutes=index)
        open_price = previous_close
        shock = trend_f + vol_f * rng.gauss(0.0, 1.0)
        close_price = max(0.01, open_price * (1.0 + shock))
        high = max(open_price, close_price) + abs(vol_f * rng.gauss(0.0, 1.0)) * open_price
        low = min(open_price, close_price) - abs(vol_f * rng.gauss(0.0, 1.0)) * min(
            open_price, close_price
        )
        low = max(0.01, low)
        high = max(high, max(open_price, close_price), low)
        low = min(low, min(open_price, close_price))
        vwap = (open_price + high + low + close_price) / 4.0
        volume = base_volume_f * (1.0 + 0.5 * abs(rng.gauss(0.0, 1.0))) + 1.0
        trade_count = max(1, int(volume / 40.0))
        bars.append(
            Bar(
                symbol=symbol,
                timeframe_minutes=1,
                start_utc=start_utc,
                end_utc=start_utc + timedelta(minutes=1),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                vwap=vwap,
                trade_count=trade_count,
                is_complete=True,
            )
        )
        previous_close = close_price
    return bars


def generate_sessions(
    symbols: Sequence[str],
    dates: Sequence[date],
    *,
    seed: int = 0,
    trend: float = 0.0,
    vol: float = 0.0015,
    n_minutes: int = 390,
    price0: Union[float, Mapping[str, float]] = 40.0,
) -> list[Bar]:
    """Generate several symbol/session series, merged and sorted by time.

    The result is directly consumable by
    :class:`~intraday_scalper.backtest.engine.EventDrivenBacktester`: sorted by
    ``(start_utc, symbol)`` and never containing a future bar before its time.
    """

    symbol_list = [str(symbol).strip().upper() for symbol in symbols]
    if not symbol_list:
        raise ValueError("symbols must not be empty")
    if len(set(symbol_list)) != len(symbol_list):
        raise ValueError("symbols contains duplicates")
    date_list = list(dates)
    if not date_list:
        raise ValueError("dates must not be empty")

    if isinstance(price0, Mapping):
        price_map: dict[str, float] = {
            str(key).strip().upper(): float(value) for key, value in price0.items()
        }
    else:
        price_map = {symbol: float(price0) for symbol in symbol_list}

    merged: list[Bar] = []
    for date_index, et_date in enumerate(date_list):
        for symbol_index, symbol in enumerate(symbol_list):
            sub_seed = int(seed) * 1_000_003 + date_index * 7_919 + symbol_index * 104_729
            merged.extend(
                generate_session(
                    symbol,
                    et_date,
                    seed=sub_seed,
                    trend=trend,
                    vol=vol,
                    n_minutes=n_minutes,
                    price0=price_map.get(symbol, 40.0),
                )
            )
    merged.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    return merged


def make_trending_session(
    symbol: str = "SOXL",
    et_date: date | None = None,
    *,
    seed: int = 1,
    direction: int = 1,
    n_minutes: int = 390,
    price0: float = 40.0,
    trend: float = 0.0004,
    vol: float = 0.0008,
) -> list[Bar]:
    """Convenience helper: a gently trending synthetic session."""

    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    return generate_session(
        symbol,
        et_date,
        seed=seed,
        trend=float(direction) * float(trend),
        vol=vol,
        n_minutes=n_minutes,
        price0=price0,
    )


def make_choppy_session(
    symbol: str = "SOXL",
    et_date: date | None = None,
    *,
    seed: int = 2,
    n_minutes: int = 390,
    price0: float = 40.0,
    vol: float = 0.0025,
) -> list[Bar]:
    """Convenience helper: a flat, choppy synthetic session (trend = 0)."""

    return generate_session(
        symbol, et_date, seed=seed, trend=0.0, vol=vol, n_minutes=n_minutes, price0=price0
    )
