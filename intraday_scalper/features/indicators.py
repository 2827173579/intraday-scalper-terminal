"""Pure technical indicators used by the deterministic strategies.

Contract
--------
* Every public function here is a **pure function**: it reads only its
  arguments, never the clock and never any global state.
* Every function returns a ``list`` with **exactly the same length** as its
  input. Entries that cannot be computed yet (warm-up) are ``None``.
* There is **no look-ahead**: ``f(xs)[k]`` depends only on ``xs[: k + 1]``.
  A prefix of the input always yields a prefix of the output, which is what
  lets a strategy decide on the latest *confirmed* bar and never be changed
  by a later bar.
* Price inputs are plain numeric sequences (usually closes); bar inputs are
  ``Sequence[Bar]`` from :mod:`intraday_scalper.core`.

This module must stay dependency-free (stdlib only) so it can be reused by the
signal engine, the dashboard and the event-driven backtester alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

from intraday_scalper.core import Bar, to_utc

__all__ = [
    "OpeningRange",
    "atr",
    "ema",
    "minutes_into_session",
    "opening_range",
    "rolling_high",
    "rolling_low",
    "rolling_mean",
    "rsi",
    "sma",
    "true_range",
    "volume_mean",
]

Number = Union[int, float]


# --------------------------------------------------------------------------- #
# Input validation helpers
# --------------------------------------------------------------------------- #
def _check_period(period: int, name: str = "period") -> int:
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError(f"{name} must be an int, got {type(period).__name__}")
    if period <= 0:
        raise ValueError(f"{name} must be > 0, got {period}")
    return period


def _as_floats(values: Sequence[Number]) -> list[float]:
    return [float(value) for value in values]


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def rolling_mean(values: Sequence[Number], period: int) -> list[Optional[float]]:
    """Simple rolling mean over ``period`` samples, ``None`` during warm-up."""

    period = _check_period(period)
    series = _as_floats(values)
    out: list[Optional[float]] = [None] * len(series)
    if len(series) < period:
        return out
    window = sum(series[:period])
    out[period - 1] = window / period
    for index in range(period, len(series)):
        window += series[index] - series[index - period]
        out[index] = window / period
    return out


def sma(values: Sequence[Number], period: int) -> list[Optional[float]]:
    """Simple moving average (alias of :func:`rolling_mean`)."""

    return rolling_mean(values, period)


def ema(values: Sequence[Number], period: int) -> list[Optional[float]]:
    """Exponential moving average seeded with the first ``period``-value SMA.

    ``out[period - 1]`` is the SMA of the first ``period`` inputs; every later
    entry follows ``out[i] = a * x[i] + (1 - a) * out[i - 1]`` with
    ``a = 2 / (period + 1)``. Entries before the seed are ``None``.
    """

    period = _check_period(period)
    series = _as_floats(values)
    out: list[Optional[float]] = [None] * len(series)
    if len(series) < period:
        return out
    previous = sum(series[:period]) / period
    out[period - 1] = previous
    alpha = 2.0 / (period + 1.0)
    for index in range(period, len(series)):
        previous = alpha * series[index] + (1.0 - alpha) * previous
        out[index] = previous
    return out


# --------------------------------------------------------------------------- #
# Range / volatility
# --------------------------------------------------------------------------- #
def true_range(bars: Sequence[Bar]) -> list[float]:
    """Wilder true range per bar.

    The first bar uses ``high - low``; later bars use the maximum of
    ``high - low``, ``|high - prev_close|`` and ``|low - prev_close|``.
    All entries are valid (no ``None``).
    """

    ranges: list[float] = []
    previous_close: Optional[float] = None
    for bar in bars:
        high = float(bar.high)
        low = float(bar.low)
        if previous_close is None:
            ranges.append(high - low)
        else:
            ranges.append(
                max(
                    high - low,
                    abs(high - previous_close),
                    abs(low - previous_close),
                )
            )
        previous_close = float(bar.close)
    return ranges


def atr(bars: Sequence[Bar], period: int = 14) -> list[Optional[float]]:
    """Average True Range with Wilder smoothing.

    The seed (index ``period - 1``) is the simple mean of the first ``period``
    true ranges; later values use ``(prev * (period - 1) + tr) / period``.
    Earlier entries are ``None``.
    """

    period = _check_period(period)
    ranges = true_range(bars)
    out: list[Optional[float]] = [None] * len(ranges)
    if len(ranges) < period:
        return out
    previous = sum(ranges[:period]) / period
    out[period - 1] = previous
    for index in range(period, len(ranges)):
        previous = (previous * (period - 1) + ranges[index]) / period
        out[index] = previous
    return out


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #
def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    if average_loss <= 0.0:
        # All gains (or a perfectly flat seed): 100 for pure up, 50 for flat.
        return 100.0 if average_gain > 0.0 else 50.0
    rs = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(values: Sequence[Number], period: int = 14) -> list[Optional[float]]:
    """Wilder Relative Strength Index.

    The first value appears at index ``period`` (it needs ``period`` price
    changes). All-up series return 100, all-down series return 0 and a flat
    series returns 50.
    """

    period = _check_period(period)
    series = _as_floats(values)
    out: list[Optional[float]] = [None] * len(series)
    if len(series) <= period:
        return out

    average_gain = 0.0
    average_loss = 0.0
    for index in range(1, period + 1):
        change = series[index] - series[index - 1]
        if change > 0.0:
            average_gain += change
        elif change < 0.0:
            average_loss -= change
    average_gain /= period
    average_loss /= period
    out[period] = _rsi_from_averages(average_gain, average_loss)

    for index in range(period + 1, len(series)):
        change = series[index] - series[index - 1]
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period
        out[index] = _rsi_from_averages(average_gain, average_loss)
    return out


# --------------------------------------------------------------------------- #
# Bar-series helpers
# --------------------------------------------------------------------------- #
def volume_mean(bars: Sequence[Bar], period: int) -> list[Optional[float]]:
    """Rolling mean of bar volumes (``None`` during warm-up)."""

    return rolling_mean([float(bar.volume) for bar in bars], period)


def rolling_high(bars: Sequence[Bar], period: int) -> list[Optional[float]]:
    """Highest high of the trailing ``period`` bars (``None`` in warm-up)."""

    period = _check_period(period)
    highs = [float(bar.high) for bar in bars]
    out: list[Optional[float]] = [None] * len(highs)
    for index in range(period - 1, len(highs)):
        out[index] = max(highs[index - period + 1 : index + 1])
    return out


def rolling_low(bars: Sequence[Bar], period: int) -> list[Optional[float]]:
    """Lowest low of the trailing ``period`` bars (``None`` in warm-up)."""

    period = _check_period(period)
    lows = [float(bar.low) for bar in bars]
    out: list[Optional[float]] = [None] * len(lows)
    for index in range(period - 1, len(lows)):
        out[index] = min(lows[index - period + 1 : index + 1])
    return out


# --------------------------------------------------------------------------- #
# Opening range / session geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OpeningRange:
    """Completed (or still forming) opening range of one session.

    ``complete`` is ``True`` once every minute of the window
    ``[start_utc, end_utc)`` was observed at least once.
    """

    high: float
    low: float
    start_utc: datetime
    end_utc: datetime
    complete: bool = False
    bar_count: int = 0

    def __post_init__(self) -> None:
        high = float(self.high)
        low = float(self.low)
        if not (math.isfinite(high) and math.isfinite(low)):
            raise ValueError("OpeningRange high/low must be finite")
        if high < low:
            raise ValueError(f"OpeningRange high {high} < low {low}")
        start = to_utc(self.start_utc)
        end = to_utc(self.end_utc)
        if end <= start:
            raise ValueError("OpeningRange.end_utc must be after start_utc")
        if int(self.bar_count) < 0:
            raise ValueError("OpeningRange.bar_count must be >= 0")
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(self, "bar_count", int(self.bar_count))

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    def contains(self, price: Number) -> bool:
        return self.low <= float(price) <= self.high


def opening_range(
    bars_1m: Sequence[Bar],
    session_open_utc: datetime,
    minutes: int,
) -> Optional[OpeningRange]:
    """Aggregate the opening range that starts at ``session_open_utc``.

    Only **completed 1m bars** whose ``start_utc`` falls inside
    ``[session_open_utc, session_open_utc + minutes)`` are counted; pre-market
    bars, after-hours bars and bars starting exactly at ``end_utc`` are
    ignored. ``complete`` becomes ``True`` only when every minute of the
    window was seen at least once. Returns ``None`` when no bar qualifies.
    """

    minutes = _check_period(minutes, "minutes")
    open_utc = to_utc(session_open_utc)
    end_utc = open_utc + timedelta(minutes=minutes)
    highs: list[float] = []
    lows: list[float] = []
    seen_minutes: set[datetime] = set()
    for bar in bars_1m:
        if not bar.is_complete:
            continue
        start = to_utc(bar.start_utc)
        if start < open_utc or start >= end_utc:
            continue
        seen_minutes.add(start)
        highs.append(float(bar.high))
        lows.append(float(bar.low))
    if not seen_minutes:
        return None
    return OpeningRange(
        high=max(highs),
        low=min(lows),
        start_utc=open_utc,
        end_utc=end_utc,
        complete=len(seen_minutes) >= minutes,
        bar_count=len(seen_minutes),
    )


def minutes_into_session(bar_start_utc: datetime, session_open_utc: datetime) -> float:
    """Minutes between the session open and ``bar_start_utc`` (negative pre-open)."""

    return (to_utc(bar_start_utc) - to_utc(session_open_utc)).total_seconds() / 60.0
