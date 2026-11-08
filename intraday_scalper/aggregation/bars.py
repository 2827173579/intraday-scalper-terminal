"""Tick -> 1m bar, 1m -> 5m bar and true session VWAP aggregation.

All timestamps are timezone-aware UTC; the *trading-session anchor* is always
derived from ``intraday_scalper.core.timeutil.session_open_utc`` (09:30 ET) so
bar boundaries follow the exchange clock across DST changes.

Rules enforced here (see ``intraday_scalper/CLAUDE.md``):

* a 1m bar is only emitted once its minute has completed (the next minute's
  first tick arrives, or an explicit :meth:`BarAggregator.flush`);
* a 5m bar is built exclusively from *completed* 1m bars and is only emitted
  when all five minutes of its bucket are present — a missing minute means no
  bar (never a fabricated one);
* VWAP is always ``sum(price * volume) / sum(volume)``. When a bucket contains
  no traded volume at all, the 1m bar falls back to its last observed price
  (still a real trade price) and the 5m bar to the simple average of its 1m
  VWAPs; nothing is invented from OHLC averages. ``SessionVWAP.value`` stays
  ``None`` until real volume arrives.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from intraday_scalper.core import (
    Bar,
    Tick,
    et_date,
    minute_floor_utc,
    session_open_utc,
    to_utc,
)
from intraday_scalper.core.timeutil import MARKET_OPEN_ET, PREMARKET_OPEN_ET

__all__ = ["BarAggregator", "FiveMinuteAggregator", "SessionVWAP"]

_REGULAR_SESSION_MINUTES = 390
_PREMARKET_OFFSET_MINUTES = int(
    (PREMARKET_OPEN_ET.hour * 60 + PREMARKET_OPEN_ET.minute)
    - (MARKET_OPEN_ET.hour * 60 + MARKET_OPEN_ET.minute)
)  # 04:00 ET relative to 09:30 ET == -330 minutes


# --------------------------------------------------------------------------- #
# Internal bucket helpers
# --------------------------------------------------------------------------- #
@dataclass
class _TickBucket:
    symbol: str
    start_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    price_volume: float = 0.0
    trade_count: int = 0

    @classmethod
    def start(cls, tick: Tick, start_utc: datetime) -> "_TickBucket":
        return cls(
            symbol=tick.symbol,
            start_utc=start_utc,
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.volume,
            price_volume=tick.price * tick.volume,
            trade_count=1,
        )

    def add(self, tick: Tick) -> None:
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.price_volume += tick.price * tick.volume
        self.trade_count += 1

    @property
    def vwap(self) -> float:
        if self.volume > 0:
            return self.price_volume / self.volume
        return self.close

    def to_bar(self, *, is_complete: bool, timeframe_minutes: int = 1) -> Bar:
        return Bar(
            symbol=self.symbol,
            timeframe_minutes=timeframe_minutes,
            start_utc=self.start_utc,
            end_utc=self.start_utc + timedelta(minutes=timeframe_minutes),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            trade_count=self.trade_count,
            is_complete=is_complete,
        )


@dataclass
class _FiveMinuteBucket:
    start_utc: datetime
    minutes: dict[datetime, Bar] = field(default_factory=dict)

    def add(self, bar: Bar) -> None:
        self.minutes[bar.start_utc] = bar

    @property
    def complete(self) -> bool:
        return len(self.minutes) == 5

    def to_bar(self) -> Bar:
        ordered = [self.minutes[self.start_utc + timedelta(minutes=i)] for i in range(5)]
        total_volume = sum(bar.volume for bar in ordered)
        if total_volume > 0:
            vwap = sum(bar.vwap * bar.volume for bar in ordered) / total_volume
        else:
            vwap = sum(bar.vwap for bar in ordered) / 5.0
        return Bar(
            symbol=ordered[0].symbol,
            timeframe_minutes=5,
            start_utc=self.start_utc,
            end_utc=self.start_utc + timedelta(minutes=5),
            open=ordered[0].open,
            high=max(bar.high for bar in ordered),
            low=min(bar.low for bar in ordered),
            close=ordered[-1].close,
            volume=total_volume,
            vwap=vwap,
            trade_count=sum(bar.trade_count for bar in ordered),
            is_complete=True,
        )


# --------------------------------------------------------------------------- #
# 1-minute aggregator (tick driven)
# --------------------------------------------------------------------------- #
class BarAggregator:
    """Build completed 1m bars from websocket ticks (UTC-minute buckets)."""

    def __init__(self, symbols: Sequence[str] | None = None) -> None:
        self._symbols: frozenset[str] | None = None
        if symbols is not None:
            normalised = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
            if not normalised:
                raise ValueError("symbols must be non-empty when provided")
            self._symbols = frozenset(normalised)
        self._lock = threading.RLock()
        self._current: dict[str, _TickBucket] = {}
        self._last_completed: dict[str, Bar] = {}
        self._late_ticks = 0

    # -- public API ----------------------------------------------------- #
    def on_tick(self, tick: Tick) -> list[Bar]:
        """Feed one tick; returns the 1m bars completed by this tick (0 or 1)."""
        if not isinstance(tick, Tick):
            raise TypeError(f"on_tick expects a Tick, got {type(tick)!r}")
        if self._symbols is not None and tick.symbol not in self._symbols:
            return []

        minute_start = minute_floor_utc(tick.ts_utc)
        with self._lock:
            bucket = self._current.get(tick.symbol)
            if bucket is None:
                self._current[tick.symbol] = _TickBucket.start(tick, minute_start)
                return []

            if minute_start == bucket.start_utc:
                bucket.add(tick)
                return []

            if minute_start < bucket.start_utc:
                # A late print cannot be folded into an already-emitted minute.
                self._late_ticks += 1
                return []

            # Minute rollover (possibly skipping several minutes): emit the
            # completed bucket and start a new one for this tick. Empty minutes
            # in between are intentionally not emitted.
            completed = bucket.to_bar(is_complete=True)
            self._last_completed[tick.symbol] = completed
            self._current[tick.symbol] = _TickBucket.start(tick, minute_start)
            return [completed]

    def current_partial(self, symbol: str) -> Bar | None:
        """The in-progress (``is_complete=False``) 1m bar, if any."""
        name = str(symbol).strip().upper()
        with self._lock:
            bucket = self._current.get(name)
            if bucket is None:
                return None
            return bucket.to_bar(is_complete=False)

    def last_completed(self, symbol: str) -> Bar | None:
        with self._lock:
            return self._last_completed.get(str(symbol).strip().upper())

    def flush(self, symbol: str | None = None) -> list[Bar]:
        """Force-complete the open minute bucket(s) — intended for shutdown.

        The caller is responsible for only flushing after the minute has really
        ended; during live trading the rollover does this automatically.
        """
        with self._lock:
            if symbol is None:
                names = list(self._current)
            else:
                name = str(symbol).strip().upper()
                names = [name] if name in self._current else []
            completed: list[Bar] = []
            for name in names:
                bar = self._current.pop(name).to_bar(is_complete=True)
                self._last_completed[name] = bar
                completed.append(bar)
            return completed

    @property
    def late_ticks(self) -> int:
        with self._lock:
            return self._late_ticks

    @property
    def symbols(self) -> frozenset[str] | None:
        return self._symbols


# --------------------------------------------------------------------------- #
# 5-minute aggregator (completed 1m bars only)
# --------------------------------------------------------------------------- #
class FiveMinuteAggregator:
    """Aggregate completed 1m bars into 5m bars aligned on the 09:30 ET anchor."""

    def __init__(
        self,
        session_open_utc: datetime | None = None,
        *,
        include_premarket: bool = False,
        symbols: Sequence[str] | None = None,
    ) -> None:
        self._explicit_session_open = (
            to_utc(session_open_utc) if session_open_utc is not None else None
        )
        self._anchor: datetime | None = self._explicit_session_open
        self._anchor_day = (
            et_date(self._explicit_session_open) if self._explicit_session_open else None
        )
        self._include_premarket = bool(include_premarket)
        self._symbols: frozenset[str] | None = None
        if symbols is not None:
            normalised = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
            if not normalised:
                raise ValueError("symbols must be non-empty when provided")
            self._symbols = frozenset(normalised)
        self._lock = threading.RLock()
        self._pending: dict[str, dict[datetime, _FiveMinuteBucket]] = {}
        self._last_completed: dict[str, Bar] = {}
        self._dropped_buckets = 0
        self._duplicate_bars = 0

    # -- public API ----------------------------------------------------- #
    def on_bar(self, bar: Bar) -> list[Bar]:
        """Feed one *completed* 1m bar; returns the 5m bars completed by it."""
        if not isinstance(bar, Bar):
            raise TypeError(f"on_bar expects a Bar, got {type(bar)!r}")
        if not bar.is_complete:
            return []
        if bar.timeframe_minutes != 1:
            raise ValueError(
                f"FiveMinuteAggregator only accepts 1m bars, got {bar.timeframe_minutes}m"
            )
        if self._symbols is not None and bar.symbol not in self._symbols:
            return []

        with self._lock:
            bucket_start = self._bucket_start_locked(bar)
            if bucket_start is None:
                return []

            buckets = self._pending.setdefault(bar.symbol, {})
            for stale_start in [start for start in buckets if start < bucket_start]:
                del buckets[stale_start]
                self._dropped_buckets += 1

            bucket = buckets.get(bucket_start)
            if bucket is None:
                bucket = _FiveMinuteBucket(start_utc=bucket_start)
                buckets[bucket_start] = bucket
            if bar.start_utc in bucket.minutes:
                self._duplicate_bars += 1
                return []
            bucket.add(bar)
            if not bucket.complete:
                return []

            del buckets[bucket_start]
            completed = bucket.to_bar()
            self._last_completed[bar.symbol] = completed
            return [completed]

    def last_completed(self, symbol: str) -> Bar | None:
        with self._lock:
            return self._last_completed.get(str(symbol).strip().upper())

    def pending_minutes(self, symbol: str) -> dict[datetime, int]:
        """For diagnostics/tests: ``{bucket_start: received_minute_count}``."""
        with self._lock:
            buckets = self._pending.get(str(symbol).strip().upper(), {})
            return {start: len(bucket.minutes) for start, bucket in buckets.items()}

    @property
    def anchor_utc(self) -> datetime | None:
        with self._lock:
            return self._anchor

    @property
    def dropped_buckets(self) -> int:
        with self._lock:
            return self._dropped_buckets

    @property
    def duplicate_bars(self) -> int:
        with self._lock:
            return self._duplicate_bars

    # -- internals ------------------------------------------------------ #
    def _bucket_start_locked(self, bar: Bar) -> datetime | None:
        bar_day = et_date(bar.start_utc)

        if self._anchor is not None and self._anchor_day != bar_day:
            # New ET trading day: pending buckets from the old day can never
            # complete, drop them and re-anchor.
            self._dropped_buckets += sum(len(buckets) for buckets in self._pending.values())
            self._pending.clear()
            self._anchor = None
            self._anchor_day = None

        if self._anchor is None:
            if self._explicit_session_open is not None and et_date(
                self._explicit_session_open
            ) == bar_day:
                self._anchor = self._explicit_session_open
            else:
                self._anchor = session_open_utc(bar.start_utc)
            self._anchor_day = et_date(self._anchor)

        offset_minutes = (bar.start_utc - self._anchor).total_seconds() / 60.0
        rounded = round(offset_minutes)
        if not math.isclose(offset_minutes, rounded, abs_tol=1e-6):
            return None
        offset = int(rounded)

        lower = _PREMARKET_OFFSET_MINUTES if self._include_premarket else 0
        if offset < lower or offset >= _REGULAR_SESSION_MINUTES:
            return None

        bucket_index = offset // 5
        return self._anchor + timedelta(minutes=bucket_index * 5)


# --------------------------------------------------------------------------- #
# Session VWAP (tick driven, resets every ET trading day)
# --------------------------------------------------------------------------- #
class SessionVWAP:
    """True session VWAP for one symbol: ``sum(p*v) / sum(v)``."""

    def __init__(
        self,
        symbol: str | None = None,
        *,
        session_open_utc: datetime | None = None,
    ) -> None:
        self._symbol = str(symbol).strip().upper() if symbol is not None else None
        if symbol is not None and not self._symbol:
            raise ValueError("symbol must not be empty")
        # The optional anchor documents/seeds the trading day for tests; live
        # ticks still roll the VWAP over automatically at the next ET day.
        self._lock = threading.RLock()
        self._price_volume = 0.0
        self._volume = 0.0
        self._tick_count = 0
        self._day = et_date(to_utc(session_open_utc)) if session_open_utc is not None else None

    # -- public API ----------------------------------------------------- #
    @property
    def symbol(self) -> str | None:
        return self._symbol

    @property
    def value(self) -> float | None:
        """Volume-weighted session price, or ``None`` before any real volume."""
        with self._lock:
            if self._volume <= 0:
                return None
            return self._price_volume / self._volume

    @property
    def volume(self) -> float:
        with self._lock:
            return self._volume

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def et_day(self):
        with self._lock:
            return self._day

    def on_tick(self, tick: Tick) -> None:
        if not isinstance(tick, Tick):
            raise TypeError(f"on_tick expects a Tick, got {type(tick)!r}")
        if self._symbol is not None and tick.symbol != self._symbol:
            return

        with self._lock:
            tick_day = et_date(tick.ts_utc)
            if self._day is None:
                self._day = tick_day
            elif tick_day != self._day:
                # New ET trading day: the session VWAP restarts from zero.
                self._reset_locked()
                self._day = tick_day
            if tick.volume <= 0:
                return
            self._price_volume += tick.price * tick.volume
            self._volume += tick.volume
            self._tick_count += 1

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
            self._day = None

    def reset_day(self, et_day: Any) -> bool:
        day = et_day.date() if isinstance(et_day, datetime) else et_day
        with self._lock:
            if self._day == day:
                return False
            self._reset_locked()
            self._day = day
            return True

    # -- internals ------------------------------------------------------ #
    def _reset_locked(self) -> None:
        self._price_volume = 0.0
        self._volume = 0.0
        self._tick_count = 0
