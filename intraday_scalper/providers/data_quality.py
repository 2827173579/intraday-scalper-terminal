"""Offline-testable data-quality monitor for the websocket tick stream.

Why this exists
---------------
The Finnhub free plan may connect successfully but never deliver US real-time
trades. The system must notice that (and show ``DATA_UNAVAILABLE``) instead of
trading on stale or fabricated data. This module is pure bookkeeping: it never
touches the network and never reads the wall clock unless no clock is injected.

Rules
-----
* no accepted tick, or ``stale_seconds > DataConfig.staleness_seconds`` -> ``STALE``
* only garbage ticks, futures beyond ``max_tick_age_seconds``, or severe
  out-of-order traffic -> ``INVALID``
* stale (or missing) data for *every* configured symbol -> aggregate ``STALE``;
  any ``INVALID`` symbol dominates the aggregate; otherwise OK when at least
  one symbol is OK (the other symbol's data is then not usable, which the
  strategies handle per-symbol).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Deque, Mapping, Sequence

from intraday_scalper.core import (
    UTC,
    DataConfig,
    DataStatus,
    Tick,
    et_date,
    now_utc,
    to_et,
)

__all__ = ["DataQualityMonitor", "QualityReport", "SymbolQuality"]

# A symbol whose stream is this badly ordered is considered unreliable.
OUT_OF_ORDER_INVALID_THRESHOLD = 5

# How many recent accepted signatures are kept to catch replayed ticks.
_RECENT_SIGNATURE_WINDOW = 256


@dataclass(frozen=True, slots=True)
class SymbolQuality:
    """Quality snapshot for one symbol."""

    symbol: str
    status: DataStatus
    last_tick_utc: datetime | None
    stale_seconds: float | None
    tick_count: int
    duplicates: int
    out_of_order: int
    invalid: int
    last_reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "last_tick_utc": self.last_tick_utc.isoformat() if self.last_tick_utc else None,
            "stale_seconds": self.stale_seconds,
            "tick_count": self.tick_count,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "invalid": self.invalid,
            "last_reject_reason": self.last_reject_reason,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Aggregate + per-symbol data-quality report."""

    aggregate: DataStatus
    per_symbol: Mapping[str, SymbolQuality]
    checked_utc: datetime
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate.value,
            "per_symbol": {symbol: quality.to_dict() for symbol, quality in self.per_symbol.items()},
            "checked_utc": self.checked_utc.isoformat(),
            "reason": self.reason,
        }


@dataclass
class _SymbolState:
    last_tick_utc: datetime | None = None
    tick_count: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    invalid: int = 0
    last_reject_reason: str = ""
    trade_ids: set[int] = field(default_factory=set)
    recent_signatures: Deque[tuple[datetime, float, float]] = field(
        default_factory=lambda: deque(maxlen=_RECENT_SIGNATURE_WINDOW)
    )


class DataQualityMonitor:
    """Tracks per-symbol freshness/duplicates/order and reports a status."""

    def __init__(
        self,
        symbols: Sequence[str],
        data_config: DataConfig,
        clock: Any = now_utc,
    ) -> None:
        normalised = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
        if not normalised:
            raise ValueError("DataQualityMonitor needs at least one symbol")
        if len(set(normalised)) != len(normalised):
            raise ValueError("DataQualityMonitor symbols contain duplicates")
        self._symbols = normalised
        self._data = data_config
        self._clock = clock
        self._lock = threading.RLock()
        self._states: dict[str, _SymbolState] = {symbol: _SymbolState() for symbol in normalised}
        self._day: date | None = None
        self._ignored_symbols = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_tick(self, tick: Tick) -> None:
        """Record one websocket tick, applying duplicate/order/age filters."""
        if not isinstance(tick, Tick):
            raise TypeError(f"record_tick expects a Tick, got {type(tick)!r}")
        now = self._clock()
        symbol = tick.symbol

        with self._lock:
            if symbol not in self._states:
                self._ignored_symbols += 1
                return

            tick_day = et_date(tick.ts_utc)
            if self._day is not None and tick_day != self._day:
                self._reset_locked()
                self._day = tick_day
            elif self._day is None:
                self._day = tick_day

            state = self._states[symbol]  # (re)fetch after a possible rollover reset

            age_seconds = (now - tick.ts_utc).total_seconds()
            if age_seconds < -float(self._data.max_tick_age_seconds):
                state.invalid += 1
                state.last_reject_reason = "tick timestamp too far in the future"
                return
            if age_seconds > float(self._data.max_tick_age_seconds):
                state.invalid += 1
                state.last_reject_reason = "tick older than max_tick_age_seconds"
                return

            signature = (tick.ts_utc, tick.price, tick.volume)
            duplicate = signature in state.recent_signatures
            if not duplicate and tick.trade_id is not None:
                duplicate = tick.trade_id in state.trade_ids
            if duplicate:
                state.duplicates += 1
                state.last_reject_reason = "duplicate tick"
                return

            if state.last_tick_utc is not None and tick.ts_utc < state.last_tick_utc:
                state.out_of_order += 1
                state.last_reject_reason = "out-of-order tick"
                return

            state.last_tick_utc = tick.ts_utc
            state.tick_count += 1
            state.recent_signatures.append(signature)
            if tick.trade_id is not None:
                state.trade_ids.add(tick.trade_id)
            state.last_reject_reason = ""

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def report(self, now: datetime | None = None) -> QualityReport:
        """Build a quality report for the configured symbols."""
        checked = self._normalise_now(now if now is not None else self._clock())
        with self._lock:
            per_symbol: dict[str, SymbolQuality] = {}
            for symbol in self._symbols:
                per_symbol[symbol] = self._symbol_quality_locked(symbol, checked)

            aggregate = self._aggregate_locked(per_symbol)
            reason = self._aggregate_reason_locked(aggregate, per_symbol)
            return QualityReport(
                aggregate=aggregate,
                per_symbol=per_symbol,
                checked_utc=checked,
                reason=reason,
            )

    @property
    def aggregate_status(self) -> DataStatus:
        """Convenience shortcut used by the engine's data gate."""
        return self.report().aggregate

    @property
    def ignored_symbols(self) -> int:
        with self._lock:
            return self._ignored_symbols

    @property
    def et_day(self) -> date | None:
        with self._lock:
            return self._day

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all counters and freshness information."""
        with self._lock:
            self._reset_locked()
            self._day = None

    def reset_day(self, et_day: date | datetime) -> bool:
        """Roll over to a new ET trading day; returns True when state was reset."""
        day = self._coerce_et_day(et_day)
        with self._lock:
            if self._day == day:
                return False
            self._reset_locked()
            self._day = day
            return True

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _reset_locked(self) -> None:
        self._states = {symbol: _SymbolState() for symbol in self._symbols}

    def _symbol_quality_locked(self, symbol: str, checked: datetime) -> SymbolQuality:
        state = self._states[symbol]
        stale_seconds: float | None = None
        if state.last_tick_utc is not None:
            stale_seconds = max((checked - state.last_tick_utc).total_seconds(), 0.0)

        if state.tick_count == 0 and state.invalid > 0:
            status = DataStatus.INVALID
        elif state.out_of_order >= OUT_OF_ORDER_INVALID_THRESHOLD:
            status = DataStatus.INVALID
        elif state.last_tick_utc is None:
            status = DataStatus.STALE
        elif stale_seconds is not None and stale_seconds > float(self._data.staleness_seconds):
            status = DataStatus.STALE
        else:
            status = DataStatus.OK

        return SymbolQuality(
            symbol=symbol,
            status=status,
            last_tick_utc=state.last_tick_utc,
            stale_seconds=stale_seconds,
            tick_count=state.tick_count,
            duplicates=state.duplicates,
            out_of_order=state.out_of_order,
            invalid=state.invalid,
            last_reject_reason=state.last_reject_reason,
        )

    @staticmethod
    def _aggregate_locked(per_symbol: Mapping[str, SymbolQuality]) -> DataStatus:
        qualities = list(per_symbol.values())
        if any(quality.status is DataStatus.INVALID for quality in qualities):
            return DataStatus.INVALID
        if all(quality.status is DataStatus.STALE for quality in qualities):
            return DataStatus.STALE
        if any(quality.status is DataStatus.OK for quality in qualities):
            return DataStatus.OK
        return DataStatus.STALE

    @staticmethod
    def _aggregate_reason_locked(
        aggregate: DataStatus, per_symbol: Mapping[str, SymbolQuality]
    ) -> str:
        if aggregate is DataStatus.OK:
            return "live ticks are fresh for at least one symbol"
        stale = [
            symbol
            for symbol, quality in per_symbol.items()
            if quality.status is DataStatus.STALE
        ]
        invalid = [
            symbol
            for symbol, quality in per_symbol.items()
            if quality.status is DataStatus.INVALID
        ]
        if aggregate is DataStatus.INVALID:
            return "invalid tick stream for: " + ", ".join(invalid or list(per_symbol))
        return "no fresh trades for: " + ", ".join(stale or list(per_symbol))

    def _normalise_now(self, now: datetime) -> datetime:
        if not isinstance(now, datetime):
            raise TypeError(f"now must be a datetime, got {type(now)!r}")
        if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(UTC)

    @staticmethod
    def _coerce_et_day(value: date | datetime) -> date:
        if isinstance(value, datetime):
            return to_et(value).date()
        if isinstance(value, date):
            return value
        raise TypeError(f"et_day must be a date or datetime, got {type(value)!r}")
