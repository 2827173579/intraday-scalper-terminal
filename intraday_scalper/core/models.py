"""Immutable data contracts for the intraday scalper.

Design rules
------------
* All timestamps are timezone-aware *UTC* ``datetime`` objects.
* Every financial value is a ``float`` that is validated to be finite; prices
  must be strictly positive, volumes non-negative.
* These records are immutable (``frozen=True``) so they can be safely shared
  between the websocket thread, the aggregation thread and the backtester.
* No I/O, no network, no clock reads here: pure data structures only.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Optional

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Mode(str, Enum):
    """Dual-track state machine mode for SOXL / SOXS."""

    LONG_SOXL = "long_soxl"
    LONG_SOXS = "long_soxs"
    DUAL_WATCH = "dual_watch"


class DataStatus(str, Enum):
    """Outcome of the data-quality monitor."""

    OK = "OK"
    STALE = "STALE"
    UNAVAILABLE = "DATA_UNAVAILABLE"
    INVALID = "INVALID"


class ConnectionState(str, Enum):
    """Finnhub websocket connection lifecycle."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    UNAVAILABLE = "UNAVAILABLE"


class SessionPhase(str, Enum):
    """US equity session phase, evaluated in America/New_York."""

    CLOSED = "CLOSED"
    PREMARKET = "PREMARKET"
    OPENING_RANGE_5 = "OPENING_RANGE_5"
    OPENING_RANGE_15 = "OPENING_RANGE_15"
    REGULAR = "REGULAR"
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"  # >= 15:30 ET
    EOD_CHECK = "EOD_CHECK"  # >= 15:50 ET
    EOD_URGENT = "EOD_URGENT"  # >= 15:55 ET
    AFTER_HOURS = "AFTER_HOURS"


class StrategyName(str, Enum):
    """Deterministic strategies implemented in phase 1."""

    ORB_5M = "ORB_5M"
    ORB_15M = "ORB_15M"
    VWAP_RECLAIM = "VWAP_RECLAIM"
    EMA_PULLBACK = "EMA_PULLBACK"


class SignalEventKind(str, Enum):
    """Kinds of events the signal engine emits to the dashboard / journal."""

    ENTRY_SIGNAL = "ENTRY_SIGNAL"
    BREAKEVEN_HINT = "BREAKEVEN_HINT"
    TAKE_PROFIT_HINT = "TAKE_PROFIT_HINT"
    TRAILING_HINT = "TRAILING_HINT"
    STOP_HIT_HINT = "STOP_HIT_HINT"
    RISK_BLOCK = "RISK_BLOCK"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    EOD_REMINDER = "EOD_REMINDER"
    STATE_CHANGE = "STATE_CHANGE"
    INFO = "INFO"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RiskBlockReason(str, Enum):
    """Why a candidate entry was rejected by the risk manager."""

    NONE = "NONE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_ENTRIES_REACHED = "MAX_ENTRIES_REACHED"
    CONSECUTIVE_LOSS_COOLDOWN = "CONSECUTIVE_LOSS_COOLDOWN"
    SWITCH_COOLDOWN = "SWITCH_COOLDOWN"
    NO_NEW_ENTRIES_TIME = "NO_NEW_ENTRIES_TIME"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    DUAL_CONFLICT = "DUAL_CONFLICT"
    DATA_NOT_READY = "DATA_NOT_READY"
    NOT_TRADING_DAY = "NOT_TRADING_DAY"
    ZERO_SHARES = "ZERO_SHARES"
    REWARD_RISK_TOO_LOW = "REWARD_RISK_TOO_LOW"
    INVALID_STOP = "INVALID_STOP"


class TradeStatus(str, Enum):
    PENDING = "PENDING"  # signal emitted, waiting for manual ENTRY
    OPEN = "OPEN"  # manually confirmed entry
    CLOSED = "CLOSED"  # manually confirmed exit
    STOPPED = "STOPPED"  # manually confirmed stop
    CANCELLED = "CANCELLED"


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _require_aware_utc(ts: datetime, name: str) -> datetime:
    if not isinstance(ts, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(ts)!r}")
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"{name} must be timezone-aware (UTC), got naive {ts!r}")
    return ts.astimezone(UTC)


def _require_finite(value: float, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _require_positive(value: float, name: str) -> float:
    v = _require_finite(value, name)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def _require_non_negative(value: float, name: str) -> float:
    v = _require_finite(value, name)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v!r}")
    return v


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Tick:
    """One websocket trade event (a single print), normalised to UTC."""

    symbol: str
    price: float
    volume: float
    ts_utc: datetime
    trade_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("Tick.symbol must be a non-empty string")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "price", _require_positive(self.price, "Tick.price"))
        object.__setattr__(self, "volume", _require_non_negative(self.volume, "Tick.volume"))
        object.__setattr__(self, "ts_utc", _require_aware_utc(self.ts_utc, "Tick.ts_utc"))

    @property
    def price_volume(self) -> float:
        return self.price * self.volume

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "ts_utc": self.ts_utc.isoformat(),
            "trade_id": self.trade_id,
        }


@dataclass(frozen=True, slots=True)
class Bar:
    """A completed (or, while open, explicitly partial) OHLCV bar.

    ``vwap`` is a *true* volume-weighted average price computed from ticks
    (1m) or from the constituent 1m bars (5m). It is never a proxy built from
    histogram/high/low values.
    """

    symbol: str
    timeframe_minutes: int
    start_utc: datetime
    end_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    trade_count: int = 0
    is_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("Bar.symbol must be a non-empty string")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if int(self.timeframe_minutes) <= 0:
            raise ValueError("Bar.timeframe_minutes must be > 0")
        object.__setattr__(self, "timeframe_minutes", int(self.timeframe_minutes))
        object.__setattr__(self, "start_utc", _require_aware_utc(self.start_utc, "Bar.start_utc"))
        object.__setattr__(self, "end_utc", _require_aware_utc(self.end_utc, "Bar.end_utc"))
        if self.end_utc <= self.start_utc:
            raise ValueError("Bar.end_utc must be after Bar.start_utc")
        expected = timedelta(minutes=self.timeframe_minutes)
        if self.end_utc - self.start_utc != expected:
            raise ValueError(
                f"Bar window must be exactly {self.timeframe_minutes}m, "
                f"got {self.end_utc - self.start_utc}"
            )
        o = _require_positive(self.open, "Bar.open")
        h = _require_positive(self.high, "Bar.high")
        lo = _require_positive(self.low, "Bar.low")
        c = _require_positive(self.close, "Bar.close")
        if h < max(o, c, lo) or lo > min(o, c, h):
            raise ValueError(f"Bar OHLC inconsistent: o={o} h={h} l={lo} c={c}")
        v = _require_non_negative(self.volume, "Bar.volume")
        vw = _require_positive(self.vwap, "Bar.vwap")
        if not (lo <= vw <= h):
            raise ValueError(f"Bar.vwap {vw} outside [low={lo}, high={h}]")
        if int(self.trade_count) < 0:
            raise ValueError("Bar.trade_count must be >= 0")
        object.__setattr__(self, "open", o)
        object.__setattr__(self, "high", h)
        object.__setattr__(self, "low", lo)
        object.__setattr__(self, "close", c)
        object.__setattr__(self, "volume", v)
        object.__setattr__(self, "vwap", vw)
        object.__setattr__(self, "trade_count", int(self.trade_count))

    # -- derived ---------------------------------------------------------- #
    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_ratio(self) -> float:
        """Body size relative to the full range (0 when the range is 0)."""
        rng = self.range
        return self.body / rng if rng > 0 else 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def notional(self) -> float:
        return self.vwap * self.volume

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe_minutes": self.timeframe_minutes,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
            "is_complete": self.is_complete,
        }


# --------------------------------------------------------------------------- #
# Signals & risk
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SignalCandidate:
    """A deterministic strategy signal *before* position sizing.

    ``stop_loss`` must be below ``entry_ref`` (these are long-only ETF
    trades: buying SOXL expresses bullish semis, buying SOXS expresses
    bearish semis; neither is a short sale).
    """

    symbol: str
    strategy: StrategyName
    entry_ref: float
    stop_loss: float
    target_r: float
    rationale: str
    created_utc: datetime
    confirmation_bar_start_utc: datetime
    structural_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not isinstance(self.strategy, StrategyName):
            object.__setattr__(self, "strategy", StrategyName(self.strategy))
        e = _require_positive(self.entry_ref, "SignalCandidate.entry_ref")
        s = _require_positive(self.stop_loss, "SignalCandidate.stop_loss")
        if s >= e:
            raise ValueError(
                f"Long-only ETF trade requires stop_loss < entry_ref, got stop={s} entry={e}"
            )
        tr = _require_positive(self.target_r, "SignalCandidate.target_r")
        object.__setattr__(self, "entry_ref", e)
        object.__setattr__(self, "stop_loss", s)
        object.__setattr__(self, "target_r", tr)
        object.__setattr__(
            self, "created_utc", _require_aware_utc(self.created_utc, "SignalCandidate.created_utc")
        )
        object.__setattr__(
            self,
            "confirmation_bar_start_utc",
            _require_aware_utc(
                self.confirmation_bar_start_utc, "SignalCandidate.confirmation_bar_start_utc"
            ),
        )
        object.__setattr__(self, "rationale", str(self.rationale))

    @property
    def risk_per_share(self) -> float:
        return self.entry_ref - self.stop_loss

    @property
    def take_profit_price(self) -> float:
        return self.entry_ref + self.target_r * self.risk_per_share

    @property
    def reward_risk(self) -> float:
        return self.target_r

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy.value,
            "entry_ref": self.entry_ref,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit_price,
            "target_r": self.target_r,
            "risk_per_share": self.risk_per_share,
            "rationale": self.rationale,
            "structural_note": self.structural_note,
            "created_utc": self.created_utc.isoformat(),
            "confirmation_bar_start_utc": self.confirmation_bar_start_utc.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RiskPlan:
    """Result of position sizing for one candidate, including real costs."""

    candidate: SignalCandidate
    shares: float
    notional: float
    theoretical_risk: float
    est_slippage_dollars: float
    est_fees_dollars: float
    actual_risk: float
    clipped_by_notional: bool
    clipped_by_risk: bool
    allow_fractional: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "shares", _require_non_negative(self.shares, "RiskPlan.shares"))
        for name in ("notional", "theoretical_risk", "est_slippage_dollars", "est_fees_dollars", "actual_risk"):
            object.__setattr__(self, name, _require_non_negative(getattr(self, name), f"RiskPlan.{name}"))

    @property
    def valid(self) -> bool:
        return self.shares > 0 and not math.isclose(self.shares, 0.0, abs_tol=1e-12)

    @property
    def reward_dollars(self) -> float:
        return self.candidate.target_r * self.actual_risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "shares": self.shares,
            "notional": self.notional,
            "theoretical_risk": self.theoretical_risk,
            "est_slippage_dollars": self.est_slippage_dollars,
            "est_fees_dollars": self.est_fees_dollars,
            "actual_risk": self.actual_risk,
            "clipped_by_notional": self.clipped_by_notional,
            "clipped_by_risk": self.clipped_by_risk,
            "allow_fractional": self.allow_fractional,
            "entry_ref": self.candidate.entry_ref,
            "stop_loss": self.candidate.stop_loss,
            "take_profit": self.candidate.take_profit_price,
            "risk_per_share": self.candidate.risk_per_share,
            "target_r": self.candidate.target_r,
        }


@dataclass
class Position:
    """A manually confirmed open position (the system never fills orders)."""

    signal_id: str
    symbol: str
    strategy: StrategyName
    mode: Mode
    shares: float
    entry_price: float
    entry_utc: datetime
    stop_loss: float
    take_profit: float
    entry_price_signal: float = 0.0

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).strip().upper()
        self.shares = _require_positive(self.shares, "Position.shares")
        self.entry_price = _require_positive(self.entry_price, "Position.entry_price")
        self.stop_loss = _require_positive(self.stop_loss, "Position.stop_loss")
        self.take_profit = _require_positive(self.take_profit, "Position.take_profit")
        self.entry_utc = _require_aware_utc(self.entry_utc, "Position.entry_utc")
        if self.entry_price_signal <= 0:
            self.entry_price_signal = self.entry_price

    @property
    def is_long_soxs(self) -> bool:
        return self.symbol == "SOXS"

    @property
    def open_risk(self) -> float:
        return max(self.entry_price - self.stop_loss, 0.0) * self.shares

    @property
    def notional(self) -> float:
        return self.entry_price * self.shares

    @property
    def entry_slippage_bps(self) -> float:
        if self.entry_price_signal <= 0:
            return 0.0
        return (self.entry_price - self.entry_price_signal) / self.entry_price_signal * 10_000.0

    def unrealized_pnl(self, price: float) -> float:
        return (float(price) - self.entry_price) * self.shares

    def unrealized_r(self, price: float) -> float:
        risk = self.open_risk
        return self.unrealized_pnl(price) / risk if risk > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_price_signal": self.entry_price_signal,
            "entry_utc": self.entry_utc.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "notional": self.notional,
            "open_risk": self.open_risk,
        }


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Closed or historical trade, mirrored into the SQLite journal."""

    signal_id: str
    symbol: str
    strategy: StrategyName
    mode: Mode
    shares: float
    entry_time_utc: datetime
    entry_price_signal: float
    entry_price_fill: float
    stop_price: float
    target_price: float
    exit_time_utc: Optional[datetime] = None
    exit_price_fill: Optional[float] = None
    exit_reason: Optional[str] = None
    fees: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    trade_id: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "entry_time_utc", _require_aware_utc(self.entry_time_utc, "TradeRecord.entry_time_utc"))
        object.__setattr__(self, "entry_price_signal", _require_positive(self.entry_price_signal, "entry_price_signal"))
        object.__setattr__(self, "entry_price_fill", _require_positive(self.entry_price_fill, "entry_price_fill"))
        object.__setattr__(self, "stop_price", _require_positive(self.stop_price, "stop_price"))
        object.__setattr__(self, "target_price", _require_positive(self.target_price, "target_price"))
        object.__setattr__(self, "shares", _require_positive(self.shares, "shares"))
        if self.exit_time_utc is not None:
            object.__setattr__(self, "exit_time_utc", _require_aware_utc(self.exit_time_utc, "exit_time_utc"))
        if self.exit_price_fill is not None:
            object.__setattr__(self, "exit_price_fill", _require_positive(self.exit_price_fill, "exit_price_fill"))
        object.__setattr__(self, "fees", _require_non_negative(self.fees, "fees"))

    @property
    def is_closed(self) -> bool:
        return self.exit_price_fill is not None and self.exit_time_utc is not None

    @property
    def pnl(self) -> float:
        if not self.is_closed:
            return 0.0
        assert self.exit_price_fill is not None
        gross = (self.exit_price_fill - self.entry_price_fill) * self.shares
        return gross - self.fees

    @property
    def risk_dollars(self) -> float:
        return max(self.entry_price_fill - self.stop_price, 0.0) * self.shares

    @property
    def r_multiple(self) -> float:
        risk = self.risk_dollars
        return self.pnl / risk if risk > 0 else 0.0

    @property
    def holding_minutes(self) -> float:
        if not self.is_closed or self.exit_time_utc is None:
            return 0.0
        return (self.exit_time_utc - self.entry_time_utc).total_seconds() / 60.0

    @property
    def entry_slippage_bps(self) -> float:
        return (self.entry_price_fill - self.entry_price_signal) / self.entry_price_signal * 10_000.0

    @property
    def exit_slippage_bps(self) -> float:
        if not self.is_closed or self.exit_price_fill is None:
            return 0.0
        # Exit is a sell: a lower fill than the planned price is negative slippage.
        return (self.exit_price_signal_or_fill - self.exit_price_fill) / self.exit_price_signal_or_fill * 10_000.0

    @property
    def exit_price_signal_or_fill(self) -> float:
        return self.exit_price_fill if self.exit_price_fill is not None else 0.0


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AlertEvent:
    """A user-facing event (sound + dashboard + journal)."""

    kind: SignalEventKind
    level: AlertLevel
    message: str
    ts_utc: datetime
    symbol: Optional[str] = None
    signal_id: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SignalEventKind):
            raise TypeError("AlertEvent.kind must be a SignalEventKind")
        if not isinstance(self.level, AlertLevel):
            raise TypeError("AlertEvent.level must be an AlertLevel")
        object.__setattr__(self, "ts_utc", _require_aware_utc(self.ts_utc, "AlertEvent.ts_utc"))
        if self.symbol is not None:
            object.__setattr__(self, "symbol", str(self.symbol).strip().upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "level": self.level.value,
            "message": self.message,
            "ts_utc": self.ts_utc.isoformat(),
            "symbol": self.symbol,
            "signal_id": self.signal_id,
            "details": dict(self.details),
        }


def new_signal_id(now: Optional[datetime] = None) -> str:
    """Stable-enough unique id for a candidate signal / manual trade."""
    compact = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    return f"SIG-{compact}-{uuid.uuid4().hex[:6].upper()}"
