"""Immutable dashboard/state snapshots produced by :class:`SignalEngine`.

The dashboard, the CLI and the tests all consume :class:`EngineSnapshot`.  It is
deliberately a *plain data* view of the engine state so that:

* it can be rendered by Rich without touching engine internals;
* it can be dumped to JSON (``to_dict``) for logging or for tests;
* nothing in it can be mutated by the display layer.

Timestamps stay timezone-aware UTC; conversion to ET/KST happens in the display
layer via :mod:`intraday_scalper.core.timeutil`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from ..core.models import (
    AlertEvent,
    ConnectionState,
    DataStatus,
    Mode,
    RiskPlan,
    SessionPhase,
    SignalCandidate,
)
from .compat import to_jsonable

__all__ = [
    "SymbolSnapshot",
    "PositionSnapshot",
    "EngineSnapshot",
]


@dataclass(frozen=True, slots=True)
class SymbolSnapshot:
    """Per-symbol view of the freshest indicators and data-age information."""

    symbol: str
    last_price: Optional[float] = None
    last_bar_utc: Optional[datetime] = None
    last_bar_end_utc: Optional[datetime] = None
    data_age_seconds: Optional[float] = None
    bars_1m_count: int = 0
    bars_5m_count: int = 0
    trend_1m: str = "flat"
    trend_5m: str = "flat"
    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    avg_volume_20: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "last_bar_utc": None if self.last_bar_utc is None else self.last_bar_utc.isoformat(),
            "last_bar_end_utc": (
                None if self.last_bar_end_utc is None else self.last_bar_end_utc.isoformat()
            ),
            "data_age_seconds": self.data_age_seconds,
            "bars_1m_count": self.bars_1m_count,
            "bars_5m_count": self.bars_5m_count,
            "trend_1m": self.trend_1m,
            "trend_5m": self.trend_5m,
            "vwap": self.vwap,
            "ema9": self.ema9,
            "ema21": self.ema21,
            "rsi": self.rsi,
            "atr": self.atr,
            "avg_volume_20": self.avg_volume_20,
        }


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Display view of the manually-held position (the system never fills)."""

    signal_id: str
    symbol: str
    strategy: str
    mode: str
    shares: float
    entry_price: float
    entry_price_signal: float
    entry_utc: datetime
    stop_loss: float
    take_profit: float
    last_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_r: Optional[float] = None
    open_risk: Optional[float] = None
    notional: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "mode": self.mode,
            "shares": self.shares,
            "entry_price": self.entry_price,
            "entry_price_signal": self.entry_price_signal,
            "entry_utc": self.entry_utc.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "last_price": self.last_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_r": self.unrealized_r,
            "open_risk": self.open_risk,
            "notional": self.notional,
        }


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Complete point-in-time engine state consumed by the Rich dashboard."""

    as_of_utc: datetime
    mode: Mode
    session_phase: SessionPhase
    symbols: Mapping[str, SymbolSnapshot]
    data_status: DataStatus = DataStatus.OK
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    position: Optional[PositionSnapshot] = None
    risk: Mapping[str, Any] = None  # type: ignore[assignment]
    entries_today: int = 0
    max_entries_per_day: int = 15
    trades_today: int = 0
    wins_today: int = 0
    stops_today: int = 0
    daily_pnl: float = 0.0
    daily_stopped: bool = False
    latest_signal: Optional[SignalCandidate] = None
    latest_plan: Optional[RiskPlan] = None
    latest_signal_id: Optional[str] = None
    strategy_names: tuple[str, ...] = ()
    events: tuple[AlertEvent, ...] = ()
    eod_close_utc: Optional[datetime] = None
    breakeven_hint_r: float = 1.2
    take_profit_hint_r: tuple[float, ...] = (2.0, 2.5)

    # -- convenience ------------------------------------------------------ #
    @property
    def data_ready(self) -> bool:
        return self.data_status is DataStatus.OK

    def symbol(self, symbol: str) -> Optional[SymbolSnapshot]:
        return self.symbols.get(str(symbol).strip().upper())

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_utc": self.as_of_utc.isoformat(),
            "mode": self.mode.value,
            "session_phase": self.session_phase.value,
            "data_status": self.data_status.value,
            "connection_state": self.connection_state.value,
            "symbols": {key: value.to_dict() for key, value in self.symbols.items()},
            "position": None if self.position is None else self.position.to_dict(),
            "risk": to_jsonable(dict(self.risk or {})),
            "entries_today": self.entries_today,
            "max_entries_per_day": self.max_entries_per_day,
            "trades_today": self.trades_today,
            "wins_today": self.wins_today,
            "stops_today": self.stops_today,
            "daily_pnl": self.daily_pnl,
            "daily_stopped": self.daily_stopped,
            "latest_signal": None if self.latest_signal is None else self.latest_signal.to_dict(),
            "latest_plan": None if self.latest_plan is None else self.latest_plan.to_dict(),
            "latest_signal_id": self.latest_signal_id,
            "strategy_names": list(self.strategy_names),
            "events": [event.to_dict() for event in self.events],
            "eod_close_utc": None if self.eod_close_utc is None else self.eod_close_utc.isoformat(),
            "breakeven_hint_r": self.breakeven_hint_r,
            "take_profit_hint_r": list(self.take_profit_hint_r),
        }
