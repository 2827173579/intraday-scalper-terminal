"""Lazy, defensive adapters for modules owned by other builders.

The backtester depends on three interfaces that are developed in parallel by
other builders:

* ``intraday_scalper.features.indicators`` — pure indicator functions;
* ``intraday_scalper.strategies.base.StrategyContext`` — the frozen dataclass
  handed to every strategy's ``evaluate()``;
* ``intraday_scalper.risk.sizing.size_position`` — position sizing.

None of these may become a hard import: the backtest engine must run (and its
tests must pass) while those modules are still in flight.  Every resolver here

1. tries the production module at *call time* (not import time),
2. validates that the expected public API exists,
3. falls back to a self-contained equivalent with the same contract.

Tests inject the built-in equivalents explicitly, so they are independent of
whatever state another builder's file happens to be in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from intraday_scalper.core import Bar, Mode, SessionPhase

from . import indicators as _builtin_indicators

__all__ = [
    "REQUIRED_INDICATORS",
    "FallbackStrategyContext",
    "builtin_indicators",
    "resolve_context_class",
    "resolve_indicators",
]

#: Public functions/attributes the engine actually uses from the indicator module.
REQUIRED_INDICATORS = (
    "ema",
    "rsi",
    "atr",
    "opening_range",
    "true_range",
    "rolling_mean",
    "volume_mean",
    "rolling_high",
    "rolling_low",
    "minutes_into_session",
)

#: The bundled fallback (also used directly by the backtest tests).
builtin_indicators = _builtin_indicators

_INDICATORS_CACHE: Any = None
_CONTEXT_CLASS_CACHE: Any = None


def _valid_indicators(module: Any) -> bool:
    return all(callable(getattr(module, name, None)) for name in REQUIRED_INDICATORS)


def resolve_indicators() -> Any:
    """Return the production indicator module if usable, else the builtin fallback.

    Resolution happens lazily on first use and the result is cached for the
    process; this keeps the import graph independent from the other builder.
    """

    global _INDICATORS_CACHE
    if _INDICATORS_CACHE is not None:
        return _INDICATORS_CACHE
    try:  # pragma: no cover - exercised only when features.* is importable
        from intraday_scalper.features import indicators as production

        if _valid_indicators(production):
            _INDICATORS_CACHE = production
            return production
    except Exception:  # noqa: BLE001 - optional integration must never crash
        pass
    _INDICATORS_CACHE = _builtin_indicators
    return _builtin_indicators


@dataclass(frozen=True, slots=True)
class FallbackStrategyContext:
    """Duck-typed stand-in for ``strategies.base.StrategyContext``.

    Field names and semantics are frozen by the cross-builder contract; the
    engine passes only *confirmed* information here:

    * ``now_utc`` is the close time of the last confirmed 1m bar;
    * ``bars_1m`` / ``bars_5m`` only contain completed regular-session bars;
    * ``vwap`` is the true session VWAP accumulated from completed bars;
    * every indicator is the value at the last confirmed bar (never later).
    """

    symbol: str
    now_utc: datetime
    session_phase: SessionPhase
    mode: Mode
    bars_1m: tuple[Bar, ...]
    bars_5m: tuple[Bar, ...]
    vwap: Optional[float]
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    rsi: Optional[float]
    atr: Optional[float]
    opening_range_5m: Any
    opening_range_15m: Any
    avg_volume_20: Optional[float]
    last_price: Optional[float]

    # -- helpers used by the strategies ---------------------------------- #
    def last(self, n: int = 1) -> Optional[Bar]:
        """The ``n``-th most recent confirmed 1m bar (``n=1`` is the latest)."""
        if n <= 0:
            raise ValueError("n must be >= 1")
        if len(self.bars_1m) < n:
            return None
        return self.bars_1m[-n]

    def prev(self) -> Optional[Bar]:
        """The bar immediately before the latest confirmed bar (or ``None``)."""
        return self.last(2)


def _valid_context_class(cls: Any) -> bool:
    required = {
        "symbol",
        "now_utc",
        "session_phase",
        "mode",
        "bars_1m",
        "bars_5m",
        "vwap",
        "ema_fast",
        "ema_slow",
        "rsi",
        "atr",
        "opening_range_5m",
        "opening_range_15m",
        "avg_volume_20",
        "last_price",
    }
    get_fields = getattr(cls, "__dataclass_fields__", None)
    if get_fields is None:
        return False
    return required.issubset(set(get_fields.keys()))


def resolve_context_class() -> Any:
    """Return the production ``StrategyContext`` class or the fallback stand-in."""

    global _CONTEXT_CLASS_CACHE
    if _CONTEXT_CLASS_CACHE is not None:
        return _CONTEXT_CLASS_CACHE
    try:  # pragma: no cover - exercised only when strategies.* is importable
        from intraday_scalper.strategies.base import StrategyContext

        if _valid_context_class(StrategyContext):
            _CONTEXT_CLASS_CACHE = StrategyContext
            return StrategyContext
    except Exception:  # noqa: BLE001 - optional integration must never crash
        pass
    _CONTEXT_CLASS_CACHE = FallbackStrategyContext
    return FallbackStrategyContext


def reset_caches() -> None:
    """Clear the lazy resolver caches (tests / long-lived processes)."""

    global _INDICATORS_CACHE, _CONTEXT_CLASS_CACHE
    _INDICATORS_CACHE = None
    _CONTEXT_CLASS_CACHE = None


def make_context(context_cls: Any, **kwargs: Any) -> Any:
    """Instantiate a strategy context, degrading to the fallback on mismatch.

    If the production class exists but has a different constructor signature
    than the frozen contract (mid-edit, version skew), falling back keeps the
    backtester runnable instead of crashing the whole report.
    """

    try:
        return context_cls(**kwargs)
    except TypeError:
        if context_cls is FallbackStrategyContext:
            raise
        return FallbackStrategyContext(**kwargs)
