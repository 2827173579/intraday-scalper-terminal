"""Shared strategy contract.

``StrategyContext`` is the immutable snapshot the signal engine hands to every
strategy for one symbol at one instant. It contains **only confirmed,
already-closed 1m bars** (oldest first, latest confirmed bar last) plus the
indicator values computed from those bars. A strategy must never look at a
later bar, a clock, the network or the filesystem.

``Strategy`` is the abstract base class. Each concrete strategy implements
``evaluate(context) -> SignalCandidate | None`` (long-only ETF trades: buying
SOXL expresses a bullish semis view, buying SOXS expresses a bearish one; the
system never shorts and never places orders).

The protected helper ``_build_candidate`` applies, in order:

1. the ATR-based structural stop distance
   (``structural_stop - atr * atr_stop_multiple``, where the multiple comes
   from the strategy's ``params.stop_atr_multiple``),
2. the ``min_stop_pct`` / ``max_stop_pct`` clamp around ``entry_ref``,
3. a final long-only sanity check (``stop < entry``).

If the result is not a valid long trade it returns ``None`` instead of raising,
so a strategy can never crash the signal engine on a degenerate bar.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence

from intraday_scalper.core import Bar, Mode, SessionPhase, SignalCandidate, StrategyName, to_utc
from intraday_scalper.features.indicators import OpeningRange

__all__ = ["Strategy", "StrategyContext"]


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Immutable per-symbol snapshot for one strategy evaluation.

    Attributes
    ----------
    symbol:
        Ticker the context was built for (``SOXL`` / ``SOXS``).
    now_utc:
        Time of the evaluation (latest confirmed bar end, or wall clock).
    session_phase / mode:
        Exchange-session phase and dual-track mode from ``core.models``.
    bars_1m:
        Confirmed regular-session 1m bars, oldest first. The **last** element
        is the latest confirmed bar; nothing after it may exist.
    bars_5m:
        Confirmed 5m bars (diagnostics / dashboard trend), oldest first.
    vwap:
        True session VWAP (``sum(price*volume)/sum(volume)``), never a proxy.
    ema_fast / ema_slow / rsi / atr:
        Indicator values computed on ``bars_1m`` (``None`` during warm-up).
    opening_range_5m / opening_range_15m:
        Opening ranges for the session, if they exist yet.
    avg_volume_20:
        Average volume of the (up to) 20 bars before the latest bar.
    last_price:
        Most recent price seen (equals the latest confirmed close in tests).
    """

    symbol: str
    now_utc: datetime
    session_phase: SessionPhase
    mode: Mode
    bars_1m: tuple[Bar, ...] = ()
    bars_5m: tuple[Bar, ...] = ()
    vwap: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    opening_range_5m: Optional[OpeningRange] = None
    opening_range_15m: Optional[OpeningRange] = None
    avg_volume_20: Optional[float] = None
    last_price: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("StrategyContext.symbol must be a non-empty string")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "now_utc", to_utc(self.now_utc))
        if not isinstance(self.session_phase, SessionPhase):
            object.__setattr__(self, "session_phase", SessionPhase(self.session_phase))
        if not isinstance(self.mode, Mode):
            object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "bars_1m", tuple(self.bars_1m))
        object.__setattr__(self, "bars_5m", tuple(self.bars_5m))
        _check_strictly_ascending(self.bars_1m, "bars_1m")
        _check_strictly_ascending(self.bars_5m, "bars_5m")

    # -- bar access ------------------------------------------------------ #
    def last(self, n: int = 1) -> Optional[Bar]:
        """The ``n``-th most recent confirmed 1m bar (``None`` if absent)."""
        if n <= 0:
            raise ValueError("n must be > 0")
        if len(self.bars_1m) < n:
            return None
        return self.bars_1m[-n]

    def prev(self) -> Optional[Bar]:
        """The bar immediately before the latest confirmed bar."""
        return self.last(2)

    @property
    def bar_count_1m(self) -> int:
        return len(self.bars_1m)

    def closes(self, n: Optional[int] = None) -> list[float]:
        """Closes of the last ``n`` bars (all bars when ``n`` is ``None``)."""
        bars = self.bars_1m if n is None else self.bars_1m[-n:]
        return [float(bar.close) for bar in bars]


def _check_strictly_ascending(bars: tuple[Bar, ...], name: str) -> None:
    for earlier, later in zip(bars, bars[1:]):
        if later.start_utc <= earlier.start_utc:
            raise ValueError(f"{name} must be ordered by start_utc (strictly ascending)")


# --------------------------------------------------------------------------- #
# Strategy base class
# --------------------------------------------------------------------------- #
class Strategy(ABC):
    """Deterministic, long-only, single-bar-confirmation strategy."""

    #: Filled in by every concrete strategy.
    name: StrategyName
    #: Concrete strategy parameter dataclass (from ``core.config``); used to
    #: read ``stop_atr_multiple`` in the shared candidate builder.
    params: Any = None

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> Optional[SignalCandidate]:
        """Return a candidate on the latest confirmed bar, or ``None``."""

    @property
    def atr_stop_multiple(self) -> float:
        """ATR multiplier for the structural stop (0.0 when params lack it)."""
        return float(getattr(self.params, "stop_atr_multiple", 0.0))

    # -- shared helpers -------------------------------------------------- #
    @staticmethod
    def _confirmed_bar(ctx: StrategyContext) -> Optional[Bar]:
        """Latest bar only if it exists and is confirmed (``is_complete``)."""
        bar = ctx.last()
        if bar is None or not bar.is_complete:
            return None
        return bar

    @staticmethod
    def _prior_average_volume(ctx: StrategyContext, lookback: int = 20) -> Optional[float]:
        """Average volume of the bars **before** the latest bar.

        Uses exactly the last ``lookback`` prior bars when that many exist
        (never the signal bar itself). Falls back to ``ctx.avg_volume_20``
        during the first minutes of a session, and then to the mean of the
        bars that do exist.
        """
        prior = ctx.bars_1m[:-1][-lookback:]
        if len(prior) >= lookback:
            total = sum(float(bar.volume) for bar in prior)
            average = total / lookback
            return average if average > 0.0 else None
        if ctx.avg_volume_20 is not None:
            value = float(ctx.avg_volume_20)
            return value if value > 0.0 else None
        if not prior:
            return None
        average = sum(float(bar.volume) for bar in prior) / len(prior)
        return average if average > 0.0 else None

    @staticmethod
    def _cumulative_vwap(bars: Sequence[Bar]) -> Optional[float]:
        """Volume-weighted VWAP of the given bars (true bar VWAPs), or ``None``.

        This is used to reconstruct "the VWAP as of the previous bar" from
        confirmed bars only; it never peeks at a future bar. Requires volume.
        """
        total_volume = 0.0
        total_notional = 0.0
        for bar in bars:
            volume = float(bar.volume)
            if volume <= 0.0:
                continue
            total_volume += volume
            total_notional += float(bar.vwap) * volume
        if total_volume <= 0.0:
            return None
        return total_notional / total_volume

    # -- candidate builder ----------------------------------------------- #
    def _build_candidate(
        self,
        symbol: str,
        entry_ref: float,
        structural_stop: float,
        target_r: float,
        atr: Optional[float],
        min_stop_pct: float,
        max_stop_pct: float,
        rationale: str,
        structural_note: str,
        now_utc: datetime,
        confirmation_bar_start_utc: datetime,
    ) -> Optional[SignalCandidate]:
        """Build a validated long candidate from a *structural* stop price.

        ``structural_stop`` is the raw structural level (for example
        ``min(opening_range.high, signal_bar.low)``). The ATR distance is
        subtracted here, then the result is clamped to
        ``[entry_ref * (1 - max_stop_pct), entry_ref * (1 - min_stop_pct)]``.
        Returns ``None`` for any non-finite / non-long / degenerate input.
        """
        try:
            entry = float(entry_ref)
            raw_stop = float(structural_stop)
            target = float(target_r)
            min_pct = float(min_stop_pct)
            max_pct = float(max_stop_pct)
        except (TypeError, ValueError):
            return None

        if not (math.isfinite(entry) and entry > 0.0):
            return None
        if not (math.isfinite(raw_stop) and raw_stop > 0.0):
            return None
        if not (math.isfinite(target) and target > 0.0):
            return None
        if not (math.isfinite(min_pct) and math.isfinite(max_pct)):
            return None
        if min_pct < 0.0 or max_pct < 0.0 or min_pct > max_pct or max_pct >= 1.0:
            raise ValueError(
                f"invalid stop clamps: min_stop_pct={min_pct} max_stop_pct={max_pct}"
            )

        atr_value: Optional[float] = None
        if atr is not None:
            try:
                candidate_atr = float(atr)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(candidate_atr) or candidate_atr < 0.0:
                return None
            atr_value = candidate_atr

        stop = raw_stop
        if atr_value is not None:
            stop = raw_stop - atr_value * self.atr_stop_multiple

        tightest = entry * (1.0 - min_pct)  # highest allowed stop price
        widest = entry * (1.0 - max_pct)  # lowest allowed stop price
        stop = min(max(stop, widest), tightest)

        if not math.isfinite(stop) or stop <= 0.0 or stop >= entry:
            return None

        try:
            return SignalCandidate(
                symbol=symbol,
                strategy=self.name,
                entry_ref=entry,
                stop_loss=stop,
                target_r=target,
                rationale=str(rationale),
                created_utc=to_utc(now_utc),
                confirmation_bar_start_utc=to_utc(confirmation_bar_start_utc),
                structural_note=str(structural_note),
            )
        except (TypeError, ValueError):
            return None
