"""Lazy adapters to sibling packages that are built in parallel.

The signal engine is specified against interfaces owned by other builders:

* builder B — ``intraday_scalper.features.indicators`` and
  ``intraday_scalper.strategies.base.StrategyContext``;
* builder C — ``intraday_scalper.risk.sizing.size_position``;
* builder A — a data-quality monitor object (duck-typed) and the session-VWAP
  callback injected into the engine.

Those packages may legitimately not exist yet while the engine and the dashboard
are developed and tested independently.  Everything here is therefore imported
**lazily** and every bridge degrades to a small, dependency-free fallback with
the same contract, so that:

* importing :mod:`intraday_scalper.signals.engine` never raises just because a
  sibling package is missing or temporarily broken;
* the engine tests run with fakes and need nothing from the other builders;
* once the real packages land they are picked up automatically — no edit and no
  re-import of this module's public API is required.

These fallbacks are deliberately conservative (they never read the future and
match the documented semantics); they are *not* a second implementation of the
strategies — only of display/plumbing indicators.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..core.config import RiskConfig
from ..core.models import Bar, DataStatus, RiskPlan, SignalCandidate

__all__ = [
    "compute_atr",
    "compute_ema",
    "compute_opening_range",
    "compute_rsi",
    "compute_volume_mean",
    "data_quality_status",
    "get_indicators",
    "get_opening_range_class",
    "get_strategy_context_class",
    "FallbackOpeningRange",
    "FallbackStrategyContext",
    "build_strategy_context",
    "get_size_position",
    "size_position",
    "last_value",
    "to_jsonable",
]


# --------------------------------------------------------------------------- #
# Data-quality monitor bridge (builder A)
# --------------------------------------------------------------------------- #
def data_quality_status(data_quality: Any, now: datetime) -> Optional[DataStatus]:
    """Ask a (duck-typed) data-quality monitor for the aggregate status.

    Supports the documented ``report(now) -> QualityReport`` contract with an
    ``aggregate: DataStatus`` attribute, plus the common convenience shapes
    (``status``/``aggregate`` on the monitor itself).  Returns ``None`` when the
    monitor cannot be interpreted; a broken monitor is treated as
    ``DATA_UNAVAILABLE`` because silently trusting bad data is worse.
    """
    if data_quality is None:
        return None

    report_fn = getattr(data_quality, "report", None)
    candidate_obj: Any = data_quality
    if callable(report_fn):
        try:
            candidate_obj = report_fn(now)
        except Exception:  # noqa: BLE001 - a broken monitor must not crash the engine
            return DataStatus.UNAVAILABLE

    for attr in ("aggregate", "status", "data_status"):
        raw = getattr(candidate_obj, attr, None)
        if raw is None and attr != "aggregate":
            raw = getattr(data_quality, attr, None)
        if isinstance(raw, DataStatus):
            return raw
        if isinstance(raw, str):
            try:
                return DataStatus(raw.strip().upper())
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Indicators bridge (builder B)
# --------------------------------------------------------------------------- #
_INDICATORS: Any = None
_INDICATORS_TRIED = False


def get_indicators() -> Any:
    """Return builder B's ``features.indicators`` module, or ``None``."""
    global _INDICATORS, _INDICATORS_TRIED
    if not _INDICATORS_TRIED:
        _INDICATORS_TRIED = True
        try:
            _INDICATORS = importlib.import_module("intraday_scalper.features.indicators")
        except Exception:  # noqa: BLE001 - module may not exist yet
            _INDICATORS = None
    return _INDICATORS


def last_value(series: Any) -> Optional[float]:
    """Last non-``None`` value of an equal-length indicator series."""
    if series is None:
        return None
    try:
        values = list(series)
    except TypeError:
        return None
    for value in reversed(values):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    return None


def _call_with_fallback(function_name: str, fallback: Any, *args: Any) -> Any:
    indicators = get_indicators()
    if indicators is not None:
        fn = getattr(indicators, function_name, None)
        if callable(fn):
            try:
                return fn(*args)
            except Exception:  # noqa: BLE001 - fall back rather than crash display
                pass
    return fallback(*args)


def compute_ema(values: Sequence[float], period: int) -> list[Optional[float]]:
    return _call_with_fallback("ema", _fallback_ema, list(values), int(period))


def compute_rsi(values: Sequence[float], period: int = 14) -> list[Optional[float]]:
    return _call_with_fallback("rsi", _fallback_rsi, list(values), int(period))


def compute_atr(bars: Sequence[Bar], period: int = 14) -> list[Optional[float]]:
    return _call_with_fallback("atr", _fallback_atr, list(bars), int(period))


def compute_volume_mean(bars: Sequence[Bar], period: int = 20) -> list[Optional[float]]:
    return _call_with_fallback("volume_mean", _fallback_volume_mean, list(bars), int(period))


def compute_opening_range(
    bars: Sequence[Bar], session_open_utc: datetime, minutes: int
) -> "FallbackOpeningRange":
    return _call_with_fallback(
        "opening_range", _fallback_opening_range, list(bars), session_open_utc, int(minutes)
    )


# -- fallback indicator implementations (same documented semantics) ---------- #
def _fallback_ema(values: Sequence[float], period: int) -> list[Optional[float]]:
    values = [None if v is None else float(v) for v in values]
    n = len(values)
    out: list[Optional[float]] = [None] * n
    if period <= 0 or n < period:
        return out
    seed = sum(value for value in values[:period] if value is not None) / period
    out[period - 1] = seed
    multiplier = 2.0 / (period + 1.0)
    previous = seed
    for index in range(period, n):
        value = values[index]
        if value is None:
            out[index] = previous
            continue
        previous = value * multiplier + previous * (1.0 - multiplier)
        out[index] = previous
    return out


def _fallback_true_range(bars: Sequence[Bar]) -> list[float]:
    out: list[float] = []
    for index, bar in enumerate(bars):
        if index == 0:
            out.append(float(bar.high - bar.low))
            continue
        previous_close = float(bars[index - 1].close)
        out.append(
            max(
                float(bar.high - bar.low),
                abs(float(bar.high) - previous_close),
                abs(float(bar.low) - previous_close),
            )
        )
    return out


def _fallback_atr(bars: Sequence[Bar], period: int = 14) -> list[Optional[float]]:
    n = len(bars)
    out: list[Optional[float]] = [None] * n
    if period <= 0 or n < period:
        return out
    true_ranges = _fallback_true_range(bars)
    seed = sum(true_ranges[:period]) / period
    out[period - 1] = seed
    previous = seed
    for index in range(period, n):
        previous = (previous * (period - 1) + true_ranges[index]) / period
        out[index] = previous
    return out


def _fallback_rsi(values: Sequence[float], period: int = 14) -> list[Optional[float]]:
    numbers = [float(v) for v in values]
    n = len(numbers)
    out: list[Optional[float]] = [None] * n
    if period <= 0 or n <= period:
        return out
    gains = 0.0
    losses = 0.0
    for index in range(1, period + 1):
        change = numbers[index] - numbers[index - 1]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period

    def _rsi(gain: float, loss: float) -> float:
        if gain == 0.0 and loss == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rsi(avg_gain, avg_loss)
    for index in range(period + 1, n):
        change = numbers[index] - numbers[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[index] = _rsi(avg_gain, avg_loss)
    return out


def _fallback_volume_mean(bars: Sequence[Bar], period: int = 20) -> list[Optional[float]]:
    volumes = [float(bar.volume) for bar in bars]
    n = len(volumes)
    out: list[Optional[float]] = [None] * n
    if period <= 0:
        return out
    for index in range(period - 1, n):
        window = volumes[index - period + 1 : index + 1]
        out[index] = sum(window) / period
    return out


@dataclass(frozen=True, slots=True)
class FallbackOpeningRange:
    """Stand-in for :class:`features.indicators.OpeningRange`."""

    high: Optional[float]
    low: Optional[float]
    start_utc: datetime
    end_utc: datetime
    complete: bool
    bar_count: int

    @property
    def range(self) -> Optional[float]:
        if self.high is None or self.low is None:
            return None
        return self.high - self.low


def _fallback_opening_range(
    bars: Sequence[Bar], session_open_utc: datetime, minutes: int
) -> FallbackOpeningRange:
    end_utc = session_open_utc + timedelta(minutes=minutes)
    window = [
        bar
        for bar in bars
        if bar.is_complete and session_open_utc <= bar.start_utc < end_utc
    ]
    complete = len(window) >= minutes
    if not window:
        return FallbackOpeningRange(
            high=None,
            low=None,
            start_utc=session_open_utc,
            end_utc=end_utc,
            complete=False,
            bar_count=0,
        )
    return FallbackOpeningRange(
        high=max(float(bar.high) for bar in window),
        low=min(float(bar.low) for bar in window),
        start_utc=session_open_utc,
        end_utc=end_utc,
        complete=complete,
        bar_count=len(window),
    )


# --------------------------------------------------------------------------- #
# StrategyContext bridge (builder B)
# --------------------------------------------------------------------------- #
_STRATEGY_CONTEXT: Any = None
_STRATEGY_CONTEXT_TRIED = False


@dataclass(frozen=True, slots=True)
class FallbackStrategyContext:
    """Stand-in for :class:`strategies.base.StrategyContext` (duck-typed)."""

    symbol: str
    now_utc: datetime
    session_phase: Any
    mode: Any
    bars_1m: tuple
    bars_5m: tuple
    vwap: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    opening_range_5m: Any = None
    opening_range_15m: Any = None
    avg_volume_20: Optional[float] = None
    last_price: Optional[float] = None

    def last(self, n: int = 1) -> Optional[Bar]:
        if n <= 0 or len(self.bars_1m) < n:
            return None
        return self.bars_1m[-n]

    def prev(self) -> Optional[Bar]:
        return self.last(2)


def get_strategy_context_class() -> Any:
    """Return builder B's ``StrategyContext`` class, or the local fallback."""
    global _STRATEGY_CONTEXT, _STRATEGY_CONTEXT_TRIED
    if not _STRATEGY_CONTEXT_TRIED:
        _STRATEGY_CONTEXT_TRIED = True
        try:
            module = importlib.import_module("intraday_scalper.strategies.base")
            _STRATEGY_CONTEXT = getattr(module, "StrategyContext", None)
        except Exception:  # noqa: BLE001 - module may not exist yet
            _STRATEGY_CONTEXT = None
    return _STRATEGY_CONTEXT if _STRATEGY_CONTEXT is not None else FallbackStrategyContext


def build_strategy_context(**kwargs: Any) -> Any:
    """Instantiate B's context when compatible, else the duck-type fallback."""
    context_class = get_strategy_context_class()
    try:
        return context_class(**kwargs)
    except TypeError:
        return FallbackStrategyContext(**kwargs)


def get_opening_range_class() -> Any:
    indicators = get_indicators()
    if indicators is not None:
        candidate = getattr(indicators, "OpeningRange", None)
        if isinstance(candidate, type):
            return candidate
    return FallbackOpeningRange


# --------------------------------------------------------------------------- #
# Position sizing bridge (builder C)
# --------------------------------------------------------------------------- #
_SIZING: Any = None
_SIZING_TRIED = False


def get_size_position() -> Any:
    """Return builder C's ``risk.sizing.size_position``, or the fallback."""
    global _SIZING, _SIZING_TRIED
    if not _SIZING_TRIED:
        _SIZING_TRIED = True
        try:
            module = importlib.import_module("intraday_scalper.risk.sizing")
            _SIZING = getattr(module, "size_position", None)
        except Exception:  # noqa: BLE001 - module may not exist yet
            _SIZING = None
    return _SIZING if _SIZING is not None else _fallback_size_position


def size_position(
    candidate: SignalCandidate, risk_config: RiskConfig, *, equity: Optional[float] = None
) -> RiskPlan:
    """Session-size a candidate using C's sizing when available."""
    sizing = get_size_position()
    try:
        return sizing(candidate, risk_config, equity=equity)
    except TypeError:
        return sizing(candidate, risk_config)


def _fallback_size_position(
    candidate: SignalCandidate, risk_config: RiskConfig, *, equity: Optional[float] = None
) -> RiskPlan:
    """Mirror of the documented sizing formula (used until builder C lands)."""
    entry = float(candidate.entry_ref)
    risk_per_share = float(candidate.risk_per_share)
    if entry <= 0 or risk_per_share <= 0:
        shares = 0.0
        notional_qty = 0.0
        risk_qty = 0.0
    else:
        risk_qty = risk_config.max_risk_per_trade / risk_per_share
        notional_qty = risk_config.max_notional_per_trade / entry
        raw = max(min(risk_qty, notional_qty), 0.0)
        if risk_config.allow_fractional_shares:
            precision = max(int(risk_config.fractional_precision), 0)
            factor = 10.0**precision
            shares = math.floor(raw * factor) / factor
        else:
            # Integer shares always round *down* so risk is never increased.
            shares = float(math.floor(raw))
    notional = shares * entry
    theoretical_risk = shares * risk_per_share
    est_slippage = (
        shares
        * (entry + float(candidate.stop_loss))
        * float(risk_config.est_slippage_bps)
        / 10_000.0
    )
    est_fees = shares * float(risk_config.commission_per_share)
    return RiskPlan(
        candidate=candidate,
        shares=shares,
        notional=notional,
        theoretical_risk=theoretical_risk,
        est_slippage_dollars=est_slippage,
        est_fees_dollars=est_fees,
        actual_risk=theoretical_risk + est_slippage + est_fees,
        clipped_by_notional=notional_qty < risk_qty,
        clipped_by_risk=risk_qty <= notional_qty,
        allow_fractional=bool(risk_config.allow_fractional_shares),
    )


# --------------------------------------------------------------------------- #
# Generic serialisation helper
# --------------------------------------------------------------------------- #
def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses / enums / datetimes to JSON-safe data."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    return value
