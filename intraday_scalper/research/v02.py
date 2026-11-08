"""V0.2 structural research backtest — 5-minute driven, mirror-regime filtered.

Why this module exists
----------------------
V0.1 (``tools/run_real_backtest.py``) is a *1-minute driven* event backtest: it
evaluates the three phase-1 strategies on every confirmed 1m bar of every
symbol and produced 2,687 losing trades over 2024-01..2026-11.  The V0.2
research hypothesis is that most of that loss is **microstructure noise**: the
1m signal fires on intrabar noise that disappears once the bar is closed.

V0.2 therefore changes *structure only* (not parameters):

1. signals are evaluated on **completed 5-minute bars** (aggregated from the
   existing 1m bars, anchored at the 09:30 ET session open);
2. the ORB_5M strategy and every 1m-triggered strategy are **disabled**; only
   three 5m strategies run (ORB_15M_5M, VWAP_RECLAIM_5M, EMA_PULLBACK_5M);
3. a **mirror-regime filter** requires the *other* ETF (SOXL <-> SOXS) to be
   **below its own session VWAP** (both directions: the inverse leg must not be
   confirming the opposite side; the SOXS-side ">" in the original task text was
   corrected as a spec typo, not a parameter change);
4. **IEX activity adaptation**: absolute-volume filters are replaced by
   ATR range expansion and IEX-scoped relative trade-count activity, because
   IEX is a single venue with a small, noisy share of consolidated volume;
5. fills happen at the **open of the next 5m bar**, exits are checked on every
   constituent **1m bar** (conservative: try 5m structure, keep 1m stops), and
   any position is force-flattened on the session's last 1m bar (EOD_EXIT).

All numeric thresholds are **one-shot a-priori values from the V0.2 research
spec**; none was tuned (let alone re-tuned per walk-forward window) on any
train/test fold.  See the ``# a-priori, not tuned`` markers below.

Data caveat (must be repeated in every report)
----------------------------------------------
The only real dataset available is Alpaca **IEX-only** 1m bars.  IEX is a
single venue (a small share of US consolidated volume), so volume, VWAP and
relative activity are IEX-scoped — **not consolidated tape**.  Results are a
structural comparison against V0.1 on the same data; they are not proof of
profitability.

Isolation
---------
This module only *imports* the frozen production contracts
(``core``/``risk``/``backtest``/``features``); it never modifies them.  It is
not imported by the live system.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import median
from typing import Any, Callable, Mapping, Sequence

from intraday_scalper.backtest.metrics import BacktestMetrics, compute_metrics
from intraday_scalper.backtest.walkforward import (
    Fold,
    assert_no_leakage,
    walk_forward_splits,
)
from intraday_scalper.core import (
    AppConfig,
    Bar,
    RiskPlan,
    SignalCandidate,
    StrategyName,
    TradeRecord,
    et_date,
    is_regular_session,
    load_config,
    session_close_utc,
    session_open_utc,
)
from intraday_scalper.features.indicators import ema as _ema_series
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.risk.sizing import size_position

__all__ = [
    "ATR_PERIOD",
    "EMA_FAST",
    "EMA_SLOW",
    "EMA_PULLBACK_5M",
    "ORB_15M_5M",
    "REGULAR_SESSION_MINUTES",
    "STRATEGY_LABELS",
    "TARGET_R",
    "VWAP_RECLAIM_5M",
    "StrategyView",
    "SymbolFeatures",
    "V02Data",
    "V02FoldResult",
    "V02Result",
    "V02WalkForwardResult",
    "build_strategy_view",
    "ema_pullback_5m_candidate",
    "evaluate_v02",
    "mirror_relation_ok",
    "orb_15m_5m_candidate",
    "prepare_v02_data",
    "resample_session_5m",
    "run_v02_backtest",
    "run_v02_walkforward",
    "vwap_reclaim_5m_candidate",
]

# --------------------------------------------------------------------------- #
# V0.2 structure constants — a-priori, not tuned
# --------------------------------------------------------------------------- #
# Every value in this block is a fixed, one-shot prior taken from the V0.2
# research specification.  They were NOT fitted, optimised or re-tuned on any
# walk-forward train or test window; changing them invalidates comparability
# with the recorded V0.2 report.

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

REL_ACTIVITY_LOOKBACK = 20
"""Trailing 5m-bar trade-count window for relative IEX activity (a-priori)."""

# Strategy identifiers used in the V0.2 report.  ``core.StrategyName`` has no
# V0.2 members and must not be modified, so trades are mapped onto the closest
# existing enum member for the production TradeRecord payload while the report
# keeps the real V0.2 label in ``trade_details['strategy_label']``.
ORB_15M_5M = "ORB_15M_5M"
VWAP_RECLAIM_5M = "VWAP_RECLAIM_5M"
EMA_PULLBACK_5M = "EMA_PULLBACK_5M"
STRATEGY_LABELS: tuple[str, ...] = (ORB_15M_5M, VWAP_RECLAIM_5M, EMA_PULLBACK_5M)

_STRATEGY_ENUM: dict[str, StrategyName] = {
    ORB_15M_5M: StrategyName.ORB_15M,
    VWAP_RECLAIM_5M: StrategyName.VWAP_RECLAIM,
    EMA_PULLBACK_5M: StrategyName.EMA_PULLBACK,
}
ENUM_TO_LABEL: dict[StrategyName, str] = {
    value: key for key, value in _STRATEGY_ENUM.items()
}

# Universal V0.2 trade geometry (a-priori, not tuned):
#   entry_ref = signal 5m bar close, target = +2.0R,
#   stop      = structural price - 1.0 * ATR5m, clamped to [0.25%, 2.0%].
TARGET_R = 2.0
STOP_ATR_MULTIPLE = 1.0
MIN_STOP_PCT = 0.0025
MAX_STOP_PCT = 0.0200

# ORB_15M_5M (a-priori, not tuned)
ORB_OPENING_BARS = 3
ORB_MIN_RANGE_EXPANSION = 1.2
ORB_MIN_REL_ACTIVITY = 1.0

# VWAP_RECLAIM_5M (a-priori, not tuned)
VWAP_RECLAIM_BUFFER_PCT = 0.0005
VWAP_RECLAIM_MIN_REL_ACTIVITY = 1.1
VWAP_RECLAIM_MIN_RANGE_EXPANSION = 1.2

# EMA_PULLBACK_5M (a-priori, not tuned)
EMA_PULLBACK_TOUCH_ATR = 0.10
EMA_PULLBACK_MIN_RANGE_EXPANSION = 1.0

REGULAR_SESSION_MINUTES = 390
"""US regular session length used for bucket-grid/skipped-slot diagnostics."""

BUCKET_MINUTES = 5
_SESSION_CLOSE_MINUTES_AFTER_OPEN = REGULAR_SESSION_MINUTES

DATA_CAVEAT = (
    "IEX 单一交易所口径：成交量/VWAP/相对活跃度均只含 IEX 份额，"
    "不是全市场 consolidated tape；V0.2 是对 V0.1 的结构性对比研究，"
    "不代表未来收益，也不构成投资建议。"
)

#: Rounding tolerance for "is the open already through the stop/target" checks.
_PRICE_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Internal prepared-data containers
# --------------------------------------------------------------------------- #
@dataclass
class _Bucket:
    """One completed 5m bar plus everything needed for exits/fills.

    ``members`` are the 1m bars that formed the bucket (chronological); the
    ``bar`` is their 5m aggregate.  ``session_vwap`` is the *progressive*
    session VWAP at the moment this 5m bar closed (1m volume-weighted, reset at
    09:30 ET).
    """

    start_utc: datetime
    end_utc: datetime
    bar: Bar
    members: tuple[Bar, ...]
    session_vwap: float | None
    index_in_session: int
    global_index: int = -1


@dataclass
class _SymbolSession:
    symbol: str
    bars_1m: tuple[Bar, ...]
    buckets: tuple[_Bucket, ...]
    bucket_starts: tuple[datetime, ...]
    bucket_by_start: Mapping[datetime, _Bucket]
    last_bar_start_utc: datetime
    orb_high: float | None
    orb_low: float | None
    skipped_slots: int


@dataclass
class _Session:
    et_date: date
    open_utc: datetime
    close_utc: datetime
    symbols: Mapping[str, _SymbolSession]
    bucket_starts: tuple[datetime, ...]


@dataclass(frozen=True, slots=True)
class SymbolFeatures:
    """Per-symbol continuous 5m feature series (indexed by ``global_index``).

    The series run across the whole dataset, session after session, because the
    V0.2 spec only mandates a *session reset for VWAP*; EMA/ATR/relative
    activity need history from the previous session to be usable early in the
    day.  Computed in one pass, so ``series[i]`` only ever uses bars ``<= i``
    (no look-ahead).
    """

    symbol: str
    ema_fast: tuple[float | None, ...]
    ema_slow: tuple[float | None, ...]
    atr: tuple[float | None, ...]
    rel_activity: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class V02Data:
    """Immutable prepared V0.2 dataset (built once, reused by every run/fold)."""

    sessions: tuple[_Session, ...]
    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    features: Mapping[str, SymbolFeatures]
    total_1m_bars: int
    total_5m_bars: int
    skipped_buckets: int

    def sessions_for(self, wanted: Sequence[date] | Sequence[Any] | None) -> list[_Session]:
        """Return the sessions whose ET date is in *wanted* (all when ``None``)."""
        if wanted is None:
            return list(self.sessions)
        wanted_set = {value if isinstance(value, date) else date.fromisoformat(str(value)) for value in wanted}
        return [session for session in self.sessions if session.et_date in wanted_set]


# --------------------------------------------------------------------------- #
# Pure 5m resampling
# --------------------------------------------------------------------------- #
def _session_open_for(bar: Bar) -> datetime:
    return session_open_utc(bar.start_utc)


def _session_close_for(bar: Bar) -> datetime:
    return session_close_utc(bar.start_utc)


def _build_buckets(
    bars: Sequence[Bar],
    open_utc: datetime,
    close_utc: datetime,
) -> tuple[list[_Bucket], int]:
    """Group sorted 1m bars into completed 5m buckets anchored at ``open_utc``.

    Returns ``(buckets, skipped_slots)``.  A bucket is emitted only when at
    least one of its 1m bars exists **and** the bucket window has fully ended by
    the session close (an unfinished bucket is never evaluated).  Empty buckets
    are not fabricated; they are only counted in ``skipped_slots``.
    """

    grouped: dict[int, list[Bar]] = {}
    for bar in bars:
        if bar.start_utc < open_utc:
            continue
        minutes = (bar.start_utc - open_utc).total_seconds() / 60.0
        index = int(math.floor(minutes / BUCKET_MINUTES))
        bucket_start = open_utc + timedelta(minutes=BUCKET_MINUTES * index)
        if bucket_start >= close_utc:
            continue
        grouped.setdefault(index, []).append(bar)

    expected_slots = max(1, int(round((close_utc - open_utc).total_seconds() / 60.0 / BUCKET_MINUTES)))
    buckets: list[_Bucket] = []
    progress_pv = 0.0
    progress_volume = 0.0
    for bucket_index in sorted(grouped):
        members = grouped[bucket_index]
        if len(members) > 1:
            members.sort(key=lambda item: item.start_utc)
        bucket_start = open_utc + timedelta(minutes=BUCKET_MINUTES * bucket_index)
        bucket_end = bucket_start + timedelta(minutes=BUCKET_MINUTES)
        if bucket_end > close_utc:
            # The 5m window has not finished by the session close: skip it.
            continue
        high = max(item.high for item in members)
        low = min(item.low for item in members)
        volume = sum(item.volume for item in members)
        if volume > 0.0:
            vwap = sum(item.vwap * item.volume for item in members) / volume
        else:  # pragma: no cover - IEX bars always carry volume; safety net only
            vwap = (high + low + members[-1].close) / 3.0
        vwap = min(max(vwap, low), high)
        five_min = Bar(
            symbol=members[0].symbol,
            timeframe_minutes=BUCKET_MINUTES,
            start_utc=bucket_start,
            end_utc=bucket_end,
            open=members[0].open,
            high=high,
            low=low,
            close=members[-1].close,
            volume=volume,
            vwap=vwap,
            trade_count=sum(int(item.trade_count) for item in members),
            is_complete=True,
        )
        progress_pv += five_min.vwap * five_min.volume
        progress_volume += five_min.volume
        session_vwap = (
            progress_pv / progress_volume if progress_volume > 0.0 else None
        )
        buckets.append(
            _Bucket(
                start_utc=bucket_start,
                end_utc=bucket_end,
                bar=five_min,
                members=tuple(members),
                session_vwap=session_vwap,
                index_in_session=len(buckets),
            )
        )
    skipped_slots = max(0, expected_slots - len(buckets))
    return buckets, skipped_slots


def resample_session_5m(
    bars_1m: Sequence[Bar],
    session_open_utc_value: datetime,
    session_close_utc_value: datetime | None = None,
) -> list[Bar]:
    """Aggregate one symbol-session of 1m bars into completed 5m bars.

    Rules (V0.2 spec §2): bucket ``k`` is ``09:30 ET + k * 5min``; open is the
    first 1m open, high/low are the extremes, close is the last 1m close,
    volume is the sum, ``vwap = Σ(bar.vwap * bar.volume) / Σvolume`` and
    ``trade_count`` is the sum.  Buckets without any 1m bar are not emitted and
    an unfinished final bucket is dropped.

    Parameters are explicit ``session_open_utc`` / ``session_close_utc`` (UTC)
    so the function is pure and trivially testable; the close defaults to
    open + 390 minutes.
    """

    if isinstance(bars_1m, (str, bytes)):
        raise TypeError("bars_1m must be a sequence of Bar, not a string")
    open_utc = session_open_utc_value
    close_utc = (
        session_close_utc_value
        if session_close_utc_value is not None
        else open_utc + timedelta(minutes=_SESSION_CLOSE_MINUTES_AFTER_OPEN)
    )
    ordered = list(bars_1m)
    previous: datetime | None = None
    for bar in ordered:
        if not isinstance(bar, Bar):
            raise TypeError(f"bars_1m must contain Bar objects, got {type(bar)!r}")
        if bar.timeframe_minutes != 1:
            raise ValueError("resample_session_5m expects 1-minute bars")
        if not bar.is_complete:
            raise ValueError("resample_session_5m must only receive completed 1m bars")
        if previous is not None and bar.start_utc < previous:
            raise ValueError("bars_1m must be sorted by start_utc ascending")
        previous = bar.start_utc
    buckets, _ = _build_buckets(ordered, open_utc, close_utc)
    return [bucket.bar for bucket in buckets]


def _opening_range_15m(
    buckets: Sequence[_Bucket],
    open_utc: datetime,
) -> tuple[float | None, float | None]:
    """High/low of the first three 5m bars (09:30/09:35/09:40 buckets).

    Returns ``(None, None)`` unless all three buckets exist, which is the V0.2
    "前 3 根 5m bar 都存在" completeness requirement.
    """

    expected = tuple(open_utc + timedelta(minutes=BUCKET_MINUTES * index) for index in range(ORB_OPENING_BARS))
    by_start = {bucket.start_utc: bucket for bucket in buckets}
    if any(start not in by_start for start in expected):
        return None, None
    highs = [by_start[start].bar.high for start in expected]
    lows = [by_start[start].bar.low for start in expected]
    return max(highs), min(lows)


# --------------------------------------------------------------------------- #
# Continuous feature series
# --------------------------------------------------------------------------- #
def _atr_across_sessions(
    session_bars: Sequence[Sequence[Bar]],
    period: int = ATR_PERIOD,
) -> tuple[float | None, ...]:
    """Wilder ATR over 5m bars, continuous across sessions but gap-safe.

    The true range of the *first* bar of every session is ``high - low``: the
    overnight gap must not inflate ATR (otherwise every 09:30 bar would look
    like a volatility explosion and corrupt the range-expansion filter).
    """

    ranges: list[float] = []
    for bars in session_bars:
        previous_close: float | None = None
        for bar in bars:
            if previous_close is None:
                ranges.append(bar.high - bar.low)
            else:
                ranges.append(
                    max(
                        bar.high - bar.low,
                        abs(bar.high - previous_close),
                        abs(bar.low - previous_close),
                    )
                )
            previous_close = bar.close
    out: list[float | None] = [None] * len(ranges)
    if len(ranges) < period:
        return tuple(out)
    previous = sum(ranges[:period]) / period
    out[period - 1] = previous
    for index in range(period, len(ranges)):
        previous = (previous * (period - 1) + ranges[index]) / period
        out[index] = previous
    return tuple(out)


def _relative_activity(trade_counts: Sequence[int], lookback: int) -> tuple[float, ...]:
    """``trade_count[i] / median(trade_count[i-lookback:i])``.

    The baseline deliberately *excludes* the current bar (a bar cannot dilute
    its own activity benchmark) and falls back to ``1.0`` while fewer than
    ``lookback`` prior bars exist — the V0.2 "样本不足用 1.0" rule.
    """

    values = [float(count) for count in trade_counts]
    out: list[float] = [1.0] * len(values)
    for index in range(lookback, len(values)):
        window = values[index - lookback : index]
        baseline = median(window)
        out[index] = (values[index] / baseline) if baseline > 0.0 else 1.0
    return tuple(out)


def _ema_series_safe(values: Sequence[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None] * len(values)
    return list(_ema_series(values, period))


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #
def prepare_v02_data(bars_1m: Sequence[Bar]) -> V02Data:
    """Validate 1m bars and build the reusable V0.2 5m dataset.

    Steps
    -----
    1. validate: every item is a 1m completed ``Bar``; the stream is sorted by
       ``start_utc``; no duplicate ``(symbol, start)``;
    2. keep regular-session bars only (09:30-16:00 ET weekdays) and group by
       ``(ET trading date, symbol)``;
    3. aggregate each symbol-session to completed 5m buckets (session anchored);
    4. assign per-symbol global 5m indexes and compute continuous EMA9/EMA21,
       gap-safe ATR14 and relative trade-count activity.

    The result is immutable and can be reused across full-period runs,
    segments and every walk-forward fold (no recomputation cost).
    """

    if isinstance(bars_1m, (str, bytes)):
        raise TypeError("bars_1m must be a sequence of Bar, not a string")
    items = list(bars_1m)

    previous_start: datetime | None = None
    for bar in items:
        if not isinstance(bar, Bar):
            raise TypeError(f"bars_1m must contain Bar objects, got {type(bar)!r}")
        if bar.timeframe_minutes != 1:
            raise ValueError(
                f"prepare_v02_data expects 1-minute bars, got {bar.timeframe_minutes}m"
            )
        if previous_start is not None and bar.start_utc < previous_start:
            raise ValueError(
                "bars_1m must be sorted by start_utc ascending; "
                f"got {bar.start_utc.isoformat()} after {previous_start.isoformat()}"
            )
        previous_start = bar.start_utc

    grouped: dict[date, dict[str, list[Bar]]] = {}
    usable = 0
    for bar in items:
        if not bar.is_complete:
            continue
        if not is_regular_session(bar.start_utc):
            continue
        trading_date = et_date(bar.start_utc)
        grouped.setdefault(trading_date, {}).setdefault(bar.symbol, []).append(bar)
        usable += 1

    sessions: list[_Session] = []
    total_5m = 0
    skipped_total = 0
    for trading_date in sorted(grouped):
        per_symbol = grouped[trading_date]
        symbol_sessions: dict[str, _SymbolSession] = {}
        for symbol, bars in per_symbol.items():
            seen: set[datetime] = set()
            for bar in bars:
                if bar.start_utc in seen:
                    raise ValueError(
                        f"duplicate 1m bar for {symbol} at {bar.start_utc.isoformat()}"
                    )
                seen.add(bar.start_utc)
            bars.sort(key=lambda item: item.start_utc)
            open_utc = _session_open_for(bars[0])
            close_utc = _session_close_for(bars[0])
            buckets, skipped_slots = _build_buckets(bars, open_utc, close_utc)
            orb_high, orb_low = _opening_range_15m(buckets, open_utc)
            symbol_sessions[symbol] = _SymbolSession(
                symbol=symbol,
                bars_1m=tuple(bars),
                buckets=tuple(buckets),
                bucket_starts=tuple(bucket.start_utc for bucket in buckets),
                bucket_by_start={bucket.start_utc: bucket for bucket in buckets},
                last_bar_start_utc=bars[-1].start_utc,
                orb_high=orb_high,
                orb_low=orb_low,
                skipped_slots=skipped_slots,
            )
            total_5m += len(buckets)
            skipped_total += skipped_slots
        union_starts = sorted(
            {start for symbol_session in symbol_sessions.values() for start in symbol_session.bucket_starts}
        )
        reference_bar = next(iter(symbol_sessions.values())).bars_1m[0]
        sessions.append(
            _Session(
                et_date=trading_date,
                open_utc=_session_open_for(reference_bar),
                close_utc=_session_close_for(reference_bar),
                symbols=symbol_sessions,
                bucket_starts=tuple(union_starts),
            )
        )

    symbols = tuple(
        sorted({symbol for session in sessions for symbol in session.symbols})
    )
    features: dict[str, SymbolFeatures] = {}
    for symbol in symbols:
        flat_buckets: list[_Bucket] = []
        session_bar_lists: list[list[Bar]] = []
        for session in sessions:
            symbol_session = session.symbols.get(symbol)
            if symbol_session is None:
                continue
            session_bar_lists.append([bucket.bar for bucket in symbol_session.buckets])
            for bucket in symbol_session.buckets:
                bucket.global_index = len(flat_buckets)
                flat_buckets.append(bucket)
        closes = [bucket.bar.close for bucket in flat_buckets]
        trade_counts = [bucket.bar.trade_count for bucket in flat_buckets]
        features[symbol] = SymbolFeatures(
            symbol=symbol,
            ema_fast=tuple(_ema_series_safe(closes, EMA_FAST)),
            ema_slow=tuple(_ema_series_safe(closes, EMA_SLOW)),
            atr=_atr_across_sessions(session_bar_lists),
            rel_activity=_relative_activity(trade_counts, REL_ACTIVITY_LOOKBACK),
        )

    return V02Data(
        sessions=tuple(sessions),
        dates=tuple(session.et_date for session in sessions),
        symbols=symbols,
        features=features,
        total_1m_bars=usable,
        total_5m_bars=total_5m,
        skipped_buckets=skipped_total,
    )


# --------------------------------------------------------------------------- #
# Strategy view + pure strategy rules
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class StrategyView:
    """Everything a V0.2 strategy may look at (all values as of the 5m close).

    Built from the prepared dataset by :func:`build_strategy_view`; tests can
    construct it directly, which keeps the strategy rules unit-testable without
    any engine or real data.
    """

    symbol: str
    now_utc: datetime
    bar: Bar
    prev_bar: Bar | None
    prior_bars: tuple[Bar, ...]
    prior_vwaps: tuple[float | None, ...]
    session_open_utc: datetime
    orb_high: float | None
    ema_fast: float | None
    ema_slow: float | None
    ema_fast_prev: float | None
    atr: float | None
    atr_prev: float | None
    range_expansion: float | None
    rel_activity: float
    session_vwap: float | None
    prev_session_vwap: float | None
    mirror_ok: bool


def mirror_relation_ok(
    data: V02Data,
    session: _Session,
    symbol: str,
    bucket_start_utc: datetime,
) -> bool:
    """V0.2 mirror-regime gate (§4, corrected mirror logic).

    For candidate symbol ``S`` the *other* ETF ``O`` (SOXL <-> SOXS) must have
    its latest completed 5m bar (same ET session, ``start <= current start``)
    **below** *its* session VWAP — i.e. the inverse leg must not be confirming
    the opposite direction:

    * long SOXL (bullish semis)  -> SOXS close < SOXS session VWAP;
    * long SOXS (bearish semis)  -> SOXL close < SOXL session VWAP.

    The SOXS side was corrected by the Lead on 2026-11-06: the original V0.2
    task text said ``SOXL close > SOXL VWAP`` (a same-direction condition and a
    spec typo); both sides now use the strict ``other close < other VWAP``
    mirror relation.  No other rule or constant changed.
    """

    if symbol == "SOXL":
        other = "SOXS"
    elif symbol == "SOXS":
        other = "SOXL"
    else:  # pragma: no cover - only the two inverse ETFs are ever traded
        return False
    other_session = session.symbols.get(other)
    if other_session is None or not other_session.buckets:
        return False
    index = bisect_right(other_session.bucket_starts, bucket_start_utc) - 1
    if index < 0:
        return False
    other_bucket = other_session.buckets[index]
    other_vwap = other_bucket.session_vwap
    if other_vwap is None or not math.isfinite(other_vwap):
        return False
    # Both directions require the other ETF to be weak vs its own session VWAP.
    return other_bucket.bar.close < other_vwap


def build_strategy_view(
    data: V02Data,
    session: _Session,
    symbol: str,
    bucket_start_utc: datetime,
) -> StrategyView | None:
    """Assemble the immutable :class:`StrategyView` for one 5m bucket."""

    symbol_session = session.symbols.get(symbol)
    if symbol_session is None:
        return None
    bucket = symbol_session.bucket_by_start.get(bucket_start_utc)
    if bucket is None:
        return None
    features = data.features[symbol]
    index = bucket.global_index
    bucket_index = bucket.index_in_session
    prior = symbol_session.buckets[max(0, bucket_index - 2) : bucket_index + 1]
    prev_bucket = symbol_session.buckets[bucket_index - 1] if bucket_index > 0 else None
    atr_value = features.atr[index] if 0 <= index < len(features.atr) else None
    atr_prev = (
        features.atr[index - 1] if index > 0 and index - 1 < len(features.atr) else None
    )
    range_expansion: float | None = None
    if atr_prev is not None and atr_prev > 0.0:
        range_expansion = bucket.bar.range / atr_prev
    return StrategyView(
        symbol=symbol,
        now_utc=bucket.end_utc,
        bar=bucket.bar,
        prev_bar=prev_bucket.bar if prev_bucket is not None else None,
        prior_bars=tuple(item.bar for item in prior),
        prior_vwaps=tuple(item.session_vwap for item in prior),
        session_open_utc=session.open_utc,
        orb_high=symbol_session.orb_high,
        ema_fast=features.ema_fast[index],
        ema_slow=features.ema_slow[index],
        ema_fast_prev=(
            features.ema_fast[index - 1]
            if index > 0 and index - 1 < len(features.ema_fast)
            else None
        ),
        atr=atr_value,
        atr_prev=atr_prev,
        range_expansion=range_expansion,
        rel_activity=features.rel_activity[index],
        session_vwap=bucket.session_vwap,
        prev_session_vwap=prev_bucket.session_vwap if prev_bucket is not None else None,
        mirror_ok=mirror_relation_ok(data, session, symbol, bucket.start_utc),
    )


def _make_candidate(
    view: StrategyView,
    strategy_label: str,
    structural_stop: float,
    rationale: str,
) -> SignalCandidate | None:
    """Apply the universal V0.2 stop/target geometry and build the candidate.

    ``stop = structural_stop - 1.0 * ATR5m`` then clamped into
    ``[entry * (1 - 2.0%), entry * (1 - 0.25%)]`` so the stop is neither a
    noise-width scratch nor an unbounded risk; target is a fixed ``+2.0R``.
    Returns ``None`` when ATR is not yet available or the geometry is invalid.
    """

    entry = view.bar.close
    atr_value = view.atr
    if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0.0:
        return None
    if not view.mirror_ok:
        return None
    if not math.isfinite(structural_stop) or structural_stop <= 0.0:
        return None
    raw_stop = structural_stop - STOP_ATR_MULTIPLE * atr_value
    widest_allowed = entry * (1.0 - MAX_STOP_PCT)  # lowest stop price allowed
    tightest_allowed = entry * (1.0 - MIN_STOP_PCT)  # highest stop price allowed
    stop = min(max(raw_stop, widest_allowed), tightest_allowed)
    if not stop < entry:
        return None
    stop_pct = (entry - stop) / entry
    note = (
        f"action={strategy_label.replace('_5M', '')}; structural_stop={structural_stop:.4f}; "
        f"raw_stop={raw_stop:.4f}; atr5m={atr_value:.4f}; "
        f"stop_pct={stop_pct:.5%} (clamp {MIN_STOP_PCT:.2%}-{MAX_STOP_PCT:.2%})"
    )
    return SignalCandidate(
        symbol=view.symbol,
        strategy=_STRATEGY_ENUM[strategy_label],
        entry_ref=entry,
        stop_loss=stop,
        target_r=TARGET_R,
        rationale=rationale,
        created_utc=view.now_utc,
        confirmation_bar_start_utc=view.bar.start_utc,
        structural_note=note,
    )


def orb_15m_5m_candidate(view: StrategyView) -> SignalCandidate | None:
    """ORB_15M_5M: break the first-three-5m-bar high after 09:45 ET (a-priori)."""

    if view.orb_high is None:
        return None
    if view.bar.start_utc < view.session_open_utc + timedelta(minutes=15):
        return None
    data_bar = view.bar
    if not data_bar.close > view.orb_high:
        return None
    if not data_bar.is_bullish:
        return None
    if (
        view.range_expansion is not None
        and view.range_expansion < ORB_MIN_RANGE_EXPANSION
    ):
        return None
    if view.rel_activity < ORB_MIN_REL_ACTIVITY:
        return None
    if not view.mirror_ok:
        return None
    structural_stop = min(view.orb_high, data_bar.low)
    rationale = (
        f"{ORB_15M_5M}: 5m close {data_bar.close:.4f} > OR15 high {view.orb_high:.4f}; "
        f"range_expansion={_fmt_optional(view.range_expansion)}, "
        f"rel_activity={view.rel_activity:.2f}; mirror=OK"
    )
    return _make_candidate(view, ORB_15M_5M, structural_stop, rationale)


def vwap_reclaim_5m_candidate(view: StrategyView) -> SignalCandidate | None:
    """VWAP_RECLAIM_5M: reclaim the session VWAP with trend + activity (a-priori)."""

    if view.prev_bar is None or view.session_vwap is None or view.prev_session_vwap is None:
        return None
    if not view.prev_bar.close < view.prev_session_vwap:
        return None
    threshold = view.session_vwap * (1.0 + VWAP_RECLAIM_BUFFER_PCT)
    if not view.bar.close > threshold:
        return None
    if not view.bar.is_bullish:
        return None
    if view.ema_fast is None or view.ema_slow is None or not view.ema_fast > view.ema_slow:
        return None
    activity_ok = view.rel_activity >= VWAP_RECLAIM_MIN_REL_ACTIVITY or (
        view.range_expansion is not None
        and view.range_expansion >= VWAP_RECLAIM_MIN_RANGE_EXPANSION
    )
    if not activity_ok:
        return None
    if not view.mirror_ok:
        return None
    structural_stop = min(view.bar.low, view.session_vwap)
    rationale = (
        f"{VWAP_RECLAIM_5M}: prev 5m close {view.prev_bar.close:.4f} < VWAP "
        f"{view.prev_session_vwap:.4f}; now close {view.bar.close:.4f} > "
        f"VWAP {view.session_vwap:.4f}; ema9>ema21; "
        f"range_expansion={_fmt_optional(view.range_expansion)}, "
        f"rel_activity={view.rel_activity:.2f}; mirror=OK"
    )
    return _make_candidate(view, VWAP_RECLAIM_5M, structural_stop, rationale)


def ema_pullback_5m_candidate(view: StrategyView) -> SignalCandidate | None:
    """EMA_PULLBACK_5M: VWAP-up trend, EMA9 touch, bullish continuation (a-priori)."""

    if view.prev_bar is None or len(view.prior_bars) < 3:
        return None
    for prior_bar, prior_vwap in zip(view.prior_bars, view.prior_vwaps):
        if prior_vwap is None or not prior_bar.close > prior_vwap:
            return None
    if view.ema_fast is None or view.ema_slow is None or not view.ema_fast > view.ema_slow:
        return None
    if view.ema_fast_prev is None or view.atr_prev is None or view.atr_prev <= 0.0:
        return None
    tolerance = EMA_PULLBACK_TOUCH_ATR * view.atr_prev
    if not view.prev_bar.low <= view.ema_fast_prev + tolerance:
        return None
    if not view.prev_bar.close >= view.ema_fast_prev - tolerance:
        return None
    if not view.bar.is_bullish:
        return None
    if not view.bar.close > view.prev_bar.high:
        return None
    if (
        view.range_expansion is not None
        and view.range_expansion < EMA_PULLBACK_MIN_RANGE_EXPANSION
    ):
        return None
    if not view.mirror_ok:
        return None
    structural_stop = min(view.prev_bar.low, view.ema_fast)
    rationale = (
        f"{EMA_PULLBACK_5M}: 3 根 5m close 均在 VWAP 上方; ema9 {view.ema_fast:.4f} > "
        f"ema21 {view.ema_slow:.4f}; 前一根 low {view.prev_bar.low:.4f} <= "
        f"ema9_prev {view.ema_fast_prev:.4f} + 0.10*ATR; 当前 close "
        f"{view.bar.close:.4f} > 前一根 high {view.prev_bar.high:.4f}; "
        f"range_expansion={_fmt_optional(view.range_expansion)}; mirror=OK"
    )
    return _make_candidate(view, EMA_PULLBACK_5M, structural_stop, rationale)


def evaluate_v02(view: StrategyView) -> SignalCandidate | None:
    """Evaluate the three V0.2 strategies in the fixed spec order.

    Order is frozen: ORB_15M_5M -> VWAP_RECLAIM_5M -> EMA_PULLBACK_5M; the
    first valid candidate wins (V0.2 spec §6).
    """

    candidate = orb_15m_5m_candidate(view)
    if candidate is not None:
        return candidate
    candidate = vwap_reclaim_5m_candidate(view)
    if candidate is not None:
        return candidate
    return ema_pullback_5m_candidate(view)


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


#: Frozen evaluation order (V0.2 spec §6): the engine offers every symbol to
#: the first rule before moving on to the next rule.
_RULE_ORDER: tuple[Callable[[StrategyView], SignalCandidate | None], ...] = (
    orb_15m_5m_candidate,
    vwap_reclaim_5m_candidate,
    ema_pullback_5m_candidate,
)


# --------------------------------------------------------------------------- #
# Replay engine
# --------------------------------------------------------------------------- #
@dataclass
class _PendingEntry:
    candidate: SignalCandidate
    plan: RiskPlan
    signal_id: str
    expected_bucket_start_utc: datetime


@dataclass
class _OpenTrade:
    signal_id: str
    candidate: SignalCandidate
    plan: RiskPlan
    shares: float
    entry_fill: float
    entry_time_utc: datetime
    stop: float
    target: float
    mae_dollars: float = 0.0
    mfe_dollars: float = 0.0
    ambiguous: bool = False


@dataclass
class V02Result:
    """One V0.2 run (full period, a segment, or a single walk-forward fold)."""

    trades: list[TradeRecord]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics
    trade_details: dict[str, dict[str, Any]]
    initial_equity: float
    final_equity: float
    sessions_run: int
    session_dates: list[date]
    buckets_evaluated: int
    skipped_buckets: int
    signals: int
    pending_cancelled: int
    ambiguous_bars: int
    breaker_days: int
    risk_block_counts: dict[str, int]
    notes: list[str]
    elapsed_seconds: float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def trades_per_session(self) -> float:
        return (len(self.trades) / self.sessions_run) if self.sessions_run else 0.0

    def extra_metrics(self) -> dict[str, Any]:
        """V0.2-only statistics that :class:`BacktestMetrics` does not carry."""
        return {
            "ambiguous_bars": int(self.ambiguous_bars),
            "breaker_days": int(self.breaker_days),
            "pending_cancelled": int(self.pending_cancelled),
            "signals": int(self.signals),
            "buckets_evaluated": int(self.buckets_evaluated),
            "skipped_buckets": int(self.skipped_buckets),
            "sessions_run": int(self.sessions_run),
            "trades_per_session": round(self.trades_per_session, 4),
            "risk_block_counts": dict(self.risk_block_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class _V02Replay:
    """Deterministic 5m-driven replay over prepared V0.2 sessions."""

    def __init__(
        self,
        data: V02Data,
        config: AppConfig,
        *,
        risk: Any | None = None,
        initial_equity: float | None = None,
    ) -> None:
        if not isinstance(data, V02Data):
            raise TypeError("data must be a V02Data prepared by prepare_v02_data")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self.data = data
        self.config = config
        self.risk = risk if risk is not None else RiskManager(config)
        equity = float(
            config.risk.account_equity if initial_equity is None else initial_equity
        )
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError(f"initial_equity must be finite and > 0, got {initial_equity!r}")
        self.initial_equity = equity
        self.symbols = tuple(
            symbol for symbol in config.data.symbols if symbol in data.features
        )
        if not self.symbols:
            raise ValueError(
                "config.data.symbols does not intersect the prepared dataset symbols "
                f"({data.symbols})"
            )

        self._slippage_rate = float(config.risk.est_slippage_bps) / 10_000.0
        self._commission_per_share = float(config.risk.commission_per_share)

        # per-run state
        self._pending: _PendingEntry | None = None
        self._open: _OpenTrade | None = None
        self._trades: list[TradeRecord] = []
        self._trade_details: dict[str, dict[str, Any]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._equity = self.initial_equity
        self._session = None  # type: _Session | None
        self._buckets_evaluated = 0
        self._skipped_buckets = 0
        self._signals = 0
        self._pending_cancelled = 0
        self._ambiguous_bars = 0
        self._risk_blocks: dict[str, int] = {}
        self._notes: list[str] = []
        self._signal_seq = 0

    # -- public --------------------------------------------------------- #
    def run(self, session_dates: Sequence[date] | None = None) -> V02Result:
        sessions = self.data.sessions_for(session_dates)
        for session in sessions:
            self._run_session(session)
        self._finalize()
        metrics = compute_metrics(
            self._trades,
            self._equity_curve,
            initial_equity=self.initial_equity,
            trade_details=self._trade_details,
        )
        dates = [session.et_date for session in sessions]
        breaker_days = self._count_breaker_days()
        notes = list(self._notes)
        notes.append(DATA_CAVEAT)
        if self._risk_blocks:
            summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(self._risk_blocks.items())
            )
            notes.append(f"risk blocks: {summary}")
        return V02Result(
            trades=list(self._trades),
            equity_curve=list(self._equity_curve),
            metrics=metrics,
            trade_details=dict(self._trade_details),
            initial_equity=self.initial_equity,
            final_equity=self._equity,
            sessions_run=len(sessions),
            session_dates=dates,
            buckets_evaluated=self._buckets_evaluated,
            skipped_buckets=self._skipped_buckets,
            signals=self._signals,
            pending_cancelled=self._pending_cancelled,
            ambiguous_bars=self._ambiguous_bars,
            breaker_days=breaker_days,
            risk_block_counts=dict(self._risk_blocks),
            notes=notes,
        )

    # -- session loop --------------------------------------------------- #
    def _run_session(self, session: _Session) -> None:
        self._session = session
        # Hard V0.2 requirement (spec §7): sync the risk clock onto the replay
        # session before any evaluation.  ``roll`` refuses to move backwards and
        # would otherwise latch the wall-clock day (the V0.1 circuit-breaker
        # lockup that froze three years of daily state).
        begin_session = getattr(self.risk, "begin_session", None)
        if not callable(begin_session):
            raise TypeError(
                "risk manager must expose begin_session(session_open_utc); "
                "roll() alone is not enough for historical replay"
            )
        begin_session(session.open_utc)
        self._pending = None
        self._open = None
        for session_symbol in session.symbols.values():
            self._skipped_buckets += session_symbol.skipped_slots
        for bucket_start in session.bucket_starts:
            self._buckets_evaluated += 1
            # 1) A pending signal may only fill at the next 5m bucket's open.
            if self._pending is not None:
                expected = self._pending.expected_bucket_start_utc
                if bucket_start != expected:
                    self._cancel_pending(
                        "missing next 5m bucket (gap); order cancelled, no stale fill"
                    )
                else:
                    self._try_fill_pending(session, bucket_start)
            # 2) Manage an open position bar-by-bar on its 1m bars (pessimistic).
            if self._open is not None:
                self._manage_open_trade(session, bucket_start)
            # 3) Evaluate completed 5m bars only when flat and uncommitted.
            if self._open is None and self._pending is None:
                self._evaluate_bucket(session, bucket_start)
        self._end_session(session)

    def _end_session(self, session: _Session) -> None:
        if self._pending is not None:
            self._cancel_pending("new session started before the pending fill")
        if self._open is not None:
            # Data-gap safety net: the symbol had no 1m bar in the final buckets.
            open_trade = self._open
            symbol_session = session.symbols.get(open_trade.candidate.symbol)
            last_bar = symbol_session.bars_1m[-1] if symbol_session is not None else None
            if last_bar is not None:
                self._notes.append(
                    "open position force-closed at session's last observed 1m bar "
                    f"({open_trade.candidate.symbol}, reason=EOD_EXIT)"
                )
                self._close_trade(
                    reference_price=last_bar.close,
                    fill_price=last_bar.close * (1.0 - self._slippage_rate),
                    ts=last_bar.end_utc,
                    reason="EOD_EXIT",
                )
            else:  # pragma: no cover - a position can only exist after a bar
                self._open = None

    def _finalize(self) -> None:
        if self._pending is not None:
            self._cancel_pending("end of data")
        if self._open is not None:  # pragma: no cover - _end_session closes first
            trade = self._open
            self._close_trade(
                reference_price=trade.entry_fill,
                fill_price=trade.entry_fill,
                ts=trade.entry_time_utc,
                reason="END_OF_DATA",
            )

    # -- pending entries / fills --------------------------------------- #
    def _try_fill_pending(self, session: _Session, bucket_start: datetime) -> None:
        pending = self._pending
        if pending is None:  # pragma: no cover - caller guarantees
            return
        symbol = pending.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        bucket = None if symbol_session is None else symbol_session.bucket_by_start.get(bucket_start)
        if bucket is None or not bucket.members:
            self._cancel_pending(
                f"next 5m bucket has no 1m bars for {symbol}; order cancelled"
            )
            return
        self._pending = None
        first_bar = bucket.members[0]
        entry_open = float(first_bar.open)
        candidate = pending.candidate
        if entry_open <= candidate.stop_loss:
            self._pending_cancelled += 1
            self._notes.append(
                "entry cancelled: next 5m open gapped through the stop "
                f"({symbol} open={entry_open:.4f} stop={candidate.stop_loss:.4f})"
            )
            self._cancel_risk_pending()
            return
        if entry_open >= candidate.take_profit_price:
            self._pending_cancelled += 1
            self._notes.append(
                "entry cancelled: next 5m open gapped through the target "
                f"({symbol} open={entry_open:.4f} target={candidate.take_profit_price:.4f})"
            )
            self._cancel_risk_pending()
            return
        entry_fill = entry_open * (1.0 + self._slippage_rate)
        try:
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=first_bar.start_utc,
                signal_id=pending.signal_id,
                mode=_mode_value(self.risk.mode),
            )
        except TypeError:  # pragma: no cover - simplified fake risk managers
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=first_bar.start_utc,
                signal_id=pending.signal_id,
            )
        self._open = _OpenTrade(
            signal_id=pending.signal_id,
            candidate=candidate,
            plan=pending.plan,
            shares=float(pending.plan.shares),
            entry_fill=entry_fill,
            entry_time_utc=first_bar.start_utc,
            stop=float(candidate.stop_loss),
            target=float(candidate.take_profit_price),
        )

    def _cancel_pending(self, reason: str) -> None:
        if self._pending is None:
            return
        self._pending = None
        self._pending_cancelled += 1
        self._notes.append(f"pending signal cancelled: {reason}")
        self._cancel_risk_pending()

    def _cancel_risk_pending(self) -> None:
        record_cancel = getattr(self.risk, "record_cancel", None)
        if callable(record_cancel):
            try:
                record_cancel()
            except Exception:  # pragma: no cover - optional integration
                pass

    # -- exits ---------------------------------------------------------- #
    def _manage_open_trade(self, session: _Session, bucket_start: datetime) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        symbol = trade.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        if symbol_session is None:  # pragma: no cover - position implies bars
            return
        bucket = symbol_session.bucket_by_start.get(bucket_start)
        if bucket is None:
            return
        for bar in bucket.members:
            if bar.start_utc < trade.entry_time_utc:
                continue
            # Track excursions on every observed 1m bar (dollar terms, like V0.1).
            trade.mae_dollars = max(
                trade.mae_dollars, max(0.0, (trade.entry_fill - bar.low) * trade.shares)
            )
            trade.mfe_dollars = max(
                trade.mfe_dollars, max(0.0, (bar.high - trade.entry_fill) * trade.shares)
            )
            stop_hit = bar.low <= trade.stop + _PRICE_EPS
            target_hit = bar.high >= trade.target - _PRICE_EPS
            if stop_hit or target_hit:
                if stop_hit and target_hit:
                    # Same-bar TP/SL collision -> conservative stop fill (spec §7).
                    self._ambiguous_bars += 1
                    trade.ambiguous = True
                    prefer_stop = True
                else:
                    prefer_stop = stop_hit
                if prefer_stop:
                    reference = bar.open if bar.open <= trade.stop else trade.stop
                    self._close_trade(
                        reference_price=reference,
                        fill_price=reference * (1.0 - self._slippage_rate),
                        ts=bar.end_utc,
                        reason="STOP",
                    )
                else:
                    reference = trade.target
                    self._close_trade(
                        reference_price=reference,
                        fill_price=reference * (1.0 - self._slippage_rate),
                        ts=bar.end_utc,
                        reason="TARGET",
                    )
                return
            if bar.start_utc == symbol_session.last_bar_start_utc:
                # 15:59 ET (or the session's observed last bar): force flat.
                self._close_trade(
                    reference_price=bar.close,
                    fill_price=bar.close * (1.0 - self._slippage_rate),
                    ts=bar.end_utc,
                    reason="EOD_EXIT",
                )
                return

    def _close_trade(
        self,
        *,
        reference_price: float,
        fill_price: float,
        ts: datetime,
        reason: str,
    ) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        record = self.risk.confirm_exit(fill_price=fill_price, fill_utc=ts, reason=reason)
        self._open = None
        self._trades.append(record)
        self._equity += record.pnl
        self._equity_curve.append((ts, self._equity))
        entry_bucket_start = trade.entry_time_utc
        if self._session is not None:
            minutes = (trade.entry_time_utc - self._session.open_utc).total_seconds() / 60.0
            index = int(math.floor(minutes / BUCKET_MINUTES))
            entry_bucket_start = self._session.open_utc + timedelta(
                minutes=BUCKET_MINUTES * index
            )
        self._trade_details[record.signal_id] = {
            "mae_dollars": trade.mae_dollars,
            "mfe_dollars": trade.mfe_dollars,
            "entry_slippage_dollars": (trade.entry_fill - trade.candidate.entry_ref)
            * trade.shares,
            "exit_slippage_dollars": (reference_price - fill_price) * trade.shares,
            "ambiguous_bar": bool(trade.ambiguous),
            "exit_reason": reason,
            "holding_minutes": record.holding_minutes,
            "fees_dollars": record.fees,
            "strategy_label": ENUM_TO_LABEL.get(record.strategy, record.strategy.value),
            "signal_bar_start_utc": trade.candidate.confirmation_bar_start_utc.isoformat(),
            "entry_bucket_start_utc": entry_bucket_start.isoformat(),
        }

    # -- strategy evaluation ------------------------------------------- #
    def _evaluate_bucket(self, session: _Session, bucket_start: datetime) -> None:
        # Build one view per symbol present in this bucket, then apply the
        # **strategy-major** fixed order from the spec: every symbol is offered
        # to ORB_15M_5M first, then VWAP_RECLAIM_5M, then EMA_PULLBACK_5M; the
        # first valid candidate wins (no cross-symbol reordering can let a
        # weaker strategy jump the queue).
        views: list[StrategyView] = []
        for symbol in self.symbols:
            symbol_session = session.symbols.get(symbol)
            if symbol_session is None:
                continue
            if bucket_start not in symbol_session.bucket_by_start:
                continue
            view = build_strategy_view(self.data, session, symbol, bucket_start)
            if view is not None:
                views.append(view)
        if not views:
            return
        candidate: SignalCandidate | None = None
        for rule in _RULE_ORDER:
            for view in views:
                candidate = rule(view)
                if candidate is not None:
                    break
            if candidate is not None:
                break
        if candidate is None:
            return
        self._signals += 1
        plan = size_position(candidate, self.config.risk, equity=self._equity)
        decision = self.risk.evaluate_entry(
            candidate, plan, candidate.created_utc, data_ready=True
        )
        if not bool(getattr(decision, "allowed", False)):
            reason = getattr(decision, "reason", None)
            key = getattr(reason, "value", str(reason))
            self._risk_blocks[key] = self._risk_blocks.get(key, 0) + 1
            return
        allowed_plan = getattr(decision, "plan", None) or plan
        self._signal_seq += 1
        self._pending = _PendingEntry(
            candidate=candidate,
            plan=allowed_plan,
            signal_id=f"V02-{self._signal_seq:06d}",
            expected_bucket_start_utc=bucket_start + timedelta(minutes=BUCKET_MINUTES),
        )

    # -- stats ---------------------------------------------------------- #
    def _count_breaker_days(self) -> int:
        threshold = -abs(float(self.config.risk.max_daily_net_loss))
        daily: dict[date, float] = {}
        for trade in self._trades:
            if trade.exit_time_utc is None:
                continue
            day = et_date(trade.exit_time_utc)
            daily[day] = daily.get(day, 0.0) + trade.pnl
        return sum(1 for pnl in daily.values() if pnl <= threshold + 1e-9)


def _mode_value(mode: Any) -> Any:
    return getattr(mode, "value", mode)


# --------------------------------------------------------------------------- #
# Public run / walk-forward entry points
# --------------------------------------------------------------------------- #
def run_v02_backtest(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    risk: Any | None = None,
    initial_equity: float | None = None,
    session_dates: Sequence[date] | None = None,
) -> V02Result:
    """Run the V0.2 replay over *data* (prepared ``V02Data`` or raw 1m bars).

    Parameters
    ----------
    data:
        A :class:`V02Data` (preferred, reusable across runs/folds) or a sorted
        sequence of completed regular-session 1m :class:`Bar` objects.
    config:
        Validated :class:`~intraday_scalper.core.config.AppConfig`; default
        config is loaded when omitted.
    risk:
        Optional ``RiskManager``-compatible instance.  ``None`` creates a fresh
        production :class:`~intraday_scalper.risk.manager.RiskManager` (fresh
        state per run, as the walk-forward contract requires).
    initial_equity:
        Starting equity; defaults to ``config.risk.account_equity``.
    session_dates:
        Optional ET-date filter (used by segments / walk-forward folds).
    """

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    replay = _V02Replay(
        prepared, app_config, risk=risk, initial_equity=initial_equity
    )
    return replay.run(session_dates)


@dataclass(frozen=True, slots=True)
class V02FoldResult:
    """One walk-forward fold: fresh engine/risk state for train *and* test."""

    fold: Fold
    train_result: V02Result
    test_result: V02Result
    leakage_checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": {
                "index": self.fold.index,
                "train_start": self.fold.train_start.isoformat() if self.fold.train_start else None,
                "train_end": self.fold.train_end.isoformat() if self.fold.train_end else None,
                "test_start": self.fold.test_start.isoformat() if self.fold.test_start else None,
                "test_end": self.fold.test_end.isoformat() if self.fold.test_end else None,
                "train_days": len(self.fold.train_dates),
                "test_days": len(self.fold.test_dates),
            },
            "train_trades": self.train_result.total_trades,
            "test_trades": self.test_result.total_trades,
            "train_metrics": self.train_result.metrics.to_dict(),
            "test_metrics": self.test_result.metrics.to_dict(),
            "test_extra": self.test_result.extra_metrics(),
            "leakage_checked": self.leakage_checked,
        }


@dataclass(frozen=True, slots=True)
class V02WalkForwardResult:
    """All walk-forward folds plus the aggregated out-of-sample summary."""

    folds: tuple[V02FoldResult, ...]
    summary: Mapping[str, Any]
    aggregate_test_metrics: BacktestMetrics
    aggregate_test_trade_details: Mapping[str, Mapping[str, Any]]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        from intraday_scalper.backtest.metrics import to_jsonable

        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "summary": to_jsonable(self.summary),
            "aggregate_test_metrics": self.aggregate_test_metrics.to_dict(),
            "aggregate_test_trade_details": to_jsonable(self.aggregate_test_trade_details),
            "notes": list(self.notes),
        }


def _collect_5m_bars(data: V02Data, dates: Sequence[date]) -> list[Bar]:
    wanted = set(dates)
    collected: list[Bar] = []
    for session in data.sessions:
        if session.et_date not in wanted:
            continue
        for symbol_session in session.symbols.values():
            collected.extend(bucket.bar for bucket in symbol_session.buckets)
    collected.sort(key=lambda bar: bar.start_utc)
    return collected


def run_v02_walkforward(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
    initial_equity: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> V02WalkForwardResult:
    """Strict chronological walk-forward using the production split helpers.

    The fold grid is exactly the one V0.1 used (``walk_forward_splits`` with
    train 120 / test 20 / step 20), and :func:`assert_no_leakage` re-validates
    every fold on the real 5m bar timestamps.  Each run (train and test) gets a
    **fresh** ``RiskManager`` and a fresh replay state; P&L is never compounded
    across folds.
    """

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    equity = float(
        app_config.risk.account_equity if initial_equity is None else initial_equity
    )
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError("initial_equity must be finite and > 0")

    dates = list(prepared.dates)
    splits = walk_forward_splits(dates, train_days, test_days, step_days)
    notes: list[str] = []
    if not splits:
        empty = compute_metrics([], [], initial_equity=equity)
        notes.append(
            "WALK-FORWARD INCOMPLETE: not enough session dates for one full fold "
            f"(have {len(dates)}, need {train_days}+{test_days})"
        )
        return V02WalkForwardResult(
            folds=(),
            summary={
                "folds": 0,
                "train_days": int(train_days),
                "test_days": int(test_days),
                "step_days": int(step_days),
                "session_dates": len(dates),
                "total_test_trades": 0,
            },
            aggregate_test_metrics=empty,
            aggregate_test_trade_details={},
            notes=tuple(notes),
        )

    fold_results: list[V02FoldResult] = []
    all_test_trades: list[TradeRecord] = []
    all_test_details: dict[str, dict[str, Any]] = {}
    all_train_trades: list[TradeRecord] = []
    breakers = 0
    ambiguous = 0
    for position, fold in enumerate(splits, start=1):
        train_bars = _collect_5m_bars(prepared, fold.train_dates)
        test_bars = _collect_5m_bars(prepared, fold.test_dates)
        assert_no_leakage(train_bars, test_bars)
        train_result = run_v02_backtest(
            prepared,
            app_config,
            risk=RiskManager(app_config),  # fresh state for the fold
            initial_equity=equity,
            session_dates=list(fold.train_dates),
        )
        test_result = run_v02_backtest(
            prepared,
            app_config,
            risk=RiskManager(app_config),
            initial_equity=equity,
            session_dates=list(fold.test_dates),
        )
        all_train_trades.extend(train_result.trades)
        all_test_trades.extend(test_result.trades)
        all_test_details.update(test_result.trade_details)
        breakers += test_result.breaker_days
        ambiguous += test_result.ambiguous_bars
        fold_results.append(
            V02FoldResult(
                fold=fold,
                train_result=train_result,
                test_result=test_result,
                leakage_checked=True,
            )
        )
        if progress is not None:
            progress(position, len(splits))

    aggregate = compute_metrics(
        all_test_trades,
        [],
        initial_equity=equity,
        trade_details=all_test_details,
    )
    test_net = sum(trade.pnl for trade in all_test_trades)
    fold_pnls = [fold.test_result.metrics.net_pnl_dollars for fold in fold_results]
    fold_drawdowns = [fold.test_result.metrics.max_drawdown_dollars for fold in fold_results]
    fold_wins = sum(1 for pnl in fold_pnls if pnl > 0.0)
    summary = {
        "folds": len(fold_results),
        "train_days": int(train_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "session_dates": len(dates),
        "total_train_trades": len(all_train_trades),
        "total_test_trades": aggregate.total_trades,
        "test_wins": aggregate.wins,
        "test_win_rate": aggregate.win_rate,
        "test_profit_factor": aggregate.profit_factor,
        "test_net_pnl_dollars": test_net,
        "test_avg_r": aggregate.avg_r,
        "test_max_drawdown_dollars_max_fold": max(fold_drawdowns) if fold_drawdowns else 0.0,
        "test_max_drawdown_pct_max_fold": max(
            (fold.test_result.metrics.max_drawdown_pct for fold in fold_results),
            default=0.0,
        ),
        "positive_folds": fold_wins,
        "best_fold_pnl": max(fold_pnls) if fold_pnls else 0.0,
        "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
        "test_breaker_days_total": breakers,
        "test_ambiguous_bars_total": ambiguous,
        "test_trades_per_session": (
            len(all_test_trades) / sum(len(fold.fold.test_dates) for fold in fold_results)
            if fold_results
            else 0.0
        ),
    }
    notes.append(
        "each fold starts from the same initial equity (no compounding across folds); "
        "train metrics are in-sample diagnostics, only test metrics are out-of-sample"
    )
    notes.append(DATA_CAVEAT)
    return V02WalkForwardResult(
        folds=tuple(fold_results),
        summary=summary,
        aggregate_test_metrics=aggregate,
        aggregate_test_trade_details=all_test_details,
        notes=tuple(notes),
    )
