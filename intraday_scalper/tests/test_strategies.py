"""Offline, deterministic tests for the three phase-1 strategies and the registry.

Every scenario is built from synthetic bars anchored to the fixed session
``2026-11-06 14:30 UTC`` (09:30 ET, US standard time). Contexts are explicit
about their indicator values so a test failure points at one conditional
rather than at an indicator warm-up. No network / clock / local data is used.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import (
    Bar,
    Mode,
    SessionPhase,
    StrategyName,
    load_config,
)
from intraday_scalper.features import OpeningRange, opening_range
from intraday_scalper.strategies import (
    EmaTrendPullback,
    OpeningRangeBreakout,
    Strategy,
    StrategyContext,
    VwapReclaim,
    build_strategies,
)

UTC = timezone.utc
SESSION_OPEN = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET (EST)
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)

CONFIG = load_config()
ORB5 = OpeningRangeBreakout(CONFIG.strategies.orb, timeframe_minutes=5)
ORB15 = OpeningRangeBreakout(CONFIG.strategies.orb, timeframe_minutes=15)
VWAP_RECLAIM = VwapReclaim(CONFIG.strategies.vwap_reclaim)
EMA_PULLBACK = EmaTrendPullback(CONFIG.strategies.ema_pullback)


# --------------------------------------------------------------------------- #
# Synthetic bar / context helpers
# --------------------------------------------------------------------------- #
def make_bar(
    start_minute: int = 0,
    *,
    open_: float = 40.0,
    high: float = 40.2,
    low: float = 39.8,
    close: float = 40.1,
    volume: float = 10_000.0,
    symbol: str = "SOXL",
    is_complete: bool = True,
    timeframe_minutes: int = 1,
    vwap: float | None = None,
) -> Bar:
    if vwap is None:
        vwap = (high + low + close) / 3.0
    start = SESSION_OPEN + timedelta(minutes=start_minute)
    return Bar(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        start_utc=start,
        end_utc=start + timedelta(minutes=timeframe_minutes),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=10,
        is_complete=is_complete,
    )


def make_ctx(bars: list[Bar] | tuple[Bar, ...], **overrides: object) -> StrategyContext:
    defaults: dict[str, object] = {
        "symbol": bars[-1].symbol if bars else "SOXL",
        "now_utc": NOW,
        "session_phase": SessionPhase.REGULAR,
        "mode": Mode.DUAL_WATCH,
        "bars_1m": tuple(bars),
        "vwap": 40.0,
        "ema_fast": 40.10,
        "ema_slow": 40.00,
        "rsi": 60.0,
        "atr": 0.05,
        "avg_volume_20": 10_000.0,
        "last_price": bars[-1].close if bars else None,
    }
    defaults.update(overrides)
    return StrategyContext(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Scenario A: opening-range breakout
# --------------------------------------------------------------------------- #
def orb_bars(
    *,
    signal: tuple[float, float, float, float, float] = (40.05, 40.35, 40.00, 40.30, 30_000.0),
) -> list[Bar]:
    """Five OR minutes (OR high 40.10 / low 39.90) plus the breakout minute."""
    opening_specs = [
        (40.00, 40.10, 39.90, 40.00),
        (40.00, 40.05, 39.92, 39.95),
        (39.95, 40.08, 39.90, 40.02),
        (40.02, 40.06, 39.93, 39.98),
        (39.98, 40.04, 39.91, 40.01),
    ]
    bars = [
        make_bar(minute, open_=open_, high=high, low=low, close=close)
        for minute, (open_, high, low, close) in enumerate(opening_specs)
    ]
    open_, high, low, close, volume = signal
    bars.append(make_bar(5, open_=open_, high=high, low=low, close=close, volume=volume))
    return bars


def orb_ctx(bars: list[Bar], **overrides: object) -> StrategyContext:
    overrides.setdefault("opening_range_5m", opening_range(bars, SESSION_OPEN, 5))
    overrides.setdefault("ema_slow", 39.50)
    return make_ctx(bars, **overrides)


def complete_or(
    *, minutes: int = 5, high: float = 40.10, low: float = 39.90, complete: bool = True
) -> OpeningRange:
    return OpeningRange(
        high=high,
        low=low,
        start_utc=SESSION_OPEN,
        end_utc=SESSION_OPEN + timedelta(minutes=minutes),
        complete=complete,
        bar_count=minutes if complete else max(minutes - 2, 0),
    )


def test_orb_5m_triggers_on_confirmed_breakout() -> None:
    candidate = ORB5.evaluate(orb_ctx(orb_bars()))
    assert candidate is not None
    assert candidate.strategy is StrategyName.ORB_5M
    assert candidate.symbol == "SOXL"
    assert candidate.entry_ref == pytest.approx(40.30)
    assert candidate.confirmation_bar_start_utc == SESSION_OPEN + timedelta(minutes=5)
    assert candidate.created_utc == NOW
    # Structural stop = min(OR high 40.10, bar low 40.00) - ATR 0.05 * 1.0 = 39.95
    assert candidate.stop_loss == pytest.approx(39.95)
    assert candidate.risk_per_share == pytest.approx(0.35)
    assert candidate.target_r == pytest.approx(2.0)
    assert candidate.take_profit_price == pytest.approx(41.00)
    assert candidate.reward_risk == pytest.approx(2.0)
    assert candidate.rationale
    assert candidate.structural_note


def test_orb_5m_rejects_weak_body() -> None:
    # body 0.25 / range 0.82 = 0.30 < min_body_ratio 0.55
    candidate = ORB5.evaluate(orb_ctx(orb_bars(signal=(40.05, 40.80, 39.98, 40.30, 30_000.0))))
    assert candidate is None


def test_orb_5m_rejects_low_volume() -> None:
    # 14,000 < 1.5 * 10,000
    candidate = ORB5.evaluate(orb_ctx(orb_bars(signal=(40.05, 40.35, 40.00, 40.30, 14_000.0))))
    assert candidate is None


def test_orb_5m_rejects_insufficient_breakout_buffer() -> None:
    # close 40.11 <= OR high 40.10 * (1 + 0.0005) = 40.12005
    candidate = ORB5.evaluate(orb_ctx(orb_bars(signal=(40.00, 40.11, 40.00, 40.11, 30_000.0))))
    assert candidate is None


def test_orb_5m_rejects_trend_mismatch() -> None:
    bars = orb_bars()
    assert ORB5.evaluate(orb_ctx(bars)) is not None  # baseline passes
    assert ORB5.evaluate(orb_ctx(bars, ema_slow=41.00)) is None


def test_orb_5m_rejects_incomplete_opening_range() -> None:
    bars = orb_bars()
    candidate = ORB5.evaluate(orb_ctx(bars, opening_range_5m=complete_or(complete=False)))
    assert candidate is None


def test_orb_5m_rejects_missing_opening_range() -> None:
    assert ORB5.evaluate(orb_ctx(orb_bars(), opening_range_5m=None)) is None


def test_orb_5m_rejects_bar_inside_the_opening_range() -> None:
    bars = orb_bars()[:4]
    bars[3] = make_bar(3, open_=40.05, high=40.35, low=40.00, close=40.30, volume=30_000.0)
    ctx = make_ctx(bars, opening_range_5m=complete_or(), ema_slow=39.50)
    # The latest bar starts at 14:33, inside the 14:30-14:35 opening range.
    assert ORB5.evaluate(ctx) is None


def test_orb_5m_rejects_premarket_signal_bar() -> None:
    premarket = make_bar(-60, open_=44.9, high=45.2, low=44.8, close=45.0, volume=30_000.0)
    ctx = make_ctx([premarket], opening_range_5m=complete_or(), ema_slow=39.50)
    assert ORB5.evaluate(ctx) is None


def test_orb_5m_rejects_incomplete_confirmation_bar() -> None:
    bars = orb_bars()
    bars[-1] = make_bar(
        5, open_=40.05, high=40.35, low=40.00, close=40.30, volume=30_000.0, is_complete=False
    )
    ctx = make_ctx(
        bars,
        opening_range_5m=complete_or(),
        ema_slow=39.50,
    )
    assert ORB5.evaluate(ctx) is None


def test_orb_5m_rejects_missing_atr() -> None:
    assert ORB5.evaluate(orb_ctx(orb_bars(), atr=None)) is None


def test_orb_15m_triggers_on_confirmed_breakout() -> None:
    opening_specs = [
        (40.00, 40.10 if minute == 0 else 40.05, 39.90 if minute == 0 else 39.95, 40.00)
        for minute in range(15)
    ]
    bars = [
        make_bar(minute, open_=open_, high=high, low=low, close=close)
        for minute, (open_, high, low, close) in enumerate(opening_specs)
    ]
    bars.append(make_bar(15, open_=40.05, high=40.35, low=40.00, close=40.30, volume=30_000.0))
    ctx = make_ctx(
        bars,
        opening_range_15m=opening_range(bars, SESSION_OPEN, 15),
        opening_range_5m=None,
        ema_slow=39.50,
    )
    candidate = ORB15.evaluate(ctx)
    assert candidate is not None
    assert candidate.strategy is StrategyName.ORB_15M
    assert candidate.confirmation_bar_start_utc == SESSION_OPEN + timedelta(minutes=15)


def test_orb_15m_rejects_incomplete_range() -> None:
    bars = orb_bars()
    candidate = ORB15.evaluate(
        make_ctx(bars, opening_range_15m=complete_or(minutes=15, complete=False), ema_slow=39.50)
    )
    assert candidate is None


def test_orb_invalid_timeframe_rejected() -> None:
    with pytest.raises(ValueError):
        OpeningRangeBreakout(CONFIG.strategies.orb, timeframe_minutes=7)


def test_orb_strategy_names() -> None:
    assert ORB5.name is StrategyName.ORB_5M
    assert ORB15.name is StrategyName.ORB_15M


def orb_bars_long_session(signal_volume: float = 30_000.0) -> list[Bar]:
    """OR (5 bars) + 15 flat minutes + one breakout minute at +20.

    With 20 prior bars the strategy must compute the volume average from those
    bars itself instead of trusting ``ctx.avg_volume_20``.
    """
    bars = orb_bars()[:5]
    for minute in range(5, 20):
        bars.append(
            make_bar(minute, open_=40.20, high=40.25, low=40.15, close=40.20, volume=12_000.0)
        )
    bars.append(
        make_bar(20, open_=40.25, high=40.50, low=40.20, close=40.45, volume=signal_volume)
    )
    return bars


def test_orb_uses_previous_20_bar_volume_average() -> None:
    bars = orb_bars_long_session(signal_volume=30_000.0)
    # Prior 20 volumes: 5 x 10,000 (OR) + 15 x 12,000 = average 11,500.
    # The deliberately wrong context value (1e9) must be ignored.
    ctx = make_ctx(
        bars,
        opening_range_5m=opening_range(bars, SESSION_OPEN, 5),
        ema_slow=39.50,
        avg_volume_20=1_000_000_000.0,
    )
    candidate = ORB5.evaluate(ctx)
    assert candidate is not None
    assert candidate.entry_ref == pytest.approx(40.45)
    assert candidate.confirmation_bar_start_utc == SESSION_OPEN + timedelta(minutes=20)


def test_orb_previous_20_bar_volume_filters_low_volume() -> None:
    # 17,000 < 1.5 * 11,500 = 17,250
    bars = orb_bars_long_session(signal_volume=17_000.0)
    ctx = make_ctx(
        bars,
        opening_range_5m=opening_range(bars, SESSION_OPEN, 5),
        ema_slow=39.50,
        avg_volume_20=1_000_000_000.0,
    )
    assert ORB5.evaluate(ctx) is None


# --------------------------------------------------------------------------- #
# Scenario B: VWAP reclaim
# --------------------------------------------------------------------------- #
def vwap_reclaim_bars(
    *,
    prev: tuple[float, float, float, float] = (40.02, 40.05, 39.90, 39.95),
    latest: tuple[float, float, float, float, float] = (40.05, 40.30, 40.02, 40.25, 20_000.0),
    volume: float = 10_000.0,
) -> list[Bar]:
    """Eight bars at VWAP 40.00, one bar closing below VWAP, one reclaim bar."""
    bars = [
        make_bar(minute, open_=40.00, high=40.20, low=39.85, close=40.02, volume=volume, vwap=40.00)
        for minute in range(8)
    ]
    prev_open, prev_high, prev_low, prev_close = prev
    bars.append(
        make_bar(
            8,
            open_=prev_open,
            high=prev_high,
            low=prev_low,
            close=prev_close,
            volume=volume,
            vwap=40.00,
        )
    )
    open_, high, low, close, trade_volume = latest
    bars.append(
        make_bar(
            9,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=trade_volume,
            vwap=(high + low + close) / 3.0,
        )
    )
    return bars


def test_vwap_reclaim_triggers_on_confirmed_reclaim() -> None:
    candidate = VWAP_RECLAIM.evaluate(make_ctx(vwap_reclaim_bars()))
    assert candidate is not None
    assert candidate.strategy is StrategyName.VWAP_RECLAIM
    assert candidate.entry_ref == pytest.approx(40.25)
    assert candidate.confirmation_bar_start_utc == SESSION_OPEN + timedelta(minutes=9)
    # min(bar low 40.02, VWAP 40.00) - ATR 0.05 * 1.2 = 39.94
    assert candidate.stop_loss == pytest.approx(39.94)
    assert candidate.risk_per_share == pytest.approx(0.31)
    assert candidate.target_r == pytest.approx(2.0)
    assert candidate.take_profit_price == pytest.approx(40.87)


def test_vwap_reclaim_rejects_bar_still_below_vwap() -> None:
    # previous close 40.05 is above the cumulative VWAP of 40.00: no reclaim
    candidate = VWAP_RECLAIM.evaluate(
        make_ctx(vwap_reclaim_bars(prev=(40.02, 40.10, 39.95, 40.05)))
    )
    assert candidate is None


def test_vwap_reclaim_rejects_weak_reclaim_close() -> None:
    # close 40.01 <= VWAP 40.00 * (1 + 0.0005) = 40.02; EMAs relaxed so the
    # buffer check is the only failing condition.
    bars = vwap_reclaim_bars(latest=(40.00, 40.02, 39.99, 40.01, 20_000.0))
    candidate = VWAP_RECLAIM.evaluate(make_ctx(bars, ema_fast=39.95, ema_slow=39.90))
    assert candidate is None


def test_vwap_reclaim_rejects_bearish_bar() -> None:
    bars = vwap_reclaim_bars(latest=(40.30, 40.35, 40.20, 40.25, 20_000.0))
    assert VWAP_RECLAIM.evaluate(make_ctx(bars)) is None


def test_vwap_reclaim_rejects_low_volume() -> None:
    bars = vwap_reclaim_bars(latest=(40.05, 40.30, 40.02, 40.25, 11_999.0))
    assert VWAP_RECLAIM.evaluate(make_ctx(bars)) is None


def test_vwap_reclaim_rejects_ema_not_rising() -> None:
    bars = vwap_reclaim_bars()
    assert VWAP_RECLAIM.evaluate(make_ctx(bars)) is not None
    assert VWAP_RECLAIM.evaluate(make_ctx(bars, ema_fast=39.90, ema_slow=40.00)) is None


def test_vwap_reclaim_rejects_close_below_fast_ema() -> None:
    assert VWAP_RECLAIM.evaluate(make_ctx(vwap_reclaim_bars(), ema_fast=40.30)) is None


def test_vwap_reclaim_rejects_missing_vwap() -> None:
    assert VWAP_RECLAIM.evaluate(make_ctx(vwap_reclaim_bars(), vwap=None)) is None


def test_vwap_reclaim_rejects_missing_atr() -> None:
    assert VWAP_RECLAIM.evaluate(make_ctx(vwap_reclaim_bars(), atr=None)) is None


def test_vwap_reclaim_rejects_bars_without_true_volume() -> None:
    bars = vwap_reclaim_bars(volume=0.0, latest=(40.05, 40.30, 40.02, 40.25, 20_000.0))
    assert VWAP_RECLAIM.evaluate(make_ctx(bars)) is None


def test_vwap_reclaim_name() -> None:
    assert VWAP_RECLAIM.name is StrategyName.VWAP_RECLAIM


# --------------------------------------------------------------------------- #
# Scenario C: EMA pullback
# --------------------------------------------------------------------------- #
def pullback_bars(
    *,
    prev: tuple[float, float, float, float] = (40.30, 40.32, 40.15, 40.22),
    latest: tuple[float, float, float, float] = (40.24, 40.46, 40.20, 40.45),
    trend_close: float | None = None,
) -> list[Bar]:
    warm = [
        make_bar(
            minute,
            open_=40.10,
            high=40.40,
            low=40.05,
            close=40.20 if minute % 2 == 0 else 40.30,
            volume=10_000.0,
        )
        for minute in range(8)
    ]
    if trend_close is not None:
        warm[6] = make_bar(
            6,
            open_=40.10,
            high=40.40,
            low=min(40.05, trend_close) - 0.05,
            close=trend_close,
            volume=10_000.0,
        )
    prev_open, prev_high, prev_low, prev_close = prev
    warm.append(
        make_bar(8, open_=prev_open, high=prev_high, low=prev_low, close=prev_close)
    )
    latest_open, latest_high, latest_low, latest_close = latest
    warm.append(
        make_bar(9, open_=latest_open, high=latest_high, low=latest_low, close=latest_close)
    )
    return warm


def pullback_ctx(bars: list[Bar], **overrides: object) -> StrategyContext:
    overrides.setdefault("vwap", 40.0)
    overrides.setdefault("ema_fast", 40.20)
    overrides.setdefault("ema_slow", 40.10)
    return make_ctx(bars, **overrides)


def test_ema_pullback_triggers_on_confirmed_pullback() -> None:
    candidate = EMA_PULLBACK.evaluate(pullback_ctx(pullback_bars()))
    assert candidate is not None
    assert candidate.strategy is StrategyName.EMA_PULLBACK
    assert candidate.entry_ref == pytest.approx(40.45)
    assert candidate.confirmation_bar_start_utc == SESSION_OPEN + timedelta(minutes=9)
    # min(prev low 40.15, EMA fast 40.20) - ATR 0.05 * 1.0 = 40.10
    assert candidate.stop_loss == pytest.approx(40.10)
    assert candidate.risk_per_share == pytest.approx(0.35)
    assert candidate.target_r == pytest.approx(2.0)
    assert candidate.take_profit_price == pytest.approx(41.15)


def test_ema_pullback_rejects_trend_below_vwap() -> None:
    bars = pullback_bars(trend_close=39.90)
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_ema_not_rising() -> None:
    bars = pullback_bars()
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is not None
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars, ema_fast=40.05, ema_slow=40.10)) is None


def test_ema_pullback_rejects_no_touch_of_fast_ema() -> None:
    # previous low 40.30 > 40.20 * 1.0005 = 40.2201: no pullback into the EMA
    bars = pullback_bars(prev=(40.35, 40.45, 40.30, 40.40))
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_previous_close_under_ema() -> None:
    # previous close 40.10 < 40.20 * 0.9995 = 40.1899: the pullback lost the EMA
    bars = pullback_bars(prev=(40.30, 40.32, 40.05, 40.10))
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_weak_confirmation_body() -> None:
    # body 0.21 / range 0.70 = 0.30 < confirmation_body_ratio 0.35
    bars = pullback_bars(latest=(40.24, 40.90, 40.20, 40.45))
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_close_not_above_pullback_high() -> None:
    bars = pullback_bars(latest=(40.24, 40.31, 40.20, 40.30))
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_bearish_confirmation_bar() -> None:
    bars = pullback_bars(latest=(40.50, 40.52, 40.40, 40.45))
    assert EMA_PULLBACK.evaluate(pullback_ctx(bars)) is None


def test_ema_pullback_rejects_missing_atr() -> None:
    assert EMA_PULLBACK.evaluate(pullback_ctx(pullback_bars(), atr=None)) is None


def test_ema_pullback_rejects_missing_vwap() -> None:
    assert EMA_PULLBACK.evaluate(pullback_ctx(pullback_bars(), vwap=None)) is None


def test_ema_pullback_rejects_insufficient_history() -> None:
    assert EMA_PULLBACK.evaluate(pullback_ctx(pullback_bars()[:3])) is None


def test_ema_pullback_name() -> None:
    assert EMA_PULLBACK.name is StrategyName.EMA_PULLBACK


# --------------------------------------------------------------------------- #
# Base-class candidate builder: ATR distance + stop clamps
# --------------------------------------------------------------------------- #
def build_candidate(strategy: Strategy, **overrides: object):
    kwargs: dict[str, object] = {
        "symbol": "SOXL",
        "entry_ref": 40.0,
        "structural_stop": 39.0,
        "target_r": 2.0,
        "atr": 0.10,
        "min_stop_pct": 0.0025,
        "max_stop_pct": 0.02,
        "rationale": "unit-test",
        "structural_note": "unit-test",
        "now_utc": NOW,
        "confirmation_bar_start_utc": SESSION_OPEN,
    }
    kwargs.update(overrides)
    return strategy._build_candidate(**kwargs)  # type: ignore[arg-type]


def test_build_candidate_clamps_to_max_stop_pct() -> None:
    # raw stop = 39.0 - 10.0 * 1.0 = 29.0 -> clamped to 40.0 * (1 - 0.02) = 39.2
    candidate = build_candidate(ORB5, structural_stop=39.0, atr=10.0)
    assert candidate is not None
    assert candidate.stop_loss == pytest.approx(39.2)


def test_build_candidate_clamps_to_min_stop_pct() -> None:
    # raw stop = 40.0 (ATR 0) -> clamped to 40.0 * (1 - 0.0025) = 39.9
    candidate = build_candidate(ORB5, structural_stop=40.0, atr=0.0)
    assert candidate is not None
    assert candidate.stop_loss == pytest.approx(39.9)


def test_build_candidate_uses_structural_stop_when_atr_missing() -> None:
    candidate = build_candidate(ORB5, structural_stop=39.8, atr=None)
    assert candidate is not None
    assert candidate.stop_loss == pytest.approx(39.8)


def test_build_candidate_rejects_stop_at_or_above_entry() -> None:
    candidate = build_candidate(ORB5, structural_stop=40.0, atr=0.0, min_stop_pct=0.0)
    assert candidate is None


def test_build_candidate_rejects_non_finite_inputs() -> None:
    assert build_candidate(ORB5, structural_stop=float("nan")) is None
    assert build_candidate(ORB5, entry_ref=float("inf")) is None
    assert build_candidate(ORB5, target_r=0.0) is None


# --------------------------------------------------------------------------- #
# No-look-ahead guarantees at the strategy level
# --------------------------------------------------------------------------- #
def test_strategies_ignore_incomplete_confirmation_bar() -> None:
    orb = orb_bars()
    orb[-1] = make_bar(
        5, open_=40.05, high=40.35, low=40.00, close=40.30, volume=30_000.0, is_complete=False
    )
    assert ORB5.evaluate(make_ctx(orb, opening_range_5m=complete_or(), ema_slow=39.50)) is None

    vwap = vwap_reclaim_bars()
    vwap[-1] = make_bar(
        9, open_=40.05, high=40.30, low=40.02, close=40.25, volume=20_000.0, is_complete=False
    )
    assert VWAP_RECLAIM.evaluate(make_ctx(vwap)) is None

    pullback = pullback_bars()
    pullback[-1] = make_bar(
        9, open_=40.24, high=40.46, low=40.20, close=40.45, is_complete=False
    )
    assert EMA_PULLBACK.evaluate(pullback_ctx(pullback)) is None


def test_orb_decision_does_not_depend_on_5m_bars() -> None:
    bars = orb_bars()
    plain = ORB5.evaluate(orb_ctx(bars))
    assert plain is not None
    five_minute_bar = make_bar(
        0, open_=10.0, high=99.0, low=1.0, close=50.0, timeframe_minutes=5
    )
    with_5m = ORB5.evaluate(orb_ctx(bars, bars_5m=(five_minute_bar,)))
    assert with_5m is not None
    assert with_5m.to_dict() == plain.to_dict()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_builds_canonical_order(config) -> None:
    strategies = build_strategies(config.strategies)
    assert [strategy.name for strategy in strategies] == [
        StrategyName.ORB_5M,
        StrategyName.ORB_15M,
        StrategyName.VWAP_RECLAIM,
        StrategyName.EMA_PULLBACK,
    ]
    assert all(isinstance(strategy, Strategy) for strategy in strategies)


def test_registry_respects_enabled_flags(config) -> None:
    strategy_config = config.strategies
    disabled = replace(
        strategy_config,
        orb=replace(strategy_config.orb, enabled_15m=False),
        vwap_reclaim=replace(strategy_config.vwap_reclaim, enabled=False),
        ema_pullback=replace(strategy_config.ema_pullback, enabled=False),
    )
    assert [strategy.name for strategy in build_strategies(disabled)] == [StrategyName.ORB_5M]


def test_registry_all_disabled_returns_empty_list(config) -> None:
    strategy_config = config.strategies
    disabled = replace(
        strategy_config,
        orb=replace(strategy_config.orb, enabled_5m=False, enabled_15m=False),
        vwap_reclaim=replace(strategy_config.vwap_reclaim, enabled=False),
        ema_pullback=replace(strategy_config.ema_pullback, enabled=False),
    )
    assert build_strategies(disabled) == []


def test_registry_rejects_wrong_type(config) -> None:
    with pytest.raises(TypeError):
        build_strategies(config)  # AppConfig, not StrategyConfig
