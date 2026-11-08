"""Offline tests for the extended-hours (pre/post market) provider helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from intraday_scalper.core.models import Bar
from intraday_scalper.providers import extended_hours_provider as ext

ET = ZoneInfo("America/New_York")
SESSION = date(2026, 11, 6)


def _bar(day: date, hour: int, minute: int, *, o: float, h: float, l: float, c: float, volume: float = 100.0) -> Bar:
    start_et = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)
    start = start_et.astimezone(timezone.utc)
    return Bar(
        symbol="SOXL",
        timeframe_minutes=5,
        start_utc=start,
        end_utc=start + timedelta(minutes=5),
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
        vwap=(h + l + c) / 3.0,
        trade_count=0,
    )


def test_phase_of_windows():
    assert ext.phase_of(datetime(2026, 11, 6, 4, 0, tzinfo=ET)) == "pre"
    assert ext.phase_of(datetime(2026, 11, 6, 9, 30, tzinfo=ET)) == "regular"
    assert ext.phase_of(datetime(2026, 11, 6, 16, 0, tzinfo=ET)) == "post"
    assert ext.phase_of(datetime(2026, 11, 6, 21, 0, tzinfo=ET)) == "closed"


def test_pre_market_anchors_are_per_session():
    day1 = date(2026, 11, 5)
    bars = [
        _bar(day1, 4, 0, o=100, h=101, l=99, c=100.5),
        _bar(day1, 9, 25, o=100, h=102, l=98, c=101),
        _bar(SESSION, 4, 0, o=200, h=201, l=199, c=200.5),
        _bar(SESSION, 9, 25, o=200, h=205, l=195, c=204),
    ]
    anchors = ext.build_pre_market_anchors(bars, SESSION)
    assert anchors["pm_high"] == pytest.approx(205.0)
    assert anchors["pm_low"] == pytest.approx(195.0)
    assert anchors["bars"] == 2


def test_session_vwap_uses_volume_and_mean_fallback():
    bars = [
        _bar(SESSION, 9, 30, o=10, h=10, l=10, c=10, volume=100),
        _bar(SESSION, 9, 35, o=20, h=20, l=20, c=20, volume=100),
    ]
    assert ext.session_vwap(bars) == pytest.approx(15.0)
    zero_volume = [_bar(SESSION, 9, 30, o=10, h=10, l=10, c=10, volume=0), _bar(SESSION, 9, 35, o=20, h=20, l=20, c=20, volume=0)]
    assert ext.session_vwap(zero_volume) is None
    assert ext.mean_price(zero_volume) == pytest.approx(15.0)


def _regular_uptrend_with_pm_sweep() -> list[Bar]:
    bars: list[Bar] = []
    # pre-market: PM low = 100
    for index in range(20):
        price = 102 + index * 0.1
        bars.append(_bar(SESSION, 4, 0 + index, o=price, h=price + 0.2, l=price - 0.2, c=price))
    bars.append(_bar(SESSION, 9, 25, o=103, h=103.2, l=100.0, c=102.8))
    # regular: warm EMA21 above 100, then sweep below PM low and reclaim
    for index in range(20):
        price = 103 + index * 0.1
        bars.append(_bar(SESSION, 9, 30, o=price, h=price + 0.2, l=price - 0.2, c=price))
    bars.append(_bar(SESSION, 9, 40, o=104.0, h=106.5, l=99.0, c=105.5, volume=200))
    return bars


def test_smc_pm_low_sweep_signal_detected():
    bars = _regular_uptrend_with_pm_sweep()
    signals = ext.evaluate_extended_signals("SOXL", bars, session_date=SESSION)
    sweeps = [signal for signal in signals if signal["type"] == "SMC_PM_LOW_SWEEP"]
    assert sweeps, f"expected PM low sweep, got {[s['type'] for s in signals]}"
    sweep = sweeps[0]
    assert sweep["stop"] < sweep["entry"]
    assert sweep["target"] > sweep["entry"]


def test_signals_are_limited_to_target_session():
    bars = _regular_uptrend_with_pm_sweep()
    later = [bar.start_utc + timedelta(days=1) for bar in bars]
    # rebuild with dates shifted one day later; evaluating SESSION must return no signals
    shifted = []
    for bar, start in zip(bars, later):
        shifted.append(
            Bar(
                symbol="SOXL",
                timeframe_minutes=5,
                start_utc=start,
                end_utc=start + timedelta(minutes=5),
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=0,
            )
        )
    assert ext.evaluate_extended_signals("SOXL", shifted, session_date=SESSION) == []


def test_pre_market_ema_pullback_has_spacing_cooldown():
    bars: list[Bar] = []
    for index in range(30):
        minute = 30 + index * 5
        hour = 7 + minute // 60
        clock_minute = minute % 60
        base = 100 + index * 0.2
        bars.append(_bar(SESSION, hour, clock_minute, o=base, h=base + 0.3, l=base - 0.5, c=base + 0.2))
    signals = ext.evaluate_extended_signals("SOXL", bars, session_date=SESSION)
    pre_pullbacks = [s for s in signals if s["type"] == "EMA_PULLBACK" and s["window"] == "pre"]
    if len(pre_pullbacks) > 1:
        times = sorted(datetime.fromisoformat(s["time_utc"]) for s in pre_pullbacks)
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
        assert min(gaps) >= 20 * 60 - 1


def test_build_extended_context_filters_to_session():
    bars = [_bar(date(2026, 11, 5), 9, 30, o=10, h=10.2, l=9.8, c=10.0), _bar(SESSION, 9, 30, o=20, h=20.2, l=19.8, c=20.0)]
    context = ext.build_extended_context("SOXL", bars, session_date=SESSION)
    assert context["session_date"] == SESSION.isoformat()
    assert len(context["bars"]) == 1
    assert context["bars"][0]["phase"] == "regular"


def test_merge_quote_updates_current_5m_bucket():
    from datetime import datetime, timezone

    bucket = datetime(2026, 11, 6, 15, 30, tzinfo=timezone.utc)
    bars = [
        Bar(
            symbol="SOXL",
            timeframe_minutes=5,
            start_utc=bucket,
            end_utc=bucket + timedelta(minutes=5),
            open=46.0,
            high=46.2,
            low=45.9,
            close=46.0,
            volume=0.0,
            vwap=46.03,
            trade_count=0,
        )
    ]
    merged = ext.merge_quote_into_extended_bars(
        "SOXL", bars, 46.5, now=datetime(2026, 11, 6, 15, 33, tzinfo=timezone.utc)
    )
    assert merged[-1].close == pytest.approx(46.5)
    assert merged[-1].high == pytest.approx(46.5)
    assert merged[-1].low == pytest.approx(45.9)
    assert merged[-1].start_utc == bucket


def test_merge_quote_appends_into_new_bucket_when_needed():
    from datetime import datetime, timezone

    old_bucket = datetime(2026, 11, 6, 15, 25, tzinfo=timezone.utc)
    new_bucket = datetime(2026, 11, 6, 15, 30, tzinfo=timezone.utc)
    bars = [
        Bar(
            symbol="SOXS",
            timeframe_minutes=5,
            start_utc=old_bucket,
            end_utc=old_bucket + timedelta(minutes=5),
            open=70.0,
            high=70.2,
            low=69.9,
            close=70.0,
            volume=0.0,
            vwap=70.03,
            trade_count=0,
        )
    ]
    merged = ext.merge_quote_into_extended_bars(
        "SOXS", bars, 70.4, now=datetime(2026, 11, 6, 15, 32, tzinfo=timezone.utc)
    )
    assert len(merged) == 2
    assert merged[-1].start_utc == new_bucket
    assert merged[-1].close == pytest.approx(70.4)
