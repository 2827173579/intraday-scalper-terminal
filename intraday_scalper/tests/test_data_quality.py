"""Offline tests for the data-quality monitor (STALE / INVALID / DATA_UNAVAILABLE).

Fixed instant: Friday 2026-11-06 14:30:30 UTC (09:30:30 ET). The clock is fully
injected so no test depends on wall time.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import DataStatus, Tick, load_config
from intraday_scalper.providers.data_quality import DataQualityMonitor

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET
FIXED_NOW = SESSION_START + timedelta(seconds=30)
SESSION_DAY = SESSION_START.date()
NEXT_DAY_START = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)  # Monday, US standard time
STALENESS = load_config().data.staleness_seconds  # 90 s
MAX_TICK_AGE = load_config().data.max_tick_age_seconds  # 300 s


class FakeClock:
    def __init__(self, now: datetime = FIXED_NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def make_tick(
    price: float = 40.0,
    volume: float = 100.0,
    ts: datetime | None = None,
    symbol: str = "SOXL",
    trade_id: int | None = None,
) -> Tick:
    return Tick(
        symbol=symbol,
        price=price,
        volume=volume,
        ts_utc=ts if ts is not None else SESSION_START,
        trade_id=trade_id,
    )


def make_monitor(symbols=("SOXL", "SOXS"), clock: FakeClock | None = None, data_config=None):
    return DataQualityMonitor(
        tuple(symbols), data_config or load_config().data, clock=clock or FakeClock()
    )


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
def test_no_ticks_at_all_is_stale():
    monitor = make_monitor()
    report = monitor.report()
    assert report.aggregate is DataStatus.STALE
    assert set(report.per_symbol) == {"SOXL", "SOXS"}
    for quality in report.per_symbol.values():
        assert quality.status is DataStatus.STALE
        assert quality.last_tick_utc is None
        assert quality.stale_seconds is None
        assert quality.tick_count == 0
    assert "no fresh trades" in report.reason


def test_fresh_tick_makes_symbol_ok():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=25)))

    report = monitor.report()
    soxl = report.per_symbol["SOXL"]
    assert soxl.status is DataStatus.OK
    assert soxl.stale_seconds == pytest.approx(5.0)
    assert soxl.tick_count == 1
    assert soxl.last_tick_utc == SESSION_START + timedelta(seconds=25)
    assert report.per_symbol["SOXS"].status is DataStatus.STALE
    # At least one live symbol keeps the aggregate usable; the engine still
    # gates each symbol on its own status.
    assert report.aggregate is DataStatus.OK
    assert monitor.aggregate_status is DataStatus.OK


def test_stale_when_older_than_threshold():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))
    monitor.record_tick(make_tick(6.0, 100.0, SESSION_START, symbol="SOXS"))

    clock.advance(STALENESS + 1.0)
    report = monitor.report()
    assert report.aggregate is DataStatus.STALE
    for quality in report.per_symbol.values():
        assert quality.status is DataStatus.STALE


def test_exactly_at_staleness_threshold_is_still_ok():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, FIXED_NOW))
    clock.advance(STALENESS)
    report = monitor.report()
    assert report.per_symbol["SOXL"].status is DataStatus.OK
    assert report.per_symbol["SOXL"].stale_seconds == pytest.approx(STALENESS)


# --------------------------------------------------------------------------- #
# Duplicates / ordering / invalid data
# --------------------------------------------------------------------------- #
def test_duplicate_trade_id_is_rejected():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START, trade_id=7))
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START, trade_id=7))

    soxl = monitor.report().per_symbol["SOXL"]
    assert soxl.tick_count == 1
    assert soxl.duplicates == 1
    assert soxl.last_reject_reason == "duplicate tick"

    # A duplicate must not refresh the freshness clock.
    clock.advance(STALENESS + 5.0)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START, trade_id=7))
    assert monitor.report().per_symbol["SOXL"].status is DataStatus.STALE


def test_identical_price_time_volume_is_a_duplicate():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))
    monitor.record_tick(make_tick(41.0, 100.0, SESSION_START + timedelta(seconds=1)))
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))  # replayed earlier print

    soxl = monitor.report().per_symbol["SOXL"]
    assert soxl.tick_count == 2
    assert soxl.duplicates == 1


def test_out_of_order_tick_is_counted_not_applied():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    latest = SESSION_START + timedelta(seconds=20)
    monitor.record_tick(make_tick(40.0, 100.0, latest))
    monitor.record_tick(make_tick(39.9, 100.0, SESSION_START + timedelta(seconds=5)))

    soxl = monitor.report().per_symbol["SOXL"]
    assert soxl.tick_count == 1
    assert soxl.out_of_order == 1
    assert soxl.last_tick_utc == latest
    assert soxl.status is DataStatus.OK


def test_future_tick_beyond_max_age_is_invalid():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(
        make_tick(40.0, 100.0, FIXED_NOW + timedelta(seconds=MAX_TICK_AGE + 1))
    )
    report = monitor.report()
    soxl = report.per_symbol["SOXL"]
    assert soxl.tick_count == 0
    assert soxl.invalid == 1
    assert soxl.status is DataStatus.INVALID
    assert report.aggregate is DataStatus.INVALID


def test_too_old_tick_is_invalid():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(
        make_tick(40.0, 100.0, FIXED_NOW - timedelta(seconds=MAX_TICK_AGE + 1))
    )
    soxl = monitor.report().per_symbol["SOXL"]
    assert soxl.tick_count == 0
    assert soxl.invalid == 1
    assert soxl.status is DataStatus.INVALID


def test_severe_out_of_order_marks_symbol_invalid():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=20)))
    for delta in range(5):
        monitor.record_tick(
            make_tick(40.0 + delta, 100.0, SESSION_START + timedelta(seconds=5 - delta))
        )
    report = monitor.report()
    soxl = report.per_symbol["SOXL"]
    assert soxl.out_of_order >= 5
    assert soxl.status is DataStatus.INVALID
    assert report.aggregate is DataStatus.INVALID


def test_invalid_symbol_dominates_aggregate():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START + timedelta(seconds=25)))
    monitor.record_tick(
        make_tick(6.0, 100.0, FIXED_NOW + timedelta(seconds=MAX_TICK_AGE + 5), symbol="SOXS")
    )
    report = monitor.report()
    assert report.per_symbol["SOXL"].status is DataStatus.OK
    assert report.per_symbol["SOXS"].status is DataStatus.INVALID
    assert report.aggregate is DataStatus.INVALID


# --------------------------------------------------------------------------- #
# Reset / day rollover
# --------------------------------------------------------------------------- #
def test_reset_clears_everything():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START, trade_id=1))
    assert monitor.et_day == SESSION_DAY
    monitor.reset()
    report = monitor.report()
    assert monitor.et_day is None
    assert report.aggregate is DataStatus.STALE
    assert report.per_symbol["SOXL"].tick_count == 0


def test_reset_day_is_a_noop_within_the_same_day():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))
    assert monitor.reset_day(SESSION_DAY) is False
    assert monitor.report().per_symbol["SOXL"].tick_count == 1


def test_reset_day_rolls_over_and_clears_counters():
    clock = FakeClock()
    monitor = make_monitor(clock=clock)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))
    assert monitor.reset_day(NEXT_DAY_START.date()) is True
    assert monitor.et_day == NEXT_DAY_START.date()
    assert monitor.report().per_symbol["SOXL"].tick_count == 0


def test_new_et_day_is_detected_automatically():
    # A cross-day tick is only accepted with a permissive max_tick_age; the point
    # here is the automatic ET-day rollover, not staleness.
    data_config = replace(load_config().data, max_tick_age_seconds=1_000_000.0)
    clock = FakeClock(NEXT_DAY_START + timedelta(seconds=30))
    monitor = make_monitor(clock=clock, data_config=data_config)
    monitor.record_tick(make_tick(40.0, 100.0, SESSION_START))  # previous day
    assert monitor.et_day == SESSION_DAY
    monitor.record_tick(make_tick(41.0, 100.0, NEXT_DAY_START))  # new day
    report = monitor.report()
    assert monitor.et_day == NEXT_DAY_START.date()
    assert report.per_symbol["SOXL"].tick_count == 1
    assert report.per_symbol["SOXL"].last_tick_utc == NEXT_DAY_START


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def test_unknown_symbols_are_ignored():
    monitor = make_monitor(("SOXL",))
    monitor.record_tick(make_tick(6.0, 100.0, SESSION_START, symbol="SOXS"))
    assert monitor.ignored_symbols == 1
    assert set(monitor.report().per_symbol) == {"SOXL"}


def test_report_is_dict_serialisable():
    monitor = make_monitor()
    report = monitor.report()
    payload = report.to_dict()
    assert payload["aggregate"] == "STALE"
    assert payload["checked_utc"] == FIXED_NOW.isoformat()
    assert payload["per_symbol"]["SOXL"]["status"] == "STALE"
    assert payload["per_symbol"]["SOXL"]["tick_count"] == 0


def test_report_requires_aware_now():
    monitor = make_monitor()
    with pytest.raises(ValueError):
        monitor.report(datetime(2026, 11, 6, 14, 30))
    with pytest.raises(TypeError):
        monitor.report("now")  # type: ignore[arg-type]


def test_record_tick_type_error():
    monitor = make_monitor()
    with pytest.raises(TypeError):
        monitor.record_tick("tick")  # type: ignore[arg-type]


def test_constructor_validation():
    data_config = load_config().data
    with pytest.raises(ValueError):
        DataQualityMonitor([], data_config)
    with pytest.raises(ValueError):
        DataQualityMonitor(["SOXL", "SOXL"], data_config)
