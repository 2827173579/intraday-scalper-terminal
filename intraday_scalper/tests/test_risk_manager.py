"""Unit tests for :mod:`intraday_scalper.risk.manager` (offline, deterministic).

Time convention: Friday 2026-11-06 is inside US *standard* time, so
09:30 ET == 14:30 UTC and 15:30 ET == 20:30 UTC.  All instants are built from
the ``America/New_York`` zone and converted to UTC, so DST bugs would surface.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from intraday_scalper.core import (
    AlertLevel,
    Mode,
    RiskBlockReason,
    SignalCandidate,
    SignalEventKind,
    StrategyName,
    TradeStatus,
)
from intraday_scalper.risk import RiskManager, size_position

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
TRADING_DAY = date(2026, 11, 6)  # Friday


def et_at(
    hour: int,
    minute: int = 0,
    second: int = 0,
    *,
    day: date = TRADING_DAY,
) -> datetime:
    """Build an exchange-clock instant and return it in UTC."""
    return datetime(
        day.year, day.month, day.day, hour, minute, second, tzinfo=ET
    ).astimezone(UTC)


class MutableClock:
    """Injectable clock: tests move it explicitly, never reading the system time."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def set(self, ts: datetime) -> None:
        self.now = ts


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock(et_at(9, 35))


@pytest.fixture
def manager(config, clock) -> RiskManager:
    return RiskManager(config, clock=clock)


def _candidate(
    now: datetime,
    symbol: str = "SOXL",
    entry: float = 20.0,
    stop: float = 19.5,
    target_r: float = 2.0,
) -> SignalCandidate:
    return SignalCandidate(
        symbol=symbol,
        strategy=StrategyName.ORB_5M,
        entry_ref=entry,
        stop_loss=stop,
        target_r=target_r,
        rationale="risk-manager unit test",
        created_utc=now,
        confirmation_bar_start_utc=now - timedelta(minutes=1),
    )


def _plan(config, candidate: SignalCandidate):
    return size_position(candidate, config.risk)


def _evaluate(
    manager: RiskManager,
    config,
    now: datetime,
    *,
    symbol: str = "SOXL",
    entry: float = 20.0,
    stop: float = 19.5,
    target_r: float = 2.0,
    data_ready: bool = True,
):
    candidate = _candidate(now, symbol, entry, stop, target_r)
    plan = _plan(config, candidate)
    return candidate, plan, manager.evaluate_entry(
        candidate, plan, now, data_ready=data_ready
    )


def _open_position(
    manager: RiskManager,
    config,
    now: datetime,
    *,
    symbol: str = "SOXL",
    entry: float = 20.0,
    stop: float = 19.5,
    signal_id: str = "SIG-TEST-1",
):
    candidate = _candidate(now, symbol, entry, stop)
    plan = _plan(config, candidate)
    decision = manager.evaluate_entry(candidate, plan, now, data_ready=True)
    assert decision.allowed is True, decision.message
    return manager.confirm_entry(
        plan,
        fill_price=entry,
        fill_utc=now,
        signal_id=signal_id,
        mode=manager.mode,
    )


def _lose_one_trade(
    manager: RiskManager,
    config,
    entry_time: datetime,
    *,
    signal_id: str,
    exit_after_minutes: int = 2,
):
    _open_position(manager, config, entry_time, signal_id=signal_id)
    return manager.confirm_exit(
        fill_price=19.5,
        fill_utc=entry_time + timedelta(minutes=exit_after_minutes),
        reason="STOP",
    )


# --------------------------------------------------------------------------- #
# Session / trading-day gates
# --------------------------------------------------------------------------- #
def test_entry_allowed_during_regular_session(manager, config):
    now = et_at(10, 0)
    candidate, plan, decision = _evaluate(manager, config, now)

    assert decision.allowed is True
    assert decision.reason is RiskBlockReason.NONE
    assert decision.plan is plan
    assert candidate.symbol in decision.message
    assert manager.pending_symbol == "SOXL"


def test_no_new_entries_at_1530_et_boundary(config):
    """15:30 ET is 20:30 UTC in November (EST = UTC-5)."""
    assert et_at(15, 30) == datetime(2026, 11, 6, 20, 30, tzinfo=UTC)

    before = RiskManager(config, clock=MutableClock(et_at(15, 29, 59)))
    _, _, allowed = _evaluate(before, config, et_at(15, 29, 59))
    assert allowed.allowed is True

    at_boundary = RiskManager(config, clock=MutableClock(et_at(15, 30)))
    _, _, blocked = _evaluate(at_boundary, config, et_at(15, 30))
    assert blocked.allowed is False
    assert blocked.reason is RiskBlockReason.NO_NEW_ENTRIES_TIME

    after_close = RiskManager(config, clock=MutableClock(et_at(16, 5)))
    _, _, blocked_after = _evaluate(after_close, config, et_at(16, 5))
    assert blocked_after.allowed is False
    assert blocked_after.reason is RiskBlockReason.NO_NEW_ENTRIES_TIME


def test_premarket_rejected(config):
    premarket = et_at(8, 0)
    manager = RiskManager(config, clock=MutableClock(premarket))
    _, _, decision = _evaluate(manager, config, premarket)

    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.NO_NEW_ENTRIES_TIME


def test_weekend_rejected(config):
    saturday = et_at(10, 0, day=date(2026, 11, 7))
    manager = RiskManager(config, clock=MutableClock(saturday))
    _, _, decision = _evaluate(manager, config, saturday)

    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.NOT_TRADING_DAY


def test_data_not_ready_rejected(manager, config):
    now = et_at(10, 0)
    _, _, decision = _evaluate(manager, config, now, data_ready=False)

    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.DATA_NOT_READY


# --------------------------------------------------------------------------- #
# Entry count / circuit breakers
# --------------------------------------------------------------------------- #
def test_fifteenth_entry_allowed_sixteenth_rejected(manager, config):
    for index in range(15):
        now = et_at(9, 35) + timedelta(minutes=index * 2)
        _open_position(manager, config, now, signal_id=f"SIG-{index}")
        manager.confirm_exit(
            fill_price=20.0, fill_utc=now + timedelta(minutes=1), reason="MANUAL"
        )

    assert manager.entries_today == 15
    assert manager.snapshot().entries_today == 15

    blocked_at = et_at(9, 35) + timedelta(minutes=15 * 2)
    _, _, decision = _evaluate(manager, config, blocked_at)
    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.MAX_ENTRIES_REACHED


def test_daily_minus_60_circuit_breaker(manager, config):
    _lose_one_trade(manager, config, et_at(10, 0), signal_id="SIG-L1")  # -20
    _lose_one_trade(manager, config, et_at(10, 5), signal_id="SIG-L2")  # -20
    assert manager.daily_realized_pnl == pytest.approx(-40.0)
    assert manager.daily_stopped is False

    _lose_one_trade(manager, config, et_at(10, 30), signal_id="SIG-L3")  # -60
    assert manager.daily_realized_pnl == pytest.approx(-60.0)
    assert manager.daily_stopped is True

    _, _, decision = _evaluate(manager, config, et_at(10, 40))
    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.DAILY_LOSS_LIMIT


def test_consecutive_stop_cooldown_boundary(manager, config):
    _lose_one_trade(manager, config, et_at(10, 0), signal_id="SIG-S1")  # exit 10:02
    _lose_one_trade(manager, config, et_at(10, 5), signal_id="SIG-S2")  # exit 10:07

    snapshot = manager.snapshot()
    assert snapshot.consecutive_losses == 2
    assert snapshot.cooldown_until_utc == et_at(10, 27)  # 10:07 + 20 min

    _, _, blocked = _evaluate(manager, config, et_at(10, 26, 59))
    assert blocked.allowed is False
    assert blocked.reason is RiskBlockReason.CONSECUTIVE_LOSS_COOLDOWN

    _, _, allowed = _evaluate(manager, config, et_at(10, 27, 0))
    assert allowed.allowed is True


def test_min_reward_risk_rejected(manager, config):
    now = et_at(10, 0)
    candidate, plan, decision = _evaluate(manager, config, now, target_r=1.0)

    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.REWARD_RISK_TOO_LOW
    # A low target still produces a valid plan; the risk gate is what rejects it.
    assert plan.valid is True


def test_zero_share_plan_rejected(manager, config):
    now = et_at(10, 0)
    _, plan, decision = _evaluate(manager, config, now, entry=100.0, stop=50.0)

    assert plan.valid is False
    assert decision.allowed is False
    assert decision.reason is RiskBlockReason.ZERO_SHARES


# --------------------------------------------------------------------------- #
# Mutual exclusion / mode handling
# --------------------------------------------------------------------------- #
def test_soxl_soxs_mutual_exclusion(manager, config):
    _open_position(manager, config, et_at(10, 0), signal_id="SIG-SOXL-1")

    _, _, cross = _evaluate(manager, config, et_at(10, 1), symbol="SOXS")
    assert cross.allowed is False
    assert cross.reason is RiskBlockReason.DUAL_CONFLICT

    _, _, same = _evaluate(manager, config, et_at(10, 1), symbol="SOXL")
    assert same.allowed is False
    assert same.reason is RiskBlockReason.POSITION_ALREADY_OPEN


def test_pending_signal_blocks_and_cancel_clears_it(manager, config):
    now = et_at(10, 0)
    _, _, first = _evaluate(manager, config, now)
    assert first.allowed is True
    assert manager.pending_symbol == "SOXL"

    _, _, second = _evaluate(manager, config, now + timedelta(minutes=1))
    assert second.allowed is False
    assert second.reason is RiskBlockReason.POSITION_ALREADY_OPEN

    _, _, cross = _evaluate(manager, config, now + timedelta(minutes=1), symbol="SOXS")
    assert cross.allowed is False
    assert cross.reason is RiskBlockReason.DUAL_CONFLICT

    assert manager.record_cancel() is True
    assert manager.pending_symbol is None
    _, _, again = _evaluate(manager, config, now + timedelta(minutes=2))
    assert again.allowed is True


def test_mode_filter_only_allows_configured_symbol(config):
    long_only = replace(config, mode=Mode.LONG_SOXL)
    manager = RiskManager(long_only, clock=MutableClock(et_at(10, 0)))

    _, _, rejected = _evaluate(manager, long_only, et_at(10, 0), symbol="SOXS")
    assert rejected.allowed is False
    assert rejected.reason is RiskBlockReason.DUAL_CONFLICT

    _, _, accepted = _evaluate(manager, long_only, et_at(10, 0), symbol="SOXL")
    assert accepted.allowed is True


def test_switch_cooldown_five_minutes(manager, config):
    switch_at = et_at(10, 0)
    assert manager.set_mode(Mode.LONG_SOXL, switch_at) is True
    assert manager.mode is Mode.LONG_SOXL

    _, _, immediately = _evaluate(manager, config, switch_at)
    assert immediately.allowed is False
    assert immediately.reason is RiskBlockReason.SWITCH_COOLDOWN

    _, _, almost = _evaluate(manager, config, et_at(10, 4, 59))
    assert almost.allowed is False
    assert almost.reason is RiskBlockReason.SWITCH_COOLDOWN

    _, _, at_boundary = _evaluate(manager, config, et_at(10, 5, 0))
    assert at_boundary.allowed is True


def test_set_mode_blocked_while_position_open(manager, config):
    _open_position(manager, config, et_at(10, 0), signal_id="SIG-POS-1")

    assert manager.set_mode(Mode.LONG_SOXS, et_at(10, 1)) is False
    assert manager.mode is Mode.DUAL_WATCH


def test_set_mode_same_mode_is_noop_success(manager, config):
    assert manager.set_mode(Mode.DUAL_WATCH, et_at(10, 0)) is True
    snapshot = manager.snapshot()
    assert snapshot.switch_cooldown_until_utc is None


# --------------------------------------------------------------------------- #
# Manual confirmations
# --------------------------------------------------------------------------- #
def test_confirm_entry_records_position(manager, config):
    now = et_at(10, 0)
    position = _open_position(manager, config, now, signal_id="SIG-E1")

    assert position.symbol == "SOXL"
    assert position.shares == 40.0
    assert position.entry_price == pytest.approx(20.0)
    assert position.entry_price_signal == pytest.approx(20.0)
    assert position.stop_loss == pytest.approx(19.5)
    assert position.take_profit == pytest.approx(21.0)

    snapshot = manager.snapshot()
    assert snapshot.entries_today == 1
    assert snapshot.position is not None
    assert snapshot.position.signal_id == "SIG-E1"
    assert snapshot.pending_symbol is None
    assert snapshot.mode is Mode.DUAL_WATCH
    assert snapshot.eod_check_sent is False
    assert snapshot.eod_urgent_sent is False


def test_confirm_entry_twice_raises(manager, config):
    now = et_at(10, 0)
    _open_position(manager, config, now, signal_id="SIG-E2")

    candidate = _candidate(now)
    plan = _plan(config, candidate)
    with pytest.raises(RuntimeError):
        manager.confirm_entry(
            plan,
            fill_price=20.0,
            fill_utc=now,
            signal_id="SIG-E2B",
            mode=Mode.DUAL_WATCH,
        )


def test_confirm_exit_without_position_raises(manager):
    with pytest.raises(RuntimeError):
        manager.confirm_exit(
            fill_price=20.0, fill_utc=et_at(10, 0), reason="MANUAL"
        )


def test_confirm_exit_win_and_stop_math(manager, config):
    win_entry = et_at(10, 0)
    _open_position(manager, config, win_entry, signal_id="SIG-WIN")
    win = manager.confirm_exit(
        fill_price=21.0,
        fill_utc=win_entry + timedelta(minutes=3),
        reason="TARGET",
    )
    assert win.status is TradeStatus.CLOSED
    assert win.pnl == pytest.approx(40.0)  # 40 shares * $1.00
    assert win.r_multiple == pytest.approx(2.0)
    assert manager.daily_realized_pnl == pytest.approx(40.0)
    assert manager.snapshot().consecutive_losses == 0

    loss_entry = et_at(10, 10)
    _open_position(manager, config, loss_entry, signal_id="SIG-LOSS")
    loss = manager.record_stop(19.5, loss_entry + timedelta(minutes=2))
    assert loss.status is TradeStatus.STOPPED
    assert loss.pnl == pytest.approx(-20.0)
    assert loss.r_multiple == pytest.approx(-1.0)
    assert manager.daily_realized_pnl == pytest.approx(20.0)
    assert manager.snapshot().consecutive_losses == 1


def test_winning_trade_resets_consecutive_losses(manager, config):
    _lose_one_trade(manager, config, et_at(10, 0), signal_id="SIG-RL-1")
    assert manager.snapshot().consecutive_losses == 1

    win_entry = et_at(10, 5)
    _open_position(manager, config, win_entry, signal_id="SIG-RL-2")
    manager.confirm_exit(
        fill_price=21.0, fill_utc=win_entry + timedelta(minutes=2), reason="TARGET"
    )
    assert manager.snapshot().consecutive_losses == 0


def test_confirm_flat_clears_position_and_pending(manager, config):
    now = et_at(10, 0)
    _open_position(manager, config, now, signal_id="SIG-FLAT")
    assert manager.confirm_flat() is True
    assert manager.position is None
    assert manager.snapshot().position is None
    assert manager.confirm_flat() is False


# --------------------------------------------------------------------------- #
# Daily stats
# --------------------------------------------------------------------------- #
def test_daily_stats(manager, config):
    win_start = et_at(10, 0)
    _open_position(manager, config, win_start, signal_id="SIG-D1")
    manager.confirm_exit(
        fill_price=21.0, fill_utc=win_start + timedelta(minutes=2), reason="TARGET"
    )
    loss_start = et_at(10, 10)
    _open_position(manager, config, loss_start, signal_id="SIG-D2")
    manager.record_stop(19.5, loss_start + timedelta(minutes=2))

    stats = manager.daily_stats()
    assert stats.et_date == TRADING_DAY
    assert stats.trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.stops == 1
    assert stats.pnl == pytest.approx(20.0)
    assert stats.entries_today == 2
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.avg_r == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# EOD reminders
# --------------------------------------------------------------------------- #
def test_eod_alerts_only_once_per_day(manager, config):
    _open_position(manager, config, et_at(10, 0), signal_id="SIG-EOD")

    assert manager.eod_alerts(et_at(15, 49)) == []

    check_events = manager.eod_alerts(et_at(15, 50))
    assert len(check_events) == 1
    check = check_events[0]
    assert check.kind is SignalEventKind.EOD_REMINDER
    assert check.level is AlertLevel.WARNING
    assert "人工平仓" in check.message
    assert "不会自动清仓" in check.message
    assert "15:50" in check.message
    assert check.symbol == "SOXL"
    assert check.details["stage"] == "EOD_CHECK"

    assert manager.eod_alerts(et_at(15, 51)) == []
    assert manager.eod_alerts(et_at(15, 54)) == []

    urgent_events = manager.eod_alerts(et_at(15, 55))
    assert len(urgent_events) == 1
    urgent = urgent_events[0]
    assert urgent.level is AlertLevel.CRITICAL
    assert "人工平仓" in urgent.message
    assert "不会自动清仓" in urgent.message
    assert "15:55" in urgent.message
    assert urgent.details["stage"] == "EOD_URGENT"

    assert manager.eod_alerts(et_at(15, 56)) == []
    snapshot = manager.snapshot()
    assert snapshot.eod_check_sent is True
    assert snapshot.eod_urgent_sent is True


def test_eod_alerts_not_sent_when_flat(manager):
    assert manager.eod_alerts(et_at(15, 50)) == []
    assert manager.eod_alerts(et_at(15, 55)) == []


def test_late_start_emits_only_urgent(manager, config):
    _open_position(manager, config, et_at(10, 0), signal_id="SIG-LATE")
    events = manager.eod_alerts(et_at(15, 56))

    assert len(events) == 1
    assert events[0].level is AlertLevel.CRITICAL


# --------------------------------------------------------------------------- #
# Session rollover
# --------------------------------------------------------------------------- #
def test_roll_resets_daily_state_and_flags(manager, config, clock):
    _lose_one_trade(manager, config, et_at(10, 0), signal_id="SIG-ROLL-1")
    manager.confirm_flat()
    # Send the EOD reminder so its once-per-day flags are set.
    win_entry = et_at(10, 5)
    _open_position(manager, config, win_entry, signal_id="SIG-ROLL-2")
    manager.eod_alerts(et_at(15, 50))
    manager.confirm_exit(
        fill_price=21.0, fill_utc=et_at(15, 52), reason="TARGET"
    )
    before = manager.snapshot()
    assert before.eod_check_sent is True
    assert before.entries_today == 2

    next_session = et_at(9, 30, day=date(2026, 11, 9))  # Monday
    clock.set(next_session)
    rolled = manager.roll(next_session)
    assert rolled == date(2026, 11, 9)

    after = manager.snapshot()
    assert after.et_date == date(2026, 11, 9)
    assert after.entries_today == 0
    assert after.consecutive_losses == 0
    assert after.daily_realized_pnl == 0.0
    assert after.daily_stopped is False
    assert after.cooldown_until_utc is None
    assert after.switch_cooldown_until_utc is None
    assert after.eod_check_sent is False
    assert after.eod_urgent_sent is False
    assert after.position is None


def test_roll_backwards_is_ignored(manager, config):
    now = et_at(10, 0)
    _open_position(manager, config, now, signal_id="SIG-BACK")
    manager.confirm_exit(
        fill_price=20.0, fill_utc=et_at(10, 1), reason="MANUAL"
    )
    entries_before = manager.entries_today

    returned = manager.roll(
        et_at(10, 0, day=date(2026, 11, 5))  # previous day
    )
    assert returned == TRADING_DAY
    assert manager.entries_today == entries_before


# --------------------------------------------------------------------------- #
# Properties / helper sanity
# --------------------------------------------------------------------------- #
def test_snapshot_after_flat_has_no_pending(manager, config):
    now = et_at(10, 0)
    _evaluate(manager, config, now)
    snapshot = manager.snapshot()
    assert snapshot.pending_symbol == "SOXL"

    manager.record_cancel()
    snapshot = manager.snapshot()
    assert snapshot.pending_symbol is None
    assert snapshot.position is None


def test_bool_of_decision(manager, config):
    now = et_at(10, 0)
    _, _, allowed = _evaluate(manager, config, now)
    assert bool(allowed) is True

    _, _, blocked = _evaluate(manager, config, now, data_ready=False)
    assert bool(blocked) is False


# --------------------------------------------------------------------------- #
# begin_session(): replay-clock sync for the backtest engine
# --------------------------------------------------------------------------- #
def test_begin_session_syncs_backward_clock_and_resets_daily_state(config):
    """A manager built with the real (future) clock must sync onto replay dates.

    ``roll`` deliberately refuses to move backwards; if the backtest engine only
    called ``roll`` the daily state would latch forever after the first loss and
    every later candidate would be blocked with DAILY_LOSS_LIMIT.
    """
    clock = MutableClock(et_at(10, 0))  # manager constructed "today" (2026-11-06)
    manager = RiskManager(config, clock=clock)
    replay_day = date(2024, 1, 3)
    replay_moment = datetime(2024, 1, 3, 15, 0, tzinfo=UTC)
    manager.begin_session(replay_moment)
    clock.set(replay_moment)  # snapshot()/daily_stats() roll with the driver clock
    assert manager.snapshot().et_date == replay_day

    for index, entry_utc in enumerate(
        (
            datetime(2024, 1, 3, 15, 2, tzinfo=UTC),  # 10:02 ET
            datetime(2024, 1, 3, 15, 7, tzinfo=UTC),  # 10:07 ET
            datetime(2024, 1, 3, 15, 30, tzinfo=UTC),  # 10:30 ET
        ),
        start=1,
    ):
        _lose_one_trade(manager, config, entry_utc, signal_id=f"REPLAY-{index}")
    assert manager.daily_stopped is True
    assert manager.daily_realized_pnl == pytest.approx(-60.0)

    next_moment = datetime(2024, 1, 4, 15, 0, tzinfo=UTC)
    manager.begin_session(next_moment)
    clock.set(next_moment)
    after = manager.snapshot()
    assert after.et_date == date(2024, 1, 4)
    assert after.daily_stopped is False
    assert after.daily_realized_pnl == 0.0
    assert after.entries_today == 0
    assert after.cooldown_until_utc is None


def test_begin_session_moves_backwards_while_roll_is_ignored(config):
    clock = MutableClock(et_at(10, 0))
    manager = RiskManager(config, clock=clock)
    manager.begin_session(et_at(10, 0))
    _open_position(manager, config, et_at(10, 1), signal_id="SIG-SYNC")
    manager.confirm_exit(fill_price=20.0, fill_utc=et_at(10, 2), reason="MANUAL")
    assert manager.entries_today == 1

    # roll() backwards stays ignored (protects live state from stale commands).
    previous = et_at(10, 0, day=date(2026, 11, 5))
    manager.roll(previous)
    assert manager.entries_today == 1
    assert manager.snapshot().et_date == TRADING_DAY

    # begin_session() is the explicit driver-level sync and may move backwards.
    manager.begin_session(previous)
    clock.set(previous)
    snapshot = manager.snapshot()
    assert snapshot.et_date == date(2026, 11, 5)
    assert snapshot.entries_today == 0
