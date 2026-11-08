"""Offline, deterministic tests for the signal engine.

Every dependency of the engine is a local fake here (strategies, risk manager,
journal, data-quality monitor, session-VWAP callback, clock), so the suite runs
without builder B/C's packages, without network and without the real clock.
The fakes mirror the documented interfaces of those modules exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

from intraday_scalper.core.config import load_config
from intraday_scalper.core.models import (
    AlertEvent,
    AlertLevel,
    Bar,
    ConnectionState,
    DataStatus,
    Mode,
    Position,
    RiskBlockReason,
    SessionPhase,
    SignalCandidate,
    SignalEventKind,
    StrategyName,
)
from intraday_scalper.signals import SignalEngine

UTC = timezone.utc

# Friday 2026-11-06, after US DST ended: 09:30 ET == 14:30 UTC.
TRADING_DATE = datetime(2026, 11, 6, tzinfo=UTC).date()
OPEN_UTC = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
CLOSE_UTC = datetime(2026, 11, 6, 21, 0, tzinfo=UTC)


def et(hour: int, minute: int) -> datetime:
    """Instant for a US/ET wall-clock time on the fixed trading date."""
    return datetime(2026, 11, 6, hour, minute, tzinfo=UTC) + timedelta(hours=5)


def minute_start(index: int) -> datetime:
    return OPEN_UTC + timedelta(minutes=index)


def make_bar(
    symbol: str = "SOXL",
    *,
    minute_index: int = 0,
    start: Optional[datetime] = None,
    open_: Optional[float] = None,
    close: float = 40.1,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: float = 10_000.0,
    vwap: Optional[float] = None,
    is_complete: bool = True,
    timeframe_minutes: int = 1,
    trade_count: int = 25,
) -> Bar:
    start_utc = start if start is not None else minute_start(minute_index)
    open_value = close if open_ is None else open_
    high_value = max(open_value, close) * 1.002 if high is None else high
    low_value = min(open_value, close) * 0.998 if low is None else low
    vwap_value = (open_value + high_value + low_value + close) / 4.0 if vwap is None else vwap
    return Bar(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        start_utc=start_utc,
        end_utc=start_utc + timedelta(minutes=timeframe_minutes),
        open=open_value,
        high=high_value,
        low=low_value,
        close=close,
        volume=volume,
        vwap=vwap_value,
        trade_count=trade_count,
        is_complete=is_complete,
    )


# --------------------------------------------------------------------------- #
# Fakes mirroring the frozen interfaces of the other builders
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


@dataclass
class FakeDecision:
    allowed: bool
    reason: RiskBlockReason = RiskBlockReason.NONE
    plan: Any = None
    message: str = ""


@dataclass
class FakeRiskState:
    mode: Mode = Mode.DUAL_WATCH
    position: Optional[Position] = None
    entries_today: int = 0
    daily_realized_pnl: float = 0.0
    daily_stopped: bool = False
    cooldown_until_utc: Optional[datetime] = None
    switch_cooldown_until_utc: Optional[datetime] = None


@dataclass
class FakeDailyStats:
    trades: int = 0
    wins: int = 0
    stops: int = 0
    pnl: float = 0.0
    entries_today: int = 0


class FakeRiskManager:
    """Minimal stand-in for builder C's ``RiskManager`` (never fills orders)."""

    def __init__(
        self,
        *,
        mode: Mode = Mode.DUAL_WATCH,
        allow: bool = True,
        block_reason: RiskBlockReason = RiskBlockReason.NONE,
        position: Optional[Position] = None,
        lock_after_allow: bool = False,
    ) -> None:
        self.state = FakeRiskState(mode=mode, position=position)
        self.allow = allow
        self.block_reason = block_reason
        self.lock_after_allow = lock_after_allow
        self.pending = False
        self.entry_calls: List[tuple] = []
        self.eod_calls: List[datetime] = []
        self.eod_queue: List[AlertEvent] = []
        self.stats = FakeDailyStats()

    def evaluate_entry(self, candidate, plan, now, *, data_ready=True):
        self.entry_calls.append((candidate, plan, now, data_ready))
        if not data_ready:
            return FakeDecision(False, RiskBlockReason.DATA_NOT_READY, None, "数据未就绪")
        if self.state.position is not None or self.pending:
            return FakeDecision(
                False, RiskBlockReason.POSITION_ALREADY_OPEN, None, "已有持仓或待确认信号"
            )
        if not self.allow:
            return FakeDecision(False, self.block_reason, None, "测试风控拦截")
        if self.lock_after_allow:
            self.pending = True
        return FakeDecision(True, RiskBlockReason.NONE, plan, "允许开仓（人工确认后执行）")

    def snapshot(self) -> FakeRiskState:
        return self.state

    def daily_stats(self) -> FakeDailyStats:
        return self.stats

    def eod_alerts(self, now: datetime) -> List[AlertEvent]:
        self.eod_calls.append(now)
        due = [event for event in self.eod_queue if event.ts_utc <= now]
        self.eod_queue = [event for event in self.eod_queue if event.ts_utc > now]
        return due


class FakeJournal:
    def __init__(self) -> None:
        self.signals: List[tuple] = []
        self.events: List[AlertEvent] = []

    def record_signal(self, candidate, plan) -> str:
        self.signals.append((candidate, plan))
        return f"SIG-TEST-{len(self.signals):03d}"

    def record_event(self, event) -> None:
        self.events.append(event)


class FakeDataQuality:
    def __init__(self, status: DataStatus = DataStatus.OK) -> None:
        self.status = status
        self.report_calls = 0

    def report(self, now: datetime):
        self.report_calls += 1
        return _Report(self.status)


@dataclass
class _Report:
    aggregate: DataStatus


class FakeStrategy:
    """Fires a valid long candidate when ``fire(ctx)`` is true."""

    def __init__(
        self,
        name: StrategyName = StrategyName.ORB_5M,
        fire=None,
        target_r: float = 2.0,
    ) -> None:
        self.name = name
        self._fire = fire
        self.target_r = target_r
        self.calls: List[Any] = []

    def evaluate(self, ctx) -> Optional[SignalCandidate]:
        self.calls.append(ctx)
        if self._fire is None or not self._fire(ctx):
            return None
        last = ctx.bars_1m[-1]
        entry = float(last.close)
        return SignalCandidate(
            symbol=ctx.symbol,
            strategy=self.name,
            entry_ref=entry,
            stop_loss=entry * 0.99,
            target_r=self.target_r,
            rationale="fake strategy for offline engine tests",
            created_utc=ctx.now_utc,
            confirmation_bar_start_utc=last.start_utc,
        )


def fire_at_minute(index: int, symbols: Optional[set] = None):
    target = minute_start(index)

    def _fire(ctx) -> bool:
        if symbols is not None and ctx.symbol not in symbols:
            return False
        return bool(ctx.bars_1m) and ctx.bars_1m[-1].start_utc == target

    return _fire


def never_fire(ctx) -> bool:  # pragma: no cover - predicate helper
    return False


def build_engine(
    *,
    strategies=None,
    risk=None,
    journal=None,
    data_quality=None,
    clock=None,
    config=None,
    on_event=None,
    vwap=None,
) -> SignalEngine:
    return SignalEngine(
        config or load_config(),
        strategies if strategies is not None else [FakeStrategy(fire=never_fire)],
        risk or FakeRiskManager(),
        journal=journal,
        data_quality=data_quality,
        on_event=on_event,
        clock=clock or FakeClock(et(10, 0)),
        current_session_vwap=vwap,
    )


def events_of(engine: SignalEngine, kind: SignalEventKind) -> List[AlertEvent]:
    return [event for event in engine.drain_events() if event.kind is kind]


# --------------------------------------------------------------------------- #
# Basic gating
# --------------------------------------------------------------------------- #
def test_incomplete_bar_is_ignored() -> None:
    strategy = FakeStrategy(fire=fire_at_minute(0))
    engine = build_engine(strategies=[strategy])
    engine.on_bar(make_bar(minute_index=0, is_complete=False))
    assert strategy.calls == []
    assert engine.drain_events() == []


def test_non_one_minute_bar_is_ignored() -> None:
    strategy = FakeStrategy(fire=lambda ctx: True)
    engine = build_engine(strategies=[strategy])
    engine.on_bar(make_bar(minute_index=0, timeframe_minutes=5))
    assert strategy.calls == []
    assert engine.drain_events() == []


def test_premarket_bar_is_ignored() -> None:
    strategy = FakeStrategy(fire=lambda ctx: True)
    engine = build_engine(strategies=[strategy])
    engine.on_bar(make_bar(start=et(8, 0), close=40.0))
    assert strategy.calls == []
    assert engine.drain_events() == []


def test_unknown_symbol_is_ignored() -> None:
    strategy = FakeStrategy(fire=lambda ctx: True)
    engine = build_engine(strategies=[strategy])
    engine.on_bar(make_bar(symbol="SPY", minute_index=0))
    assert strategy.calls == []
    assert engine.drain_events() == []


# --------------------------------------------------------------------------- #
# Signal emission
# --------------------------------------------------------------------------- #
def test_completed_bar_emits_exactly_one_entry_signal() -> None:
    strategy = FakeStrategy(fire=fire_at_minute(15))
    journal = FakeJournal()
    engine = build_engine(strategies=[strategy], journal=journal)
    for index in range(21):
        engine.on_bar(make_bar(minute_index=index, close=40.0 + 0.01 * index))

    entries = events_of(engine, SignalEventKind.ENTRY_SIGNAL)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.symbol == "SOXL"
    assert entry.level is AlertLevel.WARNING
    assert entry.signal_id and entry.signal_id.startswith("SIG-TEST-")
    assert "人工" in entry.message
    assert "不会自动下单" in entry.message
    for token in ("止损", "止盈", "股数", "名义", "风险", "R:R"):
        assert token in entry.message
    assert entry.details["candidate"]["symbol"] == "SOXL"
    assert entry.details["plan"]["shares"] > 0
    assert len(journal.signals) == 1
    candidate, plan = journal.signals[0]
    assert candidate.symbol == "SOXL"
    assert plan.shares > 0
    assert math.isclose(plan.actual_risk, plan.theoretical_risk + plan.est_slippage_dollars + plan.est_fees_dollars)


def test_same_confirmation_bar_never_emits_twice() -> None:
    strategy = FakeStrategy(fire=fire_at_minute(15))
    journal = FakeJournal()
    engine = build_engine(strategies=[strategy], journal=journal)
    for index in range(16):
        engine.on_bar(make_bar(minute_index=index, close=40.0 + 0.01 * index))
    # Duplicate delivery of the very same completed bar.
    engine.on_bar(make_bar(minute_index=15, close=40.16))
    entries = events_of(engine, SignalEventKind.ENTRY_SIGNAL)
    assert len(entries) == 1
    assert len(journal.signals) == 1


def test_risk_block_emits_event_and_no_signal() -> None:
    strategy = FakeStrategy(fire=lambda ctx: True)
    journal = FakeJournal()
    risk = FakeRiskManager(
        allow=False, block_reason=RiskBlockReason.DAILY_LOSS_LIMIT
    )
    engine = build_engine(strategies=[strategy], risk=risk, journal=journal)
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    events = engine.drain_events()
    assert not [e for e in events if e.kind is SignalEventKind.ENTRY_SIGNAL]
    blocks = [e for e in events if e.kind is SignalEventKind.RISK_BLOCK]
    assert len(blocks) == 1
    assert "DAILY_LOSS_LIMIT" in blocks[0].message
    assert journal.signals == []
    assert len(risk.entry_calls) == 1


# --------------------------------------------------------------------------- #
# Data-status gating
# --------------------------------------------------------------------------- #
def test_data_unavailable_blocks_signals() -> None:
    strategy = FakeStrategy(fire=lambda ctx: True)
    journal = FakeJournal()
    engine = build_engine(strategies=[strategy], journal=journal)
    engine.set_data_status(DataStatus.STALE, "no ticks for 90s")
    boot_events = engine.drain_events()
    assert [e.kind for e in boot_events] == [SignalEventKind.DATA_UNAVAILABLE]
    assert engine.data_status is DataStatus.STALE
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    assert strategy.calls == []
    assert journal.signals == []
    assert engine.drain_events() == []


def test_data_alert_is_rate_limited_to_once_per_second() -> None:
    clock = FakeClock(et(10, 0))
    engine = build_engine(clock=clock)
    engine.set_data_status(DataStatus.STALE, "first")
    engine.set_data_status(DataStatus.UNAVAILABLE, "same second")
    events = events_of(engine, SignalEventKind.DATA_UNAVAILABLE)
    assert len(events) == 1
    clock.advance(2.0)
    engine.set_data_status(DataStatus.STALE, "two seconds later")
    assert len(events_of(engine, SignalEventKind.DATA_UNAVAILABLE)) == 1
    assert engine.data_status is DataStatus.STALE


def test_data_recovery_emits_info_and_signals_resume() -> None:
    clock = FakeClock(et(10, 0))
    strategy = FakeStrategy(fire=lambda ctx: True)
    journal = FakeJournal()
    engine = build_engine(strategies=[strategy], journal=journal, clock=clock)
    engine.set_data_status(DataStatus.UNAVAILABLE, "ws down")
    clock.advance(2.0)
    engine.set_data_status(DataStatus.OK, "ws back")
    recovered = [
        event
        for event in engine.drain_events()
        if event.kind is SignalEventKind.INFO and "恢复" in event.message
    ]
    assert len(recovered) == 1
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    assert len(events_of(engine, SignalEventKind.ENTRY_SIGNAL)) == 1


def test_quality_monitor_drives_status() -> None:
    clock = FakeClock(et(10, 0))
    data_quality = FakeDataQuality(DataStatus.STALE)
    strategy = FakeStrategy(fire=lambda ctx: True)
    journal = FakeJournal()
    engine = build_engine(
        strategies=[strategy], journal=journal, clock=clock, data_quality=data_quality
    )
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    assert data_quality.report_calls >= 1
    assert engine.data_status is DataStatus.STALE
    events = engine.drain_events()
    assert [e.kind for e in events] == [SignalEventKind.DATA_UNAVAILABLE]
    assert strategy.calls == []

    data_quality.status = DataStatus.OK
    clock.advance(2.0)
    engine.on_bar(make_bar(minute_index=1, close=40.01))
    assert engine.data_status is DataStatus.OK
    assert len(events_of(engine, SignalEventKind.ENTRY_SIGNAL)) == 1


# --------------------------------------------------------------------------- #
# Dual-track mode / mutual exclusion
# --------------------------------------------------------------------------- #
def test_dual_watch_never_emits_two_entries_for_same_bar() -> None:
    strategy = FakeStrategy(fire=fire_at_minute(15))
    journal = FakeJournal()
    risk = FakeRiskManager(lock_after_allow=True)
    engine = build_engine(strategies=[strategy], risk=risk, journal=journal)
    engine.on_bar(make_bar(symbol="SOXL", minute_index=15, close=40.2))
    engine.on_bar(make_bar(symbol="SOXS", minute_index=15, close=10.2))

    events = engine.drain_events()
    entries = [e for e in events if e.kind is SignalEventKind.ENTRY_SIGNAL]
    blocks = [e for e in events if e.kind is SignalEventKind.RISK_BLOCK]
    assert len(entries) == 1
    assert entries[0].symbol == "SOXL"
    assert len(blocks) == 1
    assert "POSITION_ALREADY_OPEN" in blocks[0].message
    assert len(journal.signals) == 1
    assert {ctx.symbol for ctx in strategy.calls} == {"SOXL", "SOXS"}

    snapshot = engine.state_snapshot()
    assert snapshot.mode is Mode.DUAL_WATCH


def test_long_soxs_mode_filters_soxl_candidates() -> None:
    config = load_config().with_mode(Mode.LONG_SOXS)
    strategy = FakeStrategy(fire=fire_at_minute(15))
    journal = FakeJournal()
    risk = FakeRiskManager(mode=Mode.LONG_SOXS)
    engine = build_engine(
        strategies=[strategy], risk=risk, journal=journal, config=config
    )
    engine.on_bar(make_bar(symbol="SOXL", minute_index=15, close=40.2))
    assert strategy.calls == []
    assert engine.drain_events() == []

    engine.on_bar(make_bar(symbol="SOXS", minute_index=15, close=10.0))
    entries = events_of(engine, SignalEventKind.ENTRY_SIGNAL)
    assert len(entries) == 1
    assert entries[0].symbol == "SOXS"


def test_long_soxl_mode_filters_soxs_candidates() -> None:
    config = load_config().with_mode(Mode.LONG_SOXL)
    strategy = FakeStrategy(fire=fire_at_minute(15))
    risk = FakeRiskManager(mode=Mode.LONG_SOXL)
    engine = build_engine(strategies=[strategy], risk=risk, config=config)
    engine.on_bar(make_bar(symbol="SOXS", minute_index=15, close=10.0))
    assert strategy.calls == []
    engine.on_bar(make_bar(symbol="SOXL", minute_index=15, close=40.2))
    assert len(events_of(engine, SignalEventKind.ENTRY_SIGNAL)) == 1


# --------------------------------------------------------------------------- #
# Position-management hints (reminders only — nothing is closed)
# --------------------------------------------------------------------------- #
def build_position() -> Position:
    return Position(
        signal_id="SIG-POS-001",
        symbol="SOXL",
        strategy=StrategyName.ORB_5M,
        mode=Mode.DUAL_WATCH,
        shares=10.0,
        entry_price=40.0,
        entry_utc=et(10, 0),
        stop_loss=39.6,
        take_profit=40.8,
        entry_price_signal=39.98,
    )


def test_position_hints_fire_once_per_level() -> None:
    clock = FakeClock(et(10, 0))
    strategy = FakeStrategy(fire=never_fire)
    risk = FakeRiskManager(position=build_position())
    engine = build_engine(strategies=[strategy], risk=risk, clock=clock)

    closes = [40.0, 40.5, 40.81, 39.5, 39.4, 39.4]
    previous = closes[0]
    for index, close in enumerate(closes):
        engine.on_bar(make_bar(minute_index=index, open_=previous, close=close))
        previous = close
        if index == 2:
            snapshot = engine.state_snapshot()
            assert snapshot.position is not None
            assert snapshot.position.symbol == "SOXL"
            assert snapshot.position.unrealized_r is not None
            assert math.isclose(snapshot.position.unrealized_r, 2.025, rel_tol=1e-6)
            payload = snapshot.to_dict()
            json.dumps(payload)  # must be JSON serialisable for logging/tests
            assert payload["position"]["unrealized_r"] > 2.0

    events = engine.drain_events()
    breakeven = [e for e in events if e.kind is SignalEventKind.BREAKEVEN_HINT]
    take_profit = [e for e in events if e.kind is SignalEventKind.TAKE_PROFIT_HINT]
    stop_hit = [e for e in events if e.kind is SignalEventKind.STOP_HIT_HINT]
    assert len(breakeven) == 1
    assert "1.2R" in breakeven[0].message
    assert len(take_profit) == 1
    assert "2.0R" in take_profit[0].message
    assert len(stop_hit) == 1
    assert stop_hit[0].level is AlertLevel.CRITICAL
    assert "人工" in stop_hit[0].message
    # The engine never mutates the position or places an order.
    assert risk.state.position is not None
    assert len(risk.entry_calls) == 0


def test_no_hints_when_data_not_ok() -> None:
    risk = FakeRiskManager(position=build_position())
    engine = build_engine(risk=risk)
    engine.set_data_status(DataStatus.UNAVAILABLE, "ws down")
    engine.on_bar(make_bar(minute_index=0, close=39.0))
    hints = [
        e
        for e in engine.drain_events()
        if e.kind
        in {
            SignalEventKind.STOP_HIT_HINT,
            SignalEventKind.BREAKEVEN_HINT,
            SignalEventKind.TAKE_PROFIT_HINT,
        }
    ]
    assert hints == []


# --------------------------------------------------------------------------- #
# EOD reminders
# --------------------------------------------------------------------------- #
def test_eod_reminders_are_emitted_and_mention_manual_close() -> None:
    clock = FakeClock(et(15, 50))
    risk = FakeRiskManager(position=build_position())
    risk.eod_queue = [
        AlertEvent(
            kind=SignalEventKind.EOD_REMINDER,
            level=AlertLevel.WARNING,
            message="EOD 15:50 检查：若仍持仓请准备平仓",
            ts_utc=et(15, 50),
        ),
        AlertEvent(
            kind=SignalEventKind.EOD_REMINDER,
            level=AlertLevel.CRITICAL,
            message="EOD 15:55 紧急提醒",
            ts_utc=et(15, 55),
        ),
    ]
    engine = build_engine(risk=risk, clock=clock)

    engine.on_bar(make_bar(minute_index=19, close=40.0))
    first = events_of(engine, SignalEventKind.EOD_REMINDER)
    assert len(first) == 1
    assert first[0].level is AlertLevel.WARNING
    assert "人工平仓" in first[0].message
    assert "系统不会自动清仓" in first[0].message

    clock.set(et(15, 55))
    engine.on_bar(make_bar(minute_index=24, close=40.0))
    second = events_of(engine, SignalEventKind.EOD_REMINDER)
    assert len(second) == 1
    assert second[0].level is AlertLevel.CRITICAL
    assert "人工平仓" in second[0].message
    assert len(risk.eod_calls) == 2


# --------------------------------------------------------------------------- #
# Snapshots / events / context
# --------------------------------------------------------------------------- #
def test_state_snapshot_is_complete_and_serialisable() -> None:
    clock = FakeClock(et(10, 0))
    strategy = FakeStrategy(fire=never_fire, name=StrategyName.VWAP_RECLAIM)
    engine = build_engine(
        strategies=[strategy],
        clock=clock,
        vwap=lambda symbol, now: 40.25,
    )
    for index in range(30):
        close = 40.0 + 0.01 * index
        engine.on_bar(make_bar(symbol="SOXL", minute_index=index, close=close))
        engine.on_bar(make_bar(symbol="SOXS", minute_index=index, close=10.0 + 0.01 * index))

    snapshot = engine.state_snapshot()
    assert snapshot.as_of_utc == clock.now
    assert snapshot.session_phase is SessionPhase.REGULAR
    assert snapshot.data_status is DataStatus.OK
    assert snapshot.connection_state is ConnectionState.DISCONNECTED
    assert set(snapshot.symbols) == {"SOXL", "SOXS"}
    assert snapshot.strategy_names == ("VWAP_RECLAIM",)
    assert snapshot.max_entries_per_day == 15
    assert snapshot.eod_close_utc == CLOSE_UTC

    for symbol in ("SOXL", "SOXS"):
        data = snapshot.symbols[symbol]
        assert data.last_price is not None
        assert data.bars_1m_count == 30
        assert data.ema9 is not None and data.ema21 is not None
        assert data.rsi is not None
        assert data.atr is not None
        assert data.vwap == pytest.approx(40.25)
        assert data.trend_1m in {"bull", "bear", "flat"}
        assert data.data_age_seconds is not None

    payload = snapshot.to_dict()
    json.dumps(payload)  # JSON serialisable
    assert payload["symbols"]["SOXL"]["bars_1m_count"] == 30
    assert payload["mode"] == "dual_watch"
    assert snapshot.symbol("soxl") is snapshot.symbols["SOXL"]


def test_vwap_is_never_faked_from_bar_vwap() -> None:
    captured = []

    def fire(ctx) -> bool:
        captured.append(ctx)
        return False

    strategy = FakeStrategy(fire=fire)
    engine = build_engine(strategies=[strategy], vwap=None)
    for index in range(3):
        # Valid bar with a (fake) per-bar vwap that must NOT become the session VWAP.
        engine.on_bar(
            make_bar(minute_index=index, close=40.0, high=40.08, low=39.92, vwap=40.03 + 0.01 * index)
        )
    assert captured, "strategy must have been evaluated"
    # No session-VWAP callback injected -> no VWAP (bar.vwap is NOT a session VWAP).
    assert all(ctx.vwap is None for ctx in captured)


def test_context_contains_only_completed_regular_session_bars() -> None:
    contexts = []

    def fire(ctx) -> bool:
        contexts.append(ctx)
        return False

    strategy = FakeStrategy(fire=fire)
    engine = build_engine(strategies=[strategy])
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    engine.on_bar(make_bar(minute_index=1, close=40.0, is_complete=False))  # ignored
    engine.on_bar(make_bar(minute_index=1, close=40.05))

    latest = contexts[-1]
    assert all(bar.is_complete for bar in latest.bars_1m)
    assert [bar.start_utc for bar in latest.bars_1m] == [minute_start(0), minute_start(1)]
    assert latest.last_price == pytest.approx(40.05)
    assert latest.bars_5m == ()
    assert latest.last(1) is latest.bars_1m[-1]
    assert latest.prev() is latest.bars_1m[-2]


def test_on_5m_bar_updates_trend_cache() -> None:
    strategy = FakeStrategy(fire=never_fire)
    engine = build_engine(strategies=[strategy])
    for index in range(2):
        engine.on_bar(make_bar(minute_index=index, close=40.0))
    five = make_bar(
        minute_index=0, close=40.2, timeframe_minutes=5
    )
    engine.on_5m_bar(five)
    snapshot = engine.state_snapshot()
    assert snapshot.symbols["SOXL"].bars_5m_count == 1
    contexts = []
    strategy5 = FakeStrategy(fire=lambda ctx: contexts.append(ctx) or False)
    engine2 = build_engine(strategies=[strategy5])
    engine2.on_5m_bar(five)
    engine2.on_bar(make_bar(minute_index=0, close=40.0))
    assert contexts and contexts[-1].bars_5m == (five,)


def test_event_pipeline_callback_and_drain() -> None:
    received: List[AlertEvent] = []
    strategy = FakeStrategy(fire=fire_at_minute(0))
    engine = build_engine(strategies=[strategy], on_event=received.append)
    engine.on_bar(make_bar(minute_index=0, close=40.0))

    drained = engine.drain_events()
    assert [event.kind for event in drained] == [SignalEventKind.ENTRY_SIGNAL]
    assert [event.kind for event in received] == [SignalEventKind.ENTRY_SIGNAL]
    assert engine.drain_events() == []

    snapshot = engine.state_snapshot()
    assert [event.kind for event in snapshot.events] == [SignalEventKind.ENTRY_SIGNAL]
    assert snapshot.latest_signal is not None
    assert snapshot.latest_signal.symbol == "SOXL"
    assert snapshot.latest_plan is not None
    assert snapshot.latest_signal_id == drained[0].signal_id


def test_connection_state_is_reported_in_snapshot() -> None:
    engine = build_engine()
    engine.set_connection_state(ConnectionState.CONNECTED)
    assert engine.state_snapshot().connection_state is ConnectionState.CONNECTED
    engine.set_connection_state("RECONNECTING")
    assert engine.connection_state is ConnectionState.RECONNECTING


def test_broken_strategy_does_not_crash_engine() -> None:
    class ExplodingStrategy:
        name = StrategyName.ORB_5M

        def evaluate(self, ctx):
            raise RuntimeError("boom")

    journal = FakeJournal()
    engine = build_engine(strategies=[ExplodingStrategy()], journal=journal)
    engine.on_bar(make_bar(minute_index=0, close=40.0))
    events = engine.drain_events()
    assert [e.kind for e in events] == [SignalEventKind.INFO]
    assert "boom" in events[0].message
    assert journal.signals == []


# --------------------------------------------------------------------------- #
# Real integration (skipped automatically if a sibling package is absent)
# --------------------------------------------------------------------------- #
def test_integration_real_strategy_registry_builds_context() -> None:
    """Engine + builder B's real registry/indicators/strategies, fake risk only."""
    pytest.importorskip("intraday_scalper.strategies.registry")
    from intraday_scalper.strategies.registry import build_strategies

    config = load_config()
    clock = FakeClock(et(10, 0))
    engine = build_engine(
        strategies=build_strategies(config.strategies),
        risk=FakeRiskManager(),
        clock=clock,
        config=config,
        vwap=lambda symbol, now: 40.0 if symbol == "SOXL" else 10.0,
    )
    for index in range(31):
        engine.on_bar(make_bar(symbol="SOXL", minute_index=index, close=40.0))
        engine.on_bar(make_bar(symbol="SOXS", minute_index=index, close=10.0))

    snapshot = engine.state_snapshot()
    assert snapshot.strategy_names == (
        "ORB_5M",
        "ORB_15M",
        "VWAP_RECLAIM",
        "EMA_PULLBACK",
    )
    assert snapshot.data_status is DataStatus.OK
    for symbol in ("SOXL", "SOXS"):
        data = snapshot.symbols[symbol]
        assert data.bars_1m_count == 31
        assert data.ema9 is not None
        assert data.ema21 is not None
        assert data.rsi is not None
        assert data.atr is not None
    assert engine.drain_events() == []  # flat market: no signal, no crash


def test_integration_real_risk_and_journal_manual_entry_flow(tmp_path) -> None:
    """Engine + C's real RiskManager/Journal/sizing + B-free fake strategy.

    Exercises the complete manual loop: alert → journal signal row → human
    ENTRY confirmation in risk/journal → position hints → human EXIT record.
    Nothing is ever auto-filled by the engine.
    """
    pytest.importorskip("intraday_scalper.risk.manager")
    pytest.importorskip("intraday_scalper.journal.journal")
    from intraday_scalper.journal.journal import Journal
    from intraday_scalper.risk.manager import RiskManager

    config = load_config()
    clock = FakeClock(et(10, 0))
    journal = Journal(tmp_path / "journal.sqlite3", config=config, clock=clock)
    try:
        risk = RiskManager(config, clock=clock)
        strategy = FakeStrategy(fire=fire_at_minute(5))
        engine = build_engine(
            strategies=[strategy],
            risk=risk,
            journal=journal,
            clock=clock,
            config=config,
        )
        for index in range(6):
            engine.on_bar(make_bar(minute_index=index, open_=40.0, close=40.0))

        entries = events_of(engine, SignalEventKind.ENTRY_SIGNAL)
        assert len(entries) == 1
        entry = entries[0]
        signal_id = entry.signal_id
        assert signal_id
        assert risk.pending_symbol == "SOXL"
        assert journal.open_positions() == []
        assert any(row.get("kind") == "ENTRY_SIGNAL" for row in journal.recent_events(50))

        snapshot = engine.state_snapshot()
        plan = snapshot.latest_plan
        assert plan is not None and plan.shares > 0
        # Real RiskState carries a ``date`` field; the snapshot must stay JSON-safe.
        json.dumps(snapshot.to_dict())

        # A human confirms the fill manually (the engine never does this).
        risk.confirm_entry(
            plan,
            fill_price=40.05,
            fill_utc=et(10, 6),
            signal_id=signal_id,
            mode=Mode.DUAL_WATCH,
        )
        journal.record_entry(signal_id, 40.05, plan.shares, et(10, 6))
        assert len(journal.open_positions()) == 1

        clock.set(et(10, 7))
        engine.on_bar(
            make_bar(minute_index=7, open_=40.05, close=41.0, high=41.05, low=40.0)
        )
        hint_kinds = {
            event.kind
            for event in engine.drain_events()
            if event.kind
            in {
                SignalEventKind.BREAKEVEN_HINT,
                SignalEventKind.TAKE_PROFIT_HINT,
            }
        }
        assert SignalEventKind.BREAKEVEN_HINT in hint_kinds
        assert SignalEventKind.TAKE_PROFIT_HINT in hint_kinds

        trade = journal.record_exit(signal_id, 41.0, et(10, 8), "MANUAL")
        assert trade.pnl > 0
        assert journal.open_positions() == []
    finally:
        journal.close()


def test_integration_real_data_quality_blocks_then_recovers() -> None:
    """Engine + builder A's real DataQualityMonitor: DATA_UNAVAILABLE gating."""
    pytest.importorskip("intraday_scalper.providers.data_quality")
    from intraday_scalper.core.models import Tick
    from intraday_scalper.providers.data_quality import DataQualityMonitor

    config = load_config()
    clock = FakeClock(et(10, 0))
    monitor = DataQualityMonitor(list(config.data.symbols), config.data, clock=clock)
    strategy = FakeStrategy(fire=lambda ctx: True)
    journal = FakeJournal()
    engine = build_engine(
        strategies=[strategy],
        journal=journal,
        clock=clock,
        config=config,
        data_quality=monitor,
    )

    engine.on_bar(make_bar(minute_index=0, close=40.0))
    assert engine.data_status is DataStatus.STALE
    events = engine.drain_events()
    assert [event.kind for event in events] == [SignalEventKind.DATA_UNAVAILABLE]
    assert strategy.calls == []
    assert journal.signals == []

    # Real trades arrive through the websocket path -> monitor turns OK.
    for symbol, price in (("SOXL", 40.0), ("SOXS", 10.0)):
        monitor.record_tick(
            Tick(symbol=symbol, price=price, volume=100.0, ts_utc=clock.now, trade_id=1)
        )
    clock.advance(30.0)
    engine.on_bar(make_bar(minute_index=1, close=40.0))
    assert engine.data_status is DataStatus.OK
    assert len(events_of(engine, SignalEventKind.ENTRY_SIGNAL)) == 1


def test_integration_real_session_vwap_callback() -> None:
    """Engine + builder A's real SessionVWAP through the injected callback."""
    pytest.importorskip("intraday_scalper.aggregation.bars")
    from datetime import timedelta

    from intraday_scalper.core.models import Tick
    from intraday_scalper.aggregation.bars import SessionVWAP

    config = load_config()
    clock = FakeClock(et(10, 0))
    vwaps = {
        symbol: SessionVWAP(symbol, session_open_utc=OPEN_UTC) for symbol in config.data.symbols
    }
    engine = build_engine(
        strategies=[FakeStrategy(fire=never_fire)],
        clock=clock,
        config=config,
        vwap=lambda symbol, now: vwaps[symbol].value,
    )
    vwaps["SOXL"].on_tick(Tick("SOXL", 40.0, 100.0, clock.now, trade_id=1))
    vwaps["SOXL"].on_tick(
        Tick("SOXL", 40.5, 300.0, clock.now + timedelta(seconds=1), trade_id=2)
    )
    engine.on_bar(make_bar(minute_index=0, close=40.375))
    snapshot = engine.state_snapshot()
    assert snapshot.symbols["SOXL"].vwap == pytest.approx(40.375)
    assert snapshot.symbols["SOXS"].vwap is None  # no ticks -> no fake VWAP
