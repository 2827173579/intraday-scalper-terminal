"""Offline unit tests for the SQLite journal and the manual command layer."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pytest

from intraday_scalper.core import (
    AlertEvent,
    AlertLevel,
    Mode,
    SignalCandidate,
    SignalEventKind,
    StrategyName,
    TradeStatus,
    load_config,
)
from intraday_scalper.journal import Journal, execute_command, parse_command
from intraday_scalper.risk import size_position

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)  # Friday 2026-11-06, 10:00 ET
DAY_ONE = date(2026, 11, 6)
DAY_TWO = date(2026, 11, 9)  # Monday

CFG = load_config()


def _candidate(
    now: datetime = NOW,
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
        rationale="journal unit test",
        created_utc=now,
        confirmation_bar_start_utc=now - timedelta(minutes=1),
    )


def _plan(candidate: SignalCandidate):
    return size_position(candidate, CFG.risk)


def _record_signal(journal: Journal, now: datetime = NOW, **kwargs) -> str:
    candidate = _candidate(now=now, **kwargs)
    return journal.record_signal(candidate, _plan(candidate))


@pytest.fixture
def journal(tmp_path):
    instance = Journal(tmp_path / "journal.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def _sqlite_rows(db_path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM signals").fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Storage / schema
# --------------------------------------------------------------------------- #
def test_wal_mode_and_tables_exist(tmp_path):
    db_path = tmp_path / "journal.sqlite3"
    journal = Journal(db_path)
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert mode.lower() == "wal"
        assert {"signals", "trades", "events"}.issubset(tables)
        assert journal.db_path == db_path
    finally:
        journal.close()


def test_record_signal_persists_plan_fields(journal):
    candidate = _candidate()
    plan = _plan(candidate)
    signal_id = journal.record_signal(candidate, plan)

    assert signal_id.startswith("SIG-")
    row = _sqlite_rows(journal.db_path)[0]
    assert row["symbol"] == "SOXL"
    assert row["strategy"] == "ORB_5M"
    assert row["mode"] == "dual_watch"
    assert row["entry_ref"] == pytest.approx(20.0)
    assert row["stop_loss"] == pytest.approx(19.5)
    assert row["take_profit"] == pytest.approx(21.0)
    assert row["target_r"] == pytest.approx(2.0)
    assert row["risk_per_share"] == pytest.approx(0.5)
    assert row["shares"] == pytest.approx(plan.shares)
    assert row["notional"] == pytest.approx(plan.notional)
    assert row["status"] == "PENDING"
    created = datetime.fromisoformat(row["created_utc"])
    assert created.tzinfo is not None
    assert created.astimezone(UTC) == NOW


def test_record_signal_accepts_mode_override(journal):
    candidate = _candidate()
    journal.record_signal(candidate, _plan(candidate), mode=Mode.LONG_SOXL)
    assert _sqlite_rows(journal.db_path)[0]["mode"] == "long_soxl"


def test_journal_accepts_config_for_db_path(tmp_path):
    """Convenience constructors: Journal(config) / Journal(config.paths)."""
    from dataclasses import replace

    configured = replace(
        CFG, paths=replace(CFG.paths, journal_db=tmp_path / "from_config.sqlite3")
    )
    via_config = Journal(configured)
    try:
        assert via_config.db_path == tmp_path / "from_config.sqlite3"
        _record_signal(via_config)
    finally:
        via_config.close()

    via_paths = Journal(configured.paths)
    try:
        assert via_paths.db_path == tmp_path / "from_config.sqlite3"
        assert len(via_paths.open_positions()) == 0
    finally:
        via_paths.close()

    with pytest.raises(ValueError):
        Journal()


# --------------------------------------------------------------------------- #
# Entry / exit lifecycle
# --------------------------------------------------------------------------- #
def test_record_entry_and_open_positions(journal):
    signal_id = _record_signal(journal)
    trade_id = journal.record_entry(signal_id, fill_price=20.05, shares=40, ts=NOW)

    assert isinstance(trade_id, int) and trade_id > 0
    positions = journal.open_positions()
    assert len(positions) == 1
    position = positions[0]
    assert position.signal_id == signal_id
    assert position.symbol == "SOXL"
    assert position.shares == pytest.approx(40.0)
    assert position.entry_price == pytest.approx(20.05)
    assert position.entry_price_signal == pytest.approx(20.0)
    assert position.stop_loss == pytest.approx(19.5)
    assert position.take_profit == pytest.approx(21.0)
    assert position.entry_slippage_bps == pytest.approx(25.0)
    assert position.mode is Mode.DUAL_WATCH

    row = _sqlite_rows(journal.db_path)[0]
    assert row["status"] == "OPEN"
    assert row["trade_id"] == trade_id

    conn = sqlite3.connect(str(journal.db_path))
    conn.row_factory = sqlite3.Row
    try:
        trade = conn.execute("SELECT * FROM trades").fetchone()
    finally:
        conn.close()
    assert trade["status"] == "OPEN"
    assert trade["entry_price_signal"] == pytest.approx(20.0)
    assert trade["entry_price_fill"] == pytest.approx(20.05)


def test_record_exit_computes_pnl_and_stats(journal):
    signal_id = _record_signal(journal)
    journal.record_entry(signal_id, fill_price=20.05, shares=40, ts=NOW)
    exit_ts = NOW + timedelta(minutes=10)
    record = journal.record_exit(signal_id, fill_price=21.0, ts=exit_ts, reason="TARGET")

    assert record.status is TradeStatus.CLOSED
    assert record.exit_reason == "TARGET"
    assert record.exit_price_fill == pytest.approx(21.0)
    assert record.entry_price_signal == pytest.approx(20.0)
    assert record.entry_price_fill == pytest.approx(20.05)
    assert record.pnl == pytest.approx((21.0 - 20.05) * 40)
    assert record.holding_minutes == pytest.approx(10.0)
    assert record.trade_id is not None

    assert journal.open_positions() == []
    closed = journal.closed_trades()
    assert len(closed) == 1 and closed[0].signal_id == signal_id

    stats = journal.daily_stats(DAY_ONE)
    assert stats["trades"] == 1
    assert stats["wins"] == 1
    assert stats["losses"] == 0
    assert stats["pnl"] == pytest.approx(38.0)
    assert stats["win_rate"] == pytest.approx(1.0)
    assert stats["expectancy"] == pytest.approx(38.0)


def test_record_stop_marks_stopped(journal):
    signal_id = _record_signal(journal)
    journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)
    record = journal.record_stop(
        signal_id, fill_price=19.5, ts=NOW + timedelta(minutes=5)
    )

    assert record.status is TradeStatus.STOPPED
    assert record.exit_reason == "STOP"
    assert record.pnl == pytest.approx(-20.0)
    assert record.r_multiple == pytest.approx(-1.0)

    stats = journal.daily_stats(DAY_ONE)
    assert stats["stops"] == 1
    assert stats["losses"] == 1
    assert stats["pnl"] == pytest.approx(-20.0)


def test_double_entry_for_same_signal_raises(journal):
    signal_id = _record_signal(journal)
    journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)
    with pytest.raises(ValueError):
        journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)


def test_exit_without_open_trade_raises(journal):
    signal_id = _record_signal(journal)
    with pytest.raises(ValueError):
        journal.record_exit(signal_id, fill_price=21.0, ts=NOW, reason="TARGET")


def test_entry_invalid_arguments(journal):
    signal_id = _record_signal(journal)
    with pytest.raises(ValueError):
        journal.record_entry(signal_id, fill_price=0.0, shares=40, ts=NOW)
    with pytest.raises(ValueError):
        journal.record_entry(signal_id, fill_price=20.0, shares=0.0, ts=NOW)


# --------------------------------------------------------------------------- #
# Cancel / flat
# --------------------------------------------------------------------------- #
def test_cancel_pending_signal(journal):
    signal_id = _record_signal(journal)
    assert journal.cancel_signal(signal_id, "test") is True
    assert journal.cancel_signal(signal_id, "test") is False
    assert _sqlite_rows(journal.db_path)[0]["status"] == "CANCELLED"

    with pytest.raises(ValueError):
        journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)


def test_cancel_unknown_signal_raises(journal):
    with pytest.raises(ValueError):
        journal.cancel_signal("SIG-DOES-NOT-EXIST")


def test_cancel_open_trade_returns_false(journal):
    signal_id = _record_signal(journal)
    journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)
    assert journal.cancel_signal(signal_id) is False


def test_confirm_flat_clears_pending_and_open(journal):
    entered_signal = _record_signal(journal)
    journal.record_entry(entered_signal, fill_price=20.0, shares=40, ts=NOW)
    _record_signal(journal)  # second, still pending

    cleared = journal.confirm_flat()
    assert cleared == 3  # 1 pending signal + 1 open signal + 1 open trade
    assert journal.open_positions() == []
    assert journal.closed_trades() == []

    statuses = {row["status"] for row in _sqlite_rows(journal.db_path)}
    assert statuses == {"CANCELLED"}


# --------------------------------------------------------------------------- #
# Date filtering / statistics
# --------------------------------------------------------------------------- #
def test_closed_trades_and_stats_filter_by_et_date(journal):
    day_one_now = NOW
    sid_one = _record_signal(journal, day_one_now)
    journal.record_entry(sid_one, fill_price=20.0, shares=40, ts=day_one_now)
    journal.record_exit(
        sid_one, fill_price=21.0, ts=day_one_now + timedelta(minutes=5), reason="TARGET"
    )

    day_two_now = NOW + timedelta(days=3)  # Monday
    sid_two = _record_signal(journal, day_two_now)
    journal.record_entry(sid_two, fill_price=20.0, shares=40, ts=day_two_now)
    journal.record_exit(
        sid_two,
        fill_price=19.5,
        ts=day_two_now + timedelta(minutes=5),
        reason="STOP",
    )

    assert len(journal.closed_trades()) == 2
    assert len(journal.closed_trades(DAY_ONE)) == 1
    assert len(journal.closed_trades(DAY_TWO)) == 1
    assert journal.closed_trades(DAY_ONE)[0].symbol == "SOXL"

    stats_one = journal.daily_stats(DAY_ONE)
    assert stats_one["trades"] == 1
    assert stats_one["pnl"] == pytest.approx(40.0)
    stats_two = journal.daily_stats(DAY_TWO)
    assert stats_two["trades"] == 1
    assert stats_two["stops"] == 1
    assert stats_two["pnl"] == pytest.approx(-20.0)


def test_slippage_stats_entry_and_stop_exit(journal):
    signal_id = _record_signal(journal)  # signal price 20.00
    journal.record_entry(signal_id, fill_price=20.05, shares=40, ts=NOW)  # +25 bps
    journal.record_stop(signal_id, fill_price=19.45, ts=NOW + timedelta(minutes=2))

    stats = journal.slippage_stats()
    assert stats["trades"] == 1
    assert stats["entry_count"] == 1
    assert stats["entry_avg_bps"] == pytest.approx(25.0)
    assert stats["exit_count"] == 1
    # Planned stop 19.50, filled 19.45 -> 25.64 bps worse.
    assert stats["exit_avg_bps"] == pytest.approx((19.5 - 19.45) / 19.5 * 10_000)
    assert stats["exit_unknown_reference"] == 0
    assert stats["by_symbol"]["SOXL"]["trades"] == 1
    assert stats["by_symbol"]["SOXL"]["entry_avg_bps"] == pytest.approx(25.0)
    assert stats["by_strategy"]["ORB_5M"]["trades"] == 1


def test_slippage_stats_manual_exit_has_unknown_reference(journal):
    signal_id = _record_signal(journal)
    journal.record_entry(signal_id, fill_price=20.0, shares=40, ts=NOW)
    journal.record_exit(signal_id, fill_price=20.5, ts=NOW + timedelta(minutes=1), reason="MANUAL")

    stats = journal.slippage_stats()
    assert stats["entry_avg_bps"] == pytest.approx(0.0)
    assert stats["exit_count"] == 0
    assert stats["exit_unknown_reference"] == 1


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def _event(message: str, ts: datetime = NOW) -> AlertEvent:
    return AlertEvent(
        kind=SignalEventKind.ENTRY_SIGNAL,
        level=AlertLevel.INFO,
        message=message,
        ts_utc=ts,
        symbol="SOXL",
        signal_id="SIG-TEST",
        details={"shares": 40},
    )


def test_record_and_read_events(journal):
    first = journal.record_event(_event("event one"))
    second = journal.record_event(_event("event two", NOW + timedelta(minutes=1)))
    third = journal.record_event(_event("event three", NOW + timedelta(minutes=2)))
    assert first < second < third

    recent = journal.recent_events(2)
    assert len(recent) == 2
    assert recent[0]["message"] == "event three"
    assert recent[1]["message"] == "event two"
    assert recent[0]["kind"] == "ENTRY_SIGNAL"
    assert recent[0]["level"] == "INFO"
    assert recent[0]["payload"]["symbol"] == "SOXL"
    assert recent[0]["payload"]["details"] == {"shares": 40}
    assert datetime.fromisoformat(recent[0]["ts_utc"]).tzinfo is not None

    assert journal.recent_events(0) == []
    assert len(journal.recent_events(10)) == 3


def test_record_event_rejects_wrong_type(journal):
    with pytest.raises(TypeError):
        journal.record_event("not-an-event")


def test_journal_is_thread_safe(journal):
    def worker(index: int) -> None:
        journal.record_event(_event(f"thread-{index}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(40)))

    conn = sqlite3.connect(str(journal.db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()
    assert count == 40
    assert len(journal.recent_events(40)) == 40


# --------------------------------------------------------------------------- #
# Command parsing
# --------------------------------------------------------------------------- #
def test_parse_command_all_forms():
    parsed = parse_command("entry sig-abc 20.05 40")
    assert parsed.action == "ENTRY"
    assert parsed.signal_id == "SIG-ABC"
    assert parsed.price == pytest.approx(20.05)
    assert parsed.shares == pytest.approx(40.0)

    parsed = parse_command("EXIT SIG-1 21")
    assert parsed.action == "EXIT"
    assert parsed.signal_id == "SIG-1"
    assert parsed.price == pytest.approx(21.0)

    parsed = parse_command("STOP SIG-1 19.45")
    assert parsed.action == "STOP"
    assert parsed.signal_id == "SIG-1"
    assert parsed.price == pytest.approx(19.45)

    parsed = parse_command("stop 19.45")
    assert parsed.action == "STOP"
    assert parsed.signal_id is None
    assert parsed.price == pytest.approx(19.45)

    assert parse_command("CANCEL SIG-2").signal_id == "SIG-2"
    assert parse_command("Flat").action == "FLAT"


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "BUY SOXL 20 40",
        "ENTRY SIG-1 20",
        "ENTRY SIG-1 20 40 extra",
        "ENTRY SIG-1 -20 40",
        "ENTRY SIG-1 20 0",
        "EXIT SIG-1",
        "EXIT SIG-1 abc",
        "STOP",
        "STOP SIG-1 19.5 extra",
        "CANCEL",
        "FLAT now",
    ],
)
def test_parse_command_invalid_raises(line):
    with pytest.raises(ValueError):
        parse_command(line)


# --------------------------------------------------------------------------- #
# Command execution
# --------------------------------------------------------------------------- #
def test_execute_entry_then_exit(journal):
    signal_id = _record_signal(journal)

    entry_message = execute_command(
        journal, f"ENTRY {signal_id} 20.05 40", now=NOW
    )
    assert "已记录买入" in entry_message
    assert "滑点" in entry_message
    assert "不会自动下单" in entry_message
    assert len(journal.open_positions()) == 1

    exit_message = execute_command(
        journal, f"EXIT {signal_id} 21.0", now=NOW + timedelta(minutes=10)
    )
    assert "已记录卖出" in exit_message
    assert "盈亏 $+38.00" in exit_message
    # Entry filled at 20.05 against a 19.50 stop => $0.55 risk/share, R = 0.95/0.55.
    assert "+1.73R" in exit_message
    assert journal.open_positions() == []


def test_execute_stop_with_explicit_and_implicit_signal(journal):
    explicit_id = _record_signal(journal)
    execute_command(journal, f"ENTRY {explicit_id} 20.0 40", now=NOW)
    message = execute_command(
        journal, f"STOP {explicit_id} 19.45", now=NOW + timedelta(minutes=2)
    )
    assert "已记录止损卖出" in message
    assert "止损" in message

    implicit_id = _record_signal(journal, NOW + timedelta(minutes=30))
    execute_command(journal, f"ENTRY {implicit_id} 20.0 40", now=NOW + timedelta(minutes=30))
    message = execute_command(
        journal, "STOP 19.5", now=NOW + timedelta(minutes=32)
    )
    assert "已记录止损卖出" in message
    assert journal.closed_trades()[-1].status is TradeStatus.STOPPED


def test_execute_cancel_and_flat(journal):
    signal_id = _record_signal(journal)
    message = execute_command(journal, f"CANCEL {signal_id}", now=NOW)
    assert "已取消" in message
    message = execute_command(journal, f"CANCEL {signal_id}", now=NOW)
    assert "取消失败" in message

    entered_signal = _record_signal(journal, NOW + timedelta(minutes=5))
    execute_command(journal, f"ENTRY {entered_signal} 20.0 40", now=NOW + timedelta(minutes=5))
    flat_message = execute_command(journal, "FLAT", now=NOW + timedelta(minutes=6))
    assert "已确认当前空仓" in flat_message
    assert "系统不会自动下单或清仓" in flat_message
    assert journal.open_positions() == []


def test_execute_invalid_input_returns_friendly_error(journal):
    assert execute_command(journal, "", now=NOW).startswith("命令错误：")
    assert execute_command(journal, "NOPE", now=NOW).startswith("命令错误：")
    assert execute_command(
        journal, "ENTRY SIG-UNKNOWN 20.0 40", now=NOW
    ).startswith("错误：")
    assert execute_command(journal, "EXIT SIG-UNKNOWN 21.0", now=NOW).startswith(
        "错误："
    )


def test_execute_flat_with_nothing_to_clear(journal):
    message = execute_command(journal, "FLAT", now=NOW)
    assert "清理了 0 条" in message


def test_execute_stop_without_position_returns_error(journal):
    message = execute_command(journal, "STOP 19.5", now=NOW)
    assert message.startswith("错误：")
