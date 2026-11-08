"""Offline tests for the local tick/bar recorder (tmp dirs only, no network)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intraday_scalper.core import Bar, PathsConfig, Tick
from intraday_scalper.data import DataRecorder

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET
SESSION_DAY = SESSION_START.date()
NEXT_DAY_START = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
FIXED_NOW = SESSION_START + timedelta(minutes=5)


def make_paths(root: Path) -> PathsConfig:
    return PathsConfig(
        package_root=root,
        var_dir=root / "var",
        raw_tick_dir=root / "var" / "raw_ticks",
        bar_dir=root / "var" / "bars",
        log_dir=root / "var" / "logs",
        journal_db=root / "var" / "journal.sqlite3",
        backtest_dir=root / "var" / "backtest",
    )


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


def make_bar(
    start: datetime = SESSION_START,
    *,
    open_: float = 40.0,
    high: float = 40.3,
    low: float = 39.8,
    close: float = 40.1,
    volume: float = 1_000.0,
    vwap: float | None = None,
    timeframe_minutes: int = 1,
    symbol: str = "SOXL",
    trade_count: int = 10,
    is_complete: bool = True,
) -> Bar:
    if vwap is None:
        vwap = (open_ + close) / 2.0
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
        trade_count=trade_count,
        is_complete=is_complete,
    )


def make_recorder(tmp_path: Path) -> DataRecorder:
    return DataRecorder(make_paths(tmp_path), clock=lambda: FIXED_NOW)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# Raw ticks
# --------------------------------------------------------------------------- #
def test_append_tick_writes_jsonl(tmp_path):
    recorder = make_recorder(tmp_path)
    tick = make_tick(40.25, 150.0, SESSION_START + timedelta(seconds=5), trade_id=99)
    path = recorder.append_tick(tick)
    recorder.close()

    assert path == recorder.tick_path(SESSION_DAY)
    assert path.is_file()
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0] == tick.to_dict()
    assert rows[0]["symbol"] == "SOXL"
    assert rows[0]["ts_utc"] == (SESSION_START + timedelta(seconds=5)).isoformat()
    assert recorder.tick_count == 1


def test_ticks_are_grouped_by_et_trading_day(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append_tick(make_tick(ts=SESSION_START))
    recorder.append_tick(make_tick(ts=datetime(2026, 11, 6, 3, 0, tzinfo=UTC)))  # 22:00 ET previous day
    recorder.append_tick(make_tick(ts=NEXT_DAY_START))
    recorder.close()

    first = recorder.tick_path(SESSION_DAY)
    previous = recorder.tick_path(datetime(2026, 11, 5).date())
    monday = recorder.tick_path(NEXT_DAY_START.date())
    assert first.is_file() and previous.is_file() and monday.is_file()
    assert len(read_jsonl(first)) == 1
    assert len(read_jsonl(previous)) == 1
    assert len(read_jsonl(monday)) == 1


def test_threaded_tick_appends_are_not_lost(tmp_path):
    recorder = make_recorder(tmp_path)
    total_threads, per_thread = 4, 25

    def worker(worker_id: int) -> None:
        for index in range(per_thread):
            recorder.append_tick(
                make_tick(
                    40.0 + worker_id * 0.01,
                    100.0,
                    SESSION_START + timedelta(seconds=worker_id * 100 + index),
                    trade_id=worker_id * 1000 + index,
                )
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(total_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recorder.close()

    rows = read_jsonl(recorder.tick_path(SESSION_DAY))
    assert len(rows) == total_threads * per_thread
    assert recorder.tick_count == total_threads * per_thread


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def test_bar_roundtrip_via_parquet(tmp_path):
    recorder = make_recorder(tmp_path)
    bars = [
        make_bar(SESSION_START + timedelta(minutes=index), close=40.1 + index * 0.05, volume=1_000.0 + index)
        for index in range(3)
    ]
    for bar in bars:
        recorder.append_bar(bar)
    recorder.flush()
    recorder.close()

    directory = recorder.bar_directory("SOXL", 1)
    assert directory.is_dir()
    assert list(directory.glob("part-*.parquet"))
    loaded = recorder.load_bars("SOXL", 1)
    assert loaded == bars
    assert recorder.bar_count == 3


def test_multiple_shards_are_loaded_in_time_order(tmp_path):
    recorder = make_recorder(tmp_path)
    first_batch = [make_bar(SESSION_START + timedelta(minutes=index)) for index in range(2)]
    for bar in first_batch:
        recorder.append_bar(bar)
    recorder.flush()

    later = [
        make_bar(SESSION_START + timedelta(minutes=10 + index), close=41.0, high=41.3)
        for index in range(2)
    ]
    for bar in later:
        recorder.append_bar(bar)
    recorder.flush()
    recorder.close()

    loaded = recorder.load_bars("SOXL", 1)
    assert [bar.start_utc for bar in loaded] == [
        SESSION_START,
        SESSION_START + timedelta(minutes=1),
        SESSION_START + timedelta(minutes=10),
        SESSION_START + timedelta(minutes=11),
    ]
    assert recorder.bar_count == 4


def test_load_bars_returns_empty_for_unknown_symbol(tmp_path):
    recorder = make_recorder(tmp_path)
    assert recorder.load_bars("SOXL", 1) == []
    recorder.close()


def test_symbols_and_timeframes_are_stored_separately(tmp_path):
    recorder = make_recorder(tmp_path)
    soxl_1m = make_bar(SESSION_START)
    soxs_1m = make_bar(SESSION_START, symbol="SOXS", open_=6.0, high=6.2, low=5.9, close=6.1, vwap=6.05)
    soxl_5m = make_bar(SESSION_START, timeframe_minutes=5, high=40.5, low=39.5)
    for bar in (soxl_1m, soxs_1m, soxl_5m):
        recorder.append_bar(bar)
    recorder.close()

    assert recorder.load_bars("SOXL", 1) == [soxl_1m]
    assert recorder.load_bars("SOXS", 1) == [soxs_1m]
    assert recorder.load_bars("SOXL", 5) == [soxl_5m]


def test_incomplete_bar_roundtrip(tmp_path):
    recorder = make_recorder(tmp_path)
    partial = make_bar(SESSION_START, is_complete=False)
    recorder.append_bar(partial)
    recorder.close()
    loaded = recorder.load_bars("SOXL", 1)
    assert len(loaded) == 1
    assert loaded[0].is_complete is False


def test_load_bars_is_sorted_even_if_written_out_of_order(tmp_path):
    recorder = make_recorder(tmp_path)
    for minute in (2, 0, 1):
        recorder.append_bar(make_bar(SESSION_START + timedelta(minutes=minute)))
    recorder.close()
    loaded = recorder.load_bars("SOXL", 1)
    assert [bar.start_utc for bar in loaded] == [
        SESSION_START,
        SESSION_START + timedelta(minutes=1),
        SESSION_START + timedelta(minutes=2),
    ]


# --------------------------------------------------------------------------- #
# Lifecycle / robustness
# --------------------------------------------------------------------------- #
def test_append_bar_is_visible_immediately_without_flush(tmp_path):
    """Bars are written through: a crash before close() loses no completed bar."""
    recorder = make_recorder(tmp_path)
    bar = make_bar(SESSION_START)
    path = recorder.append_bar(bar)
    assert path.is_file()
    assert recorder.load_bars("SOXL", 1) == [bar]
    recorder.close()


def test_threaded_bar_appends_are_not_lost(tmp_path):
    recorder = make_recorder(tmp_path)
    total_threads, per_thread = 4, 10

    def worker(worker_id: int) -> None:
        for index in range(per_thread):
            recorder.append_bar(
                make_bar(
                    SESSION_START + timedelta(minutes=worker_id * 10 + index),
                    volume=1_000.0 + worker_id,
                )
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(total_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recorder.close()

    loaded = recorder.load_bars("SOXL", 1)
    assert len(loaded) == total_threads * per_thread
    assert recorder.bar_count == total_threads * per_thread
    assert [bar.start_utc for bar in loaded] == sorted(bar.start_utc for bar in loaded)


def test_close_flushes_tick_handles_and_blocks_further_writes(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append_tick(make_tick())
    recorder.append_bar(make_bar(SESSION_START))
    recorder.close()
    recorder.close()  # idempotent

    assert recorder.closed is True
    assert len(recorder.load_bars("SOXL", 1)) == 1
    with pytest.raises(RuntimeError):
        recorder.append_tick(make_tick())
    with pytest.raises(RuntimeError):
        recorder.append_bar(make_bar(SESSION_START))


def test_append_type_errors(tmp_path):
    recorder = make_recorder(tmp_path)
    with pytest.raises(TypeError):
        recorder.append_tick("tick")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recorder.append_bar("bar")  # type: ignore[arg-type]
    recorder.close()


def test_flush_without_data_is_a_noop(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.flush()
    assert recorder.tick_count == 0
    assert recorder.bar_count == 0
    recorder.close()


def test_writes_stay_inside_the_configured_var_directory(tmp_path):
    recorder = make_recorder(tmp_path)
    recorder.append_tick(make_tick())
    recorder.append_bar(make_bar())
    recorder.close()
    var_dir = (tmp_path / "var").resolve()
    written = [path for path in (tmp_path / "var").rglob("*") if path.is_file()]
    assert written
    for path in written:
        assert var_dir in path.resolve().parents or path.resolve() == var_dir
