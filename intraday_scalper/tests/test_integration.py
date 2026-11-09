"""End-to-end integration tests for the runtime wiring and CLI.

These tests exercise the real modules (no fakes) with deterministic synthetic
data and injected clocks.  They are still offline: no websocket network, no REST
calls, no real system clock.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intraday_scalper.core import Bar, Mode, SignalCandidate, StrategyName, load_config
from intraday_scalper.core.config import AppConfig, PathsConfig
from intraday_scalper.journal.journal import Journal
from intraday_scalper.main import Runtime, load_api_key, main
from intraday_scalper.risk.sizing import size_position

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)  # Monday 09:30 ET


def tmp_config(tmp_path: Path, **overrides) -> AppConfig:
    base = load_config()
    paths = PathsConfig(
        package_root=base.paths.package_root,
        var_dir=tmp_path / "var",
        raw_tick_dir=tmp_path / "var" / "raw_ticks",
        bar_dir=tmp_path / "var" / "bars",
        log_dir=tmp_path / "var" / "logs",
        journal_db=tmp_path / "var" / "journal.sqlite3",
        backtest_dir=tmp_path / "var" / "backtest",
    )
    config = replace(base, paths=paths)
    if overrides:
        config = replace(config, **overrides)
    return config.validate()


class SimClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment = self.moment + timedelta(seconds=seconds)


def feed_tick(runtime: Runtime, symbol: str, price: float, volume: float, ts: datetime) -> None:
    from intraday_scalper.core import Tick

    runtime._on_tick(Tick(symbol=symbol, price=price, volume=volume, ts_utc=ts))


def test_tick_to_bar_to_engine_pipeline(tmp_path: Path):
    config = tmp_config(tmp_path, mode=Mode.DUAL_WATCH)
    clock = SimClock(SESSION_START + timedelta(seconds=10))
    runtime = Runtime(config, "", clock=clock, record_data=True, quality_gate=True, audio=False)
    try:
        # Feed 5 complete minutes of SOXL prints; prices trend upward.
        price = 40.0
        for minute in range(5):
            minute_start = SESSION_START + timedelta(minutes=minute)
            for second in (5, 20, 40, 55):
                clock.moment = minute_start + timedelta(seconds=second)
                price += 0.01
                feed_tick(runtime, "SOXL", round(price, 4), 100.0, clock.moment)
        # A tick in minute 6 completes minute 5.
        clock.moment = SESSION_START + timedelta(minutes=5, seconds=5)
        feed_tick(runtime, "SOXL", round(price + 0.01, 4), 120.0, clock.moment)

        snapshot = runtime.snapshot()
        soxl = snapshot.symbol("SOXL")
        assert soxl is not None
        assert soxl.bars_1m_count >= 5
        assert soxl.last_price is not None and soxl.last_price > 40.0
        assert soxl.last_bar_utc is not None
        assert soxl.vwap is not None and 39.0 < soxl.vwap < 41.0

        # Recorder persisted raw ticks and 1m bars.
        assert runtime.recorder is not None
        tick_files = list((tmp_path / "var" / "raw_ticks").glob("*.jsonl"))
        assert tick_files, "raw tick jsonl was not written"
        first_lines = tick_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert first_lines
        assert json.loads(first_lines[0])["symbol"] == "SOXL"
        runtime.recorder.flush()
        bars = runtime.recorder.load_bars("SOXL", 1)
        assert bars and all(isinstance(b, Bar) for b in bars)
    finally:
        runtime.shutdown()


def test_five_minute_aggregation_from_completed_minutes(tmp_path: Path):
    config = tmp_config(tmp_path, mode=Mode.DUAL_WATCH)
    clock = SimClock(SESSION_START)
    runtime = Runtime(config, "", clock=clock, record_data=False, quality_gate=False, audio=False)
    five = runtime.five_minute
    completed_five = []
    try:
        for minute in range(5):
            bar = Bar(
                symbol="SOXL",
                timeframe_minutes=1,
                start_utc=SESSION_START + timedelta(minutes=minute),
                end_utc=SESSION_START + timedelta(minutes=minute + 1),
                open=40.0 + minute * 0.01,
                high=40.05 + minute * 0.01,
                low=39.95 + minute * 0.01,
                close=40.02 + minute * 0.01,
                volume=1000.0,
                vwap=40.01 + minute * 0.01,
                trade_count=10,
            )
            completed_five.extend(five.on_bar(bar))
        assert len(completed_five) == 1
        agg = completed_five[0]
        assert agg.timeframe_minutes == 5
        assert agg.start_utc == SESSION_START
        assert agg.end_utc == SESSION_START + timedelta(minutes=5)
        assert agg.volume == pytest.approx(5000.0)
        assert agg.vwap == pytest.approx(
            sum((40.01 + i * 0.01) * 1000.0 for i in range(5)) / 5000.0
        )
    finally:
        runtime.shutdown()


def test_runtime_manual_entry_and_exit_sync(tmp_path: Path):
    config = tmp_config(tmp_path, mode=Mode.DUAL_WATCH)
    now = SESSION_START + timedelta(minutes=10)
    clock = SimClock(now)
    runtime = Runtime(config, "", clock=clock, record_data=False, quality_gate=False, audio=False)
    try:
        candidate = SignalCandidate(
            symbol="SOXL",
            strategy=StrategyName.ORB_5M,
            entry_ref=40.0,
            stop_loss=39.5,
            target_r=2.0,
            rationale="integration test breakout",
            created_utc=now,
            confirmation_bar_start_utc=now - timedelta(minutes=1),
        )
        plan = size_position(candidate, config.risk)
        assert plan.valid
        signal_id = runtime.journal.record_signal(candidate, plan)
        runtime._candidates[signal_id] = candidate

        message = runtime.handle_command(f"ENTRY {signal_id} 40.02 {plan.shares:g}", now=now)
        assert "已记录开仓" in message
        assert runtime.risk.position is not None
        assert runtime.risk.position.symbol == "SOXL"
        assert len(runtime.journal.open_positions()) == 1

        # A second entry on the other ETF must be rejected by mutual exclusion.
        other = SignalCandidate(
            symbol="SOXS",
            strategy=StrategyName.ORB_5M,
            entry_ref=80.0,
            stop_loss=79.0,
            target_r=2.0,
            rationale="conflict",
            created_utc=now,
            confirmation_bar_start_utc=now - timedelta(minutes=1),
        )
        other_plan = size_position(other, config.risk)
        other_id = runtime.journal.record_signal(other, other_plan)
        runtime._candidates[other_id] = other
        blocked = runtime.handle_command(f"ENTRY {other_id} 80.0 {other_plan.shares:g}", now=now)
        assert "无法记录开仓" in blocked

        exit_time = now + timedelta(minutes=15)
        close_message = runtime.handle_command(f"EXIT {signal_id} 41.0", now=exit_time)
        assert "已记录平仓" in close_message
        assert runtime.risk.position is None
        trades = runtime.journal.closed_trades()
        assert len(trades) == 1
        trade = trades[0]
        assert trade.exit_price_fill == pytest.approx(41.0)
        assert trade.entry_price_fill == pytest.approx(40.02)
        assert trade.pnl > 0
    finally:
        runtime.shutdown()


def test_data_unavailable_blocks_signal_generation(tmp_path: Path):
    from intraday_scalper.core import DataStatus

    config = tmp_config(tmp_path)
    runtime = Runtime(config, "", clock=SimClock(SESSION_START), record_data=False, audio=False)
    try:
        runtime.engine.set_data_status(DataStatus.STALE, "no fresh ticks")
        snapshot = runtime.snapshot()
        assert snapshot.data_status is DataStatus.STALE
        assert snapshot.data_ready is False
    finally:
        runtime.shutdown()


def test_cli_demo_runs_synthetic_pipeline(capsys):
    exit_code = main(["demo", "--days", "1", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out[captured.out.index("{") :])
    assert payload["mode"] == "dual_watch"
    assert "SOXL" in payload["symbols"]
    assert "SOXS" in payload["symbols"]


def test_load_api_key_reads_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nFINNHUB_API_KEY=test_key_123\n", encoding="utf-8")
    assert load_api_key(env_file) == "test_key_123"
    assert load_api_key(tmp_path / "missing.env") == ""


def test_legacy_repository_files_untouched():
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    if result.returncode != 0:
        pytest.skip("git status failed")
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        # New work is allowed only in the isolated project dirs; legacy files never.
        assert path.startswith(("intraday_scalper/", "daily_research/", "docs/", ".github/", "README.md")), f"legacy path modified: {line}"
