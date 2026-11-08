"""Offline tests for the Rich dashboard and the sound reminders.

Everything is rendered into an in-memory :class:`rich.console.Console` and the
audio layer is exercised with an injected ``beep_fn`` so no sound, network or
real clock is involved.
"""

from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from rich.console import Console

from intraday_scalper.core.config import load_config
from intraday_scalper.core.models import (
    AlertEvent,
    AlertLevel,
    ConnectionState,
    DataStatus,
    Mode,
    RiskPlan,
    SessionPhase,
    SignalCandidate,
    SignalEventKind,
    StrategyName,
)
from intraday_scalper.dashboard import Dashboard, alert_sound
from intraday_scalper.dashboard.audio import BEEP_PATTERNS
from intraday_scalper.signals.state import EngineSnapshot, PositionSnapshot, SymbolSnapshot

UTC = timezone.utc

# Friday 2026-11-06, after US DST ended: 09:30 ET == 14:30 UTC.
TRADING_DATE = datetime(2026, 11, 6, tzinfo=UTC).date()
OPEN_UTC = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
NOW_UTC = datetime(2026, 11, 6, 15, 40, tzinfo=UTC)  # 10:40 ET


def build_snapshot(
    *,
    data_status: DataStatus = DataStatus.OK,
    with_position: bool = False,
    with_signal: bool = True,
    now: datetime = NOW_UTC,
) -> EngineSnapshot:
    candidate = SignalCandidate(
        symbol="SOXL",
        strategy=StrategyName.ORB_5M,
        entry_ref=40.0,
        stop_loss=39.6,
        target_r=2.0,
        rationale="orb breakout above 5m high with volume confirmation",
        created_utc=now,
        confirmation_bar_start_utc=NOW_UTC - timedelta(minutes=5),
    )
    plan = RiskPlan(
        candidate=candidate,
        shares=29.0,
        notional=1160.0,
        theoretical_risk=11.6,
        est_slippage_dollars=0.09,
        est_fees_dollars=0.0,
        actual_risk=11.69,
        clipped_by_notional=True,
        clipped_by_risk=False,
        allow_fractional=False,
    )
    position = None
    if with_position:
        position = PositionSnapshot(
            signal_id="SIG-DASH-001",
            symbol="SOXL",
            strategy="ORB_5M",
            mode="dual_watch",
            shares=29.0,
            entry_price=40.0,
            entry_price_signal=39.98,
            entry_utc=now - timedelta(minutes=20),
            stop_loss=39.6,
            take_profit=40.8,
            last_price=40.5,
            unrealized_pnl=14.5,
            unrealized_r=1.25,
            open_risk=11.6,
            notional=1160.0,
        )
    return EngineSnapshot(
        as_of_utc=now,
        mode=Mode.DUAL_WATCH,
        session_phase=SessionPhase.REGULAR,
        symbols={
            "SOXL": SymbolSnapshot(
                symbol="SOXL",
                last_price=40.5,
                last_bar_utc=now - timedelta(minutes=1),
                last_bar_end_utc=now,
                data_age_seconds=12.0,
                bars_1m_count=70,
                bars_5m_count=14,
                trend_1m="bull",
                trend_5m="bull",
                vwap=40.12,
                ema9=40.31,
                ema21=40.05,
                rsi=61.5,
                atr=0.18,
                avg_volume_20=12345.0,
            ),
            "SOXS": SymbolSnapshot(
                symbol="SOXS",
                last_price=9.85,
                last_bar_utc=now - timedelta(minutes=1),
                last_bar_end_utc=now,
                data_age_seconds=12.0,
                bars_1m_count=70,
                bars_5m_count=14,
                trend_1m="bear",
                trend_5m="bear",
                vwap=9.9,
                ema9=9.88,
                ema21=9.95,
                rsi=41.0,
                atr=0.05,
                avg_volume_20=54321.0,
            ),
        },
        data_status=data_status,
        connection_state=ConnectionState.CONNECTED,
        position=position,
        risk={
            "cooldown_until_utc": None,
            "switch_cooldown_until_utc": None,
        },
        entries_today=3,
        max_entries_per_day=15,
        trades_today=2,
        wins_today=1,
        stops_today=1,
        daily_pnl=12.5,
        daily_stopped=False,
        latest_signal=candidate if with_signal else None,
        latest_plan=plan if with_signal else None,
        latest_signal_id="SIG-DASH-001",
        strategy_names=(
            StrategyName.ORB_5M.value,
            StrategyName.ORB_15M.value,
            StrategyName.VWAP_RECLAIM.value,
            StrategyName.EMA_PULLBACK.value,
        ),
        events=(
            AlertEvent(
                kind=SignalEventKind.ENTRY_SIGNAL,
                level=AlertLevel.WARNING,
                message="【入场信号】买入 SOXL 参考价 40.00 止损 39.60 止盈 40.80（人工下单）",
                ts_utc=now - timedelta(minutes=5),
                symbol="SOXL",
                signal_id="SIG-DASH-001",
            ),
        ),
        eod_close_utc=datetime(2026, 11, 6, 21, 0, tzinfo=UTC),
        breakeven_hint_r=1.2,
        take_profit_hint_r=(2.0, 2.5),
    )


def render_to_text(snapshot: EngineSnapshot, width: int = 300) -> str:
    output = io.StringIO()
    console = Console(record=True, file=output, width=width, force_terminal=False)
    dashboard = Dashboard(config=load_config())
    console.print(dashboard.render(snapshot))
    return console.export_text()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_render_contains_all_core_panels() -> None:
    text = render_to_text(build_snapshot(with_position=True))
    for expected in (
        "SOXL / SOXS 日内信号台",
        "SOXL",
        "SOXS",
        "VWAP",
        "EMA9",
        "EMA21",
        "RSI",
        "DUAL_WATCH",
        "REGULAR",
        "CONNECTED",
        "40.50",
        "9.85",
        "模式 / 风控状态",
        "今日开仓",
        "3 / 15",
        "今日累计盈亏",
        "12.50",
        "持仓（人工）",
        "1.2R",
        "2.0R",
        "EOD 倒计时",
        "需人工平仓，系统不会自动清仓",
        "最近事件",
        "只提醒、人工下单、人工平仓",
    ):
        assert expected in text, f"missing from dashboard: {expected!r}"


def test_render_flat_panel_when_no_position() -> None:
    text = render_to_text(build_snapshot(with_position=False))
    assert "当前无持仓" in text
    assert "持仓（人工）" not in text
    assert "需人工平仓" not in text


def test_data_unavailable_banner_is_rendered() -> None:
    text = render_to_text(build_snapshot(data_status=DataStatus.UNAVAILABLE, with_position=True))
    assert "DATA_UNAVAILABLE — 停止产生交易信号" in text
    assert "DATA_UNAVAILABLE" in text


def test_stale_status_also_shows_the_banner() -> None:
    text = render_to_text(build_snapshot(data_status=DataStatus.STALE))
    assert "DATA_UNAVAILABLE — 停止产生交易信号" in text
    assert "STALE" in text


def test_empty_snapshot_renders_without_exception() -> None:
    snapshot = EngineSnapshot(
        as_of_utc=NOW_UTC,
        mode=Mode.LONG_SOXL,
        session_phase=SessionPhase.PREMARKET,
        symbols={},
    )
    text = render_to_text(snapshot)
    assert "暂无行情数据" in text
    assert "PREMARKET" in text
    assert "LONG_SOXL" in text
    json.dumps(snapshot.to_dict())


def test_position_snapshot_payload_is_serialisable() -> None:
    snapshot = build_snapshot(with_position=True)
    payload = snapshot.to_dict()
    json.dumps(payload)
    assert payload["position"]["symbol"] == "SOXL"
    assert payload["position"]["unrealized_r"] == pytest.approx(1.25)
    assert payload["symbols"]["SOXL"]["trend_1m"] == "bull"
    assert payload["data_status"] == "OK"


def test_snapshot_payload_handles_dates_inside_risk_state() -> None:
    import dataclasses

    # The real RiskManager's RiskState exposes ``et_date: date``; snapshots are
    # logged as JSON, so the converter must serialise it.
    snapshot = dataclasses.replace(
        build_snapshot(),
        risk={"et_date": TRADING_DATE, "entries_today": 3, "position": None},
    )
    payload = snapshot.to_dict()
    json.dumps(payload)
    assert payload["risk"]["et_date"] == TRADING_DATE.isoformat()


class EngineLike:
    """Anything exposing ``state_snapshot()`` may drive the dashboard."""

    def __init__(self) -> None:
        self.calls = 0

    def state_snapshot(self) -> EngineSnapshot:
        self.calls += 1
        return build_snapshot()


def test_run_accepts_an_engine_like_snapshot_source() -> None:
    engine = EngineLike()
    console = Console(file=io.StringIO(), force_terminal=False, width=140)
    dashboard = Dashboard(engine, console=console)
    assert dashboard.run(max_frames=1) == 1
    assert engine.calls == 1


# --------------------------------------------------------------------------- #
# Live loop
# --------------------------------------------------------------------------- #
def test_run_stops_after_max_frames() -> None:
    frames: List[EngineSnapshot] = []

    def provider() -> EngineSnapshot:
        frames.append(build_snapshot())
        return frames[-1]

    console = Console(file=io.StringIO(), force_terminal=False, width=140)
    dashboard = Dashboard(provider, console=console, config=load_config())
    rendered = dashboard.run(refresh_hz=1000.0, max_frames=2)
    assert rendered == 2
    assert len(frames) == 2


def test_run_stops_immediately_when_event_preset() -> None:
    calls: List[int] = []

    def provider() -> EngineSnapshot:
        calls.append(1)
        return build_snapshot()

    console = Console(file=io.StringIO(), force_terminal=False, width=140)
    dashboard = Dashboard(provider, console=console)
    stop_event = threading.Event()
    stop_event.set()
    assert dashboard.run(stop_event=stop_event, refresh_hz=50.0) == 0
    assert calls == []


def test_run_without_provider_raises() -> None:
    dashboard = Dashboard(console=Console(file=io.StringIO(), force_terminal=False))
    with pytest.raises(ValueError):
        dashboard.run(max_frames=1)


# --------------------------------------------------------------------------- #
# Audio reminders
# --------------------------------------------------------------------------- #
class BeepRecorder:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def __call__(self, frequency: int, duration_ms: int) -> None:
        self.calls.append((frequency, duration_ms))


def no_sleep(_seconds: float) -> None:
    return None


def test_info_sound_is_a_single_beep() -> None:
    recorder = BeepRecorder()
    played = alert_sound(AlertLevel.INFO, beep_fn=recorder, sleep_fn=no_sleep)
    assert played == 1
    assert len(recorder.calls) == len(BEEP_PATTERNS[AlertLevel.INFO])
    assert recorder.calls[0][0] > 0 and recorder.calls[0][1] > 0


def test_warning_and_critical_patterns() -> None:
    warning = BeepRecorder()
    critical = BeepRecorder()
    alert_sound(AlertLevel.WARNING, beep_fn=warning, sleep_fn=no_sleep)
    alert_sound(AlertLevel.CRITICAL, beep_fn=critical, sleep_fn=no_sleep)
    assert len(warning.calls) == 2
    assert len(critical.calls) == len(BEEP_PATTERNS[AlertLevel.CRITICAL])
    assert len(critical.calls) > len(warning.calls)
    assert all(frequency > 0 for frequency, _ in critical.calls)


def test_disabled_sound_produces_nothing() -> None:
    recorder = BeepRecorder()
    assert alert_sound(AlertLevel.CRITICAL, enabled=False, beep_fn=recorder) == 0
    assert recorder.calls == []


def test_sound_failure_is_swallowed() -> None:
    def broken_beep(_frequency: int, _duration: int) -> None:
        raise RuntimeError("no audio device")

    assert alert_sound(AlertLevel.CRITICAL, beep_fn=broken_beep, sleep_fn=no_sleep) == 0
