"""Offline tests for the local web terminal state helpers (no network, no broker)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from intraday_scalper.web import terminal_server as ts


def _et(hour: int, minute: int, day: date = date(2026, 11, 6)) -> datetime:
    """Build a UTC datetime whose America/New_York wall clock is hour:minute (EST)."""
    return datetime(day.year, day.month, day.day, hour + 5, minute, tzinfo=timezone.utc)


def _bars(count: int, start_et: tuple[int, int] = (9, 30), base: float = 10.0) -> list[dict]:
    start = _et(*start_et)
    bars = []
    for index in range(count):
        price = base + index * 0.01
        bars.append(
            {
                "start_utc": start + timedelta(minutes=index),
                "end_utc": start + timedelta(minutes=index + 1),
                "et_day": date(2026, 11, 6),
                "open": price,
                "high": price + 0.05,
                "low": price - 0.05,
                "close": price,
                "volume": 100.0,
                "vwap": price,
            }
        )
    return bars


def test_forced_flat_countdown_at_noon_et():
    moment = _et(12, 0)
    result = ts.forced_flat_countdown(moment)
    assert result["seconds_left"] == 3 * 3600 + 55 * 60
    assert 0.38 < result["fraction_elapsed"] < 0.40
    assert result["is_past"] is False


def test_forced_flat_countdown_after_deadline():
    result = ts.forced_flat_countdown(_et(16, 30))
    assert result["seconds_left"] < 0
    assert result["is_past"] is True
    assert result["fraction_elapsed"] == pytest.approx(1.0)


def test_opening_range_and_status():
    bars = _bars(15)
    orb = ts.opening_range(bars, date(2026, 11, 6))
    assert orb["ready"] is True
    assert orb["bars"] == 15
    assert orb["high"] == pytest.approx(bars[-1]["high"])
    assert orb["low"] == pytest.approx(bars[0]["low"])

    status = ts.orb_status(orb["high"] + 1.0, orb, bars[-1])
    assert status["state"] == "break_up"
    status_down = ts.orb_status(orb["low"] - 1.0, orb, bars[-1])
    assert status_down["state"] == "break_down"
    inside = ts.orb_status((orb["high"] + orb["low"]) / 2, orb, bars[-1])
    assert inside["state"] == "inside"


def test_opening_range_waits_for_data():
    orb = ts.opening_range(_bars(5), date(2026, 11, 6))
    assert orb["ready"] is False
    assert ts.orb_status(10.0, orb, None)["state"] == "waiting"


def test_detect_liquidity_sweep_bullish():
    bars = _bars(22, base=10.0)
    # prior 20 bars swing low = 9.95; last bar sweeps below and closes back above
    bars[-1]["low"] = 9.80
    bars[-1]["close"] = 10.30
    result = ts.detect_liquidity_sweep(bars)
    assert result["state"] == "bullish"


def test_detect_liquidity_sweep_none_and_short_history():
    assert ts.detect_liquidity_sweep(_bars(5))["state"] == "none"
    inside = _bars(22)
    prior = inside[:-1][-20:]
    swing_low = min(bar["low"] for bar in prior)
    swing_high = max(bar["high"] for bar in prior)
    inside[-1]["low"] = swing_low + 0.001
    inside[-1]["high"] = swing_high - 0.001
    inside[-1]["close"] = (swing_low + swing_high) / 2
    assert ts.detect_liquidity_sweep(inside)["state"] == "none"


def test_aggregate_5m_bucket():
    bars = _bars(10)
    aggregated = ts.aggregate_5m(bars, date(2026, 11, 6))
    assert len(aggregated) == 2
    first = aggregated[0]
    assert first["complete"] is True
    assert first["open"] == pytest.approx(bars[0]["open"])
    assert first["close"] == pytest.approx(bars[4]["close"])
    assert first["high"] == pytest.approx(max(bar["high"] for bar in bars[:5]))
    assert first["low"] == pytest.approx(min(bar["low"] for bar in bars[:5]))
    assert first["volume"] == pytest.approx(500.0)


def test_build_execution_card_targets_and_breakeven():
    snapshot = {
        "latest_signal": {
            "symbol": "SOXL",
            "strategy": "ORB_5M",
            "entry_ref": 100.0,
            "stop_loss": 99.0,
            "target_r": 2.0,
        },
        "latest_plan": {"shares": 20.0, "notional": 2000.0, "actual_risk": 20.0},
        "position": None,
    }
    card = ts.build_execution_card(snapshot)
    assert card["has_signal"] is True
    assert card["target"] == pytest.approx(102.0)
    assert card["risk_per_share"] == pytest.approx(1.0)
    assert card["breakeven_trigger"] == pytest.approx(101.2)
    assert card["breakeven_stop"] == pytest.approx(100.05)
    assert card["breakeven_state"] == "none"
    assert card["target_4_5r"] == pytest.approx(104.5)
    assert card["ladder_1_5r_trigger"] == pytest.approx(101.5)
    assert card["ladder_1_5r_stop"] == pytest.approx(100.20)
    assert card["ladder_2_2r_trigger"] == pytest.approx(102.2)
    assert card["ladder_2_2r_stop"] == pytest.approx(101.20)
    assert card["ladder_3_0r_trigger"] == pytest.approx(103.0)
    assert card["ladder_3_0r_stop"] == pytest.approx(102.00)


def test_build_execution_card_active_breakeven():
    snapshot = {
        "latest_signal": {"symbol": "SOXS", "strategy": "VWAP_RECLAIM", "entry_ref": 50.0, "stop_loss": 49.0, "target_r": 2.0},
        "latest_plan": {"shares": 10.0, "notional": 500.0, "actual_risk": 10.0},
        "position": {"unrealized_r": 1.5},
    }
    card = ts.build_execution_card(snapshot)
    assert card["breakeven_state"] == "active"


def test_build_ui_state_shape_and_risk_math():
    config = SimpleNamespace(risk=SimpleNamespace(daily_loss_limit=60.0))
    snapshot = {
        "mode": "DUAL_WATCH",
        "session_phase": "REGULAR",
        "data_status": "OK",
        "connection_state": "CONNECTED",
        "entries_today": 2,
        "max_entries_per_day": 15,
        "trades_today": 2,
        "daily_pnl": -30.0,
        "daily_stopped": False,
        "risk": {"consecutive_losses": 1, "cooldown_until_utc": None, "daily_realized_pnl": -30.0},
        "latest_signal": None,
        "latest_plan": None,
        "position": None,
    }
    tracker_state = {
        "symbols": {
            "SOXL": {"symbol": "SOXL", "last_price": 44.0, "volume": 1234, "bars_1m": 30, "bars_5m": 6},
            "SOXS": {"symbol": "SOXS", "last_price": 74.0, "volume": 567, "bars_1m": 28, "bars_5m": 5},
        }
    }
    state = ts.build_ui_state(snapshot, tracker_state, config, prev_closes={"SOXL": 43.0, "SOXS": 75.0})
    assert state["connection"]["mode"] == "DUAL_WATCH"
    assert state["risk"]["entries_today"] == 2
    assert state["risk"]["loss_progress"] == pytest.approx(0.5)
    assert state["symbols"]["SOXL"]["change_pct"] == pytest.approx((44 / 43 - 1) * 100, rel=1e-3)
    assert state["symbols"]["SOXS"]["change_pct"] < 0
    assert state["countdown"]["deadline_et"] == "15:55"
    assert "execution" in state and "radar" in state


def test_aggregate_timeframe_15m():
    bars = _bars(30)  # 09:30 .. 09:59
    frames = ts.aggregate_timeframe(bars, 15)
    assert len(frames) == 2
    first = frames[0]
    assert first["complete"] is True
    assert first["open"] == pytest.approx(bars[0]["open"])
    assert first["close"] == pytest.approx(bars[14]["close"])
    assert first["high"] == pytest.approx(max(bar["high"] for bar in bars[:15]))
    assert first["low"] == pytest.approx(min(bar["low"] for bar in bars[:15]))


def test_session_tracker_candles_payload():
    tracker = ts.SessionTracker()
    tracker.current_day = date(2026, 11, 6)  # fixed synthetic session under test
    for bar in _bars(30):
        core = __import__("intraday_scalper.core.models", fromlist=["Bar"]).Bar(
            symbol="SOXL",
            timeframe_minutes=1,
            start_utc=bar["start_utc"],
            end_utc=bar["end_utc"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"],
            vwap=bar["vwap"],
            trade_count=10,
        )
        tracker.on_bar(core)
    one = tracker.candles("SOXL", "1m")
    five = tracker.candles("SOXL", "5m")
    assert len(one) == 30
    assert len(five) == 6
    assert set(one[0]) == {"t", "o", "h", "l", "c", "v"}
    assert five[0]["o"] == pytest.approx(one[0]["o"])
