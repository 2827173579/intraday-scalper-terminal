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


def test_normalize_symbol_and_crypto_check():
    assert ts.normalize_symbol("BTC/USD") == "BTCUSD"
    assert ts.normalize_symbol("btcusd") == "BTCUSD"
    assert ts.normalize_symbol("btc") == "BTCUSD"
    assert ts.normalize_symbol("XBTUSD") == "BTCUSD"
    assert ts.normalize_symbol("SOXL") == "SOXL"
    assert ts.normalize_symbol("soxs") == "SOXS"
    assert ts.is_crypto_symbol("BTCUSD") is True
    assert ts.is_crypto_symbol("BTC/USD") is True
    assert ts.is_crypto_symbol("SOXL") is False
    assert ts.is_crypto_symbol("SOXS") is False


def test_btc_session_tag():
    dt_asia = datetime(2026, 11, 9, 3, 0, tzinfo=timezone.utc)
    dt_europe = datetime(2026, 11, 9, 11, 0, tzinfo=timezone.utc)
    dt_us = datetime(2026, 11, 9, 19, 0, tzinfo=timezone.utc)
    assert "亚盘" in ts.btc_session_tag(dt_asia)
    assert "欧盘" in ts.btc_session_tag(dt_europe)
    assert "美盘" in ts.btc_session_tag(dt_us)


def test_build_execution_card_crypto_fractional():
    snapshot = {
        "latest_signal": {
            "symbol": "BTCUSD",
            "strategy": "A+F_Wide(0.8%)+Macro",
            "entry_ref": 80000.0,
            "stop_loss": 79200.0,
            "target_r": 4.5,
        },
        "latest_plan": {"shares": 0.0750, "notional": 6000.0, "actual_risk": 60.0},
        "position": None,
    }
    card = ts.build_execution_card(snapshot)
    assert card["symbol"] == "BTCUSD"
    assert card["unit"] == "BTC"
    assert card["target_4_5r"] == pytest.approx(80000.0 + 4.5 * 800.0)
    assert card["ladder_1_5r_trigger"] == pytest.approx(80000.0 + 1.5 * 800.0)
    assert card["ladder_2_2r_trigger"] == pytest.approx(80000.0 + 2.2 * 800.0)
    assert card["ladder_3_0r_trigger"] == pytest.approx(80000.0 + 3.0 * 800.0)


def test_evaluate_btcusd_session_structure():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    bars = []
    base = 80000.0
    for i in range(100):
        t = start + timedelta(minutes=5 * i)
        p = base + i * 20.0
        bars.append({
            "t": int(t.timestamp() * 1000),
            "start_utc": t,
            "o": p,
            "h": p + 50.0,
            "l": p - 50.0,
            "c": p + 10.0,
            "v": 10.0,
        })
    res = ts.evaluate_btcusd_session(bars)
    assert res["symbol"] == "BTCUSD"
    assert "session_vwap" in res
    assert "radar" in res
    assert "execution" in res
    assert res["radar"]["macro_gate"]["ready"] is True
    assert res["execution"]["unit"] == "BTC"


def test_build_ui_state_with_btcusd():
    config = SimpleNamespace(risk=SimpleNamespace(daily_loss_limit=60.0))
    snapshot = {
        "mode": "DUAL_WATCH",
        "session_phase": "REGULAR",
        "data_status": "OK",
        "connection_state": "CONNECTED",
        "entries_today": 0,
        "max_entries_per_day": 15,
        "trades_today": 0,
        "daily_pnl": 0.0,
        "daily_stopped": False,
        "risk": {"consecutive_losses": 0, "cooldown_until_utc": None, "daily_realized_pnl": 0.0},
        "latest_signal": None,
        "latest_plan": None,
        "position": None,
    }
    tracker_state = {
        "symbols": {
            "SOXL": {"symbol": "SOXL", "last_price": 44.0, "volume": 100, "bars_1m": 10, "bars_5m": 2},
            "SOXS": {"symbol": "SOXS", "last_price": 74.0, "volume": 100, "bars_1m": 10, "bars_5m": 2},
            "BTCUSD": {"symbol": "BTCUSD", "last_price": 82000.0, "volume": 500, "bars_1m": 0, "bars_5m": 0},
        }
    }
    btc_state = {
        "symbol": "BTCUSD",
        "last_price": 82000.0,
        "session_vwap": 81500.0,
        "session_tag": "🌏 亚盘 00-08 UTC",
        "macro_status": "多头顺势 (EMA20 > EMA50)",
        "radar": {
            "session_vwap": 81500.0,
            "session_tag": "🌏 亚盘 00-08 UTC",
            "macro_gate": {"ready": True, "bullish": True, "label": "多头顺势"},
            "vol_filter": {"ready": True, "passed": True, "label": "波动活跃"},
            "smc_sweep": {"triggered": False, "label": "无刺穿"},
            "ema_pull": {"triggered": False, "label": "无回踩"},
            "smc_5m": {"state": "none", "label": "—"},
        },
        "execution": {
            "has_signal": False,
            "symbol": "BTCUSD",
            "strategy": "A+F_Wide(0.8%)+Macro",
            "unit": "BTC",
            "shares": 0.0,
        },
    }
    ui_state = ts.build_ui_state(
        snapshot,
        tracker_state,
        config,
        prev_closes={"SOXL": 43.0, "SOXS": 75.0, "BTCUSD": 80000.0},
        btc_state=btc_state,
    )
    assert "BTCUSD" in ui_state["symbols"]
    assert ui_state["symbols"]["BTCUSD"]["last_price"] == 82000.0
    assert "BTCUSD" in ui_state["radar"]
    assert "BTCUSD" in ui_state["execution_by_symbol"]
    assert ui_state["execution_by_symbol"]["BTCUSD"]["unit"] == "BTC"


def test_demo_state_and_candles_btcusd():
    state = ts.demo_state()
    assert "BTCUSD" in state["symbols"]
    assert "BTCUSD" in state["radar"]
    assert "BTCUSD" in state["execution_by_symbol"]

    candles = ts.demo_candles("BTCUSD", "5m")
    assert candles["symbol"] == "BTCUSD"
    assert candles["timeframe"] == "5m"
    assert len(candles["bars"]) > 0
    assert "session_vwap" in candles


# ============================================================================
# Phase 2: Comprehensive BTC/USD Spot Radar & Static Export Unit Tests
# ============================================================================

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from intraday_scalper.tools import export_github_pages as egp


def test_btc_8h_session_vwap_reset():
    start = datetime(2026, 11, 9, 7, 30, tzinfo=timezone.utc)
    bars = []
    # 6 bars from 07:30 to 07:55 (session bucket 00:00:00 UTC) with typical price ~70000
    for i in range(6):
        t = start + timedelta(minutes=5 * i)
        bars.append({
            "start_utc": t,
            "o": 70000.0,
            "h": 70100.0,
            "l": 69900.0,
            "c": 70000.0,
            "v": 100.0,
        })
    # Bars starting at 08:00 (session bucket 08:00:00 UTC) with typical price 80000
    t_0800 = start + timedelta(minutes=30)
    for i in range(16):
        t = t_0800 + timedelta(minutes=5 * i)
        bars.append({
            "start_utc": t,
            "o": 80000.0,
            "h": 80100.0,
            "l": 79900.0,
            "c": 80000.0,
            "v": 50.0,
        })
    res = ts.evaluate_btcusd_session(bars)
    # The session_vwap at the end of the 08:00 bucket should reset and be 80000.0 (not weighed down by 70000)
    assert res["session_vwap"] == pytest.approx(80000.0, abs=1.0)
    assert "欧盘" in res["session_tag"]


def test_btc_macro_ema_gate_pass_and_block():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    # Case A: Strong uptrend -> close > ema_d20 and ema_d20 >= ema_d50 * 0.99
    bars_bull = []
    for i in range(60):
        t = start + timedelta(minutes=5 * i)
        p = 70000.0 + i * 150.0
        bars_bull.append({
            "start_utc": t,
            "o": p,
            "h": p + 100.0,
            "l": p - 100.0,
            "c": p + 50.0,
            "v": 10.0,
        })
    res_bull = ts.evaluate_btcusd_session(bars_bull)
    assert res_bull["radar"]["macro_gate"]["bullish"] is True
    assert res_bull["radar"]["macro_gate"]["label"] == "多头顺势"

    # Case B: Strong downtrend -> close < ema_d20
    bars_bear = []
    for i in range(60):
        t = start + timedelta(minutes=5 * i)
        p = 80000.0 - i * 150.0
        bars_bear.append({
            "start_utc": t,
            "o": p,
            "h": p + 100.0,
            "l": p - 100.0,
            "c": p - 50.0,
            "v": 10.0,
        })
    res_bear = ts.evaluate_btcusd_session(bars_bear)
    assert res_bear["radar"]["macro_gate"]["bullish"] is False
    assert res_bear["radar"]["macro_gate"]["label"] == "日线趋势过滤 · 观望"


def test_btc_volatility_filter_f():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    # Case A: Low volatility (ATR < 0.16% of price)
    bars_flat = []
    for i in range(40):
        t = start + timedelta(minutes=5 * i)
        bars_flat.append({
            "start_utc": t,
            "o": 80000.0,
            "h": 80001.0,
            "l": 79999.0,
            "c": 80000.0,
            "v": 10.0,
        })
    res_flat = ts.evaluate_btcusd_session(bars_flat)
    assert res_flat["radar"]["vol_filter"]["passed"] is False
    assert "过滤" in res_flat["radar"]["vol_filter"]["label"]

    # Case B: Active volatility (ATR >= 0.16% and 30m range >= 1.35 * ATR)
    bars_vol = []
    for i in range(40):
        t = start + timedelta(minutes=5 * i)
        step = 50.0 if i % 2 == 0 else -50.0
        p = 80000.0 + i * 20.0
        # Give periodic expansion so 30m rolling range comfortably exceeds 1.35 * ATR
        extra_h = 300.0 if i % 6 != 0 else 600.0
        bars_vol.append({
            "start_utc": t,
            "o": p,
            "h": p + extra_h,
            "l": p - 300.0,
            "c": p + step,
            "v": 10.0,
        })
    res_vol = ts.evaluate_btcusd_session(bars_vol)
    assert res_vol["radar"]["vol_filter"]["passed"] is True
    assert res_vol["radar"]["vol_filter"]["label"] == "波动活跃"


def test_btc_smc_sweep_entry_signal():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    bars = []
    base = 80000.0
    for i in range(30):
        t = start + timedelta(minutes=5 * i)
        p = base + i * 20.0
        extra_h = 400.0 if i % 5 == 0 else 150.0
        bars.append({
            "start_utc": t,
            "o": p,
            "h": p + extra_h,
            "l": p - 100.0,
            "c": p + 50.0,
            "v": 20.0,
        })
    prior20_low = min(b["l"] for b in bars[-20:])
    t_sweep = start + timedelta(minutes=5 * 30)
    bars.append({
        "start_utc": t_sweep,
        "o": prior20_low + 20.0,
        "h": prior20_low + 600.0,
        "l": prior20_low - 50.0,
        "c": prior20_low + 550.0,
        "v": 50.0,
    })
    t_next = start + timedelta(minutes=5 * 31)
    bars.append({
        "start_utc": t_next,
        "o": prior20_low + 560.0,
        "h": prior20_low + 650.0,
        "l": prior20_low + 500.0,
        "c": prior20_low + 600.0,
        "v": 30.0,
    })
    res = ts.evaluate_btcusd_session(bars)
    entry_markers = [m for m in res["markers"] if m["type"] == "ENTRY" and m["strategy"] == "SMC_SWEEP_L"]
    assert len(entry_markers) >= 1
    assert entry_markers[0]["symbol"] == "BTCUSD"


def test_btc_ema_pullback_entry_signal():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    bars = []
    base = 80000.0
    for i in range(35):
        t = start + timedelta(minutes=5 * i)
        p = base + i * 40.0
        bars.append({
            "start_utc": t,
            "o": p,
            "h": p + 250.0,
            "l": p - 100.0,
            "c": p + 80.0,
            "v": 20.0,
        })
    t_pull = start + timedelta(minutes=5 * 35)
    last_p = bars[-1]["c"]
    bars.append({
        "start_utc": t_pull,
        "o": last_p - 50.0,
        "h": last_p + 150.0,
        "l": last_p - 150.0,
        "c": last_p + 120.0,
        "v": 30.0,
    })
    t_next = start + timedelta(minutes=5 * 36)
    bars.append({
        "start_utc": t_next,
        "o": last_p + 125.0,
        "h": last_p + 250.0,
        "l": last_p + 100.0,
        "c": last_p + 200.0,
        "v": 30.0,
    })
    res = ts.evaluate_btcusd_session(bars)
    entries = [m for m in res["markers"] if m["type"] == "ENTRY"]
    assert len(entries) >= 1


def test_btc_stop_distance_and_min_floor():
    close_price = 80000.0
    min_floor = close_price * 0.0080
    atr = 100.0
    raw_stop = close_price - 0.25 * atr
    stop_dist = max(close_price - raw_stop, close_price * 0.0080)
    assert stop_dist == pytest.approx(min_floor)

    raw_stop_wide = close_price - 960.0
    stop_dist_wide = max(close_price - raw_stop_wide, close_price * 0.0080)
    assert stop_dist_wide == pytest.approx(960.0)
    assert stop_dist_wide / close_price <= 0.0250


def test_btc_fractional_position_sizing():
    risk_per_trade = 60.0
    notional_cap = 10000.0
    entry_price = 75000.0
    stop_dist = 600.0
    shares = round(min(risk_per_trade / stop_dist, notional_cap / entry_price), 4)
    assert shares == pytest.approx(0.1000)
    assert f"{shares:.4f}" == "0.1000"

    entry_high = 90000.0
    stop_dist_wide = 400.0
    shares_capped = round(min(risk_per_trade / stop_dist_wide, notional_cap / entry_high), 4)
    assert shares_capped == pytest.approx(0.1111)


def test_btc_runner_a_ladder_stops():
    entry = 80000.0
    stop_dist = 640.0
    ladder_1_5r_trig = entry + 1.5 * stop_dist
    ladder_1_5r_stop = entry + 0.20 * stop_dist
    ladder_2_2r_trig = entry + 2.2 * stop_dist
    ladder_2_2r_stop = entry + 1.20 * stop_dist
    ladder_3_0r_trig = entry + 3.0 * stop_dist
    ladder_3_0r_stop = entry + 2.00 * stop_dist
    target_4_5r = entry + 4.5 * stop_dist

    assert ladder_1_5r_trig == pytest.approx(80960.0)
    assert ladder_1_5r_stop == pytest.approx(80128.0)
    assert ladder_2_2r_trig == pytest.approx(81408.0)
    assert ladder_2_2r_stop == pytest.approx(80768.0)
    assert ladder_3_0r_trig == pytest.approx(81920.0)
    assert ladder_3_0r_stop == pytest.approx(81280.0)
    assert target_4_5r == pytest.approx(82880.0)


def test_btc_time_6h_rolling_exit():
    start = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)
    bars = []
    base = 80000.0
    for i in range(25):
        t = start + timedelta(minutes=5 * i)
        p = base + i * 30.0
        extra_h = 450.0 if i % 5 == 0 else 200.0
        bars.append({
            "start_utc": t,
            "o": p,
            "h": p + extra_h,
            "l": p - 100.0,
            "c": p + 60.0,
            "v": 20.0,
        })
    prior_low = min(b["l"] for b in bars[-20:])
    t_sweep = start + timedelta(minutes=5 * 25)
    bars.append({
        "start_utc": t_sweep,
        "o": prior_low + 20.0,
        "h": prior_low + 600.0,
        "l": prior_low - 40.0,
        "c": prior_low + 550.0,
        "v": 40.0,
    })
    entry_p = prior_low + 560.0
    t_entry = start + timedelta(minutes=5 * 26)
    bars.append({
        "start_utc": t_entry,
        "o": entry_p,
        "h": entry_p + 100.0,
        "l": entry_p - 100.0,
        "c": entry_p + 10.0,
        "v": 20.0,
    })
    for j in range(1, 76):
        t = t_entry + timedelta(minutes=5 * j)
        bars.append({
            "start_utc": t,
            "o": entry_p + 10.0,
            "h": entry_p + 50.0,
            "l": entry_p - 50.0,
            "c": entry_p + 15.0,
            "v": 20.0,
        })
    res = ts.evaluate_btcusd_session(bars)
    time_exits = [m for m in res["markers"] if m["type"] == "TIME_6H"]
    assert len(time_exits) >= 1
    assert time_exits[0]["symbol"] == "BTCUSD"


def test_api_bars_and_candles_routes():
    server = ts.ThreadingHTTPServer(("127.0.0.1", 0), ts.TerminalHandler)
    server.auth_token = None
    server.candles_provider = lambda sym, tf: {
        "symbol": sym,
        "timeframe": tf,
        "bars": [{"t": 1700000000000, "o": 80000.0, "h": 80100.0, "l": 79900.0, "c": 80050.0, "v": 12.5}],
    }
    server.state_provider = lambda: {"ok": True}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    try:
        url_bars = f"http://127.0.0.1:{port}/api/bars?symbol=BTCUSD&tf=5m"
        with urllib.request.urlopen(url_bars, timeout=5.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["symbol"] == "BTCUSD"
            assert data["timeframe"] == "5m"
            assert len(data["bars"]) == 1

        url_candles = f"http://127.0.0.1:{port}/api/candles?symbol=BTC/USD&tf=15m"
        with urllib.request.urlopen(url_candles, timeout=5.0) as resp:
            assert resp.status == 200
            data2 = json.loads(resp.read().decode("utf-8"))
            assert data2["symbol"] == "BTCUSD"
            assert data2["timeframe"] == "15m"
    finally:
        server.shutdown()
        server.server_close()


def test_static_export_bundle_schema_and_candles():
    bundle = egp.build_static_bundle()
    assert "candles" in bundle
    assert "BTCUSD" in bundle["candles"]
    for tf in ("1m", "5m", "15m", "1d"):
        assert tf in bundle["candles"]["BTCUSD"]
        assert len(bundle["candles"]["BTCUSD"][tf]) > 0
        first_bar = bundle["candles"]["BTCUSD"][tf][0]
        assert set(first_bar.keys()) == {"t", "o", "h", "l", "c", "v"}
    assert "BTCUSD" in bundle["state"]["symbols"]
    assert bundle["state"]["symbols"]["BTCUSD"]["last_price"] is not None
    assert "BTCUSD" in bundle["state"]["radar"]
    assert "BTCUSD" in bundle["state"]["execution_by_symbol"]
    assert bundle["state"]["execution_by_symbol"]["BTCUSD"]["unit"] == "BTC"


def test_inline_js_syntax_check():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    script_pattern = re.compile(r"<script(?:\s+[^>]*)?>(.*?)</script>", re.DOTALL | re.IGNORECASE)
    for html_path in (ts.HTML_PATH, egp.DOCS_DIR / "index.html"):
        if not html_path.exists():
            continue
        content = html_path.read_text(encoding="utf-8")
        matches = script_pattern.findall(content)
        assert len(matches) >= 1, f"No script tags found in {html_path}"
        for idx, block in enumerate(matches, 1):
            code = block.strip()
            if not code:
                continue
            with tempfile.NamedTemporaryFile(suffix=".js", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(["node", "--check", tmp_path], capture_output=True, text=True, timeout=10)
                assert proc.returncode == 0, f"Syntax check failed in {html_path.name} block {idx}: {proc.stderr}"
            finally:
                Path(tmp_path).unlink(missing_ok=True)
