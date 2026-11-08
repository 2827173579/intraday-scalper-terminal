"""Tests for the Lead-owned core contracts (models / time / config)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from intraday_scalper.core import (
    Bar,
    Mode,
    Position,
    RiskPlan,
    SessionPhase,
    SignalCandidate,
    StrategyName,
    Tick,
    TradeRecord,
    TradeStatus,
    et_date,
    is_regular_session,
    is_trading_day,
    market_phase,
    new_signal_id,
    session_bounds_utc,
    to_et,
    to_kst,
)

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET (EST)
SESSION_END = datetime(2026, 11, 6, 21, 0, tzinfo=UTC)  # 16:00 ET


def make_tick(**overrides) -> Tick:
    kwargs = {
        "symbol": "SOXL",
        "price": 40.0,
        "volume": 100.0,
        "ts_utc": SESSION_START,
    }
    kwargs.update(overrides)
    return Tick(**kwargs)


class TestTick:
    def test_symbol_upper_cased_and_trimmed(self):
        assert make_tick(symbol=" soxl ").symbol == "SOXL"

    def test_naive_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            make_tick(ts_utc=datetime(2026, 11, 6, 14, 30))

    def test_non_positive_price_rejected(self):
        with pytest.raises(ValueError):
            make_tick(price=0)
        with pytest.raises(ValueError):
            make_tick(price=-1.0)

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError):
            make_tick(volume=-5)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            make_tick(price=float("nan"))

    def test_non_utc_input_is_converted(self):
        et = to_et(SESSION_START)
        tick = make_tick(ts_utc=et)
        assert tick.ts_utc.tzinfo == UTC
        assert tick.ts_utc == SESSION_START

    def test_price_volume(self):
        assert make_tick(price=2.0, volume=3.0).price_volume == 6.0


def make_bar(**overrides) -> Bar:
    kwargs = {
        "symbol": "SOXL",
        "timeframe_minutes": 1,
        "start_utc": SESSION_START,
        "end_utc": SESSION_START + timedelta(minutes=1),
        "open": 40.0,
        "high": 40.2,
        "low": 39.9,
        "close": 40.1,
        "volume": 1000.0,
        "vwap": 40.05,
    }
    kwargs.update(overrides)
    return Bar(**kwargs)


class TestBar:
    def test_valid_bar_derived_values(self):
        bar = make_bar()
        assert bar.range == pytest.approx(0.3)
        assert bar.body == pytest.approx(0.1)
        assert bar.body_ratio == pytest.approx(0.1 / 0.3)
        assert bar.is_bullish

    def test_wrong_window_rejected(self):
        with pytest.raises(ValueError, match="exactly"):
            make_bar(end_utc=SESSION_START + timedelta(minutes=2))

    def test_inconsistent_ohlc_rejected(self):
        with pytest.raises(ValueError, match="OHLC"):
            make_bar(high=39.95)
        with pytest.raises(ValueError, match="OHLC"):
            make_bar(low=40.15)

    def test_vwap_outside_range_rejected(self):
        with pytest.raises(ValueError, match="vwap"):
            make_bar(vwap=41.0)

    def test_zero_range_body_ratio_is_zero(self):
        flat = make_bar(open=40, high=40, low=40, close=40, vwap=40)
        assert flat.body_ratio == 0.0


def make_candidate(**overrides) -> SignalCandidate:
    kwargs = {
        "symbol": "SOXL",
        "strategy": StrategyName.ORB_5M,
        "entry_ref": 40.0,
        "stop_loss": 39.5,
        "target_r": 2.0,
        "rationale": "test",
        "created_utc": SESSION_START,
        "confirmation_bar_start_utc": SESSION_START,
    }
    kwargs.update(overrides)
    return SignalCandidate(**kwargs)


class TestSignalCandidate:
    def test_reward_and_target(self):
        candidate = make_candidate()
        assert candidate.risk_per_share == pytest.approx(0.5)
        assert candidate.take_profit_price == pytest.approx(41.0)
        assert candidate.reward_risk == 2.0

    def test_stop_above_entry_rejected_for_long_only_etf(self):
        with pytest.raises(ValueError, match="stop_loss < entry_ref"):
            make_candidate(stop_loss=40.5)

    def test_stop_equal_entry_rejected(self):
        with pytest.raises(ValueError):
            make_candidate(stop_loss=40.0)

    def test_invalid_target_r_rejected(self):
        with pytest.raises(ValueError):
            make_candidate(target_r=0)


def make_plan(**overrides) -> RiskPlan:
    candidate = overrides.pop("candidate", make_candidate())
    kwargs = {
        "candidate": candidate,
        "shares": 40.0,
        "notional": 1600.0,
        "theoretical_risk": 20.0,
        "est_slippage_dollars": 0.12,
        "est_fees_dollars": 0.0,
        "actual_risk": 20.12,
        "clipped_by_notional": False,
        "clipped_by_risk": True,
        "allow_fractional": False,
    }
    kwargs.update(overrides)
    return RiskPlan(**kwargs)


class TestRiskPlan:
    def test_valid(self):
        assert make_plan().valid
        assert make_plan(shares=0.0, notional=0.0, theoretical_risk=0.0, actual_risk=0.0).valid is False

    def test_reward_dollars_uses_actual_risk(self):
        assert make_plan().reward_dollars == pytest.approx(2.0 * 20.12)


class TestPosition:
    def test_open_risk_and_notional(self):
        position = Position(
            signal_id="SIG-1",
            symbol="SOXL",
            strategy=StrategyName.ORB_5M,
            mode=Mode.LONG_SOXL,
            shares=40,
            entry_price=40.0,
            entry_utc=SESSION_START,
            stop_loss=39.5,
            take_profit=41.0,
        )
        assert position.open_risk == pytest.approx(20.0)
        assert position.notional == pytest.approx(1600.0)
        assert position.unrealized_pnl(40.5) == pytest.approx(20.0)
        assert position.unrealized_r(40.5) == pytest.approx(1.0)

    def test_soxs_flag(self):
        position = Position(
            signal_id="SIG-2",
            symbol="SOXS",
            strategy=StrategyName.ORB_5M,
            mode=Mode.LONG_SOXS,
            shares=10,
            entry_price=80.0,
            entry_utc=SESSION_START,
            stop_loss=79.0,
            take_profit=82.0,
        )
        assert position.is_long_soxs


def make_trade(**overrides) -> TradeRecord:
    kwargs = {
        "signal_id": "SIG-1",
        "symbol": "SOXL",
        "strategy": StrategyName.ORB_5M,
        "mode": Mode.LONG_SOXL,
        "shares": 40.0,
        "entry_time_utc": SESSION_START,
        "entry_price_signal": 40.0,
        "entry_price_fill": 40.02,
        "stop_price": 39.5,
        "target_price": 41.0,
        "exit_time_utc": SESSION_START + timedelta(minutes=10),
        "exit_price_fill": 41.0,
        "exit_reason": "TARGET",
        "status": TradeStatus.CLOSED,
    }
    kwargs.update(overrides)
    return TradeRecord(**kwargs)


class TestTradeRecord:
    def test_pnl_and_r_multiple(self):
        trade = make_trade()
        assert trade.pnl == pytest.approx((41.0 - 40.02) * 40.0)
        assert trade.risk_dollars == pytest.approx((40.02 - 39.5) * 40.0)
        assert trade.r_multiple == pytest.approx(trade.pnl / trade.risk_dollars)
        assert trade.holding_minutes == pytest.approx(10.0)

    def test_entry_slippage_bps(self):
        trade = make_trade()
        assert trade.entry_slippage_bps == pytest.approx((0.02 / 40.0) * 10_000)

    def test_open_trade_has_zero_pnl(self):
        trade = make_trade(exit_time_utc=None, exit_price_fill=None, status=TradeStatus.OPEN)
        assert trade.is_closed is False
        assert trade.pnl == 0.0
        assert trade.holding_minutes == 0.0


class TestSignalId:
    def test_unique_and_prefixed(self):
        first = new_signal_id(SESSION_START)
        second = new_signal_id(SESSION_START)
        assert first.startswith("SIG-20261106T143000-")
        assert first != second


class TestTimeUtil:
    def test_winter_offset(self):
        assert to_et(SESSION_START).utcoffset() == timedelta(hours=-5)
        assert to_et(SESSION_START).hour == 9

    def test_summer_dst_offset(self):
        summer = datetime(2026, 7, 6, 13, 30, tzinfo=UTC)
        assert to_et(summer).utcoffset() == timedelta(hours=-4)
        assert to_et(summer).hour == 9

    def test_dst_spring_forward_boundary(self):
        # 2026-03-08 07:00 UTC is 03:00 EDT right after the spring-forward jump.
        before = datetime(2026, 3, 8, 6, 59, tzinfo=UTC)
        after = datetime(2026, 3, 8, 7, 0, tzinfo=UTC)
        assert to_et(before).utcoffset() == timedelta(hours=-5)
        assert to_et(after).utcoffset() == timedelta(hours=-4)
        assert to_et(before).hour == 1
        assert to_et(after).hour == 3

    def test_kst_offset(self):
        assert to_kst(SESSION_START).utcoffset() == timedelta(hours=9)
        assert to_kst(SESSION_START).hour == 23

    def test_et_date(self):
        late = datetime(2026, 11, 7, 3, 0, tzinfo=UTC)  # 2026-11-06 22:00 ET
        assert et_date(late) == date(2026, 11, 6)

    @pytest.mark.parametrize(
        ("utc_hour", "utc_minute", "expected"),
        [
            (14, 29, SessionPhase.PREMARKET),
            (14, 30, SessionPhase.OPENING_RANGE_5),
            (14, 34, SessionPhase.OPENING_RANGE_5),
            (14, 35, SessionPhase.OPENING_RANGE_15),
            (14, 44, SessionPhase.OPENING_RANGE_15),
            (14, 45, SessionPhase.REGULAR),
            (20, 29, SessionPhase.REGULAR),
            (20, 30, SessionPhase.NO_NEW_ENTRIES),
            (20, 50, SessionPhase.EOD_CHECK),
            (20, 55, SessionPhase.EOD_URGENT),
            (21, 0, SessionPhase.AFTER_HOURS),
            (1, 0, SessionPhase.CLOSED),
        ],
    )
    def test_market_phase_boundaries(self, utc_hour, utc_minute, expected):
        ts = datetime(2026, 11, 6, utc_hour, utc_minute, tzinfo=UTC)
        assert market_phase(ts) == expected

    def test_weekend_is_closed(self):
        saturday = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
        assert is_trading_day(saturday) is False
        assert is_regular_session(saturday) is False
        assert market_phase(saturday) == SessionPhase.CLOSED

    def test_session_bounds(self):
        open_utc, close_utc = session_bounds_utc(SESSION_START + timedelta(hours=1))
        assert open_utc == SESSION_START
        assert close_utc == SESSION_END
