"""Offline, deterministic tests for the event-driven backtester.

Every test uses synthetic bars and fake strategies: no network, no real clock,
no real historical data (the current Finnhub account has none) and no
dependency on the other builders' in-flight modules.  The tests explicitly
inject the built-in indicator module, the fallback strategy context and the
equivalent risk manager, so the suite is stable regardless of which teammate's
file is currently available.

Look-ahead safety is asserted directly: strategies only ever receive confirmed
bars up to ``i`` and signals can only fill at the open of bar ``i+1``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from intraday_scalper.backtest import indicators as bt_indicators
from intraday_scalper.backtest._compat import FallbackStrategyContext
from intraday_scalper.backtest.engine import (
    DATA_CAVEAT,
    BacktestResult,
    EventDrivenBacktester,
    run_backtest,
    run_synthetic_backtest,
    validate_bars_1m,
)
from intraday_scalper.backtest.risk_adapter import (
    SimulatedRiskManager,
    size_position_equivalent,
)
from intraday_scalper.backtest.synthetic import (
    generate_session,
    generate_sessions,
    make_choppy_session,
    make_trending_session,
    next_trading_dates,
)
from intraday_scalper.core import (
    Bar,
    DataStatus,
    Mode,
    RiskBlockReason,
    SignalCandidate,
    StrategyName,
    load_config,
    to_et,
)

UTC = timezone.utc
# Friday 2026-11-06 (US standard time): 09:30 ET == 14:30 UTC.
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SESSION_DATE = date(2026, 11, 6)


# --------------------------------------------------------------------------- #
# Bar builders
# --------------------------------------------------------------------------- #
def make_bar(
    start: datetime,
    *,
    open_: float = 40.0,
    high: float = 40.05,
    low: float = 39.95,
    close: float = 40.0,
    volume: float = 10_000.0,
    symbol: str = "SOXL",
    complete: bool = True,
    timeframe_minutes: int = 1,
    vwap: float | None = None,
) -> Bar:
    if vwap is None:
        vwap = (open_ + high + low + close) / 4.0
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
        trade_count=20,
        is_complete=complete,
    )


def flat_bars(
    count: int,
    *,
    start: datetime = SESSION_START,
    price: float = 40.0,
    spread: float = 0.05,
    symbol: str = "SOXL",
    volume: float = 10_000.0,
) -> list[Bar]:
    """``count`` 1m bars hovering around ``price`` (mildly bullish candle shape)."""
    return [
        make_bar(
            start + timedelta(minutes=index),
            open_=price,
            high=price + spread,
            low=price - spread,
            close=price,
            symbol=symbol,
            volume=volume,
        )
        for index in range(count)
    ]


def constant_bars(
    count: int,
    *,
    start: datetime = SESSION_START,
    open_: float = 40.0,
    high: float = 41.6,
    low: float = 39.9,
    close: float = 40.0,
    symbol: str = "SOXL",
) -> list[Bar]:
    """``count`` bars with the same OHLC each minute (target-hit / stop-hit tests)."""
    return [
        make_bar(
            start + timedelta(minutes=index),
            open_=open_,
            high=high,
            low=low,
            close=close,
            symbol=symbol,
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# Config / engine builders
# --------------------------------------------------------------------------- #
def make_config(**risk_overrides):
    config = load_config()
    if risk_overrides:
        config = replace(config, risk=replace(config.risk, **risk_overrides))
    return config


def build_engine(config, strategies, **overrides) -> EventDrivenBacktester:
    options = dict(
        risk=SimulatedRiskManager(config),
        sizing_fn=size_position_equivalent,
        indicators=bt_indicators,
        context_class=FallbackStrategyContext,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )
    options.update(overrides)
    return EventDrivenBacktester(config, strategies, **options)


def make_candidate(
    ctx,
    *,
    stop_pct: float = 0.0125,
    target_r: float = 2.0,
    strategy: StrategyName = StrategyName.ORB_5M,
    symbol: str | None = None,
) -> SignalCandidate:
    last = ctx.bars_1m[-1]
    return SignalCandidate(
        symbol=symbol or ctx.symbol,
        strategy=strategy,
        entry_ref=last.close,
        stop_loss=last.close * (1.0 - stop_pct),
        target_r=target_r,
        rationale="deterministic fake strategy for offline backtest tests",
        created_utc=ctx.now_utc,
        confirmation_bar_start_utc=last.start_utc,
    )


class FireAtBarCount:
    """Emit one candidate exactly when this symbol has ``trigger_at`` bars."""

    def __init__(
        self,
        trigger_at: int = 6,
        *,
        stop_pct: float = 0.0125,
        target_r: float = 2.0,
        strategy: StrategyName = StrategyName.ORB_5M,
        symbol: str | None = None,
    ) -> None:
        self.trigger_at = trigger_at
        self.stop_pct = stop_pct
        self.target_r = target_r
        self.name = strategy
        self.symbol = symbol

    def evaluate(self, ctx):
        if self.symbol is not None and ctx.symbol != self.symbol:
            return None
        if len(ctx.bars_1m) != self.trigger_at:
            return None
        return make_candidate(
            ctx,
            stop_pct=self.stop_pct,
            target_r=self.target_r,
            strategy=self.name,
            symbol=self.symbol,
        )


class FireAlways(FireAtBarCount):
    """Emit a candidate on every confirmed bar for its symbol."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("trigger_at", 1)
        super().__init__(**kwargs)

    def evaluate(self, ctx):
        if self.symbol is not None and ctx.symbol != self.symbol:
            return None
        if not ctx.bars_1m:
            return None
        return make_candidate(
            ctx,
            stop_pct=self.stop_pct,
            target_r=self.target_r,
            strategy=self.name,
            symbol=self.symbol,
        )


class FireNever:
    name = StrategyName.ORB_5M

    def evaluate(self, ctx):  # noqa: ARG002 - intentionally no signal
        return None


@dataclass(frozen=True, slots=True)
class Observation:
    now_utc: datetime
    n_bars: int
    last_start: datetime
    last_close: float
    n_5m: int
    vwap: float | None
    ema_fast: float | None
    rsi: float | None
    atr: float | None
    or5_complete: bool | None
    or5_high: float | None
    or5_low: float | None
    avg_volume_20: float | None
    symbol: str


class RecordingStrategy:
    """Records exactly what the engine exposes at each confirmed bar."""

    name = StrategyName.ORB_5M

    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def evaluate(self, ctx):
        opening_5 = ctx.opening_range_5m
        last = ctx.bars_1m[-1]
        self.observations.append(
            Observation(
                now_utc=ctx.now_utc,
                n_bars=len(ctx.bars_1m),
                last_start=last.start_utc,
                last_close=last.close,
                n_5m=len(ctx.bars_5m),
                vwap=ctx.vwap,
                ema_fast=ctx.ema_fast,
                rsi=ctx.rsi,
                atr=ctx.atr,
                or5_complete=None if opening_5 is None else opening_5.complete,
                or5_high=None if opening_5 is None else opening_5.high,
                or5_low=None if opening_5 is None else opening_5.low,
                avg_volume_20=ctx.avg_volume_20,
                symbol=ctx.symbol,
            )
        )
        return None


class FakeDataQuality:
    def __init__(self, status: DataStatus) -> None:
        self.status = status

    def report(self, now):  # noqa: ARG002 - deterministic fake
        return SimpleNamespace(aggregate=self.status)


class AlwaysRejectRiskManager:
    """Risk manager stub that always blocks; fails loudly if a fill is confirmed."""

    def __init__(self, reason: RiskBlockReason = RiskBlockReason.DATA_NOT_READY) -> None:
        self.reason = reason

    def roll(self, now=None):
        return None

    def evaluate_entry(self, candidate, plan, now, *, data_ready=True):  # noqa: ARG002
        return SimpleNamespace(allowed=False, reason=self.reason, plan=plan, message="fake reject")

    def confirm_entry(self, *args, **kwargs):
        raise AssertionError("engine must not confirm an entry after a rejection")

    def confirm_exit(self, *args, **kwargs):
        raise AssertionError("engine must not exit when no position was opened")


# --------------------------------------------------------------------------- #
# 1) Next-bar-open fills, no look-ahead
# --------------------------------------------------------------------------- #
def test_signal_fills_at_next_bar_open_not_at_signal_close():
    config = make_config()
    bars = flat_bars(12)
    # Bar 5 closes at 40 and triggers the signal (6th confirmed bar).
    # Bar 6 opens with a small gap at 40.05 -> the fill price proves the engine
    # used the *next* bar's open, not the signal bar's close.
    bars[6] = make_bar(
        SESSION_START + timedelta(minutes=6),
        open_=40.05,
        high=40.10,
        low=39.95,
        close=40.00,
    )
    # Bar 8 reaches the 41.00 target.
    bars[8] = make_bar(
        SESSION_START + timedelta(minutes=8),
        open_=40.00,
        high=41.30,
        low=39.95,
        close=41.00,
    )

    result = build_engine(config, [FireAtBarCount(6)]).run(bars)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_time_utc == bars[6].start_utc
    assert trade.entry_time_utc != bars[5].start_utc
    assert trade.entry_price_signal == pytest.approx(bars[5].close)
    assert trade.entry_price_fill == pytest.approx(40.05)
    assert trade.exit_reason == "TARGET"
    assert trade.exit_price_fill == pytest.approx(41.00)
    assert trade.shares == pytest.approx(30.0)  # $20 / $0.50 stop, capped by $1200 notional
    assert trade.pnl == pytest.approx((41.00 - 40.05) * 30.0)
    assert result.metrics.total_trades == 1
    assert result.metrics.final_equity == pytest.approx(3000.0 + trade.pnl)
    assert result.equity_curve[-1] == (trade.exit_time_utc, result.metrics.final_equity)


def test_look_ahead_future_bars_do_not_change_history():
    config = make_config()
    baseline = flat_bars(20)
    modified = flat_bars(20)
    # Rewrite only the *future* bars 10..19: huge up-spike.
    for index in range(10, 20):
        modified[index] = make_bar(
            SESSION_START + timedelta(minutes=index),
            open_=44.0,
            high=46.0,
            low=43.0,
            close=45.0,
        )

    recorder_a = RecordingStrategy()
    recorder_b = RecordingStrategy()
    build_engine(config, [recorder_a]).run(baseline)
    build_engine(config, [recorder_b]).run(modified)

    assert len(recorder_a.observations) == 20
    assert len(recorder_b.observations) == 20
    # Everything up to and including bar 9 must be bit-for-bit identical.
    assert recorder_a.observations[:10] == recorder_b.observations[:10]

    # The engine never exposes a bar the strategy has not confirmed yet.
    for index, observation in enumerate(recorder_a.observations):
        assert observation.n_bars == index + 1
        assert observation.last_start == baseline[index].start_utc
        assert observation.last_close == baseline[index].close
        assert observation.now_utc == baseline[index].end_utc


def test_context_indicators_and_5m_aggregation():
    config = make_config()
    bars = flat_bars(10)
    recorder = RecordingStrategy()
    build_engine(config, [recorder]).run(bars)
    observations = recorder.observations

    # 5m buckets are anchored at the 09:30 ET open and only emitted once complete.
    assert [observation.n_5m for observation in observations] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    # True session VWAP from completed bars (flat session -> 40).
    assert all(observation.vwap == pytest.approx(40.0) for observation in observations)
    # Opening range 5m is complete for the first time on the 09:34 bar (ends 09:35).
    assert observations[3].or5_complete is None or observations[3].or5_complete is False
    assert observations[4].or5_complete is True
    assert observations[4].or5_high == pytest.approx(40.05)
    assert observations[4].or5_low == pytest.approx(39.95)
    # EMA(9) seeds at index 8; RSI(14) at index 14; ATR(14) at index 13.
    assert observations[7].ema_fast is None
    assert observations[8].ema_fast == pytest.approx(40.0)
    assert observations[9].rsi is None
    assert observations[9].atr is None
    assert all(observation.avg_volume_20 is None for observation in observations)


def test_five_minute_bars_are_valid_and_anchor_aligned():
    config = make_config()
    bars = flat_bars(11)
    recorder = RecordingStrategy()
    build_engine(config, [recorder]).run(bars)
    last = recorder.observations[-1]
    # The 5m bar itself is validated by core.Bar when the engine builds it:
    # two completed 5m bars by the 11th 1m bar (09:30-09:35, 09:35-09:40).
    assert last.n_5m == 2


# --------------------------------------------------------------------------- #
# 2) Pessimistic TP/SL handling and costs
# --------------------------------------------------------------------------- #
def test_same_bar_tp_and_sl_collision_fills_stop_and_flags_ambiguous():
    config = make_config()
    bars = flat_bars(8)
    bars[3] = make_bar(
        SESSION_START + timedelta(minutes=3),
        open_=40.00,
        high=41.50,  # target hit
        low=39.40,  # stop hit
        close=40.50,
    )
    result = build_engine(config, [FireAtBarCount(3)]).run(bars)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "STOP"
    assert trade.exit_price_fill == pytest.approx(39.50)
    assert trade.pnl == pytest.approx((39.50 - 40.00) * 30.0)
    assert result.ambiguous_bars == 1
    assert result.trade_details[trade.signal_id]["ambiguous_bar"] is True
    assert any("ambiguous" in note for note in result.notes)


def test_target_first_pessimism_setting_is_available():
    config = make_config()
    bars = flat_bars(8)
    bars[3] = make_bar(
        SESSION_START + timedelta(minutes=3),
        open_=40.00,
        high=41.50,
        low=39.40,
        close=40.50,
    )
    result = build_engine(config, [FireAtBarCount(3)], pessimism="target_first").run(bars)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "TARGET"
    assert result.trades[0].exit_price_fill == pytest.approx(41.00)
    assert result.ambiguous_bars == 1


def test_slippage_and_commissions_are_charged_on_both_sides():
    config = make_config()
    bars = flat_bars(12)
    bars[5] = make_bar(
        SESSION_START + timedelta(minutes=5),
        open_=40.00,
        high=41.30,
        low=39.95,
        close=41.00,
    )
    result = build_engine(
        config,
        [FireAtBarCount(3)],
        slippage_bps=10.0,
        commission_per_share=0.01,
    ).run(bars)

    assert len(result.trades) == 1
    trade = result.trades[0]
    # entry: open 40 * (1 + 10bps); exit: target 41 * (1 - 10bps)
    expected_entry = 40.0 * 1.001
    expected_exit = 41.0 * 0.999
    assert trade.entry_price_fill == pytest.approx(expected_entry)
    assert trade.exit_price_fill == pytest.approx(expected_exit)
    fees = 30.0 * 0.01 * 2.0
    assert trade.fees == pytest.approx(fees)
    assert trade.pnl == pytest.approx((expected_exit - expected_entry) * 30.0 - fees)
    # Slippage decomposition, both sides explicit.
    details = result.trade_details[trade.signal_id]
    assert details["entry_slippage_dollars"] == pytest.approx((expected_entry - 40.0) * 30.0)
    assert details["exit_slippage_dollars"] == pytest.approx((41.0 - expected_exit) * 30.0)
    assert result.metrics.total_slippage_dollars == pytest.approx(
        details["entry_slippage_dollars"] + details["exit_slippage_dollars"]
    )
    assert result.metrics.total_fees_dollars == pytest.approx(fees)
    # MAE/MFE are tracked from the bars while the trade is open.
    assert details["mfe_dollars"] >= 30.0 * (41.30 - expected_entry) - 1e-9
    assert details["mae_dollars"] >= 0.0


def test_gap_through_stop_cancels_entry():
    config = make_config()
    bars = flat_bars(8)
    bars[3] = make_bar(
        SESSION_START + timedelta(minutes=3),
        open_=39.20,  # gaps below the 39.50 stop
        high=39.30,
        low=39.10,
        close=39.20,
    )
    result = build_engine(config, [FireAtBarCount(3)]).run(bars)
    assert result.trades == []
    assert any("gapped through the stop" in note for note in result.notes)


def test_pending_entry_expires_when_next_minute_is_missing():
    config = make_config()
    bars = flat_bars(5) + flat_bars(5, start=SESSION_START + timedelta(minutes=6))
    # Bars 0..4 then a gap at minute 5: the signal confirmed by bar index 4
    # (expected fill 14:35) can never fill at 14:36.
    result = build_engine(config, [FireAtBarCount(5)]).run(bars)
    assert result.trades == []
    assert any("cancelled" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 3) Risk rules through the (equivalent) risk manager
# --------------------------------------------------------------------------- #
def test_max_15_entries_per_day_enforced():
    config = make_config()
    bars = constant_bars(25)
    result = build_engine(config, [FireAlways()]).run(bars)
    assert len(result.trades) == 15
    assert result.risk_block_counts.get(RiskBlockReason.MAX_ENTRIES_REACHED.value, 0) >= 1
    assert all(trade.pnl > 0 for trade in result.trades)


def test_daily_loss_circuit_breaker_stops_new_entries():
    config = make_config(max_daily_net_loss=50.0, consecutive_stop_threshold=99)
    bars = constant_bars(12, high=40.20, low=39.40)
    result = build_engine(config, [FireAlways()]).run(bars)

    # Each stop loses $15 (30 shares * $0.50). Four losses = -$60 <= -$50.
    assert len(result.trades) == 4
    assert sum(trade.pnl for trade in result.trades) == pytest.approx(-60.0)
    assert result.risk_block_counts.get(RiskBlockReason.DAILY_LOSS_LIMIT.value, 0) >= 1
    assert all(trade.exit_reason == "STOP" for trade in result.trades)


def test_mutual_exclusion_blocks_second_symbol_while_position_open():
    config = make_config()
    soxl = flat_bars(10, symbol="SOXL")
    soxs = flat_bars(10, symbol="SOXS")
    bars = [bar for pair in zip(soxl, soxs) for bar in pair]
    bars.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    # SOXL reaches its 41.00 target on minute 5 while the SOXS strategy keeps
    # firing at minute 1 while the SOXL position is open -> must be blocked.
    soxl_target = next(bar for bar in bars if bar.symbol == "SOXL" and bar.start_utc == SESSION_START + timedelta(minutes=5))
    bars[bars.index(soxl_target)] = make_bar(
        soxl_target.start_utc,
        symbol="SOXL",
        open_=40.00,
        high=41.30,
        low=39.95,
        close=41.00,
    )

    strategies = [
        FireAtBarCount(1, symbol="SOXL"),
        FireAtBarCount(2, symbol="SOXS"),
    ]
    result = build_engine(config, strategies).run(bars)

    assert len(result.trades) == 1
    assert result.trades[0].symbol == "SOXL"
    assert (
        result.risk_block_counts.get(RiskBlockReason.DUAL_CONFLICT.value, 0)
        + result.risk_block_counts.get(RiskBlockReason.POSITION_ALREADY_OPEN.value, 0)
        >= 1
    )


def test_mode_long_soxs_filters_soxl_candidates():
    config = make_config()
    soxl = constant_bars(10, symbol="SOXL")
    soxs = constant_bars(10, symbol="SOXS")
    bars = [bar for pair in zip(soxl, soxs) for bar in pair]
    bars.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    result = build_engine(config, [FireAlways()], mode=Mode.LONG_SOXS).run(bars)
    assert result.trades
    assert all(trade.symbol == "SOXS" for trade in result.trades)
    assert result.risk_block_counts.get(RiskBlockReason.DUAL_CONFLICT.value, 0) >= 1


def test_session_end_forces_flat_with_eod_exit():
    config = make_config()
    bars = flat_bars(390)  # full 09:30..15:59 ET regular session
    assert bars[-1].start_utc == datetime(2026, 11, 6, 20, 59, tzinfo=UTC)  # 15:59 ET
    result = build_engine(
        config, [FireAtBarCount(2, stop_pct=0.05, target_r=2.0)]
    ).run(bars)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "EOD_EXIT"
    assert trade.exit_time_utc == bars[-1].end_utc
    assert trade.holding_minutes > 0
    assert trade.exit_price_fill == pytest.approx(bars[-1].close)


def test_multi_day_session_roll_resets_daily_counters():
    config = make_config(max_entries_per_day=2)
    day1 = constant_bars(8, start=SESSION_START)
    day2_start = SESSION_START + timedelta(days=3)  # Monday 2026-11-09 14:30 UTC
    day2 = constant_bars(8, start=day2_start)
    result = build_engine(config, [FireAlways()]).run(day1 + day2)
    assert len(result.trades) == 4  # 2 allowed per session, counters rolled
    assert len({trade.entry_time_utc.date() for trade in result.trades}) == 2


# --------------------------------------------------------------------------- #
# 4) Data-quality gate, injection points, ordering
# --------------------------------------------------------------------------- #
def test_data_quality_gate_blocks_all_evaluations():
    config = make_config()
    result = build_engine(
        config,
        [FireAlways()],
        data_quality=FakeDataQuality(DataStatus.STALE),
    ).run(flat_bars(6))
    assert result.trades == []
    assert any("data-quality gate" in note for note in result.notes)
    assert result.bars_processed == 6


def test_data_quality_ok_does_not_block():
    config = make_config()
    result = build_engine(
        config,
        [FireAtBarCount(3)],
        data_quality=FakeDataQuality(DataStatus.OK),
    ).run(flat_bars(6))
    # One entry happens; it may still be open at the end but must not be gated.
    assert result.bars_processed == 6
    assert not any("data-quality gate" in note for note in result.notes)


def test_injected_risk_manager_rejection_is_respected():
    config = make_config()
    fake = AlwaysRejectRiskManager(RiskBlockReason.DATA_NOT_READY)
    result = build_engine(config, [FireAlways()], risk=fake).run(flat_bars(6))
    assert result.trades == []
    assert result.risk_block_counts.get(RiskBlockReason.DATA_NOT_READY.value, 0) >= 1


def test_strategy_order_first_valid_candidate_wins():
    config = make_config()
    bars = flat_bars(10)
    strategies = [
        FireNever(),
        FireAtBarCount(3, strategy=StrategyName.VWAP_RECLAIM),
        FireAtBarCount(3, strategy=StrategyName.EMA_PULLBACK),
    ]
    result = build_engine(config, strategies).run(bars)
    assert len(result.trades) == 1
    assert result.trades[0].strategy == StrategyName.VWAP_RECLAIM


class WrongConfirmationStrategy:
    name = StrategyName.ORB_5M

    def evaluate(self, ctx):
        last = ctx.bars_1m[-1]
        if len(ctx.bars_1m) != 3:
            return None
        # Deliberately claim confirmation at the bar *close* instead of the bar
        # start: the engine must reject this as a look-ahead attempt.
        return SignalCandidate(
            symbol=ctx.symbol,
            strategy=self.name,
            entry_ref=last.close,
            stop_loss=last.close * 0.99,
            target_r=2.0,
            rationale="wrong confirmation timestamp",
            created_utc=ctx.now_utc,
            confirmation_bar_start_utc=ctx.now_utc,
        )


def test_candidate_with_wrong_confirmation_bar_is_rejected():
    config = make_config()
    result = build_engine(config, [WrongConfirmationStrategy()]).run(flat_bars(8))
    assert result.trades == []
    assert any("confirmation_bar_start_utc" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 5) Input validation / result plumbing
# --------------------------------------------------------------------------- #
def test_shuffled_bars_raise():
    bars = flat_bars(20)
    shuffled = list(bars)
    random.Random(0).shuffle(shuffled)
    with pytest.raises(ValueError, match="sorted"):
        validate_bars_1m(shuffled)
    with pytest.raises(ValueError, match="sorted"):
        build_engine(make_config(), [FireNever()]).run(shuffled)


def test_duplicate_and_wrong_timeframe_bars_raise():
    bars = flat_bars(3)
    with pytest.raises(ValueError, match="duplicate"):
        validate_bars_1m([bars[0], bars[0]])
    five_minute = make_bar(SESSION_START, timeframe_minutes=5)
    with pytest.raises(ValueError, match="1-minute"):
        validate_bars_1m([five_minute])


def test_incomplete_and_premarket_bars_are_skipped():
    premarket = make_bar(SESSION_START - timedelta(minutes=60))  # 08:30 ET
    regular = flat_bars(10)
    regular[5] = make_bar(
        regular[5].start_utc, complete=False
    )  # unfinished bar: never usable
    bars = [premarket] + regular
    recorder = RecordingStrategy()
    result = build_engine(make_config(), [recorder]).run(bars)
    assert result.bars_processed == 9
    assert result.skipped_bars == 2
    assert len(recorder.observations) == 9


def test_engine_constructor_validation():
    config = make_config()
    with pytest.raises(ValueError, match="at least one strategy"):
        EventDrivenBacktester(config, [])
    with pytest.raises(TypeError):
        EventDrivenBacktester(config, [object()])
    with pytest.raises(ValueError, match="pessimism"):
        build_engine(config, [FireNever()], pessimism="optimistic")
    with pytest.raises(ValueError, match="initial_equity"):
        build_engine(config, [FireNever()], initial_equity=0.0)
    with pytest.raises(ValueError, match="slippage_bps"):
        build_engine(config, [FireNever()], slippage_bps=-1.0)


def test_empty_bars_returns_zero_metrics():
    result = build_engine(make_config(), [FireNever()]).run([])
    assert isinstance(result, BacktestResult)
    assert result.trades == []
    assert result.metrics.total_trades == 0
    assert result.bars_processed == 0
    assert any("no bars" in note for note in result.notes)
    assert DATA_CAVEAT in result.notes


def test_backtest_result_to_dict_is_json_safe():
    config = make_config()
    bars = flat_bars(12)
    bars[5] = make_bar(
        SESSION_START + timedelta(minutes=5),
        open_=40.00,
        high=41.30,
        low=39.95,
        close=41.00,
    )
    result = run_backtest(bars, config, [FireAtBarCount(3)], risk=SimulatedRiskManager(config))
    payload = result.to_dict()
    assert payload["metrics"]["total_trades"] == 1
    assert payload["trades"][0]["exit_reason"] == "TARGET"
    assert payload["equity_curve"]
    text = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in text


def test_default_risk_manager_resolution_works_without_injection():
    """The engine must be runnable with no risk injected (Lead's report path)."""

    config = make_config()
    engine = EventDrivenBacktester(
        config,
        [FireAlways()],
        sizing_fn=size_position_equivalent,
        indicators=bt_indicators,
        context_class=FallbackStrategyContext,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )
    result = engine.run(constant_bars(6))
    assert result.bars_processed == 6
    assert isinstance(result.trades, list)
    # Resolution must always yield a manager exposing the frozen interface; when
    # the built-in equivalent is used the result notes must say so explicitly.
    assert all(
        callable(getattr(engine.risk, name, None))
        for name in ("roll", "evaluate_entry", "confirm_entry", "confirm_exit")
    )
    if isinstance(engine.risk, SimulatedRiskManager):
        assert any(note.startswith("risk:") for note in result.notes)


# --------------------------------------------------------------------------- #
# 6) Sizing fallback (mirrors the frozen risk.sizing contract)
# --------------------------------------------------------------------------- #
def _candidate(entry: float = 40.0, stop: float = 39.5) -> SignalCandidate:
    return SignalCandidate(
        symbol="SOXL",
        strategy=StrategyName.ORB_5M,
        entry_ref=entry,
        stop_loss=stop,
        target_r=2.0,
        rationale="sizing test",
        created_utc=SESSION_START,
        confirmation_bar_start_utc=SESSION_START,
    )


def test_size_position_equivalent_notional_cap():
    config = make_config()
    plan = size_position_equivalent(_candidate(40.0, 39.5), config.risk)
    assert plan.shares == pytest.approx(30.0)
    assert plan.notional == pytest.approx(1200.0)
    assert plan.theoretical_risk == pytest.approx(15.0)
    assert plan.est_slippage_dollars == pytest.approx(30.0 * 79.5 * 1.5 / 10_000.0)
    assert plan.actual_risk == pytest.approx(15.0 + plan.est_slippage_dollars)
    assert plan.clipped_by_notional is True
    assert plan.clipped_by_risk is False
    assert plan.valid is True


def test_size_position_equivalent_risk_cap_and_zero_plan():
    config = make_config()
    plan = size_position_equivalent(_candidate(40.0, 39.0), config.risk)
    assert plan.shares == pytest.approx(20.0)  # $20 / $1.00 stop
    assert plan.theoretical_risk == pytest.approx(20.0)
    assert plan.clipped_by_risk is True
    assert plan.clipped_by_notional is False

    tiny = replace(config.risk, max_notional_per_trade=5.0)
    zero = size_position_equivalent(_candidate(40.0, 39.5), tiny)
    assert zero.shares == 0.0
    assert zero.valid is False


def test_size_position_equivalent_fractional_truncation():
    config = make_config()
    fractional = replace(
        config.risk,
        allow_fractional_shares=True,
        fractional_precision=5,
        max_notional_per_trade=100.0,
    )
    plan = size_position_equivalent(_candidate(41.0, 40.0), fractional)
    assert plan.shares == pytest.approx(2.43902)
    assert plan.shares < 100.0 / 41.0  # never rounds up


# --------------------------------------------------------------------------- #
# 7) Equivalent risk-manager state machine (injected by the engine)
# --------------------------------------------------------------------------- #
def _plan(config, entry: float = 40.0, stop: float = 39.5):
    return size_position_equivalent(_candidate(entry, stop), config.risk)


def test_simulated_risk_manager_cooldown_after_two_stops():
    config = make_config()
    risk = SimulatedRiskManager(config)
    plan = _plan(config)
    t0 = SESSION_START + timedelta(minutes=5)  # 09:35 ET

    assert risk.evaluate_entry(_candidate(), plan, t0).allowed is True
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t0, signal_id="s1")
    risk.confirm_exit(fill_price=39.5, fill_utc=t0 + timedelta(minutes=1), reason="STOP")

    t1 = t0 + timedelta(minutes=2)
    assert risk.evaluate_entry(_candidate(), plan, t1).allowed is True
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t1, signal_id="s2")
    risk.confirm_exit(fill_price=39.5, fill_utc=t1 + timedelta(minutes=1), reason="STOP")

    cooldown_until = t1 + timedelta(minutes=1) + timedelta(minutes=20)
    blocked = risk.evaluate_entry(_candidate(), plan, cooldown_until - timedelta(minutes=1))
    assert blocked.allowed is False
    assert blocked.reason == RiskBlockReason.CONSECUTIVE_LOSS_COOLDOWN
    assert risk.evaluate_entry(_candidate(), plan, cooldown_until).allowed is True


def test_simulated_risk_manager_switch_cooldown():
    config = make_config()
    risk = SimulatedRiskManager(config)
    plan = _plan(config)
    t0 = SESSION_START + timedelta(minutes=5)
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t0, signal_id="s1")
    exit_time = t0 + timedelta(minutes=1)
    risk.confirm_exit(fill_price=40.5, fill_utc=exit_time, reason="TARGET")  # win: no consecutive cooldown

    soxs = SignalCandidate(
        symbol="SOXS",
        strategy=StrategyName.ORB_5M,
        entry_ref=20.0,
        stop_loss=19.7,
        target_r=2.0,
        rationale="switch",
        created_utc=t0,
        confirmation_bar_start_utc=t0,
    )
    soxs_plan = size_position_equivalent(soxs, config.risk)
    blocked = risk.evaluate_entry(soxs, soxs_plan, exit_time + timedelta(minutes=2))
    assert blocked.allowed is False
    assert blocked.reason == RiskBlockReason.SWITCH_COOLDOWN
    assert risk.evaluate_entry(soxs, soxs_plan, exit_time + timedelta(minutes=5)).allowed is True


def test_simulated_risk_manager_entries_cap_after_hours_and_mode():
    config = make_config(max_entries_per_day=1)
    risk = SimulatedRiskManager(config)
    plan = _plan(config)
    t0 = SESSION_START + timedelta(minutes=5)
    assert risk.evaluate_entry(_candidate(), plan, t0).allowed is True
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t0, signal_id="s1")
    risk.confirm_exit(fill_price=40.5, fill_utc=t0 + timedelta(minutes=1), reason="TARGET")
    capped = risk.evaluate_entry(_candidate(), plan, t0 + timedelta(minutes=2))
    assert capped.reason == RiskBlockReason.MAX_ENTRIES_REACHED

    fresh = SimulatedRiskManager(config)
    after_hours = SESSION_START + timedelta(minutes=390 + 5)  # 16:05 ET
    decision = fresh.evaluate_entry(_candidate(), _plan(config), after_hours)
    assert decision.reason == RiskBlockReason.NO_NEW_ENTRIES_TIME

    single = SimulatedRiskManager(make_config().with_mode(Mode.LONG_SOXL))
    dual_candidate = SignalCandidate(
        symbol="SOXS",
        strategy=StrategyName.ORB_5M,
        entry_ref=20.0,
        stop_loss=19.7,
        target_r=2.0,
        rationale="wrong mode",
        created_utc=t0,
        confirmation_bar_start_utc=t0,
    )
    blocked = single.evaluate_entry(
        dual_candidate, size_position_equivalent(dual_candidate, config.risk), t0
    )
    assert blocked.allowed is False
    assert blocked.reason == RiskBlockReason.DUAL_CONFLICT


def test_simulated_risk_manager_eod_alerts_only_when_position_open():
    config = make_config()
    risk = SimulatedRiskManager(config)
    plan = _plan(config)
    t0 = SESSION_START + timedelta(minutes=5)
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t0, signal_id="s1")

    eod_check = SESSION_START + timedelta(minutes=380)  # 15:50 ET
    first = risk.eod_alerts(eod_check)
    assert len(first) == 1
    assert first[0].level.value == "WARNING"
    assert "系统不会自动清仓" in first[0].message
    assert risk.eod_alerts(eod_check + timedelta(minutes=1)) == []

    eod_urgent = SESSION_START + timedelta(minutes=385)  # 15:55 ET
    urgent = risk.eod_alerts(eod_urgent)
    assert len(urgent) == 1
    assert urgent[0].level.value == "CRITICAL"
    assert "系统不会自动清仓" in urgent[0].message
    assert risk.eod_alerts(eod_urgent + timedelta(minutes=1)) == []

    flat = SimulatedRiskManager(config)
    assert flat.eod_alerts(eod_check) == []


def test_simulated_risk_manager_mode_switch_requires_flat_book():
    config = make_config().with_mode(Mode.LONG_SOXL)
    risk = SimulatedRiskManager(config)
    plan = _plan(config)
    t0 = SESSION_START + timedelta(minutes=5)
    risk.confirm_entry(plan, fill_price=40.0, fill_utc=t0, signal_id="s1")
    assert risk.set_mode(Mode.LONG_SOXL, t0) is True  # same mode: no-op success
    assert risk.set_mode(Mode.LONG_SOXS, t0) is False  # position open -> refused
    risk.confirm_exit(fill_price=40.5, fill_utc=t0 + timedelta(minutes=1), reason="TARGET")
    assert risk.set_mode(Mode.LONG_SOXS, t0 + timedelta(minutes=2)) is True
    assert risk.snapshot().mode == Mode.LONG_SOXS


# --------------------------------------------------------------------------- #
# 8) Synthetic data generator contract
# --------------------------------------------------------------------------- #
def test_synthetic_session_is_reproducible_and_bars_are_valid():
    first = generate_session("SOXL", SESSION_DATE, seed=11, n_minutes=30)
    second = generate_session("SOXL", SESSION_DATE, seed=11, n_minutes=30)
    other = generate_session("SOXL", SESSION_DATE, seed=12, n_minutes=30)

    assert [bar.to_dict() for bar in first] == [bar.to_dict() for bar in second]
    assert [bar.to_dict() for bar in first] != [bar.to_dict() for bar in other]
    assert len(first) == 30
    assert first[0].start_utc == SESSION_START
    for bar in first:
        assert isinstance(bar, Bar)
        assert bar.is_complete is True
        assert bar.timeframe_minutes == 1
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.low <= bar.vwap <= bar.high


def test_synthetic_sessions_are_sorted_and_helpers_work():
    dates = next_trading_dates(date(2026, 11, 2), 3)
    bars = generate_sessions(("SOXL", "SOXS"), dates, seed=5, n_minutes=20)
    assert len(bars) == 3 * 2 * 20
    assert bars == sorted(bars, key=lambda bar: (bar.start_utc, bar.symbol))
    with pytest.raises(ValueError):
        generate_sessions(("SOXL",), [], seed=1)
    with pytest.raises(ValueError):
        generate_session("SOXL", SESSION_DATE, n_minutes=0)
    with pytest.raises(ValueError):
        generate_session("SOXL", None)  # et_date required, never reads the clock

    trending = make_trending_session("SOXL", SESSION_DATE, seed=1, n_minutes=30, direction=1)
    choppy = make_choppy_session("SOXL", SESSION_DATE, seed=2, n_minutes=30)
    assert len(trending) == len(choppy) == 30
    assert trending[-1].close > choppy[0].open  # +3% drift over 30 minutes
    assert next_trading_dates(date(2026, 11, 6), 1) == [SESSION_DATE]
    assert [d.weekday() for d in next_trading_dates(date(2026, 11, 7), 3)] == [0, 1, 2]


def test_run_synthetic_backtest_marks_results_as_synthetic():
    config = make_config()
    result = run_synthetic_backtest(
        config,
        [FireAlways()],
        days=2,
        seed=9,
        symbols=("SOXL",),
        n_minutes=60,
        risk=SimulatedRiskManager(config),
        sizing_fn=size_position_equivalent,
        indicators=bt_indicators,
        context_class=FallbackStrategyContext,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )
    assert result.trades  # entries do happen on a seeded synthetic walk
    assert any("SYNTHETIC DATA ONLY" in note for note in result.notes)
    assert any("DATA LIMITATION" in note for note in result.notes)
    assert DATA_CAVEAT in result.notes
    assert result.bars_processed == 2 * 60


# --------------------------------------------------------------------------- #
# Risk-manager clock sync: replay dates vs. wall-clock-constructed managers
# --------------------------------------------------------------------------- #
class _RecordingRisk:
    """Minimal RiskManager stand-in that records session-sync calls."""

    def __init__(self) -> None:
        self.begin_session_calls = []
        self.roll_calls = 0

    def begin_session(self, now) -> None:
        self.begin_session_calls.append(now)

    def roll(self, now=None) -> None:
        self.roll_calls += 1

    def evaluate_entry(self, candidate, plan, now, *, data_ready):
        return SimpleNamespace(
            allowed=False,
            reason=RiskBlockReason.DATA_NOT_READY,
            plan=None,
            message="blocked by test risk manager",
        )

    def confirm_entry(self, *args, **kwargs):  # pragma: no cover - never allowed
        raise AssertionError("no entry expected with a blocking risk manager")

    def confirm_exit(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("no exit expected")


def test_engine_syncs_risk_manager_to_replay_sessions():
    """Each replay session must call begin_session(replay date), not roll()."""

    class _NoopStrategy:
        name = "NOOP"

        def evaluate(self, ctx):
            return None

    config = make_config()
    sessions = next_trading_dates(date(2024, 1, 2), 2)
    bars = generate_sessions(["SOXL"], sessions, seed=4, trend=0.0, vol=0.001)
    risk = _RecordingRisk()
    engine = EventDrivenBacktester(
        config,
        [_NoopStrategy()],
        risk=risk,
        indicators=bt_indicators,
        context_class=FallbackStrategyContext,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )
    engine.run(bars)
    assert [to_et(ts).date() for ts in risk.begin_session_calls] == list(sessions)
    assert risk.roll_calls == 0
