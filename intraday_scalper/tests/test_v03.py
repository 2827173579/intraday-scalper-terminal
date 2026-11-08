"""Offline, deterministic tests for the V0.3 structural ablation.

Synthetic data only (two weekday sessions: 2026-11-05 / 2026-11-06, 09:30 ET ==
14:30 UTC in US standard time).  The only real-data test is the explicitly
required "29-fold walk-forward on the local IEX parquet cache" check, which is
read-only and offline.

Coverage (task-11 §测试, >=15 cases): fractional-share fix for expensive SOXS,
strategy subset, 14:00 ET cutoff (evaluation block / 13:55 fill allowed /
pending cancellation), +1.2R breakeven shield (arm then reverse -> +0.05R exit,
no arm below 1.2R, same-bar collision still stop-first), Step-0 equivalence with
V0.2, determinism, variant ladder composition, and the 29-fold WF discipline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.backtest.walkforward import walk_forward_splits
from intraday_scalper.core import (
    Bar,
    SignalCandidate,
    StrategyName,
    load_config,
)
from intraday_scalper.research import v02, v03
from intraday_scalper.research.v02 import (
    EMA_PULLBACK_5M,
    ORB_15M_5M,
    STRATEGY_LABELS,
    VWAP_RECLAIM_5M,
    prepare_v02_data,
)
from intraday_scalper.research.v03 import (
    ABLATION_VARIANTS,
    BREAKEVEN_OFFSET_R,
    BREAKEVEN_TRIGGER_R,
    ENTRY_CUTOFF_ET,
    FRACTIONAL_PRECISION,
    STEP0_VARIANT,
    STEP1_VARIANT,
    STEP2_VARIANT,
    STEP3_VARIANT,
    STEP4_VARIANT,
    V03Variant,
    build_effective_config,
    run_v03_backtest,
    run_v03_walkforward,
)
from intraday_scalper.risk.sizing import size_position

UTC = timezone.utc
DAY1 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
DAY2 = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SLIPPAGE_RATE = 1.5 / 10_000.0


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def make_1m_bars(
    symbol: str,
    session_start: datetime,
    closes: list[float],
    *,
    volume: float = 1000.0,
    trade_count: int = 25,
    spread: float = 0.0008,
) -> list[Bar]:
    """Turn a 1m close path into valid 1m bars (open = previous close)."""
    bars: list[Bar] = []
    previous_close = closes[0]
    for index, close in enumerate(closes):
        open_ = previous_close
        high = max(open_, close) * (1.0 + spread)
        low = min(open_, close) * (1.0 - spread)
        vwap = (open_ + high + low + close) / 4.0
        start = session_start + timedelta(minutes=index)
        bars.append(
            Bar(
                symbol=symbol,
                timeframe_minutes=1,
                start_utc=start,
                end_utc=start + timedelta(minutes=1),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                trade_count=trade_count,
            )
        )
        previous_close = close
    return bars


def soxs_falling_path() -> list[float]:
    return [100.0 - 0.02 * index for index in range(390)]


def breakout_soxl_day2() -> list[float]:
    """09:45 ET ORB breakout then flat: exactly one V0.2-style ORB trade."""
    return [100.0] * 15 + [100.1, 100.3, 100.5, 100.7, 100.9] + [100.9] * 370


def jump_path(jump_minute: int, *, base: float = 99.9, jump: float = 101.5) -> list[float]:
    """Flat below the OR high, then a 5m-bar jump at *jump_minute*.

    ``jump_minute`` is minutes after the 09:30 ET open; the jump bucket spans
    ``[jump_minute, jump_minute + 5)`` and every later close stays at ``jump``.
    Used to place a breakout exactly inside/after the 14:00 ET cutoff.
    """
    path = [100.0] * 15 + [base] * (jump_minute - 15)
    path += [jump] * 5
    path += [jump] * (390 - jump_minute - 5)
    assert len(path) == 390, len(path)
    return path


def build_prepared(soxl_day2: list[float], *, soxs_day2: list[float] | None = None):
    """Two-session, two-symbol prepared V0.2 dataset (reused by the V0.3 engine)."""
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, soxl_day2)
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars(
        "SOXS", DAY2, soxs_day2 if soxs_day2 is not None else soxs_falling_path()
    )
    bars = sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    return prepare_v02_data(bars)


def custom_variant(
    *,
    strategies: tuple[str, ...] = (ORB_15M_5M,),
    fractional: bool = False,
    cutoff=None,
    shield: bool = False,
    step: int = 90,
) -> V03Variant:
    return V03Variant(
        step=step,
        name="test-variant",
        enabled_strategies=tuple(strategies),
        allow_fractional_shares=fractional,
        entry_cutoff_et=cutoff,
        breakeven_shield=shield,
    )


def first_trade(result: v03.V03Result):
    assert result.trades, "expected at least one trade"
    return sorted(result.trades, key=lambda trade: trade.entry_time_utc)[0]


# --------------------------------------------------------------------------- #
# Constants & variant ladder
# --------------------------------------------------------------------------- #
def test_v03_constants_are_a_priori():
    assert ENTRY_CUTOFF_ET == (14, 0) or (ENTRY_CUTOFF_ET.hour, ENTRY_CUTOFF_ET.minute) == (14, 0)
    assert BREAKEVEN_TRIGGER_R == pytest.approx(1.2)
    assert BREAKEVEN_OFFSET_R == pytest.approx(0.05)
    assert FRACTIONAL_PRECISION == 2


def test_ablation_ladder_is_cumulative():
    assert [variant.step for variant in ABLATION_VARIANTS] == [0, 1, 2, 3, 4]
    # Step 0: V0.2 exactly (integer shares, three strategies, no cutoff/shield).
    assert STEP0_VARIANT.allow_fractional_shares is False
    assert STEP0_VARIANT.enabled_strategies == tuple(STRATEGY_LABELS)
    assert STEP0_VARIANT.entry_cutoff_et is None
    assert STEP0_VARIANT.breakeven_shield is False
    # Step 1: only the fractional switch is added.
    assert STEP1_VARIANT.allow_fractional_shares is True
    assert STEP1_VARIANT.enabled_strategies == tuple(STRATEGY_LABELS)
    assert STEP1_VARIANT.entry_cutoff_et is None
    assert STEP1_VARIANT.breakeven_shield is False
    # Step 2: ORB removed, everything else as Step 1.
    assert STEP2_VARIANT.enabled_strategies == (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
    assert STEP2_VARIANT.allow_fractional_shares is True
    assert STEP2_VARIANT.entry_cutoff_et is None
    # Step 3: 14:00 ET cutoff added, no shield yet.
    assert STEP3_VARIANT.entry_cutoff_et == ENTRY_CUTOFF_ET
    assert STEP3_VARIANT.breakeven_shield is False
    # Step 4: shield added on top of Step 3.
    assert STEP4_VARIANT.breakeven_shield is True
    assert STEP4_VARIANT.entry_cutoff_et == ENTRY_CUTOFF_ET
    assert STEP4_VARIANT.enabled_strategies == (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)


def test_strategy_subset_excludes_orb():
    for variant in (STEP2_VARIANT, STEP3_VARIANT, STEP4_VARIANT):
        assert ORB_15M_5M not in variant.enabled_strategies
        assert variant.enabled_strategies == (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
    assert ORB_15M_5M in STEP0_VARIANT.enabled_strategies


def test_variant_rejects_unknown_or_empty_strategy_set():
    with pytest.raises(ValueError, match="unknown"):
        V03Variant(step=99, name="bad", enabled_strategies=("NOPE",))
    with pytest.raises(ValueError, match="at least one"):
        V03Variant(step=99, name="bad", enabled_strategies=())


# --------------------------------------------------------------------------- #
# Fractional shares
# --------------------------------------------------------------------------- #
def test_effective_config_fractional_override_is_in_memory_only():
    config = load_config()
    effective = build_effective_config(config, STEP1_VARIANT)
    assert effective.risk.allow_fractional_shares is True
    assert effective.risk.fractional_precision == FRACTIONAL_PRECISION
    # Untouched risk caps and unmodified source config.
    assert effective.risk.max_risk_per_trade == config.risk.max_risk_per_trade
    assert effective.risk.max_notional_per_trade == config.risk.max_notional_per_trade
    assert config.risk.allow_fractional_shares is False
    # Step 0 returns the config unchanged.
    assert build_effective_config(config, STEP0_VARIANT) is config


def test_fractional_sizing_fixes_expensive_soxs_zero_shares():
    config = load_config()
    # 2024 SOXS adjusted price was ~$4,438: integer-share mode gives 0 shares.
    candidate = SignalCandidate(
        symbol="SOXS",
        strategy=StrategyName.VWAP_RECLAIM,
        entry_ref=4438.0,
        stop_loss=4427.0,
        target_r=2.0,
        rationale="fractional sizing probe",
        created_utc=DAY2,
        confirmation_bar_start_utc=DAY2,
    )
    integer_plan = size_position(candidate, config.risk)
    assert integer_plan.shares == 0.0
    assert integer_plan.valid is False

    fractional_config = build_effective_config(config, STEP1_VARIANT)
    fractional_plan = size_position(candidate, fractional_config.risk)
    assert fractional_plan.shares > 0.0
    assert fractional_plan.valid is True
    assert fractional_plan.notional <= config.risk.max_notional_per_trade + 1e-9
    assert fractional_plan.actual_risk <= config.risk.max_risk_per_trade + 1e-6
    assert fractional_plan.allow_fractional is True


def test_fractional_precision_truncates_down_never_up():
    config = load_config()
    fractional_config = build_effective_config(config, STEP1_VARIANT)
    candidate = SignalCandidate(
        symbol="SOXS",
        strategy=StrategyName.VWAP_RECLAIM,
        entry_ref=1300.0,
        stop_loss=1290.0,
        target_r=2.0,
        rationale="precision probe",
        created_utc=DAY2,
        confirmation_bar_start_utc=DAY2,
    )
    plan = size_position(candidate, fractional_config.risk)
    # notional cap: 1200 / 1300 = 0.923... -> truncated to 0.92 (never rounded up).
    assert plan.shares == pytest.approx(0.92)
    assert plan.notional == pytest.approx(0.92 * 1300.0)
    assert plan.notional <= config.risk.max_notional_per_trade


# --------------------------------------------------------------------------- #
# Step 0 equivalence with V0.2 + determinism
# --------------------------------------------------------------------------- #
def _trade_signature(result) -> list[tuple]:
    return [
        (
            trade.symbol,
            trade.strategy.value,
            trade.entry_time_utc.isoformat(),
            round(trade.entry_price_signal, 8),
            round(trade.entry_price_fill, 8),
            round(trade.stop_price, 8),
            trade.exit_reason,
            round(trade.pnl, 8),
        )
        for trade in sorted(result.trades, key=lambda trade: trade.entry_time_utc)
    ]


def test_step0_engine_matches_v02_engine_exactly():
    data = build_prepared(breakout_soxl_day2())
    config = load_config()
    v02_result = v02.run_v02_backtest(data, config)
    v03_result = run_v03_backtest(data, config, variant=STEP0_VARIANT)
    assert _trade_signature(v03_result) == _trade_signature(v02_result)
    assert v03_result.metrics.total_trades == v02_result.metrics.total_trades
    assert v03_result.metrics.net_pnl_dollars == pytest.approx(
        v02_result.metrics.net_pnl_dollars, abs=1e-9
    )
    assert v03_result.metrics.profit_factor == pytest.approx(
        v02_result.metrics.profit_factor, rel=1e-9
    )


def test_step0_engine_is_deterministic_across_runs():
    data = build_prepared(breakout_soxl_day2())
    config = load_config()
    first = run_v03_backtest(data, config, variant=STEP0_VARIANT)
    second = run_v03_backtest(data, config, variant=STEP0_VARIANT)
    assert _trade_signature(first) == _trade_signature(second)
    assert first.metrics.to_dict() == second.metrics.to_dict()


# --------------------------------------------------------------------------- #
# Strategy subset (Step 2)
# --------------------------------------------------------------------------- #
def test_step2_removes_all_orb_trades():
    data = build_prepared(breakout_soxl_day2())
    config = load_config()
    # Step 0/1 fire the 09:45 ORB breakout ...
    step0 = run_v03_backtest(data, config, variant=STEP0_VARIANT)
    assert any(trade.strategy is StrategyName.ORB_15M for trade in step0.trades)
    # ... while Step 2 (two-strategy set) must not trade it at all.
    step2 = run_v03_backtest(data, config, variant=STEP2_VARIANT)
    assert step2.trades == []
    assert step2.signals == 0


# --------------------------------------------------------------------------- #
# 14:00 ET structural entry cutoff (Step 3)
# --------------------------------------------------------------------------- #
def test_cutoff_blocks_signal_from_bar_closing_at_or_after_1400():
    data = build_prepared(jump_path(270))  # 14:00 ET breakout bucket
    config = load_config()
    without_cutoff = run_v03_backtest(
        data, config, variant=custom_variant(cutoff=None)
    )
    assert without_cutoff.total_trades == 1
    entry = first_trade(without_cutoff)
    # Signal bar closes 14:05 ET -> fill at the next bucket open (14:05 ET).
    assert entry.entry_time_utc == DAY2 + timedelta(minutes=275)

    with_cutoff = run_v03_backtest(
        data, config, variant=custom_variant(cutoff=ENTRY_CUTOFF_ET)
    )
    assert with_cutoff.trades == []
    assert with_cutoff.signals == 0
    assert with_cutoff.cutoff_skipped_evaluations > 0


def test_cutoff_allows_1355_bucket_signal_and_fill():
    data = build_prepared(jump_path(260))  # 13:50-13:55 ET breakout bucket
    config = load_config()
    result = run_v03_backtest(
        data, config, variant=custom_variant(cutoff=ENTRY_CUTOFF_ET)
    )
    assert result.total_trades == 1
    trade = first_trade(result)
    # Signal bar ends 13:55 ET (< cutoff) -> fill at the 13:55 bucket open.
    assert trade.entry_time_utc == DAY2 + timedelta(minutes=265)
    assert result.cutoff_skipped_evaluations > 0


def test_cutoff_cancels_pending_fill_at_or_after_1400():
    """White-box check of the defensive pending-fill cutoff branch."""
    data = build_prepared(jump_path(270))
    config = load_config()
    variant = custom_variant(cutoff=ENTRY_CUTOFF_ET)
    engine = v03._V03Replay(data, config, variant)
    session = data.sessions[-1]
    bucket_start = DAY2 + timedelta(minutes=270)
    candidate = SignalCandidate(
        symbol="SOXL",
        strategy=StrategyName.ORB_15M,
        entry_ref=101.5,
        stop_loss=99.76,
        target_r=2.0,
        rationale="pending cutoff probe",
        created_utc=bucket_start - timedelta(minutes=5),
        confirmation_bar_start_utc=bucket_start - timedelta(minutes=5),
    )
    plan = size_position(candidate, config.risk)
    engine._pending = v03._PendingEntry(
        candidate=candidate,
        plan=plan,
        signal_id="V03-TEST",
        expected_bucket_start_utc=bucket_start,
    )
    engine._try_fill_pending(session, bucket_start)
    assert engine._pending is None
    assert engine._pending_cancelled == 1
    assert engine._trades == []
    assert engine._open is None


# --------------------------------------------------------------------------- #
# +1.2R breakeven shield (Step 4)
# --------------------------------------------------------------------------- #
def _shield_soxl_day2(post: list[float]) -> list[float]:
    pre = [100.0] * 15 + [100.1, 100.3, 100.5, 100.7, 100.9]
    assert len(post) == 370
    return pre + post


# Entry fills at 100.9*(1+slip); stop ~99.76 -> trigger ~102.30, shield stop ~100.97.
_ARM_THEN_REVERSE = (
    [100.9, 102.4, 102.2, 100.6, 100.4]  # +1.2R arm on the 2nd 1m bar
    + [100.2 - 0.02 * index for index in range(20)]  # then break the shield stop
    + [99.0] * 345
)


def test_breakeven_shield_arms_and_exits_at_plus_offset_r():
    data = build_prepared(_shield_soxl_day2(_ARM_THEN_REVERSE))
    config = load_config()
    variant = custom_variant(shield=True)
    result = run_v03_backtest(data, config, variant=variant)
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["breakeven_armed"] is True
    assert detail["shield_stop"] is not None
    assert detail["exit_stop_kind"] == "SHIELD"
    assert trade.exit_reason == "STOP"
    assert result.breakeven_armed_trades >= 1
    assert result.shield_stop_exits >= 1
    # Exited at entry_fill + 0.05R (minus exit slippage), i.e. a tiny positive P&L,
    # not the -1R initial stop.
    assert trade.exit_price_fill == pytest.approx(
        float(detail["shield_stop"]) * (1.0 - SLIPPAGE_RATE), rel=1e-9
    )
    assert trade.pnl > 0.0
    assert -0.05 < trade.r_multiple < 0.20


def test_without_shield_same_path_stops_at_initial_stop():
    data = build_prepared(_shield_soxl_day2(_ARM_THEN_REVERSE))
    config = load_config()
    result = run_v03_backtest(data, config, variant=custom_variant(shield=False))
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["breakeven_armed"] is False
    assert detail["exit_stop_kind"] == "INITIAL"
    assert trade.pnl < 0.0
    assert trade.r_multiple == pytest.approx(-1.0, abs=0.1)


def test_breakeven_shield_not_armed_below_1_2r():
    post = [100.9, 102.0, 101.8] + [101.5 - 0.03 * index for index in range(30)] + [99.0] * 337
    data = build_prepared(_shield_soxl_day2(post))
    result = run_v03_backtest(
        data, load_config(), variant=custom_variant(shield=True)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["breakeven_armed"] is False
    assert detail["shield_stop"] is None
    assert detail["exit_stop_kind"] == "INITIAL"
    assert trade.exit_reason == "STOP"


def test_breakeven_shield_same_bar_collision_still_stop_first():
    soxl_bars = make_1m_bars("SOXL", DAY2, _shield_soxl_day2(_ARM_THEN_REVERSE))
    # Second bar after entry: after arming, one bar spans both target and shield stop.
    soxl_bars[22] = Bar(
        symbol="SOXL",
        timeframe_minutes=1,
        start_utc=DAY2 + timedelta(minutes=22),
        end_utc=DAY2 + timedelta(minutes=23),
        open=102.2,
        high=104.0,  # above the 2R target (~103.18)
        low=100.0,  # below the shield stop (~100.97)
        close=102.2,
        volume=1000.0,
        vwap=102.2,
        trade_count=25,
    )
    soxs_bars = make_1m_bars("SOXS", DAY2, soxs_falling_path())
    bars = sorted(
        make_1m_bars("SOXL", DAY1, [100.0] * 390)
        + soxl_bars
        + make_1m_bars("SOXS", DAY1, [100.0] * 390)
        + soxs_bars,
        key=lambda bar: (bar.start_utc, bar.symbol),
    )
    data = prepare_v02_data(bars)
    result = run_v03_backtest(
        data, load_config(), variant=custom_variant(shield=True)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "STOP"
    assert detail["exit_stop_kind"] == "SHIELD"
    assert detail["ambiguous_bar"] is True
    assert result.ambiguous_bars >= 1


# --------------------------------------------------------------------------- #
# Engine contract
# --------------------------------------------------------------------------- #
def test_engine_requires_begin_session_capable_risk_manager():
    data = build_prepared(breakout_soxl_day2())

    class _NoBeginRisk:
        """No begin_session -> the V0.1 circuit-breaker lock-up risk."""

    with pytest.raises(TypeError, match="begin_session"):
        run_v03_backtest(
            data, load_config(), variant=STEP0_VARIANT, risk=_NoBeginRisk()
        )


def test_cutoff_skipped_counter_on_flat_session():
    """Every completed 5m bucket closing at/after 14:00 ET is skipped for entries."""
    flat = build_prepared([100.0] * 390, soxs_day2=[100.0] * 390)
    result = run_v03_backtest(flat, load_config(), variant=STEP4_VARIANT)
    assert result.trades == []
    # 25 buckets per session close at/after 14:00 ET (13:55..15:55 bucket starts)
    # on two synthetic sessions.
    assert result.cutoff_skipped_evaluations == 50


# --------------------------------------------------------------------------- #
# Real-data walk-forward discipline (explicitly required by the task)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_prepared_data():
    from intraday_scalper.tools.run_real_backtest import load_bars

    bars = load_bars()
    return prepare_v02_data(bars)


def test_real_data_v03_walkforward_has_29_folds_and_no_leakage(real_prepared_data):
    data = real_prepared_data
    config = load_config()
    splits = walk_forward_splits(list(data.dates), 120, 20, 20)
    assert len(splits) == 29
    assert splits[0].train_start == data.dates[0]
    assert max(splits[0].train_dates) < min(splits[0].test_dates)

    wf = run_v03_walkforward(data, config, variant=STEP4_VARIANT)
    assert wf.summary["folds"] == 29
    assert len(wf.folds) == 29
    test_dates: list = []
    for fold_result in wf.folds:
        assert fold_result.leakage_checked is True
        assert max(fold_result.fold.train_dates) < min(fold_result.fold.test_dates)
        test_dates.extend(fold_result.fold.test_dates)
    assert len(test_dates) == len(set(test_dates))  # non-overlapping test windows
    expected = sum(fold.test_result.total_trades for fold in wf.folds)
    assert wf.aggregate_test_metrics.total_trades == expected
    assert wf.summary["total_test_trades"] > 0


def test_real_data_step0_matches_v02_full_period(real_prepared_data):
    data = real_prepared_data
    config = load_config()
    v02_result = v02.run_v02_backtest(data, config)
    v03_result = run_v03_backtest(data, config, variant=STEP0_VARIANT)
    assert v03_result.metrics.total_trades == v02_result.metrics.total_trades == 1206
    assert v03_result.metrics.net_pnl_dollars == pytest.approx(
        v02_result.metrics.net_pnl_dollars, abs=0.01
    )
    assert v03_result.metrics.profit_factor == pytest.approx(
        v02_result.metrics.profit_factor, abs=1e-3
    )
    # Sanity anchor against the recorded V0.2 baseline used by the report tool.
    assert v03_result.metrics.total_trades == 1206
    assert v03_result.metrics.net_pnl_dollars == pytest.approx(-385.40, abs=0.01)
