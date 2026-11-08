"""Offline, deterministic tests for the V0.4 structural upgrade.

Synthetic data only except one explicit real-raw-data anchor test (the task
requires the V0.3 raw Step4 anchor 970 trades / -$214.76 / WF 814 / -$234.92 /
16 folds to be reproduced).  Synthetic sessions use weekdays 2026-11-05 /
2026-11-06 (09:30 ET == 14:30 UTC in US standard time).

Coverage: variant ladder, Step-0 == V0.3 STEP4 equivalence, 10:00-14:00 entry
window (lower bound, upper cutoff), VWAP trend filter predicate, SMC sweep rules
and deterministic stop geometry, 1.2R scale-out (partial + breakeven stop,
2R target leg, no-partial stop, same-bar collision), markers / Web trade dicts,
``evaluate_v04_session`` output shape, determinism, and the raw-data 29-fold
walk-forward anchor.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from intraday_scalper.backtest.walkforward import walk_forward_splits
from intraday_scalper.core import Bar, SignalCandidate, StrategyName, load_config
from intraday_scalper.research import v02, v03, v04
from intraday_scalper.research.v02 import (
    EMA_PULLBACK_5M,
    VWAP_RECLAIM_5M,
    prepare_v02_data,
)
from intraday_scalper.research.v04 import (
    AB_COMPARE_VARIANTS,
    ABLATION_STEPS,
    BREAKEVEN_OFFSET_R,
    COMPARE_VARIANTS,
    EMA_EXHAUSTION_GAIN_LIMIT,
    ENTRY_WINDOW_END_ET,
    ENTRY_WINDOW_START_ET,
    FRACTIONAL_PRECISION,
    OR30_BARS,
    PRODUCTION_VARIANT,
    SCALE_PORTION,
    SCALE_TRIGGER_R,
    SMC_LOOKBACK_BARS,
    SMC_RANGE_MULTIPLE,
    SMC_STOP_ATR_BUFFER,
    SMC_SWEEP_5M,
    STEP0_VARIANT,
    STEP1_VARIANT,
    STEP2_VARIANT,
    STEP3_VARIANT,
    STEP3B_VARIANT,
    STEP3C_NOCAP_VARIANT,
    STEP3C_WITHCAP_VARIANT,
    STEP4_VARIANT,
    V04_FINAL_VARIANT,
    V04EvalContext,
    V04Variant,
    build_eval_context,
    ema_pullback_exhaustion_blocked,
    evaluate_v04_session,
    result_to_trade_dicts,
    run_v04_backtest,
    smc_sweep_5m_candidate,
    smc_trend_confirm_ok,
    smc_trend_confirm_require_both,
    strategy_label_for_candidate,
    vwap_reclaim_trend_ok,
)

UTC = timezone.utc
DAY1 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
DAY2 = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SLIPPAGE_RATE = 1.5 / 10_000.0
MARKER_TYPES = {"ENTRY", "STOP", "BREAKEVEN", "TARGET", "EXIT"}


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


def sweep_day2_path(
    sweep_minute: int,
    *,
    base: float = 100.0,
    sweep_low: float = 99.3,
    recover: float = 100.6,
    post_offset: float = 0.0,
) -> list[float]:
    """Day-2 SOXL path: flat base, a 5m lower-wick sweep + reclaim, then flat.

    ``sweep_minute`` is minutes after 09:30 ET and must be a multiple of 5; the
    sweep bucket closes at ``recover`` (signal bar close) and every later close
    stays at ``recover + post_offset``.
    """
    path = [base] * sweep_minute
    path += [sweep_low, sweep_low, base, recover, recover]  # sweep + reclaim bucket
    tail = 390 - len(path)
    path += [recover + post_offset] * tail
    assert len(path) == 390, len(path)
    return path


def build_prepared(soxl_day2: list[float], *, soxs_day2: list[float] | None = None):
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
    strategies: tuple[str, ...] = (SMC_SWEEP_5M,),
    window_start=None,
    window_end=ENTRY_WINDOW_END_ET,
    trend: bool = False,
    shield: bool = False,
    scale: bool = False,
    step: int = 90,
) -> V04Variant:
    return V04Variant(
        step=step,
        name="test-variant",
        enabled_strategies=tuple(strategies),
        allow_fractional_shares=True,
        entry_window_start_et=window_start,
        entry_window_end_et=window_end,
        vwap_trend_filter=trend,
        breakeven_shield=shield,
        scale_out=scale,
    )


def first_trade(result: v04.V04Result):
    assert result.trades, "expected at least one trade"
    return sorted(result.trades, key=lambda trade: trade.entry_time_utc)[0]


# --------------------------------------------------------------------------- #
# Constants & variant ladder
# --------------------------------------------------------------------------- #
def test_v04_constants_are_a_priori():
    assert (ENTRY_WINDOW_START_ET.hour, ENTRY_WINDOW_START_ET.minute) == (10, 0)
    assert (ENTRY_WINDOW_END_ET.hour, ENTRY_WINDOW_END_ET.minute) == (14, 0)
    assert OR30_BARS == 6
    assert SMC_LOOKBACK_BARS == 20
    assert SMC_RANGE_MULTIPLE == pytest.approx(1.2)
    assert SMC_STOP_ATR_BUFFER == pytest.approx(0.25)
    assert SCALE_TRIGGER_R == pytest.approx(1.2)
    assert SCALE_PORTION == pytest.approx(0.5)
    assert BREAKEVEN_OFFSET_R == pytest.approx(0.10)
    assert FRACTIONAL_PRECISION == 2


def test_ablation_ladder_is_cumulative_and_final_uses_scale_out():
    assert [variant.step for variant in ABLATION_STEPS] == [0, 1, 2, 3, 4]
    assert V04_FINAL_VARIANT is STEP4_VARIANT
    # Step 0 = V0.3 STEP4 semantics: 2 strategies, no 10:00 lower bound, V0.3
    # 14:00 cutoff, shield on, no scale-out.
    assert STEP0_VARIANT.enabled_strategies == (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
    assert STEP0_VARIANT.entry_window_start_et is None
    assert STEP0_VARIANT.entry_window_end_et == v03.ENTRY_CUTOFF_ET
    assert STEP0_VARIANT.breakeven_shield is True and STEP0_VARIANT.scale_out is False
    # Step 1 adds the 10:00 lower bound only.
    assert STEP1_VARIANT.entry_window_start_et == ENTRY_WINDOW_START_ET
    assert STEP1_VARIANT.vwap_trend_filter is False
    assert STEP1_VARIANT.enabled_strategies == (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
    # Step 2 adds the VWAP trend filter.
    assert STEP2_VARIANT.vwap_trend_filter is True
    # Step 3 adds SMC.
    assert STEP3_VARIANT.enabled_strategies == (
        VWAP_RECLAIM_5M,
        EMA_PULLBACK_5M,
        SMC_SWEEP_5M,
    )
    assert STEP3_VARIANT.scale_out is False
    # Step 4 replaces the V0.3 shield with the 1.2R scale-out.
    assert STEP4_VARIANT.scale_out is True
    assert STEP4_VARIANT.breakeven_shield is False
    assert STEP4_VARIANT.vwap_trend_filter is True


def test_variant_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="unknown"):
        V04Variant(step=1, name="bad", enabled_strategies=("NOPE",))
    with pytest.raises(ValueError, match="at least one"):
        V04Variant(step=1, name="bad", enabled_strategies=())
    with pytest.raises(ValueError, match="mutually exclusive"):
        V04Variant(
            step=1,
            name="bad",
            breakeven_shield=True,
            scale_out=True,
        )


# --------------------------------------------------------------------------- #
# Step 0 == V0.3 STEP4 (engine equivalence)
# --------------------------------------------------------------------------- #
def _trade_signature(result) -> list[tuple]:
    return [
        (
            trade.symbol,
            trade.strategy.value,
            trade.entry_time_utc.isoformat(),
            round(trade.entry_price_fill, 8),
            round(trade.stop_price, 8),
            trade.exit_reason,
            round(trade.pnl, 8),
        )
        for trade in sorted(result.trades, key=lambda trade: trade.entry_time_utc)
    ]


def test_step0_engine_matches_v03_step4_engine():
    data = build_prepared(sweep_day2_path(60))
    config = load_config()
    v03_result = v03.run_v03_backtest(data, config, variant=v03.STEP4_VARIANT)
    v04_result = run_v04_backtest(data, config, variant=STEP0_VARIANT)
    assert _trade_signature(v04_result) == _trade_signature(v03_result)
    assert v04_result.metrics.net_pnl_dollars == pytest.approx(
        v03_result.metrics.net_pnl_dollars, abs=1e-9
    )
    assert v04_result.metrics.profit_factor == pytest.approx(
        v03_result.metrics.profit_factor, rel=1e-9
    )


def test_engine_is_deterministic_across_runs():
    data = build_prepared(sweep_day2_path(60))
    config = load_config()
    first = run_v04_backtest(data, config, variant=V04_FINAL_VARIANT)
    second = run_v04_backtest(data, config, variant=V04_FINAL_VARIANT)
    assert _trade_signature(first) == _trade_signature(second)
    assert [m["type"] for m in first.markers] == [m["type"] for m in second.markers]


# --------------------------------------------------------------------------- #
# 10:00-14:00 entry window
# --------------------------------------------------------------------------- #
def test_window_lower_bound_blocks_pre_1000_signal():
    data = build_prepared(sweep_day2_path(15))  # 09:45 bucket, signal ends 09:50
    config = load_config()
    no_window = run_v04_backtest(
        data, config, variant=custom_variant(window_start=None)
    )
    assert no_window.total_trades == 1
    assert first_trade(no_window).entry_time_utc == DAY2 + timedelta(minutes=20)

    windowed = run_v04_backtest(
        data, config, variant=custom_variant(window_start=ENTRY_WINDOW_START_ET)
    )
    assert windowed.trades == []
    assert windowed.signals == 0
    assert windowed.window_skipped_evaluations > 0


def test_window_allows_1000_bucket_signal():
    data = build_prepared(sweep_day2_path(25))  # 09:55 bucket, signal ends 10:00
    result = run_v04_backtest(
        data,
        load_config(),
        variant=custom_variant(window_start=ENTRY_WINDOW_START_ET),
    )
    assert result.total_trades == 1
    # Signal bar ends exactly 10:00 ET -> allowed; fill at the 10:00 bucket open.
    assert first_trade(result).entry_time_utc == DAY2 + timedelta(minutes=30)


def test_window_upper_bound_blocks_1355_bucket():
    data = build_prepared(sweep_day2_path(265))  # 13:55 bucket, signal ends 14:00
    result = run_v04_backtest(
        data,
        load_config(),
        variant=custom_variant(window_start=ENTRY_WINDOW_START_ET),
    )
    assert result.trades == []
    assert result.signals == 0


def test_window_allows_1350_bucket_before_cutoff():
    data = build_prepared(sweep_day2_path(260))  # 13:50 bucket, signal ends 13:55
    result = run_v04_backtest(
        data,
        load_config(),
        variant=custom_variant(window_start=ENTRY_WINDOW_START_ET),
    )
    assert result.total_trades == 1
    assert first_trade(result).entry_time_utc == DAY2 + timedelta(minutes=265)


# --------------------------------------------------------------------------- #
# VWAP_RECLAIM trend filter predicate
# --------------------------------------------------------------------------- #
def test_vwap_trend_filter_soxl_requires_up_ema_and_session_range():
    ctx = V04EvalContext(
        session_range_so_far=3.0,
        atr=1.0,
        or30_high=None,
        or30_low=None,
        prev20_high=None,
        prev20_low=None,
        avg_range20=None,
        mirror_ema_fast=None,
        mirror_ema_slow=None,
        mirror_ok=True,
    )
    bullish = SimpleNamespace(symbol="SOXL", ema_fast=101.0, ema_slow=100.0)
    bearish = SimpleNamespace(symbol="SOXL", ema_fast=99.0, ema_slow=100.0)
    assert vwap_reclaim_trend_ok("SOXL", bullish, ctx) is True
    assert vwap_reclaim_trend_ok("SOXL", bearish, ctx) is False
    # Session range too small -> reject even with the right EMA slope.
    quiet = replace(ctx, session_range_so_far=0.5)
    assert vwap_reclaim_trend_ok("SOXL", bullish, quiet) is False
    # Unknown ATR -> reject (conservative).
    no_atr = replace(ctx, atr=None)
    assert vwap_reclaim_trend_ok("SOXL", bullish, no_atr) is False


def test_vwap_trend_filter_soxs_uses_mirror_semis_trend():
    ctx = V04EvalContext(
        session_range_so_far=3.0,
        atr=1.0,
        or30_high=None,
        or30_low=None,
        prev20_high=None,
        prev20_low=None,
        avg_range20=None,
        mirror_ema_fast=99.0,
        mirror_ema_slow=100.0,
        mirror_ok=True,
    )
    view = SimpleNamespace(symbol="SOXS", ema_fast=101.0, ema_slow=100.0)
    # SOXL (mirror) EMA9 < EMA21 -> semis downtrend -> long SOXS allowed.
    assert vwap_reclaim_trend_ok("SOXS", view, ctx) is True
    ctx_up = replace(ctx, mirror_ema_fast=101.0)
    assert vwap_reclaim_trend_ok("SOXS", view, ctx_up) is False


# --------------------------------------------------------------------------- #
# SMC_SWEEP_5M rule + deterministic stop
# --------------------------------------------------------------------------- #
def _smc_bar(**overrides) -> Bar:
    defaults = dict(
        open=100.0,
        high=100.28,
        low=99.45,
        close=100.20,
        volume=1000.0,
        vwap=100.0,
        trade_count=30,
    )
    defaults.update(overrides)
    start = DAY2 + timedelta(minutes=60)
    return Bar(
        symbol="SOXL",
        timeframe_minutes=5,
        start_utc=start,
        end_utc=start + timedelta(minutes=5),
        **defaults,
    )


def _smc_ctx(**overrides) -> V04EvalContext:
    defaults = dict(
        session_range_so_far=2.0,
        atr=0.50,
        or30_high=100.5,
        or30_low=99.90,
        prev20_high=101.0,
        prev20_low=99.95,
        avg_range20=0.20,
        mirror_ema_fast=None,
        mirror_ema_slow=None,
        mirror_ok=True,
    )
    defaults.update(overrides)
    return V04EvalContext(**defaults)


def _smc_view(bar: Bar | None = None) -> SimpleNamespace:
    data_bar = bar if bar is not None else _smc_bar()
    return SimpleNamespace(
        symbol="SOXL",
        bar=data_bar,
        now_utc=data_bar.end_utc,
        ema_fast=None,
        ema_slow=None,
    )


def test_smc_sweep_candidate_triggers_with_deterministic_stop():
    bar = _smc_bar()
    ctx = _smc_ctx()
    candidate = smc_sweep_5m_candidate(_smc_view(bar), ctx)
    assert candidate is not None
    assert candidate.strategy is StrategyName.ORB_15M  # SMC maps onto the frozen enum
    assert strategy_label_for_candidate(candidate) == SMC_SWEEP_5M
    assert candidate.entry_ref == pytest.approx(bar.close)
    # anchor = max(sweep low, OR30 low) = 99.90; stop = anchor - 0.25*ATR.
    assert candidate.stop_loss == pytest.approx(99.90 - SMC_STOP_ATR_BUFFER * 0.50)
    # 2.0R target.
    assert candidate.take_profit_price == pytest.approx(
        bar.close + 2.0 * (bar.close - candidate.stop_loss)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mirror_ok": False},
        {"avg_range20": None},
        {"atr": None},
        {"prev20_low": 99.0, "or30_low": 99.0},  # never pierces a support
        {"avg_range20": 1.0},  # bar range too small vs the 1.2x filter
    ],
)
def test_smc_rejections(overrides):
    view = _smc_view()
    ctx = _smc_ctx(**overrides)
    if "avg_range20" not in overrides:
        pass
    if overrides.get("prev20_low") == 99.0:
        # close must still be above support for a reclaim; make the bar close below.
        view = _smc_view(_smc_bar(open=99.1, close=99.0, high=99.2, low=98.6, vwap=99.0))
    assert smc_sweep_5m_candidate(view, ctx) is None


def test_smc_stop_clamped_to_025pct_minimum():
    # Structural stop is tighter than 0.25% -> clamp to the tightest allowed stop.
    bar = _smc_bar(open=100.18, close=100.20, low=100.10, high=100.28, vwap=100.18)
    ctx = _smc_ctx(atr=0.001, avg_range20=0.10, or30_low=100.15, prev20_low=100.15)
    candidate = smc_sweep_5m_candidate(_smc_view(bar), ctx)
    assert candidate is not None
    assert candidate.stop_loss == pytest.approx(100.20 * (1.0 - 0.0025))


# --------------------------------------------------------------------------- #
# +1.2R scale-out engine behaviour
# --------------------------------------------------------------------------- #
def scale_day2_path(*, mode: str) -> list[float]:
    """Day-2 SOXL path for scale-out tests (sweep at 10:30 ET bucket, minute 60)."""
    base = 100.0
    sweep_low = 99.3
    recover = 100.6
    path = [base] * 60
    path += [sweep_low, sweep_low, base, recover, recover]
    if mode == "stop_after_scale":
        path += [100.6, 101.6, 101.6, 101.6, 101.6]  # entry bucket + partial on m66
        path += [100.4, 100.3, 100.3, 100.3, 100.3]  # reversal through moved stop
        path += [100.3] * (390 - len(path))
    elif mode == "target_after_scale":
        path += [100.6, 101.6, 101.6, 101.6, 101.6]  # entry bucket, partial on m66
        path += [102.5, 102.5, 102.5, 102.5, 102.5]  # 2R target hit on the next bucket
        path += [102.5] * (390 - len(path))
    elif mode == "stop_without_scale":
        path += [100.6, 99.0, 99.0, 99.0, 99.0]  # immediate full stop, no partial
        path += [99.0] * (390 - len(path))
    else:  # pragma: no cover - defensive
        raise ValueError(mode)
    assert len(path) == 390, len(path)
    return path


def test_scale_out_partial_then_breakeven_stop():
    data = build_prepared(scale_day2_path(mode="stop_after_scale"))
    result = run_v04_backtest(
        data, load_config(), variant=custom_variant(scale=True, shield=False)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["scale_target_hit"] is True
    assert detail["exit_reason_detail"] == "SCALE_TARGET+STOP"
    legs = detail["partial_legs"]
    assert len(legs) == 2
    assert sum(leg["qty"] for leg in legs) == pytest.approx(trade.shares, rel=1e-9)
    entry_fill = trade.entry_price_fill
    risk_per_share = entry_fill - trade.stop_price
    # Partial leg at the 1.2R limit and the remainder stopped at entry + 0.10R.
    assert legs[0]["reference"] == pytest.approx(
        entry_fill + SCALE_TRIGGER_R * risk_per_share, rel=1e-6
    )
    assert legs[1]["reference"] == pytest.approx(
        entry_fill + BREAKEVEN_OFFSET_R * risk_per_share, rel=1e-6
    )
    assert legs[0]["qty"] == pytest.approx(trade.shares * SCALE_PORTION, abs=0.02)
    # Aggregate P&L equals the sum of the legs and is positive (0.65R geometry).
    leg_pnl = sum((leg["fill"] - entry_fill) * leg["qty"] for leg in legs) - trade.fees
    assert trade.pnl == pytest.approx(leg_pnl, abs=1e-6)
    assert trade.pnl > 0.0
    assert result.scale_target_trades >= 1
    marker_types = [marker["type"] for marker in result.markers]
    assert "ENTRY" in marker_types
    assert "TARGET" in marker_types
    assert "BREAKEVEN" in marker_types
    assert "STOP" in marker_types


def test_scale_out_second_leg_target():
    data = build_prepared(scale_day2_path(mode="target_after_scale"))
    result = run_v04_backtest(
        data, load_config(), variant=custom_variant(scale=True, shield=False)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["scale_target_hit"] is True
    assert detail["exit_reason_detail"] == "SCALE_TARGET+TARGET"
    assert trade.exit_reason == "TARGET"
    assert len(detail["partial_legs"]) == 2
    assert trade.pnl > 0.0


def test_scale_out_full_stop_without_partial():
    data = build_prepared(scale_day2_path(mode="stop_without_scale"))
    result = run_v04_backtest(
        data, load_config(), variant=custom_variant(scale=True, shield=False)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert detail["scale_target_hit"] is False
    assert detail["exit_reason_detail"] == "STOP"
    assert trade.exit_reason == "STOP"
    assert trade.pnl < 0.0
    assert trade.r_multiple == pytest.approx(-1.0, abs=0.1)


def test_scale_out_same_bar_collision_still_stop_first():
    soxl_bars = make_1m_bars("SOXL", DAY2, scale_day2_path(mode="stop_after_scale"))
    # First bar of the reversal bucket spans both target (2R) and current stop.
    soxl_bars[70] = Bar(
        symbol="SOXL",
        timeframe_minutes=1,
        start_utc=DAY2 + timedelta(minutes=70),
        end_utc=DAY2 + timedelta(minutes=71),
        open=101.6,
        high=103.2,
        low=100.0,
        close=101.6,
        volume=1000.0,
        vwap=101.6,
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
    result = run_v04_backtest(
        data, load_config(), variant=custom_variant(scale=True, shield=False)
    )
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "STOP"
    assert detail["exit_stop_kind"] == "SCALE_STOP"
    assert result.ambiguous_bars >= 1


# --------------------------------------------------------------------------- #
# Markers / Web API
# --------------------------------------------------------------------------- #
def test_markers_and_web_trade_dicts():
    data = build_prepared(scale_day2_path(mode="target_after_scale"))
    result = run_v04_backtest(
        data, load_config(), variant=custom_variant(scale=True, shield=False)
    )
    assert result.markers
    for marker in result.markers:
        assert marker["type"] in MARKER_TYPES
        assert marker["symbol"] in {"SOXL", "SOXS"}
        assert marker["strategy"] in {
            VWAP_RECLAIM_5M,
            EMA_PULLBACK_5M,
            SMC_SWEEP_5M,
        }
        assert marker["trade_id"].startswith("V04-")
        assert isinstance(marker["time_utc"], str) and "T" in marker["time_utc"]
        assert float(marker["price"]) > 0.0

    trade_dicts = result_to_trade_dicts(result)
    assert len(trade_dicts) == result.total_trades
    trade = trade_dicts[0]
    for key in (
        "trade_id",
        "symbol",
        "strategy",
        "direction",
        "entry_time_utc",
        "entry_price_fill",
        "stop_price",
        "target_price",
        "exit_time_utc",
        "exit_price_fill",
        "exit_reason",
        "pnl",
        "r_multiple",
        "partial_legs",
        "markers",
    ):
        assert key in trade
    assert trade["direction"] == "LONG"
    assert trade["markers"]
    assert {marker["trade_id"] for marker in trade["markers"]} == {trade["trade_id"]}


def test_evaluate_v04_session_returns_web_trade_dicts():
    data = build_prepared(scale_day2_path(mode="target_after_scale"))
    session_date = DAY2.date()
    trades = evaluate_v04_session(
        data,
        session_date,
        variant=custom_variant(scale=True, shield=False),
    )
    assert isinstance(trades, list) and trades
    assert all(isinstance(item, dict) for item in trades)
    assert trades[0]["kind"] == "trade"
    assert trades[0]["markers"]
    # A session without trades returns an empty list (no exception).
    empty = evaluate_v04_session(
        data,
        DAY1.date(),
        variant=custom_variant(scale=True, shield=False),
    )
    assert empty == []


def test_build_eval_context_exposes_or30_and_lookback():
    data = build_prepared(scale_day2_path(mode="target_after_scale"))
    session = data.sessions[-1]
    bucket_start = DAY2 + timedelta(minutes=60)
    flat = v04._flat_buckets_by_symbol(data)
    ctx = build_eval_context(data, session, "SOXL", bucket_start, flat)
    assert ctx.or30_low is not None and ctx.or30_low < 100.0
    assert ctx.prev20_low is not None
    assert ctx.avg_range20 is not None and ctx.avg_range20 > 0.0
    assert ctx.atr is not None and ctx.atr > 0.0
    assert ctx.mirror_ok is True  # SOXS is below its VWAP


# --------------------------------------------------------------------------- #
# Real raw-data anchor (explicitly required by the task)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def raw_prepared_data():
    from intraday_scalper.tools.run_v03_raw_crosscheck import load_raw_bars

    return prepare_v02_data(load_raw_bars())


def test_raw_data_step0_anchor_and_29_fold_wf(raw_prepared_data):
    data = raw_prepared_data
    config = load_config()
    assert len(data.sessions) == 714
    splits = walk_forward_splits(list(data.dates), 120, 20, 20)
    assert len(splits) == 29

    # V0.3 raw_full Step4 anchor: 970 trades / -$214.76 / PF 0.9772.
    step0 = run_v04_backtest(data, config, variant=STEP0_VARIANT)
    assert step0.metrics.total_trades == 970
    assert step0.metrics.net_pnl_dollars == pytest.approx(-214.76, abs=0.01)
    assert step0.metrics.profit_factor == pytest.approx(0.9772, abs=1e-3)

    # Walk-forward anchor: 814 OOS trades / -$234.92 / PF 0.9704 / 16 folds.
    wf = v04.run_v04_walkforward(data, config, variant=STEP0_VARIANT)
    assert wf.summary["folds"] == 29
    assert wf.summary["total_test_trades"] == 814
    assert wf.summary["test_net_pnl_dollars"] == pytest.approx(-234.92, abs=0.01)
    assert wf.summary["positive_folds"] == 16

    # Final V0.4 step: scale-out and EMA9 trail actually fire on real data.
    final = run_v04_backtest(data, config, variant=V04_FINAL_VARIANT)
    assert final.total_trades > 0
    assert final.scale_target_trades > 0
    assert final.trail_exit_trades > 0
    assert final.markers
    assert any(trade.exit_reason == "TRAIL_EMA9" for trade in final.trades)


# --------------------------------------------------------------------------- #
# V0.4b Step 3b filters (task-13)
# --------------------------------------------------------------------------- #
def test_ema_exhaustion_filter_boundary():
    """>= +8.0% above the session low blocks the EMA_PULLBACK signal."""
    assert EMA_EXHAUSTION_GAIN_LIMIT == pytest.approx(0.08)
    # exactly +8.0% -> blocked (>= threshold)
    assert ema_pullback_exhaustion_blocked(108.0, 100.0) is True
    assert ema_pullback_exhaustion_blocked(100.0 * 1.08, 100.0) is True
    # just below +8.0% -> allowed
    assert ema_pullback_exhaustion_blocked(107.99, 100.0) is False
    # flat / negative gain -> allowed
    assert ema_pullback_exhaustion_blocked(100.0, 100.0) is False
    # unavailable session low cannot be evaluated -> not blocked
    assert ema_pullback_exhaustion_blocked(120.0, None) is False
    assert ema_pullback_exhaustion_blocked(120.0, 0.0) is False


def test_smc_trend_confirm_underwater_rejected():
    """close below both EMA21 and session VWAP -> underwater knife-catch rejected."""
    assert smc_trend_confirm_ok(98.0, 99.0, 99.2) is False
    assert smc_trend_confirm_ok(98.0, 99.2, 99.0) is False
    # both references unavailable -> conservative rejection
    assert smc_trend_confirm_ok(98.0, None, None) is False


def test_smc_trend_confirm_allows_ema21_or_vwap():
    """>= EMA21 alone, or >= session VWAP alone, is enough (OR rule)."""
    assert smc_trend_confirm_ok(98.0, 97.5, 99.2) is True  # reclaims EMA21 only
    assert smc_trend_confirm_ok(98.0, 99.0, 97.9) is True  # reclaims VWAP only
    assert smc_trend_confirm_ok(99.0, 99.0, 99.0) is True  # equality counts


def test_step3b_variant_ladder_and_production_default():
    flags = STEP3B_VARIANT.flags()
    assert STEP3B_VARIANT.step == 5
    assert flags["exhaustion_filter"] is True
    assert flags["smc_trend_confirm"] is True
    assert flags["scale_out_1_2r"] is False
    assert flags["breakeven_shield"] is True
    assert STEP3B_VARIANT.enabled_strategies == (
        VWAP_RECLAIM_5M,
        EMA_PULLBACK_5M,
        SMC_SWEEP_5M,
    )
    # PRODUCTION_VARIANT currently points at Step 3b (Lead may switch to Step1/Step3).
    assert PRODUCTION_VARIANT is STEP3B_VARIANT
    assert [variant.step for variant in COMPARE_VARIANTS] == [1, 3, 5]
    # Step 4 (scale-out) is retained but deprecated / not the production default.
    assert V04_FINAL_VARIANT is STEP4_VARIANT
    assert STEP4_VARIANT.scale_out is True
    assert STEP4_VARIANT is not PRODUCTION_VARIANT


def test_step3b_matches_step3_when_filters_do_not_trigger():
    """A right-side SMC sweep (not exhausted, close >= EMA21/VWAP) is identical."""
    data = build_prepared(sweep_day2_path(60))
    config = load_config()
    step3 = run_v04_backtest(data, config, variant=STEP3_VARIANT)
    step3b = run_v04_backtest(data, config, variant=STEP3B_VARIANT)
    assert step3.total_trades == step3b.total_trades > 0
    assert _trade_signature(step3b) == _trade_signature(step3)
    assert step3b.metrics.net_pnl_dollars == pytest.approx(
        step3.metrics.net_pnl_dollars, abs=1e-9
    )


def _flat_engine(variant):
    data = build_prepared([100.0] * 390)
    return v04._V04Replay(data, load_config(), variant)


def test_step3b_rule_blocks_underwater_smc_trade():
    """SMC candidate that Step 3 takes is rejected when the close is below both refs."""
    engine3 = _flat_engine(STEP3_VARIANT)
    engine3b = _flat_engine(STEP3B_VARIANT)
    bar = _smc_bar()
    view = SimpleNamespace(
        symbol="SOXL",
        bar=bar,
        now_utc=bar.end_utc,
        ema_slow=101.0,  # above close (100.20)
        session_vwap=101.2,  # above close
    )
    ctx = _smc_ctx()
    assert engine3._smc_rule(view, ctx) is not None
    assert engine3b._smc_rule(view, ctx) is None


def test_step3b_rule_allows_smc_above_ema21_or_vwap():
    engine3b = _flat_engine(STEP3B_VARIANT)
    bar = _smc_bar()
    ctx = _smc_ctx()
    above_ema = SimpleNamespace(
        symbol="SOXL", bar=bar, now_utc=bar.end_utc, ema_slow=100.0, session_vwap=101.2
    )
    assert engine3b._smc_rule(above_ema, ctx) is not None
    above_vwap = SimpleNamespace(
        symbol="SOXL", bar=bar, now_utc=bar.end_utc, ema_slow=101.2, session_vwap=100.0
    )
    assert engine3b._smc_rule(above_vwap, ctx) is not None


def _ema_pullback_view():
    """A StrategyView that triggers the unchanged V0.2 EMA_PULLBACK_5M rule."""
    bar = Bar(
        symbol="SOXL",
        timeframe_minutes=5,
        start_utc=DAY2 + timedelta(minutes=60),
        end_utc=DAY2 + timedelta(minutes=65),
        open=100.2,
        high=101.2,
        low=100.1,
        close=101.0,
        volume=1000.0,
        vwap=100.6,
        trade_count=30,
    )
    prev = Bar(
        symbol="SOXL",
        timeframe_minutes=5,
        start_utc=DAY2 + timedelta(minutes=55),
        end_utc=DAY2 + timedelta(minutes=60),
        open=100.0,
        high=100.3,
        low=99.9,
        close=100.1,
        volume=1000.0,
        vwap=100.05,
        trade_count=25,
    )
    return v02.StrategyView(
        symbol="SOXL",
        now_utc=bar.end_utc,
        bar=bar,
        prev_bar=prev,
        prior_bars=(prev, prev, bar),
        prior_vwaps=(99.0, 99.5, 100.0),
        session_open_utc=bar.start_utc - timedelta(minutes=15),
        orb_high=101.5,
        ema_fast=100.9,
        ema_slow=100.5,
        ema_fast_prev=100.05,
        atr=0.5,
        atr_prev=0.5,
        range_expansion=1.5,
        rel_activity=1.2,
        session_vwap=100.3,
        prev_session_vwap=100.2,
        mirror_ok=True,
    )


def _ema_rule_ctx(session_low: float) -> V04EvalContext:
    return V04EvalContext(
        session_range_so_far=2.0,
        atr=0.5,
        or30_high=101.0,
        or30_low=99.8,
        prev20_high=101.0,
        prev20_low=99.5,
        avg_range20=0.4,
        mirror_ema_fast=None,
        mirror_ema_slow=None,
        mirror_ok=True,
        session_low_so_far=session_low,
    )


def test_step3b_rule_blocks_exhausted_ema_pullback():
    engine3 = _flat_engine(STEP3_VARIANT)
    engine3b = _flat_engine(STEP3B_VARIANT)
    view = _ema_pullback_view()
    # close 101.0 vs session low 90.0 -> +12.2% >= 8% -> blocked by Step 3b only.
    exhausted_ctx = _ema_rule_ctx(90.0)
    assert engine3._ema_pullback_rule(view, exhausted_ctx) is not None
    assert engine3b._ema_pullback_rule(view, exhausted_ctx) is None
    # close 101.0 vs session low 95.0 -> +6.3% < 8% -> allowed by both.
    calm_ctx = _ema_rule_ctx(95.0)
    assert engine3b._ema_pullback_rule(view, calm_ctx) is not None


def _synthetic_friday_bars():
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, sweep_day2_path(60))
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars("SOXS", DAY2, soxs_falling_path())
    return sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))


def test_v04b_friday_recap_three_variant_structure():
    from intraday_scalper.tools import run_v04b_compare

    recap = run_v04b_compare.build_friday_recap(_synthetic_friday_bars())
    assert recap["session_date"] == "2026-11-06"
    assert set(recap["versions"].keys()) == {"Step1", "Step3", "Step3b"}
    assert set(recap["market"].keys()) == {"SOXL", "SOXS"}
    for label, payload in recap["versions"].items():
        summary = payload["summary"]
        for key in ("total_trades", "net_pnl_dollars", "profit_factor"):
            assert key in summary
        for trade in payload["trades"]:
            for key in (
                "trade_id",
                "symbol",
                "strategy",
                "entry_time_utc",
                "stop_price",
                "target_price",
                "exit_reason",
                "pnl",
                "r_multiple",
                "markers",
            ):
                assert key in trade
        assert isinstance(payload["markers"], list)
    # Step3 and Step3b trade the synthetic right-side SMC sweep; Step1 posts a
    # zero-trade reason (no EMA/VWAP signal on this sweep-only session).
    assert recap["versions"]["Step3b"]["summary"]["total_trades"] > 0
    assert recap["versions"]["Step1"].get("zero_trade_reason")
    assert recap["filters_effect"]["step3_trade_count"] == recap["filters_effect"]["step3b_trade_count"]


def test_v04b_markers_payload_groups_by_variant():
    from intraday_scalper.tools import run_v04b_compare

    recap = run_v04b_compare.build_friday_recap(_synthetic_friday_bars())
    payload = run_v04b_compare.build_markers_payload(recap)
    assert payload["session_date"] == "2026-11-06"
    assert set(payload["variants"]) == {"Step1", "Step3", "Step3b"}
    assert set(payload["markers_by_variant"].keys()) == {"Step1", "Step3", "Step3b"}
    assert set(payload["trades_by_variant"].keys()) == {"Step1", "Step3", "Step3b"}
    grouped = payload["markers_by_variant"]
    assert grouped["Step3b"], "Step3b should have emitters on the synthetic session"
    for label, markers in grouped.items():
        for marker in markers:
            assert marker["variant"] == label
            assert marker["type"] in MARKER_TYPES
            for key in ("time_utc", "price", "symbol", "strategy", "trade_id"):
                assert key in marker
    flat = payload["markers"]
    assert flat and all("variant" in marker for marker in flat)
    assert [marker["t"] for marker in flat] == sorted(marker["t"] for marker in flat)


# --------------------------------------------------------------------------- #
# V0.4c Step 3c A/B: SMC strict AND confirmation (task-14)
# --------------------------------------------------------------------------- #
def test_smc_and_confirmation_boundaries():
    """>= EMA21 AND >= session VWAP are both required (strict AND)."""
    # both references reclaimed -> allow
    assert smc_trend_confirm_require_both(100.0, 99.0, 98.0) is True
    assert smc_trend_confirm_require_both(100.0, 100.0, 100.0) is True  # equality
    # close < EMA21 (even when >= VWAP) -> reject
    assert smc_trend_confirm_require_both(100.0, 101.0, 98.0) is False
    assert smc_trend_confirm_require_both(100.0, 101.0, 99.0) is False
    # close >= EMA21 but < VWAP -> reject (the Step3b-OR-passes / AND-fails case)
    assert smc_trend_confirm_require_both(100.0, 99.0, 101.0) is False
    # missing reference -> conservative reject
    assert smc_trend_confirm_require_both(100.0, None, 98.0) is False
    assert smc_trend_confirm_require_both(100.0, 99.0, None) is False


def test_step3c_variant_ladder_and_production_default():
    nocap = STEP3C_NOCAP_VARIANT.flags()
    withcap = STEP3C_WITHCAP_VARIANT.flags()
    assert nocap["step"] == 6 and withcap["step"] == 7
    assert nocap["smc_require_both"] is True
    assert withcap["smc_require_both"] is True
    assert nocap["smc_trend_confirm"] is True and withcap["smc_trend_confirm"] is True
    assert nocap["exhaustion_filter"] is False  # NoCap keeps no 8% cap
    assert withcap["exhaustion_filter"] is True  # WithCap keeps the 8% cap
    for flags in (nocap, withcap):
        assert flags["breakeven_shield"] is True
        assert flags["scale_out_1_2r"] is False
        assert flags["strategies"] == [VWAP_RECLAIM_5M, EMA_PULLBACK_5M, SMC_SWEEP_5M]
    assert [variant.step for variant in AB_COMPARE_VARIANTS] == [6, 7]
    # PRODUCTION_VARIANT deliberately stays at Step 3b until the Lead decides.
    assert PRODUCTION_VARIANT is STEP3B_VARIANT


def _smc_rule_view(ema21, vwap):
    bar = _smc_bar()
    return SimpleNamespace(
        symbol="SOXL", bar=bar, now_utc=bar.end_utc, ema_slow=ema21, session_vwap=vwap
    )


def test_step3c_and_rule_vs_step3b_or_rule():
    """Step3b OR passes when close >= EMA21 only; Step3c AND rejects it."""
    ctx = _smc_ctx()
    engine_or = _flat_engine(STEP3B_VARIANT)
    engine_and_nocap = _flat_engine(STEP3C_NOCAP_VARIANT)
    engine_and_withcap = _flat_engine(STEP3C_WITHCAP_VARIANT)
    view = _smc_rule_view(ema21=100.0, vwap=101.0)  # close 100.20 >= EMA21, < VWAP
    assert engine_or._smc_rule(view, ctx) is not None
    assert engine_and_nocap._smc_rule(view, ctx) is None
    assert engine_and_withcap._smc_rule(view, ctx) is None
    # The symmetric case: close >= VWAP only (EMA21 above close) -> OR passes, AND rejects.
    view_vwap_only = _smc_rule_view(ema21=101.0, vwap=100.0)
    assert engine_or._smc_rule(view_vwap_only, ctx) is not None
    assert engine_and_nocap._smc_rule(view_vwap_only, ctx) is None


def test_step3c_and_rule_allows_when_both_reclaimed():
    ctx = _smc_ctx()
    engines = (
        _flat_engine(STEP3B_VARIANT),
        _flat_engine(STEP3C_NOCAP_VARIANT),
        _flat_engine(STEP3C_WITHCAP_VARIANT),
    )
    view = _smc_rule_view(ema21=100.0, vwap=100.0)  # both below close 100.20
    assert all(engine._smc_rule(view, ctx) is not None for engine in engines)


def test_nocap_restores_exhausted_ema_signal():
    """NoCap (no 8% cap) takes the EMA_PULLBACK signal WithCap/Step3b rejects."""
    view = _ema_pullback_view()
    exhausted_ctx = _ema_rule_ctx(90.0)  # close 101 vs session low 90 -> +12.2%
    nocap = _flat_engine(STEP3C_NOCAP_VARIANT)
    withcap = _flat_engine(STEP3C_WITHCAP_VARIANT)
    step3b = _flat_engine(STEP3B_VARIANT)
    assert nocap._ema_pullback_rule(view, exhausted_ctx) is not None
    assert withcap._ema_pullback_rule(view, exhausted_ctx) is None
    assert step3b._ema_pullback_rule(view, exhausted_ctx) is None


def test_withcap_blocks_exhausted_ema_signal_but_not_below_threshold():
    view = _ema_pullback_view()
    withcap = _flat_engine(STEP3C_WITHCAP_VARIANT)
    assert withcap._ema_pullback_rule(view, _ema_rule_ctx(90.0)) is None
    # +6.3% < 8% -> still allowed
    assert withcap._ema_pullback_rule(view, _ema_rule_ctx(95.0)) is not None


def test_step3c_filters_do_not_change_exits_when_not_triggered():
    """When the AND rule passes and no chase exists, trades are identical."""
    data = build_prepared(sweep_day2_path(60))
    config = load_config()
    step3b = run_v04_backtest(data, config, variant=STEP3B_VARIANT)
    nocap = run_v04_backtest(data, config, variant=STEP3C_NOCAP_VARIANT)
    withcap = run_v04_backtest(data, config, variant=STEP3C_WITHCAP_VARIANT)
    assert step3b.total_trades == nocap.total_trades == withcap.total_trades > 0
    assert _trade_signature(nocap) == _trade_signature(step3b)
    assert _trade_signature(withcap) == _trade_signature(step3b)
    # Existing-position exits are untouched: same exit reason/time/price.
    assert [
        (t.exit_reason, t.exit_time_utc, t.exit_price_fill) for t in nocap.trades
    ] == [(t.exit_reason, t.exit_time_utc, t.exit_price_fill) for t in step3b.trades]


def test_evaluate_v04_session_supports_step3c_variant():
    data = build_prepared(sweep_day2_path(60))
    trades = evaluate_v04_session(
        data, DAY2.date(), variant=STEP3C_NOCAP_VARIANT
    )
    assert isinstance(trades, list) and trades
    trade = trades[0]
    assert trade["symbol"] in {"SOXL", "SOXS"}
    assert trade["markers"]
    assert all(marker["trade_id"] == trade["trade_id"] for marker in trade["markers"])


def test_v04c_friday_recap_and_markers_structure():
    from intraday_scalper.tools import run_v04c_ab

    recap = run_v04c_ab.build_friday_recap(_synthetic_friday_bars())
    assert set(recap["versions"].keys()) == {"Step3b", "NoCap", "WithCap"}
    for label, payload in recap["versions"].items():
        for key in ("total_trades", "net_pnl_dollars", "profit_factor"):
            assert key in payload["summary"]
        for trade in payload["trades"]:
            for key in (
                "trade_id",
                "symbol",
                "strategy",
                "entry_time_utc",
                "stop_price",
                "target_price",
                "exit_reason",
                "pnl",
                "r_multiple",
                "markers",
            ):
                assert key in trade
    assert set(recap["traps"].keys()) == {
        "11:25 SOXL SMC 接刀",
        "12:40 SOXS EMA 山顶追高",
    }
    effect = recap["filter_effect_vs_step3b"]
    assert "blocked_by_and_confirmation" in effect
    assert "blocked_by_8pct_cap" in effect

    markers_payload = run_v04c_ab.build_markers_payload(recap)
    assert markers_payload["variants"] == ["NoCap", "WithCap"]
    assert set(markers_payload["markers_by_variant"].keys()) == {"NoCap", "WithCap"}
    for label, markers in markers_payload["markers_by_variant"].items():
        for marker in markers:
            assert marker["variant"] == label
            assert marker["type"] in MARKER_TYPES
            for key in ("time_utc", "price", "symbol", "strategy", "trade_id", "t"):
                assert key in marker
    flat = markers_payload["markers"]
    assert [marker["t"] for marker in flat] == sorted(marker["t"] for marker in flat)
    assert set(markers_payload["trades_by_variant"].keys()) == {
        "NoCap",
        "WithCap",
        "Step3b",
    }
