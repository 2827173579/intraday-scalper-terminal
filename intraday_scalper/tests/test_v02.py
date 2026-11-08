"""Offline, deterministic tests for the V0.2 research backtest.

Everything here is synthetic: two weekday sessions built from explicit 1m close
paths (2026-11-05 Thursday / 2026-11-06 Friday, 09:30 ET == 14:30 UTC in US
standard time).  No network, no real data, no wall-clock dependence.

Coverage (V0.2 task §11): 5m resampling geometry/OHLCV/VWAP, empty-bucket
handling, unfinished-bucket behaviour, mirror-regime accept/reject in both
directions, each of the three strategies accepting and rejecting (including the
range-expansion / relative-activity filters), next-5m-open fills, same-1m-bar
TP/SL collision -> stop, EOD flattening, pending cancellation on a missing
bucket, risk-manager mutual exclusion / daily circuit-breaker smoke, the hard
``begin_session`` requirement, and walk-forward split reuse / leakage checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import (
    Bar,
    Mode,
    SignalCandidate,
    StrategyName,
    load_config,
    session_open_utc,
)
from intraday_scalper.research.v02 import (
    ATR_PERIOD,
    EMA_PULLBACK_5M,
    ORB_15M_5M,
    VWAP_RECLAIM_5M,
    StrategyView,
    build_strategy_view,
    ema_pullback_5m_candidate,
    evaluate_v02,
    mirror_relation_ok,
    orb_15m_5m_candidate,
    prepare_v02_data,
    resample_session_5m,
    run_v02_backtest,
    run_v02_walkforward,
    vwap_reclaim_5m_candidate,
)
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.risk.sizing import size_position

UTC = timezone.utc
#: Thursday 2026-11-05 09:30 ET == 14:30 UTC (US standard time).
DAY1 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
#: Friday 2026-11-06 09:30 ET == 14:30 UTC.
DAY2 = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SLIPPAGE_RATE = 1.5 / 10_000.0  # config.risk.est_slippage_bps default


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


def make_5m_bar(
    *,
    symbol: str = "SOXL",
    start: datetime | None = None,
    open_: float = 100.0,
    high: float = 100.4,
    low: float = 99.8,
    close: float = 100.2,
    volume: float = 5000.0,
    vwap: float = 100.1,
    trade_count: int = 30,
) -> Bar:
    start_utc = start if start is not None else DAY2
    return Bar(
        symbol=symbol,
        timeframe_minutes=5,
        start_utc=start_utc,
        end_utc=start_utc + timedelta(minutes=5),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        trade_count=trade_count,
    )


def make_view(**overrides) -> StrategyView:
    """A strategy view whose default state satisfies every ORB/ VWAP filter."""
    bar = make_5m_bar(close=101.0, open_=100.2, high=101.2, low=100.1, vwap=100.6)
    prev_bar = make_5m_bar(
        start=DAY2 - timedelta(minutes=5),
        close=100.1,
        open_=100.0,
        high=100.3,
        low=99.9,
        vwap=100.05,
    )
    defaults = dict(
        symbol="SOXL",
        now_utc=bar.end_utc,
        bar=bar,
        prev_bar=prev_bar,
        prior_bars=(prev_bar, prev_bar, bar),
        prior_vwaps=(99.0, 99.5, 100.0),
        session_open_utc=bar.start_utc - timedelta(minutes=15),
        orb_high=100.8,
        ema_fast=100.9,
        ema_slow=100.5,
        ema_fast_prev=100.7,
        atr=0.5,
        atr_prev=0.5,
        range_expansion=1.5,
        rel_activity=1.2,
        session_vwap=100.3,
        prev_session_vwap=100.2,
        mirror_ok=True,
    )
    defaults.update(overrides)
    return StrategyView(**defaults)


def soxl_day2_path(post: list[float]) -> list[float]:
    """Day-2 SOXL path: flat open, a 09:45 ET breakout bar, then *post*."""
    pre = [100.0] * 15 + [100.1, 100.3, 100.5, 100.7, 100.9]
    path = pre + list(post)
    assert len(path) == 390, len(path)
    return path


def soxs_falling_path() -> list[float]:
    """SOXS drifts down all day -> SOXL long candidates pass the mirror gate."""
    return [100.0 - 0.02 * index for index in range(390)]


def build_prepared(soxl_day2: list[float], *, soxs_day2: list[float] | None = None):
    """Two-session, two-symbol prepared V0.2 dataset."""
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, soxl_day2)
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars("SOXS", DAY2, soxs_day2 if soxs_day2 is not None else soxs_falling_path())
    bars = sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    return prepare_v02_data(bars)


def first_trade(result):
    assert result.trades, "expected at least one trade"
    return sorted(result.trades, key=lambda trade: trade.entry_time_utc)[0]


# --------------------------------------------------------------------------- #
# 5m resampling
# --------------------------------------------------------------------------- #
def test_resample_5m_ohlcv_vwap_and_trade_count():
    """Aggregation must be exact: first open, extremes, last close, Σv, Σvwap*v."""
    session_open = DAY2
    rows = [
        # open, high, low, close, volume, vwap, trade_count
        (10.00, 10.50, 9.80, 10.20, 100.0, 10.10, 1),
        (10.20, 10.40, 10.00, 10.10, 200.0, 10.20, 2),
        (10.10, 10.30, 9.90, 10.30, 300.0, 10.05, 3),
        (10.30, 10.60, 10.20, 10.40, 400.0, 10.40, 4),
        (10.40, 10.50, 10.10, 10.20, 500.0, 10.30, 5),
    ]
    bars = []
    for index, (open_, high, low, close, volume, vwap, trade_count) in enumerate(rows):
        start = session_open + timedelta(minutes=index)
        bars.append(
            Bar(
                symbol="SOXL",
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
    result = resample_session_5m(bars, session_open)
    assert len(result) == 1
    bar = result[0]
    assert bar.timeframe_minutes == 5
    assert bar.start_utc == session_open
    assert bar.end_utc == session_open + timedelta(minutes=5)
    assert bar.open == pytest.approx(10.00)
    assert bar.high == pytest.approx(10.60)
    assert bar.low == pytest.approx(9.80)
    assert bar.close == pytest.approx(10.20)
    assert bar.volume == pytest.approx(1500.0)
    expected_vwap = (10.10 * 100 + 10.20 * 200 + 10.05 * 300 + 10.40 * 400 + 10.30 * 500) / 1500.0
    assert bar.vwap == pytest.approx(expected_vwap)
    assert bar.trade_count == 15
    assert bar.is_complete is True


def test_resample_5m_anchors_at_session_open_and_skips_empty_buckets():
    session_open = DAY2
    minutes = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]  # 09:30 bucket + 09:40 bucket
    bars = []
    for minute in minutes:
        start = session_open + timedelta(minutes=minute)
        price = 100.0 + minute * 0.01
        bars.append(
            Bar(
                symbol="SOXL",
                timeframe_minutes=1,
                start_utc=start,
                end_utc=start + timedelta(minutes=1),
                open=price,
                high=price + 0.1,
                low=price - 0.1,
                close=price,
                volume=100.0,
                vwap=price,
                trade_count=5,
            )
        )
    result = resample_session_5m(bars, session_open)
    assert [bar.start_utc for bar in result] == [
        session_open,
        session_open + timedelta(minutes=10),
    ]
    # The empty 09:35 bucket was not fabricated into a bar.
    assert all(bar.start_utc != session_open + timedelta(minutes=5) for bar in result)


def test_resample_5m_drops_unfinished_final_bucket():
    """A bucket whose 5m window is not over by the session close is not emitted."""
    session_open = DAY2
    bars = make_1m_bars("SOXL", session_open, [100.0 + 0.01 * index for index in range(7)])
    close_utc = session_open + timedelta(minutes=7)  # 09:37: bucket 09:35 not finished
    result = resample_session_5m(bars, session_open, close_utc)
    assert [bar.start_utc for bar in result] == [session_open]


def test_resample_5m_rejects_unsorted_input():
    bars = make_1m_bars("SOXL", DAY2, [100.0, 100.1, 100.2])
    with pytest.raises(ValueError, match="sorted"):
        resample_session_5m([bars[1], bars[0], bars[2]], DAY2)


# --------------------------------------------------------------------------- #
# Mirror-regime filter
# --------------------------------------------------------------------------- #
def test_mirror_filter_accepts_both_directions():
    """Both legs below their own VWAP: the mirror relation holds for both longs.

    Corrected mirror logic (Lead-confirmed): long SOXL needs SOXS < SOXS VWAP;
    long SOXS needs SOXL < SOXL VWAP — i.e. the other ETF must be weak.
    """
    falling_soxl = [100.0 - 0.02 * index for index in range(390)]
    falling_soxs = [100.0 - 0.02 * index for index in range(390)]
    data = build_prepared(falling_soxl, soxs_day2=falling_soxs)
    session = data.sessions[-1]  # day 2
    bucket_start = DAY2 + timedelta(minutes=15)  # 09:45 ET bucket
    assert mirror_relation_ok(data, session, "SOXL", bucket_start) is True
    assert mirror_relation_ok(data, session, "SOXS", bucket_start) is True


def test_mirror_filter_rejects_both_directions():
    """Both legs above their own VWAP: neither mirror relation holds."""
    rising_soxl = [100.0 + 0.02 * index for index in range(390)]
    rising_soxs = [100.0 + 0.02 * index for index in range(390)]
    data = build_prepared(rising_soxl, soxs_day2=rising_soxs)
    session = data.sessions[-1]
    bucket_start = DAY2 + timedelta(minutes=15)
    assert mirror_relation_ok(data, session, "SOXL", bucket_start) is False
    assert mirror_relation_ok(data, session, "SOXS", bucket_start) is False


def test_mirror_filter_soxs_long_requires_soxl_below_vwap():
    """A rising SOXL must block long-SOXS candidates (SOXL is above its VWAP)."""
    data = build_prepared(soxl_day2_path([100.9] * 370), soxs_day2=soxs_falling_path())
    session = data.sessions[-1]
    bucket_start = DAY2 + timedelta(minutes=15)
    # SOXL is above its session VWAP -> the corrected SOXS mirror gate fails.
    assert mirror_relation_ok(data, session, "SOXS", bucket_start) is False
    # SOXS is below its VWAP -> the SOXL mirror gate passes.
    assert mirror_relation_ok(data, session, "SOXL", bucket_start) is True


def test_build_view_wires_features_and_mirror():
    data = build_prepared(soxl_day2_path([100.9] * 370), soxs_day2=soxs_falling_path())
    session = data.sessions[-1]
    view = build_strategy_view(data, session, "SOXL", DAY2 + timedelta(minutes=15))
    assert view is not None
    assert view.mirror_ok is True
    assert view.orb_high == pytest.approx(100.08)
    assert view.atr is not None and view.atr > 0
    assert view.rel_activity == pytest.approx(1.0)
    assert view.range_expansion is not None and view.range_expansion > 1.2


# --------------------------------------------------------------------------- #
# Strategy rules: accept / reject
# --------------------------------------------------------------------------- #
def test_orb_triggers_with_stop_and_target_geometry():
    view = make_view()
    candidate = orb_15m_5m_candidate(view)
    assert candidate is not None
    assert candidate.strategy is StrategyName.ORB_15M
    assert candidate.entry_ref == pytest.approx(101.0)
    # structural stop = min(OR high 100.8, bar low 100.1) - 1.0*ATR(0.5)
    assert candidate.stop_loss == pytest.approx(99.6)
    assert candidate.take_profit_price == pytest.approx(101.0 + 2.0 * (101.0 - 99.6))
    assert candidate.target_r == pytest.approx(2.0)
    assert candidate.confirmation_bar_start_utc == view.bar.start_utc
    assert candidate.created_utc == view.now_utc


@pytest.mark.parametrize(
    "overrides",
    [
        {"mirror_ok": False},
        {"orb_high": 101.5},  # no breakout
        {"bar": make_5m_bar(close=100.6, open_=101.0, high=101.2, low=100.1, vwap=100.6)},  # bearish
        {"range_expansion": 1.1, "rel_activity": 1.0},  # expansion filter fails
        {"rel_activity": 0.9},  # activity filter fails (range passes)
    ],
)
def test_orb_rejections(overrides):
    view = make_view(**overrides)
    assert orb_15m_5m_candidate(view) is None


def test_vwap_reclaim_triggers():
    view = make_view(prev_session_vwap=100.5, session_vwap=100.3)
    candidate = vwap_reclaim_5m_candidate(view)
    assert candidate is not None
    assert candidate.strategy is StrategyName.VWAP_RECLAIM
    # structural stop = min(bar low 100.1, session VWAP 100.3) - 1.0*ATR(0.5)
    assert candidate.stop_loss == pytest.approx(99.6)
    assert candidate.entry_ref == pytest.approx(101.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prev_session_vwap": 100.0},  # prev close 100.1 was NOT below VWAP
        {"session_vwap": 101.0},  # no reclaim buffer (101 <= 101*1.0005)
        {"bar": make_5m_bar(close=100.9, open_=101.0, high=101.2, low=100.1, vwap=100.9)},  # bearish
        {"ema_fast": 100.4},  # ema9 must be above ema21
        {"rel_activity": 1.0, "range_expansion": 1.1},  # both activity filters fail
        {"mirror_ok": False},
    ],
)
def test_vwap_reclaim_rejections(overrides):
    view = make_view(**overrides)
    assert vwap_reclaim_5m_candidate(view) is None


def test_vwap_reclaim_activity_or_filter_allows_range_expansion():
    view = make_view(
        rel_activity=1.0,
        range_expansion=1.3,
        prev_session_vwap=100.5,
        session_vwap=100.3,
    )
    assert vwap_reclaim_5m_candidate(view) is not None


def test_ema_pullback_triggers():
    view = make_view(
        ema_fast_prev=100.05,
        prev_session_vwap=101.0,  # ensure VWAP reclaim rule would fail
        session_vwap=100.3,
        orb_high=101.5,  # ensure ORB would fail
    )
    candidate = ema_pullback_5m_candidate(view)
    assert candidate is not None
    assert candidate.strategy is StrategyName.EMA_PULLBACK
    assert candidate.entry_ref == pytest.approx(101.0)
    # structural stop = min(prev low 99.9, current ema9 100.9) - 1.0*ATR(0.5)
    assert candidate.stop_loss == pytest.approx(99.4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ema_fast_prev": 101.5},  # prev low never touched the EMA9
        {"bar": make_5m_bar(close=100.9, open_=101.0, high=101.2, low=100.1, vwap=100.9)},  # bearish
        {"prev_bar": make_5m_bar(start=DAY2 - timedelta(minutes=5), close=100.1, open_=100.0, high=100.2, low=99.9, vwap=100.05),
         "prior_bars": (make_5m_bar(), make_5m_bar(), make_5m_bar(close=101.0, open_=100.2, high=101.2, low=100.1, vwap=100.6))},
        {"range_expansion": 0.8},  # expansion filter fails
        {"ema_slow": 101.0},  # ema9 must exceed ema21
        {"mirror_ok": False},
    ],
)
def test_ema_pullback_rejections(overrides):
    view = make_view(**overrides)
    assert ema_pullback_5m_candidate(view) is None


def test_stop_clamp_widest_and_tightest():
    # Structural stop far below the 2% floor -> clamped to entry * (1 - 2%).
    wide_view = make_view(orb_high=98.0, bar=make_5m_bar(close=101.0, open_=99.0, high=101.2, low=97.5, vwap=100.0), atr=1.0)
    wide = orb_15m_5m_candidate(wide_view)
    assert wide is not None
    assert wide.stop_loss == pytest.approx(101.0 * (1 - 0.02))

    # Structural stop tighter than 0.25% -> clamped up to entry * (1 - 0.25%).
    tight_view = make_view(
        orb_high=100.9,
        bar=make_5m_bar(close=101.0, open_=100.95, high=101.2, low=100.8, vwap=101.0),
        atr=0.01,
        range_expansion=1.5,
        rel_activity=1.2,
    )
    tight = orb_15m_5m_candidate(tight_view)
    assert tight is not None
    assert tight.stop_loss == pytest.approx(101.0 * (1 - 0.0025))


def test_evaluate_v02_uses_fixed_strategy_order():
    # Default view satisfies both ORB and VWAP; ORB must win.
    assert evaluate_v02(make_view()).strategy is StrategyName.ORB_15M
    # ORB filtered out by range expansion -> VWAP_RECLAIM wins.
    vwap_only = make_view(range_expansion=1.1, prev_session_vwap=100.5, session_vwap=100.3)
    assert evaluate_v02(vwap_only).strategy is StrategyName.VWAP_RECLAIM
    # Both ORB and VWAP filtered out -> EMA_PULLBACK.
    ema_only = make_view(
        range_expansion=1.0,
        prev_session_vwap=100.0,  # VWAP reclaim needs prev close < VWAP
        ema_fast_prev=100.05,
    )
    assert evaluate_v02(ema_only).strategy is StrategyName.EMA_PULLBACK


# --------------------------------------------------------------------------- #
# Engine: fills, exits, pending lifecycle
# --------------------------------------------------------------------------- #
def test_engine_fills_at_next_5m_bucket_open_with_slippage():
    data = build_prepared(soxl_day2_path([100.9 + 0.051 * index for index in range(1, 101)] + [106.0] * 270))
    result = run_v02_backtest(data, load_config())
    trade = first_trade(result)
    assert trade.symbol == "SOXL"
    assert trade.strategy is StrategyName.ORB_15M
    # Signal on the 09:45 bucket close -> fill at the 09:50 bucket's first 1m open.
    assert trade.entry_time_utc == DAY2 + timedelta(minutes=20)
    assert trade.entry_price_signal == pytest.approx(100.9)
    assert trade.entry_price_fill == pytest.approx(100.9 * (1.0 + SLIPPAGE_RATE))
    assert trade.exit_reason == "TARGET"
    assert trade.exit_time_utc is not None
    assert trade.exit_price_fill == pytest.approx(trade.target_price * (1.0 - SLIPPAGE_RATE))
    assert trade.pnl > 0
    # Metrics / equity curve bookkeeping.
    assert result.metrics.total_trades == len(result.trades)
    assert len(result.equity_curve) == len(result.trades)
    assert result.sessions_run == 2


def test_engine_same_1m_bar_tp_sl_collision_fills_stop():
    soxl_bars = make_1m_bars("SOXL", DAY2, soxl_day2_path([100.9] * 370))
    # Second 1m bar of the 09:50 bucket spans both the stop and the target.
    soxl_bars[21] = Bar(
        symbol="SOXL",
        timeframe_minutes=1,
        start_utc=DAY2 + timedelta(minutes=21),
        end_utc=DAY2 + timedelta(minutes=22),
        open=100.9,
        high=110.0,
        low=90.0,
        close=100.9,
        volume=1000.0,
        vwap=100.9,
        trade_count=25,
    )
    soxs_bars = make_1m_bars("SOXS", DAY2, soxs_falling_path())
    bars = sorted(
        make_1m_bars("SOXL", DAY1, [100.0] * 390) + soxl_bars
        + make_1m_bars("SOXS", DAY1, [100.0] * 390) + soxs_bars,
        key=lambda bar: (bar.start_utc, bar.symbol),
    )
    result = run_v02_backtest(prepare_v02_data(bars), load_config())
    trade = first_trade(result)
    assert trade.exit_reason == "STOP"
    assert trade.exit_time_utc == DAY2 + timedelta(minutes=22)
    assert trade.exit_price_fill == pytest.approx(trade.stop_price * (1.0 - SLIPPAGE_RATE))
    assert result.ambiguous_bars == 1
    detail = result.trade_details[trade.signal_id]
    assert detail["ambiguous_bar"] is True


def test_engine_flattens_at_eod():
    data = build_prepared(soxl_day2_path([100.9] * 370))
    result = run_v02_backtest(data, load_config())
    trade = first_trade(result)
    assert trade.exit_reason == "EOD_EXIT"
    last_bar_end = DAY2 + timedelta(minutes=390)  # 16:00 ET
    assert trade.exit_time_utc == last_bar_end
    assert trade.exit_price_fill == pytest.approx(100.9 * (1.0 - SLIPPAGE_RATE))


def test_engine_cancels_pending_when_next_bucket_is_missing():
    soxl_day2 = soxl_day2_path([100.9] * 370)
    soxl_bars = make_1m_bars("SOXL", DAY2, soxl_day2)
    soxs_bars = make_1m_bars("SOXS", DAY2, soxs_falling_path())
    # Remove the entire 09:50 bucket (minutes 20-24) from both symbols.
    drop = {DAY2 + timedelta(minutes=minute) for minute in range(20, 25)}
    soxl_bars = [bar for bar in soxl_bars if bar.start_utc not in drop]
    soxs_bars = [bar for bar in soxs_bars if bar.start_utc not in drop]
    bars = sorted(
        make_1m_bars("SOXL", DAY1, [100.0] * 390) + soxl_bars
        + make_1m_bars("SOXS", DAY1, [100.0] * 390) + soxs_bars,
        key=lambda bar: (bar.start_utc, bar.symbol),
    )
    result = run_v02_backtest(prepare_v02_data(bars), load_config())
    assert result.trades == []
    assert result.signals == 1
    assert result.pending_cancelled >= 1


def test_engine_ignores_intrabar_breakout_before_5m_close():
    """A 1m spike above the ORB high must not fire: only the completed 5m close counts."""
    path = [100.0] * 15 + [99.95] * 5 + [99.5] * 370
    soxl_bars = make_1m_bars("SOXL", DAY2, path)
    soxl_bars[15] = Bar(
        symbol="SOXL",
        timeframe_minutes=1,
        start_utc=DAY2 + timedelta(minutes=15),
        end_utc=DAY2 + timedelta(minutes=16),
        open=100.0,
        high=102.0,  # intrabar spike above the 100.08 opening-range high
        low=99.9,
        close=99.95,
        volume=1000.0,
        vwap=100.5,
        trade_count=25,
    )
    soxs_bars = make_1m_bars("SOXS", DAY2, soxs_falling_path())
    bars = sorted(
        make_1m_bars("SOXL", DAY1, [100.0] * 390) + soxl_bars
        + make_1m_bars("SOXS", DAY1, [100.0] * 390) + soxs_bars,
        key=lambda bar: (bar.start_utc, bar.symbol),
    )
    result = run_v02_backtest(prepare_v02_data(bars), load_config())
    assert result.signals == 0
    assert result.trades == []


class _RecordingRisk:
    """Risk-manager proxy recording the hard ``begin_session`` contract."""

    def __init__(self, inner: RiskManager) -> None:
        self._inner = inner
        self.begin_calls: list[datetime] = []

    def begin_session(self, now: datetime):
        self.begin_calls.append(now)
        return self._inner.begin_session(now)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_engine_calls_begin_session_once_per_session():
    data = build_prepared(soxl_day2_path([100.9 + 0.051 * index for index in range(1, 101)] + [106.0] * 270))
    config = load_config()
    proxy = _RecordingRisk(RiskManager(config))
    run_v02_backtest(data, config, risk=proxy)
    assert proxy.begin_calls == [DAY1, DAY2]


def test_engine_requires_begin_session_capable_risk_manager():
    data = build_prepared(soxl_day2_path([100.9] * 370))

    class _NoBeginRisk:
        """A risk manager with no ``begin_session`` (the V0.1 lock-up risk)."""

    with pytest.raises(TypeError, match="begin_session"):
        run_v02_backtest(data, load_config(), risk=_NoBeginRisk())


def test_engine_validates_session_ordering_of_input():
    good = make_1m_bars("SOXL", DAY2, [100.0] * 10)
    with pytest.raises(ValueError, match="sorted"):
        prepare_v02_data([good[1], good[0]] + good[2:])


# --------------------------------------------------------------------------- #
# Risk-manager integration smoke (mutual exclusion + daily circuit breaker)
# --------------------------------------------------------------------------- #
def _candidate(symbol: str = "SOXL", entry: float = 20.0, stop: float = 19.5) -> SignalCandidate:
    return SignalCandidate(
        symbol=symbol,
        strategy=StrategyName.ORB_15M,
        entry_ref=entry,
        stop_loss=stop,
        target_r=2.0,
        rationale="smoke",
        created_utc=DAY2 + timedelta(minutes=10),
        confirmation_bar_start_utc=DAY2 + timedelta(minutes=5),
    )


def test_risk_manager_mutual_exclusion_smoke():
    config = load_config()
    risk = RiskManager(config)
    risk.begin_session(DAY2)
    candidate = _candidate("SOXL")
    plan = size_position(candidate, config.risk)
    assert plan.valid
    decision = risk.evaluate_entry(candidate, plan, DAY2 + timedelta(minutes=10), data_ready=True)
    assert decision.allowed
    risk.confirm_entry(
        plan,
        fill_price=20.0,
        fill_utc=DAY2 + timedelta(minutes=10),
        signal_id="SMOKE-1",
        mode=Mode.DUAL_WATCH,
    )
    other = _candidate("SOXS")
    other_plan = size_position(other, config.risk)
    blocked = risk.evaluate_entry(
        other, other_plan, DAY2 + timedelta(minutes=20), data_ready=True
    )
    assert not blocked.allowed
    assert blocked.reason.value in {"POSITION_ALREADY_OPEN", "DUAL_CONFLICT"}


def test_risk_manager_daily_circuit_breaker_smoke():
    config = load_config()
    risk = RiskManager(config)
    risk.begin_session(DAY2)
    candidate = _candidate("SOXL")
    plan = size_position(candidate, config.risk)  # 40 shares, $20 risk each
    assert plan.shares == pytest.approx(40.0)
    offsets = [10, 20, 45]  # 2nd stop starts the 20-minute cooldown
    for index, offset in enumerate(offsets):
        now = DAY2 + timedelta(minutes=offset)
        decision = risk.evaluate_entry(candidate, plan, now, data_ready=True)
        assert decision.allowed, decision.message
        risk.confirm_entry(
            plan,
            fill_price=20.0,
            fill_utc=now,
            signal_id=f"SMOKE-{index}",
            mode=Mode.DUAL_WATCH,
        )
        risk.confirm_exit(
            fill_price=19.5, fill_utc=now + timedelta(minutes=2), reason="STOP"
        )
    assert risk.daily_stopped is True
    blocked = risk.evaluate_entry(
        candidate, plan, DAY2 + timedelta(minutes=60), data_ready=True
    )
    assert not blocked.allowed
    assert blocked.reason.value == "DAILY_LOSS_LIMIT"
    assert risk.daily_realized_pnl <= -config.risk.max_daily_net_loss


# --------------------------------------------------------------------------- #
# Walk-forward reuse
# --------------------------------------------------------------------------- #
def test_walkforward_uses_production_splits_and_fresh_state():
    # 30 flat synthetic sessions (weekdays), two symbols -> no clean trend, but
    # the split/leakage/fresh-state contract is what is under test here.
    dates = [
        datetime(2026, 9, 1, 14, 30, tzinfo=UTC) + timedelta(days=offset)
        for offset in range(60)
        if (datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=offset)).weekday() < 5
    ][:30]
    bars = []
    for start in dates:
        bars.extend(make_1m_bars("SOXL", start, [100.0] * 390))
        bars.extend(make_1m_bars("SOXS", start, [100.0] * 390))
    bars.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    data = prepare_v02_data(bars)
    wf = run_v02_walkforward(data, load_config(), train_days=10, test_days=5, step_days=5)
    assert wf.summary["folds"] == 4
    assert wf.summary["session_dates"] == 30
    assert len(wf.folds) == 4
    seen_test_dates: list = []
    for fold_result in wf.folds:
        assert fold_result.leakage_checked is True
        # Strictly chronological, no overlap between test windows.
        assert max(fold_result.fold.train_dates) < min(fold_result.fold.test_dates)
        seen_test_dates.extend(fold_result.fold.test_dates)
    assert len(seen_test_dates) == len(set(seen_test_dates))  # non-overlapping
    expected_trades = sum(fold.test_result.total_trades for fold in wf.folds)
    assert wf.aggregate_test_metrics.total_trades == expected_trades


def test_run_accepts_raw_bars_and_empty_result_is_zeroed():
    bars = []
    for start in (DAY1, DAY2):
        bars.extend(make_1m_bars("SOXL", start, [100.0] * 390))
        bars.extend(make_1m_bars("SOXS", start, [100.0] * 390))
    bars.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    result = run_v02_backtest(bars, load_config())  # raw bars -> prepared internally
    assert result.trades == []
    assert result.metrics.total_trades == 0
    assert result.metrics.profit_factor == 0.0
    assert result.metrics.net_pnl_dollars == 0.0
    assert "IEX" in result.notes[-1] or "IEX" in " ".join(result.notes)


def _breakout_day(start: datetime, scale: float = 1.0) -> tuple[list[Bar], list[Bar]]:
    """A session with one 09:45 ET ORB breakout, then flat until the EOD flatten."""
    soxl_base = [100.0] * 15 + [100.1, 100.3, 100.5, 100.7, 100.9] + [100.9] * 370
    soxs_base = [100.0 - 0.02 * index for index in range(390)]
    return (
        make_1m_bars("SOXL", start, [value * scale for value in soxl_base]),
        make_1m_bars("SOXS", start, [value * scale for value in soxs_base]),
    )


def _four_session_market(last_session_scale: float = 1.0):
    starts = [
        datetime(2026, 11, 3, 14, 30, tzinfo=UTC),
        datetime(2026, 11, 4, 14, 30, tzinfo=UTC),
        DAY1,
        DAY2,
    ]
    bars: list[Bar] = []
    for index, start in enumerate(starts):
        scale = last_session_scale if index == len(starts) - 1 else 1.0
        soxl_bars, soxs_bars = _breakout_day(start, scale)
        bars.extend(soxl_bars)
        bars.extend(soxs_bars)
    bars.sort(key=lambda bar: (bar.start_utc, bar.symbol))
    return prepare_v02_data(bars)


def test_no_lookahead_future_bars_do_not_change_past_trades():
    """Scaling every price of the *last* session must not alter earlier trades."""
    config = load_config()
    baseline = run_v02_backtest(_four_session_market(1.0), config)
    distorted = run_v02_backtest(_four_session_market(1.5), config)
    cutoff = DAY2  # signals of the last session start at 09:45 ET after the open

    def signatures(result):
        return [
            (
                trade.signal_id,
                trade.symbol,
                trade.strategy.value,
                trade.entry_time_utc.isoformat(),
                round(trade.entry_price_signal, 6),
                round(trade.entry_price_fill, 6),
                round(trade.stop_price, 6),
                round(trade.target_price, 6),
                trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
                trade.exit_reason,
                round(trade.pnl, 6),
            )
            for trade in result.trades
            if trade.entry_time_utc < cutoff
        ]

    before = signatures(baseline)
    after = signatures(distorted)
    assert len(before) >= 2, f"expected at least two pre-cutoff trades, got {before}"
    assert before == after
