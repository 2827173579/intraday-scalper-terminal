"""Offline, deterministic tests for V0.5 R-Ladder exits + OR30 dead-fish filter.

Synthetic sessions only.  Entries are frozen to V0.4 Step3c-A NoCap; the tests
here cover the new V0.5 exit structures and the OR30 day filter:

* R-ladder levels (Runner-A) / monotonic stops / no early breakeven;
* Runner-B lock + trailing levels and the EMA21 exit;
* fixed 4.5R target for Runner-A, no fixed cap for Runner-B;
* same-bar stop/target collision -> conservative stop, and no same-bar look-ahead
  for ladder updates (updates come from completed 5m bars);
* EOD 15:59 flatten; OR30 pct / literal OR conditions + same-day entry skip;
* baseline exit equals the frozen V0.4 Step3c-NoCap exit at the same B1 sizing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import Bar, load_config
from intraday_scalper.research import v04, v05
from intraday_scalper.research.v02 import prepare_v02_data

UTC = timezone.utc
DAY1 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
DAY2 = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
SESSION_END = DAY2 + timedelta(minutes=390)  # 21:00 UTC = 16:00 ET


# --------------------------------------------------------------------------- #
# Synthetic builders (SMC sweep entry at 10:30 ET -> fill at 10:35 ET)
# --------------------------------------------------------------------------- #
def make_1m_bars(
    symbol: str,
    start: datetime,
    closes: list[float],
    *,
    volume: float = 1000.0,
    trade_count: int = 25,
    spread: float = 0.0008,
) -> list[Bar]:
    bars: list[Bar] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        open_ = previous
        high = max(open_, close) * (1.0 + spread)
        low = min(open_, close) * (1.0 - spread)
        vwap = (open_ + high + low + close) / 4.0
        bar_start = start + timedelta(minutes=index)
        bars.append(
            Bar(
                symbol=symbol,
                timeframe_minutes=1,
                start_utc=bar_start,
                end_utc=bar_start + timedelta(minutes=1),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                vwap=vwap,
                trade_count=trade_count,
            )
        )
        previous = close
    return bars


def soxs_falling_path() -> list[float]:
    return [100.0 - 0.02 * index for index in range(390)]


def sweep_day2_path(post: list[float] | None = None) -> list[float]:
    """Day-2 SOXL: flat 100, SMC lower-wick sweep at 10:30 ET, then *post*."""
    path = [100.0] * 60 + [99.3, 99.3, 100.0, 100.6, 100.6]
    remaining = 390 - len(path)
    path += list(post) if post is not None else [100.6] * remaining
    assert len(path) == 390, len(path)
    return path


def build_prepared(
    soxl_day2: list[float],
    *,
    soxs_day2: list[float] | None = None,
):
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, soxl_day2)
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars(
        "SOXS", DAY2, soxs_day2 if soxs_day2 is not None else soxs_falling_path()
    )
    bars = sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    return prepare_v02_data(bars)


def replace_bar(
    data_bars: list[Bar],
    symbol: str,
    minute: int,
    **overrides,
) -> None:
    """Replace the 1m bar of *symbol* starting at DAY2 + *minute* minutes."""

    target_start = DAY2 + timedelta(minutes=minute)
    for index, bar in enumerate(data_bars):
        if bar.symbol == symbol and bar.start_utc == target_start:
            fields = dict(
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count,
            )
            fields.update(overrides)
            data_bars[index] = Bar(
                symbol=symbol,
                timeframe_minutes=1,
                start_utc=bar.start_utc,
                end_utc=bar.end_utc,
                **fields,
            )
            return
    raise AssertionError(f"bar not found: {symbol} at {target_start}")


def run_variant(data, variant: v05.V05Variant, *, session_dates=None):
    return v05.run_v05_backtest(data, variant=variant, session_dates=session_dates)


def first_trade(result: v05.V05Result):
    assert result.trades, "expected at least one trade"
    return sorted(result.trades, key=lambda trade: trade.entry_time_utc)[0]


# --------------------------------------------------------------------------- #
# B1 risk config + pure ladder helpers
# --------------------------------------------------------------------------- #
def test_b1_risk_config_matches_task_spec():
    config = v05.build_b1_config(load_config(), 60.0)
    risk = config.risk
    assert risk.account_equity == pytest.approx(10_000.0)
    assert risk.max_risk_per_trade == pytest.approx(60.0)
    assert risk.max_notional_per_trade == pytest.approx(10_000.0)
    assert risk.max_daily_net_loss == pytest.approx(320.0)
    assert risk.max_entries_per_day == 15
    assert risk.consecutive_stop_threshold == 3
    assert risk.consecutive_stop_cooldown_minutes == 15
    assert risk.allow_fractional_shares is True
    assert risk.fractional_precision == 2
    assert risk.est_slippage_bps == pytest.approx(1.5)
    assert risk.commission_per_share == pytest.approx(0.0)
    # risk-$80 sensitivity override
    config_80 = v05.build_b1_config(load_config(), 80.0)
    assert config_80.risk.max_risk_per_trade == pytest.approx(80.0)


def test_runner_a_ladder_levels_are_monotonic():
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 0.0) == (-1.0, "INITIAL")
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 1.49)[0] == pytest.approx(-1.0)
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 1.5) == (0.20, "LADDER_1.5R")
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 2.19) == (0.20, "LADDER_1.5R")
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 2.2) == (1.20, "LADDER_2.2R")
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 2.99) == (1.20, "LADDER_2.2R")
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 3.0) == (2.00, "LADDER_3R")
    # No level above 3.0R -> stop never moves beyond +2.00R (only up, never down).
    assert v05.ladder_stop(v05.EXIT_RUNNER_A, 9.0) == (2.00, "LADDER_3R")


def test_runner_a_no_breakeven_before_1_5r():
    # The V0.3 shield would push the stop at 1.2R; Runner-A must not.
    for mfe in (0.0, 0.5, 1.19, 1.2, 1.49):
        level, kind = v05.ladder_stop(v05.EXIT_RUNNER_A, mfe)
        assert level == pytest.approx(-1.0)
        assert kind == "INITIAL"


def test_runner_b_ladder_and_trailing_lock():
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 0.0) == (-1.0, "INITIAL")
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 1.5) == (0.20, "LADDER_1.5R")
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 2.0) == (1.00, "LADDER_2R")
    # Above 2.0R the stop trails at max(1.0R, MFE - 1.0R) and never goes down.
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 2.5) == (1.50, "TRAIL_LOCK")
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 4.0) == (3.00, "TRAIL_LOCK")
    assert v05.ladder_stop(v05.EXIT_RUNNER_B, 1.0) == (-1.0, "INITIAL")


def test_target_prices_per_exit_mode():
    entry, risk_per_share = 100.0, 1.0
    assert v05.target_price_for_mode(v05.EXIT_RUNNER_A, entry, risk_per_share) == pytest.approx(104.5)
    assert v05.target_price_for_mode(v05.EXIT_RUNNER_B, entry, risk_per_share) is None
    assert v05.target_price_for_mode(v05.EXIT_BASELINE, entry, risk_per_share) is None


# --------------------------------------------------------------------------- #
# OR30 filter
# --------------------------------------------------------------------------- #
def _or30_metrics(range_pct: float | None, or30_range: float | None, atr: float | None):
    return {
        "range_pct": range_pct,
        "or30_range": or30_range,
        "atr14_5m_1000": atr,
    }


def test_or30_pct_only_definition():
    assert v05.or30_filter_blocks(_or30_metrics(0.017, 1.0, 1.0), v05.OR30_FILTER_PCT_ONLY)
    assert not v05.or30_filter_blocks(
        _or30_metrics(0.018, 1.0, 1.0), v05.OR30_FILTER_PCT_ONLY
    )
    assert not v05.or30_filter_blocks(
        _or30_metrics(0.030, 0.5, 1.0), v05.OR30_FILTER_PCT_ONLY
    )
    # unavailable OR30 data never blocks (and never silently trades a filter day)
    assert not v05.or30_filter_blocks(
        _or30_metrics(None, None, None), v05.OR30_FILTER_PCT_ONLY
    )
    # "none" filter never blocks
    assert not v05.or30_filter_blocks(
        _or30_metrics(0.001, 0.01, 1.0), v05.OR30_FILTER_NONE
    )


def test_or30_literal_or_definition():
    mode = v05.OR30_FILTER_PCT_OR_ATR
    # pct condition alone blocks
    assert v05.or30_filter_blocks(_or30_metrics(0.015, 5.0, 1.0), mode)
    # range < 1.2 * ATR blocks even though the pct is fine (literal OR)
    assert v05.or30_filter_blocks(_or30_metrics(0.030, 1.0, 1.0), mode)
    # neither condition -> pass
    assert not v05.or30_filter_blocks(_or30_metrics(0.030, 1.5, 1.0), mode)
    # ATR unavailable -> only pct condition applies
    assert not v05.or30_filter_blocks(_or30_metrics(0.030, 1.0, None), mode)


def test_or30_compute_blocks_and_same_day_entry_skip():
    data = build_prepared(sweep_day2_path())
    blocks, rows = v05.compute_or30_blocks(data, v05.OR30_FILTER_PCT_ONLY)
    assert (DAY2.date(), "SOXL") in blocks  # flat OR30 -> dead-fish day
    assert rows and rows[0]["blocked"] in (True, False)
    # Baseline takes the synthetic SMC trade without the filter ...
    unfiltered = run_variant(data, v05.BASELINE_VARIANT, session_dates=[DAY2.date()])
    assert unfiltered.total_trades == 1
    # ... and skips all new entries on the blocked day with it.
    filtered_variant = replace(
        v05.BASELINE_VARIANT,
        name="B1+F",
        or30_filter=v05.OR30_FILTER_PCT_ONLY,
    )
    filtered = run_variant(data, filtered_variant, session_dates=[DAY2.date()])
    assert filtered.total_trades == 0
    assert filtered.or30_skipped_evaluations > 0


# --------------------------------------------------------------------------- #
# Baseline equivalence with the frozen V0.4 Step3c-A NoCap exit
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
            trade.exit_time_utc.isoformat() if trade.exit_time_utc else None,
            round(trade.pnl, 8),
            round(trade.r_multiple, 8),
        )
        for trade in sorted(result.trades, key=lambda trade: trade.entry_time_utc)
    ]


def test_baseline_equals_v04_step3c_nocap_at_same_sizing():
    data = build_prepared(sweep_day2_path())
    b1_config = v05.build_b1_config(load_config(), 60.0)
    v04_result = v04.run_v04_backtest(
        data,
        b1_config,
        variant=v04.STEP3C_NOCAP_VARIANT,
        initial_equity=v05.B1_EQUITY,
        session_dates=[DAY2.date()],
    )
    v05_result = run_variant(data, v05.BASELINE_VARIANT, session_dates=[DAY2.date()])
    assert _trade_signature(v05_result) == _trade_signature(v04_result)
    assert v05_result.metrics.net_pnl_dollars == pytest.approx(
        v04_result.metrics.net_pnl_dollars, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Runner-A engine behaviour
# --------------------------------------------------------------------------- #
# Entry fill ~100.615, initial stop ~99.857 -> R ~= 0.7583.
_RUNNER_A_LADDER_POST = (
    [100.7, 101.5, 102.0, 102.3, 102.2]  # bucket 10:35: MFE ~2.33R
    + [101.4, 101.3, 101.2, 101.2, 101.2]  # next bucket breaks +1.20R stop
    + [101.2] * (390 - 75)
)
_RUNNER_A_TARGET_POST = (
    [101.0, 102.5, 103.5, 104.2, 104.2]  # bar high clears +4.5R (~104.03)
    + [104.2] * (390 - 70)
)


def test_runner_a_ladder_stop_exit():
    data = build_prepared(sweep_day2_path(_RUNNER_A_LADDER_POST))
    result = run_variant(data, v05.RUNNER_A_VARIANT)
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "STOP"
    assert detail["exit_stop_kind"] == "LADDER_2.2R"
    assert detail["exit_reference_r"] == pytest.approx(1.2, abs=1e-6)
    assert detail["max_mfe_r"] >= 2.2
    assert trade.pnl > 0.0  # +1.2R locked instead of the -1R initial stop


def test_runner_a_fixed_45r_target():
    data = build_prepared(sweep_day2_path(_RUNNER_A_TARGET_POST))
    result = run_variant(data, v05.RUNNER_A_VARIANT)
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "TARGET"
    assert detail["exit_stop_kind"] == "TARGET"
    assert detail["exit_reference_r"] == pytest.approx(4.5, abs=1e-6)
    assert trade.r_multiple > 4.0


# --------------------------------------------------------------------------- #
# Runner-B engine behaviour
# --------------------------------------------------------------------------- #
_EMA21_EXIT_POST = (
    [100.1, 99.95, 99.95, 99.95, 99.95]  # 5m close 99.95 < EMA21 (~100.1)
    + [99.95] * (390 - 70)
)


def test_runner_b_ema21_exit_next_bar_open():
    data = build_prepared(sweep_day2_path(_EMA21_EXIT_POST))
    result = run_variant(data, v05.RUNNER_B_VARIANT)
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "TRAIL_EMA21"
    assert detail["exit_stop_kind"] == "EMA21_CLOSE"
    # Exit executes at the next 1m bar open after the completed 5m close.
    assert trade.exit_time_utc == DAY2 + timedelta(minutes=70)


def test_runner_b_has_no_fixed_r_cap_and_runs_to_eod():
    data = build_prepared(sweep_day2_path(_RUNNER_A_TARGET_POST))
    result = run_variant(data, v05.RUNNER_B_VARIANT)
    trade = first_trade(result)
    assert trade.exit_reason == "EOD_EXIT"
    # Uncapped: well beyond the 4.5R cap Runner-A applies.
    assert trade.r_multiple > 4.5


def test_eod_1559_flatten_for_runner_a():
    data = build_prepared(sweep_day2_path())
    result = run_variant(data, v05.RUNNER_A_VARIANT)
    trade = first_trade(result)
    assert trade.exit_reason == "EOD_EXIT"
    assert trade.exit_time_utc == SESSION_END  # 16:00 ET bar end = 15:59 ET close


# --------------------------------------------------------------------------- #
# No look-ahead: 5m-bar ladder updates apply from the next bar only
# --------------------------------------------------------------------------- #
def test_ladder_update_is_not_applied_within_same_bar():
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, sweep_day2_path())
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars("SOXS", DAY2, soxs_falling_path())
    # One bar in the fist post-entry bucket spans both a would-be 2.2R-high and
    # the initial stop; the ladder cannot have been armed before the 5m close.
    replace_bar(
        soxl,
        "SOXL",
        minute=66,
        open=100.6,
        high=103.0,
        low=99.5,
        close=101.0,
        vwap=101.5,
    )
    bars = sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    data = prepare_v02_data(bars)
    result = run_variant(data, v05.RUNNER_A_VARIANT)
    trade = first_trade(result)
    detail = result.trade_details[trade.signal_id]
    assert trade.exit_reason == "STOP"
    assert detail["exit_stop_kind"] == "INITIAL"


# --------------------------------------------------------------------------- #
# OR30 / raw-bar input sanity
# --------------------------------------------------------------------------- #
def test_run_accepts_raw_bars_and_reports_exit_distribution():
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, sweep_day2_path())
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars("SOXS", DAY2, soxs_falling_path())
    bars = sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    result = v05.run_v05_backtest(bars, variant=v05.RUNNER_A_VARIANT)
    extra = result.extra_metrics()
    assert result.total_trades == 1
    assert extra["exit_reasons"].get("EOD_EXIT") == 1
    assert extra["sessions_run"] == 2
    assert "target_exits" in extra
