"""Offline, deterministic tests for V0.5b pre-registered F1/F2 + Web API.

Covers the frozen Runner-A exits plus the new filter rules:

* F1 threshold boundaries (2.8% / 2.2 x ATR14_5m) and no effect on the original
  A+F thresholds unless they are met;
* F2 trend-box rule: OR30 breakout, upper-half + rising 3-bar EMA21 slope,
  rejection inside the box / flat or falling slope, conservative missing data;
* SMC_SWEEP_5M is exempt from F2 (structural + engine behaviour);
* causal usage: the slope uses the EMA21 value from 3 completed bars back;
* the pre-registered selection rule (WF>0 gate, composite score, <5% lead keeps
  A+F);
* `v05.evaluate_v05_session` Web API and the winner's Friday recap/markers
  payload structure.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import Bar, load_config
from intraday_scalper.research import v05
from intraday_scalper.research.v02 import prepare_v02_data

UTC = timezone.utc
DAY1 = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
DAY2 = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)


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
    path = [100.0] * 60 + [99.3, 99.3, 100.0, 100.6, 100.6]
    remaining = 390 - len(path)
    path += list(post) if post is not None else [100.6] * remaining
    assert len(path) == 390, len(path)
    return path


def build_prepared(soxl_day2: list[float], soxs_day2: list[float] | None = None):
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, soxl_day2)
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars(
        "SOXS", DAY2, soxs_day2 if soxs_day2 is not None else soxs_falling_path()
    )
    return prepare_v02_data(
        sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))
    )


def build_raw_bars(soxl_day2: list[float] | None = None) -> list[Bar]:
    soxl = make_1m_bars("SOXL", DAY1, [100.0] * 390)
    soxl += make_1m_bars("SOXL", DAY2, soxl_day2 or sweep_day2_path())
    soxs = make_1m_bars("SOXS", DAY1, [100.0] * 390)
    soxs += make_1m_bars("SOXS", DAY2, soxs_falling_path())
    return sorted(soxl + soxs, key=lambda bar: (bar.start_utc, bar.symbol))


# --------------------------------------------------------------------------- #
# F1 boundaries
# --------------------------------------------------------------------------- #
def _or30_metrics(range_pct: float | None, or30_range: float | None, atr: float | None):
    return {"range_pct": range_pct, "or30_range": or30_range, "atr14_5m_1000": atr}


def test_f1_pct_threshold_boundaries():
    mode = v05.OR30_FILTER_F1
    assert v05.or30_filter_blocks(_or30_metrics(0.0279, 5.0, 1.0), mode)
    assert not v05.or30_filter_blocks(_or30_metrics(0.0280, 5.0, 1.0), mode)
    assert not v05.or30_filter_blocks(_or30_metrics(0.0400, 5.0, 1.0), mode)
    # Original A+F thresholds are untouched by F1 (2.7% passes the 1.8% rule).
    assert not v05.or30_filter_blocks(
        _or30_metrics(0.0270, 5.0, 1.0), v05.OR30_FILTER_PCT_OR_ATR
    )
    assert v05.or30_filter_blocks(
        _or30_metrics(0.0270, 5.0, 1.0), v05.OR30_FILTER_F1
    )


def test_f1_atr_threshold_boundaries():
    mode = v05.OR30_FILTER_F1
    # range below 2.2 x ATR -> block (pct deliberately wide)
    assert v05.or30_filter_blocks(_or30_metrics(0.05, 2.19, 1.0), mode)
    # range exactly at 2.2 x ATR -> pass (strict <)
    assert not v05.or30_filter_blocks(_or30_metrics(0.05, 2.20, 1.0), mode)
    # missing ATR -> only the pct leg applies
    assert not v05.or30_filter_blocks(_or30_metrics(0.05, 0.1, None), mode)
    assert v05.or30_filter_blocks(_or30_metrics(0.027, 0.1, None), mode)


# --------------------------------------------------------------------------- #
# F2 trend-box rule
# --------------------------------------------------------------------------- #
def test_f2_breakout_above_or30_high_allowed():
    assert v05.trend_box_allows(101.0, 100.0, 99.0, None, None)
    assert v05.trend_box_allows(100.0, 100.0, 99.0, 99.0, 99.5)  # equality


def test_f2_upper_half_with_rising_slope_allowed():
    # close 99.6 -> upper half of [99, 100]; EMA21 slope +0.16% over 3 bars.
    assert v05.trend_box_allows(99.6, 100.0, 99.0, 100.16, 100.0)
    # exactly at the 0.15% threshold passes (>=)
    assert v05.trend_box_allows(99.6, 100.0, 99.0, 100.15, 100.0)


def test_f2_upper_half_flat_or_falling_slope_rejected():
    assert not v05.trend_box_allows(99.6, 100.0, 99.0, 100.14, 100.0)
    assert not v05.trend_box_allows(99.6, 100.0, 99.0, 99.90, 100.0)


def test_f2_inside_lower_half_rejected():
    # close 99.2 is below the 99.5 mid, even with a steep rising slope
    assert not v05.trend_box_allows(99.2, 100.0, 99.0, 101.0, 100.0)


def test_f2_missing_references_rejected_conservatively():
    assert not v05.trend_box_allows(99.6, None, None, 100.16, 100.0)
    assert not v05.trend_box_allows(99.6, 100.0, 99.0, None, 100.0)
    assert not v05.trend_box_allows(99.6, 100.0, 99.0, 100.16, None)


def test_f2_smc_sweep_is_exempt_structurally():
    assert v05._TREND_LABELS == ("VWAP_RECLAIM_5M", "EMA_PULLBACK_5M")
    assert "SMC_SWEEP_5M" not in v05._TREND_LABELS


# --------------------------------------------------------------------------- #
# Engine-level behaviour on synthetic data
# --------------------------------------------------------------------------- #
def test_f2_engine_keeps_smc_trades():
    """F2 never blocks SMC_SWEEP_5M; the synthetic entry is SMC-only."""
    data = build_prepared(sweep_day2_path())
    result = v05.run_v05_backtest(
        data, variant=v05.A_F2_VARIANT, session_dates=[DAY2.date()]
    )
    assert result.total_trades == 1
    trade = result.trades[0]
    detail = result.trade_details[trade.signal_id]
    assert detail["strategy_label"] == "SMC_SWEEP_5M"
    assert result.extra_metrics()["trend_box_rejected"] == 0


def test_original_filter_blocks_narrow_or30_day_but_f2_does_not():
    """Flat OR30 (0.16%) is a dead-fish day for A+F/A+F1; F2 alone keeps SMC."""
    data = build_prepared(sweep_day2_path())
    dates = [DAY2.date()]
    unfiltered = v05.run_v05_backtest(
        data, variant=v05.A_VARIANT, session_dates=dates
    )
    original = v05.run_v05_backtest(
        data, variant=v05.A_PLUS_F_VARIANT, session_dates=dates
    )
    f1 = v05.run_v05_backtest(
        data, variant=v05.A_F1_VARIANT, session_dates=dates
    )
    f2 = v05.run_v05_backtest(
        data, variant=v05.A_F2_VARIANT, session_dates=dates
    )
    assert unfiltered.total_trades == 1
    assert original.total_trades == 0  # 1.8% / 1.2 ATR day filter blocks
    assert f1.total_trades == 0  # 2.8% / 2.2 ATR day filter also blocks
    assert f2.total_trades == 1  # no day filter; SMC exempt from F2


def test_f1_engine_skips_evaluations_in_3m_like_window():
    data = build_prepared(sweep_day2_path())
    result = v05.run_v05_backtest(
        data, variant=v05.A_F1_VARIANT, session_dates=[DAY2.date()]
    )
    assert result.extra_metrics()["or30_skipped_evaluations"] > 0
    assert result.total_trades == 0


# --------------------------------------------------------------------------- #
# Web API + marker types
# --------------------------------------------------------------------------- #
def test_web_evaluate_v05_session_returns_markers():
    bars = build_raw_bars()
    trades = v05.evaluate_v05_session(bars, DAY2.date(), v05.A_VARIANT)
    assert isinstance(trades, list) and trades
    trade = trades[0]
    for key in (
        "kind",
        "trade_id",
        "symbol",
        "strategy",
        "strategy_label",
        "entry_time_utc",
        "entry_time_et",
        "entry_price_fill",
        "stop_price",
        "target_price",
        "exit_reason",
        "pnl",
        "r_multiple",
        "markers",
    ):
        assert key in trade
    assert trade["kind"] == "trade"
    assert trade["markers"], "Web trades must carry markers"
    for marker in trade["markers"]:
        for key in ("type", "time_utc", "price", "symbol", "strategy", "trade_id", "t"):
            assert key in marker
        assert marker["trade_id"] == trade["trade_id"]
    assert {marker["type"] for marker in trade["markers"]} <= {
        "ENTRY",
        "STOP",
        "TARGET",
        "EOD_EXIT",
        "TRAIL_EMA21",
        "EXIT",
    }


def test_web_evaluate_unknown_session_raises():
    bars = build_raw_bars()
    with pytest.raises(ValueError):
        v05.evaluate_v05_session(bars, "2026-11-04", v05.A_VARIANT)


def test_marker_types_include_trail_and_eod():
    # Runner-B EMA21 exit path -> explicit TRAIL_EMA21 marker type.
    ema_post = [100.1, 99.95, 99.95, 99.95, 99.95] + [99.95] * (390 - 70)
    data = build_prepared(sweep_day2_path(ema_post))
    trail_result = v05.run_v05_backtest(
        data, variant=v05.RUNNER_B_VARIANT, session_dates=[DAY2.date()]
    )
    types = {marker["type"] for marker in trail_result.markers}
    assert "TRAIL_EMA21" in types
    # Runner-A flat path -> EOD_EXIT marker type.
    flat_data = build_prepared(sweep_day2_path())
    eod_result = v05.run_v05_backtest(
        flat_data, variant=v05.A_VARIANT, session_dates=[DAY2.date()]
    )
    eod_types = {marker["type"] for marker in eod_result.markers}
    assert "EOD_EXIT" in eod_types


# --------------------------------------------------------------------------- #
# Pre-registered selection rule
# --------------------------------------------------------------------------- #
def _selection_payload(
    window_net: dict[str, float],
    full_net: dict[str, float],
    wf_net: dict[str, float],
) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for label in v05.V05_VARIANTS:
        payload[label] = {
            "window": {"net_pnl_dollars": window_net.get(label, 0.0)},
            "full": {"net_pnl_dollars": full_net.get(label, 0.0)},
            "walkforward": {
                "net_pnl_dollars": wf_net.get(label, 0.0),
                "positive_folds": 14,
            },
        }
    return payload


def test_selection_rule_keeps_a_plus_f_when_lead_under_5pct():
    from intraday_scalper.tools.run_v05b_filters import select_winner

    payload = _selection_payload(
        window_net={"A+F": 100.0, "A+F1": 101.0},  # improves 3M by only 1%
        full_net={"A+F": 100.0, "A+F1": 103.0},
        wf_net={"A+F": 100.0, "A+F1": 105.0},  # composite lead ~3% (<5%)
    )
    selection = select_winner(payload)
    # A+F1 improves both 3M and WF but leads A+F by <5% -> keep A+F.
    assert "A+F1" in selection["improved_both_vs_A+F"]
    assert selection["winner"] == "A+F"
    assert selection["no_significant_beat"] is True


def test_selection_rule_takes_clear_winner_when_lead_over_5pct():
    from intraday_scalper.tools.run_v05b_filters import select_winner

    payload = _selection_payload(
        window_net={"A+F": 100.0, "A+F1": 200.0},
        full_net={"A+F": 100.0, "A+F1": 200.0},
        wf_net={"A+F": 100.0, "A+F1": 200.0},
    )
    selection = select_winner(payload)
    assert selection["winner"] == "A+F1"
    assert selection["no_significant_beat"] is False


def test_selection_rule_requires_positive_wf():
    from intraday_scalper.tools.run_v05b_filters import select_winner

    payload = _selection_payload(
        window_net={"A+F": 100.0, "A+F1": 1000.0},
        full_net={"A+F": 100.0, "A+F1": 1000.0},
        wf_net={"A+F": 100.0, "A+F1": -10.0},  # huge score but WF negative
    )
    selection = select_winner(payload)
    assert selection["winner"] == "A+F"
    assert "A+F1" not in selection["eligible"]


# --------------------------------------------------------------------------- #
# Winner Friday recap / markers payload (tool helper)
# --------------------------------------------------------------------------- #
def test_winner_friday_recap_and_markers_payload_structure():
    from intraday_scalper.tools import run_v05b_filters

    prepared = build_prepared(sweep_day2_path())
    friday_bars = build_raw_bars()
    selection = {"verdict": "unit-test winner"}
    recap, markers_payload = run_v05b_filters.build_friday_payload(
        friday_bars,
        prepared,
        winner_label="A+F",
        selection=selection,
    )
    assert recap["version"] == "V0.5b"
    assert recap["session_date"] == DAY2.date().isoformat()
    assert recap["winner"] == "A+F"
    assert recap["session_date"] == "2026-11-06"
    for key in (
        "variant",
        "summary",
        "exit_reason_distribution",
        "trades",
        "markers",
        "market",
        "filters",
    ):
        assert key in recap
    assert markers_payload["winner"] == "A+F"
    assert isinstance(markers_payload["markers"], list)
    for marker in markers_payload["markers"]:
        for key in ("type", "time_utc", "price", "symbol", "strategy", "trade_id", "t"):
            assert key in marker
    # marker grouping keyed by trade_id for Web replay
    for trade_id, markers in markers_payload["markers_by_trade"].items():
        assert isinstance(trade_id, str)
        assert all(marker["trade_id"] == trade_id for marker in markers)


def test_v05b_variants_and_production_default():
    assert set(v05.V05_VARIANTS) == {"A", "A+F", "A+F1", "A+F2", "A+F1+F2"}
    assert v05.V05_PRODUCTION_VARIANT.name == "A+F"
    assert [variant.name for variant in v05.V05B_VARIANTS] == [
        "A+F",
        "A+F1",
        "A+F2",
        "A+F1+F2",
    ]
    # A+F keeps the original 1.8% / 1.2 ATR filter; F1 the 2.8% / 2.2 ATR one.
    assert v05.A_PLUS_F_VARIANT.or30_filter == v05.OR30_FILTER_PCT_OR_ATR
    assert v05.A_F1_VARIANT.or30_filter == v05.OR30_FILTER_F1
    assert v05.A_F2_VARIANT.trend_box_filter is True
    assert v05.A_F1_F2_VARIANT.trend_box_filter is True
