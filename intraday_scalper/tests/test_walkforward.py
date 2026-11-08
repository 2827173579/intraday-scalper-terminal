"""Offline, deterministic tests for walk-forward splitting and validation."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from intraday_scalper.backtest import indicators as bt_indicators
from intraday_scalper.backtest._compat import FallbackStrategyContext
from intraday_scalper.backtest.risk_adapter import (
    SimulatedRiskManager,
    size_position_equivalent,
)
from intraday_scalper.backtest.synthetic import generate_session, generate_sessions, next_trading_dates
from intraday_scalper.backtest.walkforward import (
    Fold,
    FoldResult,
    WalkForwardResult,
    assert_no_leakage,
    run_walkforward,
    walk_forward_splits,
)
from intraday_scalper.core import StrategyName, SignalCandidate, load_config

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)
FIRST_MONDAY = date(2026, 11, 2)
MINUTES_PER_SESSION = 30


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class FireEveryBar:
    """Tiny deterministic fake strategy: one candidate per confirmed bar."""

    name = StrategyName.ORB_5M

    def evaluate(self, ctx):
        last = ctx.bars_1m[-1]
        return SignalCandidate(
            symbol=ctx.symbol,
            strategy=self.name,
            entry_ref=last.close,
            stop_loss=last.close * 0.99,
            target_r=2.0,
            rationale="walk-forward fake strategy",
            created_utc=ctx.now_utc,
            confirmation_bar_start_utc=last.start_utc,
        )


class FireNever:
    name = StrategyName.ORB_5M

    def evaluate(self, ctx):  # noqa: ARG002 - intentionally no signal
        return None


def engine_kwargs() -> dict:
    return dict(
        sizing_fn=size_position_equivalent,
        indicators=bt_indicators,
        context_class=FallbackStrategyContext,
        slippage_bps=0.0,
        commission_per_share=0.0,
    )


def make_risk_factory(config):
    return lambda: SimulatedRiskManager(config)


def synthetic_bars(sessions: int = 8, *, seed: int = 5, symbol: str = "SOXL"):
    dates = next_trading_dates(FIRST_MONDAY, sessions)
    return generate_sessions((symbol,), dates, seed=seed, n_minutes=MINUTES_PER_SESSION)


# --------------------------------------------------------------------------- #
# Split logic
# --------------------------------------------------------------------------- #
def test_walk_forward_splits_are_chronological_and_non_overlapping():
    dates = next_trading_dates(FIRST_MONDAY, 10)
    folds = walk_forward_splits(dates, train_days=3, test_days=2, step_days=2)

    assert len(folds) == 3
    assert folds[0].train_dates == tuple(dates[0:3])
    assert folds[0].test_dates == tuple(dates[3:5])
    assert folds[1].train_dates == tuple(dates[2:5])
    assert folds[1].test_dates == tuple(dates[5:7])
    assert folds[2].train_dates == tuple(dates[4:7])
    assert folds[2].test_dates == tuple(dates[7:9])
    assert [fold.index for fold in folds] == [0, 1, 2]

    # No leakage: training strictly before the test window in every fold.
    for fold in folds:
        assert max(fold.train_dates) < min(fold.test_dates)
        assert not set(fold.train_dates) & set(fold.test_dates)

    # Test windows are pairwise disjoint.
    seen: set[date] = set()
    for fold in folds:
        assert not seen & set(fold.test_dates)
        seen |= set(fold.test_dates)


def test_walk_forward_splits_default_step_and_input_hygiene():
    dates = next_trading_dates(FIRST_MONDAY, 6)
    shuffled_and_duplicated = [dates[3], dates[0], dates[3], dates[1], dates[5], dates[2], dates[4]]
    folds = walk_forward_splits(shuffled_and_duplicated, train_days=2, test_days=2)
    # step defaults to test_days (2): starts 0 and 2 for 6 dates.
    assert len(folds) == 2
    assert folds[0].train_dates == tuple(dates[0:2])
    assert folds[0].test_dates == tuple(dates[2:4])
    assert folds[1].test_dates == tuple(dates[4:6])
    assert all(isinstance(fold, Fold) for fold in folds)


def test_walk_forward_splits_insufficient_dates_returns_empty():
    dates = next_trading_dates(FIRST_MONDAY, 3)
    assert walk_forward_splits(dates, train_days=2, test_days=2) == []
    assert walk_forward_splits([], train_days=1, test_days=1) == []


def test_walk_forward_splits_larger_step_keeps_windows_disjoint():
    dates = next_trading_dates(FIRST_MONDAY, 10)
    folds = walk_forward_splits(dates, train_days=2, test_days=2, step_days=3)
    assert len(folds) == 3
    seen: set[date] = set()
    for fold in folds:
        assert fold.test_dates[0] > fold.train_dates[-1]
        assert not seen & set(fold.test_dates)
        seen |= set(fold.test_dates)


@pytest.mark.parametrize(
    "train_days,test_days,step_days",
    [(0, 2, 2), (2, 0, 2), (2, 2, 0), (2, 2, 1)],
)
def test_walk_forward_splits_reject_invalid_parameters(train_days, test_days, step_days):
    dates = next_trading_dates(FIRST_MONDAY, 10)
    with pytest.raises(ValueError):
        walk_forward_splits(dates, train_days, test_days, step_days)


def test_walk_forward_splits_reject_non_int_parameters():
    dates = next_trading_dates(FIRST_MONDAY, 10)
    with pytest.raises(TypeError):
        walk_forward_splits(dates, 2.5, 2)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Leakage assertion
# --------------------------------------------------------------------------- #
def test_assert_no_leakage_accepts_disjoint_windows():
    dates = next_trading_dates(FIRST_MONDAY, 2)
    train = generate_session("SOXL", dates[0], seed=1, n_minutes=MINUTES_PER_SESSION)
    test = generate_session("SOXL", dates[1], seed=2, n_minutes=MINUTES_PER_SESSION)
    assert_no_leakage(train, test)  # must not raise


def test_assert_no_leakage_rejects_overlapping_timestamps():
    dates = next_trading_dates(FIRST_MONDAY, 2)
    train = generate_session("SOXL", dates[0], seed=1, n_minutes=MINUTES_PER_SESSION)
    test = generate_session("SOXL", dates[1], seed=2, n_minutes=MINUTES_PER_SESSION)
    contaminated_train = list(train) + list(test[:1])
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(contaminated_train, test)


def test_assert_no_leakage_rejects_shared_session_date():
    date_one = next_trading_dates(FIRST_MONDAY, 1)[0]
    train = generate_session("SOXL", date_one, seed=1, n_minutes=5)
    test = generate_session("SOXL", date_one, seed=2, n_minutes=5)
    # Same ET session date -> leakage even if the individual minute bars differ.
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(train, test)


# --------------------------------------------------------------------------- #
# run_walkforward
# --------------------------------------------------------------------------- #
def test_run_walkforward_structure_and_summary():
    config = load_config()
    bars = synthetic_bars(8)
    result = run_walkforward(
        bars,
        config,
        strategy_factory=lambda: [FireEveryBar()],
        train_days=3,
        test_days=2,
        step_days=2,
        risk_factory=make_risk_factory(config),
        engine_kwargs=engine_kwargs(),
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.folds) == 2
    assert result.summary["folds"] == 2
    assert result.summary["train_days"] == 3
    assert result.summary["test_days"] == 2
    assert result.summary["test_days_total"] == 4

    for fold_result in result.folds:
        assert isinstance(fold_result, FoldResult)
        assert fold_result.leakage_checked is True
        # Every test fold only sees its own test dates (fresh engine).
        assert fold_result.train_result.bars_processed == 3 * MINUTES_PER_SESSION
        assert fold_result.test_result.bars_processed == 2 * MINUTES_PER_SESSION
        assert fold_result.train_result.initial_equity == pytest.approx(3000.0)
        assert fold_result.test_result.initial_equity == pytest.approx(3000.0)
        # Fresh risk manager per run: entries never leak across folds.
        assert fold_result.train_result.trades
        assert fold_result.test_result.trades

    expected_test_trades = sum(len(fold.test_result.trades) for fold in result.folds)
    assert result.summary["total_test_trades"] == expected_test_trades
    assert result.aggregate_test_metrics.total_trades == expected_test_trades
    assert result.summary["total_test_trades"] > 0
    assert any("no compounding" in note for note in result.notes)
    assert any("DATA LIMITATION" in note for note in result.notes)

    payload = result.to_dict()
    assert payload["folds"][0]["fold"]["train_dates"]
    assert payload["summary"]["folds"] == 2
    json.dumps(payload, allow_nan=False)


def test_run_walkforward_insufficient_data_returns_incomplete_notes():
    config = load_config()
    dates = next_trading_dates(FIRST_MONDAY, 2)
    bars = generate_sessions(("SOXL",), dates, seed=3, n_minutes=MINUTES_PER_SESSION)
    result = run_walkforward(
        bars,
        config,
        strategy_factory=lambda: [FireEveryBar()],
        train_days=2,
        test_days=1,
        risk_factory=make_risk_factory(config),
        engine_kwargs=engine_kwargs(),
    )
    assert result.folds == ()
    assert result.summary["folds"] == 0
    assert result.aggregate_test_metrics.total_trades == 0
    notes = " ".join(result.notes)
    assert "WALK-FORWARD INCOMPLETE" in notes


def test_run_walkforward_test_folds_without_trades_note_incomplete():
    config = load_config()
    bars = synthetic_bars(6)
    result = run_walkforward(
        bars,
        config,
        strategy_factory=lambda: [FireNever()],
        train_days=2,
        test_days=1,
        step_days=1,
        risk_factory=make_risk_factory(config),
        engine_kwargs=engine_kwargs(),
    )
    assert result.folds
    assert result.summary["total_test_trades"] == 0
    assert any("cannot support any statistical claim" in note for note in result.notes)


def test_run_walkforward_one_argument_factory_receives_train_bars_only():
    config = load_config()
    bars = synthetic_bars(5)
    calls: list[int] = []

    def factory(train_bars):
        calls.append(len(train_bars))
        return [FireEveryBar()]

    result = run_walkforward(
        bars,
        config,
        factory,
        train_days=2,
        test_days=1,
        step_days=1,
        risk_factory=make_risk_factory(config),
        engine_kwargs=engine_kwargs(),
    )
    # 5 sessions, train=2/test=1/step=1 -> 3 folds; 2 factory calls per fold.
    assert len(result.folds) == 3
    assert len(calls) == 6
    assert all(count == 2 * MINUTES_PER_SESSION for count in calls)  # train window only


def test_run_walkforward_rejects_stateful_risk_instance():
    config = load_config()
    bars = synthetic_bars(4)
    with pytest.raises(ValueError, match="risk_factory"):
        run_walkforward(
            bars,
            config,
            strategy_factory=lambda: [FireEveryBar()],
            train_days=2,
            test_days=1,
            engine_kwargs={"risk": SimulatedRiskManager(config)},
        )


def test_run_walkforward_initial_equity_is_shared_and_not_compounded():
    config = load_config()
    bars = synthetic_bars(5)
    result = run_walkforward(
        bars,
        config,
        strategy_factory=lambda: [FireEveryBar()],
        train_days=2,
        test_days=1,
        step_days=1,
        initial_equity=5000.0,
        risk_factory=make_risk_factory(config),
        engine_kwargs=engine_kwargs(),
    )
    assert result.folds
    for fold_result in result.folds:
        assert fold_result.train_result.initial_equity == pytest.approx(5000.0)
        assert fold_result.test_result.initial_equity == pytest.approx(5000.0)
    assert result.aggregate_test_metrics.initial_equity == pytest.approx(5000.0)


def test_run_walkforward_rejects_shuffled_input():
    config = load_config()
    bars = synthetic_bars(4)
    shuffled = list(bars)
    shuffled.reverse()
    with pytest.raises(ValueError, match="sorted"):
        run_walkforward(
            shuffled,
            config,
            strategy_factory=lambda: [FireEveryBar()],
            train_days=2,
            test_days=1,
        )
