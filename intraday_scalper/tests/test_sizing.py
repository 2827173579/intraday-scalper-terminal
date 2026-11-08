"""Unit tests for :mod:`intraday_scalper.risk.sizing` (offline, deterministic)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intraday_scalper.core import RiskConfig, SignalCandidate, StrategyName, load_config
from intraday_scalper.risk import describe_plan, size_position

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 15, 0, tzinfo=UTC)  # Friday 2026-11-06, 10:00 ET


def _candidate(
    entry: float = 20.0,
    stop: float = 19.5,
    target_r: float = 2.0,
    symbol: str = "SOXL",
) -> SignalCandidate:
    return SignalCandidate(
        symbol=symbol,
        strategy=StrategyName.ORB_5M,
        entry_ref=entry,
        stop_loss=stop,
        target_r=target_r,
        rationale="sizing unit test",
        created_utc=NOW,
        confirmation_bar_start_utc=NOW - timedelta(minutes=1),
    )


@pytest.fixture
def risk(config) -> RiskConfig:
    return config.risk


# --------------------------------------------------------------------------- #
# Core formula
# --------------------------------------------------------------------------- #
def test_risk_budget_formula_20_dollars_over_50c_stop(risk):
    """$20 risk / $0.50 stop distance = 40 shares (notional still below cap)."""
    plan = size_position(_candidate(entry=20.0, stop=19.5), risk)

    assert plan.valid is True
    assert plan.shares == 40.0
    assert plan.notional == pytest.approx(800.0)
    assert plan.theoretical_risk == pytest.approx(20.0)
    assert plan.est_slippage_dollars == pytest.approx(
        40.0 * (20.0 + 19.5) * risk.est_slippage_bps / 10_000.0
    )
    assert plan.est_fees_dollars == pytest.approx(0.0)
    assert plan.actual_risk == pytest.approx(
        plan.theoretical_risk + plan.est_slippage_dollars + plan.est_fees_dollars
    )
    # The risk cap binds, the notional cap does not (40 * $20 = $800 < $1,200).
    assert plan.clipped_by_risk is True
    assert plan.clipped_by_notional is False
    assert plan.allow_fractional is False


def test_notional_cap_clips_share_count(risk):
    """A $100 entry only fits 12 shares under the $1,200 notional cap."""
    plan = size_position(_candidate(entry=100.0, stop=99.0), risk)

    assert plan.shares == 12.0
    assert plan.notional == pytest.approx(1_200.0)
    assert plan.theoretical_risk == pytest.approx(12.0)  # far below the $20 budget
    assert plan.clipped_by_notional is True
    assert plan.clipped_by_risk is False


def test_risk_cap_clips_share_count(risk):
    """A $2.00 stop distance only allows 10 shares under the $20 risk cap."""
    plan = size_position(_candidate(entry=20.0, stop=18.0), risk)

    assert plan.shares == 10.0
    assert plan.theoretical_risk == pytest.approx(20.0)
    assert plan.notional == pytest.approx(200.0)
    assert plan.clipped_by_risk is True
    assert plan.clipped_by_notional is False


def test_integer_floor_never_increases_risk(risk):
    """28.57... shares must floor to 28; risk must never exceed the budget."""
    candidate = _candidate(entry=20.0, stop=19.3)  # $0.70 risk per share
    plan = size_position(candidate, risk)

    assert plan.shares == 28.0
    assert plan.shares == float(math.floor(plan.shares))
    assert plan.theoretical_risk <= risk.max_risk_per_trade + 1e-12
    assert plan.theoretical_risk == pytest.approx(28.0 * 0.7)
    assert plan.actual_risk >= plan.theoretical_risk


def test_fractional_shares_truncated_to_precision():
    """Fractional sizing floors (never rounds up) to 5 decimal places."""
    config = load_config()
    fractional_risk = replace(
        config.risk, allow_fractional_shares=True, fractional_precision=5
    )
    plan = size_position(_candidate(entry=20.0, stop=19.3), fractional_risk)

    assert plan.allow_fractional is True
    assert plan.shares == pytest.approx(28.57142, abs=1e-9)
    assert plan.shares == math.floor(plan.shares * 1e5) / 1e5
    assert plan.theoretical_risk <= fractional_risk.max_risk_per_trade + 1e-12


def test_equity_argument_tightens_notional_cap(risk):
    """With $500 equity the notional budget shrinks from $1,200 to $500."""
    plan = size_position(_candidate(entry=20.0, stop=19.5), risk, equity=500.0)

    assert plan.shares == 25.0  # 500 / 20
    assert plan.notional == pytest.approx(500.0)
    assert plan.clipped_by_notional is True
    assert plan.clipped_by_risk is False


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_slippage_and_fees_are_added_to_actual_risk(risk):
    costly = replace(risk, est_slippage_bps=10.0, commission_per_share=0.01)
    plan = size_position(_candidate(entry=20.0, stop=19.5), costly)

    expected_slippage = 40.0 * (20.0 + 19.5) * 10.0 / 10_000.0
    expected_fees = 40.0 * 0.01
    assert plan.est_slippage_dollars == pytest.approx(expected_slippage)
    assert plan.est_fees_dollars == pytest.approx(expected_fees)
    assert plan.actual_risk == pytest.approx(
        20.0 + expected_slippage + expected_fees
    )
    assert plan.actual_risk > plan.theoretical_risk


# --------------------------------------------------------------------------- #
# Invalid / zero plans
# --------------------------------------------------------------------------- #
def test_zero_shares_plan_is_invalid(risk):
    """A $50 stop distance cannot fit even one share inside the $20 budget."""
    plan = size_position(_candidate(entry=100.0, stop=50.0), risk)

    assert plan.shares == 0.0
    assert plan.valid is False
    assert plan.notional == 0.0
    assert plan.theoretical_risk == 0.0
    assert plan.actual_risk == 0.0


def test_zero_shares_when_equity_too_small(risk):
    plan = size_position(_candidate(entry=20.0, stop=19.5), risk, equity=10.0)

    assert plan.shares == 0.0
    assert plan.valid is False


def test_non_positive_equity_produces_invalid_plan(risk):
    plan = size_position(_candidate(), risk, equity=-100.0)
    assert plan.valid is False
    assert plan.shares == 0.0


def test_non_finite_equity_raises(risk):
    with pytest.raises(ValueError):
        size_position(_candidate(), risk, equity=float("nan"))


def test_sweep_never_exceeds_budgets(risk):
    for entry, stop in [
        (10.0, 9.99),
        (25.0, 24.5),
        (40.0, 39.0),
        (80.0, 78.0),
        (150.0, 149.0),
        (20.0, 19.3),
    ]:
        plan = size_position(_candidate(entry=entry, stop=stop), risk)
        assert plan.notional <= risk.max_notional_per_trade + 1e-9
        assert plan.theoretical_risk <= risk.max_risk_per_trade + 1e-9
        assert plan.shares >= 0.0


# --------------------------------------------------------------------------- #
# describe_plan
# --------------------------------------------------------------------------- #
def test_describe_plan_contains_expected_fields(risk):
    plan = size_position(_candidate(entry=20.0, stop=19.5), risk)
    described = describe_plan(plan)

    assert set(described) == {
        "计划股数",
        "实际名义仓位",
        "理论止损风险",
        "预估滑点",
        "计入成本实际风险",
        "风险收益比",
    }
    assert described["计划股数"] == 40.0
    assert described["实际名义仓位"] == "$800.00"
    assert described["理论止损风险"] == "$20.00"
    assert described["预估滑点"] == "$0.24"
    assert described["计入成本实际风险"] == "$20.24"
    # Effective (after-cost) reward/risk: 40 * ($21 - $20) / $20.237.
    assert described["风险收益比"] == "1.98R"


def test_plan_preserves_candidate_and_target(risk):
    candidate = _candidate(entry=20.0, stop=19.5, target_r=2.0)
    plan = size_position(candidate, risk)

    assert plan.candidate is candidate
    assert plan.candidate.take_profit_price == pytest.approx(21.0)
    assert plan.reward_dollars == pytest.approx(2.0 * plan.actual_risk)
