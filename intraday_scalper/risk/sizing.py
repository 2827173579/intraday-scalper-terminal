"""Position sizing for long-only SOXL / SOXS trades.

The output is a :class:`~intraday_scalper.core.models.RiskPlan`: it contains the
planned share count, the planned notional, the theoretical stop risk and the
*real* risk once estimated slippage and fees are added.  The system never sizes
up: when both the risk budget and the notional budget allow a fractional share
count the value is floored, so rounding can only ever reduce risk.

This module is pure: no clock, no I/O, no network.  It is safe to call from the
signal engine, the backtester and unit tests.
"""

from __future__ import annotations

import math

from ..core import RiskConfig, RiskPlan, SignalCandidate


# Tolerances used only to decide which of the two budgets was the binding one.
# They are relative to the size of the caps so ordinary float noise cannot flip
# the flags.
_REL_TOL = 1e-12


def _zero_plan(candidate: SignalCandidate, risk: RiskConfig) -> RiskPlan:
    """A plan representing "do not trade": shares = 0 makes ``valid`` False."""
    return RiskPlan(
        candidate=candidate,
        shares=0.0,
        notional=0.0,
        theoretical_risk=0.0,
        est_slippage_dollars=0.0,
        est_fees_dollars=0.0,
        actual_risk=0.0,
        clipped_by_notional=False,
        clipped_by_risk=False,
        allow_fractional=bool(risk.allow_fractional_shares),
    )


def _truncate_fractional(shares: float, precision: int) -> float:
    """Floor *shares* to *precision* decimal places (never round up)."""
    if precision <= 0:
        return float(math.floor(shares))
    scale = 10 ** int(precision)
    return math.floor(shares * scale) / scale


def size_position(
    candidate: SignalCandidate,
    risk: RiskConfig,
    *,
    equity: float | None = None,
) -> RiskPlan:
    """Size one trade from the $ risk budget and the notional budget.

    Parameters
    ----------
    candidate:
        A long-only :class:`SignalCandidate` (``stop_loss < entry_ref`` is
        guaranteed by its constructor).
    risk:
        The :class:`RiskConfig` with ``max_risk_per_trade``,
        ``max_notional_per_trade`` and the fractional-share switch.
    equity:
        Optional account equity.  When supplied it *tightens* both budgets:
        ``max_risk = min(max_risk_per_trade, equity)`` and
        ``max_notional = min(max_notional_per_trade, equity)`` so the account
        can never be over-committed.  ``None`` (default) keeps the configured
        budgets untouched.

    Returns
    -------
    RiskPlan
        ``plan.shares == 0`` (``plan.valid is False``) when the caps do not
        allow even one share; callers must not trade such a plan.
    """
    entry = candidate.entry_ref
    stop = candidate.stop_loss
    risk_per_share = candidate.risk_per_share  # entry - stop, > 0 by contract

    max_risk = float(risk.max_risk_per_trade)
    max_notional = float(risk.max_notional_per_trade)

    if equity is not None:
        equity_value = float(equity)
        if not math.isfinite(equity_value):
            raise ValueError(f"equity must be finite, got {equity!r}")
        if equity_value <= 0:
            # No capital -> no position.  Avoid a division by zero below.
            return _zero_plan(candidate, risk)
        max_risk = min(max_risk, equity_value)
        max_notional = min(max_notional, equity_value)

    if risk_per_share <= 0 or entry <= 0:  # pragma: no cover - contract guard
        return _zero_plan(candidate, risk)

    risk_qty = max_risk / risk_per_share
    notional_qty = max_notional / entry
    raw_shares = min(risk_qty, notional_qty)

    if risk.allow_fractional_shares:
        shares = _truncate_fractional(raw_shares, risk.fractional_precision)
    else:
        shares = float(math.floor(raw_shares))

    # Floating point can never turn a non-negative size negative, but clamp for
    # clarity so downstream arithmetic never sees "-0.0" or a tiny negative.
    shares = max(shares, 0.0)

    # Which budget was binding?  Mark only a *strict* winner; an exact tie means
    # both budgets bind simultaneously and neither "clipped" the other.
    tolerance = _REL_TOL * max(1.0, abs(risk_qty), abs(notional_qty))
    clipped_by_risk = risk_qty < notional_qty - tolerance
    clipped_by_notional = notional_qty < risk_qty - tolerance

    notional = shares * entry
    theoretical_risk = shares * risk_per_share
    est_slippage_dollars = shares * (entry + stop) * float(risk.est_slippage_bps) / 10_000.0
    est_fees_dollars = shares * float(risk.commission_per_share)
    actual_risk = theoretical_risk + est_slippage_dollars + est_fees_dollars

    return RiskPlan(
        candidate=candidate,
        shares=shares,
        notional=notional,
        theoretical_risk=theoretical_risk,
        est_slippage_dollars=est_slippage_dollars,
        est_fees_dollars=est_fees_dollars,
        actual_risk=actual_risk,
        clipped_by_notional=clipped_by_notional,
        clipped_by_risk=clipped_by_risk,
        allow_fractional=bool(risk.allow_fractional_shares),
    )


def describe_plan(plan: RiskPlan) -> dict[str, str | float]:
    """Human-readable summary of a :class:`RiskPlan` for the dashboard.

    The reward/risk value is computed *after costs*::

        effective_rr = shares * (take_profit - entry_ref) / actual_risk

    which is slightly below the strategy's nominal ``target_r`` because
    slippage and fees are charged against the plan.
    """
    effective_rr = 0.0
    if plan.actual_risk > 0:
        reward_dollars = plan.shares * (
            plan.candidate.take_profit_price - plan.candidate.entry_ref
        )
        effective_rr = reward_dollars / plan.actual_risk
    return {
        "计划股数": float(plan.shares),
        "实际名义仓位": f"${plan.notional:,.2f}",
        "理论止损风险": f"${plan.theoretical_risk:,.2f}",
        "预估滑点": f"${plan.est_slippage_dollars:,.2f}",
        "计入成本实际风险": f"${plan.actual_risk:,.2f}",
        "风险收益比": f"{effective_rr:.2f}R",
    }
