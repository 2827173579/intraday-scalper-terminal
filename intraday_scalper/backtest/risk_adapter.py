"""Risk adapters for the event-driven backtester.

Two responsibilities live here:

1. **Fallback position sizing.**  ``size_position_equivalent`` mirrors the
   frozen sizing contract from :mod:`intraday_scalper.risk.sizing` so the
   backtester can size trades before builder C's module exists.  When the
   production module is importable :func:`resolve_sizing_fn` prefers it.

2. **Equivalent risk state machine.**  ``SimulatedRiskManager`` implements the
   same public interface as ``intraday_scalper.risk.manager.RiskManager``
   (``roll`` / ``evaluate_entry`` / ``confirm_entry`` / ``confirm_exit`` /
   ``record_stop`` / ``set_mode`` / ``eod_alerts`` / ``snapshot`` /
   ``daily_stats``) using only :mod:`intraday_scalper.core`.  It exists so the
   backtester is runnable and deterministic before C's module lands; tests
   inject it explicitly.  ``resolve_risk_manager`` prefers the production class
   when it is importable, so a finished project uses the real risk manager.

Hard rules enforced here (never re-implemented inside the engine):

* no entry on non-trading days / outside the regular session / after
  ``session.no_new_entries_et`` (15:30 ET by default);
* ``data_ready`` gate;
* daily net-loss circuit breaker (``max_daily_net_loss``);
* max entries per day;
* consecutive-stop cooldown;
* SOXL <-> SOXS switch cooldown;
* at most one open position at a time (the two ETFs are mutually exclusive);
* minimum reward/risk and non-zero share count.

This module never places orders and never talks to a broker.  It is pure
in-memory state, deterministic when driven by an injected clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from intraday_scalper.core import (
    AlertEvent,
    AlertLevel,
    AppConfig,
    Mode,
    Position,
    RiskBlockReason,
    RiskConfig,
    RiskPlan,
    SignalCandidate,
    SignalEventKind,
    TradeRecord,
    TradeStatus,
)
from intraday_scalper.core.timeutil import (
    et_date,
    is_regular_session,
    is_trading_day,
    now_utc,
    to_et,
    to_utc,
)

__all__ = [
    "DailyStats",
    "EntryDecision",
    "RiskState",
    "SimulatedRiskManager",
    "resolve_risk_manager",
    "resolve_sizing_fn",
    "risk_manager_is_simulated",
    "size_position_equivalent",
]


# --------------------------------------------------------------------------- #
# Position sizing fallback
# --------------------------------------------------------------------------- #
def _zero_plan(candidate: SignalCandidate, risk: RiskConfig) -> RiskPlan:
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
    if precision <= 0:
        return float(math.floor(shares))
    scale = 10 ** int(precision)
    return math.floor(shares * scale) / scale


def size_position_equivalent(
    candidate: SignalCandidate,
    risk: RiskConfig,
    *,
    equity: float | None = None,
) -> RiskPlan:
    """Fallback sizing: ``shares = floor(min(risk/stop_distance, notional/price))``.

    Never rounds up (rounding may only reduce risk).  ``shares == 0`` makes the
    plan invalid.  Mirrors :func:`intraday_scalper.risk.sizing.size_position`.
    """

    entry = float(candidate.entry_ref)
    risk_per_share = float(candidate.risk_per_share)

    max_risk = float(risk.max_risk_per_trade)
    max_notional = float(risk.max_notional_per_trade)
    if equity is not None:
        equity_value = float(equity)
        if not math.isfinite(equity_value):
            raise ValueError(f"equity must be finite, got {equity!r}")
        if equity_value <= 0:
            return _zero_plan(candidate, risk)
        max_risk = min(max_risk, equity_value)
        max_notional = min(max_notional, equity_value)

    if risk_per_share <= 0 or entry <= 0:  # pragma: no cover - core guarantees this
        return _zero_plan(candidate, risk)

    risk_qty = max_risk / risk_per_share
    notional_qty = max_notional / entry
    raw_shares = min(risk_qty, notional_qty)

    if risk.allow_fractional_shares:
        shares = _truncate_fractional(raw_shares, risk.fractional_precision)
    else:
        shares = float(math.floor(raw_shares))
    shares = max(shares, 0.0)

    tolerance = 1e-12 * max(1.0, abs(risk_qty), abs(notional_qty))
    clipped_by_risk = risk_qty < notional_qty - tolerance
    clipped_by_notional = notional_qty < risk_qty - tolerance

    notional = shares * entry
    theoretical_risk = shares * risk_per_share
    est_slippage_dollars = shares * (entry + float(candidate.stop_loss)) * float(risk.est_slippage_bps) / 10_000.0
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


_SIZING_CACHE: Any = None


def resolve_sizing_fn() -> Any:
    """Prefer the production sizing function, fall back to the equivalent."""
    global _SIZING_CACHE
    if _SIZING_CACHE is not None:
        return _SIZING_CACHE
    try:  # pragma: no cover - exercised only when risk.* is importable
        from intraday_scalper.risk.sizing import size_position as production

        if callable(production):
            _SIZING_CACHE = production
            return production
    except Exception:  # noqa: BLE001 - optional integration must never crash
        pass
    _SIZING_CACHE = size_position_equivalent
    return size_position_equivalent


# --------------------------------------------------------------------------- #
# Equivalent risk state machine (frozen interface with risk.manager.RiskManager)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EntryDecision:
    """Result of :meth:`SimulatedRiskManager.evaluate_entry`."""

    allowed: bool
    reason: RiskBlockReason = RiskBlockReason.NONE
    plan: Optional[RiskPlan] = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class RiskState:
    """Snapshot of the risk state machine (read-only)."""

    et_date: Optional[date] = None
    entries_today: int = 0
    consecutive_losses: int = 0
    daily_realized_pnl: float = 0.0
    daily_stopped: bool = False
    cooldown_until_utc: Optional[datetime] = None
    switch_cooldown_until_utc: Optional[datetime] = None
    position: Optional[Position] = None
    mode: Mode = Mode.DUAL_WATCH
    eod_check_sent: bool = False
    eod_urgent_sent: bool = False
    last_closed_symbol: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DailyStats:
    """Per-session counters used by the dashboard / report."""

    et_date: Optional[date] = None
    trades: int = 0
    wins: int = 0
    stops: int = 0
    pnl: float = 0.0
    entries_today: int = 0


class SimulatedRiskManager:
    """In-memory equivalent of the production ``RiskManager``.

    The state rolls over at the America/New_York calendar date (``roll``).
    Entries are counted when an entry is *confirmed* (filled), matching the
    production contract where the user confirms a manual fill.
    """

    def __init__(self, config: AppConfig, clock: Any = now_utc) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self._config = config
        self._clock = clock
        self._mode: Mode = Mode(config.mode)
        self._et_date: Optional[date] = None
        self._entries_today = 0
        self._consecutive_losses = 0
        self._daily_realized_pnl = 0.0
        self._daily_stopped = False
        self._cooldown_until: Optional[datetime] = None
        self._switch_cooldown_until: Optional[datetime] = None
        self._last_closed_symbol: Optional[str] = None
        self._position: Optional[Position] = None
        self._eod_check_sent = False
        self._eod_urgent_sent = False
        self._closed_trades: list[TradeRecord] = []

    # -- helpers ---------------------------------------------------------- #
    def _block(self, reason: RiskBlockReason, message: str, plan: RiskPlan | None = None) -> EntryDecision:
        return EntryDecision(allowed=False, reason=reason, plan=plan, message=message)

    def _et_date_of(self, ts: datetime) -> date:
        return et_date(ts)

    # -- session roll ----------------------------------------------------- #
    def begin_session(self, now: datetime) -> None:
        """Force the simulated manager onto the ET date of *now* (backtest clock)."""
        self.roll(now)

    def roll(self, now: datetime) -> None:
        """Reset the per-session counters when the ET date changes."""
        current_date = self._et_date_of(now)
        if self._et_date == current_date:
            return
        self._et_date = current_date
        self._entries_today = 0
        self._consecutive_losses = 0
        self._daily_realized_pnl = 0.0
        self._daily_stopped = False
        self._cooldown_until = None
        self._switch_cooldown_until = None
        self._last_closed_symbol = None
        self._eod_check_sent = False
        self._eod_urgent_sent = False
        self._closed_trades = []

    # -- queries ---------------------------------------------------------- #
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def position(self) -> Optional[Position]:
        return self._position

    def snapshot(self) -> RiskState:
        return RiskState(
            et_date=self._et_date,
            entries_today=self._entries_today,
            consecutive_losses=self._consecutive_losses,
            daily_realized_pnl=self._daily_realized_pnl,
            daily_stopped=self._daily_stopped,
            cooldown_until_utc=self._cooldown_until,
            switch_cooldown_until_utc=self._switch_cooldown_until,
            position=self._position,
            mode=self._mode,
            eod_check_sent=self._eod_check_sent,
            eod_urgent_sent=self._eod_urgent_sent,
            last_closed_symbol=self._last_closed_symbol,
        )

    def daily_stats(self) -> DailyStats:
        trades = self._closed_trades
        return DailyStats(
            et_date=self._et_date,
            trades=len(trades),
            wins=sum(1 for trade in trades if trade.pnl > 0),
            stops=sum(1 for trade in trades if trade.exit_reason == "STOP"),
            pnl=sum(trade.pnl for trade in trades),
            entries_today=self._entries_today,
        )

    # -- entry gate ------------------------------------------------------- #
    def evaluate_entry(
        self,
        candidate: SignalCandidate,
        plan: RiskPlan | None,
        now: datetime,
        *,
        data_ready: bool = True,
    ) -> EntryDecision:
        """Apply the hard risk rules in their documented order."""
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        self.roll(now)

        risk = self._config.risk
        session = self._config.session

        # 1) trading day / session / 15:30 ET cut-off
        if not is_trading_day(now):
            return self._block(RiskBlockReason.NOT_TRADING_DAY, "非交易日，禁止开仓", plan)
        if not is_regular_session(now):
            return self._block(RiskBlockReason.NO_NEW_ENTRIES_TIME, "不在常规交易时段，禁止开仓", plan)
        if to_et(now).time() >= session.no_new_entries_et:
            return self._block(
                RiskBlockReason.NO_NEW_ENTRIES_TIME,
                f"已过 {session.no_new_entries_et:%H:%M} ET，禁止新开仓",
                plan,
            )

        # 2) data quality gate
        if not data_ready:
            return self._block(RiskBlockReason.DATA_NOT_READY, "数据未就绪（DATA_UNAVAILABLE），禁止开仓", plan)

        # 3) daily circuit breaker
        if self._daily_stopped:
            return self._block(
                RiskBlockReason.DAILY_LOSS_LIMIT,
                f"单日净亏损熔断（${self._daily_realized_pnl:.2f} <= -${risk.max_daily_net_loss:.2f}）",
                plan,
            )

        # 4) max entries
        if self._entries_today >= risk.max_entries_per_day:
            return self._block(
                RiskBlockReason.MAX_ENTRIES_REACHED,
                f"今日开仓次数已达上限 {risk.max_entries_per_day}",
                plan,
            )

        # 5) consecutive-stop cooldown
        if self._cooldown_until is not None and now < self._cooldown_until:
            return self._block(
                RiskBlockReason.CONSECUTIVE_LOSS_COOLDOWN,
                f"连续止损冷却中，至 {self._cooldown_until.isoformat()}",
                plan,
            )

        # 6) SOXL <-> SOXS switch cooldown
        if (
            self._last_closed_symbol is not None
            and candidate.symbol != self._last_closed_symbol
            and self._switch_cooldown_until is not None
            and now < self._switch_cooldown_until
        ):
            return self._block(
                RiskBlockReason.SWITCH_COOLDOWN,
                f"切换标的冷却中，至 {self._switch_cooldown_until.isoformat()}",
                plan,
            )

        # 7) mutual exclusion: at most one open position
        if self._position is not None:
            reason = (
                RiskBlockReason.POSITION_ALREADY_OPEN
                if self._position.symbol == candidate.symbol
                else RiskBlockReason.DUAL_CONFLICT
            )
            return self._block(reason, "已有持仓（SOXL/SOXS 同一时间只允许一个持仓）", plan)

        # 8) mode filter (single-track modes only allow their ETF)
        if self._mode == Mode.LONG_SOXL and candidate.symbol != "SOXL":
            return self._block(RiskBlockReason.DUAL_CONFLICT, "当前模式 long_soxl，只允许 SOXL", plan)
        if self._mode == Mode.LONG_SOXS and candidate.symbol != "SOXS":
            return self._block(RiskBlockReason.DUAL_CONFLICT, "当前模式 long_soxs，只允许 SOXS", plan)

        # 9) reward/risk floor
        if candidate.target_r + 1e-9 < risk.min_reward_risk:
            return self._block(
                RiskBlockReason.REWARD_RISK_TOO_LOW,
                f"R:R {candidate.target_r:.2f} < 最低要求 {risk.min_reward_risk:.2f}",
                plan,
            )

        # 10) plan must be tradable
        if plan is None or not plan.valid:
            return self._block(RiskBlockReason.ZERO_SHARES, "计划股数为 0（资金/名义上限不足）", plan)

        return EntryDecision(allowed=True, reason=RiskBlockReason.NONE, plan=plan, message="允许开仓")

    # -- fills ------------------------------------------------------------ #
    def confirm_entry(
        self,
        plan: RiskPlan,
        *,
        fill_price: float,
        fill_utc: datetime,
        signal_id: str,
        mode: Mode | None = None,
    ) -> Position:
        """Record a *manually confirmed* fill and open the position."""
        if self._position is not None:
            raise RuntimeError("already have an open position; one position at a time")
        if not plan.valid:
            raise ValueError("cannot confirm an entry with a non-tradable plan")
        fill_utc = to_utc(fill_utc)
        self.roll(fill_utc)
        active_mode = Mode(mode) if mode is not None else self._mode
        self._entries_today += 1
        position = Position(
            signal_id=str(signal_id),
            symbol=plan.candidate.symbol,
            strategy=plan.candidate.strategy,
            mode=active_mode,
            shares=plan.shares,
            entry_price=float(fill_price),
            entry_utc=to_utc(fill_utc),
            stop_loss=plan.candidate.stop_loss,
            take_profit=plan.candidate.take_profit_price,
            entry_price_signal=plan.candidate.entry_ref,
        )
        self._position = position
        return position

    def confirm_exit(
        self,
        *,
        fill_price: float,
        fill_utc: datetime,
        reason: str = "EXIT",
        fees: float = 0.0,
        shares: float | None = None,
    ) -> TradeRecord:
        """Record a *manually confirmed* exit, update P&L and cooldowns."""
        position = self._position
        if position is None:
            raise RuntimeError("confirm_exit called with no open position")
        fill_utc = to_utc(fill_utc)
        self.roll(fill_utc)
        realized_shares = position.shares if shares is None else float(shares)
        record = TradeRecord(
            signal_id=position.signal_id,
            symbol=position.symbol,
            strategy=position.strategy,
            mode=position.mode,
            shares=realized_shares,
            entry_time_utc=position.entry_utc,
            entry_price_signal=position.entry_price_signal,
            entry_price_fill=position.entry_price,
            stop_price=position.stop_loss,
            target_price=position.take_profit,
            exit_time_utc=fill_utc,
            exit_price_fill=float(fill_price),
            exit_reason=str(reason),
            fees=float(fees),
            status=TradeStatus.STOPPED if str(reason).upper() == "STOP" else TradeStatus.CLOSED,
        )
        risk = self._config.risk
        pnl = record.pnl
        self._daily_realized_pnl += pnl
        self._closed_trades.append(record)
        self._position = None

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        if (
            self._consecutive_losses >= risk.consecutive_stop_threshold
            and risk.consecutive_stop_cooldown_minutes > 0
        ):
            self._cooldown_until = fill_utc + timedelta(minutes=risk.consecutive_stop_cooldown_minutes)
        if self._daily_realized_pnl <= -float(risk.max_daily_net_loss) + 1e-9:
            self._daily_stopped = True
        if risk.switch_cooldown_minutes > 0:
            self._switch_cooldown_until = fill_utc + timedelta(minutes=risk.switch_cooldown_minutes)
        self._last_closed_symbol = position.symbol
        return record

    def record_stop(self, fill_price: float, fill_utc: datetime) -> TradeRecord:
        """Convenience wrapper for a stop-loss exit."""
        return self.confirm_exit(fill_price=fill_price, fill_utc=fill_utc, reason="STOP")

    def record_cancel(self) -> None:
        """Cancel a pending signal: this equivalent manager has no pending state."""

    def confirm_flat(self) -> None:
        """The user confirms flat (no open position) — clears any stale position."""
        self._position = None

    # -- mode ------------------------------------------------------------- #
    def set_mode(self, new_mode: Mode | str, now: datetime) -> bool:
        """Switch the dual-track mode; switching requires a flat book + cooldown.

        Returns ``True`` when the mode was applied.  Re-selecting the current
        mode is a no-op success; switching while a position is open fails.
        """
        target = Mode(new_mode)
        if target == self._mode:
            return True
        if self._position is not None:
            return False
        switch_until = self._switch_cooldown_until
        self._mode = target
        if self._config.risk.switch_cooldown_minutes > 0:
            self._switch_cooldown_until = to_utc(now) + timedelta(
                minutes=self._config.risk.switch_cooldown_minutes
            )
        else:  # pragma: no cover - config guard
            self._switch_cooldown_until = switch_until
        return True

    # -- end of day reminders (reminders only; never auto-liquidates) ----- #
    def eod_alerts(self, now: datetime) -> list[AlertEvent]:
        """15:50 / 15:55 ET reminders while a position is open (once each)."""
        self.roll(now)
        if self._position is None:
            return []
        local = to_et(now).time()
        session = self._config.session
        events: list[AlertEvent] = []
        if local >= session.eod_urgent_et and not self._eod_urgent_sent:
            self._eod_urgent_sent = True
            events.append(
                AlertEvent(
                    kind=SignalEventKind.EOD_REMINDER,
                    level=AlertLevel.CRITICAL,
                    message="15:55 ET：立即检查持仓，需人工平仓；系统不会自动清仓。",
                    ts_utc=to_utc(now),
                    symbol=self._position.symbol,
                    signal_id=self._position.signal_id,
                    details={"stage": "EOD_URGENT", "manual_action_required": True},
                )
            )
        elif local >= session.eod_check_et and not self._eod_check_sent:
            self._eod_check_sent = True
            events.append(
                AlertEvent(
                    kind=SignalEventKind.EOD_REMINDER,
                    level=AlertLevel.WARNING,
                    message="15:50 ET：请检查持仓并准备人工平仓；系统不会自动清仓。",
                    ts_utc=to_utc(now),
                    symbol=self._position.symbol,
                    signal_id=self._position.signal_id,
                    details={"stage": "EOD_CHECK", "manual_action_required": True},
                )
            )
        return events


# --------------------------------------------------------------------------- #
# Production / fallback resolver
# --------------------------------------------------------------------------- #
_RISK_REQUIRED_METHODS = ("roll", "evaluate_entry", "confirm_entry", "confirm_exit", "snapshot")


def resolve_risk_manager(config: AppConfig, *, clock: Any = now_utc) -> Any:
    """Prefer ``risk.manager.RiskManager``; fall back to the equivalent manager.

    The production class is only used when it imports *and* exposes the frozen
    public interface.  Any failure degrades gracefully so the backtester (and
    the Lead's report script) still runs.
    """

    try:  # pragma: no cover - exercised only when risk.* is importable
        from intraday_scalper.risk.manager import RiskManager

        instance = RiskManager(config, clock=clock)
        if all(callable(getattr(instance, name, None)) for name in _RISK_REQUIRED_METHODS):
            return instance
    except Exception:  # noqa: BLE001 - optional integration must never crash
        pass
    return SimulatedRiskManager(config, clock=clock)


def risk_manager_is_simulated(risk: Any) -> bool:
    """True when *risk* is the built-in equivalent (used for result notes)."""

    return isinstance(risk, SimulatedRiskManager)
