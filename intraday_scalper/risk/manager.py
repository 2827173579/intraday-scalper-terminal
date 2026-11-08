"""Risk-control state machine for the manual SOXL / SOXS scalper.

The system *never* places orders: every state transition below is driven by a
human typing ``ENTRY`` / ``EXIT`` / ``STOP`` / ``CANCEL`` / ``FLAT``.  This
manager is the single authority that decides whether a freshly generated signal
is allowed through, and it owns the hard circuit breakers:

* at most ``max_entries_per_day`` entries per ET session,
* flat when the day's realised PnL reaches ``-max_daily_net_loss``,
* consecutive-stop cooldown (2 stops -> 20 min by default),
* SOXL / SOXS mutual exclusion (one position at a time, never both),
* switch cooldown after a mode change,
* no new entries at or after 15:30 ET,
* EOD reminders at 15:50 / 15:55 ET (reminders only: they never close a
  position and the message explicitly says so).

All state rolls on the **America/New_York trading date**, not on UTC midnight.
``now`` is always injected by the caller (tests use fixed UTC instants).  The
class is internally locked and safe to share between the engine loop and the
manual command loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable

from ..core import (
    AlertEvent,
    AlertLevel,
    AppConfig,
    Mode,
    Position,
    RiskBlockReason,
    RiskPlan,
    SignalCandidate,
    SignalEventKind,
    TradeRecord,
    TradeStatus,
    et_date,
    is_trading_day,
    market_phase,
    now_utc,
    phase_allows_new_entries,
    to_et,
    to_utc,
)

#: The instrument each single-symbol mode is allowed to trade.
_MODE_SYMBOL: dict[Mode, str] = {Mode.LONG_SOXL: "SOXL", Mode.LONG_SOXS: "SOXS"}


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class EntryDecision:
    """Outcome of ``RiskManager.evaluate_entry``."""

    allowed: bool
    reason: RiskBlockReason
    plan: RiskPlan | None
    message: str

    def __bool__(self) -> bool:  # convenience: ``if decision:``
        return self.allowed


@dataclass(frozen=True, slots=True)
class RiskState:
    """Immutable snapshot of the manager (safe to hand to the dashboard)."""

    et_date: date
    entries_today: int
    consecutive_losses: int
    daily_realized_pnl: float
    daily_stopped: bool
    cooldown_until_utc: datetime | None
    switch_cooldown_until_utc: datetime | None
    position: Position | None
    mode: Mode
    eod_check_sent: bool
    eod_urgent_sent: bool
    pending_symbol: str | None = None


@dataclass(frozen=True, slots=True)
class DailyStats:
    """Per-session trade statistics (only human-confirmed closed trades)."""

    et_date: date
    trades: int
    wins: int
    losses: int
    stops: int
    pnl: float
    entries_today: int
    win_rate: float
    avg_r: float
    consecutive_losses: int = 0
    daily_stopped: bool = False


@dataclass
class _PendingSignal:
    """A candidate whose ENTRY_SIGNAL was emitted but not yet confirmed."""

    candidate: SignalCandidate
    plan: RiskPlan
    created_utc: datetime


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_utc(ts: datetime, name: str = "now") -> datetime:
    if not isinstance(ts, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(ts)!r}")
    return to_utc(ts)


class RiskManager:
    """Deterministic risk state machine.  No I/O, no clock reads except ``clock``."""

    def __init__(self, config: AppConfig, clock: Callable[[], datetime] = now_utc) -> None:
        self._config = config
        self._risk = config.risk
        self._session = config.session
        self._clock = clock
        if not callable(clock):
            raise TypeError("clock must be a callable returning a datetime")

        start = _as_utc(clock(), "clock()")
        self._lock = threading.RLock()

        # --- session-scoped state (reset by roll) -----------------------
        self._et_date: date = et_date(start)
        self._entries_today = 0
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._daily_stopped = False
        self._cooldown_until: datetime | None = None
        self._switch_cooldown_until: datetime | None = None
        self._eod_check_sent = False
        self._eod_urgent_sent = False
        self._day_trades: list[TradeRecord] = []

        # --- cross-session state ----------------------------------------
        self._position: Position | None = None
        self._pending: _PendingSignal | None = None
        self._mode: Mode = Mode(config.mode)

    # ------------------------------------------------------------------ #
    # Session rollover
    # ------------------------------------------------------------------ #
    def roll(self, now: datetime) -> date:
        """Advance to the ET trading day of *now*, resetting daily counters.

        Rolling backwards (an out-of-order manual command) is ignored: the
        manager never forgets a later day's state because an older timestamp
        showed up.  An open :class:`Position` is **not** silently dropped on
        rollover - the user must confirm the exit or ``confirm_flat()``.
        """
        moment = _as_utc(now)
        with self._lock:
            current = et_date(moment)
            if current > self._et_date:
                self._reset_day(current)
            return self._et_date

    def begin_session(self, now: datetime) -> date:
        """Force the manager onto the ET trading day of *now* (driver clock sync).

        Unlike :meth:`roll`, this also moves **backwards**.  It exists for
        drivers that own the clock — the backtest engine, which replays
        historical sessions while the manager was constructed with the real
        wall clock, and any future replay harness.  Live code must keep using
        :meth:`roll` so an out-of-order manual command cannot rewind state.
        """
        moment = _as_utc(now)
        with self._lock:
            current = et_date(moment)
            if current != self._et_date:
                self._reset_day(current)
            return self._et_date

    def _reset_day(self, new_date: date) -> None:
        self._et_date = new_date
        self._entries_today = 0
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._daily_stopped = False
        self._cooldown_until = None
        self._switch_cooldown_until = None
        self._eod_check_sent = False
        self._eod_urgent_sent = False
        self._day_trades = []
        # A signal that was never confirmed does not survive the session.
        self._pending = None

    # ------------------------------------------------------------------ #
    # Gate
    # ------------------------------------------------------------------ #
    def evaluate_entry(
        self,
        candidate: SignalCandidate,
        plan: RiskPlan | None,
        now: datetime,
        *,
        data_ready: bool,
    ) -> EntryDecision:
        """Decide whether *candidate* may be presented as a manual entry.

        The checks run in a fixed, documented order; the first failure wins.
        A successful evaluation stores the candidate as *pending*, so a second
        evaluation cannot emit a duplicate / conflicting signal until the user
        confirms it (``confirm_entry``) or cancels it (``record_cancel``).
        """
        if not isinstance(candidate, SignalCandidate):
            raise TypeError(f"candidate must be a SignalCandidate, got {type(candidate)!r}")
        moment = _as_utc(now)
        with self._lock:
            self.roll(moment)

            # 1) trading day / session window ---------------------------
            if not is_trading_day(moment):
                return self._deny(
                    RiskBlockReason.NOT_TRADING_DAY,
                    f"非交易日（{to_et(moment):%Y-%m-%d} ET），禁止新开仓",
                )
            et_time = to_et(moment).time()
            if et_time >= self._session.no_new_entries_et:
                return self._deny(
                    RiskBlockReason.NO_NEW_ENTRIES_TIME,
                    f"已到 {self._session.no_new_entries_et:%H:%M} ET，禁止新开仓（仅可管理已有持仓）",
                )
            phase = market_phase(moment)
            if not phase_allows_new_entries(phase):
                return self._deny(
                    RiskBlockReason.NO_NEW_ENTRIES_TIME,
                    f"当前时段 {phase.value} 不允许新开仓",
                )

            # 2) data quality ------------------------------------------
            if not data_ready:
                return self._deny(
                    RiskBlockReason.DATA_NOT_READY,
                    "行情数据未就绪（DATA_UNAVAILABLE/STALE），禁止开仓",
                )

            # 3) daily loss circuit breaker ----------------------------
            if self._daily_stopped or self._daily_pnl <= -self._risk.max_daily_net_loss:
                return self._deny(
                    RiskBlockReason.DAILY_LOSS_LIMIT,
                    f"当日净亏损 ${self._daily_pnl:.2f} 已达熔断线 "
                    f"${-self._risk.max_daily_net_loss:.2f}，停止开仓",
                )

            # 4) entries per day ---------------------------------------
            if self._entries_today >= self._risk.max_entries_per_day:
                return self._deny(
                    RiskBlockReason.MAX_ENTRIES_REACHED,
                    f"今日已开仓 {self._entries_today} 次，达到上限 "
                    f"{self._risk.max_entries_per_day} 次",
                )

            # 5) consecutive-stop cooldown -----------------------------
            if self._cooldown_until is not None and moment < self._cooldown_until:
                remaining = (self._cooldown_until - moment).total_seconds() / 60.0
                return self._deny(
                    RiskBlockReason.CONSECUTIVE_LOSS_COOLDOWN,
                    f"连续 {self._consecutive_losses} 笔止损后的冷却中，"
                    f"剩余 {remaining:.1f} 分钟（至 {to_et(self._cooldown_until):%H:%M:%S} ET）",
                )

            # 6) mode-switch cooldown ----------------------------------
            if (
                self._switch_cooldown_until is not None
                and moment < self._switch_cooldown_until
            ):
                remaining = (self._switch_cooldown_until - moment).total_seconds() / 60.0
                return self._deny(
                    RiskBlockReason.SWITCH_COOLDOWN,
                    f"SOXL/SOXS 切换冷却中，剩余 {remaining:.1f} 分钟"
                    f"（至 {to_et(self._switch_cooldown_until):%H:%M:%S} ET）",
                )

            # 7) mode filter + one-position mutual exclusion -----------
            allowed_symbol = _MODE_SYMBOL.get(self._mode)
            if allowed_symbol is not None and candidate.symbol != allowed_symbol:
                return self._deny(
                    RiskBlockReason.DUAL_CONFLICT,
                    f"当前模式 {self._mode.value} 只允许 {allowed_symbol}，"
                    f"忽略 {candidate.symbol} 候选",
                )
            if self._position is not None:
                if self._position.symbol == candidate.symbol:
                    return self._deny(
                        RiskBlockReason.POSITION_ALREADY_OPEN,
                        f"已持有 {self._position.symbol} {self._position.shares:g} 股，禁止加仓",
                    )
                return self._deny(
                    RiskBlockReason.DUAL_CONFLICT,
                    f"已持有 {self._position.symbol}，SOXL/SOXS 互斥，"
                    f"不能同时持有 {candidate.symbol}",
                )
            if self._pending is not None:
                if self._pending.candidate.symbol == candidate.symbol:
                    return self._deny(
                        RiskBlockReason.POSITION_ALREADY_OPEN,
                        f"已有一个待人工确认的 {candidate.symbol} 信号，"
                        f"请先 ENTRY 或 CANCEL",
                    )
                return self._deny(
                    RiskBlockReason.DUAL_CONFLICT,
                    f"已有待人工确认的 {self._pending.candidate.symbol} 信号，"
                    f"SOXL/SOXS 互斥",
                )

            # 8) reward / risk -----------------------------------------
            if candidate.reward_risk < self._risk.min_reward_risk - 1e-9:
                return self._deny(
                    RiskBlockReason.REWARD_RISK_TOO_LOW,
                    f"目标盈亏比 {candidate.reward_risk:.2f}R 低于最低要求 "
                    f"{self._risk.min_reward_risk:.2f}R",
                )

            # 9) sizing viability --------------------------------------
            if plan is None:
                return self._deny(
                    RiskBlockReason.ZERO_SHARES,
                    "没有可用的仓位计划（sizing 未返回 RiskPlan），禁止开仓",
                )
            if plan.candidate.risk_per_share <= 0:
                return self._deny(
                    RiskBlockReason.INVALID_STOP,
                    "止损价必须低于入场参考价（Long-only ETF）",
                )
            if not plan.valid:
                return self._deny(
                    RiskBlockReason.ZERO_SHARES,
                    "按风险/名义上限计算股数为 0，禁止开仓",
                )

            # --- allowed ---------------------------------------------
            self._pending = _PendingSignal(candidate, plan, moment)
            message = (
                f"允许人工开仓 {candidate.symbol} {candidate.strategy.value}: "
                f"{plan.shares:g} 股 @ ${candidate.entry_ref:.2f}"
                f"（止损 ${candidate.stop_loss:.2f}，止盈 ${candidate.take_profit_price:.2f}），"
                f"名义 ${plan.notional:,.2f}，实际风险 ${plan.actual_risk:,.2f}"
            )
            return EntryDecision(
                allowed=True,
                reason=RiskBlockReason.NONE,
                plan=plan,
                message=message,
            )

    # ------------------------------------------------------------------ #
    # Manual confirmations (the only way state changes)
    # ------------------------------------------------------------------ #
    def confirm_entry(
        self,
        plan: RiskPlan,
        *,
        fill_price: float,
        fill_utc: datetime,
        signal_id: str,
        mode: Mode | str,
    ) -> Position:
        """Register a human-confirmed entry fill (never placed by this system)."""
        moment = _as_utc(fill_utc, "fill_utc")
        with self._lock:
            self.roll(moment)
            if self._position is not None:
                raise RuntimeError(
                    f"已有持仓 {self._position.symbol}，不能重复确认开仓；"
                    f"请先记录 EXIT/STOP 或 FLAT"
                )
            if not isinstance(plan, RiskPlan):
                raise TypeError(f"plan must be a RiskPlan, got {type(plan)!r}")
            if not plan.valid:
                raise ValueError("无效的仓位计划（股数为 0），不能开仓")
            fill = float(fill_price)
            if fill <= 0:
                raise ValueError(f"fill_price must be > 0, got {fill_price!r}")
            if not signal_id:
                raise ValueError("signal_id must be a non-empty string")

            position = Position(
                signal_id=str(signal_id),
                symbol=plan.candidate.symbol,
                strategy=plan.candidate.strategy,
                mode=Mode(mode),
                shares=plan.shares,
                entry_price=fill,
                entry_utc=moment,
                stop_loss=plan.candidate.stop_loss,
                take_profit=plan.candidate.take_profit_price,
                entry_price_signal=plan.candidate.entry_ref,
            )
            self._position = position
            self._entries_today += 1
            self._pending = None
            return position

    def confirm_exit(
        self,
        *,
        fill_price: float,
        fill_utc: datetime,
        reason: str,
    ) -> TradeRecord:
        """Register a human-confirmed exit fill and apply the risk consequences."""
        moment = _as_utc(fill_utc, "fill_utc")
        with self._lock:
            self.roll(moment)
            position = self._position
            if position is None:
                raise RuntimeError("当前没有持仓，无法记录平仓")
            if moment < position.entry_utc:
                raise ValueError(
                    f"平仓时间 {moment.isoformat()} 早于开仓时间 "
                    f"{position.entry_utc.isoformat()}"
                )
            fill = float(fill_price)
            if fill <= 0:
                raise ValueError(f"fill_price must be > 0, got {fill_price!r}")

            reason_text = str(reason or "MANUAL").strip().upper() or "MANUAL"
            status = (
                TradeStatus.STOPPED
                if reason_text in {"STOP", "STOPPED"}
                else TradeStatus.CLOSED
            )
            fees = round(position.shares * float(self._risk.commission_per_share), 10)
            record = TradeRecord(
                signal_id=position.signal_id,
                symbol=position.symbol,
                strategy=position.strategy,
                mode=position.mode,
                shares=position.shares,
                entry_time_utc=position.entry_utc,
                entry_price_signal=position.entry_price_signal,
                entry_price_fill=position.entry_price,
                stop_price=position.stop_loss,
                target_price=position.take_profit,
                exit_time_utc=moment,
                exit_price_fill=fill,
                exit_reason=reason_text,
                fees=fees,
                status=status,
            )

            self._position = None
            self._day_trades.append(record)
            self._daily_pnl += record.pnl

            if record.pnl < 0:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self._risk.consecutive_stop_threshold:
                    self._cooldown_until = moment + timedelta(
                        minutes=self._risk.consecutive_stop_cooldown_minutes
                    )
            elif record.pnl > 0:
                self._consecutive_losses = 0

            if self._daily_pnl <= -self._risk.max_daily_net_loss:
                self._daily_stopped = True

            return record

    def record_stop(self, fill_price: float, fill_utc: datetime) -> TradeRecord:
        """Convenience wrapper around ``confirm_exit(reason="STOP")``."""
        return self.confirm_exit(fill_price=fill_price, fill_utc=fill_utc, reason="STOP")

    def record_cancel(self) -> bool:
        """Drop the pending (not yet confirmed) signal.  Returns True if any."""
        with self._lock:
            if self._pending is None:
                return False
            self._pending = None
            return True

    def confirm_flat(self) -> bool:
        """User asserts they are flat now.

        Administrative action only: it clears the tracked position and any
        pending signal **without** creating a trade record and **without**
        touching the daily PnL.  Returns True if something was cleared.
        """
        with self._lock:
            changed = self._position is not None or self._pending is not None
            self._position = None
            self._pending = None
            return changed

    # ------------------------------------------------------------------ #
    # Mode switching
    # ------------------------------------------------------------------ #
    def set_mode(self, new_mode: Mode | str, now: datetime) -> bool:
        """Switch the dual-track mode.

        Returns False (and changes nothing) while a position is open.  A real
        change sets ``switch_cooldown_until = now + switch_cooldown_minutes``
        and clears any stale pending signal.  Switching to the current mode is
        an idempotent success and does not start a cooldown.
        """
        moment = _as_utc(now)
        mode = Mode(new_mode)
        with self._lock:
            self.roll(moment)
            if mode == self._mode:
                return True
            if self._position is not None:
                return False
            self._mode = mode
            self._pending = None
            self._switch_cooldown_until = moment + timedelta(
                minutes=self._risk.switch_cooldown_minutes
            )
            return True

    # ------------------------------------------------------------------ #
    # End-of-day reminders (not actions!)
    # ------------------------------------------------------------------ #
    def eod_alerts(self, now: datetime) -> list[AlertEvent]:
        """Return 15:50 / 15:55 ET reminders while a position is still open.

        Reminders only: the message states that the user must close manually
        and that the system never flattens positions.  Each reminder fires at
        most once per trading day; without a position nothing is emitted.
        """
        moment = _as_utc(now)
        with self._lock:
            self.roll(moment)
            position = self._position
            if position is None:
                return []
            et_time = to_et(moment).time()
            events: list[AlertEvent] = []
            if et_time >= self._session.eod_urgent_et and not self._eod_urgent_sent:
                self._eod_urgent_sent = True
                self._eod_check_sent = True  # the 15:50 note would be stale now
                events.append(
                    self._eod_event(AlertLevel.CRITICAL, position, moment, "15:55")
                )
            elif et_time >= self._session.eod_check_et and not self._eod_check_sent:
                self._eod_check_sent = True
                events.append(
                    self._eod_event(AlertLevel.WARNING, position, moment, "15:50")
                )
            return events

    def _eod_event(
        self, level: AlertLevel, position: Position, moment: datetime, marker: str
    ) -> AlertEvent:
        prefix = "尾盘紧急提醒" if level is AlertLevel.CRITICAL else "尾盘检查提醒"
        message = (
            f"{marker} ET {prefix}（当前 {to_et(moment):%H:%M:%S} ET）："
            f"仍持有 {position.symbol} {position.shares:g} 股 @ "
            f"${position.entry_price:.2f}（止损 ${position.stop_loss:.2f}）。"
            f"需人工平仓，系统不会自动清仓。"
        )
        details: dict[str, Any] = {
            "stage": "EOD_URGENT" if level is AlertLevel.CRITICAL else "EOD_CHECK",
            "marker": marker,
            "et_date": self._et_date.isoformat(),
            "symbol": position.symbol,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
        }
        return AlertEvent(
            kind=SignalEventKind.EOD_REMINDER,
            level=level,
            message=message,
            ts_utc=moment,
            symbol=position.symbol,
            signal_id=position.signal_id,
            details=details,
        )

    # ------------------------------------------------------------------ #
    # Read models
    # ------------------------------------------------------------------ #
    def snapshot(self) -> RiskState:
        """Current state (rolls the session forward using ``clock``)."""
        with self._lock:
            self.roll(self._clock())
            return RiskState(
                et_date=self._et_date,
                entries_today=self._entries_today,
                consecutive_losses=self._consecutive_losses,
                daily_realized_pnl=self._daily_pnl,
                daily_stopped=self._daily_stopped,
                cooldown_until_utc=self._cooldown_until,
                switch_cooldown_until_utc=self._switch_cooldown_until,
                position=self._position,
                mode=self._mode,
                eod_check_sent=self._eod_check_sent,
                eod_urgent_sent=self._eod_urgent_sent,
                pending_symbol=(
                    self._pending.candidate.symbol if self._pending is not None else None
                ),
            )

    def daily_stats(self) -> DailyStats:
        """Statistics for the current ET session."""
        with self._lock:
            self.roll(self._clock())
            trades = len(self._day_trades)
            wins = sum(1 for trade in self._day_trades if trade.pnl > 0)
            losses = sum(1 for trade in self._day_trades if trade.pnl < 0)
            stops = sum(
                1 for trade in self._day_trades if trade.status is TradeStatus.STOPPED
            )
            avg_r = (
                sum(trade.r_multiple for trade in self._day_trades) / trades
                if trades
                else 0.0
            )
            return DailyStats(
                et_date=self._et_date,
                trades=trades,
                wins=wins,
                losses=losses,
                stops=stops,
                pnl=self._daily_pnl,
                entries_today=self._entries_today,
                win_rate=(wins / trades) if trades else 0.0,
                avg_r=avg_r,
                consecutive_losses=self._consecutive_losses,
                daily_stopped=self._daily_stopped,
            )

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def position(self) -> Position | None:
        return self._position

    @property
    def pending_symbol(self) -> str | None:
        return self._pending.candidate.symbol if self._pending is not None else None

    @property
    def entries_today(self) -> int:
        return self._entries_today

    @property
    def daily_realized_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_stopped(self) -> bool:
        return self._daily_stopped

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RiskManager(mode={self._mode.value}, et_date={self._et_date.isoformat()}, "
            f"entries={self._entries_today}, pnl={self._daily_pnl:.2f}, "
            f"position={self._position.symbol if self._position else None})"
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _deny(self, reason: RiskBlockReason, message: str) -> EntryDecision:
        return EntryDecision(allowed=False, reason=reason, plan=None, message=message)
