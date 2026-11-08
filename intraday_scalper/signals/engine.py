"""Signal engine state machine — "system only alerts, a human trades".

Design rules (see ``intraday_scalper/CLAUDE.md`` and the shared task board):

* The engine consumes **completed 1-minute bars** only.  An incomplete bar or a
  bar outside the regular session never reaches a strategy.
* Strategies are deterministic functions of the past: the context built here
  contains only completed bars plus indicators computed from them.
* The engine **never places, fills or cancels an order**.  A signal is an
  :class:`AlertEvent` plus a journal row; a position becomes "open" only after
  ``risk.confirm_entry`` is called manually by the operator (through the journal
  command layer).  Every hint message says so explicitly.
* Data quality gates signals: any non-``OK`` :class:`DataStatus` blocks new
  entries and raises a ``DATA_UNAVAILABLE`` alert (rate limited to one per
  second).
* The engine is dependency injected: strategies (builder B), risk manager
  (builder C), journal (builder C), data-quality monitor (builder A) and the
  session-VWAP callback all arrive through the constructor, so it can be tested
  fully offline with fakes and later wired to the real modules unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.config import AppConfig
from ..core.models import (
    AlertEvent,
    AlertLevel,
    Bar,
    ConnectionState,
    DataStatus,
    Mode,
    Position,
    RiskBlockReason,
    RiskPlan,
    SessionPhase,
    SignalCandidate,
    SignalEventKind,
    StrategyName,
    new_signal_id,
)
from ..core.timeutil import (
    is_regular_session,
    market_phase,
    now_utc,
    session_close_utc,
    session_open_utc,
)
from . import compat
from .state import EngineSnapshot, PositionSnapshot, SymbolSnapshot

__all__ = ["SignalEngine"]

EventCallback = Callable[[AlertEvent], None]
Clock = Callable[[], datetime]
VwapProvider = Callable[[str, datetime], Optional[float]]

#: Minimum spacing between repeated ``DATA_UNAVAILABLE`` alerts.
DATA_ALERT_MIN_INTERVAL_SECONDS = 1.0

#: Upper bound on remembered ``(symbol, strategy, confirmation bar)`` keys.
_SIGNAL_KEY_LIMIT = 4_000


class SignalEngine:
    """Event-driven SOXL/SOXS signal engine (dual-track state machine)."""

    def __init__(
        self,
        config: AppConfig,
        strategies: Sequence[Any],
        risk: Any,
        *,
        journal: Any = None,
        data_quality: Any = None,
        on_event: Optional[EventCallback] = None,
        clock: Clock = now_utc,
        max_bars_1m: int = 400,
        max_bars_5m: int = 120,
        current_session_vwap: Optional[VwapProvider] = None,
        sizing: Optional[Callable[..., RiskPlan]] = None,
    ) -> None:
        if config is None:
            raise ValueError("SignalEngine requires an AppConfig")
        self._config = config
        self._strategies: Tuple[Any, ...] = tuple(strategies or ())
        self._risk = risk
        self._journal = journal
        self._data_quality = data_quality
        self._on_event = on_event
        self._clock: Clock = clock
        self._max_bars_1m = max(2, int(max_bars_1m))
        self._max_bars_5m = max(2, int(max_bars_5m))
        self._vwap_provider = current_session_vwap
        self._sizing = sizing

        self._symbols: Tuple[str, ...] = tuple(str(s).upper() for s in config.data.symbols)
        self._bars_1m: Dict[str, Deque[Bar]] = {
            symbol: deque(maxlen=self._max_bars_1m) for symbol in self._symbols
        }
        self._bars_5m: Dict[str, Deque[Bar]] = {
            symbol: deque(maxlen=self._max_bars_5m) for symbol in self._symbols
        }
        self._last_price: Dict[str, float] = {}
        self._metrics: Dict[str, Dict[str, Any]] = {}

        self._data_status: DataStatus = DataStatus.OK
        self._connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self._last_data_alert_ts: Optional[datetime] = None

        self._events: Deque[AlertEvent] = deque(maxlen=200)
        self._pending_events: List[AlertEvent] = []

        self._latest_signal: Optional[SignalCandidate] = None
        self._latest_plan: Optional[RiskPlan] = None
        self._latest_signal_id: Optional[str] = None

        self._signal_keys: set[Tuple[str, str, datetime]] = set()
        self._hint_keys: Dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API — market data
    # ------------------------------------------------------------------ #
    def on_bar(self, bar: Bar) -> None:
        """Handle one completed 1-minute regular-session :class:`Bar`."""
        if not isinstance(bar, Bar):
            raise TypeError(f"on_bar expects a core.Bar, got {type(bar)!r}")
        if not bar.is_complete:
            return  # never trade an unfinished bar
        if bar.timeframe_minutes != 1:
            return
        if not is_regular_session(bar.start_utc):
            return
        symbol = bar.symbol
        if symbol not in self._bars_1m:
            return  # only configured symbols

        window = self._bars_1m[symbol]
        if window and bar.start_utc < window[-1].start_utc:
            return  # out-of-order print; drop it rather than corrupt history
        if window and bar.start_utc == window[-1].start_utc:
            window.pop()  # duplicate delivery: replace with the latest copy
        window.append(bar)
        self._last_price[symbol] = float(bar.close)

        now = self._clock()
        self._refresh_data_quality(now)
        self._handle_eod_alerts(now)
        if self._data_status is not DataStatus.OK:
            return  # data not trusted: no signals, no position hints
        self._handle_position_hints(now)
        self._evaluate_candidates(triggering_symbol=symbol, now=now)

    def on_5m_bar(self, bar: Bar) -> None:
        """Handle one completed 5-minute bar (trend cache for contexts/dashboard)."""
        if not isinstance(bar, Bar):
            raise TypeError(f"on_5m_bar expects a core.Bar, got {type(bar)!r}")
        if not bar.is_complete or bar.timeframe_minutes != 5:
            return
        if not is_regular_session(bar.start_utc):
            return
        symbol = bar.symbol
        if symbol not in self._bars_5m:
            return
        window = self._bars_5m[symbol]
        if window and bar.start_utc < window[-1].start_utc:
            return
        if window and bar.start_utc == window[-1].start_utc:
            window.pop()
        window.append(bar)

    # ------------------------------------------------------------------ #
    # Public API — data quality / connection
    # ------------------------------------------------------------------ #
    def set_data_status(self, status: DataStatus | str, reason: str = "") -> None:
        """Explicitly override the data status (used by the main loop)."""
        now = self._clock()
        try:
            data_status = DataStatus(status)
        except ValueError:
            data_status = DataStatus.UNAVAILABLE
            reason = reason or f"unknown status {status!r}"
        self._apply_data_status(data_status, reason, now)

    def set_connection_state(self, state: ConnectionState | str) -> None:
        """Update the websocket state shown on the dashboard (no signal impact)."""
        try:
            self._connection_state = ConnectionState(state)
        except ValueError:
            self._connection_state = ConnectionState.DISCONNECTED

    @property
    def data_status(self) -> DataStatus:
        return self._data_status

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    @property
    def mode(self) -> Mode:
        return self._current_mode()

    # ------------------------------------------------------------------ #
    # Public API — events / snapshots
    # ------------------------------------------------------------------ #
    def drain_events(self) -> List[AlertEvent]:
        """Return (and clear) the events accumulated since the last drain."""
        drained = list(self._pending_events)
        self._pending_events.clear()
        return drained

    def state_snapshot(self) -> EngineSnapshot:
        """Build the immutable snapshot consumed by the Rich dashboard."""
        now = self._clock()
        mode = self._current_mode()
        phase = market_phase(now)
        risk_state = self._risk_state()
        position = self._build_position_snapshot(risk_state)
        stats = self._daily_stats()

        symbols: Dict[str, SymbolSnapshot] = {}
        for symbol in self._symbols:
            symbols[symbol] = self._build_symbol_snapshot(symbol, now)

        entries_today = self._risk_int(risk_state, "entries_today", stats.get("entries_today", 0))
        daily_pnl = self._risk_float(
            risk_state, ("daily_realized_pnl", "daily_pnl"), float(stats.get("pnl", 0.0) or 0.0)
        )
        daily_stopped = bool(
            getattr(risk_state, "daily_stopped", False) or stats.get("daily_stopped", False)
        )

        return EngineSnapshot(
            as_of_utc=now,
            mode=mode,
            session_phase=phase,
            symbols=symbols,
            data_status=self._data_status,
            connection_state=self._connection_state,
            position=position,
            risk=compat.to_jsonable(risk_state) if risk_state is not None else {},
            entries_today=entries_today,
            max_entries_per_day=int(self._config.risk.max_entries_per_day),
            trades_today=int(stats.get("trades", 0) or 0),
            wins_today=int(stats.get("wins", 0) or 0),
            stops_today=int(stats.get("stops", 0) or 0),
            daily_pnl=daily_pnl,
            daily_stopped=daily_stopped,
            latest_signal=self._latest_signal,
            latest_plan=self._latest_plan,
            latest_signal_id=self._latest_signal_id,
            strategy_names=self._strategy_names(),
            events=tuple(list(self._events)[-20:]),
            eod_close_utc=session_close_utc(now),
            breakeven_hint_r=float(self._config.risk.breakeven_hint_r),
            take_profit_hint_r=tuple(self._config.risk.take_profit_hint_r),
        )

    # ------------------------------------------------------------------ #
    # Internal — status / alerts
    # ------------------------------------------------------------------ #
    def _refresh_data_quality(self, now: datetime) -> None:
        if self._data_quality is None:
            return
        status = compat.data_quality_status(self._data_quality, now)
        if status is None or status is self._data_status:
            return
        self._apply_data_status(status, f"数据质量监控判定为 {status.value}", now)

    def _apply_data_status(self, status: DataStatus, reason: str, now: datetime) -> None:
        previous = self._data_status
        self._data_status = status
        if status is DataStatus.OK:
            if previous is not DataStatus.OK:
                self._last_data_alert_ts = now
                self._emit(
                    SignalEventKind.INFO,
                    AlertLevel.INFO,
                    f"数据状态恢复为 OK（此前 {previous.value}）：{reason or '数据流正常，恢复信号评估'}",
                    ts=now,
                )
            return

        last = self._last_data_alert_ts
        if last is not None and (now - last).total_seconds() < DATA_ALERT_MIN_INTERVAL_SECONDS:
            return  # rate limit: at most one DATA_UNAVAILABLE alert per second
        self._last_data_alert_ts = now
        if status is DataStatus.UNAVAILABLE:
            headline = "DATA_UNAVAILABLE"
        else:
            headline = f"数据不可用（{status.value}）"
        level = AlertLevel.CRITICAL if status is DataStatus.UNAVAILABLE else AlertLevel.WARNING
        self._emit(
            SignalEventKind.DATA_UNAVAILABLE,
            level,
            f"{headline}：{reason or status.value} —— 已停止产生交易信号，请勿依据过期数据下单",
            ts=now,
        )

    def _emit(
        self,
        kind: SignalEventKind,
        level: AlertLevel,
        message: str,
        *,
        symbol: Optional[str] = None,
        signal_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        ts: Optional[datetime] = None,
    ) -> AlertEvent:
        event = AlertEvent(
            kind=kind,
            level=level,
            message=str(message),
            ts_utc=ts if ts is not None else self._clock(),
            symbol=symbol,
            signal_id=signal_id,
            details=dict(details or {}),
        )
        self._events.append(event)
        self._pending_events.append(event)
        callback = self._on_event
        if callback is not None:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - alerting must never stop the engine
                pass
        journal = self._journal
        record_event = getattr(journal, "record_event", None) if journal is not None else None
        if callable(record_event):
            try:
                record_event(event)
            except Exception:  # noqa: BLE001 - journal persistence is best effort here
                pass
        return event

    # ------------------------------------------------------------------ #
    # Internal — EOD reminders (risk manager owns the once-per-day flags)
    # ------------------------------------------------------------------ #
    def _handle_eod_alerts(self, now: datetime) -> None:
        eod_fn = getattr(self._risk, "eod_alerts", None)
        if not callable(eod_fn):
            return
        try:
            events = eod_fn(now)
        except Exception as exc:  # noqa: BLE001 - a broken risk module must not stop alerts
            self._emit(
                SignalEventKind.INFO,
                AlertLevel.WARNING,
                f"EOD 提醒检查失败：{exc}",
                ts=now,
            )
            return
        for event in events or ():
            if not isinstance(event, AlertEvent):
                continue
            message = event.message
            if event.kind is SignalEventKind.EOD_REMINDER and "人工平仓" not in message:
                message = f"{message} —— 需人工平仓，系统不会自动清仓"
            self._emit(
                event.kind,
                event.level,
                message,
                symbol=event.symbol,
                signal_id=event.signal_id,
                details=event.details,
                ts=now,
            )

    # ------------------------------------------------------------------ #
    # Internal — position hints (never closes anything)
    # ------------------------------------------------------------------ #
    def _handle_position_hints(self, now: datetime) -> None:
        risk_state = self._risk_state()
        position = getattr(risk_state, "position", None) if risk_state is not None else None
        if position is None:
            self._hint_keys.clear()
            return
        symbol = str(getattr(position, "symbol", "")).upper()
        price = self._last_price.get(symbol)
        if price is None:
            price = self._safe_float(getattr(position, "entry_price", None))
        if price is None or price <= 0:
            return

        signal_id = str(getattr(position, "signal_id", "") or f"{symbol}-pos")
        hints = self._hint_keys.setdefault(signal_id, set())
        entry = self._safe_float(getattr(position, "entry_price", None)) or price
        stop = self._safe_float(getattr(position, "stop_loss", None)) or 0.0
        take_profit = self._safe_float(getattr(position, "take_profit", None))
        shares = self._safe_float(getattr(position, "shares", None)) or 0.0
        open_risk = max(entry - stop, 0.0) * shares
        unrealized_pnl = (price - entry) * shares
        r_multiple = unrealized_pnl / open_risk if open_risk > 0 else 0.0

        # 1) Hard stop touched → critical manual-action alert.
        if stop > 0 and price <= stop and "STOP" not in hints:
            hints.add("STOP")
            self._emit(
                SignalEventKind.STOP_HIT_HINT,
                AlertLevel.CRITICAL,
                f"{symbol} 最新价 {price:.2f} 已触及止损 {stop:.2f}：请立即人工止损/平仓"
                "（系统只提醒、不会自动平仓）",
                symbol=symbol,
                signal_id=signal_id,
                details={"price": price, "stop": stop, "r_multiple": r_multiple},
                ts=now,
            )

        # 2) Breakeven hint at +1.2R (configurable).
        breakeven_r = float(self._config.risk.breakeven_hint_r)
        if r_multiple >= breakeven_r and "BREAKEVEN" not in hints:
            hints.add("BREAKEVEN")
            self._emit(
                SignalEventKind.BREAKEVEN_HINT,
                AlertLevel.WARNING,
                f"{symbol} 已达 {r_multiple:.2f}R（≥{breakeven_r:.1f}R 保本提示）："
                f"建议将止损上移至成本价 {entry:.2f}（人工操作，系统不会自动改单）",
                symbol=symbol,
                signal_id=signal_id,
                details={"price": price, "r_multiple": r_multiple, "breakeven_r": breakeven_r},
                ts=now,
            )

        # 3) Take-profit tiers, each tier emitted once per position.
        for tier in self._config.risk.take_profit_hint_r:
            token = f"TP:{tier:g}"
            if r_multiple >= float(tier) and token not in hints:
                hints.add(token)
                self._emit(
                    SignalEventKind.TAKE_PROFIT_HINT,
                    AlertLevel.WARNING,
                    f"{symbol} 已达 {r_multiple:.2f}R（≥{float(tier):.1f}R 止盈提示）："
                    f"目标价 {take_profit:.2f}，考虑分批止盈（人工操作，系统不会自动平仓）"
                    if take_profit is not None
                    else f"{symbol} 已达 {r_multiple:.2f}R（≥{float(tier):.1f}R 止盈提示）："
                    "考虑分批止盈（人工操作，系统不会自动平仓）",
                    symbol=symbol,
                    signal_id=signal_id,
                    details={"price": price, "r_multiple": r_multiple, "tier_r": float(tier)},
                    ts=now,
                )

        # 4) Optional trailing hint: close loses EMA9 while still in profit.
        if r_multiple > 0 and "TRAILING" not in hints:
            metrics = self._metrics.get(symbol)
            if metrics is None:
                metrics = self._compute_metrics(symbol, now)
            ema9 = metrics.get("ema9") if metrics else None
            if ema9 is not None and price < float(ema9):
                hints.add("TRAILING")
                self._emit(
                    SignalEventKind.TRAILING_HINT,
                    AlertLevel.INFO,
                    f"{symbol} 最新价 {price:.2f} 跌破 EMA9 {float(ema9):.2f} 且仍浮盈 "
                    f"{r_multiple:.2f}R：考虑上移止损/止盈（人工操作）",
                    symbol=symbol,
                    signal_id=signal_id,
                    details={"price": price, "ema9": float(ema9), "r_multiple": r_multiple},
                    ts=now,
                )

    # ------------------------------------------------------------------ #
    # Internal — candidate evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_candidates(self, triggering_symbol: str, now: datetime) -> None:
        mode = self._current_mode()
        targets = self._evaluation_targets(triggering_symbol, mode)
        for symbol in targets:
            context = self._build_strategy_context(symbol, now)
            if context is None:
                continue
            for strategy in self._strategies:
                try:
                    candidate = strategy.evaluate(context)
                except Exception as exc:  # noqa: BLE001 - one bad strategy must not kill the engine
                    self._emit(
                        SignalEventKind.INFO,
                        AlertLevel.INFO,
                        f"策略 {self._strategy_key(strategy)} 计算异常，已跳过：{exc}",
                        symbol=symbol,
                        ts=now,
                    )
                    continue
                if candidate is None:
                    continue
                if not isinstance(candidate, SignalCandidate):
                    continue
                key = (
                    candidate.symbol,
                    self._strategy_key(strategy),
                    candidate.confirmation_bar_start_utc,
                )
                if key in self._signal_keys:
                    continue  # same confirmation bar already produced a candidate
                self._remember_signal_key(key)
                self._handle_candidate(candidate, now)
                return  # dual-watch: at most one candidate per completed bar

    def _evaluation_targets(self, triggering_symbol: str, mode: Mode) -> List[str]:
        """Symbols to evaluate for this bar, in configured order (max 1 entry)."""
        if mode is Mode.LONG_SOXL:
            allowed = {"SOXL"}
        elif mode is Mode.LONG_SOXS:
            allowed = {"SOXS"}
        else:  # DUAL_WATCH
            allowed = set(self._symbols)
        if triggering_symbol not in allowed:
            return []
        targets = [triggering_symbol]
        triggering_window = self._bars_1m.get(triggering_symbol)
        if triggering_window:
            latest_bar_start = triggering_window[-1].start_utc
            for symbol in self._symbols:
                if symbol == triggering_symbol or symbol not in allowed:
                    continue
                window = self._bars_1m.get(symbol)
                if window and window[-1].start_utc == latest_bar_start:
                    targets.append(symbol)
        ordered = [symbol for symbol in self._symbols if symbol in targets]
        return ordered or ([triggering_symbol] if triggering_symbol in allowed else [])

    def _handle_candidate(self, candidate: SignalCandidate, now: datetime) -> None:
        sizing = self._sizing if self._sizing is not None else compat.get_size_position()
        try:
            plan = sizing(candidate, self._config.risk)
        except Exception as exc:  # noqa: BLE001 - sizing failure blocks the trade, never crashes
            self._emit(
                SignalEventKind.INFO,
                AlertLevel.WARNING,
                f"仓位计算失败，信号 {candidate.symbol} 未发出：{exc}",
                symbol=candidate.symbol,
                ts=now,
            )
            return

        try:
            try:
                decision = self._risk.evaluate_entry(
                    candidate, plan, now, data_ready=self._data_status is DataStatus.OK
                )
            except TypeError:  # tolerate a simpler duck-typed risk manager
                decision = self._risk.evaluate_entry(candidate, plan, now)
        except Exception as exc:  # noqa: BLE001 - fail closed (no signal) on risk errors
            self._emit(
                SignalEventKind.RISK_BLOCK,
                AlertLevel.INFO,
                f"风控评估异常，{candidate.symbol} 信号已拦截：{exc}",
                symbol=candidate.symbol,
                ts=now,
            )
            return

        allowed = bool(getattr(decision, "allowed", False))
        if not allowed:
            reason = getattr(decision, "reason", RiskBlockReason.NONE)
            reason_value = reason.value if hasattr(reason, "value") else str(reason)
            message = str(getattr(decision, "message", "") or reason_value)
            self._emit(
                SignalEventKind.RISK_BLOCK,
                AlertLevel.INFO,
                f"风控拦截 {candidate.symbol} {candidate.strategy.value}：{reason_value} {message}",
                symbol=candidate.symbol,
                details={
                    "reason": reason_value,
                    "candidate": candidate.to_dict(),
                    "plan": plan.to_dict(),
                },
                ts=now,
            )
            return

        approved_plan = getattr(decision, "plan", None)
        if isinstance(approved_plan, RiskPlan):
            plan = approved_plan
        signal_id = self._record_signal(candidate, plan, now)
        self._latest_signal = candidate
        self._latest_plan = plan
        self._latest_signal_id = signal_id
        self._emit(
            SignalEventKind.ENTRY_SIGNAL,
            AlertLevel.WARNING,
            self._entry_message(candidate, plan),
            symbol=candidate.symbol,
            signal_id=signal_id,
            details={
                "candidate": candidate.to_dict(),
                "plan": plan.to_dict(),
                "signal_id": signal_id,
            },
            ts=now,
        )

    # ------------------------------------------------------------------ #
    # Internal — context / indicators
    # ------------------------------------------------------------------ #
    def _build_strategy_context(self, symbol: str, now: datetime) -> Any:
        bars = tuple(self._bars_1m.get(symbol, ()))
        if not bars:
            return None
        metrics = self._compute_metrics(symbol, now)
        bars_5m = tuple(self._bars_5m.get(symbol, ()))
        try:
            context = compat.build_strategy_context(
                symbol=symbol,
                now_utc=now,
                session_phase=market_phase(now),
                mode=self._current_mode(),
                bars_1m=bars,
                bars_5m=bars_5m,
                vwap=metrics.get("vwap"),
                ema_fast=metrics.get("ema9"),
                ema_slow=metrics.get("ema21"),
                rsi=metrics.get("rsi"),
                atr=metrics.get("atr"),
                opening_range_5m=metrics.get("opening_range_5m"),
                opening_range_15m=metrics.get("opening_range_15m"),
                avg_volume_20=metrics.get("avg_volume_20"),
                last_price=self._last_price.get(symbol, bars[-1].close),
            )
        except Exception as exc:  # noqa: BLE001 - a broken sibling package must not kill the engine
            self._emit(
                SignalEventKind.INFO,
                AlertLevel.INFO,
                f"策略上下文构建失败，{symbol} 本轮跳过：{exc}",
                symbol=symbol,
                ts=now,
            )
            return None
        return context

    def _compute_metrics(self, symbol: str, now: datetime) -> Dict[str, Any]:
        bars = list(self._bars_1m.get(symbol, ()))
        closes = [float(bar.close) for bar in bars]
        strategy_config = self._config.strategies
        ema9 = compat.last_value(compat.compute_ema(closes, strategy_config.ema_fast))
        ema21 = compat.last_value(compat.compute_ema(closes, strategy_config.ema_slow))
        rsi = compat.last_value(compat.compute_rsi(closes, strategy_config.rsi_period))
        atr = compat.last_value(compat.compute_atr(bars, strategy_config.atr_period))
        prior_bars = bars[:-1] if len(bars) > 1 else []
        avg_volume = compat.last_value(compat.compute_volume_mean(prior_bars, 20)) if prior_bars else None

        if bars:
            anchor = session_open_utc(bars[-1].start_utc)
        else:
            anchor = session_open_utc(now)
        opening_range_5m = compat.compute_opening_range(bars, anchor, 5)
        opening_range_15m = compat.compute_opening_range(bars, anchor, 15)

        bars_5m = list(self._bars_5m.get(symbol, ()))
        closes_5m = [float(bar.close) for bar in bars_5m]
        ema9_5m = compat.last_value(compat.compute_ema(closes_5m, strategy_config.ema_fast))
        ema21_5m = compat.last_value(compat.compute_ema(closes_5m, strategy_config.ema_slow))

        metrics: Dict[str, Any] = {
            "symbol": symbol,
            "last_price": self._last_price.get(symbol, closes[-1] if closes else None),
            "vwap": self._resolve_vwap(symbol, now),
            "ema9": ema9,
            "ema21": ema21,
            "rsi": rsi,
            "atr": atr,
            "avg_volume_20": avg_volume,
            "opening_range_5m": opening_range_5m,
            "opening_range_15m": opening_range_15m,
            "trend_1m": self._trend(ema9, ema21),
            "trend_5m": self._trend(ema9_5m, ema21_5m),
            "bars_1m_count": len(bars),
            "bars_5m_count": len(bars_5m),
        }
        self._metrics[symbol] = metrics
        return metrics

    def _resolve_vwap(self, symbol: str, now: datetime) -> Optional[float]:
        """True session VWAP from the injected callback only (never bar.vwap)."""
        provider = self._vwap_provider
        if provider is None:
            data_quality = self._data_quality
            for name in ("current_session_vwap", "session_vwap", "get_session_vwap"):
                candidate = getattr(data_quality, name, None) if data_quality is not None else None
                if callable(candidate):
                    provider = candidate
                    break
        if provider is None:
            return None
        try:
            value = provider(symbol, now)
        except Exception:  # noqa: BLE001 - display value only
            return None
        number = self._safe_float(value)
        if number is None or number <= 0:
            return None
        return number

    @staticmethod
    def _trend(fast: Optional[float], slow: Optional[float]) -> str:
        if fast is None or slow is None:
            return "flat"
        if fast > slow:
            return "bull"
        if fast < slow:
            return "bear"
        return "flat"

    # ------------------------------------------------------------------ #
    # Internal — snapshots helpers
    # ------------------------------------------------------------------ #
    def _build_symbol_snapshot(self, symbol: str, now: datetime) -> SymbolSnapshot:
        metrics = self._compute_metrics(symbol, now)
        window = self._bars_1m.get(symbol)
        last_bar = window[-1] if window else None
        last_price = self._last_price.get(symbol)
        if last_price is None and last_bar is not None:
            last_price = float(last_bar.close)
        age: Optional[float] = None
        if last_bar is not None:
            age = max((now - last_bar.end_utc).total_seconds(), 0.0)
        return SymbolSnapshot(
            symbol=symbol,
            last_price=last_price,
            last_bar_utc=last_bar.start_utc if last_bar is not None else None,
            last_bar_end_utc=last_bar.end_utc if last_bar is not None else None,
            data_age_seconds=age,
            bars_1m_count=len(window) if window else 0,
            bars_5m_count=len(self._bars_5m.get(symbol, ())),
            trend_1m=metrics.get("trend_1m", "flat"),
            trend_5m=metrics.get("trend_5m", "flat"),
            vwap=metrics.get("vwap"),
            ema9=metrics.get("ema9"),
            ema21=metrics.get("ema21"),
            rsi=metrics.get("rsi"),
            atr=metrics.get("atr"),
            avg_volume_20=metrics.get("avg_volume_20"),
        )

    def _build_position_snapshot(self, risk_state: Any) -> Optional[PositionSnapshot]:
        position = getattr(risk_state, "position", None) if risk_state is not None else None
        if position is None:
            return None
        symbol = str(getattr(position, "symbol", "")).upper()
        last_price = self._last_price.get(symbol)
        if last_price is None:
            last_price = self._safe_float(getattr(position, "entry_price", None))
        entry = self._safe_float(getattr(position, "entry_price", None)) or 0.0
        stop = self._safe_float(getattr(position, "stop_loss", None)) or 0.0
        shares = self._safe_float(getattr(position, "shares", None)) or 0.0
        take_profit = self._safe_float(getattr(position, "take_profit", None)) or 0.0
        open_risk = max(entry - stop, 0.0) * shares
        unrealized_pnl: Optional[float] = None
        unrealized_r: Optional[float] = None
        if last_price is not None:
            pnl_fn = getattr(position, "unrealized_pnl", None)
            r_fn = getattr(position, "unrealized_r", None)
            try:
                unrealized_pnl = (
                    float(pnl_fn(last_price)) if callable(pnl_fn) else (last_price - entry) * shares
                )
            except Exception:  # noqa: BLE001
                unrealized_pnl = (last_price - entry) * shares
            try:
                unrealized_r = (
                    float(r_fn(last_price))
                    if callable(r_fn)
                    else (unrealized_pnl / open_risk if open_risk > 0 else 0.0)
                )
            except Exception:  # noqa: BLE001
                unrealized_r = unrealized_pnl / open_risk if open_risk > 0 else 0.0
        mode_value = getattr(position, "mode", None)
        strategy_value = getattr(position, "strategy", None)
        entry_utc = getattr(position, "entry_utc", None)
        if not isinstance(entry_utc, datetime):
            entry_utc = self._clock()
        return PositionSnapshot(
            signal_id=str(getattr(position, "signal_id", "") or ""),
            symbol=symbol,
            strategy=(
                strategy_value.value
                if hasattr(strategy_value, "value")
                else str(strategy_value or "")
            ),
            mode=mode_value.value if hasattr(mode_value, "value") else str(mode_value or ""),
            shares=shares,
            entry_price=entry,
            entry_price_signal=self._safe_float(
                getattr(position, "entry_price_signal", entry)
            )
            or entry,
            entry_utc=entry_utc,
            stop_loss=stop,
            take_profit=take_profit,
            last_price=last_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_r=unrealized_r,
            open_risk=open_risk,
            notional=entry * shares,
        )

    # ------------------------------------------------------------------ #
    # Internal — risk helpers
    # ------------------------------------------------------------------ #
    def _risk_state(self) -> Any:
        snapshot_fn = getattr(self._risk, "snapshot", None)
        if not callable(snapshot_fn):
            return None
        try:
            return snapshot_fn()
        except Exception:  # noqa: BLE001
            return None

    def _current_mode(self) -> Mode:
        risk_state = self._risk_state()
        raw = getattr(risk_state, "mode", None) if risk_state is not None else None
        if isinstance(raw, Mode):
            return raw
        if isinstance(raw, str):
            try:
                return Mode(raw)
            except ValueError:
                pass
        return self._config.mode

    def _daily_stats(self) -> Dict[str, Any]:
        stats_fn = getattr(self._risk, "daily_stats", None)
        if not callable(stats_fn):
            return {}
        try:
            stats = stats_fn()
        except Exception:  # noqa: BLE001
            return {}
        converted = compat.to_jsonable(stats)
        if isinstance(converted, Mapping):
            return dict(converted)
        return {}

    @staticmethod
    def _risk_int(state: Any, key: str, default: int = 0) -> int:
        value = getattr(state, key, None) if state is not None else None
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default or 0)

    @staticmethod
    def _risk_float(state: Any, keys: Iterable[str], default: float = 0.0) -> float:
        for key in keys:
            value = getattr(state, key, None) if state is not None else None
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default or 0.0)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    # ------------------------------------------------------------------ #
    # Internal — signal bookkeeping / formatting
    # ------------------------------------------------------------------ #
    def _record_signal(
        self, candidate: SignalCandidate, plan: RiskPlan, now: datetime
    ) -> str:
        journal = self._journal
        record_signal = getattr(journal, "record_signal", None) if journal is not None else None
        if callable(record_signal):
            try:
                return str(record_signal(candidate, plan))
            except Exception as exc:  # noqa: BLE001 - journal write failure must not lose the alert
                self._emit(
                    SignalEventKind.INFO,
                    AlertLevel.WARNING,
                    f"信号写入 journal 失败，仍发出提醒：{exc}",
                    symbol=candidate.symbol,
                    ts=now,
                )
        return new_signal_id(now)

    def _entry_message(self, candidate: SignalCandidate, plan: RiskPlan) -> str:
        direction = "看涨半导体" if candidate.symbol == "SOXL" else "看跌半导体（做多 SOXS）"
        shares = plan.shares
        shares_text = f"{shares:.5f}".rstrip("0").rstrip(".") if shares % 1 else f"{int(shares)}"
        return (
            f"【入场信号】买入 {candidate.symbol}（{direction}）"
            f"｜策略 {candidate.strategy.value}"
            f"｜参考价 {candidate.entry_ref:.2f}"
            f"｜止损 {candidate.stop_loss:.2f}"
            f"｜止盈 {candidate.take_profit_price:.2f}"
            f"｜股数 {shares_text}"
            f"｜名义 ${plan.notional:.2f}"
            f"｜风险 ${plan.actual_risk:.2f}"
            f"｜R:R {candidate.target_r:.1f}"
            f" —— 需人工确认后在券商手动下单；系统只提醒，不会自动下单或平仓"
        )

    def _remember_signal_key(self, key: Tuple[str, str, datetime]) -> None:
        if len(self._signal_keys) >= _SIGNAL_KEY_LIMIT:
            self._signal_keys.clear()
        self._signal_keys.add(key)

    @staticmethod
    def _strategy_key(strategy: Any) -> str:
        name = getattr(strategy, "name", None)
        if isinstance(name, StrategyName):
            return name.value
        if hasattr(name, "value"):
            return str(name.value)
        return str(name or type(strategy).__name__)

    def _strategy_names(self) -> Tuple[str, ...]:
        return tuple(self._strategy_key(strategy) for strategy in self._strategies)
