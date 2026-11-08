"""Command-line entry point for the SOXL / SOXS intraday scalper.

Modes
-----
``live``
    Start the Finnhub websocket, aggregate ticks into 1m/5m bars, evaluate the
    deterministic strategies, enforce the risk state machine and show the Rich
    dashboard.  Alerts are audible.  If the account has no realtime US trade
    permission (the usual case for the free tier) the engine shows
    ``DATA_UNAVAILABLE`` and does **not** produce signals.

``demo``
    Replay deterministic *synthetic* bars through the same engine and render the
    dashboard.  This validates the wiring only -- it is never real market data.

``backtest``
    Run the event-driven backtester on synthetic sessions (real historical data
    is unavailable with the current Finnhub account) and print the metrics plus
    the explicit data caveat.

This program never places orders and never talks to a broker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .aggregation.bars import BarAggregator, FiveMinuteAggregator, SessionVWAP
from rich.console import Console

from .core import (
    AlertEvent,
    AlertLevel,
    AppConfig,
    Bar,
    ConnectionState,
    DataStatus,
    Mode,
    SignalCandidate,
    SignalEventKind,
    market_phase,
    now_utc,
    phase_allows_new_entries,
    to_et,
    to_kst,
)
from .core.config import PACKAGE_ROOT, load_config
from .dashboard.audio import alert_sound
from .dashboard.dashboard import Dashboard
from .data.recorder import DataRecorder
from .journal.commands import parse_command
from .journal.journal import Journal
from .providers.data_quality import DataQualityMonitor
from .providers.finnhub_rest import DataUnavailableError, FinnhubRestClient
from .providers.finnhub_ws import FinnhubWebSocketClient
from .risk.manager import RiskManager
from .risk.sizing import size_position
from .signals.engine import SignalEngine
from .strategies.registry import build_strategies

DATA_CAVEAT = (
    "当前 Finnhub 账户无历史 K 线（/stock/candle 403）与实时美股成交权限，"
    "因此这里的结果只用于验证引擎逻辑，**不是真实市场回测成绩**。"
)

DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
def load_api_key(env_file: str | Path | None = None) -> str:
    """Read ``FINNHUB_API_KEY`` from the environment or the git-ignored .env file.

    Never logs the value.  Returns an empty string when no key is configured
    (live mode then simply reports DATA_UNAVAILABLE instead of crashing).
    """
    value = os.environ.get("FINNHUB_API_KEY", "").strip()
    if value:
        return value
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        if key.strip() != "FINNHUB_API_KEY":
            continue
        cleaned = raw_value.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return ""


# --------------------------------------------------------------------------- #
# Runtime wiring
# --------------------------------------------------------------------------- #
class Runtime:
    """Wire providers, aggregation, engine, risk, journal and dashboard.

    ``clock`` is injectable so tests and the synthetic demo can control time
    deterministically; the live runtime uses the real UTC clock.
    """

    def __init__(
        self,
        config: AppConfig,
        token: str,
        *,
        clock: Callable[[], datetime] = now_utc,
        record_data: bool = True,
        quality_gate: bool = True,
        audio: bool = True,
        provider: str = "finnhub",
    ) -> None:
        """``provider`` selects the realtime feed: ``finnhub`` or ``alpaca``."""
        if provider not in {"finnhub", "alpaca"}:
            raise ValueError(f"unsupported provider {provider!r}")
        self.config = config
        self.token = token or ""
        self.provider = provider
        self._clock = clock
        self._audio = audio
        self._stop = threading.Event()
        self._events: Deque[AlertEvent] = deque(maxlen=100)
        self._candidates: Dict[str, SignalCandidate] = {}
        self._demo_vwap: Dict[str, float] = {}
        self.console = Console()

        config.paths.ensure_dirs()

        self.journal = Journal(config, clock=clock)
        self.risk = RiskManager(config, clock=clock)
        self.strategies = build_strategies(config.strategies)
        self.quality = DataQualityMonitor(config.data.symbols, config.data, clock=clock)
        self.vwap: Dict[str, SessionVWAP] = {symbol: SessionVWAP(symbol) for symbol in config.data.symbols}
        self.aggregator = BarAggregator(config.data.symbols)
        self.five_minute = FiveMinuteAggregator()
        self.recorder: Optional[DataRecorder] = (
            DataRecorder(config.paths, clock=clock) if record_data else None
        )
        self.engine = SignalEngine(
            config,
            self.strategies,
            self.risk,
            journal=self.journal,
            data_quality=self.quality if quality_gate else None,
            on_event=self._on_alert,
            clock=clock,
            current_session_vwap=self._resolve_vwap,
        )
        self.dashboard = Dashboard(self.snapshot, console=self.console, config=config)
        self.ws: Optional[FinnhubWebSocketClient] = None
        self.rest: Optional[FinnhubRestClient] = None
        self.alpaca = None  # AlpacaIEXProvider when provider == "alpaca"
        if provider == "alpaca":
            from .providers.alpaca_provider import AlpacaIEXProvider, load_alpaca_credentials

            self.alpaca_credentials = load_alpaca_credentials()
            self.alpaca = AlpacaIEXProvider(
                self.alpaca_credentials,
                config.data,
                on_tick=self._on_tick,
                on_bar=self._on_bar,
                on_state=self._on_ws_state,
                clock=clock,
            )
        else:
            self.alpaca_credentials = None
        if provider != "alpaca" and self.token:
            self.ws = FinnhubWebSocketClient(
                self.token,
                config.data.symbols,
                config.data,
                self._on_tick,
                self._on_ws_state,
                clock=clock,
            )
            self.rest = FinnhubRestClient(self.token, config.data, clock=clock)

    # -- dashboard / snapshot ------------------------------------------- #
    def snapshot(self):
        return self.engine.state_snapshot()

    # -- provider callbacks --------------------------------------------- #
    def _on_ws_state(self, state: ConnectionState, reason: str) -> None:
        self.engine.set_connection_state(state)
        self._events.append(
            AlertEvent(
                kind=SignalEventKind.STATE_CHANGE,
                level=AlertLevel.INFO,
                message=f"Finnhub WebSocket: {state.value} {reason}".strip(),
                ts_utc=self._clock(),
            )
        )

    def _on_tick(self, tick) -> None:
        """Called from the provider thread for every trade."""
        try:
            self.quality.record_tick(tick)
            tracker = self.vwap.get(tick.symbol)
            if tracker is not None:
                tracker.on_tick(tick)
            if self.recorder is not None:
                self.recorder.append_tick(tick)
            if self.alpaca is None:
                # Finnhub has no native 1m bars: aggregate locally from ticks.
                for bar in self.aggregator.on_tick(tick):
                    self._on_bar(bar)
            # Alpaca emits native completed 1m bars through ``_on_bar`` directly,
            # so feeding the tick aggregator here would duplicate every minute.
        except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the feed
            self._events.append(
                AlertEvent(
                    kind=SignalEventKind.INFO,
                    level=AlertLevel.WARNING,
                    message=f"行情处理异常：{exc}",
                    ts_utc=self._clock(),
                )
            )

    def _on_bar(self, bar: Bar) -> None:
        if self.recorder is not None:
            self.recorder.append_bar(bar)
        for five in self.five_minute.on_bar(bar):
            if self.recorder is not None:
                self.recorder.append_bar(five)
            self.engine.on_5m_bar(five)
        self.engine.on_bar(bar)

    def _resolve_vwap(self, symbol: str, now: datetime) -> Optional[float]:
        if self._demo_vwap:
            return self._demo_vwap.get(symbol)
        tracker = self.vwap.get(symbol)
        return None if tracker is None else tracker.value

    # -- alerts --------------------------------------------------------- #
    def _on_alert(self, event: AlertEvent) -> None:
        self._events.append(event)
        if event.kind.value == "ENTRY_SIGNAL":
            details = dict(event.details or {})
            candidate_data = details.get("candidate")
            if isinstance(candidate_data, dict) and event.signal_id:
                try:
                    candidate = _candidate_from_dict(candidate_data)
                except Exception:  # noqa: BLE001 - alert must still be shown
                    candidate = None
                if candidate is not None:
                    self._candidates[event.signal_id] = candidate
        if self._audio and event.level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
            try:
                alert_sound(event.level)
            except Exception:  # noqa: BLE001 - sound failure is never fatal
                pass

    def drain_events(self) -> List[AlertEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    # -- background loops ----------------------------------------------- #
    def _status_loop(self) -> None:
        while not self._stop.is_set():
            try:
                moment = self._clock()
                self.risk.roll(moment)
                if self.alpaca is None and not self.token:
                    if self.engine.data_status is not DataStatus.UNAVAILABLE:
                        self.engine.set_data_status(DataStatus.UNAVAILABLE, "未配置行情凭证")
                elif phase_allows_new_entries(market_phase(moment)):
                    report = self.quality.report(moment)
                    if report.aggregate is not self.engine.data_status:
                        self.engine.set_data_status(report.aggregate, report.reason)
            except Exception:  # noqa: BLE001 - monitor thread must survive
                pass
            self._stop.wait(1.0)

    def _command_loop(self) -> None:
        console = self.console
        for line in sys.stdin:
            if self._stop.is_set():
                return
            text = line.strip()
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                console.print("[cyan]收到退出指令，正在停止...[/cyan]")
                self._stop.set()
                return
            try:
                message = self.handle_command(text)
            except Exception as exc:  # noqa: BLE001 - a bad command must not crash the app
                message = f"命令执行失败：{exc}"
            console.print(f"[magenta]交易记录[/magenta] {message}")

    # -- manual trade journal ------------------------------------------- #
    def _plan_with_shares(self, plan, shares: float):
        entry = plan.candidate.entry_ref
        stop = plan.candidate.stop_loss
        risk_per_share = plan.candidate.risk_per_share
        theoretical = shares * risk_per_share
        slippage = shares * (entry + stop) * self.config.risk.est_slippage_bps / 10_000.0
        fees = shares * self.config.risk.commission_per_share
        return replace(
            plan,
            shares=shares,
            notional=shares * entry,
            theoretical_risk=theoretical,
            est_slippage_dollars=slippage,
            est_fees_dollars=fees,
            actual_risk=theoretical + slippage + fees,
        )

    def handle_command(self, line: str, now: Optional[datetime] = None) -> str:
        """Execute one manual journal command and keep risk + journal in sync."""
        moment = now or self._clock()
        try:
            parsed = parse_command(line)
        except ValueError as exc:
            return f"命令无效：{exc}"
        action = parsed.action.upper()

        if action == "ENTRY":
            if parsed.price is None or parsed.price <= 0:
                return "用法：ENTRY <signal_id> <price> <shares>"
            signal_id = parsed.signal_id
            if not signal_id:
                return "用法：ENTRY <signal_id> <price> <shares>"
            candidate = self._candidates.get(signal_id)
            if candidate is None:
                return f"未知信号 {signal_id}：只能录入本系统发出过的信号（或信号已过期）"
            plan = size_position(candidate, self.config.risk)
            if not plan.valid:
                return f"信号 {signal_id} 的仓位计算结果为 0 股，无法开仓"
            if parsed.shares is not None and parsed.shares > 0 and parsed.shares != plan.shares:
                plan = self._plan_with_shares(plan, parsed.shares)
            try:
                position = self.risk.confirm_entry(
                    plan,
                    fill_price=parsed.price,
                    fill_utc=moment,
                    signal_id=signal_id,
                    mode=self.risk.mode,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                return f"无法记录开仓：{exc}"
            try:
                self.journal.record_entry(
                    signal_id, parsed.price, plan.shares, moment, mode=position.mode
                )
            except Exception as exc:  # noqa: BLE001 - journal inconsistency is reported
                return f"开仓已进入风控状态，但 journal 写入失败：{exc}"
            slippage_bps = (parsed.price - candidate.entry_ref) / candidate.entry_ref * 10_000.0
            return (
                f"已记录开仓 {position.symbol} {plan.shares:g} 股 @ {parsed.price:g}"
                f"（信号价 {candidate.entry_ref:g}，滑点 {slippage_bps:+.1f} bps，"
                f"名义 ${plan.notional:,.2f}，风险 ${plan.actual_risk:,.2f}）"
            )

        if action in {"EXIT", "STOP"}:
            if parsed.price is None or parsed.price <= 0:
                return f"用法：{action} <signal_id> <price>"
            if self.risk.position is None:
                return "当前没有系统跟踪的持仓；若已平仓请用 FLAT 确认空仓"
            if parsed.signal_id and parsed.signal_id != self.risk.position.signal_id:
                return (
                    f"信号 {parsed.signal_id} 与当前持仓 {self.risk.position.signal_id} 不一致；"
                    "请核对持仓或使用不带信号ID的 STOP <price>"
                )
            try:
                if action == "STOP":
                    trade = self.risk.record_stop(parsed.price, moment)
                    self.journal.record_stop(trade.signal_id, parsed.price, moment)
                else:
                    trade = self.risk.confirm_exit(
                        fill_price=parsed.price, fill_utc=moment, reason="EXIT"
                    )
                    self.journal.record_exit(trade.signal_id, parsed.price, moment, "EXIT")
            except (RuntimeError, TypeError, ValueError) as exc:
                return f"无法记录{action}：{exc}"
            return (
                f"已记录{'止损' if action == 'STOP' else '平仓'} {trade.symbol} "
                f"{trade.shares:g} 股 @ {parsed.price:g}，"
                f"盈亏 ${trade.pnl:+,.2f}（{trade.r_multiple:+.2f}R）"
            )

        if action == "CANCEL":
            cleared = self.risk.record_cancel()
            signal_id = parsed.signal_id
            if signal_id:
                self.journal.cancel_signal(signal_id, "MANUAL")
            if not cleared:
                return "没有待执行的信号可取消"
            return "已取消待执行信号" + (f" {signal_id}" if signal_id else "")

        if action == "FLAT":
            risk_cleared = self.risk.confirm_flat()
            journal_cleared = self.journal.confirm_flat()
            if not risk_cleared and not journal_cleared:
                return "当前已是空仓状态"
            return f"已确认空仓（journal 关闭未平仓记录 {journal_cleared} 条）"

        return f"未知命令 {action}；支持 ENTRY / EXIT / STOP / CANCEL / FLAT"

    # -- startup -------------------------------------------------------- #
    def startup_validation(self) -> Dict[str, Any]:
        """One-shot connectivity check; its prices are never used as signal data."""
        if self.alpaca is not None:
            from .providers.alpaca_provider import probe_alpaca

            try:
                assert self.alpaca_credentials is not None
                result = probe_alpaca(self.alpaca_credentials)
            except Exception as exc:  # noqa: BLE001 - validation is best effort
                self._events.append(
                    AlertEvent(
                        kind=SignalEventKind.DATA_UNAVAILABLE,
                        level=AlertLevel.WARNING,
                        message=f"Alpaca 启动校验异常：{exc}",
                        ts_utc=self._clock(),
                    )
                )
                return {"provider": "alpaca", "ok": False, "error": str(exc)}
            payload = result.to_dict()
            payload["provider"] = "alpaca"
            if not result.ok:
                self._events.append(
                    AlertEvent(
                        kind=SignalEventKind.DATA_UNAVAILABLE,
                        level=AlertLevel.WARNING,
                        message=f"Alpaca 启动校验失败：{payload}",
                        ts_utc=self._clock(),
                    )
                )
            return payload
        if self.rest is None:
            return {}
        try:
            quotes = self.rest.startup_validation(list(self.config.data.symbols))
        except DataUnavailableError as exc:
            self._events.append(
                AlertEvent(
                    kind=SignalEventKind.DATA_UNAVAILABLE,
                    level=AlertLevel.WARNING,
                    message=f"启动行情校验失败（REST 权限不足）：{exc}",
                    ts_utc=self._clock(),
                )
            )
            return {}
        except Exception as exc:  # noqa: BLE001 - validation is best effort
            self._events.append(
                AlertEvent(
                    kind=SignalEventKind.INFO,
                    level=AlertLevel.WARNING,
                    message=f"启动行情校验异常：{exc}",
                    ts_utc=self._clock(),
                )
            )
            return {}
        result: Dict[str, Any] = {}
        for symbol, snapshot in quotes.items():
            result[symbol] = None if snapshot is None else snapshot.to_dict()
        return result

    # -- lifecycle ------------------------------------------------------ #
    def run(self, *, refresh_hz: float = 1.0, max_frames: Optional[int] = None) -> None:
        self.journal  # ensure constructed
        startup = self.startup_validation()
        if self.alpaca is None and not self.token:
            self.engine.set_data_status(DataStatus.UNAVAILABLE, "未配置行情凭证")
        active_provider = self.alpaca if self.alpaca is not None else self.ws
        if active_provider is not None:
            active_provider.start()
            active_provider.wait_until_connected(timeout=15.0)
        if startup:
            self.dashboard._console.print(f"[dim]启动校验：{json.dumps(startup, ensure_ascii=False)}[/dim]")
        monitor = threading.Thread(target=self._status_loop, name="status-loop", daemon=True)
        monitor.start()
        commands = threading.Thread(target=self._command_loop, name="command-loop", daemon=True)
        commands.start()
        try:
            self.dashboard.run(
                self._stop, refresh_hz=refresh_hz, max_frames=max_frames
            )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        for provider in (self.alpaca, self.ws):
            if provider is None:
                continue
            try:
                provider.stop()
                provider.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.journal.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _candidate_from_dict(payload: Dict[str, Any]) -> SignalCandidate:
    strategy = payload.get("strategy")
    return SignalCandidate(
        symbol=payload["symbol"],
        strategy=strategy,
        entry_ref=float(payload["entry_ref"]),
        stop_loss=float(payload["stop_loss"]),
        target_r=float(payload.get("target_r", 2.0)),
        rationale=str(payload.get("rationale", "")),
        created_utc=datetime.fromisoformat(payload["created_utc"]),
        confirmation_bar_start_utc=datetime.fromisoformat(
            payload["confirmation_bar_start_utc"]
        ),
        structural_note=str(payload.get("structural_note", "")),
    )


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #
def _has_alpaca_credentials(env_file: str | None = None) -> bool:
    try:
        from .providers.alpaca_provider import load_alpaca_credentials

        load_alpaca_credentials(env_file)
        return True
    except Exception:  # noqa: BLE001 - absence is normal for finnhub-only users
        return False


def cmd_live(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    console = Console()
    provider = args.provider
    if provider == "auto":
        provider = "alpaca" if _has_alpaca_credentials(args.env_file) else "finnhub"
    token = load_api_key(args.env_file) if provider == "finnhub" else ""
    if provider == "finnhub" and not token:
        console.print(
            "[bold yellow]未找到 FINNHUB_API_KEY：实时模式将显示 DATA_UNAVAILABLE，不会产生信号。[/bold yellow]"
        )
    console.print(
        f"[bold]SOXL / SOXS Intraday Scalper[/bold] · provider={provider} · "
        f"mode={config.mode.value} · symbols={', '.join(config.data.symbols)}"
    )
    if provider == "alpaca":
        console.print(
            "[dim]数据口径：Alpaca IEX 单一交易所实时行情（成交量约为全市场的一小部分），"
            "VWAP 为 IEX-trade VWAP，非全市场 consolidated tape。[/dim]"
        )
    console.print(
        "[dim]提醒：系统只做分析/信号/提醒，绝不自动下单；成交必须人工录入。[/dim]"
    )
    try:
        runtime = Runtime(config, token, provider=provider)
    except Exception as exc:  # noqa: BLE001 - e.g. missing/invalid Alpaca credentials
        console.print(f"[bold red]初始化 {provider} 数据源失败：{exc}[/bold red]")
        return 2
    try:
        runtime.run(refresh_hz=args.refresh, max_frames=args.max_frames)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .backtest.synthetic import generate_sessions, next_trading_dates

    config = _config_from_args(args)
    console = Console()
    start = args.start_date or date(2026, 11, 9)
    dates = next_trading_dates(start, args.days)
    bars = generate_sessions(
        list(config.data.symbols),
        dates,
        seed=args.seed,
        trend=args.trend,
        vol=args.vol,
    )
    clock_state = {"now": bars[0].start_utc if bars else now_utc()}

    def sim_clock() -> datetime:
        return clock_state["now"]

    runtime = Runtime(
        config,
        "",
        clock=sim_clock,
        record_data=False,
        quality_gate=False,
        audio=False,
    )
    accumulators = {symbol: [0.0, 0.0] for symbol in config.data.symbols}
    five = FiveMinuteAggregator()
    try:
        for bar in bars:
            clock_state["now"] = bar.end_utc
            acc = accumulators.setdefault(bar.symbol, [0.0, 0.0])
            acc[0] += bar.vwap * bar.volume
            acc[1] += bar.volume
            if acc[1] > 0:
                runtime._demo_vwap[bar.symbol] = acc[0] / acc[1]
            runtime.engine.on_bar(bar)
            for five_bar in five.on_bar(bar):
                runtime.engine.on_5m_bar(five_bar)
        runtime.engine.set_data_status(DataStatus.OK, "synthetic demo")
        snapshot = runtime.snapshot()
        if args.json:
            print(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2))
        else:
            console.print(runtime.dashboard.render(snapshot))
            console.print(
                f"[bold yellow]SYNTHETIC DEMO ONLY[/bold yellow] · {len(bars)} 根合成 1m bar · "
                "不是真实行情，不构成任何交易信号"
            )
            for event in runtime.drain_events():
                console.print(f"[dim]{event.kind.value}[/dim] {event.message}")
    finally:
        runtime.shutdown()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """One-shot Alpaca IEX connectivity check (prints JSON, read-only)."""
    from .providers.alpaca_provider import (
        AlpacaCredentialsError,
        load_alpaca_credentials,
        probe_alpaca,
    )

    console = Console()
    try:
        credentials = load_alpaca_credentials(args.env_file)
    except AlpacaCredentialsError as exc:
        console.print(f"[bold red]Alpaca 凭证缺失/不可读：{exc}[/bold red]")
        return 2
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    result = probe_alpaca(credentials, symbols=symbols)
    payload = result.to_dict()
    payload.update({"feed": credentials.feed, "ok": result.ok, "any_ok": result.any_ok})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result.ok:
        console.print("[bold green]ALPACA OK[/bold green] · latest bar / quote / trade 均可读取")
        return 0
    console.print("[bold red]ALPACA FAILED[/bold red] · 详见上方各项 error")
    return 1


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.engine import run_synthetic_backtest
    from .backtest.synthetic import next_trading_dates

    config = _config_from_args(args)
    strategies = build_strategies(config.strategies)
    result = run_synthetic_backtest(
        config,
        strategies,
        days=args.days,
        seed=args.seed,
        symbols=config.data.symbols,
    )
    payload = {
        "data_limitation": DATA_CAVEAT,
        "synthetic_only": True,
        "days": args.days,
        "seed": args.seed,
        "metrics": result.metrics.to_dict(),
        "trades": len(result.trades),
        "notes": list(result.notes),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    if getattr(args, "mode", None):
        config = config.with_mode(args.mode)
    return config.validate()


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intraday_scalper",
        description="SOXL/SOXS minute-level intraday signal system (never places orders).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None, help="YAML config path (default: config/default.yaml)")
        p.add_argument("--mode", choices=[m.value for m in Mode], default=None, help="override trading mode")

    live = sub.add_parser("live", help="realtime signal engine + dashboard (Alpaca IEX or Finnhub)")
    add_common(live)
    live.add_argument("--refresh", type=float, default=1.0, help="dashboard refresh seconds")
    live.add_argument("--env-file", default=None, help="path to .env holding ALPACA_* / FINNHUB_API_KEY")
    live.add_argument(
        "--provider",
        choices=["auto", "alpaca", "finnhub"],
        default="auto",
        help="data feed selection; auto prefers Alpaca when credentials exist",
    )
    live.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after N dashboard frames (mainly for smoke tests)",
    )
    live.set_defaults(func=cmd_live)

    check = sub.add_parser("check", help="one-shot Alpaca IEX connectivity check (Alpaca only)")
    check.add_argument("--env-file", default=None, help="path to .env holding ALPACA_* credentials")
    check.add_argument("--symbols", default="SOXL,SOXS", help="comma-separated symbols")
    check.set_defaults(func=cmd_check)

    demo = sub.add_parser("demo", help="synthetic-data demo of the full pipeline")
    add_common(demo)
    demo.add_argument("--days", type=int, default=1, help="synthetic trading days")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--trend", type=float, default=0.0)
    demo.add_argument("--vol", type=float, default=0.0015)
    demo.add_argument("--start-date", type=date.fromisoformat, default=None)
    demo.add_argument("--json", action="store_true", help="print the engine snapshot as JSON")
    demo.set_defaults(func=cmd_demo)

    backtest = sub.add_parser("backtest", help="synthetic-data logic check (no real history available)")
    add_common(backtest)
    backtest.add_argument("--days", type=int, default=5)
    backtest.add_argument("--seed", type=int, default=7)
    backtest.set_defaults(func=cmd_backtest)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    try:  # Windows 控制台默认 GBK，中文/符号输出需显式 UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - reconfigure is best effort
        pass
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
