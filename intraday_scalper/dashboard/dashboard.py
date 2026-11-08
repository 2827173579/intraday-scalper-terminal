"""Rich terminal dashboard for the intraday signal engine.

The dashboard is a *display only*.  It renders an immutable
:class:`~intraday_scalper.signals.state.EngineSnapshot`; it can neither create a
signal nor close a position, and every panel that talks about exits says that a
human must act.  Data problems (``DATA_UNAVAILABLE`` / ``STALE``) are highlighted
at the very top in red because the operator must stop trading on bad data.

Usage::

    dashboard = Dashboard(engine.state_snapshot, config=config)
    dashboard.run(stop_event=stop_event)          # live 1 Hz refresh

For tests ``render()`` returns a plain Rich renderable and ``run()`` accepts
``max_frames`` so it never blocks.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Sequence

from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.config import AppConfig
from ..core.models import AlertEvent, AlertLevel, DataStatus, Mode
from ..core.timeutil import to_et, to_kst
from ..signals.state import EngineSnapshot, PositionSnapshot, SymbolSnapshot

__all__ = ["Dashboard"]

_STALE_BANNER = "DATA_UNAVAILABLE — 停止产生交易信号"


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_shares(value: Optional[float]) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number == int(number):
        return str(int(number))
    return f"{number:.5f}".rstrip("0").rstrip(".")


def _age_text(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    try:
        age = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if age < 120:
        return f"{age:.0f}s"
    return f"{age / 60.0:.1f}m"


def _trend_text(trend: Optional[str]) -> Text:
    if trend == "bull":
        return Text("▲ bull", style="bold green")
    if trend == "bear":
        return Text("▼ bear", style="bold red")
    return Text("— flat", style="dim")


def _level_style(level: AlertLevel) -> str:
    if level is AlertLevel.CRITICAL:
        return "bold red"
    if level is AlertLevel.WARNING:
        return "bold yellow"
    return "dim"


def _as_snapshot_provider(provider: Any) -> Optional[Callable[[], EngineSnapshot]]:
    """Accept a callable snapshot provider, or an engine with ``state_snapshot``."""
    if provider is None:
        return None
    if callable(provider):
        return provider
    method = getattr(provider, "state_snapshot", None)
    if callable(method):
        return method
    raise TypeError(
        "Dashboard needs a snapshot provider callable or an object with "
        f"state_snapshot(); got {type(provider)!r}"
    )


class Dashboard:
    """Render one :class:`EngineSnapshot` as a Rich layout."""

    def __init__(
        self,
        snapshot_provider: Optional[Callable[[], EngineSnapshot]] = None,
        *,
        console: Optional[Console] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self._snapshot_provider = _as_snapshot_provider(snapshot_provider)
        self._console = console if console is not None else Console()
        self._config = config

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, snapshot: EngineSnapshot) -> RenderableType:
        """Build the complete dashboard renderable for *snapshot*."""
        parts: List[RenderableType] = []
        if snapshot.data_status is not DataStatus.OK:
            parts.append(self._data_banner(snapshot))
        parts.append(self._header(snapshot))
        parts.append(self._symbols_table(snapshot))
        panels: List[RenderableType] = [self._mode_status_panel(snapshot)]
        if snapshot.position is not None:
            panels.append(self._position_panel(snapshot, snapshot.position))
        else:
            panels.append(self._flat_panel())
        if snapshot.latest_signal is not None:
            panels.append(self._signal_panel(snapshot))
        parts.append(Columns(panels, equal=True, expand=True))
        parts.append(self._events_panel(snapshot))
        parts.append(self._footer(snapshot))
        return Group(*parts)

    def _data_banner(self, snapshot: EngineSnapshot) -> RenderableType:
        status = snapshot.data_status
        lines = [
            Text(_STALE_BANNER, style="bold white on red"),
            Text(
                f"当前数据状态：{status.value}；数据异常时禁止新开仓，请检查行情连接。",
                style="bold red",
            ),
        ]
        return Panel(Group(*lines), border_style="red", box=box.HEAVY)

    def _header(self, snapshot: EngineSnapshot) -> RenderableType:
        et = to_et(snapshot.as_of_utc)
        kst = to_kst(snapshot.as_of_utc)
        header = Text()
        header.append("SOXL / SOXS 日内信号台", style="bold cyan")
        header.append("   ")
        header.append(f"{et:%Y-%m-%d %H:%M:%S} ET", style="bold")
        header.append("  /  ")
        header.append(f"{kst:%Y-%m-%d %H:%M:%S} KST", style="bold")
        header.append("   ")
        header.append(f"模式 {snapshot.mode.value.upper()}", style="bold magenta")
        header.append("   阶段 ")
        header.append(snapshot.session_phase.value, style="bold blue")
        return header

    def _symbols_table(self, snapshot: EngineSnapshot) -> RenderableType:
        table = Table(
            title="行情 / 指标（真实成交数据）",
            box=box.SIMPLE_HEAVY,
            expand=True,
            title_justify="left",
        )
        for column in ("标的", "最新价", "数据年龄", "1m 趋势", "5m 趋势", "VWAP", "EMA9", "EMA21", "RSI", "ATR"):
            table.add_column(column, justify="left" if column in ("标的", "1m 趋势", "5m 趋势") else "right")
        for symbol, data in self._ordered_symbols(snapshot):
            table.add_row(
                Text(symbol, style="bold"),
                self._price_text(data.last_price),
                Text(_age_text(data.data_age_seconds), style=self._age_style(data)),
                _trend_text(data.trend_1m),
                _trend_text(data.trend_5m),
                Text(_fmt(data.vwap)),
                Text(_fmt(data.ema9)),
                Text(_fmt(data.ema21)),
                Text(_fmt(data.rsi, 1)),
                Text(_fmt(data.atr, 3)),
            )
        if not snapshot.symbols:
            table.add_row(Text("暂无行情数据", style="dim"), *[Text("—") for _ in range(9)])
        return table

    def _price_text(self, price: Optional[float]) -> Text:
        if price is None:
            return Text("—", style="dim")
        return Text(_fmt(price), style="bold white")

    @staticmethod
    def _age_style(data: SymbolSnapshot) -> str:
        age = data.data_age_seconds
        if age is None or age < 0:
            return "dim"
        if age > 90:
            return "bold red"
        if age > 30:
            return "yellow"
        return "green"

    def _mode_status_panel(self, snapshot: EngineSnapshot) -> RenderableType:
        mode = snapshot.mode
        mode_color = {
            Mode.LONG_SOXL: "green",
            Mode.LONG_SOXS: "red",
            Mode.DUAL_WATCH: "magenta",
        }.get(mode, "white")
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left")
        grid.add_column(justify="left")
        grid.add_row("当前模式", Text(mode.value.upper(), style=f"bold {mode_color}"))
        grid.add_row("会话阶段", Text(snapshot.session_phase.value))
        data_color = "green" if snapshot.data_status is DataStatus.OK else "bold red"
        grid.add_row("数据状态", Text(snapshot.data_status.value, style=data_color))
        connection_color = "green" if snapshot.connection_state.value == "CONNECTED" else "yellow"
        grid.add_row("行情连接", Text(snapshot.connection_state.value, style=connection_color))
        grid.add_row(
            "今日开仓",
            Text(f"{snapshot.entries_today} / {snapshot.max_entries_per_day}", style="bold"),
        )
        grid.add_row(
            "交易 / 胜 / 止损",
            Text(f"{snapshot.trades_today} / {snapshot.wins_today} / {snapshot.stops_today}"),
        )
        pnl_style = "green" if snapshot.daily_pnl >= 0 else "red"
        grid.add_row(
            "今日累计盈亏",
            Text(f"${snapshot.daily_pnl:+.2f}", style=f"bold {pnl_style}"),
        )
        if snapshot.daily_stopped:
            grid.add_row("熔断", Text("已触发单日亏损熔断，今日不再开仓", style="bold red"))
        cooldown = self._risk_field(snapshot, "cooldown_until_utc")
        if cooldown:
            grid.add_row("连亏冷却", Text(str(cooldown), style="yellow"))
        switch_cooldown = self._risk_field(snapshot, "switch_cooldown_until_utc")
        if switch_cooldown:
            grid.add_row("切换冷却", Text(str(switch_cooldown), style="yellow"))
        grid.add_row(
            "策略",
            Text(", ".join(snapshot.strategy_names) if snapshot.strategy_names else "—", style="dim"),
        )
        eod_text = self._eod_countdown(snapshot)
        if eod_text:
            grid.add_row("EOD 倒计时", Text(eod_text, style="bold yellow"))
        return Panel(grid, title="模式 / 风控状态", border_style="blue", box=box.ROUNDED)

    def _flat_panel(self) -> RenderableType:
        content = Text()
        content.append("当前无持仓\n", style="bold")
        content.append("等待确定性信号；系统只提醒、不自动下单。\n", style="dim")
        content.append("入场需人工确认后在券商手动买入；平仓也必须人工执行。", style="yellow")
        return Panel(content, title="持仓", border_style="green", box=box.ROUNDED)

    def _position_panel(
        self, snapshot: EngineSnapshot, position: PositionSnapshot
    ) -> RenderableType:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left")
        grid.add_column(justify="left")
        grid.add_row(
            "持仓",
            Text(f"{position.symbol} {_fmt_shares(position.shares)} 股", style="bold white"),
        )
        grid.add_row(
            "入场价 / 信号价",
            Text(f"{_fmt(position.entry_price)} / {_fmt(position.entry_price_signal)}"),
        )
        grid.add_row(
            "止损 / 止盈",
            Text(f"{_fmt(position.stop_loss)} / {_fmt(position.take_profit)}", style="bold"),
        )
        if position.last_price is not None:
            grid.add_row("最新价", Text(_fmt(position.last_price), style="bold"))
        if position.unrealized_pnl is not None:
            pnl_style = "green" if position.unrealized_pnl >= 0 else "red"
            grid.add_row(
                "浮动盈亏",
                Text(f"${position.unrealized_pnl:+.2f}", style=f"bold {pnl_style}"),
            )
        if position.unrealized_r is not None:
            r_style = "green" if position.unrealized_r >= 0 else "red"
            grid.add_row("浮动 R", Text(f"{position.unrealized_r:+.2f}R", style=f"bold {r_style}"))
        if position.notional is not None:
            grid.add_row("名义仓位", Text(f"${position.notional:.2f}"))
        if position.open_risk is not None:
            grid.add_row("持仓风险", Text(f"${position.open_risk:.2f}"))
        hints = Text()
        hints.append(
            f"{snapshot.breakeven_hint_r:.1f}R 保本提示线 / "
            f"{'/'.join(f'{tier:.1f}R' for tier in snapshot.take_profit_hint_r)} 止盈提示线",
            style="yellow",
        )
        r_value = position.unrealized_r or 0.0
        hints.append("\n")
        if r_value >= snapshot.breakeven_hint_r:
            hints.append(
                f"已达 {r_value:.2f}R：建议将止损上移至成本价（人工操作）", style="bold yellow"
            )
        elif r_value >= 0:
            hints.append("尚未到保本提示线", style="dim")
        else:
            hints.append("当前低于成本，注意止损纪律", style="red")
        hints.append("\n")
        if snapshot.take_profit_hint_r and r_value >= min(snapshot.take_profit_hint_r):
            hints.append(
                f"已达 {r_value:.2f}R：考虑分批止盈（人工操作）", style="bold green"
            )
        hints.append("\n需人工平仓，系统不会自动清仓。", style="bold red")
        grid.add_row("提示", hints)
        eod_text = self._eod_countdown(snapshot)
        if eod_text:
            grid.add_row("EOD 倒计时", Text(eod_text, style="bold yellow"))
        return Panel(grid, title="持仓（人工）", border_style="yellow", box=box.ROUNDED)

    def _signal_panel(self, snapshot: EngineSnapshot) -> RenderableType:
        candidate = snapshot.latest_signal
        assert candidate is not None
        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left")
        grid.add_column(justify="left")
        grid.add_row("标的 / 策略", Text(f"{candidate.symbol} / {candidate.strategy.value}"))
        grid.add_row("入场参考价", Text(_fmt(candidate.entry_ref), style="bold"))
        grid.add_row("止损 / 止盈", Text(f"{_fmt(candidate.stop_loss)} / {_fmt(candidate.take_profit_price)}"))
        grid.add_row("R:R", Text(f"{candidate.target_r:.1f}", style="bold"))
        plan = snapshot.latest_plan
        if plan is not None:
            grid.add_row("计划股数", Text(_fmt_shares(plan.shares), style="bold"))
            grid.add_row("计划名义", Text(f"${plan.notional:.2f}"))
            grid.add_row("风险预算", Text(f"${plan.actual_risk:.2f}", style="bold"))
        if snapshot.latest_signal_id:
            grid.add_row("信号 ID", Text(snapshot.latest_signal_id, style="dim"))
        footer = Text("\n需人工确认后在券商手动下单；系统只提醒、不自动下单。", style="bold yellow")
        grid.add_row("说明", footer)
        return Panel(grid, title="最近入场信号", border_style="magenta", box=box.ROUNDED)

    def _events_panel(self, snapshot: EngineSnapshot) -> RenderableType:
        table = Table(
            title="最近事件",
            box=box.SIMPLE,
            expand=True,
            title_justify="left",
        )
        table.add_column("KST", style="dim", no_wrap=True)
        table.add_column("级别", no_wrap=True)
        table.add_column("类型", no_wrap=True)
        table.add_column("消息")
        events = list(snapshot.events)[-8:]
        if not events:
            table.add_row("—", "—", "—", Text("暂无事件", style="dim"))
        for event in events:
            table.add_row(
                self._event_time(event),
                Text(event.level.value, style=_level_style(event.level)),
                Text(event.kind.value),
                Text(event.message),
            )
        return table

    def _footer(self, snapshot: EngineSnapshot) -> RenderableType:
        footer = Text()
        footer.append(
            "本系统只产生提醒与建议，不构成投资建议，不保证盈利。"
            "系统只提醒、人工下单、人工平仓；绝不自动下单（无券商 API）。",
            style="dim",
        )
        return footer

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ordered_symbols(
        self, snapshot: EngineSnapshot
    ) -> Sequence[tuple[str, SymbolSnapshot]]:
        preferred: Iterable[str]
        if self._config is not None:
            preferred = self._config.data.symbols
        else:
            preferred = ("SOXL", "SOXS")
        ordered: List[tuple[str, SymbolSnapshot]] = []
        seen: set[str] = set()
        for symbol in preferred:
            if symbol in snapshot.symbols and symbol not in seen:
                ordered.append((symbol, snapshot.symbols[symbol]))
                seen.add(symbol)
        for symbol, data in snapshot.symbols.items():
            if symbol not in seen:
                ordered.append((symbol, data))
        return ordered

    @staticmethod
    def _risk_field(snapshot: EngineSnapshot, key: str) -> Any:
        return (snapshot.risk or {}).get(key)

    def _eod_countdown(self, snapshot: EngineSnapshot) -> str:
        close_utc = snapshot.eod_close_utc
        if close_utc is None:
            return ""
        minutes = (close_utc - snapshot.as_of_utc).total_seconds() / 60.0
        if minutes <= 0:
            return "已收盘（无新开仓）"
        return f"EOD 倒计时 {minutes:.0f} 分钟（15:50/15:55 ET 提醒，16:00 收盘）"

    @staticmethod
    def _event_time(event: AlertEvent) -> str:
        try:
            return f"{to_kst(event.ts_utc):%H:%M:%S}"
        except Exception:  # noqa: BLE001
            return "—"

    # ------------------------------------------------------------------ #
    # Live loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        stop_event: Optional[threading.Event] = None,
        refresh_hz: float = 1.0,
        max_frames: Optional[int] = None,
        *,
        snapshot_provider: Optional[Callable[[], EngineSnapshot]] = None,
    ) -> int:
        """Refresh the dashboard until *stop_event* is set (or *max_frames*).

        ``max_frames`` and a pre-set ``stop_event`` make this testable without
        blocking.  Returns the number of rendered frames.
        """
        provider = _as_snapshot_provider(snapshot_provider) or self._snapshot_provider
        if provider is None:
            raise ValueError("Dashboard.run needs a snapshot_provider callable")
        try:
            hz = float(refresh_hz)
        except (TypeError, ValueError):
            hz = 1.0
        interval = 1.0 / hz if hz > 0 else 1.0
        frames = 0
        with Live(
            console=self._console,
            auto_refresh=False,
            refresh_per_second=max(1, min(int(hz) if hz >= 1 else 1, 20)),
            transient=False,
        ) as live:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_frames is not None and frames >= max_frames:
                    break
                snapshot = provider()
                live.update(self.render(snapshot), refresh=True)
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
                if stop_event is not None:
                    if stop_event.wait(timeout=interval):
                        break
                else:
                    time.sleep(interval)
        return frames
