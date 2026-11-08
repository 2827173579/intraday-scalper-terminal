"""Manual trade commands typed by the user (never executed by this system).

Supported grammar (case-insensitive)::

    ENTRY  <signal_id> <fill_price> <shares>
    EXIT   <signal_id> <fill_price>
    STOP   <signal_id> <fill_price>
    STOP   <fill_price>                 # uses the single open position
    CANCEL <signal_id>
    FLAT

``execute_command`` always returns a Chinese, user-facing string.  Invalid
input produces a friendly ``命令错误：...`` / ``错误：...`` message instead of an
exception (database errors are the documented exception and propagate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ..core import to_utc
from .journal import Journal

_SUPPORTED = "ENTRY / EXIT / STOP / CANCEL / FLAT"

_ENTRY_USAGE = "用法：ENTRY <signal_id> <成交价> <股数>"
_EXIT_USAGE = "用法：EXIT <signal_id> <成交价>"
_STOP_USAGE = "用法：STOP <signal_id> <成交价> 或 STOP <成交价>"
_CANCEL_USAGE = "用法：CANCEL <signal_id>"
_FLAT_USAGE = "用法：FLAT（无参数）"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """A validated manual command; produced by :func:`parse_command`."""

    action: str
    signal_id: str | None = None
    price: float | None = None
    shares: float | None = None
    raw: str = ""

    def __str__(self) -> str:  # pragma: no cover - display helper
        bits = [self.action]
        if self.signal_id is not None:
            bits.append(self.signal_id)
        if self.price is not None:
            bits.append(f"{self.price:g}")
        if self.shares is not None:
            bits.append(f"{self.shares:g}")
        return " ".join(bits)


def _parse_price(token: str) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"成交价必须是数字，收到 {token!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"成交价必须是正的有限数字，收到 {token!r}")
    return value


def _parse_shares(token: str) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"股数必须是数字，收到 {token!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"股数必须是正的有限数字，收到 {token!r}")
    return value


def _parse_signal_id(token: str) -> str:
    text = str(token).strip().upper()
    if not text:
        raise ValueError("signal_id 不能为空")
    return text


def parse_command(line: str) -> ParsedCommand:
    """Parse one command line.  Raises ``ValueError`` with a Chinese message."""
    if not isinstance(line, str):
        raise ValueError("命令必须是字符串")
    text = line.strip()
    if not text:
        raise ValueError("命令为空")
    parts = text.split()
    action = parts[0].upper()
    args = parts[1:]

    if action == "ENTRY":
        if len(args) != 3:
            raise ValueError(_ENTRY_USAGE)
        return ParsedCommand(
            action="ENTRY",
            signal_id=_parse_signal_id(args[0]),
            price=_parse_price(args[1]),
            shares=_parse_shares(args[2]),
            raw=text,
        )
    if action == "EXIT":
        if len(args) != 2:
            raise ValueError(_EXIT_USAGE)
        return ParsedCommand(
            action="EXIT",
            signal_id=_parse_signal_id(args[0]),
            price=_parse_price(args[1]),
            raw=text,
        )
    if action == "STOP":
        if len(args) == 1:
            return ParsedCommand(action="STOP", price=_parse_price(args[0]), raw=text)
        if len(args) == 2:
            return ParsedCommand(
                action="STOP",
                signal_id=_parse_signal_id(args[0]),
                price=_parse_price(args[1]),
                raw=text,
            )
        raise ValueError(_STOP_USAGE)
    if action == "CANCEL":
        if len(args) != 1:
            raise ValueError(_CANCEL_USAGE)
        return ParsedCommand(
            action="CANCEL", signal_id=_parse_signal_id(args[0]), raw=text
        )
    if action == "FLAT":
        if args:
            raise ValueError(_FLAT_USAGE)
        return ParsedCommand(action="FLAT", raw=text)
    raise ValueError(f"未知命令 {action!r}，支持 {_SUPPORTED}")


def execute_command(
    journal: Journal,
    line: str,
    *,
    now: datetime | None = None,
) -> str:
    """Validate, apply and describe one manual command.  Never raises for bad input."""
    if not isinstance(journal, Journal):
        raise TypeError(f"journal must be a Journal, got {type(journal)!r}")
    try:
        parsed = parse_command(line)
    except (ValueError, TypeError) as exc:
        return f"命令错误：{exc}"

    moment = to_utc(now) if now is not None else journal.now()
    try:
        if parsed.action == "ENTRY":
            return _do_entry(journal, parsed, moment)
        if parsed.action == "EXIT":
            return _do_exit(journal, parsed, moment)
        if parsed.action == "STOP":
            return _do_stop(journal, parsed, moment)
        if parsed.action == "CANCEL":
            return _do_cancel(journal, parsed)
        if parsed.action == "FLAT":
            return _do_flat(journal)
    except (ValueError, RuntimeError, KeyError) as exc:
        return f"错误：{exc}"
    return f"命令错误：未知命令 {parsed.action!r}"  # pragma: no cover - defensive


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
def _do_entry(journal: Journal, parsed: ParsedCommand, moment: datetime) -> str:
    assert parsed.signal_id is not None and parsed.price is not None
    assert parsed.shares is not None
    trade_id = journal.record_entry(
        parsed.signal_id, parsed.price, parsed.shares, moment
    )
    position = next(
        (
            item
            for item in journal.open_positions()
            if item.signal_id == parsed.signal_id
        ),
        None,
    )
    if position is None:  # pragma: no cover - defensive
        return f"已记录买入（trade_id={trade_id}），请到券商端确认成交。"
    return (
        f"已记录买入：{position.symbol} {position.shares:g} 股 @ "
        f"${position.entry_price:.2f}（信号价 ${position.entry_price_signal:.2f}，"
        f"入场滑点 {position.entry_slippage_bps:+.1f} bps；trade_id={trade_id}）。"
        f"请确认券商端已真实成交，系统不会自动下单。"
    )


def _do_exit(journal: Journal, parsed: ParsedCommand, moment: datetime) -> str:
    assert parsed.signal_id is not None and parsed.price is not None
    record = journal.record_exit(parsed.signal_id, parsed.price, moment, "MANUAL")
    return (
        f"已记录卖出：{record.symbol} {record.shares:g} 股 @ "
        f"${record.exit_price_fill:.2f}，盈亏 ${record.pnl:+.2f}"
        f"（{record.r_multiple:+.2f}R），退出原因 {record.exit_reason}。"
        f"系统不会自动下单。"
    )


def _do_stop(journal: Journal, parsed: ParsedCommand, moment: datetime) -> str:
    assert parsed.price is not None
    signal_id = parsed.signal_id
    if signal_id is None:
        open_positions = journal.open_positions()
        if not open_positions:
            raise ValueError("当前没有持仓，无法记录 STOP；请使用 STOP <signal_id> <成交价>")
        if len(open_positions) > 1:
            raise ValueError(
                "存在多个未平仓记录，请使用 STOP <signal_id> <成交价> 指定信号"
            )
        signal_id = open_positions[0].signal_id
    record = journal.record_stop(signal_id, parsed.price, moment)
    return (
        f"已记录止损卖出：{record.symbol} {record.shares:g} 股 @ "
        f"${record.exit_price_fill:.2f}，盈亏 ${record.pnl:+.2f}"
        f"（{record.r_multiple:+.2f}R），退出原因 {record.exit_reason}。"
        f"系统不会自动下单。"
    )


def _do_cancel(journal: Journal, parsed: ParsedCommand) -> str:
    assert parsed.signal_id is not None
    if journal.cancel_signal(parsed.signal_id, "MANUAL"):
        return f"已取消待确认信号 {parsed.signal_id}（状态改为 CANCELLED）。"
    return (
        f"取消失败：信号 {parsed.signal_id} 当前不是待确认状态"
        f"（可能已成交、已取消或已平仓）。"
    )


def _do_flat(journal: Journal) -> str:
    cleared = journal.confirm_flat()
    return (
        f"已确认当前空仓：清理了 {cleared} 条待确认/未平仓记录。"
        f"（仅更新本地日志，系统不会自动下单或清仓）"
    )


__all__ = ["ParsedCommand", "execute_command", "parse_command"]
