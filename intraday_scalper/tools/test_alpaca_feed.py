"""Alpaca Markets (IEX feed) connectivity probe.

Reads credentials from ``intraday_scalper/.env`` (or the process environment)
and verifies, in order:

1. **Account** — ``paper-api.alpaca.markets`` paper account status.
2. **REST data** — latest 1-minute bar, latest quote and latest trade for
   SOXL / SOXS plus a few recent 1-minute bars (feed=IEX).
3. **WebSocket** — raw protocol handshake against
   ``wss://stream.data.alpaca.markets/v2/iex`` followed by a short
   trades+bars subscription; counts messages and reports auth errors verbatim.

Never prints the secret. Read-only: it places no orders and touches no account
state. Exit codes:

* ``0`` — at least one data endpoint returned data
* ``1`` — authentication failed (HTTP 401/403 or WS ``auth failed``)
* ``2`` — other failure (network, subscription, unknown)

Usage::

    python -m intraday_scalper.tools.test_alpaca_feed --check all
    python -m intraday_scalper.tools.test_alpaca_feed --check rest
    python -m intraday_scalper.tools.test_alpaca_feed --check ws --ws-seconds 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"
SYMBOLS = ("SOXL", "SOXS")
WS_URL = "wss://stream.data.alpaca.markets/v2/iex"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_env(path: Path = DEFAULT_ENV_FILE) -> Dict[str, str]:
    """Parse the git-ignored .env file (simple KEY=VALUE parser)."""
    values: Dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    # process environment wins (lets CI/其它 secret 注入方式覆盖 .env)
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_DATA_FEED",
                "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def resolve_credentials(env: Mapping[str, str]) -> tuple[str, str, str]:
    key = env.get("ALPACA_API_KEY") or env.get("APCA_API_KEY_ID") or ""
    secret = env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY") or ""
    feed = (env.get("ALPACA_DATA_FEED") or "iex").strip().lower()
    return key.strip(), secret.strip(), feed


def masked_identity(key: str) -> str:
    if not key:
        return "<missing>"
    return f"{key[:2]}***{key[-2:]} ({len(key)} chars)"


# --------------------------------------------------------------------------- #
# REST checks (official SDK)
# --------------------------------------------------------------------------- #
def check_account(key: str, secret: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": "paper account", "ok": False}
    try:
        from alpaca.trading.client import TradingClient

        client = TradingClient(key, secret, paper=True)
        account = client.get_account()
        result.update(
            ok=True,
            status=str(account.status),
            account_number=account.account_number,
            buying_power=str(account.buying_power),
            currency=str(account.currency),
        )
    except Exception as exc:  # noqa: BLE001 - probe reports any failure
        result.update(error=f"{type(exc).__name__}: {_short(exc)}")
    return result


def check_rest(key: str, secret: str, feed: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import (
            StockBarsRequest,
            StockLatestBarRequest,
            StockLatestQuoteRequest,
            StockLatestTradeRequest,
        )
        from alpaca.data.timeframe import TimeFrame
    except Exception as exc:  # noqa: BLE001
        return [{"name": "alpaca-py import", "ok": False, "error": f"{type(exc).__name__}: {exc}"}]

    feed_enum = DataFeed.IEX if feed == "iex" else DataFeed.SIP
    client = StockHistoricalDataClient(key, secret)

    try:
        payload = client.get_stock_latest_bar(
            StockLatestBarRequest(symbol_or_symbols=list(SYMBOLS), feed=feed_enum)
        )
        rows = {}
        for symbol, bar in payload.items():
            rows[symbol] = {
                "t": bar.timestamp.astimezone(timezone.utc).isoformat(),
                "o": float(bar.open),
                "h": float(bar.high),
                "l": float(bar.low),
                "c": float(bar.close),
                "v": float(bar.volume),
            }
        results.append({"name": "latest 1m bar", "ok": True, "data": rows})
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "latest 1m bar", "ok": False, "error": f"{type(exc).__name__}: {_short(exc)}"})

    try:
        payload = client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=list(SYMBOLS), feed=feed_enum)
        )
        rows = {}
        for symbol, quote in payload.items():
            rows[symbol] = {
                "t": quote.timestamp.astimezone(timezone.utc).isoformat(),
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "bid_size": float(quote.bid_size),
                "ask_size": float(quote.ask_size),
            }
        results.append({"name": "latest quote", "ok": True, "data": rows})
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "latest quote", "ok": False, "error": f"{type(exc).__name__}: {_short(exc)}"})

    try:
        payload = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=list(SYMBOLS), feed=feed_enum)
        )
        rows = {}
        for symbol, trade in payload.items():
            rows[symbol] = {
                "t": trade.timestamp.astimezone(timezone.utc).isoformat(),
                "price": float(trade.price),
                "size": float(trade.size),
            }
        results.append({"name": "latest trade", "ok": True, "data": rows})
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "latest trade", "ok": False, "error": f"{type(exc).__name__}: {_short(exc)}"})

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)
        payload = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=list(SYMBOLS),
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                limit=5,
                feed=feed_enum,
            )
        )
        rows = {}
        data = getattr(payload, "data", None) or {}
        for symbol, bars in data.items():
            rows[symbol] = [
                {
                    "t": b.timestamp.astimezone(timezone.utc).isoformat(),
                    "o": float(b.open), "h": float(b.high), "l": float(b.low),
                    "c": float(b.close), "v": float(b.volume),
                }
                for b in bars[:5]
            ]
        results.append({"name": "recent 1m bars", "ok": bool(rows), "data": rows})
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "recent 1m bars", "ok": False, "error": f"{type(exc).__name__}: {_short(exc)}"})
    return results


# --------------------------------------------------------------------------- #
# WebSocket check (raw protocol, so auth errors are unambiguous)
# --------------------------------------------------------------------------- #
async def check_ws(key: str, secret: str, seconds: float) -> Dict[str, Any]:
    try:
        import websockets
    except Exception as exc:  # noqa: BLE001
        return {"name": "websocket", "ok": False, "error": f"websockets import failed: {exc}"}

    result: Dict[str, Any] = {
        "name": "websocket",
        "url": WS_URL,
        "ok": False,
        "counts": {},
        "samples": [],
        "auth_ok": False,
        "error": "",
    }
    counter: Dict[str, int] = {}
    try:
        async with websockets.connect(WS_URL, open_timeout=10, close_timeout=3) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if not isinstance(first, list):
                first = [first]
            if any(item.get("T") == "error" for item in first if isinstance(item, dict)):
                result["error"] = str(first)
                return result
            await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if isinstance(reply, dict):
                reply = [reply]
            for item in reply:
                if not isinstance(item, dict):
                    continue
                if item.get("T") == "error":
                    result["error"] = f"auth failed: code={item.get('code')} msg={item.get('msg')}"
                    return result
                if item.get("T") in {"success", "authorization"}:
                    result["auth_ok"] = str(item.get("msg", "")).lower() in {
                        "connected", "authenticated", "authorized"
                    } or item.get("T") == "authorization"
            if not result["auth_ok"]:
                result["error"] = f"unexpected auth reply: {reply}"
                return result

            await ws.send(
                json.dumps(
                    {
                        "action": "subscribe",
                        "trades": list(SYMBOLS),
                        "bars": list(SYMBOLS),
                        "quotes": list(SYMBOLS),
                    }
                )
            )
            deadline = asyncio.get_running_loop().time() + seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                for item in _as_list(raw):
                    kind = str(item.get("T", "?"))
                    counter[kind] = counter.get(kind, 0) + 1
                    if len(result["samples"]) < 5:
                        result["samples"].append(item)
        result["ok"] = True
        result["counts"] = counter
        result["error"] = "" if counter else "authenticated but no market data in window（可能为盘前/休市或 IEX 无成交）"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {_short(exc)}"
    return result


def _as_list(payload: Any) -> List[Any]:
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    return payload if isinstance(payload, list) else [payload]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _short(exc: Exception, limit: int = 220) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def classify(results: List[Dict[str, Any]]) -> int:
    errors = [str(r.get("error", "")) for r in results]
    joined = " ".join(errors).lower()
    if "401" in joined or "unauthorized" in joined or "auth failed" in joined or "403" in joined:
        return 1
    if any(r.get("ok") for r in results):
        return 0
    return 2


def run(check: str, ws_seconds: float) -> int:
    try:  # Windows 控制台默认 GBK，中文/符号输出需显式 UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - reconfigure is best effort
        pass
    env = load_env()
    key, secret, feed = resolve_credentials(env)
    print("=" * 78)
    print("Alpaca Markets IEX 数据源连通性探测")
    print(f"  API Key : {masked_identity(key)}")
    print(f"  Secret  : {'[set]' if secret else '<missing>'}")
    print(f"  Feed    : {feed}")
    print(f"  .env    : {DEFAULT_ENV_FILE}")
    print("=" * 78)
    if not key or not secret:
        print("[FAIL] 缺少 ALPACA_API_KEY / ALPACA_SECRET_KEY：请写入 intraday_scalper/.env")
        return 1

    results: List[Dict[str, Any]] = []
    if check in {"rest", "all"}:
        results.append(check_account(key, secret))
        results.extend(check_rest(key, secret, feed))
    if check in {"ws", "all"}:
        ws_result = asyncio.run(check_ws(key, secret, ws_seconds))
        results.append(ws_result)

    for item in results:
        status = "OK" if item.get("ok") else "FAIL"
        print(f"\n[{status}] {item['name']}")
        if item.get("error"):
            print(f"    error: {item['error']}")
        if item.get("data"):
            print("    " + json.dumps(item["data"], ensure_ascii=False, indent=2).replace("\n", "\n    "))
        if item.get("counts") is not None and item["name"] == "websocket":
            print(f"    count: {json.dumps(item['counts'], ensure_ascii=False)}")
            if item.get("samples"):
                print(f"    sample: {json.dumps(item['samples'][:2], ensure_ascii=False, default=str)[:500]}")

    code = classify(results)
    print("\n" + "=" * 78)
    if code == 0:
        print("结论：至少一个数据端点返回了数据（其余见上方明细）。")
    elif code == 1:
        print("结论：AUTH FAILED —— 凭证无效/被撤销/不属于该环境，数据源不可用。")
    else:
        print("结论：FAILED —— 网络、订阅或未知错误，详见上方明细。")
    print("=" * 78)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Alpaca IEX data-feed connectivity probe (read-only).")
    parser.add_argument("--check", choices=["rest", "ws", "all"], default="all")
    parser.add_argument("--ws-seconds", type=float, default=15.0, help="WebSocket observation window")
    args = parser.parse_args(argv)
    return run(args.check, args.ws_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
