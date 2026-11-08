"""Finnhub real-time trade websocket client.

Design
------
* The client owns one background thread. The thread opens a websocket through an
  injectable *connector*, subscribes to the configured symbols and stays in
  ``run_forever`` until the socket drops or :meth:`stop` is requested.
* All callbacks (``on_tick`` / ``on_state``) are invoked from that background
  thread. They are wrapped so that a raising consumer can never kill the
  websocket loop.
* Reconnects use the exponential backoff configured in :class:`DataConfig`
  (``reconnect_backoff_initial_seconds`` * ``reconnect_backoff_factor`` **
  (attempt-1), capped at ``reconnect_backoff_max_seconds``) up to
  ``max_reconnect_attempts``. After that the state becomes
  ``ConnectionState.UNAVAILABLE`` and the thread exits.
* Tests inject a fake connector: no network, no real clock, no sleeps. The only
  contract a connector must honour is

      connector(WsConnectOptions(url, ping_interval_seconds, open_timeout_seconds),
                WsHandlers(on_open, on_message, on_error, on_close)) -> connection

  where ``connection`` exposes ``send(str)``, ``close()`` and a blocking
  ``run_forever()`` that drives the callbacks. Callbacks must only be fired from
  inside ``run_forever`` (the real WebSocketApp behaves that way).

The API token is passed to Finnhub as a query parameter, as required by their
protocol. It is never logged and never included in ``repr()``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, Sequence

from intraday_scalper.core import UTC, ConnectionState, DataConfig, Tick, now_utc

logger = logging.getLogger(__name__)

__all__ = [
    "FinnhubWebSocketClient",
    "WsConnectOptions",
    "WsConnection",
    "WsHandlers",
    "websocket_connector",
]

_AUTH_ERROR_MARKERS = (
    "401",
    "403",
    "unauthor",
    "forbidden",
    "invalid token",
    "authentication",
)


@dataclass(frozen=True, slots=True)
class WsHandlers:
    """Callbacks handed to a connector; fired from the websocket thread."""

    on_open: Callable[[], None]
    on_message: Callable[[str], None]
    on_error: Callable[[Any], None]
    on_close: Callable[[], None]


@dataclass(frozen=True, slots=True)
class WsConnectOptions:
    """Everything a connector needs to open the socket."""

    url: str
    ping_interval_seconds: float = 20.0
    open_timeout_seconds: float = 15.0


class WsConnection(Protocol):
    """Minimal blocking connection contract used by the client."""

    def send(self, payload: str) -> None:  # pragma: no cover - protocol only
        ...

    def close(self) -> None:  # pragma: no cover - protocol only
        ...

    def run_forever(self) -> None:  # pragma: no cover - protocol only
        ...


Connector = Callable[[WsConnectOptions, WsHandlers], WsConnection]


def websocket_connector(options: WsConnectOptions, handlers: WsHandlers) -> WsConnection:
    """Default connector built on ``websocket-client``'s ``WebSocketApp``.

    ``open_timeout_seconds`` is applied through the library's process-wide
    default socket timeout *only while connecting* and is reset immediately in
    ``on_open``/``finally`` so the steady-state read loop keeps blocking.
    """
    try:
        import websocket
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "websocket-client is required for live Finnhub data; "
            "install it with `pip install websocket-client`"
        ) from exc

    def _restore_default_timeout() -> None:
        try:
            websocket.setdefaulttimeout(None)
        except Exception:  # pragma: no cover - defensive
            pass

    def _on_open(ws: Any) -> None:
        try:
            if ws.sock is not None:
                ws.sock.settimeout(None)
        except Exception:  # pragma: no cover - defensive
            pass
        _restore_default_timeout()
        handlers.on_open()

    def _on_message(ws: Any, message: Any) -> None:
        handlers.on_message(message)

    def _on_error(ws: Any, error: Any) -> None:
        handlers.on_error(error)

    def _on_close(ws: Any, status_code: Any = None, message: Any = None) -> None:
        handlers.on_close()

    app = websocket.WebSocketApp(
        options.url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )

    class _WebSocketAppConnection:
        def send(self, payload: str) -> None:
            if app.sock is None:
                raise ConnectionError("websocket is not connected")
            app.send(payload)

        def close(self) -> None:
            try:
                app.close()
            except Exception:  # pragma: no cover - best effort shutdown
                pass

        def run_forever(self) -> None:
            ping = float(options.ping_interval_seconds or 0.0)
            try:
                if options.open_timeout_seconds and options.open_timeout_seconds > 0:
                    websocket.setdefaulttimeout(float(options.open_timeout_seconds))
                app.run_forever(
                    ping_interval=ping if ping > 0 else 0,
                    ping_timeout=max(ping / 2.0, 1.0) if ping > 0 else None,
                )
            finally:
                _restore_default_timeout()

    return _WebSocketAppConnection()


class FinnhubWebSocketClient:
    """Threaded Finnhub trade websocket client with reconnect + resubscribe."""

    def __init__(
        self,
        token: str,
        symbols: Sequence[str],
        data_config: DataConfig,
        on_tick: Callable[[Tick], None],
        on_state: Callable[[ConnectionState, str], None] | None = None,
        clock: Callable[[], datetime] = now_utc,
        connector: Connector | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Finnhub websocket token must be a non-empty string")
        normalised = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
        if not normalised:
            raise ValueError("Finnhub websocket needs at least one symbol")

        self._token = token.strip()
        self._symbols = normalised
        self._symbol_set = frozenset(normalised)
        self._data = data_config
        self._on_tick = on_tick
        self._on_state = on_state
        self._clock = clock
        self._connector: Connector = connector or websocket_connector
        self._sleep = sleep_fn

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn: WsConnection | None = None

        self._state = ConnectionState.DISCONNECTED
        self._last_message_utc: datetime | None = None
        self._message_count = 0
        self._tick_count = 0
        self._malformed_count = 0
        self._skipped_symbol_count = 0
        self._skipped_volume_count = 0
        self._handler_errors = 0
        self._reconnect_count = 0
        self._fatal_error: str | None = None
        self._last_error = ""

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def last_message_utc(self) -> datetime | None:
        with self._lock:
            return self._last_message_utc

    @property
    def message_count(self) -> int:
        with self._lock:
            return self._message_count

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def malformed_count(self) -> int:
        with self._lock:
            return self._malformed_count

    @property
    def skipped_symbol_count(self) -> int:
        with self._lock:
            return self._skipped_symbol_count

    @property
    def skipped_volume_count(self) -> int:
        with self._lock:
            return self._skipped_volume_count

    @property
    def handler_errors(self) -> int:
        with self._lock:
            return self._handler_errors

    @property
    def reconnect_count(self) -> int:
        with self._lock:
            return self._reconnect_count

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def __repr__(self) -> str:  # never show the token
        return (
            f"FinnhubWebSocketClient(symbols={self._symbols!r}, "
            f"state={self.state.value}, ticks={self.tick_count})"
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background thread; returns immediately."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._connected_event.clear()
            self._fatal_error = None
            self._thread = threading.Thread(
                target=self._run_loop, name="finnhub-ws", daemon=True
            )
        self._set_state(ConnectionState.CONNECTING, "starting websocket thread")
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown: stop reconnecting and close the live socket."""
        self._stop_event.set()
        with self._lock:
            conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort shutdown
                logger.debug("error while closing websocket", exc_info=True)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the background thread to exit; returns ``True`` when it did."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def wait_until_connected(self, timeout: float | None = None) -> bool:
        """Block until the socket is open (convenience for ``main.py``)."""
        return self._connected_event.wait(timeout)

    # ------------------------------------------------------------------ #
    # Connection loop (background thread)
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        reconnects = 0
        while not self._stop_event.is_set():
            if reconnects == 0:
                self._set_state(ConnectionState.CONNECTING, "connecting")
            else:
                with self._lock:
                    self._reconnect_count = reconnects
                self._set_state(
                    ConnectionState.RECONNECTING,
                    f"reconnect attempt {reconnects}/{self._data.max_reconnect_attempts}",
                )

            handlers = WsHandlers(
                on_open=self._handle_open,
                on_message=self._handle_message,
                on_error=self._handle_error,
                on_close=self._handle_close,
            )
            options = WsConnectOptions(
                url=self._build_url(),
                ping_interval_seconds=float(self._data.ws_ping_interval_seconds),
                open_timeout_seconds=float(self._data.ws_open_timeout_seconds),
            )

            connection: WsConnection | None = None
            try:
                connection = self._connector(options, handlers)
            except Exception as exc:  # noqa: BLE001 - connectors are untrusted
                self._handle_error(exc)

            if connection is not None:
                with self._lock:
                    self._conn = connection
                try:
                    connection.run_forever()
                except Exception as exc:  # noqa: BLE001 - keep the thread alive
                    self._handle_error(exc)
                finally:
                    with self._lock:
                        self._conn = None

            if self._stop_event.is_set():
                self._set_state(ConnectionState.DISCONNECTED, "stopped")
                return
            if self._fatal_error is not None:
                return

            reconnects += 1
            if reconnects > self._data.max_reconnect_attempts:
                self._set_state(
                    ConnectionState.UNAVAILABLE,
                    "websocket unavailable: reconnect attempts exhausted "
                    f"({reconnects} connection attempts). The Finnhub account may "
                    "not have US real-time trade permissions.",
                )
                return

            delay = self._backoff_seconds(reconnects)
            self._set_state(
                ConnectionState.RECONNECTING,
                f"reconnecting in {delay:.2f}s (attempt {reconnects})",
            )
            self._wait(delay)

        self._set_state(ConnectionState.DISCONNECTED, "stopped")

    def _build_url(self) -> str:
        separator = "&" if "?" in self._data.ws_url else "?"
        return f"{self._data.ws_url}{separator}token={self._token}"

    def _backoff_seconds(self, attempt: int) -> float:
        initial = float(self._data.reconnect_backoff_initial_seconds)
        factor = float(self._data.reconnect_backoff_factor)
        maximum = float(self._data.reconnect_backoff_max_seconds)
        delay = initial * (factor ** max(attempt - 1, 0))
        return min(delay, maximum)

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        # With the real sleep the wait must stay interruptible by stop().
        if self._sleep is time.sleep:
            self._stop_event.wait(seconds)
        else:
            self._sleep(seconds)

    # ------------------------------------------------------------------ #
    # Websocket callbacks
    # ------------------------------------------------------------------ #
    def _handle_open(self) -> None:
        self._set_state(ConnectionState.CONNECTED, "websocket open")
        for symbol in self._symbols:
            payload = json.dumps({"type": "subscribe", "symbol": symbol})
            try:
                self._send(payload)
            except Exception as exc:  # noqa: BLE001
                self._handle_error(exc)
                self._close_current_connection()
                return

    def _send(self, payload: str) -> None:
        with self._lock:
            conn = self._conn
        if conn is None:
            raise ConnectionError("websocket is not connected")
        conn.send(payload)

    def _handle_message(self, raw: Any) -> None:
        with self._lock:
            self._message_count += 1
            self._last_message_utc = self._clock()

        if not isinstance(raw, str):
            self._bump_malformed("non-text websocket frame")
            return
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            self._bump_malformed("invalid JSON frame")
            return
        if not isinstance(payload, dict):
            self._bump_malformed("JSON frame is not an object")
            return

        message_type = payload.get("type")
        if message_type == "trade":
            data = payload.get("data")
            if not isinstance(data, list):
                self._bump_malformed("trade frame without data list")
                return
            for item in data:
                self._handle_trade_item(item)
        # Any other message type (ping, error, subscription confirmations, ...)
        # is ignored on purpose: it must not crash the loop.

    def _handle_trade_item(self, item: Any) -> None:
        if not isinstance(item, dict):
            self._bump_malformed("trade entry is not an object")
            return

        symbol_raw = item.get("s")
        if not isinstance(symbol_raw, str) or not symbol_raw.strip():
            self._bump_malformed("trade entry without symbol")
            return
        symbol = symbol_raw.strip().upper()
        if symbol not in self._symbol_set:
            with self._lock:
                self._skipped_symbol_count += 1
            return

        try:
            price = float(item["p"])
            volume = float(item["v"])
            ts_millis = float(item["t"])
        except (KeyError, TypeError, ValueError):
            self._bump_malformed("trade entry with non-numeric price/volume/time")
            return
        if not (math.isfinite(price) and math.isfinite(volume) and math.isfinite(ts_millis)):
            self._bump_malformed("trade entry with non-finite fields")
            return
        if volume <= 0:
            with self._lock:
                self._skipped_volume_count += 1
            return
        if price <= 0:
            self._bump_malformed("trade entry with non-positive price")
            return

        try:
            ts_utc = datetime.fromtimestamp(ts_millis / 1000.0, tz=UTC)
            tick = Tick(symbol=symbol, price=price, volume=volume, ts_utc=ts_utc)
        except (OverflowError, OSError, TypeError, ValueError):
            self._bump_malformed("trade entry outside the supported timestamp range")
            return

        with self._lock:
            self._tick_count += 1
        try:
            self._on_tick(tick)
        except Exception:  # noqa: BLE001 - consumer errors must not kill the loop
            with self._lock:
                self._handler_errors += 1
            logger.debug("on_tick callback raised", exc_info=True)

    def _handle_error(self, error: Any) -> None:
        message = str(error).strip() or type(error).__name__
        with self._lock:
            self._last_error = message[:300]

        if self._looks_like_auth_failure(message):
            with self._lock:
                self._fatal_error = message
            self._set_state(
                ConnectionState.UNAUTHENTICATED,
                f"authentication failed: {message[:120]}",
            )
            self._close_current_connection()
            return

        if not self._stop_event.is_set():
            self._set_state(
                ConnectionState.RECONNECTING, f"websocket error: {message[:120]}"
            )

    def _handle_close(self) -> None:
        with self._lock:
            self._conn = None
        if self._stop_event.is_set() or self._fatal_error is not None:
            return
        self._set_state(ConnectionState.DISCONNECTED, "websocket closed")

    def _close_current_connection(self) -> None:
        with self._lock:
            conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("error while closing websocket", exc_info=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _bump_malformed(self, reason: str) -> None:
        with self._lock:
            self._malformed_count += 1
            self._last_error = reason
        logger.debug("ignored malformed websocket payload: %s", reason)

    def _looks_like_auth_failure(self, message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in _AUTH_ERROR_MARKERS)

    def _set_state(self, state: ConnectionState, reason: str = "") -> None:
        with self._lock:
            changed = state is not self._state
            self._state = state
            callback = self._on_state
            if state is ConnectionState.CONNECTED:
                self._connected_event.set()
            else:
                self._connected_event.clear()
        if changed and callback is not None:
            try:
                callback(state, reason)
            except Exception:  # noqa: BLE001 - UI callbacks are untrusted
                with self._lock:
                    self._handler_errors += 1
                logger.debug("on_state callback raised", exc_info=True)
