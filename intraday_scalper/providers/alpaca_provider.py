"""Alpaca IEX market-data provider (Paper credentials, IEX feed only).

IMPORTANT — what "IEX data" means
---------------------------------
IEX is a *single exchange*. Its trades/bars represent only the volume that
executed on IEX, which is a small fraction of consolidated US market volume.
Every tick, 1-minute bar and VWAP produced by this provider is therefore an
**IEX-only** view: it must never be described (in logs, dashboards or reports)
as full-market or consolidated volume. Trading decisions based on this data are
consequently based on a subset of the tape.

Design notes
------------
* The only network code lives behind :class:`AlpacaIEXProvider`'s injectable
  ``stream_factory`` (and :func:`probe_alpaca`'s injectable ``client_factory``).
  Tests use fakes and never open a socket.
* Alpaca's :class:`alpaca.data.live.stock.StockDataStream` (alpaca-py 0.45) is
  wrapped in a small subclass that reports "authenticated/connected" through
  callbacks and hands reconnection back to this provider's own backoff loop
  (``DataConfig.reconnect_*`` / ``max_reconnect_attempts``). alpaca-py keeps the
  websocket protocol, authentication, decoders and frame dispatch; this module
  owns the lifecycle/state machine and normalises dict frames into the core
  :class:`~intraday_scalper.core.models.Tick` / ``Bar`` contracts.
* Secrets are handled by :class:`AlpacaCredentials` (``repr`` hides the secret),
  are read only from the environment / git-ignored ``intraday_scalper/.env``,
  and are redacted from every error string produced here.
* bars received from the stream are *already completed 1-minute bars*; they are
  passed to ``on_bar`` directly and are **not** fed into the local tick
  aggregator (that would double-count). Ticks are passed to ``on_tick`` so the
  caller can maintain IEX-only session VWAP / quality metrics / raw recording.
* If a bar frame has no ``vw`` (VWAP), the bar is skipped and counted — a
  synthetic ``(h+l+c)/3`` VWAP is never produced, because the strategy layer
  explicitly requires true volume-weighted prices.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from intraday_scalper.core import Bar, ConnectionState, DataConfig, Tick, UTC, now_utc, to_utc

logger = logging.getLogger(__name__)

__all__ = [
    "AlpacaCredentials",
    "AlpacaCredentialsError",
    "AlpacaIEXProvider",
    "AlpacaStreamHooks",
    "ProbeItem",
    "ProbeResult",
    "load_alpaca_credentials",
    "probe_alpaca",
]

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"

DEFAULT_FEED = "iex"
_SUPPORTED_FEEDS = (
    "iex",
    "sip",
    "delayed_sip",
    "otc",
    "boats",
    "overnight",
)

# Environment variable names (the APCA_* aliases are what alpaca-py itself uses).
_API_KEY_ENV_NAMES = ("ALPACA_API_KEY", "APCA_API_KEY_ID")
_SECRET_KEY_ENV_NAMES = ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
_FEED_ENV_NAMES = ("ALPACA_DATA_FEED", "APCA_DATA_FEED")

_AUTH_ERROR_MARKERS = (
    "401",
    "402",
    "403",
    "unauthorized",
    "unauthorised",
    "forbidden",
    "auth failed",
    "authentication",
    "credential",
    "invalid api",
    "invalid key",
    "not authorized",
    "permission denied",
)
_FRACTION_RE = re.compile(r"\.(\d{6})\d+")


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
class AlpacaCredentialsError(RuntimeError):
    """Alpaca credentials are missing/unreadable/invalid (never contains secrets)."""


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Alpaca market-data credentials.

    ``secret_key`` is excluded from ``repr`` so accidental logging never leaks it.
    """

    api_key: str
    secret_key: str = field(repr=False)
    feed: str = DEFAULT_FEED

    def __post_init__(self) -> None:
        api_key = str(self.api_key or "").strip()
        secret_key = str(self.secret_key or "").strip()
        feed = str(self.feed or DEFAULT_FEED).strip().lower()
        if not api_key:
            raise AlpacaCredentialsError("Alpaca api_key must not be empty")
        if not secret_key:
            raise AlpacaCredentialsError("Alpaca secret_key must not be empty")
        if not feed:
            feed = DEFAULT_FEED
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "secret_key", secret_key)
        object.__setattr__(self, "feed", feed)

    @property
    def feed_enum(self) -> str:
        """Normalised feed name restricted to the feeds alpaca-py understands."""
        return self.feed if self.feed in _SUPPORTED_FEEDS else DEFAULT_FEED


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` dotenv file (BOM/comments/quotes/export)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise AlpacaCredentialsError(
            f"cannot read Alpaca env file {path}: {exc.strerror or exc}"
        ) from exc

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _first_present(names: Sequence[str], *sources: Mapping[str, str]) -> str:
    """Return the first non-empty value, preferring earlier *sources*."""
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def load_alpaca_credentials(env_file: Path | None = None) -> AlpacaCredentials:
    """Load Alpaca credentials from the environment, then ``intraday_scalper/.env``.

    Field-level precedence: process environment first (``ALPACA_API_KEY`` /
    ``ALPACA_SECRET_KEY`` / ``ALPACA_DATA_FEED``, plus the ``APCA_*`` aliases),
    then the dotenv file. Raises :class:`AlpacaCredentialsError` when the API key
    or secret is missing; error messages contain variable names and the checked
    path only — never a credential value.
    """
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    file_values = _read_env_file(path)
    env_values = dict(os.environ)

    api_key = _first_present(_API_KEY_ENV_NAMES, env_values, file_values)
    secret_key = _first_present(_SECRET_KEY_ENV_NAMES, env_values, file_values)
    feed = _first_present(_FEED_ENV_NAMES, env_values, file_values) or DEFAULT_FEED

    missing = []
    if not api_key:
        missing.append("ALPACA_API_KEY")
    if not secret_key:
        missing.append("ALPACA_SECRET_KEY")
    if missing:
        raise AlpacaCredentialsError(
            "missing Alpaca credentials: "
            + ", ".join(missing)
            + f" (checked the process environment and {path})"
        )
    return AlpacaCredentials(api_key=api_key, secret_key=secret_key, feed=feed)


# --------------------------------------------------------------------------- #
# Error helpers (safe for logs/UI)
# --------------------------------------------------------------------------- #
def _redact(text: str, credentials: AlpacaCredentials | None) -> str:
    """Remove credential values from a message and truncate it."""
    message = str(text or "")
    if credentials is not None:
        for secret in (credentials.secret_key, credentials.api_key):
            if secret and len(secret) >= 4:
                message = message.replace(secret, "***")
    return message[:500]


def _describe_error(error: Any, credentials: AlpacaCredentials | None = None) -> str:
    status = getattr(error, "status_code", None)
    prefix = f"HTTP {status} " if status else ""
    return _redact(f"{prefix}{type(error).__name__}: {error}", credentials)


def _looks_like_auth_error(value: Any) -> bool:
    """True for normalised 401/402/403-ish auth failures."""
    text = str(value or "").lower()
    status = getattr(value, "status_code", None)
    if status in (401, 402, 403):
        return True
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


def _normalise_error(value: Any) -> str:
    return "auth" if _looks_like_auth_error(value) else "error"


# --------------------------------------------------------------------------- #
# PROBE (one-shot REST validation for `main.py check`)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProbeItem:
    """Result of one probe call (latest bar / quote / trade)."""

    name: str
    ok: bool
    error: str = ""
    normalized_error: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_auth_error(self) -> bool:
        return self.normalized_error == "auth"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "error": self.error,
            "normalized_error": self.normalized_error,
            "data": dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Structured result of :func:`probe_alpaca` (never contains secrets)."""

    feed: str
    symbols: tuple[str, ...]
    bars: ProbeItem
    quotes: ProbeItem
    trades: ProbeItem

    @property
    def items(self) -> tuple[ProbeItem, ...]:
        return (self.bars, self.quotes, self.trades)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.items)

    @property
    def any_ok(self) -> bool:
        return any(item.ok for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "any_ok": self.any_ok,
            "feed": self.feed,
            "symbols": list(self.symbols),
            "bars": self.bars.to_dict(),
            "quotes": self.quotes.to_dict(),
            "trades": self.trades.to_dict(),
        }


def _default_historical_client(credentials: AlpacaCredentials) -> Any:
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise RuntimeError(
            "alpaca-py is required for probe_alpaca; install it with `pip install alpaca-py`"
        ) from exc
    return StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )


def _data_feed(credentials: AlpacaCredentials) -> Any:
    from alpaca.data.enums import DataFeed

    try:
        return DataFeed(credentials.feed_enum)
    except ValueError:  # pragma: no cover - guarded by feed_enum
        return DataFeed.IEX


def _make_latest_request(request_name: str, symbols: Sequence[str], credentials: AlpacaCredentials) -> Any:
    from alpaca.data import requests as data_requests

    request_class = getattr(data_requests, request_name)
    try:
        return request_class(symbol_or_symbols=list(symbols), feed=_data_feed(credentials))
    except TypeError:  # pragma: no cover - older/newer alpaca variants
        return request_class(symbol_or_symbols=list(symbols))


def _jsonable(value: Any) -> Any:
    """Convert pydantic models / datetimes / enums into JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return to_utc(value).isoformat()
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:  # pragma: no cover - pydantic v1 fallback
            return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def probe_alpaca(
    credentials: AlpacaCredentials,
    *,
    symbols: Sequence[str] = ("SOXL", "SOXS"),
    client_factory: Callable[[AlpacaCredentials], Any] | None = None,
) -> ProbeResult:
    """One-shot REST check: latest 1m bar / quote / trade via alpaca-py.

    Never raises for network/API errors: each item carries ``ok``/``error``/
    ``normalized_error`` (``"auth"`` for 401/402/403-ish failures) and JSON-safe
    ``data``. ``client_factory`` is injectable for offline tests; it must accept
    the credentials and return a ``StockHistoricalDataClient``-like object.
    """
    normalised: list[str] = []
    for symbol in symbols:
        name = str(symbol).strip().upper()
        if name and name not in normalised:
            normalised.append(name)
    if not normalised:
        raise ValueError("symbols must not be empty")

    client: Any = None
    client_error = ""
    client_normalized = "error"
    try:
        factory = client_factory if client_factory is not None else _default_historical_client
        client = factory(credentials)
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        client_error = _describe_error(exc, credentials)
        client_normalized = _normalise_error(exc)

    checks = (
        ("bars", "get_stock_latest_bar", "StockLatestBarRequest"),
        ("quotes", "get_stock_latest_quote", "StockLatestQuoteRequest"),
        ("trades", "get_stock_latest_trade", "StockLatestTradeRequest"),
    )
    items: dict[str, ProbeItem] = {}
    for name, method_name, request_name in checks:
        if client is None:
            items[name] = ProbeItem(
                name=name,
                ok=False,
                error=client_error,
                normalized_error=client_normalized,
            )
            continue
        try:
            request = _make_latest_request(request_name, normalised, credentials)
            response = getattr(client, method_name)(request)
            items[name] = ProbeItem(name=name, ok=True, data=_jsonable(response) or {})
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            items[name] = ProbeItem(
                name=name,
                ok=False,
                error=_describe_error(exc, credentials),
                normalized_error=_normalise_error(exc),
            )

    return ProbeResult(
        feed=credentials.feed_enum,
        symbols=tuple(normalised),
        bars=items["bars"],
        quotes=items["quotes"],
        trades=items["trades"],
    )


# --------------------------------------------------------------------------- #
# Stream hook contract (injectable for offline tests)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AlpacaStreamHooks:
    """Callbacks a stream implementation must fire from its own thread.

    ``on_connected``: after connect + auth succeeded (and before data flows).
    ``on_auth_failed``: on non-recoverable authentication failure (401/402/403).
    """

    on_connected: Callable[[], None]
    on_auth_failed: Callable[[str], None]


class _StreamReconnect(Exception):
    """Internal signal: the stream run ended and should be retried."""


class _StreamAuthFailed(Exception):
    """Internal signal: the stream authentication failed permanently."""


def _default_stream_factory(credentials: AlpacaCredentials, hooks: AlpacaStreamHooks) -> Any:
    """Build a real alpaca-py ``StockDataStream`` wrapped in our lifecycle hooks."""
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.live.stock import StockDataStream
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise RuntimeError(
            "alpaca-py is required for the live Alpaca IEX provider; "
            "install it with `pip install alpaca-py`"
        ) from exc

    class _HookableStockDataStream(StockDataStream):
        """StockDataStream whose reconnect policy belongs to our supervisor."""

        async def _start_ws(self) -> None:
            try:
                await super()._start_ws()
            except Exception as exc:  # noqa: BLE001 - classify then re-raise
                message = _describe_error(exc, credentials)
                if _looks_like_auth_error(exc):
                    hooks.on_auth_failed(message)
                    raise _StreamAuthFailed(message) from exc
                raise
            hooks.on_connected()

        async def _dispatch(self, msg: Any) -> None:
            # alpaca-py logs ``{"T":"error"}`` frames but keeps the socket open;
            # auth/subscription errors (401/402/403) must instead be fatal.
            if isinstance(msg, Mapping) and msg.get("T") == "error":
                candidate = f"{msg.get('code', '')} {msg.get('msg', '')}".strip()
                if _looks_like_auth_error(candidate):
                    hooks.on_auth_failed(_redact(candidate, credentials) or "authentication failed")
                    self.stop()
                    return
            await super()._dispatch(msg)

        async def _wait_before_reconnect(self, retries: int, run: Any) -> None:
            # alpaca-py retries internally forever; instead surface the
            # disconnect to AlpacaIEXProvider, which owns the configured
            # backoff / max-attempt policy and re-subscribes explicitly.
            if run is not None and getattr(run, "stopping", False):
                return
            raise _StreamReconnect(f"connection lost after {retries} attempt(s)")

    return _HookableStockDataStream(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
        raw_data=True,
        feed=DataFeed.IEX,
    )


# --------------------------------------------------------------------------- #
# Frame parsing (pure, offline-testable)
# --------------------------------------------------------------------------- #
class _MalformedMessage(ValueError):
    """A websocket frame that cannot be normalised; counted, never fatal."""


class _MissingVWAP(_MalformedMessage):
    """A 1m bar frame without a true VWAP; skipped (never synthesised)."""


def _require_mapping(message: Any) -> Mapping[str, Any]:
    if not isinstance(message, Mapping):
        raise _MalformedMessage(f"frame is not a mapping ({type(message).__name__})")
    return message


def _positive_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _MalformedMessage(f"{name} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise _MalformedMessage(f"{name} must be a positive finite number")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _MalformedMessage(f"{name} is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise _MalformedMessage(f"{name} must be a non-negative finite number")
    return number


def _symbol_from_message(message: Mapping[str, Any]) -> str:
    symbol = message.get("S")
    if not isinstance(symbol, str) or not symbol.strip():
        raise _MalformedMessage("frame has no symbol")
    return symbol.strip().upper()


def _coerce_utc_timestamp(value: Any, name: str = "timestamp") -> datetime:
    """Parse ISO-8601/RFC3339, msgpack ``Timestamp``, epoch seconds/millis, datetime."""
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _MalformedMessage(f"{name} is empty")
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        text = _FRACTION_RE.sub(r".\1", text)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise _MalformedMessage(f"{name} is not valid ISO-8601: {value!r}") from exc
        return to_utc(parsed)
    if isinstance(value, bool):
        raise _MalformedMessage(f"{name} is a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise _MalformedMessage(f"{name} is not finite")
        if abs(number) >= 1e12:
            number /= 1000.0  # milliseconds since epoch
        try:
            return datetime.fromtimestamp(number, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise _MalformedMessage(f"{name} epoch value is out of range") from exc
    to_datetime = getattr(value, "to_datetime", None)
    if callable(to_datetime):
        try:
            parsed = to_datetime()
        except Exception as exc:  # noqa: BLE001 - msgpack Timestamp
            raise _MalformedMessage(f"{name} cannot be converted") from exc
        if isinstance(parsed, datetime):
            return to_utc(parsed)
    raise _MalformedMessage(f"{name} is not a supported timestamp ({type(value).__name__})")


def _trade_from_message(message: Any) -> Tick:
    """``{"T":"t","S":...,"p":...,"s":...,"t":...}`` -> core.Tick (IEX-only)."""
    data = _require_mapping(message)
    if data.get("T") != "t":
        raise _MalformedMessage(f"not a trade frame (T={data.get('T')!r})")
    symbol = _symbol_from_message(data)
    price = _positive_float(data.get("p"), "trade price")
    volume = _non_negative_float(data.get("s"), "trade size")
    ts_utc = _coerce_utc_timestamp(data.get("t"), "trade timestamp")
    try:
        return Tick(symbol=symbol, price=price, volume=volume, ts_utc=ts_utc, trade_id=None)
    except (TypeError, ValueError) as exc:
        raise _MalformedMessage(str(exc)) from exc


def _bar_from_message(message: Any) -> Bar:
    """``{"T":"b","S":...,"o,h,l,c,v,vw","t"}`` -> completed core.Bar (IEX-only)."""
    data = _require_mapping(message)
    if data.get("T") != "b":
        raise _MalformedMessage(f"not a bar frame (T={data.get('T')!r})")
    symbol = _symbol_from_message(data)
    open_ = _positive_float(data.get("o"), "bar open")
    high = _positive_float(data.get("h"), "bar high")
    low = _positive_float(data.get("l"), "bar low")
    close = _positive_float(data.get("c"), "bar close")
    volume = _non_negative_float(data.get("v"), "bar volume")
    vwap = data.get("vw")
    if vwap is None:
        raise _MissingVWAP("bar frame has no true VWAP (vw)")
    vwap_value = _positive_float(vwap, "bar vwap")
    start_utc = _coerce_utc_timestamp(data.get("t"), "bar timestamp")
    try:
        return Bar(
            symbol=symbol,
            timeframe_minutes=1,
            start_utc=start_utc,
            end_utc=start_utc + timedelta(minutes=1),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            vwap=vwap_value,
            trade_count=0,
            is_complete=True,
        )
    except (TypeError, ValueError) as exc:
        raise _MalformedMessage(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class AlpacaIEXProvider:
    """Threaded Alpaca IEX trade+bar stream with reconnect/state supervision.

    ``stream_factory`` (injectable for offline tests) is called as
    ``factory(credentials, hooks)`` and must return a stream-like object with
    ``subscribe_trades(handler, *symbols)``, ``subscribe_bars(handler, *symbols)``,
    ``run()`` (blocking; may return on disconnect) and ``stop()``. The handlers
    are coroutine functions awaited by the stream in its own event-loop thread.
    The default factory returns an instrumented alpaca-py ``StockDataStream``.
    """

    def __init__(
        self,
        credentials: AlpacaCredentials,
        data_config: DataConfig,
        *,
        on_tick: Callable[[Tick], None],
        on_bar: Callable[[Bar], None] | None = None,
        on_state: Callable[[ConnectionState, str], None] | None = None,
        clock: Callable[[], datetime] = now_utc,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not callable(on_tick):
            raise TypeError("on_tick must be a callable")
        if on_bar is not None and not callable(on_bar):
            raise TypeError("on_bar must be a callable")
        if on_state is not None and not callable(on_state):
            raise TypeError("on_state must be a callable")
        if not data_config.symbols:
            raise ValueError("data_config.symbols must not be empty")

        self._credentials = credentials
        self._data = data_config
        self._symbols = tuple(str(s).strip().upper() for s in data_config.symbols)
        self._symbol_set = frozenset(self._symbols)
        self._on_tick = on_tick
        self._on_bar = on_bar
        self._on_state = on_state
        self._clock = clock
        self._stream_factory = stream_factory if stream_factory is not None else _default_stream_factory

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._state = ConnectionState.DISCONNECTED
        self._auth_failure = ""
        self._last_error = ""
        self._last_message_utc: datetime | None = None
        self._tick_count = 0
        self._bar_count = 0
        self._reconnect_count = 0
        self._connect_attempts = 0
        self._malformed_count = 0
        self._skipped_symbol_count = 0
        self._skipped_no_vwap_count = 0
        self._handler_errors = 0

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def feed(self) -> str:
        return self._credentials.feed_enum

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def last_message_utc(self) -> datetime | None:
        with self._lock:
            return self._last_message_utc

    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def bar_count(self) -> int:
        with self._lock:
            return self._bar_count

    @property
    def reconnect_count(self) -> int:
        with self._lock:
            return self._reconnect_count

    @property
    def connect_attempts(self) -> int:
        with self._lock:
            return self._connect_attempts

    @property
    def malformed_count(self) -> int:
        with self._lock:
            return self._malformed_count

    @property
    def skipped_symbol_count(self) -> int:
        with self._lock:
            return self._skipped_symbol_count

    @property
    def skipped_no_vwap_count(self) -> int:
        with self._lock:
            return self._skipped_no_vwap_count

    @property
    def handler_errors(self) -> int:
        with self._lock:
            return self._handler_errors

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def __repr__(self) -> str:  # never show credentials
        return (
            f"AlpacaIEXProvider(feed={self.feed!r}, symbols={self._symbols!r}, "
            f"state={self.state.value}, ticks={self.tick_count}, bars={self.bar_count})"
        )

    def wait_until_connected(self, timeout: float | None = None) -> bool:
        """Block until the stream authenticates (convenience for ``main.py``)."""
        return self._connected_event.wait(timeout)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the background supervisor thread; returns immediately."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._connected_event.clear()
            self._auth_failure = ""
            self._thread = threading.Thread(
                target=self._run_loop, name="alpaca-iex", daemon=True
            )
        self._set_state(ConnectionState.CONNECTING, "starting Alpaca IEX provider")
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown (safe from any thread; idempotent)."""
        self._stop_event.set()
        with self._lock:
            stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # pragma: no cover - best effort shutdown
                logger.debug("error while stopping Alpaca stream", exc_info=True)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the supervisor thread to exit; returns ``True`` when it did."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    # ------------------------------------------------------------------ #
    # Supervisor loop (background thread)
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        reconnects = 0
        while not self._stop_event.is_set():
            attempt = reconnects + 1
            with self._lock:
                self._connect_attempts = attempt
            if attempt == 1:
                self._set_state(ConnectionState.CONNECTING, "connecting to Alpaca IEX stream")
            else:
                with self._lock:
                    self._reconnect_count = reconnects
                self._set_state(
                    ConnectionState.RECONNECTING,
                    f"reconnect attempt {reconnects}/{self._data.max_reconnect_attempts}",
                )

            hooks = AlpacaStreamHooks(
                on_connected=self._handle_stream_connected,
                on_auth_failed=self._handle_stream_auth_failed,
            )
            stream: Any = None
            try:
                stream = self._stream_factory(self._credentials, hooks)
            except Exception as exc:  # noqa: BLE001 - factory is untrusted
                self._record_error(exc)

            if stream is not None:
                with self._lock:
                    self._stream = stream
                try:
                    self._subscribe(stream)
                    if not self._stop_event.is_set():
                        stream.run()  # blocks until stopped / disconnect / auth failure
                except Exception as exc:  # noqa: BLE001 - stream errors are recoverable
                    self._record_error(exc)
                finally:
                    with self._lock:
                        self._stream = None

            if self._stop_event.is_set():
                self._set_state(ConnectionState.DISCONNECTED, "stopped")
                return
            if self._auth_failure:
                # State was already set to UNAUTHENTICATED by the hook.
                return

            reconnects += 1
            if reconnects > self._data.max_reconnect_attempts:
                self._set_state(
                    ConnectionState.UNAVAILABLE,
                    "Alpaca IEX stream unavailable: reconnect attempts exhausted "
                    f"({reconnects} attempts)",
                )
                return
            delay = self._backoff_seconds(reconnects)
            self._set_state(
                ConnectionState.RECONNECTING,
                f"reconnecting in {delay:.2f}s (attempt {reconnects}/{self._data.max_reconnect_attempts})",
            )
            self._wait(delay)

        self._set_state(ConnectionState.DISCONNECTED, "stopped")

    def _subscribe(self, stream: Any) -> None:
        stream.subscribe_trades(self._async_trade_handler, *self._symbols)
        stream.subscribe_bars(self._async_bar_handler, *self._symbols)

    def _backoff_seconds(self, attempt: int) -> float:
        initial = float(self._data.reconnect_backoff_initial_seconds)
        factor = float(self._data.reconnect_backoff_factor)
        maximum = float(self._data.reconnect_backoff_max_seconds)
        return min(initial * (factor ** max(attempt - 1, 0)), maximum)

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._stop_event.wait(seconds)

    # ------------------------------------------------------------------ #
    # Stream hooks
    # ------------------------------------------------------------------ #
    def _handle_stream_connected(self) -> None:
        self._set_state(ConnectionState.CONNECTED, "Alpaca IEX stream authenticated")

    def _handle_stream_auth_failed(self, message: str) -> None:
        safe = _redact(message, self._credentials) or "authentication failed"
        with self._lock:
            self._auth_failure = safe
        self._set_state(ConnectionState.UNAUTHENTICATED, f"authentication failed: {safe}")

    # ------------------------------------------------------------------ #
    # Frame handlers (called from the stream's event-loop thread)
    # ------------------------------------------------------------------ #
    async def _async_trade_handler(self, message: Any) -> None:
        self._process_trade_message(message)

    async def _async_bar_handler(self, message: Any) -> None:
        self._process_bar_message(message)

    def _process_trade_message(self, message: Any) -> None:
        symbol = None
        if isinstance(message, Mapping):
            raw_symbol = message.get("S")
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                symbol = raw_symbol.strip().upper()
        if symbol is not None and symbol not in self._symbol_set:
            self._increment("skipped_symbol_count")
            return
        try:
            tick = _trade_from_message(message)
        except _MalformedMessage as exc:
            self._increment("malformed_count")
            logger.debug("ignored malformed Alpaca trade frame: %s", exc)
            return
        self._record_message("tick")
        try:
            self._on_tick(tick)
        except Exception:  # noqa: BLE001 - consumer errors must not kill the stream
            self._increment("handler_errors")
            logger.debug("on_tick callback raised", exc_info=True)

    def _process_bar_message(self, message: Any) -> None:
        symbol = None
        if isinstance(message, Mapping):
            raw_symbol = message.get("S")
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                symbol = raw_symbol.strip().upper()
        if symbol is not None and symbol not in self._symbol_set:
            self._increment("skipped_symbol_count")
            return
        try:
            bar = _bar_from_message(message)
        except _MissingVWAP as exc:
            self._increment("skipped_no_vwap_count")
            logger.debug("skipped Alpaca bar without true VWAP: %s", exc)
            return
        except _MalformedMessage as exc:
            self._increment("malformed_count")
            logger.debug("ignored malformed Alpaca bar frame: %s", exc)
            return
        self._record_message("bar")
        if self._on_bar is None:
            return
        try:
            self._on_bar(bar)
        except Exception:  # noqa: BLE001 - consumer errors must not kill the stream
            self._increment("handler_errors")
            logger.debug("on_bar callback raised", exc_info=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _record_message(self, kind: str) -> None:
        with self._lock:
            self._last_message_utc = self._clock()
            if kind == "tick":
                self._tick_count += 1
            else:
                self._bar_count += 1

    def _increment(self, counter: str) -> None:
        with self._lock:
            setattr(self, f"_{counter}", getattr(self, f"_{counter}") + 1)

    def _record_error(self, error: Any) -> None:
        message = _describe_error(error, self._credentials)
        with self._lock:
            self._last_error = message

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
