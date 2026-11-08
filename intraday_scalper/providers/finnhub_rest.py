"""Finnhub REST client used *only* for startup quote validation.

The free Finnhub plan is heavily rate-limited and ``/quote`` returns the last
close outside the session, so this client is deliberately tiny:

* :meth:`FinnhubRestClient.quote` performs exactly one ``/quote`` request (with
  429 retries) and never loops.
* :meth:`FinnhubRestClient.startup_validation` is the only supported entry point
  for the live system: it is hard-capped at
  ``DataConfig.startup_quote_checks_max`` requests for the lifetime of the
  client. Any attempt to exceed the cap raises :class:`RuntimeError`.
* 401/403 (and persistent 429) raise :class:`DataUnavailableError`; the UI layer
  is then responsible for showing ``DATA_UNAVAILABLE`` and suppressing signals.

Nothing in the trading loop may poll this client. The websocket is the only
real-time source.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol, Sequence

from intraday_scalper.core import UTC, DataConfig, now_utc

__all__ = [
    "DataUnavailableError",
    "FinnhubRestClient",
    "HttpSession",
    "HttpResponse",
    "QuoteSnapshot",
]

_QUOTE_URL = "https://finnhub.io/api/v1/quote"


class DataUnavailableError(RuntimeError):
    """The configured data source cannot supply trustworthy data right now.

    Raised for authentication failures, exhausted 429 retries, network errors
    and unexpected HTTP status codes. Callers must surface
    ``DATA_UNAVAILABLE`` instead of fabricating prices.
    """


class HttpResponse(Protocol):
    """The minimal ``requests.Response`` surface used by this module."""

    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any:  # pragma: no cover - protocol only
        ...


class HttpSession(Protocol):
    """The minimal ``requests.Session`` surface used by this module."""

    def get(
        self, url: str, params: Mapping[str, Any] | None = None, timeout: float | None = None
    ) -> HttpResponse:  # pragma: no cover - protocol only
        ...


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """A single validated ``/quote`` response.

    ``quote_utc`` is the Finnhub ``t`` field (unix seconds, usually the last
    trade time) converted to an aware UTC datetime. This snapshot is *not* used
    for bar construction; it exists only to prove the symbol/token work at
    startup.
    """

    symbol: str
    current: float
    high: float
    low: float
    open: float
    prev_close: float
    quote_utc: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        for name in ("current", "high", "low", "open", "prev_close"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"QuoteSnapshot.{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.current <= 0:
            raise ValueError("QuoteSnapshot.current must be > 0")
        quote_utc = self.quote_utc
        if quote_utc.tzinfo is None or quote_utc.tzinfo.utcoffset(quote_utc) is None:
            raise ValueError("QuoteSnapshot.quote_utc must be timezone-aware")
        object.__setattr__(self, "quote_utc", quote_utc.astimezone(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current": self.current,
            "high": self.high,
            "low": self.low,
            "open": self.open,
            "prev_close": self.prev_close,
            "quote_utc": self.quote_utc.isoformat(),
        }


class FinnhubRestClient:
    """Minimal, rate-limit-aware Finnhub ``/quote`` client (startup only)."""

    _QUOTE_URL = _QUOTE_URL

    def __init__(
        self,
        token: str,
        data_config: DataConfig,
        session: HttpSession | None = None,
        clock: Callable[[], datetime] = now_utc,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Finnhub REST token must be a non-empty string")
        self._token = token.strip()
        self._data = data_config
        self._session = session if session is not None else self._default_session()
        self._clock = clock
        self._sleep = sleep_fn

        self._lock = threading.Lock()
        self._last_request_utc: datetime | None = None
        self._startup_calls_used = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def startup_calls_used(self) -> int:
        with self._lock:
            return self._startup_calls_used

    @property
    def min_interval_seconds(self) -> float:
        return float(self._data.rest_min_interval_seconds)

    def __repr__(self) -> str:  # never show the token
        return (
            f"FinnhubRestClient(startup_calls={self.startup_calls_used}, "
            f"max_calls={self._data.startup_quote_checks_max})"
        )

    def quote(self, symbol: str) -> QuoteSnapshot | None:
        """Fetch one ``/quote`` snapshot.

        Returns ``None`` when Finnhub answered but the payload has no usable
        price (for example ``c == 0``). Raises :class:`DataUnavailableError`
        when the data cannot be trusted. This method performs a single request
        (plus explicitly bounded 429 retries); callers must not poll it.
        """
        normalised = str(symbol).strip().upper()
        if not normalised:
            raise ValueError("symbol must be a non-empty string")
        payload = self._request_quote(normalised)
        return self._parse_quote(normalised, payload)

    def startup_validation(
        self, symbols: Sequence[str], max_calls: int | None = None
    ) -> dict[str, QuoteSnapshot | None]:
        """Validate a few symbols once at startup, respecting the hard cap.

        At most ``DataConfig.startup_quote_checks_max`` successful *or*
        attempted ``/quote`` calls may ever happen through this method. The cap
        is cumulative for the lifetime of the client; exceeding it raises
        :class:`RuntimeError` before any network call is made.
        """
        normalised: list[str] = []
        for symbol in symbols:
            name = str(symbol).strip().upper()
            if not name:
                raise ValueError("symbols must not contain empty strings")
            if name not in normalised:
                normalised.append(name)
        if not normalised:
            raise ValueError("symbols must not be empty")

        allowed = int(self._data.startup_quote_checks_max)
        if max_calls is not None:
            if not isinstance(max_calls, int) or isinstance(max_calls, bool):
                raise TypeError("max_calls must be an int or None")
            if max_calls < 0:
                raise ValueError("max_calls must be >= 0")

        with self._lock:
            remaining = allowed - self._startup_calls_used
        if remaining <= 0:
            raise RuntimeError(
                "startup_validation budget exhausted: "
                f"startup_quote_checks_max={allowed} (/quote polling is forbidden)"
            )

        planned = normalised[: max_calls] if max_calls is not None else normalised
        if not planned:
            raise ValueError("max_calls=0 leaves nothing to validate")
        if len(planned) > remaining:
            raise RuntimeError(
                f"startup_validation would issue {len(planned)} calls but only "
                f"{remaining} of startup_quote_checks_max={allowed} remain"
            )

        results: dict[str, QuoteSnapshot | None] = {}
        for symbol in planned:
            with self._lock:
                if self._startup_calls_used >= allowed:
                    raise RuntimeError(
                        "startup_validation budget exhausted: "
                        f"startup_quote_checks_max={allowed} (/quote polling is forbidden)"
                    )
                self._startup_calls_used += 1
            results[symbol] = self.quote(symbol)
        return results

    # ------------------------------------------------------------------ #
    # HTTP plumbing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_session() -> HttpSession:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError(
                "requests is required for Finnhub REST validation; "
                "install it with `pip install requests`"
            ) from exc
        session = requests.Session()
        session.headers.update({"User-Agent": "intraday-scalper/0.1 (startup-validation-only)"})
        return session

    def _request_quote(self, symbol: str) -> dict[str, Any]:
        self._respect_min_interval()
        retries_used = 0
        while True:
            try:
                response = self._session.get(
                    self._QUOTE_URL,
                    params={"symbol": symbol, "token": self._token},
                    timeout=float(self._data.rest_timeout_seconds),
                )
            except DataUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - requests raises many types
                self._mark_request()
                raise DataUnavailableError(
                    f"Finnhub /quote network error for {symbol}: {type(exc).__name__}"
                ) from exc

            self._mark_request()
            status = int(getattr(response, "status_code", 0))

            if status == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise DataUnavailableError(
                        f"Finnhub /quote returned invalid JSON for {symbol}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise DataUnavailableError(
                        f"Finnhub /quote returned a non-object payload for {symbol}"
                    )
                return payload

            if status in (401, 403):
                raise DataUnavailableError(
                    f"Finnhub /quote not permitted for {symbol} (HTTP {status}); "
                    "check the API token and account tier"
                )

            if status == 429:
                if retries_used >= int(self._data.rest_max_retries_429):
                    raise DataUnavailableError(
                        f"Finnhub /quote rate-limited for {symbol} (HTTP 429) after "
                        f"{retries_used} retries"
                    )
                retries_used += 1
                delay = self._retry_delay_seconds(retries_used, self._retry_after(response))
                self._sleep(delay)
                continue

            if status >= 500:
                raise DataUnavailableError(
                    f"Finnhub /quote server error for {symbol} (HTTP {status})"
                )

            raise DataUnavailableError(
                f"Finnhub /quote unexpected HTTP {status} for {symbol}"
            )

    def _respect_min_interval(self) -> None:
        with self._lock:
            last = self._last_request_utc
        if last is None:
            return
        elapsed = (self._clock() - last).total_seconds()
        remaining = float(self._data.rest_min_interval_seconds) - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def _mark_request(self) -> None:
        with self._lock:
            self._last_request_utc = self._clock()

    def _retry_delay_seconds(self, retries_used: int, retry_after: float | None) -> float:
        base = float(self._data.rest_min_interval_seconds) or 1.0
        factor = float(self._data.reconnect_backoff_factor)
        exponential = base * (factor ** max(retries_used - 1, 0))
        delay = max(exponential, retry_after or 0.0)
        return min(delay, float(self._data.reconnect_backoff_max_seconds))

    @staticmethod
    def _retry_after(response: HttpResponse) -> float | None:
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:  # pragma: no cover - defensive
            return None
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            return None
        return max(seconds, 0.0)

    @staticmethod
    def _parse_quote(symbol: str, payload: Mapping[str, Any]) -> QuoteSnapshot | None:
        def _number(key: str) -> float | None:
            raw = payload.get(key)
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        current = _number("c")
        timestamp = _number("t")
        if current is None or current <= 0:
            return None
        if timestamp is None or timestamp <= 0:
            return None
        try:
            quote_utc = datetime.fromtimestamp(timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

        high = _number("h") or current
        low = _number("l") or current
        open_ = _number("o") or current
        prev_close = _number("pc") or current
        if high < low:
            high, low = low, high
        return QuoteSnapshot(
            symbol=symbol,
            current=current,
            high=high,
            low=low,
            open=open_,
            prev_close=prev_close,
            quote_utc=quote_utc,
        )
