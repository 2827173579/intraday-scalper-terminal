"""Offline tests for the Finnhub websocket/REST providers.

Every test uses injected fake connectors/sessions: no network, no real sockets,
no real sleeps. Time is frozen at Friday 2026-11-06 14:30:05 UTC (09:30:05 ET).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intraday_scalper.core import ConnectionState, DataConfig, Tick, load_config
from intraday_scalper.providers.finnhub_rest import (
    DataUnavailableError,
    FinnhubRestClient,
    QuoteSnapshot,
)
from intraday_scalper.providers.finnhub_ws import (
    FinnhubWebSocketClient,
    WsConnectOptions,
    WsHandlers,
)

UTC = timezone.utc
FIXED_NOW = datetime(2026, 11, 6, 14, 30, 5, tzinfo=UTC)  # 09:30:05 ET
TICK_TS = datetime(2026, 11, 6, 14, 30, 1, tzinfo=UTC)
TICK_MS = int(TICK_TS.timestamp() * 1000)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def base_data_config(**overrides) -> DataConfig:
    return replace(load_config().data, **overrides)


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def trade_frame(*trades: tuple[str, float, float, datetime]) -> str:
    """Build a Finnhub trade websocket frame from (symbol, price, volume, ts)."""
    payload = [
        {"s": symbol, "p": price, "v": volume, "t": int(ts.timestamp() * 1000)}
        for symbol, price, volume, ts in trades
    ]
    return json.dumps({"type": "trade", "data": payload})


class FakeConnection:
    """In-process stand-in for ``websocket.WebSocketApp``."""

    def __init__(
        self,
        options: WsConnectOptions,
        handlers: WsHandlers,
        script: list[tuple[str, object]],
        drop_after_script: bool,
    ) -> None:
        self.options = options
        self.handlers = handlers
        self.script = script
        self.drop_after_script = drop_after_script
        self.sent: list[str] = []
        self.closed = threading.Event()
        self.run_calls = 0

    def send(self, payload: str) -> None:
        if self.closed.is_set():
            raise ConnectionError("fake websocket is closed")
        self.sent.append(payload)

    def close(self) -> None:
        self.closed.set()

    def run_forever(self) -> None:
        self.run_calls += 1
        self.handlers.on_open()
        for action, payload in self.script:
            if self.closed.is_set():
                return
            if action == "message":
                self.handlers.on_message(payload)
            elif action == "error":
                self.handlers.on_error(payload)
            else:  # pragma: no cover - defensive
                raise AssertionError(f"unknown fake action {action!r}")
        if self.drop_after_script:
            self.handlers.on_close()
            return
        self.closed.wait(timeout=10.0)
        self.handlers.on_close()


class FakeConnector:
    """Connection factory injected into the client; supports scripted behaviours."""

    def __init__(self, behaviours: list[dict] | None = None) -> None:
        self.behaviours = list(behaviours or [{}])
        self.connections: list[FakeConnection] = []
        self.options: list[WsConnectOptions] = []

    def __call__(self, options: WsConnectOptions, handlers: WsHandlers) -> FakeConnection:
        index = len(self.connections)
        behaviour = (
            self.behaviours[index] if index < len(self.behaviours) else self.behaviours[-1]
        )
        connection = FakeConnection(
            options,
            handlers,
            script=list(behaviour.get("script", [])),
            drop_after_script=bool(behaviour.get("drop_after_script", False)),
        )
        self.connections.append(connection)
        self.options.append(options)
        return connection


def subscribed_symbols(connection: FakeConnection) -> list[str]:
    return [json.loads(payload)["symbol"] for payload in connection.sent]


def make_ws_client(
    connector: FakeConnector,
    *,
    ticks: list[Tick] | None = None,
    states: list[tuple[ConnectionState, str]] | None = None,
    on_tick=None,
    data_config: DataConfig | None = None,
    sleep_fn=None,
) -> FinnhubWebSocketClient:
    sink = ticks if ticks is not None else []
    state_log = states if states is not None else []
    return FinnhubWebSocketClient(
        "unit-test-token",
        ["SOXL", "SOXS"],
        data_config or base_data_config(),
        on_tick=on_tick if on_tick is not None else sink.append,
        on_state=lambda state, reason: state_log.append((state, reason)),
        clock=lambda: FIXED_NOW,
        connector=connector,
        sleep_fn=sleep_fn if sleep_fn is not None else (lambda seconds: None),
    )


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload=None,
        headers: dict | None = None,
        invalid_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("response is not JSON")
        return self._payload


class FakeSession:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self._responses:
            raise AssertionError("unexpected extra /quote request in test")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


QUOTE_TS = datetime(2026, 11, 6, 14, 29, 59, tzinfo=UTC)


def quote_payload(
    c: float = 40.5,
    h: float = 40.9,
    low: float = 40.0,
    o: float = 40.1,
    pc: float = 39.9,
    t: float | None = None,
) -> dict:
    return {
        "c": c,
        "h": h,
        "l": low,
        "o": o,
        "pc": pc,
        "t": QUOTE_TS.timestamp() if t is None else t,
    }


def make_rest_client(
    session: FakeSession,
    *,
    data_config: DataConfig | None = None,
    sleeps: list[float] | None = None,
) -> FinnhubRestClient:
    sleep_log = sleeps if sleeps is not None else []
    return FinnhubRestClient(
        "unit-test-token",
        data_config or base_data_config(),
        session=session,
        clock=lambda: FIXED_NOW,
        sleep_fn=sleep_log.append,
    )


# --------------------------------------------------------------------------- #
# Websocket client
# --------------------------------------------------------------------------- #
def test_ws_open_subscribes_and_dispatches_tick():
    connector = FakeConnector(
        [{"script": [("message", trade_frame(("SOXL", 40.25, 150.0, TICK_TS)))]}]
    )
    ticks: list[Tick] = []
    states: list[tuple[ConnectionState, str]] = []
    client = make_ws_client(connector, ticks=ticks, states=states)

    client.start()
    try:
        assert client.wait_until_connected(5.0) is True
        assert wait_until(lambda: client.tick_count == 1)

        assert client.state is ConnectionState.CONNECTED
        assert subscribed_symbols(connector.connections[0]) == ["SOXL", "SOXS"]
        assert [json.loads(payload)["type"] for payload in connector.connections[0].sent] == [
            "subscribe",
            "subscribe",
        ]

        assert len(ticks) == 1
        tick = ticks[0]
        assert tick.symbol == "SOXL"
        assert tick.price == pytest.approx(40.25)
        assert tick.volume == pytest.approx(150.0)
        assert tick.ts_utc == TICK_TS
        assert tick.trade_id is None
        assert client.last_message_utc == FIXED_NOW
        assert client.message_count == 1

        options = connector.options[0]
        assert options.url.startswith("wss://ws.finnhub.io")
        assert "token=unit-test-token" in options.url
        assert options.ping_interval_seconds > 0

        assert any(state is ConnectionState.CONNECTED for state, _ in states)
    finally:
        client.stop()
    assert client.join(5.0) is True
    assert client.state is ConnectionState.DISCONNECTED


def test_ws_dispatches_every_trade_in_a_frame():
    frame = trade_frame(
        ("SOXL", 40.10, 100.0, TICK_TS),
        ("SOXS", 6.25, 500.0, TICK_TS),
    )
    connector = FakeConnector([{"script": [("message", frame)]}])
    ticks: list[Tick] = []
    client = make_ws_client(connector, ticks=ticks)
    client.start()
    try:
        assert wait_until(lambda: client.tick_count == 2)
        assert [tick.symbol for tick in ticks] == ["SOXL", "SOXS"]
        assert ticks[1].ts_utc == TICK_TS
    finally:
        client.stop()
        client.join(5.0)


def test_ws_filters_unknown_symbols_and_zero_volume():
    frame = trade_frame(
        ("SOXL", 40.0, 100.0, TICK_TS),
        ("SOXS", 6.0, 0.0, TICK_TS),
        ("AAPL", 200.0, 10.0, TICK_TS),
        ("SOXS", 6.1, 20.0, TICK_TS),
    )
    connector = FakeConnector([{"script": [("message", frame)]}])
    ticks: list[Tick] = []
    client = make_ws_client(connector, ticks=ticks)
    client.start()
    try:
        assert wait_until(lambda: client.tick_count == 2)
        assert [tick.symbol for tick in ticks] == ["SOXL", "SOXS"]
        assert client.skipped_volume_count == 1
        assert client.skipped_symbol_count == 1
        assert client.malformed_count == 0
    finally:
        client.stop()
        client.join(5.0)


def test_ws_malformed_messages_are_tolerated():
    bad_messages = [
        "not-json",
        json.dumps([1, 2, 3]),
        json.dumps({"type": "trade", "data": "oops"}),
        json.dumps({"type": "trade", "data": [{"s": "SOXL", "p": 40.0, "v": 100.0}]}),
        json.dumps({"type": "trade", "data": [{"s": 123, "p": 40.0, "v": 100.0, "t": TICK_MS}]}),
        json.dumps({"type": "trade", "data": [{"s": "SOXL", "p": 0.0, "v": 100.0, "t": TICK_MS}]}),
        json.dumps({"type": "ping"}),
    ]
    good = trade_frame(("SOXL", 40.0, 100.0, TICK_TS))
    connector = FakeConnector([{"script": [("message", raw) for raw in bad_messages + [good]]}])
    ticks: list[Tick] = []
    client = make_ws_client(connector, ticks=ticks)
    client.start()
    try:
        assert wait_until(lambda: client.tick_count == 1)
        assert client.malformed_count == 6
        assert len(ticks) == 1
        # every raw frame is counted, including ignored/unknown types
        assert client.message_count == len(bad_messages) + 1
        assert client.state is ConnectionState.CONNECTED
    finally:
        client.stop()
        client.join(5.0)


def test_ws_on_tick_exception_does_not_kill_stream():
    frame = trade_frame(("SOXL", 40.0, 100.0, TICK_TS))
    connector = FakeConnector([{"script": [("message", frame)] * 3}])
    received: list[Tick] = []

    def handler(tick: Tick) -> None:
        received.append(tick)
        if len(received) == 1:
            raise RuntimeError("consumer bug")

    client = make_ws_client(connector, on_tick=handler)
    client.start()
    try:
        assert wait_until(lambda: client.tick_count == 3)
        assert len(received) == 3
        assert client.handler_errors == 1
        assert client.state is ConnectionState.CONNECTED
    finally:
        client.stop()
        client.join(5.0)


def test_ws_reconnects_with_backoff_and_resubscribes():
    connector = FakeConnector(
        [
            {"script": [], "drop_after_script": True},
            {"script": [("message", trade_frame(("SOXL", 40.0, 100.0, TICK_TS)))]},
        ]
    )
    sleeps: list[float] = []
    ticks: list[Tick] = []
    data_config = base_data_config(
        max_reconnect_attempts=5,
        reconnect_backoff_initial_seconds=1.0,
        reconnect_backoff_factor=2.0,
        reconnect_backoff_max_seconds=60.0,
    )
    client = make_ws_client(
        connector, ticks=ticks, data_config=data_config, sleep_fn=sleeps.append
    )
    client.start()
    try:
        assert wait_until(
            lambda: len(connector.connections) >= 2
            and client.tick_count == 1
            and client.state is ConnectionState.CONNECTED
        )
        assert client.reconnect_count == 1
        assert sleeps == [1.0]
        assert subscribed_symbols(connector.connections[1]) == ["SOXL", "SOXS"]
    finally:
        client.stop()
        client.join(5.0)


def test_ws_gives_up_after_max_reconnect_attempts():
    connector = FakeConnector([{"script": [], "drop_after_script": True}])
    sleeps: list[float] = []
    data_config = base_data_config(
        max_reconnect_attempts=2,
        reconnect_backoff_initial_seconds=1.0,
        reconnect_backoff_factor=2.0,
        reconnect_backoff_max_seconds=60.0,
    )
    client = make_ws_client(connector, data_config=data_config, sleep_fn=sleeps.append)
    client.start()
    assert client.join(5.0) is True
    assert client.state is ConnectionState.UNAVAILABLE
    assert len(connector.connections) == 3
    assert sleeps == [1.0, 2.0]
    assert client.reconnect_count == 2


def test_ws_auth_failure_sets_unauthenticated_without_retry():
    connector = FakeConnector(
        [{"script": [("error", "401 Unauthorized: invalid API key")], "drop_after_script": True}]
    )
    data_config = base_data_config(max_reconnect_attempts=5)
    client = make_ws_client(connector, data_config=data_config)
    client.start()
    assert client.join(5.0) is True
    assert client.state is ConnectionState.UNAUTHENTICATED
    assert client.reconnect_count == 0
    assert len(connector.connections) == 1


def test_ws_stop_before_start_is_harmless():
    connector = FakeConnector()
    client = make_ws_client(connector)
    client.stop()
    assert client.join(0.5) is True
    assert client.state is ConnectionState.DISCONNECTED
    assert connector.connections == []


def test_ws_constructor_validation():
    data_config = base_data_config()
    with pytest.raises(ValueError):
        FinnhubWebSocketClient("", ["SOXL"], data_config, on_tick=lambda tick: None)
    with pytest.raises(ValueError):
        FinnhubWebSocketClient("   ", ["SOXL"], data_config, on_tick=lambda tick: None)
    with pytest.raises(ValueError):
        FinnhubWebSocketClient("token", [], data_config, on_tick=lambda tick: None)


def test_ws_repr_hides_token():
    connector = FakeConnector()
    client = make_ws_client(connector)
    text = repr(client)
    assert "unit-test-token" not in text
    assert "SOXL" in text
    assert "DISCONNECTED" in text


# --------------------------------------------------------------------------- #
# REST client
# --------------------------------------------------------------------------- #
def test_rest_quote_success():
    session = FakeSession([FakeResponse(200, quote_payload())])
    sleeps: list[float] = []
    client = make_rest_client(session, sleeps=sleeps)

    snapshot = client.quote("soxl")

    assert isinstance(snapshot, QuoteSnapshot)
    assert snapshot.symbol == "SOXL"
    assert snapshot.current == pytest.approx(40.5)
    assert snapshot.high == pytest.approx(40.9)
    assert snapshot.low == pytest.approx(40.0)
    assert snapshot.open == pytest.approx(40.1)
    assert snapshot.prev_close == pytest.approx(39.9)
    assert snapshot.quote_utc == QUOTE_TS
    assert snapshot.to_dict()["quote_utc"] == QUOTE_TS.isoformat()

    assert sleeps == []
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://finnhub.io/api/v1/quote"
    assert call["params"]["symbol"] == "SOXL"
    assert call["params"]["token"] == "unit-test-token"
    assert call["timeout"] == pytest.approx(load_config().data.rest_timeout_seconds)
    assert "unit-test-token" not in repr(client)


def test_rest_quote_returns_none_when_no_usable_price():
    session = FakeSession([FakeResponse(200, quote_payload(c=0.0)), FakeResponse(200, {"d": 1.0})])
    client = make_rest_client(session)
    assert client.quote("SOXL") is None
    assert client.quote("SOXL") is None


@pytest.mark.parametrize("status", [401, 403])
def test_rest_quote_auth_errors_raise_data_unavailable(status):
    session = FakeSession([FakeResponse(status, {"error": "no permission"})])
    client = make_rest_client(session)
    with pytest.raises(DataUnavailableError, match="not permitted"):
        client.quote("SOXL")
    assert len(session.calls) == 1


def test_rest_429_retry_honours_retry_after():
    session = FakeSession(
        [
            FakeResponse(429, {"error": "rate limited"}, headers={"Retry-After": "2"}),
            FakeResponse(200, quote_payload()),
        ]
    )
    sleeps: list[float] = []
    data_config = base_data_config(
        rest_max_retries_429=2,
        rest_min_interval_seconds=1.0,
        reconnect_backoff_factor=2.0,
        reconnect_backoff_max_seconds=60.0,
    )
    client = make_rest_client(session, data_config=data_config, sleeps=sleeps)
    snapshot = client.quote("SOXS")
    assert snapshot is not None
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_rest_429_exhausted_raises():
    session = FakeSession([FakeResponse(429), FakeResponse(429), FakeResponse(429)])
    sleeps: list[float] = []
    data_config = base_data_config(
        rest_max_retries_429=2,
        rest_min_interval_seconds=1.0,
        reconnect_backoff_factor=2.0,
        reconnect_backoff_max_seconds=60.0,
    )
    client = make_rest_client(session, data_config=data_config, sleeps=sleeps)
    with pytest.raises(DataUnavailableError, match="rate-limited"):
        client.quote("SOXL")
    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_rest_min_interval_between_requests():
    session = FakeSession([FakeResponse(200, quote_payload()), FakeResponse(200, quote_payload())])
    sleeps: list[float] = []
    data_config = base_data_config(rest_min_interval_seconds=1.1)
    client = make_rest_client(session, data_config=data_config, sleeps=sleeps)

    client.quote("SOXL")
    assert sleeps == []
    client.quote("SOXS")
    assert sleeps == [pytest.approx(1.1)]
    assert len(session.calls) == 2


def test_rest_server_error_raises():
    session = FakeSession([FakeResponse(503, {"error": "down"})])
    client = make_rest_client(session)
    with pytest.raises(DataUnavailableError, match="server error"):
        client.quote("SOXL")


def test_rest_invalid_json_raises():
    session = FakeSession([FakeResponse(200, invalid_json=True)])
    client = make_rest_client(session)
    with pytest.raises(DataUnavailableError, match="invalid JSON"):
        client.quote("SOXL")


def test_rest_network_error_raises():
    session = FakeSession([ConnectionError("socket exploded")])
    client = make_rest_client(session)
    with pytest.raises(DataUnavailableError, match="network error"):
        client.quote("SOXL")


def test_rest_quote_rejects_empty_symbol():
    session = FakeSession([])
    client = make_rest_client(session)
    with pytest.raises(ValueError):
        client.quote("   ")


def test_rest_startup_validation_respects_hard_cap():
    session = FakeSession(
        [FakeResponse(200, quote_payload()), FakeResponse(200, quote_payload(c=6.1))]
    )
    data_config = base_data_config(startup_quote_checks_max=2)
    client = make_rest_client(session, data_config=data_config)

    result = client.startup_validation(["SOXL", "SOXS"])
    assert list(result) == ["SOXL", "SOXS"]
    assert all(isinstance(value, QuoteSnapshot) for value in result.values())
    assert len(session.calls) == 2
    assert client.startup_calls_used == 2

    with pytest.raises(RuntimeError, match="budget"):
        client.startup_validation(["SOXL"])
    assert len(session.calls) == 2  # no extra HTTP request happened


def test_rest_startup_validation_max_calls_subset():
    session = FakeSession([FakeResponse(200, quote_payload())])
    data_config = base_data_config(startup_quote_checks_max=2)
    client = make_rest_client(session, data_config=data_config)

    result = client.startup_validation(["SOXL", "SOXS"], max_calls=1)
    assert list(result) == ["SOXL"]
    assert client.startup_calls_used == 1

    session._responses.append(FakeResponse(200, quote_payload(c=6.2)))
    result = client.startup_validation(["SOXS"])
    assert list(result) == ["SOXS"]
    assert client.startup_calls_used == 2


def test_rest_startup_validation_rejects_oversized_request():
    session = FakeSession([])
    data_config = base_data_config(startup_quote_checks_max=2)
    client = make_rest_client(session, data_config=data_config)
    with pytest.raises(RuntimeError, match="would issue 3 calls"):
        client.startup_validation(["SOXL", "SOXS", "SPY"])
    assert session.calls == []
    assert client.startup_calls_used == 0


def test_rest_startup_validation_counts_failed_attempts():
    session = FakeSession([FakeResponse(403)])
    data_config = base_data_config(startup_quote_checks_max=1)
    client = make_rest_client(session, data_config=data_config)
    with pytest.raises(DataUnavailableError):
        client.startup_validation(["SOXL"])
    assert client.startup_calls_used == 1
    with pytest.raises(RuntimeError, match="budget"):
        client.startup_validation(["SOXS"])
    assert len(session.calls) == 1


def test_rest_startup_validation_with_zero_budget():
    session = FakeSession([])
    data_config = base_data_config(startup_quote_checks_max=0)
    client = make_rest_client(session, data_config=data_config)
    with pytest.raises(RuntimeError, match="budget"):
        client.startup_validation(["SOXL"])
    assert session.calls == []


def test_rest_constructor_validation():
    data_config = base_data_config()
    with pytest.raises(ValueError):
        FinnhubRestClient("", data_config, session=FakeSession([]))
    with pytest.raises(ValueError):
        FinnhubRestClient("   ", data_config, session=FakeSession([]))
