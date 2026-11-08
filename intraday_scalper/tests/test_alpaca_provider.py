"""Offline tests for the Alpaca IEX provider.

Everything here is offline: fake streams/clients are injected, no socket is ever
opened, no real credentials are read (the autouse fixture clears Alpaca env
variables; dotenv tests use tmp files with fake values only).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intraday_scalper.core import Bar, ConnectionState, DataConfig, Tick, load_config
from intraday_scalper.providers import alpaca_provider as alp
from intraday_scalper.providers.alpaca_provider import (
    AlpacaCredentials,
    AlpacaCredentialsError,
    AlpacaIEXProvider,
    AlpacaStreamHooks,
    ProbeResult,
    load_alpaca_credentials,
    probe_alpaca,
)

UTC = timezone.utc
SESSION_START = datetime(2026, 11, 6, 14, 30, tzinfo=UTC)  # 09:30 ET (EST)
FIXED_NOW = SESSION_START + timedelta(seconds=5)
FAKE_API_KEY = "AKFAKEKEYFORTESTS"
FAKE_SECRET = "fake-secret-value-for-tests"

ALPACA_ENV_NAMES = (
    "ALPACA_API_KEY",
    "APCA_API_KEY_ID",
    "ALPACA_SECRET_KEY",
    "APCA_API_SECRET_KEY",
    "ALPACA_DATA_FEED",
    "APCA_DATA_FEED",
)


@pytest.fixture(autouse=True)
def _isolate_alpaca_env(monkeypatch):
    for name in ALPACA_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield


def make_credentials(feed: str = "iex") -> AlpacaCredentials:
    return AlpacaCredentials(api_key=FAKE_API_KEY, secret_key=FAKE_SECRET, feed=feed)


def fast_config(**overrides) -> DataConfig:
    base = replace(
        load_config().data,
        reconnect_backoff_initial_seconds=0.01,
        reconnect_backoff_max_seconds=0.05,
        reconnect_backoff_factor=2.0,
        max_reconnect_attempts=5,
    )
    return replace(base, **overrides) if overrides else base


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def stop_and_join(provider: AlpacaIEXProvider, timeout: float = 5.0) -> None:
    provider.stop()
    assert provider.join(timeout) is True, "provider thread did not stop"


def trade_message(
    symbol: str = "SOXL",
    price: float = 40.0,
    volume: float = 100.0,
    ts: object = "2026-11-06T14:30:00Z",
) -> dict:
    return {"T": "t", "S": symbol, "p": price, "s": volume, "t": ts}


def bar_message(
    symbol: str = "SOXL",
    open_: float = 40.0,
    high: float = 40.3,
    low: float = 39.8,
    close: float = 40.1,
    volume: float = 1_000.0,
    vwap: object = 40.05,
    ts: object = "2026-11-06T14:30:00Z",
) -> dict:
    message = {
        "T": "b",
        "S": symbol,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
        "t": ts,
    }
    if vwap is not None:
        message["vw"] = vwap
    return message


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeStream:
    """Scripted stand-in for alpaca-py ``StockDataStream`` (no network)."""

    def __init__(self, behaviour: dict, credentials, hooks: AlpacaStreamHooks) -> None:
        self.behaviour = dict(behaviour)
        self.credentials = credentials
        self.hooks = hooks
        self.stop_event = threading.Event()
        self.started = threading.Event()
        self.subscriptions: list[tuple[str, tuple[str, ...]]] = []
        self.handlers: dict[str, object] = {}
        self.stop_calls = 0
        self.run_calls = 0

    def subscribe_trades(self, handler, *symbols) -> None:
        assert inspect.iscoroutinefunction(handler), "alpaca-py requires coroutine handlers"
        self.subscriptions.append(("trades", tuple(symbols)))
        self.handlers["trade"] = handler

    def subscribe_bars(self, handler, *symbols) -> None:
        assert inspect.iscoroutinefunction(handler), "alpaca-py requires coroutine handlers"
        self.subscriptions.append(("bars", tuple(symbols)))
        self.handlers["bar"] = handler

    def run(self) -> None:
        self.run_calls += 1
        self.started.set()
        if self.behaviour.get("connected", True):
            self.hooks.on_connected()
        for channel, payload in self.behaviour.get("script", ()):
            asyncio.run(self.handlers[channel](payload))
        outcome = self.behaviour.get("outcome", "block")
        if outcome == "auth_failed":
            self.hooks.on_auth_failed(self.behaviour.get("auth_error", "auth failed (401)"))
            return
        if outcome == "disconnect":
            return
        if outcome == "raise":
            raise self.behaviour.get("error", RuntimeError("stream exploded"))
        self.stop_event.wait(15.0)

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_event.set()


class FakeStreamFactory:
    """Returns one :class:`FakeStream` per connection attempt."""

    def __init__(self, behaviours: list[dict] | None = None) -> None:
        self.behaviours = list(behaviours) if behaviours else [{"outcome": "block"}]
        self.streams: list[FakeStream] = []

    def __call__(self, credentials, hooks: AlpacaStreamHooks) -> FakeStream:
        index = len(self.streams)
        behaviour = (
            self.behaviours[index] if index < len(self.behaviours) else self.behaviours[-1]
        )
        stream = FakeStream(behaviour, credentials, hooks)
        self.streams.append(stream)
        return stream


class FlakyFactory:
    """First factory call raises, then behaves like a blocking stream."""

    def __init__(self) -> None:
        self.calls = 0
        self.streams: list[FakeStream] = []

    def __call__(self, credentials, hooks: AlpacaStreamHooks) -> FakeStream:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary factory failure")
        stream = FakeStream({"outcome": "block"}, credentials, hooks)
        self.streams.append(stream)
        return stream


def make_provider(
    factory,
    *,
    data_config: DataConfig | None = None,
    on_tick=None,
    on_bar=None,
    on_state=None,
    clock=lambda: FIXED_NOW,
):
    ticks: list[Tick] = []
    bars: list[Bar] = []
    states: list[tuple[ConnectionState, str]] = []
    provider = AlpacaIEXProvider(
        make_credentials(),
        data_config if data_config is not None else fast_config(),
        on_tick=on_tick if on_tick is not None else ticks.append,
        on_bar=on_bar if on_bar is not None else bars.append,
        on_state=on_state if on_state is not None else (lambda s, r: states.append((s, r))),
        clock=clock,
        stream_factory=factory,
    )
    return provider, ticks, bars, states


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def write_env(path: Path, text: str, *, bom: bool = False) -> Path:
    data = text.encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + data)
    return path


def test_load_credentials_prefers_environment(tmp_path, monkeypatch):
    env_file = write_env(
        tmp_path / ".env",
        "ALPACA_API_KEY=file-key\nALPACA_SECRET_KEY=file-secret\nALPACA_DATA_FEED=sip\n",
    )
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")

    credentials = load_alpaca_credentials(env_file=env_file)
    assert credentials.api_key == "env-key"
    assert credentials.secret_key == "env-secret"
    assert credentials.feed == "iex"


def test_load_credentials_falls_back_to_dotenv(tmp_path):
    env_file = write_env(
        tmp_path / ".env",
        "# comment line\n"
        "ALPACA_API_KEY='file-key'\n"
        'ALPACA_SECRET_KEY="file-secret"\n'
        "export ALPACA_DATA_FEED=sip\n"
        "\n"
        "# FINNHUB_API_KEY=ignored\n",
        bom=True,
    )
    credentials = load_alpaca_credentials(env_file=env_file)
    assert credentials.api_key == "file-key"
    assert credentials.secret_key == "file-secret"
    assert credentials.feed == "sip"


def test_load_credentials_accepts_apca_aliases(tmp_path, monkeypatch):
    env_file = write_env(tmp_path / ".env", "ALPACA_API_KEY=file-key\nALPACA_SECRET_KEY=file-secret\n")
    monkeypatch.setenv("APCA_API_KEY_ID", "apca-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "apca-secret")
    credentials = load_alpaca_credentials(env_file=env_file)
    assert credentials.api_key == "apca-key"
    assert credentials.secret_key == "apca-secret"


def test_load_credentials_missing_raises_without_leaking_secret(tmp_path):
    env_file = tmp_path / "missing.env"
    with pytest.raises(AlpacaCredentialsError) as excinfo:
        load_alpaca_credentials(env_file=env_file)
    message = str(excinfo.value)
    assert "ALPACA_API_KEY" in message
    assert "ALPACA_SECRET_KEY" in message
    assert str(env_file) in message
    assert FAKE_SECRET not in message


def test_load_credentials_empty_values_are_missing(tmp_path, monkeypatch):
    env_file = write_env(tmp_path / ".env", "ALPACA_SECRET_KEY=file-secret\n")
    monkeypatch.setenv("ALPACA_API_KEY", "   ")
    with pytest.raises(AlpacaCredentialsError) as excinfo:
        load_alpaca_credentials(env_file=env_file)
    assert "ALPACA_API_KEY" in str(excinfo.value)
    assert "file-secret" not in str(excinfo.value)


def test_load_credentials_tolerates_missing_env_file_when_environment_is_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
    credentials = load_alpaca_credentials(env_file=tmp_path / "nope.env")
    assert credentials.api_key == "env-key"
    assert credentials.secret_key == "env-secret"


def test_load_credentials_default_feed_is_iex(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
    credentials = load_alpaca_credentials(env_file=tmp_path / "nope.env")
    assert credentials.feed == "iex"
    assert credentials.feed_enum == "iex"


def test_credentials_repr_hides_secret_and_is_frozen():
    import dataclasses

    credentials = make_credentials()
    text = repr(credentials)
    assert FAKE_SECRET not in text
    assert FAKE_API_KEY in text
    assert "iex" in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        credentials.secret_key = "other"  # type: ignore[misc]


def test_credentials_validation_and_normalisation():
    with pytest.raises(AlpacaCredentialsError):
        AlpacaCredentials(api_key="", secret_key="secret")
    with pytest.raises(AlpacaCredentialsError):
        AlpacaCredentials(api_key="key", secret_key="   ")
    credentials = AlpacaCredentials(api_key=" key ", secret_key=" secret ", feed="IEX")
    assert credentials.api_key == "key"
    assert credentials.secret_key == "secret"
    assert credentials.feed == "iex"
    assert AlpacaCredentials("key", "secret", "bogus-feed").feed_enum == "iex"


def test_default_env_file_location():
    assert alp.DEFAULT_ENV_FILE == alp.PACKAGE_ROOT / ".env"
    assert alp.DEFAULT_ENV_FILE.name == ".env"
    assert alp.PACKAGE_ROOT.name == "intraday_scalper"


# --------------------------------------------------------------------------- #
# Frame parsing
# --------------------------------------------------------------------------- #
def test_trade_frame_to_tick():
    tick = alp._trade_from_message(trade_message(symbol="soxl", price=40.5, volume=150.0))
    assert isinstance(tick, Tick)
    assert tick.symbol == "SOXL"
    assert tick.price == pytest.approx(40.5)
    assert tick.volume == pytest.approx(150.0)
    assert tick.ts_utc == SESSION_START
    assert tick.ts_utc.tzinfo is not None
    assert tick.trade_id is None


def test_trade_timestamp_iso_offset_and_fractional_seconds():
    message = trade_message(ts="2026-11-06T16:30:00.123456789+02:00")
    tick = alp._trade_from_message(message)
    assert tick.ts_utc == SESSION_START + timedelta(microseconds=123456)
    assert tick.ts_utc.tzinfo == UTC


def test_trade_timestamp_msgpack_and_epoch_forms():
    import msgpack

    epoch_seconds = int(SESSION_START.timestamp())
    packed = msgpack.Timestamp.from_unix(epoch_seconds)
    assert alp._trade_from_message(trade_message(ts=packed)).ts_utc == SESSION_START
    assert alp._trade_from_message(trade_message(ts=epoch_seconds)).ts_utc == SESSION_START
    # millisecond-epoch form
    assert alp._trade_from_message(trade_message(ts=epoch_seconds * 1000)).ts_utc == SESSION_START


def test_trade_frame_rejects_malformed_payloads():
    bad_messages = [
        "not-a-mapping",
        {},
        {"T": "q", "S": "SOXL", "p": 40.0, "s": 1.0, "t": "2026-11-06T14:30:00Z"},
        {"T": "t", "S": "SOXL", "p": -1.0, "s": 1.0, "t": "2026-11-06T14:30:00Z"},
        {"T": "t", "S": "SOXL", "p": 40.0, "s": -1.0, "t": "2026-11-06T14:30:00Z"},
        {"T": "t", "S": "", "p": 40.0, "s": 1.0, "t": "2026-11-06T14:30:00Z"},
        {"T": "t", "S": "SOXL", "p": 40.0, "s": 1.0},
        {"T": "t", "S": "SOXL", "p": 40.0, "s": 1.0, "t": "not-a-timestamp"},
        {"T": "t", "S": "SOXL", "p": 40.0, "s": 1.0, "t": True},
    ]
    for message in bad_messages:
        with pytest.raises(alp._MalformedMessage):
            alp._trade_from_message(message)


def test_bar_frame_to_bar():
    bar = alp._bar_from_message(bar_message())
    assert isinstance(bar, Bar)
    assert bar.symbol == "SOXL"
    assert bar.timeframe_minutes == 1
    assert bar.start_utc == SESSION_START
    assert bar.end_utc == SESSION_START + timedelta(minutes=1)
    assert bar.open == pytest.approx(40.0)
    assert bar.high == pytest.approx(40.3)
    assert bar.low == pytest.approx(39.8)
    assert bar.close == pytest.approx(40.1)
    assert bar.volume == pytest.approx(1_000.0)
    assert bar.vwap == pytest.approx(40.05)
    assert bar.is_complete is True


def test_bar_frame_without_vwap_is_skipped_not_synthesised():
    with pytest.raises(alp._MissingVWAP) as excinfo:
        alp._bar_from_message(bar_message(vwap=None))
    assert isinstance(excinfo.value, alp._MalformedMessage)


def test_bar_frame_rejects_invalid_ohlcv():
    with pytest.raises(alp._MalformedMessage):
        alp._bar_from_message(bar_message(vwap=50.0))  # vwap outside [low, high]
    with pytest.raises(alp._MalformedMessage):
        alp._bar_from_message(bar_message(open_=-1.0))
    with pytest.raises(alp._MalformedMessage):
        alp._bar_from_message({"T": "b", "S": "SOXL", "o": 40.0})  # missing fields


def test_auth_error_classification():
    assert alp._looks_like_auth_error("401 Unauthorized")
    assert alp._looks_like_auth_error("auth failed")
    assert alp._looks_like_auth_error("invalid credentials")
    assert alp._looks_like_auth_error("Forbidden")
    assert not alp._looks_like_auth_error("connection reset by peer")
    assert not alp._looks_like_auth_error("timed out")

    class StatusError(Exception):
        status_code = 403

    assert alp._looks_like_auth_error(StatusError("denied"))
    assert alp._normalise_error(StatusError("denied")) == "auth"
    assert alp._normalise_error("connection reset") == "error"


# --------------------------------------------------------------------------- #
# Provider lifecycle + frame dispatch (fake stream)
# --------------------------------------------------------------------------- #
def test_start_is_non_blocking_and_connects():
    factory = FakeStreamFactory([{"outcome": "block", "script": [("trade", trade_message())]}])
    provider, ticks, bars, states = make_provider(factory)

    started = time.monotonic()
    provider.start()
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.5, "start() must not block on the stream"
        assert provider.wait_until_connected(5.0) is True
        assert provider.state is ConnectionState.CONNECTED
        assert wait_until(lambda: provider.tick_count == 1)
        assert wait_until(lambda: len(ticks) == 1)
        assert provider.is_running is True
        assert provider.last_message_utc == FIXED_NOW
        assert [state for state, _ in states][:2] == [
            ConnectionState.CONNECTING,
            ConnectionState.CONNECTED,
        ]
    finally:
        stop_and_join(provider)
    assert provider.state is ConnectionState.DISCONNECTED
    assert bars == []


def test_stream_is_subscribed_to_both_channels_before_run():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.start()
    try:
        assert provider.wait_until_connected(5.0) is True
        assert wait_until(lambda: bool(factory.streams))
        stream = factory.streams[0]
        assert stream.subscriptions == [
            ("trades", ("SOXL", "SOXS")),
            ("bars", ("SOXL", "SOXS")),
        ]
        assert stream.handlers["trade"].__name__ == "_async_trade_handler"
        assert stream.handlers["bar"].__name__ == "_async_bar_handler"
    finally:
        stop_and_join(provider)


def test_trade_messages_become_ticks():
    factory = FakeStreamFactory(
        [
            {
                "outcome": "block",
                "script": [
                    ("trade", trade_message("SOXL", 40.0, 100.0)),
                    ("trade", trade_message("SOXS", 6.1, 200.0)),
                    ("trade", trade_message("SOXL", 40.1, 50.0, ts="2026-11-06T14:30:01.5Z")),
                ],
            }
        ]
    )
    provider, ticks, bars, _ = make_provider(factory)
    provider.start()
    try:
        assert wait_until(lambda: len(ticks) == 3)
        assert [tick.symbol for tick in ticks] == ["SOXL", "SOXS", "SOXL"]
        assert ticks[0].price == pytest.approx(40.0)
        assert ticks[2].ts_utc == SESSION_START + timedelta(seconds=1, milliseconds=500)
        assert provider.tick_count == 3
        assert provider.bar_count == 0
        assert bars == []
    finally:
        stop_and_join(provider)


def test_foreign_symbol_trades_are_ignored():
    factory = FakeStreamFactory([{"outcome": "block", "script": [("trade", trade_message("AAPL"))]}])
    provider, ticks, _, _ = make_provider(factory)
    provider.start()
    try:
        assert wait_until(lambda: provider.skipped_symbol_count == 1)
        assert provider.tick_count == 0
        assert ticks == []
    finally:
        stop_and_join(provider)


def test_bar_messages_become_one_minute_bars():
    factory = FakeStreamFactory([{"outcome": "block", "script": [("bar", bar_message())]}])
    provider, ticks, bars, _ = make_provider(factory)
    provider.start()
    try:
        assert wait_until(lambda: len(bars) == 1)
        bar = bars[0]
        assert bar.symbol == "SOXL"
        assert bar.timeframe_minutes == 1
        assert bar.start_utc == SESSION_START
        assert bar.end_utc == SESSION_START + timedelta(minutes=1)
        assert bar.vwap == pytest.approx(40.05)
        assert provider.bar_count == 1
        assert ticks == []
    finally:
        stop_and_join(provider)


def test_bar_without_vwap_is_skipped_and_counted():
    factory = FakeStreamFactory(
        [{"outcome": "block", "script": [("bar", bar_message(vwap=None))]}]
    )
    provider, _, bars, _ = make_provider(factory)
    provider.start()
    try:
        assert wait_until(lambda: provider.skipped_no_vwap_count == 1)
        assert provider.bar_count == 0
        assert bars == []
        assert provider.state is ConnectionState.CONNECTED
    finally:
        stop_and_join(provider)


def test_malformed_frames_are_counted_without_crashing():
    factory = FakeStreamFactory(
        [
            {
                "outcome": "block",
                "script": [
                    ("trade", "not-a-mapping"),
                    ("trade", {"T": "t", "S": "SOXL", "p": -1.0, "s": 1.0, "t": "2026-11-06T14:30:00Z"}),
                    ("bar", {"T": "b", "S": "SOXL", "o": -1.0, "h": 40.0, "l": -2.0, "c": 39.0, "v": 1.0, "vw": 39.5, "t": "2026-11-06T14:30:00Z"}),
                    ("trade", trade_message()),
                ],
            }
        ]
    )
    provider, ticks, bars, _ = make_provider(factory)
    provider.start()
    try:
        assert wait_until(lambda: provider.tick_count == 1)
        assert provider.malformed_count == 3
        assert len(ticks) == 1
        assert bars == []
        assert provider.state is ConnectionState.CONNECTED
    finally:
        stop_and_join(provider)


def test_consumer_callback_errors_are_counted_and_do_not_kill_stream():
    def failing_tick(tick: Tick) -> None:
        raise RuntimeError("tick consumer bug")

    def failing_bar(bar: Bar) -> None:
        raise RuntimeError("bar consumer bug")

    factory = FakeStreamFactory(
        [{"outcome": "block", "script": [("trade", trade_message()), ("bar", bar_message())]}]
    )
    provider, _, _, _ = make_provider(factory, on_tick=failing_tick, on_bar=failing_bar)
    provider.start()
    try:
        assert wait_until(lambda: provider.handler_errors == 2)
        assert provider.tick_count == 1
        assert provider.bar_count == 1
        assert provider.state is ConnectionState.CONNECTED
    finally:
        stop_and_join(provider)


def test_on_bar_is_optional():
    factory = FakeStreamFactory([{"outcome": "block", "script": [("bar", bar_message())]}])
    provider, ticks, _, _ = make_provider(factory, on_bar=None)
    provider.start()
    try:
        assert wait_until(lambda: provider.bar_count == 1)
        assert ticks == []
    finally:
        stop_and_join(provider)


def test_disconnect_reconnects_and_resubscribes():
    factory = FakeStreamFactory(
        [
            {"outcome": "disconnect"},
            {"outcome": "block", "script": [("trade", trade_message())]},
        ]
    )
    provider, ticks, _, states = make_provider(factory)
    provider.start()
    try:
        assert wait_until(
            lambda: len(factory.streams) >= 2
            and provider.tick_count == 1
            and provider.state is ConnectionState.CONNECTED
        ), f"state={provider.state}"
        assert provider.reconnect_count == 1
        assert provider.connect_attempts == 2
        assert wait_until(lambda: len(ticks) == 1)
        # Every connection attempt subscribes to both channels again.
        for stream in factory.streams:
            assert stream.subscriptions == [
                ("trades", ("SOXL", "SOXS")),
                ("bars", ("SOXL", "SOXS")),
            ]
        seen = [state for state, _ in states]
        assert ConnectionState.CONNECTING in seen
        assert seen.count(ConnectionState.CONNECTED) >= 2
        assert ConnectionState.RECONNECTING in seen
    finally:
        stop_and_join(provider)


def test_auth_failure_maps_to_unauthenticated_and_stops_retrying():
    factory = FakeStreamFactory(
        [{"outcome": "auth_failed", "auth_error": "401 unauthorized: invalid API key"}]
    )
    provider, _, _, states = make_provider(factory)
    provider.start()
    assert provider.join(5.0) is True
    assert provider.state is ConnectionState.UNAUTHENTICATED
    assert len(factory.streams) == 1
    assert provider.reconnect_count == 0
    assert states[-1][0] is ConnectionState.UNAUTHENTICATED


def test_reconnect_attempts_are_capped_and_state_becomes_unavailable():
    factory = FakeStreamFactory([{"outcome": "disconnect"}])
    provider, _, _, states = make_provider(factory, data_config=fast_config(max_reconnect_attempts=2))
    provider.start()
    assert provider.join(5.0) is True
    assert provider.state is ConnectionState.UNAVAILABLE
    assert len(factory.streams) == 3  # initial attempt + 2 reconnects
    assert provider.reconnect_count == 2
    assert states[-1][0] is ConnectionState.UNAVAILABLE


def test_factory_failure_is_retried_with_backoff():
    factory = FlakyFactory()
    provider, _, _, states = make_provider(factory)
    provider.start()
    try:
        assert wait_until(
            lambda: factory.calls >= 2 and provider.state is ConnectionState.CONNECTED
        ), f"state={provider.state}"
        assert provider.reconnect_count == 1
        assert "temporary factory failure" in provider.last_error
        assert ConnectionState.RECONNECTING in [state for state, _ in states]
    finally:
        stop_and_join(provider)


def test_stop_is_idempotent_and_joins():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.start()
    assert provider.wait_until_connected(5.0) is True
    provider.stop()
    provider.stop()
    assert provider.join(5.0) is True
    assert provider.state is ConnectionState.DISCONNECTED
    assert factory.streams[0].stop_calls >= 1


def test_join_returns_false_while_stream_is_running():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.start()
    try:
        assert provider.join(0.02) is False
    finally:
        stop_and_join(provider)


def test_start_twice_is_idempotent():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.start()
    try:
        assert provider.wait_until_connected(5.0) is True
        provider.start()
        time.sleep(0.02)
        assert len(factory.streams) == 1
    finally:
        stop_and_join(provider)


def test_stop_before_start_is_harmless():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.stop()
    assert provider.join(0.1) is True
    assert provider.state is ConnectionState.DISCONNECTED
    assert factory.streams == []


def test_can_restart_after_stop():
    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory)
    provider.start()
    assert provider.wait_until_connected(5.0) is True
    stop_and_join(provider)

    provider.start()
    try:
        assert provider.wait_until_connected(5.0) is True
        assert wait_until(lambda: len(factory.streams) == 2)
        assert provider.state is ConnectionState.CONNECTED
    finally:
        stop_and_join(provider)
    assert provider.state is ConnectionState.DISCONNECTED


def test_on_state_callback_errors_are_contained():
    def broken_state_callback(state, reason):
        raise RuntimeError("dashboard bug")

    factory = FakeStreamFactory()
    provider, _, _, _ = make_provider(factory, on_state=broken_state_callback)
    provider.start()
    try:
        assert provider.wait_until_connected(5.0) is True
        assert provider.handler_errors >= 1
        assert provider.state is ConnectionState.CONNECTED
    finally:
        stop_and_join(provider)


def test_constructor_validation():
    with pytest.raises(TypeError):
        AlpacaIEXProvider(make_credentials(), fast_config(), on_tick=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AlpacaIEXProvider(make_credentials(), fast_config(), on_tick=lambda tick: None, on_bar="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AlpacaIEXProvider(
            make_credentials(),
            replace(load_config().data, symbols=()),
            on_tick=lambda tick: None,
        )


def test_repr_hides_credentials():
    provider, _, _, _ = make_provider(FakeStreamFactory())
    text = repr(provider)
    assert FAKE_SECRET not in text
    assert FAKE_API_KEY not in text
    assert "AlpacaIEXProvider" in text
    assert "iex" in text


def test_provider_uses_iex_feed_name():
    provider, _, _, _ = make_provider(FakeStreamFactory())
    assert provider.feed == "iex"
    assert provider.symbols == ("SOXL", "SOXS")


# --------------------------------------------------------------------------- #
# Real stream factory (constructed offline, never run) + probe
# --------------------------------------------------------------------------- #
def test_default_stream_factory_builds_iex_raw_stream_without_network():
    hooks = AlpacaStreamHooks(on_connected=lambda: None, on_auth_failed=lambda message: None)
    stream = alp._default_stream_factory(make_credentials(), hooks)

    assert stream._endpoint.endswith("/v2/iex")
    assert stream._raw_data is True
    assert callable(stream.run) and callable(stream.stop)
    assert callable(stream.subscribe_trades) and callable(stream.subscribe_bars)

    async def handler(message):
        return None

    stream.subscribe_trades(handler, "SOXL", "SOXS")
    stream.subscribe_bars(handler, "SOXL", "SOXS")
    assert set(stream._handlers["trades"]) == {"SOXL", "SOXS"}
    assert set(stream._handlers["bars"]) == {"SOXL", "SOXS"}
    assert stream._running is False  # subscribing offline must not connect


def _hook_recorder():
    calls = {"connected": 0, "auth": []}
    hooks = AlpacaStreamHooks(
        on_connected=lambda: calls.__setitem__("connected", calls["connected"] + 1),
        on_auth_failed=lambda message: calls["auth"].append(message),
    )
    return hooks, calls


def test_hookable_stream_reports_connected_offline(monkeypatch):
    from alpaca.data.live.websocket import DataStream

    async def ok_connect(self):
        return None

    async def ok_auth(self):
        return None

    monkeypatch.setattr(DataStream, "_connect", ok_connect)
    monkeypatch.setattr(DataStream, "_auth", ok_auth)
    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    asyncio.run(stream._start_ws())
    assert calls["connected"] == 1
    assert calls["auth"] == []


def test_hookable_stream_maps_auth_failure_to_hook(monkeypatch):
    from alpaca.data.live.websocket import DataStream

    async def ok_connect(self):
        return None

    async def bad_auth(self):
        raise ValueError("401 Unauthorized: invalid key")

    monkeypatch.setattr(DataStream, "_connect", ok_connect)
    monkeypatch.setattr(DataStream, "_auth", bad_auth)
    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    with pytest.raises(alp._StreamAuthFailed):
        asyncio.run(stream._start_ws())
    assert calls["connected"] == 0
    assert len(calls["auth"]) == 1
    assert "401" in calls["auth"][0]


def test_hookable_stream_does_not_classify_network_errors_as_auth(monkeypatch):
    from alpaca.data.live.websocket import DataStream

    async def bad_connect(self):
        raise ValueError("connected message not received")

    async def ok_auth(self):
        return None

    monkeypatch.setattr(DataStream, "_connect", bad_connect)
    monkeypatch.setattr(DataStream, "_auth", ok_auth)
    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(stream._start_ws())
    assert not isinstance(excinfo.value, alp._StreamAuthFailed)
    assert calls["connected"] == 0
    assert calls["auth"] == []


def test_hookable_stream_surfaces_disconnect_as_reconnect_signal(monkeypatch):
    from alpaca.data.live.websocket import DataStream
    from websockets.exceptions import WebSocketException

    async def ok_connect(self):
        return None

    async def ok_auth(self):
        return None

    async def noop_send(self):
        return None

    async def dropped_consume(self, run=None):
        raise WebSocketException("socket dropped")

    monkeypatch.setattr(DataStream, "_connect", ok_connect)
    monkeypatch.setattr(DataStream, "_auth", ok_auth)
    monkeypatch.setattr(DataStream, "_send_subscribe_msg", noop_send)
    monkeypatch.setattr(DataStream, "_consume", dropped_consume)

    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    async def handler(message):
        return None

    stream.subscribe_trades(handler, "SOXL")
    stream.subscribe_bars(handler, "SOXL")
    with pytest.raises(alp._StreamReconnect):
        asyncio.run(stream._run_forever())
    assert calls["connected"] == 1


def test_hookable_stream_maps_auth_error_frame_to_hook():
    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    asyncio.run(stream._dispatch({"T": "error", "code": 401, "msg": "unauthorized"}))
    assert len(calls["auth"]) == 1
    assert "401" in calls["auth"][0]
    assert stream._stop_before_first_run is True  # forces the socket to stop


def test_hookable_stream_ignores_non_auth_error_frames():
    hooks, calls = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    asyncio.run(stream._dispatch({"T": "error", "code": 500, "msg": "internal error"}))
    assert calls["auth"] == []
    assert stream._stop_before_first_run is False


def test_wait_before_reconnect_honours_stop_and_hands_back_supervision():
    import types

    hooks, _ = _hook_recorder()
    stream = alp._default_stream_factory(make_credentials(), hooks)

    stopping = types.SimpleNamespace(stopping=True)
    asyncio.run(stream._wait_before_reconnect(2, stopping))  # must not raise

    with pytest.raises(alp._StreamReconnect):
        asyncio.run(stream._wait_before_reconnect(2, types.SimpleNamespace(stopping=False)))


class FakeModel:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, mode=None):
        return dict(self._payload)


class FakeProbeClient:
    def __init__(self, results: dict[str, object]) -> None:
        self.results = results
        self.requests: list[tuple[str, object]] = []

    def _resolve(self, name: str):
        value = self.results.get(name, {})
        if isinstance(value, BaseException):
            raise value
        return value

    def get_stock_latest_bar(self, request):
        self.requests.append(("bar", request))
        return self._resolve("bar")

    def get_stock_latest_quote(self, request):
        self.requests.append(("quote", request))
        return self._resolve("quote")

    def get_stock_latest_trade(self, request):
        self.requests.append(("trade", request))
        return self._resolve("trade")


def test_probe_success_returns_structured_json_safe_result():
    client = FakeProbeClient(
        {
            "bar": {"SOXL": FakeModel({"o": 40.0, "c": 40.1, "vw": 40.05, "t": "2026-11-06T14:30:00Z"})},
            "quote": {"SOXL": {"bp": 40.0, "ap": 40.02}},
            "trade": {"SOXS": {"p": 6.1, "s": 10.0}},
        }
    )
    result = probe_alpaca(
        make_credentials(),
        symbols=("soxl", "SOXS", "SOXL"),
        client_factory=lambda credentials: client,
    )

    assert isinstance(result, ProbeResult)
    assert result.ok is True
    assert result.any_ok is True
    assert result.feed == "iex"
    assert result.symbols == ("SOXL", "SOXS")
    assert result.bars.data["SOXL"]["c"] == 40.1
    assert result.quotes.data["SOXL"]["ap"] == 40.02
    assert result.trades.data["SOXS"]["p"] == 6.1
    json.dumps(result.to_dict())  # must be JSON serialisable for main.py check

    from alpaca.data.enums import DataFeed

    first_request = client.requests[0][1]
    assert list(first_request.symbol_or_symbols) == ["SOXL", "SOXS"]
    assert first_request.feed == DataFeed.IEX


def test_probe_auth_failure_is_normalised_not_raised():
    class FakeAPIError(Exception):
        status_code = 401

    client = FakeProbeClient(
        {
            "bar": FakeAPIError("invalid key"),
            "quote": FakeAPIError("invalid key"),
            "trade": FakeAPIError("invalid key"),
        }
    )
    result = probe_alpaca(make_credentials(), client_factory=lambda credentials: client)
    assert result.ok is False
    assert all(item.normalized_error == "auth" for item in result.items)
    assert "HTTP 401" in result.bars.error


def test_probe_partial_failure_is_structured():
    client = FakeProbeClient(
        {
            "bar": {"SOXL": {"c": 40.0}},
            "quote": RuntimeError("market data boom"),
            "trade": {"SOXL": {"p": 40.0}},
        }
    )
    result = probe_alpaca(make_credentials(), client_factory=lambda credentials: client)
    assert result.ok is False
    assert result.any_ok is True
    assert result.bars.ok is True and result.trades.ok is True
    assert result.quotes.ok is False
    assert result.quotes.normalized_error == "error"
    assert "market data boom" in result.quotes.error


def test_probe_client_factory_failure_fills_every_item():
    def broken_factory(credentials):
        raise RuntimeError("cannot create client")

    result = probe_alpaca(make_credentials(), client_factory=broken_factory)
    assert result.ok is False
    assert all(not item.ok for item in result.items)
    assert all("cannot create client" in item.error for item in result.items)
    assert all(item.normalized_error == "error" for item in result.items)


def test_probe_redacts_secret_from_errors():
    def leaking_factory(credentials):
        raise RuntimeError(f"init failed for {credentials.secret_key}")

    result = probe_alpaca(make_credentials(), client_factory=leaking_factory)
    payload = json.dumps(result.to_dict())
    assert FAKE_SECRET not in payload
    assert "***" in result.bars.error


def test_probe_rejects_empty_symbols():
    with pytest.raises(ValueError):
        probe_alpaca(make_credentials(), symbols=())
