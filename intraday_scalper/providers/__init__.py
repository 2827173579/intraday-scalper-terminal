"""Data providers: Finnhub websocket/REST and the data-quality monitor.

Nothing in this package places orders; it only obtains and validates market
data. The websocket is the only real-time source — REST ``/quote`` is limited to
``startup_quote_checks_max`` calls per run and is never polled in the loop.
"""

from .data_quality import DataQualityMonitor, QualityReport, SymbolQuality
from .finnhub_rest import DataUnavailableError, FinnhubRestClient, QuoteSnapshot
from .finnhub_ws import (
    FinnhubWebSocketClient,
    WsConnectOptions,
    WsConnection,
    WsHandlers,
    websocket_connector,
)

__all__ = [
    "DataQualityMonitor",
    "DataUnavailableError",
    "FinnhubRestClient",
    "FinnhubWebSocketClient",
    "QualityReport",
    "QuoteSnapshot",
    "SymbolQuality",
    "WsConnectOptions",
    "WsConnection",
    "WsHandlers",
    "websocket_connector",
]
