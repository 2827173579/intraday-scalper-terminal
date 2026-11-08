"""Feature engineering: pure technical indicators and opening-range geometry.

Only :mod:`intraday_scalper.features.indicators` exists in phase 1. It has no
I/O and no clock access, so the signal engine and the backtester can call it
without behavioural differences.
"""

from .indicators import (
    OpeningRange,
    atr,
    ema,
    minutes_into_session,
    opening_range,
    rolling_high,
    rolling_low,
    rolling_mean,
    rsi,
    sma,
    true_range,
    volume_mean,
)

__all__ = [
    "OpeningRange",
    "atr",
    "ema",
    "minutes_into_session",
    "opening_range",
    "rolling_high",
    "rolling_low",
    "rolling_mean",
    "rsi",
    "sma",
    "true_range",
    "volume_mean",
]
