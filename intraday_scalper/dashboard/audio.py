"""Audible alerts for the terminal dashboard.

The sound is a *reminder only*: the system never places or closes an order.
``winsound`` is Windows-only, so on other platforms this is a silent no-op.
Tests inject ``beep_fn``/``sleep_fn`` so no sound is ever produced in CI.

Patterns (by :class:`AlertLevel`):

* ``INFO``     — one short beep;
* ``WARNING``  — two short beeps;
* ``CRITICAL`` — four fast, rising beeps (EOD urgent / stop touched / data lost).

Any failure inside the sound layer is swallowed: a missing audio device must
never crash the alerting pipeline.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple

from ..core.models import AlertLevel

__all__ = ["alert_sound", "BEEP_PATTERNS"]

BeepFn = Callable[[int, int], None]
SleepFn = Callable[[float], None]

#: (frequency Hz, duration ms) sequences per alert level.
BEEP_PATTERNS: dict[AlertLevel, Tuple[Tuple[int, int], ...]] = {
    AlertLevel.INFO: ((880, 70),),
    AlertLevel.WARNING: ((700, 80), (700, 80)),
    AlertLevel.CRITICAL: ((1050, 60), (1150, 60), (1250, 60), (1450, 90)),
}

#: Pause inserted between two beeps of a pattern.
_GAP_SECONDS = 0.04


def alert_sound(
    level: AlertLevel,
    *,
    enabled: bool = True,
    beep_fn: Optional[BeepFn] = None,
    sleep_fn: Optional[SleepFn] = None,
) -> int:
    """Play the reminder pattern for *level*; return the number of beeps played.

    Parameters are injectable for tests: with ``beep_fn`` given no signal is
    produced and ``sleep_fn`` defaults to a no-op when a fake is provided.
    """
    if not enabled:
        return 0
    try:
        pattern = BEEP_PATTERNS.get(AlertLevel(level))
    except (ValueError, TypeError):
        pattern = BEEP_PATTERNS[AlertLevel.INFO]
    if not pattern:
        return 0

    if beep_fn is None:
        if sys.platform != "win32":
            return 0
        try:
            import winsound  # type: ignore[import-not-found]

            beep_fn = winsound.Beep
        except Exception:  # noqa: BLE001 - no audio device / winsound missing
            return 0
    sleeper: SleepFn = sleep_fn if sleep_fn is not None else time.sleep

    played = 0
    for index, (frequency, duration_ms) in enumerate(pattern):
        if index:
            try:
                sleeper(_GAP_SECONDS)
            except Exception:  # noqa: BLE001
                pass
        try:
            beep_fn(int(frequency), int(duration_ms))
            played += 1
        except Exception:  # noqa: BLE001 - audio must never break the pipeline
            return played
    return played


def alert_sound_many(
    levels: Sequence[AlertLevel],
    *,
    enabled: bool = True,
    beep_fn: Optional[BeepFn] = None,
    sleep_fn: Optional[SleepFn] = None,
) -> int:
    """Convenience helper: play one pattern per level; return total beeps."""
    played = 0
    for level in levels:
        played += alert_sound(level, enabled=enabled, beep_fn=beep_fn, sleep_fn=sleep_fn)
    return played
