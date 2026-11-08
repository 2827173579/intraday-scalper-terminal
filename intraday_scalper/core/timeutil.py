"""Time-zone and US-market-session helpers.

Every internal timestamp is UTC. Display layers convert to
``America/New_York`` (exchange clock) and ``Asia/Seoul`` (user clock) using
``zoneinfo`` so daylight-saving transitions are handled by the IANA database
rather than by hand.

Known limitation: exchange *holidays* are not modelled here. ``is_trading_day``
only excludes weekends; callers that need holiday accuracy should supply an
optional calendar (see ``SessionCalendar`` in later phases) or simply observe
that no fresh trades arrive, which the data-quality monitor reports as
``DATA_UNAVAILABLE``.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .models import SessionPhase

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)
PREMARKET_OPEN_ET = time(4, 0)
AFTERHOURS_CLOSE_ET = time(20, 0)

OPENING_RANGE_5_END_ET = time(9, 35)
OPENING_RANGE_15_END_ET = time(9, 45)
NO_NEW_ENTRIES_ET = time(15, 30)
EOD_CHECK_ET = time(15, 50)
EOD_URGENT_ET = time(15, 55)


# --------------------------------------------------------------------------- #
# Clock & conversion
# --------------------------------------------------------------------------- #
def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_utc(ts: datetime) -> datetime:
    """Return *ts* in UTC; a naive datetime is assumed to already be UTC."""
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def to_et(ts: datetime) -> datetime:
    """Convert a datetime to the exchange clock (America/New_York)."""
    return to_utc(ts).astimezone(ET)


def to_kst(ts: datetime) -> datetime:
    """Convert a datetime to the user clock (Asia/Seoul)."""
    return to_utc(ts).astimezone(KST)


def et_date(ts: datetime) -> date:
    """Exchange-local calendar date for *ts*."""
    return to_et(ts).date()


def minute_floor_utc(ts: datetime) -> datetime:
    """Floor a UTC timestamp down to the start of its minute."""
    utc = to_utc(ts)
    return utc.replace(second=0, microsecond=0)


def minute_floor_et(ts: datetime) -> datetime:
    """Floor an instant to the start of its *exchange-local* minute.

    For minutes inside the session this equals ``minute_floor_utc`` because the
    US/Eastern offset is a whole number of minutes; it exists to make bar
    boundaries explicit at the exchange clock.
    """
    local = to_et(ts)
    floored = local.replace(second=0, microsecond=0)
    return floored.astimezone(UTC)


def add_minutes(ts: datetime, minutes: int) -> datetime:
    return to_utc(ts) + timedelta(minutes=minutes)


# --------------------------------------------------------------------------- #
# Trading calendar stubs (weekends only; holidays intentionally not guessed)
# --------------------------------------------------------------------------- #
def is_trading_day(ts: datetime) -> bool:
    """True when the exchange-local weekday is Monday..Friday."""
    return to_et(ts).weekday() < 5


def is_regular_session(ts: datetime) -> bool:
    """True between 09:30 and 16:00 ET on a weekday."""
    if not is_trading_day(ts):
        return False
    local = to_et(ts).time()
    return MARKET_OPEN_ET <= local < MARKET_CLOSE_ET


def is_premarket(ts: datetime) -> bool:
    if not is_trading_day(ts):
        return False
    local = to_et(ts).time()
    return PREMARKET_OPEN_ET <= local < MARKET_OPEN_ET


def is_after_hours(ts: datetime) -> bool:
    if not is_trading_day(ts):
        return False
    local = to_et(ts).time()
    return MARKET_CLOSE_ET <= local < AFTERHOURS_CLOSE_ET


def market_phase(ts: datetime) -> SessionPhase:
    """Map an instant to the trading-session phase used by the signal engine."""
    if not is_trading_day(ts):
        return SessionPhase.CLOSED
    local = to_et(ts).time()
    if local < PREMARKET_OPEN_ET:
        return SessionPhase.CLOSED
    if local < MARKET_OPEN_ET:
        return SessionPhase.PREMARKET
    if local < OPENING_RANGE_5_END_ET:
        return SessionPhase.OPENING_RANGE_5
    if local < OPENING_RANGE_15_END_ET:
        return SessionPhase.OPENING_RANGE_15
    if local < NO_NEW_ENTRIES_ET:
        return SessionPhase.REGULAR
    if local < EOD_CHECK_ET:
        return SessionPhase.NO_NEW_ENTRIES
    if local < EOD_URGENT_ET:
        return SessionPhase.EOD_CHECK
    if local < MARKET_CLOSE_ET:
        return SessionPhase.EOD_URGENT
    if local < AFTERHOURS_CLOSE_ET:
        return SessionPhase.AFTER_HOURS
    return SessionPhase.CLOSED


def phase_allows_new_entries(phase: SessionPhase) -> bool:
    """Only the pre-15:30 phases of an open session may create entries.

    The opening-range phases are marked as ``OPENING_RANGE_5`` /
    ``OPENING_RANGE_15``: the ORB strategy still needs the range to be complete,
    which the strategy itself checks via bar timestamps.
    """
    return phase in {
        SessionPhase.OPENING_RANGE_5,
        SessionPhase.OPENING_RANGE_15,
        SessionPhase.REGULAR,
    }


def session_open_utc(ts: datetime) -> datetime:
    """09:30 ET of the exchange-local date of *ts*, in UTC."""
    local = to_et(ts)
    return datetime.combine(local.date(), MARKET_OPEN_ET, tzinfo=ET).astimezone(UTC)


def session_close_utc(ts: datetime) -> datetime:
    """16:00 ET of the exchange-local date of *ts*, in UTC."""
    local = to_et(ts)
    return datetime.combine(local.date(), MARKET_CLOSE_ET, tzinfo=ET).astimezone(UTC)


def session_bounds_utc(ts: datetime) -> tuple[datetime, datetime]:
    """``(open_utc, close_utc)`` for the exchange session containing *ts*."""
    return session_open_utc(ts), session_close_utc(ts)


def elapsed_session_minutes(ts: datetime) -> float:
    """Minutes since 09:30 ET; negative before the open."""
    return (to_utc(ts) - session_open_utc(ts)).total_seconds() / 60.0


def opening_range_bounds_utc(ts: datetime, minutes: int) -> tuple[datetime, datetime]:
    """``(start, end)`` UTC boundaries of the opening range for *minutes*."""
    if minutes not in (5, 15):
        raise ValueError("opening range only supports 5 or 15 minutes")
    open_utc = session_open_utc(ts)
    return open_utc, open_utc + timedelta(minutes=minutes)


def human_time(ts: datetime) -> str:
    """Compact display string: ET + KST side by side."""
    et = to_et(ts)
    kst = to_kst(ts)
    return f"{et:%Y-%m-%d %H:%M:%S} ET / {kst:%m-%d %H:%M:%S} KST"
