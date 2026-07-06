"""Google Calendar tool for the meeting-scheduling case (Phase 2, case 1).

Same separation as gmail_client / llm: this is the only module that talks to the
Calendar API. It reuses the single OAuth token (see gmail_client.get_credentials;
SCOPES there include calendar.events + calendar.freebusy).

Read tools (get_freebusy / is_free / find_free_slots) may run freely. create_event
is an ACTION tool and is only ever called from the human-approval path
(slack_approval.handle_approve) — never by the model.

Time handling: everything is in IST (Asia/Kolkata) unless a caller passes
tz-aware datetimes in another zone. Callers inject the datetimes (and the
"now" anchor via now_ist()); nothing here guesses a date from free text.
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- Config -----------------------------------------------------------------

TIMEZONE = os.getenv("SCHEDULING_TIMEZONE", "Asia/Kolkata")
IST = ZoneInfo(TIMEZONE)

DEFAULT_DURATION_MIN = 30
# Working hours (IST). This MIRRORS the "Working hours" set in Google Calendar:
# the Calendar API does not reliably expose that setting, so it is duplicated
# here — if you change your Calendar working hours, update these constants too.
# Used both to bound proposed slots AND for the availability rule: a slot counts
# as available only if it is within these hours AND the calendar is free.
WORK_START = (10, 30)   # 10:30 IST
WORK_END = (18, 30)     # 18:30 IST
WORKDAYS = frozenset({0, 1, 2, 3, 4})  # Mon..Fri (Python weekday())

CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# httplib2 (under googleapiclient) is not thread-safe; serialize API calls.
_API_LOCK = threading.Lock()


# --- Time helpers -----------------------------------------------------------

def now_ist() -> datetime:
    """The single source of 'now' for scheduling. Callers capture this once and
    pass it down so relative-time resolution and slot math share one anchor."""
    return datetime.now(IST)


def _to_ist(dt: datetime) -> datetime:
    """Normalize any datetime to IST (assume IST if it's naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _parse_rfc3339(value: str) -> datetime:
    """Parse a Calendar API RFC3339 timestamp to an IST-aware datetime."""
    return _to_ist(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _ceil_to_minutes(dt: datetime, step_min: int) -> datetime:
    """Round dt UP to the next multiple of step_min minutes (drop sub-minute)."""
    dt = dt.replace(second=0, microsecond=0)
    rem = dt.minute % step_min
    if rem:
        dt += timedelta(minutes=step_min - rem)
    return dt


def _overlaps_any(start: datetime, end: datetime, busy: list[tuple[datetime, datetime]]) -> bool:
    return any(b_start < end and b_end > start for b_start, b_end in busy)


def within_working_hours(start: datetime, end: datetime) -> bool:
    """True if [start, end] falls entirely within working hours on a single
    workday (Mon-Fri, WORK_START..WORK_END IST). The availability rule treats a
    slot outside these hours as unavailable even when the calendar is free."""
    start, end = _to_ist(start), _to_ist(end)
    if start.weekday() not in WORKDAYS:
        return False
    day_open = start.replace(hour=WORK_START[0], minute=WORK_START[1], second=0, microsecond=0)
    day_close = start.replace(hour=WORK_END[0], minute=WORK_END[1], second=0, microsecond=0)
    return start >= day_open and end <= day_close and start.date() == end.date()


# --- Service ----------------------------------------------------------------

def get_service():
    from googleapiclient.discovery import build

    from app import gmail_client

    creds = gmail_client.get_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _execute(request):
    with _API_LOCK:
        return request.execute()


# --- Read tools (free to run) -----------------------------------------------

def get_freebusy(start: datetime, end: datetime, *, calendar_id: str | None = None,
                 service=None) -> list[tuple[datetime, datetime]]:
    """Return busy intervals (IST-aware) on the calendar within [start, end)."""
    service = service or get_service()
    calendar_id = calendar_id or CALENDAR_ID
    body = {
        "timeMin": _to_ist(start).isoformat(),
        "timeMax": _to_ist(end).isoformat(),
        "timeZone": TIMEZONE,
        "items": [{"id": calendar_id}],
    }
    resp = _execute(service.freebusy().query(body=body))
    cal = resp.get("calendars", {}).get(calendar_id, {})
    busy = [(_parse_rfc3339(b["start"]), _parse_rfc3339(b["end"])) for b in cal.get("busy", [])]
    busy.sort()
    return busy


def is_free(start: datetime, end: datetime, *, calendar_id: str | None = None, service=None) -> bool:
    """True if [start, end) does not overlap any busy interval on the calendar."""
    busy = get_freebusy(start, end, calendar_id=calendar_id, service=service)
    return not _overlaps_any(_to_ist(start), _to_ist(end), busy)


def find_free_slots(window_start: datetime, window_end: datetime, *,
                    duration_min: int = DEFAULT_DURATION_MIN, max_slots: int = 3,
                    granularity_min: int = 30, calendar_id: str | None = None,
                    service=None) -> list[tuple[datetime, datetime]]:
    """Open slots within working hours (Mon-Fri, WORK_START..WORK_END IST) in
    [window_start, window_end). Never returns a slot starting before window_start,
    so passing now as window_start excludes past times. Returns up to max_slots.
    """
    window_start = _to_ist(window_start)
    window_end = _to_ist(window_end)
    busy = get_freebusy(window_start, window_end, calendar_id=calendar_id, service=service)
    dur = timedelta(minutes=duration_min)

    slots: list[tuple[datetime, datetime]] = []
    day = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day.date() <= window_end.date() and len(slots) < max_slots:
        if day.weekday() in WORKDAYS:
            work_start = day.replace(hour=WORK_START[0], minute=WORK_START[1])
            work_end = day.replace(hour=WORK_END[0], minute=WORK_END[1])
            cand = _ceil_to_minutes(max(work_start, window_start), granularity_min)
            while cand + dur <= work_end and cand + dur <= window_end and len(slots) < max_slots:
                if cand >= window_start and not _overlaps_any(cand, cand + dur, busy):
                    slots.append((cand, cand + dur))
                cand += timedelta(minutes=granularity_min)
        day += timedelta(days=1)
    return slots


# --- Action tool (human-approval path ONLY) ---------------------------------

def _extract_meet_link(event: dict) -> str | None:
    """Pull the Google Meet URL from a created event (hangoutLink, or the video
    entry point in conferenceData)."""
    if event.get("hangoutLink"):
        return event["hangoutLink"]
    for ep in event.get("conferenceData", {}).get("entryPoints", []) or []:
        if ep.get("entryPointType") == "video" and ep.get("uri"):
            return ep["uri"]
    return None


def create_event(summary: str, start: datetime, end: datetime, attendee_email: str, *,
                 description: str | None = None, calendar_id: str | None = None,
                 service=None) -> dict:
    """Create a calendar event with the customer as attendee, a Google Meet video
    link, and email the invite (sendUpdates='all'). ACTION TOOL — call only after
    human approval.

    Returns {"event_id", "html_link", "meet_link", "start", "end"}.
    """
    if not attendee_email:
        raise RuntimeError("create_event requires an attendee email.")
    service = service or get_service()
    calendar_id = calendar_id or CALENDAR_ID
    body = {
        "summary": summary,
        "description": description or "",
        "start": {"dateTime": _to_ist(start).isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": _to_ist(end).isoformat(), "timeZone": TIMEZONE},
        "attendees": [{"email": attendee_email}],
        # Always attach Google Meet video conferencing. requestId is the
        # idempotency key for the conference-create; conferenceDataVersion=1 is
        # required for the API to honor conferenceData.
        "conferenceData": {
            "createRequest": {
                "requestId": uuid.uuid4().hex,
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }
    event = _execute(
        service.events().insert(calendarId=calendar_id, body=body,
                                sendUpdates="all", conferenceDataVersion=1)
    )
    return {
        "event_id": event.get("id"),
        "html_link": event.get("htmlLink"),
        "meet_link": _extract_meet_link(event),
        "start": _to_ist(start).isoformat(),
        "end": _to_ist(end).isoformat(),
    }


def update_event(event_id: str, start: datetime, end: datetime, *,
                 calendar_id: str | None = None, service=None) -> dict:
    """Move an EXISTING event to a new start/end and email the update
    (sendUpdates='all'). Used to reschedule rather than create a duplicate;
    attendees and the existing Google Meet are preserved (patch). ACTION TOOL —
    call only after human approval.

    Returns {"event_id", "html_link", "meet_link", "start", "end"}.
    """
    if not event_id:
        raise RuntimeError("update_event requires an event_id.")
    service = service or get_service()
    calendar_id = calendar_id or CALENDAR_ID
    body = {
        "start": {"dateTime": _to_ist(start).isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": _to_ist(end).isoformat(), "timeZone": TIMEZONE},
    }
    event = _execute(
        service.events().patch(calendarId=calendar_id, eventId=event_id, body=body,
                               sendUpdates="all", conferenceDataVersion=1)
    )
    return {
        "event_id": event.get("id"),
        "html_link": event.get("htmlLink"),
        "meet_link": _extract_meet_link(event),
        "start": _to_ist(start).isoformat(),
        "end": _to_ist(end).isoformat(),
    }
