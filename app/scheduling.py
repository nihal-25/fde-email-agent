"""Meeting-scheduling orchestrator (Phase 2, case 1).

Given a meeting_request thread, the prior negotiation state, and a fixed "now"
anchor (IST), this decides what to do and produces:
  - the reply draft text,
  - a booking dict IF the draft is a confirmation (else None),
  - the updated per-thread scheduling state,
  - review flags.

It is deliberately pure of I/O persistence and Slack/approval: the worker calls
handle(), persists the result (db.persist_processing booking=...,
db.set_scheduling_state), and posts to Slack. Calendar reads and the LLM are the
only external calls, and both are injectable for tests.

Time rules (per spec):
  * IST (Asia/Kolkata), 30-min default meetings.
  * The LLM resolves relative expressions anchored to the passed-in `now` only.
  * A resolved time that is missing/unparseable/in the past never books — it
    degrades to a clarify-draft. (Defense-in-depth on top of the LLM's
    next-occurrence resolution.)
"""

from __future__ import annotations

import email.utils
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import calendar_client as cal
from app import llm
from app.calendar_client import IST
from app.cli import render_thread

PROPOSE_HORIZON_DAYS = 7
MAX_PROPOSALS = 3

# Scheduling stages stored in SchedulingState.state["stage"].
STAGE_AWAITING_TIME = "awaiting_time"   # we asked them for a time
STAGE_PROPOSED = "proposed"             # we offered specific slot(s)
STAGE_CONFIRMING = "confirming"         # a booking draft is awaiting approval
STAGE_CLARIFY = "clarify"               # we asked them to clarify a time


@dataclass
class SchedulingResult:
    draft_text: str
    booking: dict | None
    state: dict
    flags: list = field(default_factory=list)


# --- Formatting -------------------------------------------------------------

def _fmt_time(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d}"


def _meridiem(dt: datetime) -> str:
    return "am" if dt.hour < 12 else "pm"


def _one_zone(start: datetime, end: datetime, label: str, *, with_day: bool = True) -> str:
    if _meridiem(start) == _meridiem(end):
        times = f"{_fmt_time(start)}-{_fmt_time(end)} {_meridiem(end)}"
    else:
        times = f"{_fmt_time(start)} {_meridiem(start)}-{_fmt_time(end)} {_meridiem(end)}"
    if with_day:
        day = f"{start.strftime('%a')} {start.day} {start.strftime('%b')}"
        return f"{day}, {times} {label}"
    return f"{times} {label}"


def format_slot(start: datetime, end: datetime, other_tz: str | None = None) -> str:
    """Human label for a slot. IST is the reference:
        'Thu 25 Jun, 3:00-3:30 pm IST'
    If the customer stated another timezone, show theirs first, then IST:
        'Thu 25 Jun, 4:00-4:30 pm PST / 3:30-4:00 am IST'  (IST day shown too if
        it differs across the date line). Used in drafts and the Slack card."""
    ist_s, ist_e = cal._to_ist(start), cal._to_ist(end)
    if not other_tz:
        return _one_zone(ist_s, ist_e, "IST")
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(other_tz)
    except Exception:
        return _one_zone(ist_s, ist_e, "IST")  # bad/unknown zone -> IST only
    cust_s, cust_e = ist_s.astimezone(zone), ist_e.astimezone(zone)
    cust_label = cust_s.tzname() or other_tz  # e.g. PST/PDT, DST-correct
    cust_part = _one_zone(cust_s, cust_e, cust_label)
    # Include the IST day only when it lands on a different calendar date.
    ist_part = _one_zone(ist_s, ist_e, "IST", with_day=(ist_s.date() != cust_s.date()))
    return f"{cust_part} / {ist_part}"


# Deterministic timezone guard. The LLM is told to set stated_timezone only when
# the customer explicitly writes a zone, but it is not perfectly reliable (it can
# infer a zone from the sender's -0700 headers). So we ENFORCE the rule in code:
# honor stated_timezone only if an explicit zone token appears in the customer's
# own (de-quoted) message text. Otherwise everything is IST.
_TZ_TOKEN_RE = re.compile(
    r"\b(?:IST|PST|PDT|PT|EST|EDT|ET|CST|CDT|CT|MST|MDT|MT|GMT|UTC|BST|WET|"
    r"CET|CEST|EET|EEST|JST|KST|SGT|HKT|AEST|AEDT|NZST|AST)\b"
    r"|\btime\s?zone\b|\b(?:my|your|local)\s+time\b",
    re.IGNORECASE,
)
_QUOTE_HEADER_RE = re.compile(r"^\s*On .*wrote:\s*$")


def _strip_quoted(body: str) -> str:
    """Drop quoted reply history so we only inspect the customer's NEW text
    (quoted history can contain 'IST' from our own prior draft)."""
    out = []
    for line in (body or "").splitlines():
        if line.lstrip().startswith(">") or _QUOTE_HEADER_RE.match(line):
            break
        out.append(line)
    return "\n".join(out)


def _explicit_tz(stated_tz: str | None, customer_body: str | None) -> str | None:
    """Return stated_tz only if the customer actually wrote a timezone word in
    their new message; else None (default IST). Never trust an inferred zone."""
    if not stated_tz:
        return None
    return stated_tz if _TZ_TOKEN_RE.search(_strip_quoted(customer_body)) else None


def _parse_ist(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return cal._to_ist(datetime.fromisoformat(iso.replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return None


def _resolve_requested(ext: dict, customer_body: str | None) -> datetime | None:
    """Deterministically resolve the requested start to an IST-aware datetime —
    the PARSE-side of the assume-IST spec (twin of the display-side _explicit_tz
    guard). The LLM supplies only a NAIVE wall-clock ("11:30 on Fri" -> the Friday
    date at 11:30:00, no zone); the CODE decides the zone: an EXPLICIT zone word ->
    interpret there and convert to IST; otherwise -> IST literal. No LLM zone
    inference can shift the instant."""
    if not ext.get("has_time"):
        return None
    wall = ext.get("requested_wall_clock")
    if not wall:  # legacy fallback: the model gave only the (already-shifted) IST
        return _parse_ist(ext.get("requested_start_ist"))
    try:
        naive = datetime.fromisoformat(wall.split("+")[0].replace("Z", "")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return _parse_ist(ext.get("requested_start_ist"))
    tz = _explicit_tz(ext.get("stated_timezone"), customer_body)  # explicit word only
    if tz:
        try:
            return naive.replace(tzinfo=ZoneInfo(tz)).astimezone(IST)
        except Exception:
            pass  # unknown zone -> fall through to IST literal
    return naive.replace(tzinfo=IST)                              # bare -> IST literal


def _sender_tzinfo(thread: dict):
    """The tzinfo from the latest customer message's Date header — the sender's
    real zone, used for DISCLOSURE ONLY (never the parse). None if unparseable."""
    msgs = thread.get("messages") or []
    d = (msgs[-1] if msgs else {}).get("date")
    if not d:
        return None
    try:
        return email.utils.parsedate_to_datetime(d).tzinfo
    except (ValueError, TypeError):
        return None


def _ist_assumption_note(requested: datetime, sender_tz, explicit_tz: str | None) -> str | None:
    """When we defaulted a zoneless time to IST but the SENDER is in a different
    zone, surface the AMBIGUITY: state our assumption (IST) and name the alternative
    in the CUSTOMER'S OWN terms — the same clock time they typed, read as their
    local time — then invite correction. We do NOT convert their time backwards and
    assert a clock nobody meant, and we never print a raw UTC offset. None when a
    zone was explicitly stated, there's no sender zone, or the sender is IST."""
    if explicit_tz or sender_tz is None:
        return None
    off_sender = sender_tz.utcoffset(requested)
    if off_sender is None or off_sender == IST.utcoffset(requested):
        return None
    clock = requested.strftime("%I:%M %p").lstrip("0").lower()   # the customer's stated clock
    return (f"I assumed IST — if you meant {clock} your local time, "
            f"just say so and I'll adjust")


def _title(classification: dict) -> str:
    company = (classification or {}).get("company")
    name = (classification or {}).get("customer_name")
    if company:
        return f"Plivo <> {company}"
    if name:
        return f"Plivo <> {name}"
    return "Plivo intro call"


def _booking(start: datetime, end: datetime, attendee: str, classification: dict, label: str) -> dict:
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_min": int((end - start).total_seconds() // 60),
        "attendee_email": attendee,
        "title": _title(classification),
        "timezone": cal.TIMEZONE,
        "label": label,  # human "Thu 25 Jun, 3:00-3:30 pm IST" for the Slack card
    }


def _base_state(duration_min: int, attendee: str) -> dict:
    return {
        "stage": STAGE_AWAITING_TIME,
        "duration_min": duration_min,
        "timezone": cal.TIMEZONE,
        "attendee_email": attendee,
        "proposed_slots": [],
        "candidate_slot": None,
        # Set once a meeting has been booked on this thread; a later confirmation
        # then RESCHEDULES this event instead of creating a duplicate.
        "booked_event_id": None,
    }


# --- Orchestrator -----------------------------------------------------------

def handle(thread: dict, classification: dict, prior_state: dict | None,
           now: datetime, *, calendar_service=None) -> SchedulingResult:
    """Decide + draft the next scheduling move. See module docstring."""
    now = cal._to_ist(now)
    thread_text = render_thread(thread)
    # The reply_context "to" is a full "Name <email>" header (right for the email
    # To: line). The calendar attendee needs a BARE address, so parse it out.
    attendee = email.utils.parseaddr((thread.get("reply_context") or {}).get("to") or "")[1] or None
    customer_name = (classification or {}).get("customer_name")

    ext = llm.extract_scheduling(thread_text, prior_state, now.isoformat())
    # Enforce the IST-default rule deterministically: only honor a stated zone if
    # the customer's own message actually contains a timezone word.
    latest_customer_body = (thread.get("messages") or [{}])[-1].get("body")
    other_tz = _explicit_tz(ext.get("stated_timezone"), latest_customer_body)

    duration_min = (ext.get("duration_min")
                    or (prior_state or {}).get("duration_min")
                    or cal.DEFAULT_DURATION_MIN)
    dur = timedelta(minutes=duration_min)
    state = _base_state(duration_min, attendee)
    # Carry forward any already-booked event so all branches retain it.
    booked_event_id = (prior_state or {}).get("booked_event_id")
    state["booked_event_id"] = booked_event_id

    # --- Branch 1: customer is flexible -> propose open slots -----------------
    if ext.get("open_to_any"):
        slots = cal.find_free_slots(now, now + timedelta(days=PROPOSE_HORIZON_DAYS),
                                    duration_min=duration_min, max_slots=MAX_PROPOSALS,
                                    service=calendar_service)
        return _propose(thread_text, customer_name, slots, state, classification, other_tz=other_tz)

    # Deterministic zone resolution (code owns it — LLM gives only the wall-clock).
    requested = _resolve_requested(ext, latest_customer_body)

    # --- Branch 2: a specific time, valid and in the future ------------------
    if requested is not None and requested > now:
        end = requested + dur
        # Available = inside working hours AND calendar-free. An out-of-hours
        # time is treated like busy (propose a nearby in-hours slot), even if
        # the calendar happens to be free then. within_working_hours is checked
        # first so a free/busy lookup is skipped for out-of-hours requests.
        if cal.within_working_hours(requested, end) and cal.is_free(requested, end, service=calendar_service):
            # Available -> confirmation + booking (approval-gated). If a meeting
            # is already booked on this thread, this RESCHEDULES it (updates the
            # same event) instead of creating a duplicate.
            # Disclosure: if we defaulted a zoneless time to IST while the sender is
            # in a different zone (per their Date header), SAY the assumption + their
            # local equivalent — a silent wrong-from-their-seat time causes no-shows;
            # a disclosed one gets a one-reply correction. Rides in the verbatim label.
            note = _ist_assumption_note(requested, _sender_tzinfo(thread), other_tz)
            slot_label = format_slot(requested, end, other_tz)
            if note:
                slot_label = f"{slot_label} ({note})"
            action = "reschedule" if booked_event_id else "confirm"
            draft = llm.draft_meeting_reply(action, thread_text=thread_text,
                                            times=[slot_label], customer_name=customer_name)
            state.update(stage=STAGE_CONFIRMING,
                         candidate_slot={"start": requested.isoformat(), "end": end.isoformat()})
            if attendee:
                booking = _booking(requested, end, attendee, classification, slot_label)
                if booked_event_id:
                    booking["event_id"] = booked_event_id  # -> update, not create
                return SchedulingResult(draft, booking, state, [])
            return SchedulingResult(draft, None, state, [{"type": "no_attendee",
                "text": "No customer email on thread — cannot book; review."}])
        # Busy -> propose nearby free slots from the requested time onward.
        slots = cal.find_free_slots(max(now, requested), requested + timedelta(days=PROPOSE_HORIZON_DAYS),
                                    duration_min=duration_min, max_slots=MAX_PROPOSALS,
                                    service=calendar_service)
        return _propose(thread_text, customer_name, slots, state, classification,
                        nearby=True, other_tz=other_tz)

    # --- Branch 3: a time was named but it's past/unparseable -> clarify -----
    if ext.get("has_time"):
        draft = llm.draft_meeting_reply("clarify_time", thread_text=thread_text,
                                        customer_name=customer_name)
        state["stage"] = STAGE_CLARIFY
        return SchedulingResult(draft, None, state, [])

    # --- Branch 4: no time given -> ask what works (don't propose) -----------
    draft = llm.draft_meeting_reply("ask_time", thread_text=thread_text, customer_name=customer_name)
    state["stage"] = STAGE_AWAITING_TIME
    return SchedulingResult(draft, None, state, [])


def _propose(thread_text, customer_name, slots, state, classification, *, nearby=False, other_tz=None):
    """Draft a propose_slots/propose_nearby reply (no booking yet)."""
    if not slots:
        # Nothing free in the horizon -> fall back to asking them for a time.
        draft = llm.draft_meeting_reply("ask_time", thread_text=thread_text, customer_name=customer_name)
        state["stage"] = STAGE_AWAITING_TIME
        return SchedulingResult(draft, None, state, [])
    labels = [format_slot(s, e, other_tz) for s, e in slots]
    action = "propose_nearby" if nearby else "propose_slots"
    draft = llm.draft_meeting_reply(action, thread_text=thread_text, times=labels,
                                    customer_name=customer_name)
    state.update(stage=STAGE_PROPOSED,
                 proposed_slots=[s.isoformat() for s, _ in slots])
    return SchedulingResult(draft, None, state, [])
