"""Scheduling timezone-parse class (the draft-1221 bug): the code owns the zone
decision, not the LLM. Bare time -> IST literal; explicit zone word -> convert.
Proves BOTH failure directions are dead: the polite false-unavailable (PT sender)
AND the catastrophic false-confirmation/double-book (a +0900 shift). Plus the
disclosure, the relative-weekday anchor (live), and availability-error-holds.
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from unittest import mock

from app import scheduling, llm, calendar_client as cal
from app.calendar_client import IST

FRI = datetime(2026, 7, 10, 11, 30, tzinfo=IST)


def _ext(wall, stated=None):
    return {"has_time": True, "requested_wall_clock": wall, "requested_start_ist": None,
            "stated_timezone": stated, "open_to_any": False}


# ---------- (1) polite failure: PT sender, bare local time -> IST -----------
def test_bare_time_from_pt_sender_resolves_to_ist_literal():
    # "11:30 on Friday" from a -0700 sender, no tz word -> 11:30 IST (NOT 23:00).
    got = scheduling._resolve_requested(_ext("2026-07-10T11:30:00"), "How about 11:30 on Friday")
    assert got == FRI
    assert got.hour == 11 and got.utcoffset() == timedelta(hours=5, minutes=30)


# ---------- (2) CATASTROPHIC twin: a shift would FALSELY CONFIRM -------------
def test_catastrophic_shift_books_ist_literal_not_shifted():
    # +0900 sender, "15:00 tomorrow", LLM even mis-infers Asia/Tokyo — but no tz
    # WORD, so the deterministic guard books 15:00 IST, NOT 15:00 JST(=11:30 IST).
    # A shift here would have silently confirmed the WRONG instant and double-booked.
    got = scheduling._resolve_requested(_ext("2026-07-08T15:00:00", stated="Asia/Tokyo"),
                                        "15:00 tomorrow works")
    assert got == datetime(2026, 7, 8, 15, 0, tzinfo=IST)      # IST literal
    assert got != datetime(2026, 7, 8, 11, 30, tzinfo=IST)     # NOT the JST->IST shift


# ---------- (4) explicit zone WORD -> honored (converted), not literal ------
def test_explicit_zone_word_is_converted_to_ist():
    # "11:30 PST on Friday" — an explicit word, so honor it: 11:30 America/LA -> IST.
    got = scheduling._resolve_requested(_ext("2026-07-10T11:30:00", stated="America/Los_Angeles"),
                                        "11:30 PST on Friday")
    expected = datetime(2026, 7, 10, 11, 30, tzinfo=ZoneInfo("America/Los_Angeles")).astimezone(IST)
    assert got == expected
    assert got != FRI                                          # NOT re-anchored to literal IST


def test_inferred_zone_without_word_is_ignored():
    # stated_timezone present (LLM inferred) but NO tz word -> guard ignores it -> IST.
    got = scheduling._resolve_requested(_ext("2026-07-10T11:30:00", stated="America/Los_Angeles"),
                                        "How about 11:30 on Friday")   # no zone word
    assert got == FRI


# ---------- disclosure: assumed-IST for a cross-zone sender -----------------
def test_ist_assumption_note_surfaces_ambiguity_not_backwards_time():
    pt = timezone(timedelta(hours=-7))          # FRI is 11:30 IST; sender is -0700
    note = scheduling._ist_assumption_note(FRI, pt, explicit_tz=None)
    assert note and "assumed IST" in note and "your local time" in note
    # SEMANTICS: it names the customer's OWN stated clock time (11:30) as the alternative
    assert "11:30" in note
    # and must NOT assert a backwards-converted time nobody meant...
    for backwards in ("11:00 PM", "11:00 pm", "23:00", "11 PM"):
        assert backwards not in note
    # ...nor print a raw UTC offset (name the zone or say 'your local time')
    for raw in ("UTC-", "-07:00", "-0700", "UTC-07:00"):
        assert raw not in note
    # no note when the sender is effectively IST, or a real zone was explicitly stated
    assert scheduling._ist_assumption_note(FRI, IST, None) is None
    assert scheduling._ist_assumption_note(FRI, pt, "America/Los_Angeles") is None


def test_ist_assumption_note_offset_only_zone_says_your_local_time_plainly():
    # A fixed-offset tzinfo (no zone name) must NOT leak "UTC-07:00" — plain phrasing.
    pt = timezone(timedelta(hours=-7))          # tzname() is like 'UTC-07:00'
    note = scheduling._ist_assumption_note(FRI, pt, explicit_tz=None)
    assert "your local time" in note and "UTC" not in note


# ---------- (3) relative-weekday anchoring (live LLM) -----------------------
def test_relative_weekday_anchors_to_next_friday_live():
    now = datetime(2026, 7, 6, 21, 29, tzinfo=IST)             # a Monday
    thread_text = ('From: Sam <x@ext.com>\nDate: Mon, 6 Jul 2026 09:00:53 -0700\n\n'
                   'How about 11:30 on Friday')
    ext = llm.extract_scheduling(thread_text, None, now.isoformat())
    wall = ext.get("requested_wall_clock") or ""
    assert wall.startswith("2026-07-10T11:30")                # Friday = Jul 10, 11:30 as written
    assert not ext.get("stated_timezone")                     # no zone inferred from the header


# ---------- (5) availability ERROR holds (never a false 'unavailable') ------
def test_availability_error_propagates_not_false_unavailable():
    # An error in the free/busy check must bubble to the worker's safety net
    # ([scheduling-failed] keeps the generic draft) — it must NOT be swallowed into
    # a confident "unavailable" answer.
    thread = {"thread_id": "t", "reply_context": {"to": "x@ext.com"},
              "messages": [{"date": "Mon, 6 Jul 2026 09:00:53 -0700", "body": "11:30 on Friday"}]}
    now = datetime(2026, 7, 6, 21, 29, tzinfo=IST)
    with mock.patch.object(llm, "extract_scheduling",
                           return_value=_ext("2026-07-10T11:30:00")), \
         mock.patch.object(cal, "within_working_hours", return_value=True), \
         mock.patch.object(cal, "is_free", side_effect=RuntimeError("freebusy down")):
        try:
            scheduling.handle(thread, {"customer_name": "Sam"}, None, now)
            assert False, "expected the availability error to propagate"
        except RuntimeError as e:
            assert "freebusy down" in str(e)
