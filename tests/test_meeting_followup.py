"""Meeting follow-up: parse, the two safety guards (tested by INJECTION — feed a
misbehaving draft and assert the fallback fires, not just observe it not-firing),
thread selection, and the polish fixes. No network — llm / calendar / gmail seams
are mocked.
"""
from unittest import mock

from app import meeting_followup as mf

MY = "nihal.manjunath@plivo.com"
SUBJECT = 'Notes: “Foo / Bar” Jul 6, 2026'
BODY = """Notes from “Foo / Bar”
These notes have been sent to invited guests in your organization.
Open meeting notes
The content was auto-generated on July 6, 2026, 4:45 PM IST and may contain errors.

Summary
We covered onboarding and next steps for integration.

Onboarding topic
Discussed the setup flow in detail.

Suggested next steps

[Alice] Send the API docs: share the streaming guide link.
[Bob] Follow up: confirm the number provisioning timeline.

Meeting records Document Notes by Gemini Video Recording
Is the Next Steps section in this email helpful?
Google LLC, 1600"""


# ---------- parse -----------------------------------------------------------
def test_parse_extracts_sections_and_strips_footer():
    p = mf.parse_notes(SUBJECT, BODY)
    assert p["title"] == "Foo / Bar" and p["date"] == "Jul 6, 2026"
    assert "onboarding" in p["summary"].lower()
    assert [s["owner"] for s in p["next_steps"]] == ["Alice", "Bob"]
    assert "streaming guide link" in p["next_steps"][0]["text"]
    assert "Google LLC" not in mf.render_notes(p)      # footer dropped
    assert not mf.is_thin(p)


# ---------- guard 1: commitment phrasing, by INJECTION ----------------------
def test_commitment_guard_catches_first_person_plural_and_promised():
    # direct: the model's more natural phrasings must all trip the guard
    for phrase in ("we agreed", "as we agreed", "as discussed and agreed",
                   "per our agreement", "as promised", "you agreed", "you committed"):
        assert mf.commitment_violations(f"Great call. {phrase.capitalize()} to proceed.")


def test_commitment_guard_fires_and_falls_back():
    p = mf.parse_notes(SUBJECT, BODY)
    bad = ("Subject: Recap\n\nHi Alice,\n\nGreat call — as we agreed, we agreed you'd send the "
           "docs, as promised.\n\nBest regards,\nNihal")
    with mock.patch.object(mf.llm, "draft_meeting_followup", return_value=bad), \
         mock.patch.object(mf.llm, "rag_groundedness", return_value={"grounded": True}):
        draft, flags = mf.build_followup(p, "Alice")
    assert any(f["type"] == "meeting_commitment_blocked" for f in flags)
    assert mf.commitment_violations(draft) == []                       # fallback is clean
    assert "we agreed" not in draft.lower() and "as promised" not in draft.lower()
    assert "the notes captured" in draft.lower()                       # recorded-not-agreed framing


# ---------- guard 2: groundedness, by INJECTION -----------------------------
def test_groundedness_downgrade_on_untraceable_claim():
    p = mf.parse_notes(SUBJECT, BODY)
    ungrounded = ("Hi Alice,\n\nThanks for the call. I'll send you a $500 credit next week and "
                  "waive the setup fee.\n\nBest regards,\nNihal")   # nothing in the notes
    with mock.patch.object(mf.llm, "draft_meeting_followup", return_value=ungrounded), \
         mock.patch.object(mf.llm, "rag_groundedness",
                           return_value={"grounded": False, "unsupported_claims": ["$500 credit", "waive the setup fee"]}):
        draft, flags = mf.build_followup(p, "Alice")
    assert any(f["type"] == "meeting_ungrounded" for f in flags)
    assert "$500" not in draft and "waive" not in draft                # invented claims gone
    assert "the notes captured" in draft.lower()


# ---------- thin notes -> thin draft (no model call, nothing invented) ------
def test_thin_notes_yield_thin_draft():
    thin = ("Notes from “Empty”\nOpen meeting notes\nThe content was auto-generated on "
            "July 6, 2026 and may contain errors.\n\nMeeting records Document Notes by Gemini\nGoogle LLC")
    p = mf.parse_notes('Notes: “Empty” Jul 6, 2026', thin)
    assert mf.is_thin(p)
    with mock.patch.object(mf.llm, "draft_meeting_followup",
                           side_effect=AssertionError("thin notes must NOT call the model")):
        draft, flags = mf.build_followup(p, None)
    assert any(f["type"] == "meeting_thin_notes" for f in flags)
    assert "taking the time to meet" in draft.lower()
    assert "next step" not in draft.lower()                            # not enriched


# ---------- paths -----------------------------------------------------------
def test_notes_failed_visible_note():
    note = mf.note_notes_failed('Problem with the notes: “Robin / Nihal” Jul 3, 2026')
    assert "Robin / Nihal" in note and "no follow-up" in note.lower()


def test_no_calendar_match_is_fresh_with_reason():
    with mock.patch.object(mf, "_find_meeting_event", return_value=None):
        sel = mf.select_thread({"title": "Foo", "date": "Jul 6, 2026"},
                               service=None, cal_service=None, my_email=MY)
    assert sel["mode"] == "fresh" and "couldn't match a calendar event" in sel["reason"]
    assert sel["recipient_email"] is None


def test_internal_only_is_fresh_with_reason():
    ev = {"summary": "Foo", "attendees": [{"email": "x@plivo.com"}, {"email": MY}]}
    with mock.patch.object(mf, "_find_meeting_event", return_value=ev):
        sel = mf.select_thread({"title": "Foo", "date": "Jul 6, 2026"},
                               service=None, cal_service=None, my_email=MY)
    assert sel["mode"] == "fresh" and "no external attendee" in sel["reason"]


def test_thread_selected_and_displayname_preferred():
    ev = {"summary": "Foo", "attendees": [{"email": "jordan@ext.com", "displayName": "Jordan L"},
                                          {"email": MY}]}
    hit = {"thread_id": "T1", "subject": "Prior thread", "reply_context": {"to": "jordan@ext.com"}}
    with mock.patch.object(mf, "_find_meeting_event", return_value=ev), \
         mock.patch.object(mf, "_most_recent_thread_with", return_value=hit):
        sel = mf.select_thread({"title": "Foo", "date": "Jul 6, 2026"},
                               service=None, cal_service=None, my_email=MY)
    assert sel["mode"] == "thread" and sel["thread_id"] == "T1"
    assert sel["recipient_name"] == "Jordan L" and "jordan@ext.com" in sel["reason"]


def test_displayname_absent_gives_none_not_localpart():
    ev = {"summary": "Foo", "attendees": [{"email": "jordan@ext.com"}, {"email": MY}]}
    with mock.patch.object(mf, "_find_meeting_event", return_value=ev), \
         mock.patch.object(mf, "_most_recent_thread_with", return_value=None):
        sel = mf.select_thread({"title": "Foo", "date": "Jul 6, 2026"},
                               service=None, cal_service=None, my_email=MY)
    assert sel["mode"] == "fresh" and sel["recipient_name"] is None    # NOT "neeraj"


# ---------- polish: Subject-line strip --------------------------------------
def test_subject_line_stripped_from_body():
    p = mf.parse_notes(SUBJECT, BODY)
    withsubj = "Subject: Follow-up\n\nHi Alice,\n\nThanks for the call about onboarding.\n\nBest regards,\nNihal"
    with mock.patch.object(mf.llm, "draft_meeting_followup", return_value=withsubj), \
         mock.patch.object(mf.llm, "rag_groundedness", return_value={"grounded": True}):
        draft, flags = mf.build_followup(p, "Alice")
    assert not draft.lower().startswith("subject:") and draft.startswith("Hi Alice,")
