"""Decision-matrix tests for app/scheduling.handle — LLM + calendar mocked.

Covers each branch: no-time, named-free, named-busy, flexible, past-time guard,
continuation (agree to a proposed slot), and the no-attendee safety case.

Run: .venv/bin/python -m unittest tests.test_scheduling
"""

import unittest
from datetime import datetime
from unittest import mock

from app import scheduling
from app.calendar_client import IST


NOW = datetime(2026, 6, 22, 10, 0, tzinfo=IST)  # Monday 10:00 IST


def thread(to="priya@acme.io"):
    return {
        "thread_id": "gt-1",
        "subject": "Quick call?",
        "messages": [{"from": to, "date": "Mon, 22 Jun 2026", "body": "Can we talk?"}],
        "reply_context": {"to": to, "subject": "Re: Quick call?"},
    }


CLASSIFICATION = {"intent": "meeting_request", "customer_name": "Priya", "company": "Acme"}


class SchedulingHarness(unittest.TestCase):
    """Patches the LLM (extract + draft) and the calendar reads for each test."""

    def run_handle(self, ext, *, is_free=True, free_slots=None, prior_state=None, th=None):
        self.draft_calls = []

        def fake_draft(action, *, thread_text, times=None, customer_name=None):
            self.draft_calls.append({"action": action, "times": times or []})
            return f"DRAFT:{action}"

        with mock.patch.object(scheduling.llm, "extract_scheduling", lambda *a, **k: ext), \
             mock.patch.object(scheduling.llm, "draft_meeting_reply", fake_draft), \
             mock.patch.object(scheduling.cal, "is_free", lambda *a, **k: is_free), \
             mock.patch.object(scheduling.cal, "find_free_slots", lambda *a, **k: free_slots or []):
            return scheduling.handle(th or thread(), CLASSIFICATION, prior_state, NOW)

    def action(self):
        return self.draft_calls[-1]["action"]


class TestDecisionMatrix(SchedulingHarness):
    def test_no_time_asks_without_proposing(self):
        r = self.run_handle({"has_time": False, "open_to_any": False})
        self.assertEqual(self.action(), "ask_time")
        self.assertIsNone(r.booking)
        self.assertEqual(r.state["stage"], scheduling.STAGE_AWAITING_TIME)

    def test_named_time_free_confirms_and_books(self):
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T15:00:00+05:30"},
            is_free=True)
        self.assertEqual(self.action(), "confirm")
        self.assertIsNotNone(r.booking)
        self.assertEqual(r.booking["start"], "2026-06-22T15:00:00+05:30")
        self.assertEqual(r.booking["duration_min"], 30)
        self.assertEqual(r.booking["attendee_email"], "priya@acme.io")
        self.assertEqual(r.booking["title"], "Plivo <> Acme")
        self.assertEqual(r.state["stage"], scheduling.STAGE_CONFIRMING)
        # The reviewer-facing time label is passed to the draft verbatim.
        self.assertIn("3:00-3:30 pm IST", self.draft_calls[-1]["times"][0])

    def test_out_of_hours_free_time_proposes_instead_of_booking(self):
        # 20:00 is free on the calendar but outside 10:30-18:30 working hours ->
        # must be treated like busy: propose a nearby in-hours slot, no booking,
        # and free/busy must NOT even be consulted (within-hours fails first).
        nearby = [(datetime(2026, 6, 23, 11, 0, tzinfo=IST), datetime(2026, 6, 23, 11, 30, tzinfo=IST))]

        def boom(*a, **k):
            raise AssertionError("is_free must not be called for an out-of-hours time")

        self.draft_calls = []

        def fake_draft(action, *, thread_text, times=None, customer_name=None):
            self.draft_calls.append({"action": action, "times": times or []})
            return f"DRAFT:{action}"

        with mock.patch.object(scheduling.llm, "extract_scheduling",
                               lambda *a, **k: {"has_time": True, "requested_start_ist": "2026-06-22T20:00:00+05:30"}), \
             mock.patch.object(scheduling.llm, "draft_meeting_reply", fake_draft), \
             mock.patch.object(scheduling.cal, "is_free", boom), \
             mock.patch.object(scheduling.cal, "find_free_slots", lambda *a, **k: nearby):
            r = scheduling.handle(thread(), CLASSIFICATION, None, NOW)

        self.assertEqual(self.action(), "propose_nearby")
        self.assertIsNone(r.booking)
        self.assertEqual(r.state["stage"], scheduling.STAGE_PROPOSED)

    def test_named_time_busy_proposes_nearby(self):
        nearby = [(datetime(2026, 6, 22, 16, 0, tzinfo=IST), datetime(2026, 6, 22, 16, 30, tzinfo=IST))]
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T15:00:00+05:30"},
            is_free=False, free_slots=nearby)
        self.assertEqual(self.action(), "propose_nearby")
        self.assertIsNone(r.booking)
        self.assertEqual(r.state["stage"], scheduling.STAGE_PROPOSED)
        self.assertEqual(r.state["proposed_slots"], ["2026-06-22T16:00:00+05:30"])

    def test_flexible_proposes_slots(self):
        slots = [(datetime(2026, 6, 22, 11, 0, tzinfo=IST), datetime(2026, 6, 22, 11, 30, tzinfo=IST)),
                 (datetime(2026, 6, 22, 11, 30, tzinfo=IST), datetime(2026, 6, 22, 12, 0, tzinfo=IST))]
        r = self.run_handle({"open_to_any": True}, free_slots=slots)
        self.assertEqual(self.action(), "propose_slots")
        self.assertIsNone(r.booking)
        self.assertEqual(len(r.state["proposed_slots"]), 2)

    def test_past_time_clarifies_never_books(self):
        # 08:00 is before NOW (10:00) -> must not book, must not even check free/busy.
        def boom(*a, **k):
            raise AssertionError("is_free must not be called for a past time")

        with mock.patch.object(scheduling.llm, "extract_scheduling",
                               lambda *a, **k: {"has_time": True, "requested_start_ist": "2026-06-22T08:00:00+05:30"}), \
             mock.patch.object(scheduling.llm, "draft_meeting_reply", lambda action, **k: f"DRAFT:{action}"), \
             mock.patch.object(scheduling.cal, "is_free", boom):
            r = scheduling.handle(thread(), CLASSIFICATION, None, NOW)
        self.assertEqual(r.draft_text, "DRAFT:clarify_time")
        self.assertIsNone(r.booking)
        self.assertEqual(r.state["stage"], scheduling.STAGE_CLARIFY)

    def test_continuation_agree_books_the_proposed_slot(self):
        prior = {"stage": scheduling.STAGE_PROPOSED, "duration_min": 30,
                 "attendee_email": "priya@acme.io",
                 "proposed_slots": ["2026-06-22T16:00:00+05:30"], "candidate_slot": None}
        r = self.run_handle(
            {"has_time": True, "agrees_to_prior": True, "requested_start_ist": "2026-06-22T16:00:00+05:30"},
            is_free=True, prior_state=prior)
        self.assertEqual(self.action(), "confirm")
        self.assertEqual(r.booking["start"], "2026-06-22T16:00:00+05:30")
        self.assertEqual(r.state["stage"], scheduling.STAGE_CONFIRMING)

    def test_reschedule_when_already_booked_updates_same_event(self):
        # A meeting is already booked on the thread -> a new agreed time must
        # carry the existing event_id (so it reschedules, not duplicates).
        prior = {"stage": "booked", "duration_min": 30, "timezone": "Asia/Kolkata",
                 "attendee_email": "priya@acme.io", "proposed_slots": [],
                 "candidate_slot": None, "booked_event_id": "evt-existing"}
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T17:00:00+05:30"},
            is_free=True, prior_state=prior)
        self.assertEqual(self.action(), "reschedule")
        self.assertEqual(r.booking["event_id"], "evt-existing")
        self.assertEqual(r.state["booked_event_id"], "evt-existing")

    def test_attendee_parsed_to_bare_email_for_calendar(self):
        # reply_context "to" carries a display name; the booking attendee (used
        # as the Calendar attendee) must be the bare address.
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T15:00:00+05:30"},
            is_free=True, th=thread(to="Sam Rivera <cust@example.com>"))
        self.assertEqual(r.booking["attendee_email"], "cust@example.com")

    def test_no_attendee_does_not_book(self):
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T15:00:00+05:30"},
            is_free=True, th=thread(to=None))
        self.assertEqual(self.action(), "confirm")
        self.assertIsNone(r.booking)  # cannot book without an attendee
        self.assertTrue(any(f["type"] == "no_attendee" for f in r.flags))


class TestExplicitTimezoneGuard(unittest.TestCase):
    def test_honors_explicit_zone(self):
        self.assertEqual(
            scheduling._explicit_tz("America/Los_Angeles", "4pm PST works for me"),
            "America/Los_Angeles")

    def test_drops_inferred_zone_when_no_token(self):
        # The model inferred a zone from headers, but the customer wrote none.
        self.assertIsNone(scheduling._explicit_tz("America/Los_Angeles", "I want a call tmr at 2pm"))

    def test_ignores_zone_word_in_quoted_history(self):
        # "IST" here is from our own quoted draft, not the customer's new text.
        body = "Actually can you move it to 3\n\nOn Sun, Nihal wrote:\n> invite for 2:30 pm IST"
        self.assertIsNone(scheduling._explicit_tz("America/Los_Angeles", body))

    def test_none_stays_none(self):
        self.assertIsNone(scheduling._explicit_tz(None, "4pm PST"))


class TestHandleTimezoneGuard(SchedulingHarness):
    def test_inferred_zone_ignored_end_to_end(self):
        # Extractor returns a zone, but the bare-time message has no zone word ->
        # the slot label must be IST-only (no PDT).
        th = thread(to="cust@example.com")
        th["messages"][-1]["body"] = "I want a call tmr at 2pm"
        r = self.run_handle(
            {"has_time": True, "requested_start_ist": "2026-06-22T14:00:00+05:30",
             "stated_timezone": "America/Los_Angeles"},
            is_free=True, th=th)
        self.assertEqual(self.action(), "confirm")
        self.assertNotIn("PDT", self.draft_calls[-1]["times"][0])
        self.assertIn("IST", self.draft_calls[-1]["times"][0])


class TestFormatSlot(unittest.TestCase):
    def test_same_meridiem(self):
        s = scheduling.format_slot(datetime(2026, 6, 25, 15, 0, tzinfo=IST),
                                   datetime(2026, 6, 25, 15, 30, tzinfo=IST))
        self.assertEqual(s, "Thu 25 Jun, 3:00-3:30 pm IST")

    def test_cross_meridiem(self):
        s = scheduling.format_slot(datetime(2026, 6, 25, 11, 45, tzinfo=IST),
                                   datetime(2026, 6, 25, 12, 15, tzinfo=IST))
        self.assertEqual(s, "Thu 25 Jun, 11:45 am-12:15 pm IST")

    def test_dual_zone_shows_customer_then_ist(self):
        # IST 15:00 -> Los Angeles 02:30 (PDT, same calendar date in June).
        s = scheduling.format_slot(datetime(2026, 6, 25, 15, 0, tzinfo=IST),
                                   datetime(2026, 6, 25, 15, 30, tzinfo=IST),
                                   "America/Los_Angeles")
        self.assertIn(" / ", s)
        self.assertTrue(s.endswith("IST"))
        self.assertIn("3:00-3:30 pm IST", s)   # IST stays the reference
        self.assertIn("PDT", s)                # customer zone (DST-correct) echoed

    def test_bad_zone_falls_back_to_ist_only(self):
        s = scheduling.format_slot(datetime(2026, 6, 25, 15, 0, tzinfo=IST),
                                   datetime(2026, 6, 25, 15, 30, tzinfo=IST), "Not/AZone")
        self.assertEqual(s, "Thu 25 Jun, 3:00-3:30 pm IST")


if __name__ == "__main__":
    unittest.main()
