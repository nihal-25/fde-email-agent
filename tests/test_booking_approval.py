"""Tests for the booking-on-approval action (slack_approval._book_and_send).

Verifies the order and the safety rule: calendar event + invite is created
BEFORE the confirmation email, and if booking fails the email is NOT sent.
Uses a SQLite-backed db and fake gmail/calendar modules — no network.

Run: .venv/bin/python -m unittest tests.test_booking_approval
"""

import os
import tempfile
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db, slack_approval
from app.db import Draft


class FakeGmail:
    def __init__(self):
        self.sent = []

    def send_reply(self, reply_context, body):
        self.sent.append((reply_context, body))
        return {"id": "gmail-1"}


class FakeCalendar:
    def __init__(self, fail=False, log=None):
        self.created = []
        self.updated = None
        self.fail = fail
        self._log = log if log is not None else []

    def create_event(self, summary, start, end, attendee_email, *, description=None):
        self._log.append("create_event")
        if self.fail:
            raise RuntimeError("calendar 500")
        self.created.append({"summary": summary, "attendee": attendee_email})
        return {"event_id": "evt-1", "html_link": "https://cal/evt-1",
                "meet_link": "https://meet.google.com/abc-defg-hij"}

    def update_event(self, event_id, start, end):
        self._log.append("update_event")
        self.updated = event_id
        return {"event_id": event_id, "html_link": "https://cal/" + event_id,
                "meet_link": "https://meet.google.com/abc-defg-hij"}


class _SqliteBacked(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        db._engine = create_engine(f"sqlite:///{self._tmp.name}", future=True)
        db._SessionFactory = sessionmaker(bind=db._engine, future=True)
        db.Base.metadata.create_all(db._engine)

    def tearDown(self):
        db._engine.dispose()
        db._engine = None
        db._SessionFactory = None
        os.unlink(self._tmp.name)

    def _booking_draft(self):
        booking = {"start": "2026-06-22T15:00:00+05:30", "end": "2026-06-22T15:30:00+05:30",
                   "duration_min": 30, "attendee_email": "me@myown.com",
                   "title": "Plivo <> Acme", "timezone": "Asia/Kolkata",
                   "label": "Mon 22 Jun, 3:00-3:30 pm IST"}
        thread = {"thread_id": "gt-1", "subject": "Call?",
                  "messages": [{"from": "me@myown.com", "date": "d", "body": "4pm?"}]}
        ids = db.persist_processing(thread, {"intent": "meeting_request"}, "I'll send an invite.",
                                    source="gmail", reply_context={"to": "me@myown.com", "subject": "Re: Call?"},
                                    booking=booking)
        return ids["draft_id"]


class TestBookAndSend(_SqliteBacked):
    def test_booking_created_before_email_and_marks_sent(self):
        did = self._booking_draft()
        order = []
        gmail = FakeGmail()
        cal = FakeCalendar(log=order)
        # wrap send so we can see ordering relative to create_event
        orig_send = gmail.send_reply
        gmail.send_reply = lambda rc, b: (order.append("send_reply"), orig_send(rc, b))[1]

        ctx = db.get_send_context(did)
        note = slack_approval._book_and_send(did, "I'll send an invite.", "gt-1", ctx,
                                             gmail=gmail, calendar=cal)

        self.assertEqual(order, ["create_event", "send_reply"])  # booking first
        self.assertEqual(len(cal.created), 1)
        self.assertEqual(len(gmail.sent), 1)
        self.assertIn("Booked", note)
        self.assertIn("Meet: https://meet.google.com/abc-defg-hij", note)
        self.assertIn("📤 Sent", note)
        self.assertEqual(db.get_send_context(did)["status"], "sent")

    def test_booking_failure_aborts_email(self):
        did = self._booking_draft()
        gmail = FakeGmail()
        cal = FakeCalendar(fail=True)

        ctx = db.get_send_context(did)
        note = slack_approval._book_and_send(did, "I'll send an invite.", "gt-1", ctx,
                                             gmail=gmail, calendar=cal)

        self.assertEqual(gmail.sent, [])  # email NOT sent
        self.assertIn("BOOKING FAILED", note)
        self.assertNotEqual(db.get_send_context(did)["status"], "sent")  # not marked sent

    def test_reschedule_updates_existing_event_not_create(self):
        # A booking carrying an event_id must UPDATE that event, not create one.
        booking = {"start": "2026-06-22T17:00:00+05:30", "end": "2026-06-22T17:30:00+05:30",
                   "duration_min": 30, "attendee_email": "me@myown.com", "title": "Plivo <> Acme",
                   "timezone": "Asia/Kolkata", "label": "Mon 22 Jun, 5:00-5:30 pm IST",
                   "event_id": "evt-existing"}
        thread = {"thread_id": "gt-1", "subject": "Call?",
                  "messages": [{"from": "me@myown.com", "date": "d", "body": "move to 5"}]}
        ids = db.persist_processing(thread, {"intent": "meeting_request"}, "I'll move the invite.",
                                    source="gmail", reply_context={"to": "me@myown.com", "subject": "Re: Call?"},
                                    booking=booking)
        gmail, cal = FakeGmail(), FakeCalendar()
        ctx = db.get_send_context(ids["draft_id"])
        note = slack_approval._book_and_send(ids["draft_id"], "I'll move the invite.", "gt-1", ctx,
                                             gmail=gmail, calendar=cal)
        self.assertEqual(cal.created, [])          # no new event
        self.assertEqual(cal.updated, "evt-existing")  # moved the existing one
        self.assertIn("Rescheduled", note)
        self.assertEqual(len(gmail.sent), 1)
        # The thread now remembers the (same) event for any further change.
        self.assertEqual(db.get_scheduling_state("gt-1")["booked_event_id"], "evt-existing")

    def test_non_booking_draft_just_sends(self):
        thread = {"thread_id": "gt-2", "subject": "Pricing",
                  "messages": [{"from": "p@acme.io", "date": "d", "body": "rates?"}]}
        ids = db.persist_processing(thread, {"intent": "pricing_question"}, "Here are the rates.",
                                    source="gmail", reply_context={"to": "p@acme.io", "subject": "Re: Pricing"})
        gmail = FakeGmail()
        cal = FakeCalendar()
        ctx = db.get_send_context(ids["draft_id"])
        note = slack_approval._book_and_send(ids["draft_id"], "Here are the rates.", "gt-2", ctx,
                                             gmail=gmail, calendar=cal)
        self.assertEqual(cal.created, [])      # no calendar action
        self.assertEqual(len(gmail.sent), 1)
        self.assertIn("📤 Sent", note)


class TestBookingCard(unittest.TestCase):
    def test_build_blocks_shows_booking_line(self):
        booking = {"label": "Mon 22 Jun, 3:00-3:30 pm IST", "attendee_email": "p@acme.io"}
        blocks = slack_approval.build_blocks("orig", "reply text", 7, booking=booking)
        text = " ".join(b.get("text", {}).get("text", "") for b in blocks if b.get("type") == "section")
        self.assertIn("Will book on approval", text)
        self.assertIn("Mon 22 Jun, 3:00-3:30 pm IST", text)
        self.assertIn("p@acme.io", text)


if __name__ == "__main__":
    unittest.main()
