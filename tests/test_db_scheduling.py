"""Tests for the scheduling-related DB additions: SchedulingState get/set and
the Draft.booking column flowing through persist_processing -> get_send_context.

Run: .venv/bin/python -m unittest tests.test_db_scheduling
"""

import os
import tempfile
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db


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


class TestSchedulingState(_SqliteBacked):
    def test_none_before_set(self):
        self.assertIsNone(db.get_scheduling_state("t-thread"))

    def test_roundtrip_and_update(self):
        db.set_scheduling_state("t-thread", {"stage": "proposed", "proposed_slots": ["2026-06-22T15:00:00+05:30"]})
        got = db.get_scheduling_state("t-thread")
        self.assertEqual(got["stage"], "proposed")

        db.set_scheduling_state("t-thread", {"stage": "agreed", "proposed_slots": []})
        self.assertEqual(db.get_scheduling_state("t-thread")["stage"], "agreed")


class TestBookingColumn(_SqliteBacked):
    def _thread(self):
        return {"thread_id": "gt-1", "subject": "Call?", "messages": [{"from": "p@acme.io", "date": "d", "body": "let's talk"}]}

    def test_booking_persists_and_surfaces_in_send_context(self):
        booking = {"start": "2026-06-22T15:00:00+05:30", "end": "2026-06-22T15:30:00+05:30",
                   "duration_min": 30, "attendee_email": "p@acme.io",
                   "title": "Plivo <> Acme", "timezone": "Asia/Kolkata"}
        ids = db.persist_processing(self._thread(), {"intent": "meeting_request"}, "I'll send an invite.",
                                    source="gmail", reply_context={"to": "p@acme.io"}, booking=booking)
        ctx = db.get_send_context(ids["draft_id"])
        self.assertEqual(ctx["booking"]["attendee_email"], "p@acme.io")
        self.assertEqual(ctx["booking"]["duration_min"], 30)

    def test_non_booking_draft_has_null_booking(self):
        ids = db.persist_processing(self._thread(), {"intent": "pricing_question"}, "Here are the rates.",
                                    source="gmail", reply_context={"to": "p@acme.io"})
        self.assertIsNone(db.get_send_context(ids["draft_id"])["booking"])


if __name__ == "__main__":
    unittest.main()
