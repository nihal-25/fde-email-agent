"""Tests for the terminal-state guard in record_human_decision and the
Slack-coordinate persistence helper. Uses a throwaway SQLite file so no
Postgres/docker is required.

Run: .venv/bin/python -m unittest tests.test_decision_guard
"""

import os
import tempfile
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db
from app.db import Draft, Email, DRAFT_DRAFTED, DRAFT_SENT, DecisionConflict


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

    def _make_draft(self, status=DRAFT_DRAFTED) -> int:
        with db.get_session() as s:
            e = Email(thread_id="t1", subject="s", source="cli", raw_thread={})
            s.add(e)
            s.flush()
            d = Draft(email_id=e.id, classification={}, draft_text="hi", status=status)
            s.add(d)
            s.flush()
            did = d.id
            s.commit()
        return did


class TestDecisionGuard(_SqliteBacked):
    def test_normal_approve_from_drafted_ok(self):
        did = self._make_draft()
        res = db.record_human_decision(did, "approve", actor="t")
        self.assertEqual(res["status"], "approved")

    def test_rejected_cannot_be_reapproved(self):
        """The priority case: a stale Approve on a rejected draft must not fire."""
        did = self._make_draft()
        db.record_human_decision(did, "reject", actor="t")
        with self.assertRaises(DecisionConflict) as ctx:
            db.record_human_decision(did, "approve", actor="t")
        self.assertEqual(ctx.exception.current_status, "rejected")
        self.assertEqual(ctx.exception.attempted, "approve")

    def test_double_reject_blocked(self):
        did = self._make_draft()
        db.record_human_decision(did, "reject", actor="t")
        with self.assertRaises(DecisionConflict):
            db.record_human_decision(did, "reject", actor="t")

    def test_sent_cannot_be_reapproved(self):
        """Re-approving a sent draft would double-send to the customer."""
        did = self._make_draft(status=DRAFT_SENT)
        with self.assertRaises(DecisionConflict):
            db.record_human_decision(did, "approve", actor="t")

    def test_rejected_cannot_be_edited(self):
        did = self._make_draft()
        db.record_human_decision(did, "reject", actor="t")
        with self.assertRaises(DecisionConflict):
            db.record_human_decision(did, "edit", actor="t", edited_text="x")


class TestSlackCoordinates(_SqliteBacked):
    def test_set_slack_message_persists(self):
        did = self._make_draft()
        db.set_slack_message(did, "C123", "1700000000.000100")
        with db.get_session() as s:
            d = s.get(Draft, did)
            self.assertEqual(d.slack_channel, "C123")
            self.assertEqual(d.slack_ts, "1700000000.000100")

    def test_set_slack_message_noop_without_ts(self):
        did = self._make_draft()
        db.set_slack_message(did, "C123", None)  # must not raise
        with db.get_session() as s:
            self.assertIsNone(s.get(Draft, did).slack_ts)


if __name__ == "__main__":
    unittest.main()
