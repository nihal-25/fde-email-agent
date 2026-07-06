"""Fetch-side reconciliation (silent-loss slice, path 1).

Proves BOTH branches of the transient-vs-permanent split in
worker._process_since_cursor:

  TRANSIENT  -> cursor HELD, window re-fetched next round, and the re-processed
                successes do NOT duplicate (idempotency_key anchor collapses the
                replay) — no-double-send.
  PERMANENT  -> 404 AND unknown errors SKIP + ADVANCE the cursor; the worker
                moves on and does NOT wedge — a permanent error mistaken for
                transient would stall the whole pipeline, which is worse than
                losing the one message.

Gmail + process_thread are mocked; the cursor is isolated in a local store; the
idempotency anchor runs against the real dev DB (then cleaned up).
"""
import sys
import unittest
from unittest import mock

sys.path.insert(0, "/Users/nihal.manjunath/fde-email-agent")
from dotenv import load_dotenv
load_dotenv("/Users/nihal.manjunath/fde-email-agent/.env")

from googleapiclient.errors import HttpError  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import db, worker as wk  # noqa: E402

_TEST_THREADS = ("T-mA", "T-mB", "T-mC", "T-mD")


class _Resp(dict):
    """Minimal httplib2-style response for building an HttpError with a status."""
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.reason = "err"


def _msg(mid):
    return {"threadId": f"T-{mid}",
            "payload": {"headers": [{"name": "From", "value": "cust@example.com"}]},
            "labelIds": []}


def _clean():
    with db.get_engine().begin() as c:
        for tid in _TEST_THREADS:
            c.execute(text("DELETE FROM audit_log WHERE thread_id=:t"), {"t": tid})
            c.execute(text("DELETE FROM drafts WHERE email_id IN "
                           "(SELECT id FROM emails WHERE thread_id=:t)"), {"t": tid})
            c.execute(text("DELETE FROM emails WHERE thread_id=:t"), {"t": tid})


class TestFetchReconcile(unittest.TestCase):
    def setUp(self):
        db.init_db()
        _clean()
        self.store = {}
        mock.patch.object(db, "kv_get", lambda k, default=None: self.store.get(k, default)).start()
        mock.patch.object(db, "kv_set", lambda k, v: self.store.__setitem__(k, v)).start()
        mock.patch.object(wk.gmail_client, "is_automated", lambda *a, **k: (False, "")).start()

        # process_thread -> a draft keyed on the thread (STABLE across a replay,
        # so re-processing the same window dedups via the anchor).
        def fake_process(thread_id, my_email, service):
            db.persist_processing(
                {"thread_id": thread_id,
                 "messages": [{"id": thread_id, "body": "b", "from": "c@x", "date": "d"}]},
                {"intent": "technical_support"}, "draft", source="reconcile-test")
        mock.patch.object(wk, "process_thread", fake_process).start()

    def tearDown(self):
        mock.patch.stopall()
        _clean()

    def _count(self, tid):
        from app.db import Draft, Email, get_session
        s = get_session()
        try:
            return s.query(Draft).join(Email, Draft.email_id == Email.id).filter(
                Email.thread_id == tid).count()
        finally:
            s.close()  # release the pooled connection (else the pool exhausts + blocks)

    def _cursor(self):
        return self.store[wk.gmail_client.KV_LAST_HISTORY]["history_id"]

    # --- BRANCH 1: transient -------------------------------------------------
    def test_transient_holds_cursor_refetches_and_no_double_send(self):
        self.store[wk.gmail_client.KV_LAST_HISTORY] = {"history_id": "H1"}
        calls = {"mB": 0}

        def get_msg(mid, **kw):
            if mid == "mB":
                calls["mB"] += 1
                if calls["mB"] == 1:
                    raise BrokenPipeError(32, "broken pipe")  # transient
            return _msg(mid)

        with mock.patch.object(wk.gmail_client, "list_new_message_ids",
                               lambda h, service=None: (["mA", "mB"], "H2")), \
             mock.patch.object(wk.gmail_client, "get_message", get_msg):
            # Round 1: mA ok, mB transient -> cursor HELD at H1.
            wk._process_since_cursor({"historyId": "H2"}, service=None, my_email="me")
            self.assertEqual(self._cursor(), "H1", "cursor must be HELD on transient")
            self.assertEqual(calls["mB"], 1)
            # Round 2: same window re-fetched; mB now ok -> cursor ADVANCES.
            wk._process_since_cursor({"historyId": "H2"}, service=None, my_email="me")
            self.assertEqual(self._cursor(), "H2", "cursor must ADVANCE once clean")
            self.assertEqual(calls["mB"], 2, "mB must be RE-FETCHED next round")

        # T-mA was processed in BOTH rounds; the anchor collapses it to ONE draft.
        self.assertEqual(self._count("T-mA"), 1, "no double-send: replay must dedup")
        self.assertEqual(self._count("T-mB"), 1)

    # --- BRANCH 2: permanent (404) — advance, no wedge -----------------------
    def test_permanent_404_advances_and_does_not_wedge(self):
        self.store[wk.gmail_client.KV_LAST_HISTORY] = {"history_id": "H2"}

        def get_msg(mid, **kw):
            if mid == "mC":
                raise HttpError(_Resp(404), b"not found")
            return _msg(mid)

        with mock.patch.object(wk.gmail_client, "list_new_message_ids",
                               lambda h, service=None: (["mC"], "H3")), \
             mock.patch.object(wk.gmail_client, "get_message", get_msg):
            wk._process_since_cursor({"historyId": "H3"}, service=None, my_email="me")  # must not raise
        self.assertEqual(self._cursor(), "H3", "404 must ADVANCE the cursor (no wedge)")

    # --- BRANCH 2b: unknown/permanent error — advance, no wedge --------------
    def test_unknown_permanent_advances_and_does_not_wedge(self):
        self.store[wk.gmail_client.KV_LAST_HISTORY] = {"history_id": "H2"}

        def get_msg(mid, **kw):
            if mid == "mD":
                raise ValueError("weird permanent parse error")  # NOT a known transient
            return _msg(mid)

        with mock.patch.object(wk.gmail_client, "list_new_message_ids",
                               lambda h, service=None: (["mD"], "H3")), \
             mock.patch.object(wk.gmail_client, "get_message", get_msg):
            wk._process_since_cursor({"historyId": "H3"}, service=None, my_email="me")
        self.assertEqual(self._cursor(), "H3",
                         "unknown error must be treated PERMANENT -> advance, never wedge")


if __name__ == "__main__":
    unittest.main(verbosity=2)
