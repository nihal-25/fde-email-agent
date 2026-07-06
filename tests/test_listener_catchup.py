"""Listener reconnect catch-up (silent-loss slice, path 3).

HEADLINE — the COLLISION: a pending case's #verify reply is picked up by the
reconnect catch-up AND the live event fires for the SAME case. Both drive
resume() concurrently -> the atomic claim-and-clear means resume runs EXACTLY
ONCE: one final #debugging post, one approval card, one draft; the second
attempt no-ops.

Also: the catch-up scan is BOUNDED (only recent cases, capped) so a reconnect
storm can't become an API-rate storm.

LLM + Slack posting are mocked; the atomic claim + drafts run against the real
dev DB (then cleaned up).
"""
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, "/Users/nihal.manjunath/fde-email-agent")
from dotenv import load_dotenv
load_dotenv("/Users/nihal.manjunath/fde-email-agent/.env")

from sqlalchemy import text  # noqa: E402
from app import db, slack_approval, debug_orchestrator as orch  # noqa: E402

_TS = "collide-999.001"
_TID = "T-collide"
_BOUNDS_KEYS = [f"{orch._PENDING_PREFIX}bound-{i}" for i in range(5)]


def _clean():
    with db.get_engine().begin() as c:
        c.execute(text("DELETE FROM kv_state WHERE key = :k"), {"k": orch._PENDING_PREFIX + _TS})
        for k in _BOUNDS_KEYS:
            c.execute(text("DELETE FROM kv_state WHERE key = :k"), {"k": k})
        c.execute(text("DELETE FROM audit_log WHERE thread_id = :t"), {"t": _TID})
        c.execute(text("DELETE FROM drafts WHERE email_id IN "
                       "(SELECT id FROM emails WHERE thread_id = :t)"), {"t": _TID})
        c.execute(text("DELETE FROM emails WHERE thread_id = :t"), {"t": _TID})


def _count(tid):
    from app.db import Draft, Email, get_session
    s = get_session()
    try:
        return s.query(Draft).join(Email, Draft.email_id == Email.id).filter(
            Email.thread_id == tid).count()
    finally:
        s.close()


class TestListenerCatchup(unittest.TestCase):
    def setUp(self):
        db.init_db()
        _clean()

    def tearDown(self):
        _clean()

    # --- HEADLINE: catch-up + live event collide -> resume runs exactly once --
    def test_collision_resume_runs_exactly_once(self):
        db.kv_set(orch._PENDING_PREFIX + _TS, {
            "account_id": 10000001, "uuid": "11111111-1111-4111-8111-111111111111",
            "facts": "F", "interpretation": "I", "case_title": "Collision case",
            "email_text": "Hi", "sender": "c@x",
            "thread": {"thread_id": _TID,
                       "messages": [{"id": "MCOL", "body": "b", "from": "c@x", "date": "d"}]}})

        dbg_posts, cards, lk = [], [], threading.Lock()

        def fake_post(channel, textmsg, thread_ts=None):     # orch._post (final #debugging)
            with lk:
                dbg_posts.append(channel)
            return {"ts": "1"}

        def fake_card(original, draft_text, draft_id, **kw):  # post_draft_for_approval
            with lk:
                cards.append(draft_id)
            return {"channel": "C0", "ts": "card1"}

        results = {}
        barrier = threading.Barrier(2)

        def run(label):
            barrier.wait()   # catch-up's resume and the live event's resume fire together
            results[label] = orch.resume(_TS, "the reviewer: US-region trunk -> India anchoring 403")

        with mock.patch.object(orch, "_post", fake_post), \
             mock.patch.object(orch.llm, "debug_final_findings",
                               lambda *a, **k: {"final_interpretation": "confirmed",
                                                "customer_safe_explanation": "US-based region; use India trunk"}), \
             mock.patch.object(orch.llm, "debug_customer_draft", lambda *a, **k: "draft body"), \
             mock.patch.object(slack_approval, "post_draft_for_approval", fake_card):
            t1 = threading.Thread(target=run, args=("catchup",))
            t2 = threading.Thread(target=run, args=("live",))
            t1.start(); t2.start(); t1.join(); t2.join()

        winners = [v for v in results.values() if v is not None]
        self.assertEqual(len(winners), 1, "resume must run EXACTLY ONCE (one winner, one no-op)")
        self.assertEqual(len(dbg_posts), 1, "exactly one final #debugging post")
        self.assertEqual(len(cards), 1, "exactly one approval card")
        self.assertEqual(_count(_TID), 1, "exactly one draft")
        self.assertIsNone(db.kv_get(orch._PENDING_PREFIX + _TS), "pending case is cleared")

    # --- bounded scan: only recent cases, capped ----------------------------
    def test_catch_up_scan_is_bounded(self):
        for k in _BOUNDS_KEYS:
            db.kv_set(k, {"case": k})
        # Backdate two of them beyond a 1-hour window.
        with db.get_engine().begin() as c:
            for k in _BOUNDS_KEYS[:2]:
                c.execute(text("UPDATE kv_state SET updated_at = now() - make_interval(secs => 7200) "
                               "WHERE key = :k"), {"k": k})
        keys = db.list_pending_case_keys(orch._PENDING_PREFIX, max_age_seconds=3600, max_cases=2)
        self.assertLessEqual(len(keys), 2, "cap (max_cases) is enforced")
        for k in keys:
            self.assertNotIn(k, _BOUNDS_KEYS[:2], "cases older than max_age are excluded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
