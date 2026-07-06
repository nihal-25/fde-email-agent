"""Orphaned-draft sweeper (silent-loss slice, path 2), on the race-proven claim.

HEADLINE (the real production race): a draft with no slack_ts, the normal post
path AND the sweeper posting CONCURRENTLY -> both call claim_draft_for_posting,
exactly one wins and posts (one chat_postMessage), the other stands down.

Also:
- a failed post RELEASES the claim so the next attempt re-claims promptly
  (doesn't burn the full lease);
- the sweeper query is grace-perioded: a fresh draft is NOT swept, an old
  unposted one IS.

post_draft_for_approval (the real chat_postMessage) is mocked to count posts;
the claim + drafts run against the real dev DB (then cleaned up).
"""
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, "/Users/nihal.manjunath/fde-email-agent")
from dotenv import load_dotenv
load_dotenv("/Users/nihal.manjunath/fde-email-agent/.env")

from sqlalchemy import text  # noqa: E402
from app import db, slack_approval  # noqa: E402

_TIDS = ("T-sw1", "T-sw2", "T-sw3", "T-sw4")


def _mk_unposted(tid, mid):
    return db.persist_processing(
        {"thread_id": tid, "subject": "s",
         "messages": [{"id": mid, "body": "b", "from": "c@x", "date": "d"}]},
        {"intent": "technical_support"}, "draft body", source="sweeptest")["draft_id"]


def _slack_ts(did):
    from app.db import Draft, get_session
    s = get_session()
    try:
        return s.get(Draft, did).slack_ts
    finally:
        s.close()


def _clean():
    with db.get_engine().begin() as c:
        for tid in _TIDS:
            c.execute(text("DELETE FROM audit_log WHERE thread_id=:t"), {"t": tid})
            c.execute(text("DELETE FROM drafts WHERE email_id IN "
                           "(SELECT id FROM emails WHERE thread_id=:t)"), {"t": tid})
            c.execute(text("DELETE FROM emails WHERE thread_id=:t"), {"t": tid})


class TestSweeperReconcile(unittest.TestCase):
    def setUp(self):
        db.init_db()
        _clean()

    def tearDown(self):
        _clean()

    # --- HEADLINE: concurrent normal-vs-sweeper -> exactly one card ----------
    def test_concurrent_normal_and_sweeper_post_exactly_one_card(self):
        did = _mk_unposted("T-sw1", "M1")
        posts, lk = [], threading.Lock()

        def fake_post_for_approval(original, draft_text, draft_id, **kw):
            with lk:
                posts.append(draft_id)          # count real chat_postMessage attempts
            return {"channel": "C0", "ts": f"ts-{len(posts)}"}

        results = {}
        barrier = threading.Barrier(2)

        def run(label):
            barrier.wait()                       # both fire at once
            results[label] = slack_approval.post_draft_once("orig", "draft", did)

        with mock.patch.object(slack_approval, "post_draft_for_approval", fake_post_for_approval):
            t1 = threading.Thread(target=run, args=("normal",))
            t2 = threading.Thread(target=run, args=("sweeper",))
            t1.start(); t2.start(); t1.join(); t2.join()

        self.assertEqual(len(posts), 1, "exactly ONE chat_postMessage must land")
        winners = [v for v in results.values() if v is not None]
        self.assertEqual(len(winners), 1, "exactly one poster wins; the other stands down (None)")
        self.assertIsNotNone(_slack_ts(did), "the winner recorded slack_ts")

    # --- failed post releases the claim for a prompt retry -------------------
    def test_post_failure_releases_claim_for_prompt_retry(self):
        did = _mk_unposted("T-sw2", "M2")

        def boom(*a, **k):
            raise RuntimeError("slack down")

        with mock.patch.object(slack_approval, "post_draft_for_approval", boom):
            with self.assertRaises(RuntimeError):
                slack_approval.post_draft_once("orig", "draft", did)
        # Claim released -> immediately re-claimable (not waiting out the lease).
        self.assertTrue(db.claim_draft_for_posting(did),
                        "a failed post must release the claim for a prompt retry")

    # --- grace period: fresh excluded, old included -------------------------
    def test_grace_period_excludes_fresh_includes_old(self):
        fresh = _mk_unposted("T-sw3", "M3")
        old = _mk_unposted("T-sw4", "M4")
        with db.get_engine().begin() as c:
            c.execute(text("UPDATE drafts SET created_at = now() - make_interval(secs => 300) WHERE id=:i"),
                      {"i": old})
        found = {r["draft_id"] for r in db.find_unposted_drafts(grace_seconds=120, limit=1000)}
        self.assertIn(old, found, "an old unposted draft IS swept")
        self.assertNotIn(fresh, found, "a fresh draft within the grace window is NOT swept")


if __name__ == "__main__":
    unittest.main(verbosity=2)
