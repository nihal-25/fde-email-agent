"""Tests for the oversized-thread guard: truncate to recent messages and
always produce a postable draft (manual-review fallback) so an over-limit
thread can't vanish with only a log line.

Run: .venv/bin/python -m unittest tests.test_oversized_thread
"""

import unittest
from unittest import mock

from app import llm, worker


class TestSelectRecentMessages(unittest.TestCase):
    def _msgs(self, n, body_len=100):
        return [{"from": f"a{i}@x.com", "date": "d", "body": "x" * body_len} for i in range(n)]

    def test_keeps_most_recent_and_drops_oldest(self):
        msgs = self._msgs(10, body_len=100)  # ~164 chars each
        kept, dropped = llm.select_recent_messages(msgs, char_budget=500)
        self.assertEqual(len(kept) + dropped, 10)
        self.assertLess(len(kept), 10)
        # The survivors are the newest ones, still in oldest-first order.
        self.assertEqual(kept[-1], msgs[-1])
        self.assertEqual([m["from"] for m in kept], [m["from"] for m in msgs[-len(kept):]])

    def test_always_keeps_at_least_latest(self):
        msgs = self._msgs(3, body_len=10_000)  # each far exceeds the budget
        kept, dropped = llm.select_recent_messages(msgs, char_budget=100)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], msgs[-1])
        self.assertEqual(dropped, 2)

    def test_no_truncation_when_within_budget(self):
        msgs = self._msgs(4, body_len=10)
        kept, dropped = llm.select_recent_messages(msgs, char_budget=10_000)
        self.assertEqual(dropped, 0)
        self.assertEqual(kept, msgs)

    def test_estimate_tokens_scales_with_length(self):
        self.assertGreater(llm.estimate_tokens("x" * 4000), llm.estimate_tokens("x" * 40))


class TestClassifyAndDraftGuard(unittest.TestCase):
    def _thread(self, n, body_len):
        return {"subject": "S", "messages":
                [{"from": f"a{i}@x.com", "date": "d", "body": "x" * body_len} for i in range(n)]}

    def test_oversized_thread_is_truncated_before_drafting(self):
        thread = self._thread(20, body_len=500)
        captured = {}

        def fake_classify(text):
            captured["text"] = text
            return {"intent": "general_inquiry"}

        with mock.patch.object(worker.llm, "classify", fake_classify), \
             mock.patch.object(worker.llm, "draft_reply", lambda t, i: "BODY"), \
             mock.patch.object(worker.draft_mod, "flag_unverified_specifics", lambda d: []), \
             mock.patch.object(worker.llm, "THREAD_TOKEN_BUDGET", 50), \
             mock.patch.object(worker.llm, "THREAD_CHAR_BUDGET", 1500):
            text, cls, draft, flags = worker._classify_and_draft(thread)

        # The model saw fewer than all 20 messages...
        self.assertLess(captured["text"].count("From:"), 20)
        # ...the draft carries the reviewer note, and a truncation flag is set.
        self.assertIn("most recent", draft)
        self.assertTrue(any(f["type"] == "thread_truncated" for f in flags))

    def test_draft_failure_yields_postable_manual_review_fallback(self):
        """If drafting still fails (e.g. context_length even after truncation),
        a manual-review card must still be produced — nothing vanishes."""
        thread = self._thread(1, body_len=20)

        def boom(*a, **k):
            raise RuntimeError("context_length_exceeded")

        with mock.patch.object(worker.llm, "classify", boom), \
             mock.patch.object(worker.llm, "draft_reply", boom):
            text, cls, draft, flags = worker._classify_and_draft(thread)

        self.assertEqual(cls["intent"], "needs_manual_review")
        self.assertIn("manual", draft.lower())
        self.assertTrue(any(f["type"] == "auto_draft_failed" for f in flags))
        self.assertTrue(text)  # there is still something to show in Slack

    def test_normal_thread_unchanged(self):
        thread = self._thread(2, body_len=20)
        with mock.patch.object(worker.llm, "classify", lambda t: {"intent": "pricing_question"}), \
             mock.patch.object(worker.llm, "draft_reply", lambda t, i: "Hi there"), \
             mock.patch.object(worker.draft_mod, "flag_unverified_specifics", lambda d: []):
            text, cls, draft, flags = worker._classify_and_draft(thread)
        self.assertEqual(draft, "Hi there")  # no reviewer note prepended
        self.assertEqual(flags, [])


if __name__ == "__main__":
    unittest.main()
