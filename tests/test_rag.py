"""Routing tests for app/rag.answer — the layered stack, all LLM/db mocked.

Verifies the deterministic control flow: score floor, answerability gate,
tier selection, groundedness downgrade, and flags. The acceptance test against
real adjacent-domain cases (groups C/D/B) is a separate live check.
"""

import unittest
from unittest import mock

from app import rag


def chunks(top1, n=8):
    # descending similarities starting at top1
    return [{"similarity": max(0.0, top1 - 0.02 * i), "url": f"https://plivo.com/docs/p{i}",
             "title": f"P{i}", "heading": f"H{i}", "content": f"content {i}", "source_type": "docs"}
            for i in range(n)]


def run(top1, *, supported=True, qtype="factual", grounded=True, name="Priya"):
    with mock.patch.object(rag.llm, "rag_answerability",
                           lambda q, c: {"supported": supported, "question_type": qtype, "reason": "r"}), \
         mock.patch.object(rag.llm, "rag_draft", lambda q, c, t: "Here is the answer. https://plivo.com/docs/p0"), \
         mock.patch.object(rag.llm, "rag_groundedness",
                           lambda d, c: {"grounded": grounded, "unsupported_claims": [] if grounded else ["x"]}):
        return rag.answer("how do I do X?", customer_name=name,
                          embed=lambda q: [0.0], search=lambda v, k: chunks(top1))


class TestRagRouting(unittest.TestCase):
    def test_below_floor_is_weak_holding_no_llm(self):
        # top1 < 0.55 -> holding, and the answerability gate must NOT be called.
        with mock.patch.object(rag.llm, "rag_answerability",
                               side_effect=AssertionError("gate must not run below floor")):
            r = rag.answer("x", customer_name="P", embed=lambda q: [0.0],
                           search=lambda v, k: chunks(0.40))
        self.assertEqual(r.path, rag.PATH_WEAK)
        self.assertIn("confirm", r.draft_text.lower())
        self.assertTrue(any(f["type"] == "rag_holding" for f in r.flags))

    def test_gate_unsupported_forces_holding_even_with_high_score(self):
        # High score but the answerability gate says no -> holding (the
        # adjacent-domain fabrication defense).
        r = run(0.76, supported=False)
        self.assertEqual(r.path, rag.PATH_WEAK)
        self.assertIn("gate_unsupported", r.reason)

    def test_strong_path_answers_directly(self):
        r = run(0.70, supported=True)
        self.assertEqual(r.path, rag.PATH_STRONG)
        self.assertNotIn("confirm the details", r.draft_text.lower())
        self.assertTrue(r.citations)

    def test_partial_path_between_floor_and_strong(self):
        # 0.55 <= top1 < 0.62 -> PARTIAL, with the verify flag.
        r = run(0.58, supported=True)
        self.assertEqual(r.path, rag.PATH_PARTIAL)
        self.assertTrue(any(f["type"] == "rag_partial" for f in r.flags))

    def test_ungrounded_downgrades_to_holding(self):
        r = run(0.70, supported=True, grounded=False)
        self.assertEqual(r.path, rag.PATH_WEAK)
        self.assertTrue(any(f["type"] == "rag_ungrounded" for f in r.flags))

    def test_no_chunks_holding(self):
        r = rag.answer("x", customer_name="P", embed=lambda q: [0.0], search=lambda v, k: [])
        self.assertEqual(r.path, rag.PATH_WEAK)

    def test_technical_support_holding_asks_for_debug_inputs(self):
        # Below floor -> holding; technical_support wording asks for the API
        # request + error response, and must NOT contain a diagnosis.
        r = rag.answer("calls failing error 3020", customer_name="Sam", intent="technical_support",
                       embed=lambda q: [0.0], search=lambda v, k: chunks(0.40))
        self.assertEqual(r.path, rag.PATH_WEAK)
        self.assertIn("error response", r.draft_text.lower())
        self.assertIn("api request", r.draft_text.lower())

    def test_platform_holding_uses_generic_wording(self):
        r = rag.answer("how do I X", customer_name="Sam", intent="platform_query",
                       embed=lambda q: [0.0], search=lambda v, k: chunks(0.40))
        self.assertIn("confirm the details", r.draft_text.lower())
        self.assertNotIn("error response", r.draft_text.lower())

    def test_strong_requires_mean3_corroboration(self):
        # top1 high but the rest collapse so mean3 < 0.52 -> PARTIAL, not STRONG.
        lopsided = [{"similarity": s, "url": f"u{i}", "title": "t", "heading": "h",
                     "content": "c", "source_type": "docs"}
                    for i, s in enumerate([0.66, 0.40, 0.39])]
        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "factual", "reason": "r"}), \
             mock.patch.object(rag.llm, "rag_draft", lambda q, c, t: "ans"), \
             mock.patch.object(rag.llm, "rag_groundedness",
                               lambda d, c: {"grounded": True, "unsupported_claims": []}):
            r = rag.answer("x", customer_name="P", embed=lambda q: [0.0], search=lambda v, k: lopsided)
        self.assertEqual(r.path, rag.PATH_PARTIAL)


class TestHowtoTemplate(unittest.TestCase):
    def test_guide_url_prefers_docs_over_github(self):
        ch = [{"source_type": "github", "url": "https://github.com/plivo/x/buy.py", "similarity": 0.70},
              {"source_type": "support", "url": "https://support.plivo.com/hc/rent", "similarity": 0.60},
              {"source_type": "docs", "url": "https://plivo.com/docs/numbers/buy", "similarity": 0.58}]
        # docs/support preferred; among them the highest-count/sim wins.
        self.assertIn("plivo.com", rag._guide_url(ch))
        self.assertNotIn("github", rag._guide_url(ch))

    def test_guide_url_majority(self):
        ch = [{"source_type": "docs", "url": "A", "similarity": 0.61},
              {"source_type": "docs", "url": "B", "similarity": 0.66},
              {"source_type": "docs", "url": "A", "similarity": 0.60}]
        self.assertEqual(rag._guide_url(ch), "A")  # 2 chunks beat B's higher sim

    def test_howto_returns_template_no_llm_draft(self):
        ch = [{"similarity": 0.70, "url": "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit",
               "title": "LiveKit", "heading": "LiveKit", "content": "guide", "source_type": "docs"}] * 3
        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "howto", "reason": "r"}), \
             mock.patch.object(rag.llm, "rag_draft", side_effect=AssertionError("no LLM draft for howto template")), \
             mock.patch.object(rag.llm, "rag_groundedness", side_effect=AssertionError("no groundedness for howto template")):
            r = rag.answer("connect livekit", customer_name="Priya",
                           embed=lambda q: [0.0], search=lambda v, k: ch)
        self.assertEqual(r.path, rag.PATH_STRONG)
        self.assertIn("Here's the guide for that:", r.draft_text)
        self.assertIn("integration-guides/livekit", r.draft_text)
        self.assertEqual(r.reason, "howto_template")

    def test_limitation_uses_llm_draft_not_template(self):
        # A grounded-negative ("limitation") must go through the LLM draft +
        # groundedness path (to state the restriction), NOT the how-to link
        # template.
        ch = [{"similarity": 0.66, "url": "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/vapi",
               "title": "Vapi", "heading": "Prerequisites",
               "content": "India not supported: Indian phone numbers are not compatible with Vapi due to TRAI regulations.",
               "source_type": "docs"}] * 3
        used = {"draft": False, "ground": False}

        def d(q, c, t):
            used["draft"] = True
            assert t == "limitation"
            return "Indian numbers are not supported with Vapi due to TRAI rules."
        def g(d_, c):
            used["ground"] = True
            return {"grounded": True, "unsupported_claims": []}

        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "limitation", "reason": "India not supported"}), \
             mock.patch.object(rag.llm, "rag_draft", d), \
             mock.patch.object(rag.llm, "rag_groundedness", g):
            r = rag.answer("connect vapi in indian org", customer_name="P",
                           embed=lambda q: [0.0], search=lambda v, k: ch)
        self.assertTrue(used["draft"] and used["ground"])
        self.assertIn(r.path, ("strong", "partial"))
        self.assertIn("not supported", r.draft_text.lower())

    def test_factual_still_uses_llm_draft_and_groundedness(self):
        ch = [{"similarity": 0.70, "url": "https://plivo.com/docs/messaging/x", "title": "SMS",
               "heading": "SMS", "content": "1600 chars", "source_type": "docs"}] * 3
        called = {"draft": False, "ground": False}

        def d(q, c, t): called["draft"] = True; return "GSM: 1600 chars."
        def g(d_, c): called["ground"] = True; return {"grounded": True, "unsupported_claims": []}

        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "factual", "reason": "r"}), \
             mock.patch.object(rag.llm, "rag_draft", d), \
             mock.patch.object(rag.llm, "rag_groundedness", g):
            r = rag.answer("max sms length", embed=lambda q: [0.0], search=lambda v, k: ch)
        self.assertTrue(called["draft"] and called["ground"])
        self.assertEqual(r.path, rag.PATH_STRONG)


class TestNumericGroundingGate(unittest.TestCase):
    CH = [{"content": "Maximum: 1,600 characters (splits into ~11 segments). "
                      "GSM-7: 160 characters per segment. UCS-2: 70 characters. "
                      "See article 360041449012 for details.",
           "url": "https://support.plivo.com/hc/en-us/articles/360041449012-X"}]

    def test_fabricated_number_caught(self):
        # 737 is not in the docs -> must be caught, deterministically.
        ok, missing = rag._numbers_grounded("UCS-2 max is 737 characters.", self.CH)
        self.assertFalse(ok)
        self.assertIn("737", missing)

    def test_comma_normalization(self):
        # "1600" in draft matches "1,600" in docs.
        ok, _ = rag._numbers_grounded("Up to 1600 characters.", self.CH)
        self.assertTrue(ok)

    def test_doc_backed_numbers_pass(self):
        ok, missing = rag._numbers_grounded(
            "GSM-7 fits 160 characters; UCS-2 fits 70; max 1,600 over 11 segments.", self.CH)
        self.assertTrue(ok, f"unexpected missing: {missing}")

    def test_url_article_id_not_counted_as_claim(self):
        # A number only inside the draft's URL must not be flagged.
        ok, _ = rag._numbers_grounded(
            "See https://support.plivo.com/hc/en-us/articles/999999999-Z for details.", self.CH)
        self.assertTrue(ok)

    def test_substring_is_not_a_match(self):
        # "16" must NOT be considered grounded just because "160"/"1600" contain it.
        ok, missing = rag._numbers_grounded("It supports 16 lanes.", self.CH)
        self.assertFalse(ok)
        self.assertIn("16", missing)

    def test_factual_path_holds_on_bad_number(self):
        ch = [{"similarity": 0.70, "url": "https://plivo.com/docs/m", "title": "SMS",
               "heading": "SMS", "content": "GSM-7: 160 characters. UCS-2: 70 characters.",
               "source_type": "docs"}] * 3
        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "factual", "reason": "r"}), \
             mock.patch.object(rag.llm, "rag_draft", lambda q, c, t: "GSM: 160, UCS-2: 737 characters."), \
             mock.patch.object(rag.llm, "rag_groundedness",
                               lambda d, c: {"grounded": True, "unsupported_claims": []}):  # LLM gate WRONGLY passes
            r = rag.answer("max sms length", customer_name="P",
                           embed=lambda q: [0.0], search=lambda v, k: ch)
        # numeric gate must still hold it despite the LLM gate passing.
        self.assertEqual(r.path, rag.PATH_WEAK)
        self.assertEqual(r.reason, "ungrounded_number")


class TestLinkForwardShortcut(unittest.TestCase):
    CH = [{"url": "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit",
           "content": "LiveKit integration guide ...", "title": "LiveKit"}]

    def test_pure_link_forward_is_grounded(self):
        d = ("Here's the guide for connecting your LiveKit agent to Plivo: "
             "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit "
             "— feel free to ask if you hit anything.\n\nBest regards,\nNihal Manjunath")
        self.assertTrue(rag._link_forward_grounded(d, self.CH))

    def test_link_forward_with_greeting(self):
        d = ("Hi Priya,\n\nHere's the guide for that: "
             "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit "
             "— happy to help if you get stuck.\n\nBest regards,\nNihal Manjunath")
        self.assertTrue(rag._link_forward_grounded(d, self.CH))

    def test_extra_factual_claim_does_not_shortcut(self):
        # A real claim beyond the link ("supports SIP over TLS on port 5061")
        # must fall through to the LLM gate, not be auto-grounded.
        d = ("Here's the guide: "
             "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit "
             "Plivo supports SIP over TLS on port 5061.\n\nBest regards,\nNihal")
        self.assertFalse(rag._link_forward_grounded(d, self.CH))

    def test_url_not_in_chunks_does_not_shortcut(self):
        d = ("Here's the guide: https://plivo.com/docs/totally-made-up-page "
             "— ask away.\n\nBest regards,\nNihal")
        self.assertFalse(rag._link_forward_grounded(d, self.CH))

    def test_no_url_does_not_shortcut(self):
        d = "You can do this by configuring your trunk.\n\nBest regards,\nNihal"
        self.assertFalse(rag._link_forward_grounded(d, self.CH))

    def test_digits_block_shortcut(self):
        d = ("Here's the guide: "
             "https://plivo.com/docs/voice-agents/sip-trunking/integration-guides/livekit "
             "Use 3 retries.\n\nBest regards,\nNihal")
        self.assertFalse(rag._link_forward_grounded(d, self.CH))

    def test_strong_link_forward_skips_llm_groundedness(self):
        # End-to-end: a link-forward STRONG answer must NOT call llm.rag_groundedness.
        ch = [{"similarity": 0.70, "url": self.CH[0]["url"], "title": "LiveKit",
               "heading": "LiveKit", "content": "guide", "source_type": "docs"}] * 3
        draft = ("Here's the guide for that: " + self.CH[0]["url"] +
                 " — feel free to ask.\n\nBest regards,\nNihal Manjunath")
        with mock.patch.object(rag.llm, "rag_answerability",
                               lambda q, c: {"supported": True, "question_type": "howto", "reason": "r"}), \
             mock.patch.object(rag.llm, "rag_draft", lambda q, c, t: draft), \
             mock.patch.object(rag.llm, "rag_groundedness",
                               side_effect=AssertionError("groundedness LLM must be skipped for link-forward")):
            r = rag.answer("connect livekit", customer_name="P",
                           embed=lambda q: [0.0], search=lambda v, k: ch)
        self.assertEqual(r.path, rag.PATH_STRONG)


if __name__ == "__main__":
    unittest.main()
