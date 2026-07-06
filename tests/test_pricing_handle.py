"""Tests for the SMS pricing draft templating + numeric post-check + routing
(app/pricing.draft_sms_pricing / handle). LLM + db mocked. The rate is never
model-generated; these lock that the templates quote only looked-up figures and
hold/ask correctly.
"""

import unittest
from unittest import mock

from app import pricing


class TestDraftTemplates(unittest.TestCase):
    def test_quote_contains_only_looked_up_rate(self):
        r = {"status": "quote", "iso": "AE", "country": "United Arab Emirates",
             "rate": "0.1089", "currency": "USD"}
        d = pricing.draft_sms_pricing(r, "Priya")
        self.assertIn("$0.1089 USD", d)
        self.assertTrue(pricing._numeric_post_check(d, pricing._allowed_rates(r)))
        # no other rate-shaped number leaked
        self.assertEqual(set(pricing._RATE_NUM_RE.findall(d)), {"0.1089"})

    def test_us_routes_quotes_three_rates(self):
        r = {"status": "us_routes", "routes": [
            {"iso": "US-10DLC", "rate_usd": "0.0077"},
            {"iso": "US-SC", "rate_usd": "0.0077"},
            {"iso": "US-TF", "rate_usd": "0.0079"}]}
        d = pricing.draft_sms_pricing(r, None)
        self.assertIn("0.0077", d)
        self.assertIn("0.0079", d)
        self.assertTrue(pricing._numeric_post_check(d, pricing._allowed_rates(r)))
        self.assertLessEqual(set(pricing._RATE_NUM_RE.findall(d)), {"0.0077", "0.0079"})

    def test_india_and_holds_contain_no_rate(self):
        for r in [{"status": "india_unavailable"},
                  {"status": "hold_not_in_sheet", "iso": "KP"},
                  {"status": "ask_country"},
                  {"status": "ask_channel"},
                  {"status": "ask_disambiguate", "candidates": [("Republic of the Congo", "CG"),
                                                                ("Democratic Republic of the Congo", "CD")]},
                  {"status": "channel_deferred", "channel": "voice"}]:
            d = pricing.draft_sms_pricing(r, "Sam")
            self.assertEqual(pricing._RATE_NUM_RE.findall(d), [], f"{r['status']} leaked a number: {d}")

    def test_india_message(self):
        d = pricing.draft_sms_pricing({"status": "india_unavailable"}, "Sam")
        self.assertIn("don't currently offer SMS to India", d)

    def test_disambiguate_lists_candidates(self):
        d = pricing.draft_sms_pricing({"status": "ask_disambiguate",
            "candidates": [("Republic of the Congo", "CG"), ("Democratic Republic of the Congo", "CD")]}, None)
        self.assertIn("Republic of the Congo", d)
        self.assertIn("Democratic Republic of the Congo", d)


class TestHandleRouting(unittest.TestCase):
    THREAD = {"thread_id": "t", "subject": "pricing",
              "messages": [{"from": "p@x.com", "date": "d", "body": "rate?"}]}

    def _handle(self, channel, country):
        chans = [channel] if channel and channel != "unknown" else []
        ctys = [country] if country else []
        with mock.patch.object(pricing.llm, "extract_pricing",
                               lambda t: {"channels": chans, "countries": ctys}):
            return pricing.handle(self.THREAD, {"customer_name": "P"})

    def test_sms_uae_quotes(self):
        with mock.patch.object(pricing, "lookup_sms_price",
                               lambda c: {"status": "quote", "iso": "AE", "country": "United Arab Emirates",
                                          "rate": "0.1089", "currency": "USD"}):
            draft, flags = self._handle("sms", "UAE")
        self.assertIn("$0.1089 USD", draft)
        self.assertEqual(flags, [])

    def test_voice_routes_to_web_not_defer(self):
        # Voice now resolves via web_pricing (stage 2). With no row -> hold, NOT
        # the old "channel_deferred". (web_quote path covered in test_pricing_web.)
        with mock.patch.object(pricing.db, "get_web_pricing", lambda ch, iso: None):
            draft, flags = self._handle("voice", "UAE")
        self.assertEqual(flags[0]["type"], "pricing_hold")
        self.assertEqual(pricing._RATE_NUM_RE.findall(draft), [])

    def test_rcs_holds(self):
        draft, flags = self._handle("rcs", "US")
        self.assertEqual(flags[0]["type"], "pricing_channel_unsupported")

    def test_unknown_channel_asks(self):
        draft, flags = self._handle("unknown", None)
        self.assertEqual(flags[0]["type"], "pricing_ask")


if __name__ == "__main__":
    unittest.main()
