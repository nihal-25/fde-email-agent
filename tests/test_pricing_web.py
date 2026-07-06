"""Tests for the web-pricing (voice/WhatsApp/numbers) handler: faithful
range-quoting (qualifier inseparable from number), channel routing, and the
critical India-override-is-SMS-only proof. LLM + db mocked.
"""

import unittest
from unittest import mock

from app import pricing

VOICE_IN = {  # India voice page (₹) as parsed
    "channel": "voice", "iso": "IN", "country_name": "India", "currency": "INR",
    "imported_at": "2026-06-22T10:00:00+00:00",
    "tables": [{"section": "Route Type | To Make Calls (Outbound) | To Receive Calls (Inbound)",
                "rows": [
                    {"label": "Local Calls", "prices": {"To Make Calls (Outbound)": "₹0.60/min",
                                                        "To Receive Calls (Inbound)": "₹0.60/min"}},
                    {"label": "Toll-Free Calls", "prices": {"To Make Calls (Outbound)": "Not Supported",
                                                            "To Receive Calls (Inbound)": "₹1.30/min"}},
                ]}]}
VOICE_GB = {  # has a "Starts at" range
    "channel": "voice", "iso": "GB", "country_name": "United Kingdom", "currency": "USD",
    "imported_at": "2026-06-22T10:00:00+00:00",
    "tables": [{"section": "Route Type | To Make Calls (Outbound) | To Receive Calls (Inbound)",
                "rows": [{"label": "Local Calls", "prices": {"To Make Calls (Outbound)": "Starts at $0.0075/min",
                                                            "To Receive Calls (Inbound)": "$0.0055/min"}}]}]}


class TestWebPostCheck(unittest.TestCase):
    def test_flattening_fails(self):
        allowed = {"Starts at $0.0075/min", "$0.0055/min"}
        self.assertTrue(pricing._web_post_check("out: Starts at $0.0075/min, in: $0.0055/min", allowed))
        self.assertFalse(pricing._web_post_check("out: $0.0075/min, in: $0.0055/min", allowed))  # qualifier dropped

    def test_not_supported_preserved(self):
        self.assertTrue(pricing._web_post_check("Toll-Free out: Not Supported, in: ₹1.30/min",
                                                {"Not Supported", "₹1.30/min"}))

    def test_fabricated_extra_number_fails(self):
        self.assertFalse(pricing._web_post_check("Local: ₹0.60/min and also ₹9.99/min",
                                                 {"₹0.60/min"}))


class TestWebLookupAndDraft(unittest.TestCase):
    def test_voice_quote_renders_full_strings(self):
        with mock.patch.object(pricing.db, "get_web_pricing", lambda ch, iso: VOICE_GB if iso == "GB" else None):
            r = pricing.lookup_web_price("voice", "United Kingdom")
        self.assertEqual(r["status"], "web_quote")
        d = pricing.draft_web_pricing(r, "Sam")
        self.assertIn("Starts at $0.0075/min", d)          # qualifier travels into the draft
        self.assertNotIn("\n- Local Calls — outbound: $0.0075/min", d)  # not flattened
        self.assertIn("as of 2026-06-22", d)               # freshness stamp
        self.assertTrue(pricing._web_post_check(d, pricing._web_allowed_strings(r["table"])))

    def test_not_covered_holds(self):
        with mock.patch.object(pricing.db, "get_web_pricing", lambda ch, iso: None):
            r = pricing.lookup_web_price("voice", "Kiribati")
        self.assertEqual(r["status"], "hold_not_covered")

    def test_ambiguous_asks(self):
        r = pricing.lookup_web_price("voice", "Congo")
        self.assertEqual(r["status"], "ask_disambiguate")


class TestChannelScopedIndiaOverride(unittest.TestCase):
    """The critical proof: India override is SMS-only. India voice/WhatsApp quote
    in INR; India SMS still says not-offered."""

    def _handle(self, channel, country):
        with mock.patch.object(pricing.llm, "extract_pricing",
                               lambda t: {"channels": [channel], "countries": [country]}), \
             mock.patch.object(pricing.db, "get_web_pricing", lambda ch, iso: VOICE_IN if iso == "IN" else None):
            return pricing.handle({"thread_id": "t", "subject": "p",
                                   "messages": [{"from": "c@x", "date": "d", "body": "?"}]},
                                  {"customer_name": "C"})

    def test_india_voice_quotes_inr_not_unavailable(self):
        draft, flags = self._handle("voice", "India")
        self.assertIn("₹0.60/min", draft)
        self.assertNotIn("don't currently offer", draft)

    def test_india_sms_still_unavailable(self):
        with mock.patch.object(pricing.llm, "extract_pricing",
                               lambda t: {"channels": ["sms"], "countries": ["India"]}):
            draft, flags = pricing.handle({"thread_id": "t", "subject": "p",
                                           "messages": [{"from": "c@x", "date": "d", "body": "?"}]},
                                          {"customer_name": "C"})
        self.assertIn("don't currently offer SMS to India", draft)

    def test_rcs_holds(self):
        draft, flags = self._handle("rcs", "United States")
        self.assertEqual(flags[0]["type"], "pricing_channel_unsupported")


class TestMultiQuote(unittest.TestCase):
    """List-all behavior: same per-pair lookup concatenated; per-line honest;
    mixed currencies labeled; India override per (channel, country)."""

    def _multi(self, channels, countries, web=lambda ch, iso: VOICE_IN if iso == "IN" else None):
        with mock.patch.object(pricing.llm, "extract_pricing",
                               lambda t: {"channels": channels, "countries": countries}), \
             mock.patch.object(pricing.db, "get_web_pricing", web), \
             mock.patch.object(pricing, "lookup_sms_price", _fake_sms):
            return pricing.handle({"thread_id": "t", "subject": "p",
                                   "messages": [{"from": "c@x", "date": "d", "body": "?"}]},
                                  {"customer_name": "C"})

    def test_multi_channel_india_sms_notoffered_voice_inr(self):
        # "SMS and voice to India" -> SMS not-offered + voice ₹ in ONE reply.
        draft, flags = self._multi(["sms", "voice"], ["India"])
        self.assertIn("don't currently offer SMS to India", draft)   # override per-(channel,country)
        self.assertIn("₹0.60/min", draft)                            # voice India quotes ₹
        self.assertEqual(flags[0]["type"], "pricing_multi")

    def test_mixed_currency_labeled_not_converted(self):
        # voice India (₹) + voice US ($) in one reply, each its own currency.
        web = lambda ch, iso: VOICE_IN if iso == "IN" else (VOICE_US if iso == "US" else None)
        draft, flags = self._multi(["voice"], ["India", "US"], web=web)
        self.assertIn("₹0.60/min", draft)
        self.assertIn("$0.0115/min", draft)
        self.assertIn("(INR", draft)
        self.assertIn("(USD", draft)

    def test_mixed_resolve_and_hold_never_drops(self):
        # one covered (US voice $) + one uncovered (Kiribati voice) -> quote one,
        # hold the other, never drop the uncovered one.
        web = lambda ch, iso: VOICE_US if iso == "US" else None
        draft, flags = self._multi(["voice"], ["US", "Kiribati"], web=web)
        self.assertIn("$0.0115/min", draft)                          # US quoted
        self.assertIn("Let me confirm", draft)                       # Kiribati held, not dropped

    def test_cap_points_to_page(self):
        draft, flags = self._multi(["voice", "whatsapp"], ["US", "India", "GB", "AU"])
        self.assertIn("plivo.com/pricing", draft)
        self.assertEqual(flags[0]["type"], "pricing_too_many")


VOICE_US = {"channel": "voice", "iso": "US", "country_name": "United States", "currency": "USD",
            "imported_at": "2026-06-22T10:00:00+00:00",
            "tables": [{"section": "Route Type | To Make Calls (Outbound) | To Receive Calls (Inbound)",
                        "rows": [{"label": "Local Calls", "prices": {"To Make Calls (Outbound)": "$0.0115/min",
                                                                    "To Receive Calls (Inbound)": "$0.0055/min"}}]}]}


def _fake_sms(country):
    from app import pricing as p
    r = p.resolve_country(country)
    if r.get("iso") == "IN":
        return {"status": "india_unavailable"}
    return {"status": "hold_not_in_sheet", "iso": r.get("iso", "XX")}


if __name__ == "__main__":
    unittest.main()
