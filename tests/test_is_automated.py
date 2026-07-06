"""Tests for the automated-sender pre-filter (gmail_client.is_automated).

Run with either:
    .venv/bin/python -m unittest tests.test_is_automated
    .venv/bin/python -m pytest tests/test_is_automated.py   # once pytest is installed
"""

import unittest

from app.gmail_client import is_automated


def _headers(**kwargs) -> list[dict]:
    """Build Gmail-style [{name, value}] headers from keyword pairs."""
    return [{"name": k.replace("_", "-"), "value": v} for k, v in kwargs.items()]


class TestIsAutomated(unittest.TestCase):
    # --- The Fireflies regression: friendly From name, VERP bounce envelope ---

    def test_fireflies_friendly_name_verp_envelope_is_skipped(self):
        """The exact shape that slipped through: human From, ESP bounce Return-Path."""
        from_hdr = '"Fred from Fireflies.ai" <fred@fireflies.ai>'
        headers = _headers(
            From=from_hdr,
            Return_Path="<bounces+1696684-321a-nihal.manjunath=plivo.com@send.fireflies.ai>",
        )
        skip, reason = is_automated(from_hdr, headers, ["UNREAD", "CATEGORY_UPDATES", "INBOX"])
        self.assertTrue(skip, "Fireflies notification mail must be skipped")
        self.assertIn("bounce", reason.lower())

    # --- Generalizes across ESPs without hardcoding a vendor ---

    def test_verp_recipient_encoding_any_domain(self):
        """The '=' recipient-encoding tell, on an arbitrary sending domain."""
        from_hdr = '"Casey at Acme" <casey@acme.io>'
        headers = _headers(
            From=from_hdr,
            Return_Path="<msprvs1=18ab=foo=bar.com@bounce.mail.acme.io>",
        )
        skip, _ = is_automated(from_hdr, headers, None)
        self.assertTrue(skip)

    def test_esp_bounce_prefix_variants(self):
        for rp in (
            "<bounce-12345@mg.example.com>",
            "<prvs=abc123=user@sub.example.com>",
            "<fbl+xyz@email.vendor.net>",
        ):
            with self.subTest(return_path=rp):
                from_hdr = '"Jamie" <jamie@example.com>'
                headers = _headers(From=from_hdr, Return_Path=rp)
                skip, _ = is_automated(from_hdr, headers, None)
                self.assertTrue(skip, f"{rp} should be detected as a bulk envelope")

    def test_automated_envelope_in_sender_header(self):
        """Some mail puts the automated address in Sender rather than Return-Path."""
        from_hdr = '"Notifications Team" <hello@product.com>'
        headers = _headers(From=from_hdr, Sender="<bounces+9=user=product.com@sg.product.com>")
        skip, _ = is_automated(from_hdr, headers, None)
        self.assertTrue(skip)

    # --- Real human mail must NOT be filtered ---

    def test_genuine_customer_mail_is_not_skipped(self):
        from_hdr = '"Sushil Kumar" <sushil@bigcorp.com>'
        headers = _headers(
            From=from_hdr,
            Return_Path="<sushil@bigcorp.com>",
            Reply_To="sushil@bigcorp.com",
        )
        skip, reason = is_automated(from_hdr, headers, ["UNREAD", "INBOX"])
        self.assertFalse(skip, f"genuine mail wrongly skipped: {reason}")

    def test_plus_tag_in_human_address_is_not_verp(self):
        """A '+tag' on a normal human address must not trip the VERP check."""
        from_hdr = "<nihal+plivo@gmail.com>"
        headers = _headers(From=from_hdr, Return_Path="<nihal+plivo@gmail.com>")
        skip, reason = is_automated(from_hdr, headers, None)
        self.assertFalse(skip, f"plus-tagged human address wrongly skipped: {reason}")

    # --- Notification-service signals (Jira / Gemini / integration relays) ---

    def test_feedback_id_header_is_bulk(self):
        """ESP feedback-loop id (AmazonSES/SendGrid) marks bulk/automated mail."""
        from_hdr = '"hemanth.kb (Jira)" <jira@plivo-team.atlassian.net>'
        headers = _headers(
            From=from_hdr,
            **{"Feedback-Id": "ip.0f2fa25c:la.jira.PO-jira-project-recap:1.us-east-1.abc:AmazonSES"},
        )
        skip, reason = is_automated(from_hdr, headers, None)
        self.assertTrue(skip)
        self.assertIn("Feedback-ID", reason)

    def test_bounce_subdomain_envelope(self):
        """Gemini-style: human-ish From, envelope via a *.bounces.* subdomain."""
        from_hdr = "Gemini <gemini-notes@google.com>"
        headers = _headers(
            From=from_hdr,
            Return_Path="<3w9o0agwKD6QKIQ@rtc-meetings-api.bounces.google.com>",
        )
        skip, reason = is_automated(from_hdr, headers, None)
        self.assertTrue(skip)
        self.assertIn("bounce", reason.lower())

    def test_atlassian_bounces_label_envelope(self):
        from_hdr = "Jira automation <automation@gokwik.atlassian.net>"
        headers = _headers(
            From=from_hdr,
            Return_Path="<0101019ed5787519-7235957a@atlassian-bounces.atlassian.net>",
        )
        self.assertTrue(is_automated(from_hdr, headers, None)[0])

    def test_integration_relay_localpart(self):
        """Zapier-relayed summary from an integration alias, no header signals."""
        from_hdr = '"Fireflies Meeting Summary (Zapier)" <marketing-integrations@plivo.com>'
        headers = _headers(From=from_hdr, Return_Path="<marketing-integrations@plivo.com>")
        skip, _ = is_automated(from_hdr, headers, None)
        self.assertTrue(skip)

    def test_automation_localpart(self):
        from_hdr = "<automation@acme.com>"
        self.assertTrue(is_automated(from_hdr, _headers(From=from_hdr), None)[0])

    def test_genuine_corp_mail_not_flagged_by_new_rules(self):
        """The new domain/header rules must not catch ordinary customer replies."""
        for from_hdr, rp in (
            ('"Trupti Shetty" <Trupti.T.Shetty@shell.com>', "<Trupti.T.Shetty@shell.com>"),
            ('"Surbhi Vasoya" <surbhi@deepvox.ai>', "<surbhi@deepvox.ai>"),
            ('"Laura-Jade Brooks" <laura-jade.brooks@tagmarshal.com>', "<laura-jade.brooks@tagmarshal.com>"),
        ):
            with self.subTest(sender=from_hdr):
                skip, reason = is_automated(from_hdr, _headers(From=from_hdr, Return_Path=rp), ["INBOX"])
                self.assertFalse(skip, f"genuine mail wrongly skipped: {reason}")

    # --- Pre-existing signals still fire ---

    def test_auto_submitted_header(self):
        from_hdr = "<svc@example.com>"
        skip, _ = is_automated(from_hdr, _headers(From=from_hdr, Auto_Submitted="auto-generated"), None)
        self.assertTrue(skip)

    def test_noreply_localpart(self):
        from_hdr = '"Acme" <no-reply@acme.com>'
        skip, _ = is_automated(from_hdr, _headers(From=from_hdr), None)
        self.assertTrue(skip)

    def test_list_unsubscribe(self):
        from_hdr = '"News" <hello@acme.com>'
        skip, _ = is_automated(from_hdr, _headers(From=from_hdr, List_Unsubscribe="<mailto:u@acme.com>"), None)
        self.assertTrue(skip)


if __name__ == "__main__":
    unittest.main()
