#!/usr/bin/env python3
"""Stage synthetic demo posts for the README screenshots.

Drives the REAL posting + rendering code paths (run_sms_case → resume_account,
the SMS three-bucket finding renderer, the approval-card block builder) with
SYNTHETIC fixtures — so the screenshots are genuine product output, but nothing
real is touched:

  * the warehouse is mocked (synthetic account 10000002, carrier_code 451),
  * the DB is mocked (no drafts/emails rows written),
  * pending-case state is an in-memory dict (no kv_state writes).

The ONLY real side effect is the Slack posts you screenshot.

Posts, in the test workspace:
  #debugging      : SMS account-ask  →  (simulated "10000002" reply)  →
                    three-bucket finding + labelled representative message
  approval channel: SMS conservative customer draft (from the finding),
                    a docs-RAG grounded card, and a grounded-or-hold card

Run (provide Slack creds via your environment / .env):
    set -a; source .env; set +a
    python scripts/stage_demo.py
"""
import contextlib
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app import debug_orchestrator as orch, slack_approval, rag  # noqa: E402

# --- synthetic fixtures ------------------------------------------------------
ACCOUNT = 10000002
CARRIER = "451"
SENDER = "Jordan Lee <jordan@acme-demo.example>"
CUSTOMER = "Jordan"
_REP_UUID = "a1b2c3d4-0000-4a5b-8c9d-000000000451"
_WINDOW = {"start": "2026-07-07 00:00:00", "end": "2026-07-07 12:00:00", "has_hint": True}


def _histogram(*a, **k):
    # Exact (delivery_state, carrier_code, country) aggregate — no row cap.
    # Three buckets: failed=45 (451 dominant), delivered=140, limbo=8  → CLEAR.
    return [
        {"delivery_state": "failed",      "carrier_code": CARRIER, "country_iso": "IN", "n": 32},
        {"delivery_state": "failed",      "carrier_code": CARRIER, "country_iso": "US", "n": 8},
        {"delivery_state": "undelivered", "carrier_code": "300",   "country_iso": "IN", "n": 5},
        {"delivery_state": "delivered",   "carrier_code": "000",   "country_iso": "IN", "n": 140},
        {"delivery_state": "sent",        "carrier_code": "",      "country_iso": "IN", "n": 6},
        {"delivery_state": "queued",      "carrier_code": "",      "country_iso": "US", "n": 2},
    ]


def _rep_row(*a, **k):
    return {"found": True, "channel": "sms", "source_table": "messages_fact",
            "message_uuid": _REP_UUID, "account_id": ACCOUNT, "sub_account_id": 0,
            "delivery_state": "failed", "message_direction": "outbound", "message_type": "sms",
            "to_number_redacted": "9198765****", "from_number_redacted": "9188888****",
            "country_iso": "IN", "carrier_code": CARRIER, "error_code": 64, "units": 1,
            "message_time": "2026-07-07 09:14:22"}


def _messages_for_account(*a, **k):
    return [{"message_uuid": _REP_UUID, "delivery_state": "failed", "carrier_code": CARRIER,
             "country_iso": "IN", "message_time": "2026-07-07 09:14:22"}]


# --- in-memory pending store (never touches real kv_state) -------------------
_KV: dict = {}
_DRAFT_ID = [4000]


def _persist(*a, **k):
    _DRAFT_ID[0] += 1
    return {"draft_id": _DRAFT_ID[0]}


# --- synthetic content for the two extra approval cards ----------------------
DOCS_Q = ("Hi — how do I receive delivery receipts (DLRs) for my outbound SMS? "
          "Where do I set the callback URL?")
DOCS_ANSWER = (
    "Hi,\n\nYou can receive delivery reports by passing a `url` parameter (your callback "
    "URL) when you send a message, or by setting a default Message callback URL on your "
    "application. Plivo then POSTs the delivery status to that URL as the message moves "
    "through queued → sent → delivered/undelivered. See the Message API docs for the exact "
    "parameters.\n\nBest regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo")
HOLD_Q = ("Hi — can you confirm the exact GST percentage applied to my last invoice and "
          "share a sample calculation?")


def main() -> int:
    if not os.getenv("SLACK_BOT_TOKEN"):
        print("SLACK_BOT_TOKEN not set — export it (or source your .env) first.", file=sys.stderr)
        return 2

    seams = [
        # warehouse → synthetic fixtures
        mock.patch.object(orch.redshift_tools, "get_message_histogram", _histogram),
        mock.patch.object(orch.redshift_tools, "get_message_type_breakdown", lambda *a, **k: []),
        mock.patch.object(orch.redshift_tools, "get_messages_for_account", _messages_for_account),
        mock.patch.object(orch.redshift_tools, "get_message_by_uuid", _rep_row),
        mock.patch.object(orch, "_extract_window", lambda thread: _WINDOW),
        # DB → no rows written
        mock.patch.object(orch.db, "persist_processing", _persist),
        mock.patch.object(slack_approval.db, "claim_draft_for_posting", lambda did, **k: True),
        mock.patch.object(slack_approval.db, "set_slack_message", lambda *a, **k: None),
        # pending-case state → in-memory only
        mock.patch.object(orch.db, "kv_set", lambda k, v: _KV.__setitem__(k, v)),
        mock.patch.object(orch.db, "kv_get", lambda k: _KV.get(k)),
        mock.patch.object(orch.db, "claim_pending_case", lambda k: _KV.pop(k, None)),
    ]

    thread = {
        "thread_id": "demo-sms", "subject": "SMS delivery failures this morning",
        "customer_name": CUSTOMER, "reply_context": {"to": SENDER},
        "messages": [{"from": SENDER, "body": (
            "A lot of our outbound SMS started failing this morning — can you check "
            "what's going on?")}],
    }

    with contextlib.ExitStack() as es:
        for s in seams:
            es.enter_context(s)

        print("Staging demo posts (synthetic data; no live mail / DB / warehouse touched)…\n")

        # (1) #debugging: account-ask → simulated account-id reply → finding + customer draft
        art = orch.run_sms_case(thread, dry_run=False)
        ts = art["debug_ts"]
        print(f"  ✓ #debugging: SMS account-ask posted (ts={ts})")
        res = orch.resume_account(ts, str(ACCOUNT))  # simulate the reviewer replying "10000002"
        print(f"  ✓ #debugging: three-bucket finding + representative posted "
              f"(verdict branch={res.get('branch')})")
        print(f"  ✓ approval : SMS conservative customer draft card (draft_id={res.get('draft_id')})")

        # (2) docs-RAG grounded approval card
        slack_approval.post_draft_for_approval(
            DOCS_Q, DOCS_ANSWER, 4101,
            flags=[{"type": "rag_grounded",
                    "text": "Answer grounded in retrieved docs (strong match). "
                            "Source: plivo.com/docs/sms/api/message#callback-url"}])
        print("  ✓ approval : docs-RAG grounded card")

        # (3) grounded-or-hold approval card (ungroundable question → safe hold, never fabricated)
        slack_approval.post_draft_for_approval(
            HOLD_Q, rag.holding_reply("Priya", "account_billing"), 4102,
            flags=[{"type": "rag_holding",
                    "text": "No confident/grounded doc answer (weak_score) — a fact-free holding "
                            "reply was drafted; needs a manual answer. Nothing fabricated."}])
        print("  ✓ approval : grounded-or-hold card")

    print("\nAll staged posts are up. Nothing was written to the DB, kv_state, or the warehouse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
