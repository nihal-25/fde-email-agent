#!/usr/bin/env python3
"""Two-phase demo staging for the Loom recording.

PHASE 1 (arm): clean up any OLD staged posts (so only fresh posts are on camera),
post the fresh approval cards (docs-RAG grounded, grounded-or-hold) and ONLY the
SMS account-ask in #debugging, then WAIT — poll the account-ask thread for a
human (non-bot) reply.

PHASE 2 (on your reply): treat the reply text as the account id and post, IN THE
SAME THREAD, the three-bucket finding + labelled representative; then post the
conservative SMS customer draft to the approval channel.

Synthetic fixtures throughout (account 10000002, carrier_code 451, synthetic
sender). Pure renderers + post_draft_for_approval + _post only — NO DB, NO
warehouse, NO pending-case state is touched. ~10 min timeout so it never hangs.

Usage (creds via env / .env):
    python scripts/stage_demo.py --cleanup-only   # just delete old staged posts
    python scripts/stage_demo.py --skip-cleanup   # arm without deleting
    python scripts/stage_demo.py                  # cleanup, then arm + wait
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from slack_sdk import WebClient  # noqa: E402
from app import debug_orchestrator as orch, slack_approval, rag  # noqa: E402

# --- synthetic fixtures ------------------------------------------------------
ACCOUNT = 10000002
CARRIER = "451"
SENDER = "Jordan Lee <jordan@acme-demo.example>"
CUSTOMER = "Jordan"
SUBJECT = "SMS delivery failures this morning"
CUST_MAIL = "A lot of our outbound SMS started failing this morning — can you check what's going on?"
_REP_UUID = "a1b2c3d4-0000-4a5b-8c9d-000000000451"
_WINDOW = {"start": "2026-07-07 00:00:00", "end": "2026-07-07 12:00:00", "has_hint": True}

POLL_S = 3
TIMEOUT_S = 600  # ~10 min

DEBUG_CHANNEL = orch.DEBUG_CHANNEL
APPROVAL_CHANNEL = os.getenv("SLACK_APPROVAL_CHANNEL")

# High-specificity markers of THIS demo's staged posts — synthetic strings that
# real customer mail will not contain, so cleanup never touches a real post.
STAGED_MARKERS = [
    "10000002", "acme-demo", SUBJECT, "outbound SMS started failing this morning",
    _REP_UUID, "callback-url", "delivery receipts (DLRs)",
    "GST percentage applied to my last invoice",
]

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


def _histogram():
    # Exact (delivery_state, carrier_code, country) aggregate — three buckets:
    # failed=45 (451 dominant), delivered=140, limbo=8 → CLEAR.
    return [
        {"delivery_state": "failed",      "carrier_code": CARRIER, "country_iso": "IN", "n": 32},
        {"delivery_state": "failed",      "carrier_code": CARRIER, "country_iso": "US", "n": 8},
        {"delivery_state": "undelivered", "carrier_code": "300",   "country_iso": "IN", "n": 5},
        {"delivery_state": "delivered",   "carrier_code": "000",   "country_iso": "IN", "n": 140},
        {"delivery_state": "sent",        "carrier_code": "",      "country_iso": "IN", "n": 6},
        {"delivery_state": "queued",      "carrier_code": "",      "country_iso": "US", "n": 2},
    ]


def _rep_row():
    return {"found": True, "channel": "sms", "source_table": "messages_fact",
            "message_uuid": _REP_UUID, "account_id": ACCOUNT, "sub_account_id": 0,
            "delivery_state": "failed", "message_direction": "outbound", "message_type": "sms",
            "to_number_redacted": "9198765****", "from_number_redacted": "9188888****",
            "country_iso": "IN", "carrier_code": CARRIER, "error_code": 64, "units": 1,
            "message_time": "2026-07-07 09:14:22"}


def _build_inv():
    b = orch._sms_buckets(_histogram())
    v = orch._sms_verdict(b)
    return {"account_id": ACCOUNT, "window": _WINDOW, **v, "buckets": b,
            "representative_uuid": (_REP_UUID if v.get("failed_finding") else None), "cap_note": ""}


def cleanup_old_staged(client):
    """Delete this demo's OLD staged posts from both channels (marker-matched, so a
    real post is never touched). Prints each deletion for audit."""
    deleted = 0
    for ch in (DEBUG_CHANNEL, APPROVAL_CHANNEL):
        if not ch:
            continue
        try:
            msgs = client.conversations_history(channel=ch, limit=100).get("messages", [])
        except Exception as e:
            print(f"[cleanup] history fetch failed for {ch}: {type(e).__name__}: {e}", flush=True)
            continue
        for m in msgs:
            if any(mk in json.dumps(m) for mk in STAGED_MARKERS):
                try:
                    client.chat_delete(channel=ch, ts=m["ts"])
                    deleted += 1
                    snip = (m.get("text") or "").replace("\n", " ")[:55]
                    print(f"[cleanup] deleted {ch} ts={m['ts']} :: {snip!r}", flush=True)
                except Exception as e:
                    print(f"[cleanup] delete FAILED {ch} ts={m.get('ts')}: {type(e).__name__}: {e}", flush=True)
    print(f"[cleanup] removed {deleted} old staged post(s)", flush=True)
    return deleted


def _wait_for_human_reply(client, ts):
    """Poll the account-ask thread for the first human (non-bot) reply; return its
    text, or None on timeout."""
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        try:
            msgs = client.conversations_replies(channel=DEBUG_CHANNEL, ts=ts, limit=50).get("messages", [])
        except Exception as e:
            print(f"[wait] replies poll error: {type(e).__name__}: {e}", flush=True)
            time.sleep(POLL_S)
            continue
        for m in msgs:
            if m.get("ts") == ts:  # the bot's own account-ask (thread parent)
                continue
            if m.get("bot_id") or m.get("subtype") or not m.get("user"):
                continue
            return (m.get("text") or "").strip()
        time.sleep(POLL_S)
    return None


def phase2(client, ask_ts, reply_text):
    print(f"[phase2] human reply detected: {reply_text!r} — treating as the account id", flush=True)
    inv = _build_inv()
    facts = (f"\n*Representative failed message (one example — most recent with the dominant "
             f"code carrier_code={inv.get('dom_code')}; the counts above are the full-window "
             f"aggregate, not this single row):*\n" + orch._sms_facts(_rep_row()))
    finding = (f"*Account-wide SMS — {ACCOUNT}*\n{orch._sms_header(inv)}\n"
               f"*Interpretation (pattern-descriptive — not a cause):*\n"
               f"{orch._sms_interpretation(inv)}{facts}")
    orch._post(DEBUG_CHANNEL, finding, thread_ts=ask_ts)
    print("[phase2] #debugging: three-bucket finding + representative posted IN-THREAD", flush=True)

    draft = orch._strip_internal(orch._sms_customer_message(inv, CUSTOMER), ACCOUNT)
    slack_approval.post_draft_for_approval(
        CUST_MAIL, draft, 4001,
        flags=[{"type": "sms_debug",
                "text": "SMS finding — conservative customer draft (states only what the data "
                        "shows; no invented cause)."}])
    print("[phase2] approval: conservative SMS customer draft card", flush=True)
    print("DONE — phase 2 posted.", flush=True)


def main():
    if not os.getenv("SLACK_BOT_TOKEN"):
        print("SLACK_BOT_TOKEN not set — source your .env first.", file=sys.stderr)
        return 2
    args = set(sys.argv[1:])
    client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

    if "--skip-cleanup" not in args:
        cleanup_old_staged(client)
    if "--cleanup-only" in args:
        return 0

    # PHASE 1 — fresh approval cards + the account-ask, then wait.
    slack_approval.post_draft_for_approval(
        DOCS_Q, DOCS_ANSWER, 4101,
        flags=[{"type": "rag_grounded",
                "text": "Answer grounded in retrieved docs (strong match). "
                        "Source: plivo.com/docs/sms/api/message#callback-url"}])
    print("[phase1] approval: docs-RAG grounded card", flush=True)
    slack_approval.post_draft_for_approval(
        HOLD_Q, rag.holding_reply("Priya", "account_billing"), 4102,
        flags=[{"type": "rag_holding",
                "text": "No confident/grounded doc answer (weak_score) — a fact-free holding reply "
                        "was drafted; needs a manual answer. Nothing fabricated."}])
    print("[phase1] approval: grounded-or-hold card", flush=True)

    ask = (f"*New SMS debugging case* — {SUBJECT} — from {SENDER}  (no UUID in mail)\n"
           f"What's the *account_id* for this SMS case? (reply with just the number)")
    ask_ts = orch._post(DEBUG_CHANNEL, ask).get("ts")
    print(f"[phase1] #debugging: account-ask posted (ts={ask_ts})", flush=True)
    print(f"PHASE 1 ARMED — reply IN THE #debugging account-ask THREAD with an account id "
          f"(e.g. {ACCOUNT}); polling every {POLL_S}s, up to {TIMEOUT_S // 60} min…", flush=True)

    reply = _wait_for_human_reply(client, ask_ts)
    if reply is None:
        print("PHASE 2 SKIPPED — timed out with no reply (no phase-2 posts made).", flush=True)
        return 0
    phase2(client, ask_ts, reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
