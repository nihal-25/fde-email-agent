"""Main worker loop (Milestone 5): pull -> classify -> draft -> Slack.

Flow:
1. Ensure a Gmail watch is registered (renewed daily) -> Pub/Sub topic.
2. Pull change notifications from the Pub/Sub subscription (no inbound webhook).
3. For each notification, find new INBOX messages since the last historyId,
   fetch each full thread, classify + draft, persist with reply context, and
   post to Slack for approval.

Sending is NOT done here — it happens in app/slack_approval.py when a human
clicks Approve. This worker only ingests and proposes.

Run:  python -m app.worker        (Ctrl+C to stop)
Also run the approval listener in parallel:  python -m app.slack_approval
"""

from __future__ import annotations

import json
import os
import threading
import time

from dotenv import load_dotenv

from app import (
    calendar_client,
    db,
    draft as draft_mod,
    gmail_client,
    llm,
    pricing,
    rag,
    scheduling,
    slack_approval,
)
from app.cli import render_thread

load_dotenv()

SUBSCRIPTION = os.getenv("GMAIL_PUBSUB_SUBSCRIPTION")
WATCH_CHECK_INTERVAL_S = 6 * 3600  # re-check watch expiry every 6h
SWEEP_INTERVAL_S = int(os.getenv("SWEEP_INTERVAL_S", "300"))  # orphaned-draft sweep cadence

# Serializes notification processing so the startup catch-up and live Pub/Sub
# callbacks can't read the same history cursor and double-process messages.
_PROCESS_LOCK = threading.Lock()

# Intents routed through the layered RAG stack's grounded-or-hold gate instead of
# the generic LLM free-write. This is the single fabrication gate for every intent
# that would otherwise assert prose facts: the two doc-answer intents plus the
# generic ones (account_billing, general_inquiry, feature_request, other) that
# used to free-write hedged-but-ungrounded claims (see draft 1081's GST/tax
# assertions). No carve-out — the classifier boundary between these is too fuzzy
# to hang a safety property on, and hedging is not grounding. meeting_request and
# pricing_question have their own dedicated grounded paths and are excluded here.
_GROUNDED_OR_HOLD_INTENTS = frozenset({
    "platform_query", "technical_support",
    "account_billing", "general_inquiry", "feature_request", "other",
})


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fallback_draft_text(thread: dict, err: Exception) -> str:
    """A clearly-marked placeholder so a thread we couldn't auto-draft still
    reaches Slack for a human, instead of vanishing with only a log line."""
    subject = thread.get("subject") or "(no subject)"
    msgs = thread.get("messages") or []
    last_from = msgs[-1].get("from") if msgs else "(unknown)"
    return (
        "[Auto-draft unavailable — please write this reply manually.]\n\n"
        f"The assistant could not generate a draft for this thread "
        f"(reason: {type(err).__name__}: {err}).\n"
        f"Subject: {subject}\n"
        f"Latest message from: {last_from}\n"
        f"Messages in thread: {len(msgs)}\n"
    )


def _classify_and_draft(thread: dict) -> tuple[str, dict, str, list[dict]]:
    """Render -> (truncate if oversized) -> classify -> draft.

    Guarantees a postable result: if the thread exceeds the model's context
    budget it is truncated to its most recent messages and the draft notes
    that; if classification/drafting still fails for any reason, a clearly
    marked manual-review fallback is returned so something always reaches Slack.
    Returns (thread_text, classification, draft, flags).
    """
    messages = thread.get("messages") or []
    total = len(messages)
    thread_text = render_thread(thread)

    dropped = 0
    if llm.estimate_tokens(thread_text) > llm.THREAD_TOKEN_BUDGET:
        kept, dropped = llm.select_recent_messages(messages, llm.THREAD_CHAR_BUDGET)
        thread_text = render_thread({**thread, "messages": kept})
        print(f"[truncated] thread too large for model: kept {total - dropped}/{total} most recent messages")

    try:
        classification = llm.classify(thread_text)
        draft = llm.draft_reply(thread_text, classification.get("intent", "other"))
        flags = draft_mod.flag_unverified_specifics(draft)
    except Exception as e:
        # Never drop the thread on a model error — surface a manual-review card.
        print(f"[draft-failed] posting manual-review fallback instead: {type(e).__name__}: {e}")
        classification = {
            "intent": "needs_manual_review",
            "summary": "Automatic drafting failed; this thread needs manual handling.",
            "customer_name": None, "company": None, "key_points": [], "urgency": "normal",
        }
        draft = _fallback_draft_text(thread, e)
        flags = [{"type": "auto_draft_failed",
                  "text": f"Auto-draft failed ({type(e).__name__}). Please draft this one manually."}]
        return thread_text, classification, draft, flags

    if dropped:
        kept_n = total - dropped
        note = (
            f"[Note for reviewer: this thread is very long — the draft below is based on the "
            f"most recent {kept_n} of {total} messages; earlier history was omitted. Please "
            f"verify nothing important from earlier in the thread is missing before sending.]"
        )
        draft = f"{note}\n\n{draft}"
        flags = flags + [{"type": "thread_truncated",
                          "text": f"{dropped} older message(s) omitted to fit the model context "
                                  f"(drafted from the {kept_n} most recent)."}]
    return thread_text, classification, draft, flags


def _rag_query(classification: dict, thread: dict) -> str:
    """Build the retrieval/answer query from the classification (clean intent
    summary + key points), falling back to the latest customer message."""
    parts = []
    if classification.get("summary"):
        parts.append(classification["summary"])
    parts += classification.get("key_points") or []
    if parts:
        return " ".join(parts)
    msgs = thread.get("messages") or []
    return (msgs[-1].get("body") if msgs else "") or thread.get("subject", "")


def process_thread(thread_id: str, my_email: str, service) -> None:
    """Ingest one thread: fetch -> classify -> draft -> persist -> Slack."""
    thread = gmail_client.fetch_thread(thread_id, service=service)

    # Skip threads whose latest message is from us (e.g. our own sent reply) to
    # avoid replying to ourselves / loops.
    last_from = (thread.get("reply_context") or {}).get("to") or ""
    if my_email and my_email.lower() in last_from.lower():
        print(f"[skip] thread {thread_id}: latest message is from self")
        return

    # Slice-2 re-attach hook: if this thread has a SUSPENDED customer debug case,
    # route the reply back into the investigation instead of classifying it as
    # fresh mail. FAIL-OPEN — returns False on ANY error, falling through to the
    # normal path, so a broken hook can never drop mail or wedge ingest.
    thread.setdefault("thread_id", thread_id)
    from app import debug_orchestrator
    if debug_orchestrator.maybe_reattach_debug_case(thread):
        print(f"[reattach] thread {thread_id}: routed into suspended debug case")
        return

    thread_text, classification, draft, flags = _classify_and_draft(thread)

    # Debug auto-trigger (flag-gated, default OFF): a Call/Message UUID (Tier 1)
    # or a diagnosable traffic-failure (Tier 2) routes to the debugging flow — an
    # account-ask in #debugging, NOT the approval gate. The generic safety-net
    # draft above is DISCARDED here (never persisted or posted), so a routed mail
    # never leaves an orphaned second card at the gate. FAIL-OPEN: any error falls
    # through to normal drafting, so the trigger can't drop mail or wedge ingest.
    if os.getenv("DEBUG_AUTODETECT") == "1":
        try:
            if debug_orchestrator.maybe_debug_autotrigger(thread, classification):
                print(f"[debug-autotrigger] thread={thread_id}: routed to debugging flow")
                return
        except Exception:
            import traceback
            print("[debug-autotrigger] FAILED — falling through to normal drafting:")
            traceback.print_exc()

    _route_and_post(thread, thread_id, thread_text, classification, draft, flags, service)


def _route_and_post(thread, thread_id, thread_text, classification, draft, flags, service):
    """Intent routing (meeting / grounded-or-hold / pricing) -> persist -> post ONE
    approval card. Shared by process_thread and the debug 'not a debugging case'
    bounce (draft_and_post_normally), so a bounce lands exactly one normal draft.
    The pre-computed generic `draft` is the safety net kept if a router errors."""
    booking = None
    sched_state = None
    intent = classification.get("intent")
    if intent == "meeting_request":
        try:
            result = scheduling.handle(
                thread, classification,
                db.get_scheduling_state(thread_id),
                calendar_client.now_ist(),
            )
            draft, booking, flags, sched_state = (
                result.draft_text, result.booking, result.flags, result.state)
        except Exception as e:
            print(f"[scheduling-failed] keeping generic meeting draft: {type(e).__name__}: {e}")
    elif intent in _GROUNDED_OR_HOLD_INTENTS:
        # Grounded-or-hold: answer ONLY from retrieved Plivo docs via the layered
        # RAG stack, else a fixed, fact-free holding reply (technical_support's
        # asks for debugging inputs; the rest get a generic confirm-and-follow-up).
        # This is the fabrication gate for every free-write intent — none may
        # assert an ungrounded fact. A hold must ACTUALLY hold: on a RAG error we
        # do NOT fall back to the generic free-write draft (that reopens exactly
        # the leak we're closing) — we post the clean holding reply and flag it
        # for a manual answer.
        try:
            rr = rag.answer(_rag_query(classification, thread),
                            customer_name=classification.get("customer_name"), intent=intent)
            draft, flags = rr.draft_text, rr.flags
            print(f"[rag] thread={thread_id} intent={intent} path={rr.path} top1={rr.top1:.3f} reason={rr.reason[:60]!r}")
        except Exception as e:
            draft = rag.holding_reply(classification.get("customer_name"), intent)
            flags = [{"type": "rag_error_hold",
                      "text": f"RAG failed ({type(e).__name__}); posted a fact-free holding reply "
                              f"instead of an ungrounded draft — needs a manual answer."}]
            print(f"[rag-failed] holding (no free-write fallback): {type(e).__name__}: {e}")
    elif intent == "pricing_question":
        # Exact structured lookup — the rate is NEVER model-generated (replaces
        # the generic free-write that could fabricate a rate). SMS now;
        # voice/WhatsApp/etc. defer to stage 2. Generic draft is the safety net.
        try:
            draft, flags = pricing.handle(thread, classification)
            print(f"[pricing] thread={thread_id} flags={[f['type'] for f in flags]}")
        except Exception as e:
            print(f"[pricing-failed] keeping generic draft: {type(e).__name__}: {e}")
    elif intent == "meeting_followup":
        # The guarded builder is the ONLY entry for meeting_followup. A meeting-
        # notes / follow-up mail that reaches the CLASSIFIER (e.g. auto-generated
        # notes from a sender is_notes_mail() doesn't recognize) must NOT free-write
        # via draft_reply — route it through the SAME commitment + groundedness
        # guards the notes-mail detector uses. Thin/absent notes yield a VISIBLE,
        # un-enriched fallback (never an invented recap); a fabricated commitment to
        # a customer is exactly the failure these guards exist to prevent.
        try:
            from app import meeting_followup
            msgs = thread.get("messages") or []
            notes_body = (msgs[-1].get("body") if msgs else "") or thread_text
            parsed = meeting_followup.parse_notes(thread.get("subject", ""), notes_body)
            draft, flags = meeting_followup.build_followup(parsed, classification.get("customer_name"))
            print(f"[meeting-followup-classified] thread={thread_id} "
                  f"thin={meeting_followup.is_thin(parsed)} flags={[f['type'] for f in flags]}")
        except Exception as e:
            # Never free-write and never crash the thread — a VISIBLE hold.
            draft = ("Hi,\n\nThanks for the time today — I'll follow up with a recap shortly.\n\n"
                     "Best regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo")
            flags = [{"type": "meeting_followup_error",
                      "text": f"meeting_followup builder errored ({type(e).__name__}); no usable notes "
                              f"drafted — HELD for manual follow-up (recap not invented)."}]
            print(f"[meeting-followup-classified] FAILED → held: {type(e).__name__}: {e}")

    ids = db.persist_processing(
        thread, classification, draft,
        source="gmail",
        reply_context=thread.get("reply_context"),
        booking=booking,
    )
    if sched_state is not None:
        db.set_scheduling_state(thread_id, sched_state)
    print(f"[ingested] thread={thread_id} draft_id={ids['draft_id']} intent={classification.get('intent')} flags={len(flags)} booking={bool(booking)}")

    # Post via the shared claim-then-post primitive so a concurrent sweeper can
    # never double-post this card (post_draft_once records slack_ts on success).
    resp = slack_approval.post_draft_once(thread_text, draft, ids["draft_id"], flags=flags, booking=booking)
    if resp:
        print(f"[posted to slack] draft_id={ids['draft_id']} ts={resp.get('ts')}")
    else:
        print(f"[post-skipped] draft_id={ids['draft_id']} — posting already owned (sweeper/other)")


def draft_and_post_normally(thread: dict, service=None) -> None:
    """Normal classify -> route -> post path WITHOUT the debug auto-trigger. Used
    by the debug 'not a debugging case' bounce so the mail lands as an ordinary
    draft; persist_processing's idempotency anchor collapses any replay to ONE."""
    thread_text, classification, draft, flags = _classify_and_draft(thread)
    _route_and_post(thread, thread.get("thread_id"), thread_text, classification,
                    draft, flags, service)


def handle_notification(data: dict, service, my_email: str) -> int:
    """Process new mail since the stored cursor. Returns # threads ingested.

    The stored historyId cursor (Postgres) is authoritative; the notification
    payload's historyId is only a fallback for the very first run. This is why
    a single notification after downtime recovers the whole backlog.
    """
    with _PROCESS_LOCK:
        return _process_since_cursor(data, service, my_email)


def _process_since_cursor(data: dict, service, my_email: str) -> int:
    last = db.kv_get(gmail_client.KV_LAST_HISTORY)
    start_history_id = (last or {}).get("history_id") or str(data.get("historyId"))

    message_ids, latest = gmail_client.list_new_message_ids(start_history_id, service=service)
    if not message_ids:
        # Still advance the cursor so we don't rescan the same window forever.
        if latest:
            db.kv_set(gmail_client.KV_LAST_HISTORY, {"history_id": latest})
        return 0

    from googleapiclient.errors import HttpError

    ingested = 0
    # Reconciliation: if a per-message fetch fails TRANSIENTLY we hold the cursor
    # so this window is re-fetched next round (idempotency_key dedups the retries);
    # 404/permanent skips still advance so a bad message never wedges the pipeline.
    had_transient = False
    # Map messages -> their threads (dedup threads).
    seen_threads: set[str] = set()
    for mid in message_ids:
        try:
            msg = gmail_client.get_message(
                mid, service=service, fmt="metadata",
                metadata_headers=[
                    "From", "Subject", "Reply-To", "Return-Path", "Sender",
                    "Auto-Submitted", "Precedence", "Feedback-ID",
                    "List-Unsubscribe", "List-Id", "X-Auto-Response-Suppress",
                ],
            )
        except HttpError as e:
            status = getattr(getattr(e, "resp", None), "status", None) or getattr(e, "status_code", None)
            if status == 404:
                # Genuinely gone (deleted/expired between listing and fetch) —
                # skip AND let the cursor advance; never wedge on a dead message.
                print(f"[skip] message {mid} not found (404 — deleted/expired); continuing")
            elif gmail_client.is_transient_error(e):
                # Server-side 5xx — hold the cursor, retry this window next round.
                had_transient = True
                print(f"[transient] message {mid} (HTTP {status}); cursor HELD for retry")
            else:
                # Other HTTP (4xx etc.) — permanent; skip AND advance (no wedge).
                print(f"[error] message {mid} (HTTP {status}); skipping (permanent): {e}")
            continue
        except Exception as e:
            if gmail_client.is_transient_error(e):
                # Network blip (BrokenPipe / Errno 49 / timeout) — hold + retry.
                had_transient = True
                print(f"[transient] message {mid}; cursor HELD for retry: {type(e).__name__}")
            else:
                # Unknown/permanent — skip AND advance so one bad message can never
                # wedge the whole pipeline (a stall is worse than losing the one).
                print(f"[error] message {mid}; skipping (permanent): {type(e).__name__}: {e}")
            continue
        thread_id = msg.get("threadId")
        if not thread_id or thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)

        headers = msg.get("payload", {}).get("headers", [])
        from_hdr = next((h["value"] for h in headers if h["name"].lower() == "from"), None)
        subject_hdr = next((h["value"] for h in headers if h["name"].lower() == "subject"), None)

        # Meeting-follow-up carve-out: gemini-notes / notes-failed mail are
        # "automated" and would be [skip-automated] below, so route them FIRST.
        # Flag-gated (default OFF) so auto-detect stays dormant until the invoke-
        # only smoke round-trips; a failure falls through (never wedges ingest).
        if os.getenv("MEETING_FOLLOWUP_AUTODETECT") == "1":
            from app import meeting_followup
            if (meeting_followup.is_notes_mail(from_hdr)
                    or meeting_followup.is_notes_failed_mail(from_hdr, subject_hdr)):
                try:
                    from app import calendar_client
                    art = meeting_followup.handle_gmail_message(
                        mid, from_hdr, subject_hdr, service=service,
                        cal_service=calendar_client.get_service(), my_email=my_email)
                    print(f"[meeting-followup] thread={thread_id} status={art.get('status')}")
                    continue
                except Exception:
                    import traceback
                    print(f"[meeting-followup] FAILED — falling through to normal filter:")
                    traceback.print_exc()

        # Pre-filter: never draft replies to automated / bulk senders.
        skip, reason = gmail_client.is_automated(from_hdr, headers, msg.get("labelIds"))
        if skip:
            print(f"[skip-automated] thread={thread_id} from={from_hdr!r}: {reason}")
            continue

        try:
            process_thread(thread_id, my_email, service)
            ingested += 1
        except Exception:
            # Never fabricate success; log and continue with the next thread.
            import traceback
            print(f"[error] processing thread {thread_id}:")
            traceback.print_exc()

    # Advance the cursor ONLY if no transient skip occurred. A transient hold
    # re-fetches this window next notification (idempotency_key dedups the
    # re-processed successes → no double-draft); a 404/permanent skip still
    # advances, so a bad message never wedges the pipeline.
    if latest and not had_transient:
        db.kv_set(gmail_client.KV_LAST_HISTORY, {"history_id": latest})
    elif had_transient:
        print(f"[cursor-held] transient fetch error this round; not advancing past {start_history_id} (will retry)")
    return ingested


def sweep_orphaned_drafts(grace_seconds: int = 120) -> int:
    """Reconciliation: repost approval cards for drafts whose post never landed
    (slack_ts IS NULL) and that are older than the grace period — e.g. the draft
    saved but the Slack post failed (the orphaned-draft-1032 case). Each repost
    goes through post_draft_once (claim), so it can never double-post a card that
    a normal post is still delivering. Returns the number reposted."""
    from app.cli import render_thread
    reposted = 0
    for row in db.find_unposted_drafts(grace_seconds):
        did = row["draft_id"]
        original = render_thread(row["raw_thread"]) if row.get("raw_thread") else ""
        try:
            resp = slack_approval.post_draft_once(original, row["draft_text"], did)
            if resp:
                reposted += 1
                print(f"[sweep] reposted orphaned draft {did} ts={resp.get('ts')}")
            else:
                print(f"[sweep] draft {did} already owned (normal post in flight) — skipped")
        except Exception as e:
            # post_draft_once released the claim; the next sweep retries promptly.
            print(f"[sweep] repost FAILED for draft {did} (claim released, will retry): {e}")
    return reposted


def catch_up(service, my_email: str) -> None:
    """On startup, pull any mail that arrived while the worker was down.

    Uses the stored history cursor so a restart deterministically drains the
    backlog instead of waiting for the next live notification.
    """
    cursor = db.kv_get(gmail_client.KV_LAST_HISTORY)
    if not (cursor and cursor.get("history_id")):
        print("[startup] no stored history cursor; starting fresh from next notification")
        return
    print(f"[startup] catch-up: pulling mail since historyId={cursor['history_id']}…")
    try:
        n = handle_notification({}, service, my_email)
        print(f"[startup] catch-up complete: {n} thread(s) ingested")
    except Exception as e:
        # Most likely the stored historyId is too old (Gmail 404 after a long
        # outage). We cannot recover that gap; move the cursor forward to the
        # current watch historyId so we resume cleanly, and say so loudly.
        print(f"[startup] catch-up FAILED ({type(e).__name__}: {e}).")
        watch = db.kv_get(gmail_client.KV_WATCH)
        if watch and watch.get("history_id"):
            db.kv_set(gmail_client.KV_LAST_HISTORY, {"history_id": watch["history_id"]})
            print("[startup] cursor advanced to current historyId — mail during the "
                  "outage gap was NOT recovered (outage likely exceeded Gmail's "
                  "history window). Newer mail will process normally.")


def run() -> None:
    if not SUBSCRIPTION:
        raise RuntimeError("GMAIL_PUBSUB_SUBSCRIPTION is not set (see .env).")
    from google.cloud import pubsub_v1

    db.init_db()
    service = gmail_client.get_service()
    my_email = gmail_client.get_profile_email(service)
    print(f"Worker authorized as {my_email}")

    gmail_client.ensure_watch(service, now_ms=_now_ms())
    last_watch_check = time.time()

    # Drain anything that arrived while we were down, BEFORE going live, so the
    # backlog pull can't race with incoming callbacks.
    catch_up(service, my_email)

    # Reconcile any drafts whose approval card never posted (orphaned) — once now,
    # then on the SWEEP_INTERVAL_S cadence in the loop below.
    try:
        print(f"[sweep] startup pass: {sweep_orphaned_drafts()} reposted")
    except Exception:
        import traceback
        traceback.print_exc()
    last_sweep = time.time()

    subscriber = pubsub_v1.SubscriberClient()

    def callback(message) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
            handle_notification(payload, service, my_email)
            message.ack()
        except Exception:
            import traceback
            traceback.print_exc()
            message.nack()

    # Process one notification at a time. Gmail calls are also lock-serialized
    # in gmail_client (httplib2 is not thread-safe); this avoids piling up
    # callback threads that would all block on that lock.
    flow_control = pubsub_v1.types.FlowControl(max_messages=1)
    future = subscriber.subscribe(SUBSCRIPTION, callback=callback, flow_control=flow_control)
    print(f"Listening on {SUBSCRIPTION} (Ctrl+C to stop)…")
    try:
        while True:
            time.sleep(30)
            if time.time() - last_watch_check >= WATCH_CHECK_INTERVAL_S:
                gmail_client.ensure_watch(service, now_ms=_now_ms())
                last_watch_check = time.time()
            if time.time() - last_sweep >= SWEEP_INTERVAL_S:
                try:
                    print(f"[sweep] pass complete: {sweep_orphaned_drafts()} reposted")
                except Exception:
                    import traceback
                    traceback.print_exc()
                last_sweep = time.time()
    except KeyboardInterrupt:
        print("Shutting down…")
        future.cancel()


if __name__ == "__main__":
    run()
