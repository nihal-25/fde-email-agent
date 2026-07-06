"""Slack approval over Socket Mode (Milestone 4).

Posts a drafted reply to a Slack channel as a Block Kit message showing the
original email thread + the draft, with Approve / Edit / Reject buttons, and
handles the button callbacks over Socket Mode (an outbound WebSocket — no
public inbound endpoint, per CLAUDE.md).

What the buttons do in Milestone 4:
- Approve -> mark the draft APPROVED in the DB and print "would send". It does
  NOT send anything. Real sending arrives in Milestone 5.
- Edit    -> open a modal pre-filled with the draft; on submit, store the
  edited text and mark the draft EDITED.
- Reject  -> mark the draft REJECTED.

Two entry points:
- post_draft_for_approval(...)  — called by the CLI/worker to post a draft.
- run_socket_mode()             — long-running listener for button callbacks
  (`python -m app.slack_approval`). The listener must be running for the
  buttons to do anything.
"""

from __future__ import annotations

import json
import os
import threading
import time

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app import db

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_APPROVAL_CHANNEL = os.getenv("SLACK_APPROVAL_CHANNEL")
# Nihal's Slack user id — the ONLY user whose #verify thread reply resumes a
# debug case. If unset, we fall back to "any human (non-bot) reply in the case
# thread" (still excludes the bot and the reviewer, which post as apps/bots).
SLACK_NIHAL_USER_ID = os.getenv("SLACK_NIHAL_USER_ID")
# Periodic pending-case catch-up cadence — covers mid-run socket flaps that a
# startup-only scan would miss (the atomic claim in resume() makes each tick safe).
CATCHUP_INTERVAL_S = int(os.getenv("CATCHUP_INTERVAL_S", "120"))

# Slack text objects cap at 3000 chars; leave headroom for code fences.
_MAX_SECTION_CHARS = 2800

# action_ids — the draft id rides in each button's `value`.
ACTION_APPROVE = "draft_approve"
ACTION_EDIT = "draft_edit"
ACTION_REJECT = "draft_reject"
EDIT_MODAL_CALLBACK = "draft_edit_submit"
EDIT_INPUT_BLOCK = "edit_block"
EDIT_INPUT_ACTION = "edit_input"

# Edit-learning loop — #learning ratify/reject cards. Style ratify rides the
# rule_id in `value`; fact ratify rides {fact_text, origin_draft_id} as JSON.
SLACK_LEARNING_CHANNEL = os.getenv("SLACK_LEARNING_CHANNEL")
ACTION_STYLE_RATIFY = "learn_style_ratify"
ACTION_STYLE_REJECT = "learn_style_reject"
ACTION_FACT_RATIFY = "learn_fact_ratify"
ACTION_FACT_REJECT = "learn_fact_reject"


def _truncate(text: str, limit: int = _MAX_SECTION_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…(truncated)…"


def _format_flags_block(flags: list[dict] | None) -> dict | None:
    """A Slack section warning the reviewer about unverified specifics."""
    if not flags:
        return None
    lines = ["⚠️ *Unverified specifics — please verify each is real:*"]
    for f in flags[:12]:
        lines.append(f"• `{f['type']}`: {f['text']}")
    if len(flags) > 12:
        lines.append(f"• …and {len(flags) - 12} more")
    return {"type": "section", "text": {"type": "mrkdwn", "text": _truncate("\n".join(lines))}}


def _format_booking_block(booking: dict | None) -> dict | None:
    """A Slack section spelling out the calendar booking the approval performs,
    so the human approves the actual booking, not just the reply text."""
    if not booking:
        return None
    label = booking.get("label") or f"{booking.get('start')} – {booking.get('end')}"
    attendee = booking.get("attendee_email", "(unknown)")
    verb = "Will reschedule to" if booking.get("event_id") else "Will book on approval:"
    return {"type": "section", "text": {"type": "mrkdwn",
            "text": f"📅 *{verb}* {label} · invite to `{attendee}`"}}


def build_blocks(original_email: str, draft_text: str, draft_id: int,
                 *, status_note: str | None = None,
                 flags: list[dict] | None = None,
                 booking: dict | None = None) -> list[dict]:
    """Build the Block Kit message for a draft awaiting approval."""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📧 Draft reply needs approval"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original thread:*\n```{_truncate(original_email)}```"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Proposed reply:*\n```{_truncate(draft_text)}```"},
        },
    ]
    booking_block = _format_booking_block(booking)
    if booking_block:
        blocks.append(booking_block)
    flags_block = _format_flags_block(flags)
    if flags_block:
        blocks.append(flags_block)
    if status_note:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": status_note}],
        })
    else:
        blocks.append({
            "type": "actions",
            "block_id": f"draft_actions_{draft_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "style": "primary",
                    "action_id": ACTION_APPROVE,
                    "value": str(draft_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️ Edit"},
                    "action_id": ACTION_EDIT,
                    "value": str(draft_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject"},
                    "style": "danger",
                    "action_id": ACTION_REJECT,
                    "value": str(draft_id),
                },
            ],
        })
    return blocks


# --- Posting (called by CLI / worker) ---------------------------------------

def post_draft_for_approval(original_email: str, draft_text: str, draft_id: int,
                            *, channel: str | None = None,
                            flags: list[dict] | None = None,
                            booking: dict | None = None) -> dict:
    """Post a draft to the approval channel. Returns the Slack API response."""
    if not SLACK_BOT_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN is not set (see .env).")
    channel = channel or SLACK_APPROVAL_CHANNEL
    if not channel:
        raise RuntimeError("SLACK_APPROVAL_CHANNEL is not set (see .env).")

    # A standalone WebClient so posting works without starting the listener.
    from slack_sdk import WebClient

    client = WebClient(token=SLACK_BOT_TOKEN)
    resp = client.chat_postMessage(
        channel=channel,
        text="Draft reply needs approval",  # notification fallback
        blocks=build_blocks(original_email, draft_text, draft_id, flags=flags, booking=booking),
    )
    return resp.data


def post_draft_once(original_email: str, draft_text: str, draft_id: int,
                    *, flags: list[dict] | None = None, booking: dict | None = None) -> dict | None:
    """Claim-then-post: post the approval card at most once, whoever calls.

    Both the normal ingest path AND the orphaned-draft sweeper call this, so they
    can't both post the same card: each must first win `claim_draft_for_posting`.
    The winner posts (one chat_postMessage) and records slack_ts; the loser stands
    down and returns None. If the winner's post FAILS, the claim is released so the
    next attempt (e.g. the next sweep) can re-claim immediately rather than waiting
    out the lease.
    """
    if not db.claim_draft_for_posting(draft_id):
        return None  # another poster/sweeper owns it, or it's already posted
    try:
        resp = post_draft_for_approval(original_email, draft_text, draft_id,
                                       flags=flags, booking=booking)
    except Exception:
        db.release_post_claim(draft_id)  # failed -> let the next attempt re-claim now
        raise
    db.set_slack_message(draft_id, resp.get("channel"), resp.get("ts"))
    return resp


# --- Approval actions (booking + send) --------------------------------------

def _book_and_send(draft_id: int, effective_text: str, thread_id,
                   ctx: dict | None, *, gmail=None, calendar=None) -> str:
    """Run the actions an approval authorizes, in order, and return a status note.

    If the draft carries a booking, create the calendar event + invite FIRST
    (Google emails the attendee); if that fails, the confirmation email is NOT
    sent — we must never promise an invite we didn't create. Then send the reply
    (when there's Gmail reply context). Pure of Slack so it is unit-testable;
    gmail/calendar are injectable.
    """
    ctx = ctx or {}
    reply_context = ctx.get("reply_context")
    booking = ctx.get("booking")

    booking_note = None
    if booking:
        if calendar is None:
            from app import calendar_client as calendar
        from datetime import datetime
        start = datetime.fromisoformat(booking["start"])
        end = datetime.fromisoformat(booking["end"])
        existing_event = booking.get("event_id")
        try:
            if existing_event:
                # A meeting already exists on this thread -> move it, don't dup.
                ev = calendar.update_event(existing_event, start, end)
                verb = "Rescheduled"
            else:
                ev = calendar.create_event(
                    booking["title"], start, end, booking["attendee_email"],
                    description=f"Scheduled via email with {booking['attendee_email']}.",
                )
                verb = "Booked"
        except Exception as e:  # never fabricate success; abort BEFORE the email
            print(f"[BOOKING-FAILED] draft_id={draft_id}: {e}")
            return (f"⚠️ BOOKING FAILED: `{e}` — calendar not updated and confirmation "
                    f"email NOT sent. Draft remains approved.")
        db.record_booking(draft_id, ev)
        # Remember the event on the thread so a later change reschedules THIS one.
        st = db.get_scheduling_state(thread_id) or {}
        st["booked_event_id"] = ev.get("event_id")
        st["stage"] = "booked"
        db.set_scheduling_state(thread_id, st)
        meet = ev.get("meet_link")
        booking_note = (f"📅 {verb} {booking.get('label')} · invite to "
                        f"{booking['attendee_email']}"
                        + (f" · Meet: {meet}" if meet else " · ⚠️ no Meet link returned")
                        + f" · event `{ev.get('event_id')}`")
        print(f"[{verb.upper()}] draft_id={draft_id} event_id={ev.get('event_id')} meet={meet} attendee={booking['attendee_email']}")

    if reply_context and reply_context.get("to"):
        if gmail is None:
            from app import gmail_client as gmail
        try:
            sent = gmail.send_reply(reply_context, effective_text)
            db.mark_draft_sent(draft_id, effective_text, sent.get("id"))
            sent_note = f"📤 Sent to {reply_context['to']} · gmail_id `{sent.get('id')}` · status `sent`"
            print(f"[SENT] draft_id={draft_id} gmail_id={sent.get('id')} to={reply_context['to']}")
        except Exception as e:  # never fabricate success; surface the failure
            print(f"[SEND-FAILED] draft_id={draft_id}: {e}")
            sent_note = f"⚠️ Approved, but SEND FAILED: `{e}` — draft remains approved, not sent."
    else:
        print("=" * 60)
        print(f"[WOULD SEND] draft_id={draft_id} thread_id={thread_id} (no Gmail reply context)")
        print("--- text that would be sent ---")
        print(effective_text)
        print("=" * 60)
        sent_note = "_not sent — no Gmail reply context (CLI-sourced draft)_"

    return "\n".join(x for x in [booking_note, sent_note] if x)


# --- Socket Mode listener (handles button callbacks) ------------------------

def _evidence_block(evidence: list[dict]) -> list[dict]:
    """Inline before/after diff snippets from the cited drafts — Nihal ratifies an
    inference against its source, never a bare distilled rule."""
    out = []
    for ev in (evidence or [])[:3]:
        o = (ev.get("original") or "")[:500]
        e = (ev.get("edited") or "")[:500]
        out.append({"type": "section", "text": {"type": "mrkdwn",
                    "text": f"*Evidence — draft {ev.get('draft_id')}*\n"
                            f"• _before:_ {o}\n• _after:_ {e}"}})
    return out


def build_style_card_blocks(cand: dict) -> list[dict]:
    """Style-rule ratify card: the rule, its SCOPE, inline evidence, Ratify/Reject."""
    rid = cand.get("rule_id")
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn",
         "text": f"*📐 Style rule candidate*  ·  scope: `{cand.get('scope')}`\n"
                 f">{cand.get('rule_text')}"}},
        *_evidence_block(cand.get("evidence")),
        {"type": "actions", "elements": [
            {"type": "button", "style": "primary", "text": {"type": "plain_text", "text": "Ratify"},
             "action_id": ACTION_STYLE_RATIFY, "value": str(rid)},
            {"type": "button", "style": "danger", "text": {"type": "plain_text", "text": "Reject"},
             "action_id": ACTION_STYLE_REJECT, "value": str(rid)},
        ]},
    ]
    return blocks


def build_fact_card_blocks(cand: dict) -> list[dict]:
    """Fact ratify card: the VERBATIM text that will be embedded + cited (not a
    summary), inline evidence, Ratify/Reject."""
    payload = json.dumps({"fact_text": cand.get("fact_text"),
                          "origin_draft_id": cand.get("origin_draft_id")})[:1900]
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn",
         "text": "*📎 Fact candidate for the knowledge base*\n"
                 "_This exact text will be embedded and cited by future drafts:_\n"
                 f"```{cand.get('fact_text')}```"}},
        *_evidence_block(cand.get("evidence")),
        {"type": "actions", "elements": [
            {"type": "button", "style": "primary", "text": {"type": "plain_text", "text": "Ratify"},
             "action_id": ACTION_FACT_RATIFY, "value": payload},
            {"type": "button", "style": "danger", "text": {"type": "plain_text", "text": "Reject"},
             "action_id": ACTION_FACT_REJECT, "value": "reject"},
        ]},
    ]
    return blocks


def post_lesson_cards(result: dict) -> int:
    """Publish style + fact candidates to #learning for ratification. Returns count."""
    if not SLACK_LEARNING_CHANNEL:
        raise RuntimeError("SLACK_LEARNING_CHANNEL is not set (see .env).")
    from slack_sdk import WebClient
    client = WebClient(token=SLACK_BOT_TOKEN)
    n = 0
    for cand in result.get("style_candidates", []):
        client.chat_postMessage(channel=SLACK_LEARNING_CHANNEL, text="Style rule candidate",
                                blocks=build_style_card_blocks(cand)); n += 1
    for cand in result.get("fact_candidates", []):
        client.chat_postMessage(channel=SLACK_LEARNING_CHANNEL, text="Fact candidate",
                                blocks=build_fact_card_blocks(cand)); n += 1
    return n


def build_app() -> App:
    if not SLACK_BOT_TOKEN:
        raise RuntimeError("SLACK_BOT_TOKEN is not set (see .env).")
    app = App(token=SLACK_BOT_TOKEN)

    def _decision_note(actor_id: str, label: str) -> str:
        return f"*{label}* by <@{actor_id}>"

    def _render_finalized(client, channel_id, message_ts, draft_id, status):
        """Re-render a card whose draft is already finalized: show the real
        status and drop the (now stale) action buttons."""
        view = db.get_draft_view(draft_id)
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="Draft already finalized",
            blocks=build_blocks(
                _original_for(view),
                (view or {}).get("effective_text", "") if view else "",
                draft_id,
                status_note=(
                    f"⚠️ This draft is already *{status}* — no action taken "
                    f"(stale button)."
                ),
            ),
        )

    def _finalize_lesson_card(client, body, note: str):
        # Replace the card's buttons with a status line so a lesson can't be
        # double-ratified from a stale card.
        blocks = [b for b in body["message"]["blocks"] if b.get("type") != "actions"]
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": note}]})
        client.chat_update(channel=body["channel"]["id"], ts=body["message"]["ts"],
                           text="lesson decided", blocks=blocks)

    @app.action(ACTION_STYLE_RATIFY)
    def handle_style_ratify(ack, body, client, logger):
        ack()
        from app import learn
        rid = int(body["actions"][0]["value"]); user = body["user"]["id"]
        ok = learn.ratify_style(rid)
        _finalize_lesson_card(client, body, f"✅ Ratified rule #{rid} by <@{user}> — now in drafting prompts.")

    @app.action(ACTION_STYLE_REJECT)
    def handle_style_reject(ack, body, client, logger):
        ack()
        from app import learn
        rid = int(body["actions"][0]["value"]); user = body["user"]["id"]
        learn.reject_style(rid)
        _finalize_lesson_card(client, body, f"🚫 Rejected rule #{rid} by <@{user}> — not applied.")

    @app.action(ACTION_FACT_RATIFY)
    def handle_fact_ratify(ack, body, client, logger):
        ack()
        from app import learn
        user = body["user"]["id"]
        payload = json.loads(body["actions"][0]["value"])
        cid = learn.ratify_fact(payload["fact_text"], payload.get("origin_draft_id"))
        _finalize_lesson_card(client, body, f"✅ Ratified fact by <@{user}> — added to the corpus (chunk #{cid}, citable).")

    @app.action(ACTION_FACT_REJECT)
    def handle_fact_reject(ack, body, client, logger):
        ack()
        user = body["user"]["id"]
        _finalize_lesson_card(client, body, f"🚫 Rejected fact by <@{user}> — not added to the corpus.")

    @app.action(ACTION_APPROVE)
    def handle_approve(ack, body, client, logger):
        ack()
        draft_id = int(body["actions"][0]["value"])
        user = body["user"]["id"]
        # Step 1: record the human approval (the only path that authorizes a send).
        try:
            result = db.record_human_decision(draft_id, "approve", actor=f"slack:{user}")
        except db.DecisionConflict as conflict:
            # A stale Approve on a rejected/sent draft must NOT send anything.
            _render_finalized(client, body["channel"]["id"], body["message"]["ts"],
                              draft_id, conflict.current_status)
            return

        # Step 2: perform the approved actions (booking-then-send) and report.
        ctx = db.get_send_context(draft_id)
        note = _book_and_send(draft_id, result["effective_text"], result["thread_id"], ctx)
        view = db.get_draft_view(draft_id)
        client.chat_update(
            channel=body["channel"]["id"],
            ts=body["message"]["ts"],
            text="Draft approved",
            blocks=build_blocks(
                _original_for(view), result["effective_text"], draft_id,
                status_note=f"✅ {_decision_note(user, 'Approved')}\n{note}",
                booking=(ctx or {}).get("booking"),
            ),
        )

    @app.action(ACTION_REJECT)
    def handle_reject(ack, body, client, logger):
        ack()
        draft_id = int(body["actions"][0]["value"])
        user = body["user"]["id"]
        try:
            result = db.record_human_decision(draft_id, "reject", actor=f"slack:{user}")
        except db.DecisionConflict as conflict:
            _render_finalized(client, body["channel"]["id"], body["message"]["ts"],
                              draft_id, conflict.current_status)
            return
        view = db.get_draft_view(draft_id)
        client.chat_update(
            channel=body["channel"]["id"],
            ts=body["message"]["ts"],
            text="Draft rejected",
            blocks=build_blocks(
                _original_for(view), result["effective_text"], draft_id,
                status_note=f"❌ {_decision_note(user, 'Rejected')} · status `rejected`",
            ),
        )

    @app.action(ACTION_EDIT)
    def handle_edit(ack, body, client, logger):
        ack()
        draft_id = int(body["actions"][0]["value"])
        view = db.get_draft_view(draft_id)
        # Don't let a finalized draft be edited back into play.
        if view and view["status"] in db._TERMINAL_STATUSES:
            _render_finalized(client, body["channel"]["id"], body["message"]["ts"],
                              draft_id, view["status"])
            return
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": EDIT_MODAL_CALLBACK,
                # Carry the context the submit handler needs to update the message.
                "private_metadata": f"{draft_id}|{body['channel']['id']}|{body['message']['ts']}",
                "title": {"type": "plain_text", "text": "Edit draft"},
                "submit": {"type": "plain_text", "text": "Save"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": EDIT_INPUT_BLOCK,
                        "label": {"type": "plain_text", "text": "Reply body"},
                        "element": {
                            "type": "plain_text_input",
                            "action_id": EDIT_INPUT_ACTION,
                            "multiline": True,
                            "initial_value": view["effective_text"] if view else "",
                        },
                    }
                ],
            },
        )

    @app.event("message")
    def handle_debug_reply(event, logger):
        """Resume a pending debug case ONLY when Nihal replies in that case's
        thread — the reviewer-verify (#verify → resume) OR account-ask (#debugging →
        resume_account). Guards (all required): right channel, a thread reply
        whose ts matches a pending case OF THAT KIND, a human message (no
        bot_id/subtype — excludes the bot and the reviewer), and (when configured) from
        Nihal's user id. Cannot be triggered by the bot's own post or any other
        thread."""
        from app import debug_orchestrator as orch
        ch, thread_ts = event.get("channel"), event.get("thread_ts")
        if not thread_ts:
            return
        # Cheap SILENT filters: identify which kind of pending case (if any).
        if ch == orch.VERIFY_CHANNEL and db.kv_get(f"{orch._PENDING_PREFIX}{thread_ts}"):
            kind, resume_fn = "verify", orch.resume
        elif ch == orch.DEBUG_CHANNEL and db.kv_get(f"{orch._ACCT_PREFIX}{thread_ts}"):
            kind, resume_fn = "acct", orch.resume_account
        else:
            return  # not a pending thread we track
        # In a pending thread — from here LOG any skip so a non-firing resume is
        # never silent (that silence cost an hour once).
        if event.get("bot_id") or event.get("subtype"):
            print(f"[debug-skip] {kind} thread={thread_ts}: not a human reply "
                  f"(bot_id={event.get('bot_id')} subtype={event.get('subtype')})")
            return
        if SLACK_NIHAL_USER_ID and event.get("user") != SLACK_NIHAL_USER_ID:
            print(f"[debug-skip] {kind} thread={thread_ts}: user {event.get('user')} "
                  f"!= Nihal {SLACK_NIHAL_USER_ID} — ignored")
            return
        try:
            res = resume_fn(thread_ts, event.get("text", "")) or {}
            print(f"[debug-resume] {kind} thread={thread_ts} user={event.get('user')} "
                  f"result={ {k: res.get(k) for k in ('branch', 'draft_id', 'verify_ts')} }")
        except Exception:
            logger.exception("debug-case resume failed")

    @app.view(EDIT_MODAL_CALLBACK)
    def handle_edit_submit(ack, body, client, view, logger):
        ack()
        user = body["user"]["id"]
        draft_id_s, channel_id, message_ts = view["private_metadata"].split("|")
        draft_id = int(draft_id_s)
        new_text = view["state"]["values"][EDIT_INPUT_BLOCK][EDIT_INPUT_ACTION]["value"]
        try:
            result = db.record_human_decision(
                draft_id, "edit", actor=f"slack:{user}", edited_text=new_text
            )
        except db.DecisionConflict as conflict:
            _render_finalized(client, channel_id, message_ts, draft_id, conflict.current_status)
            return
        dview = db.get_draft_view(draft_id)
        # Re-post the original action buttons so the human can now approve the edit.
        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="Draft edited",
            blocks=build_blocks(_original_for(dview), result["effective_text"], draft_id),
        )

    return app


def _original_for(view: dict | None) -> str:
    """Best-effort original-thread text for re-rendering an updated message.

    The button payload doesn't carry the original thread, so we note that the
    full thread lives in the audit log rather than fabricating it here.
    """
    if not view:
        return "(thread unavailable)"
    return f"(subject: {view.get('subject') or 'n/a'} · thread_id: {view.get('thread_id')})"


def run_socket_mode() -> None:
    if not SLACK_APP_TOKEN:
        raise RuntimeError("SLACK_APP_TOKEN is not set (needs Socket Mode; see .env).")
    import logging
    logging.basicConfig(level=getattr(logging, os.getenv("SLACK_LOG_LEVEL", "INFO").upper(), logging.INFO))
    app = build_app()
    # Reconciliation: pick up #verify replies that landed while we were
    # disconnected (bounded scan). Safe vs. live events — resume() claims the
    # case atomically, so a reply caught here AND re-fired live resumes once.
    try:
        from app import debug_orchestrator as orch
        n = orch.catch_up_pending_cases()
        if n:
            print(f"[startup] catch-up resumed {n} pending debug case(s)")
    except Exception as e:
        print(f"[startup] pending-case catch-up skipped: {type(e).__name__}: {e}")

    # Periodic reconciliation: re-scan pending cases for #verify replies missed
    # during a mid-run socket flap (startup-only doesn't cover a flap). Each tick
    # is safe vs. live events — resume() claims the case atomically.
    def _periodic_catchup():
        while True:
            time.sleep(CATCHUP_INTERVAL_S)
            try:
                from app import debug_orchestrator as orch
                orch.catch_up_pending_cases()
            except Exception as e:
                print(f"[catch-up] periodic scan error: {type(e).__name__}: {e}")
    threading.Thread(target=_periodic_catchup, daemon=True, name="periodic-catchup").start()
    print(f"[startup] periodic catch-up armed (every {CATCHUP_INTERVAL_S}s)")

    print("Starting Slack Socket Mode listener (Ctrl+C to stop)…")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()


if __name__ == "__main__":
    run_socket_mode()
