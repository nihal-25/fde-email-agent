"""Phase 4 — Slice 1 debugging orchestrator (voice/sip_trunk, Call UUID given).

The model COMPOSES over the validated, account-scoped tools in
app/redshift_tools.py — it never writes SQL and never sees/sets account_id.
Flow (Slice 1, UUID + account already known, so no account-id ask):

  1. detect a debugging mail carrying a Call UUID
  2. investigate via the scoped tools (model decides which/params)
  3. post to #debugging: GROUNDED FACTS (from tools) + INTERPRETATION (model's
     read) — labeled separately; report-only, does not wait
  4. post a targeted verify-prompt to #verify (leads only, no findings dump),
     then WAIT for Nihal to paste the reviewer's reply
  5. on the reply: final findings -> #debugging
  6. draft the customer mail to the EXISTING approval channel — the ONLY
     human-approval gate; internal fields stripped at THIS stage only

Hard rules: account scope is the tools' job (never bypassed — account_id is
injected here, never exposed to the model); facts vs interpretation always
labeled; interpretation never reaches the customer without the the reviewer step + the
approval gate; "couldn't resolve -> say so in #debugging, draft nothing" is a
valid terminal state.
"""

from __future__ import annotations

import os
import re

from app import db, llm, redshift_tools

DEBUG_CHANNEL = os.getenv("SLACK_DEBUG_CHANNEL", "C0BE57VLGH0")   # #debugging
VERIFY_CHANNEL = os.getenv("SLACK_VERIFY_CHANNEL", "C0BE357F9TQ")   # #verify
APPROVAL_CHANNEL = os.getenv("SLACK_APPROVAL_CHANNEL")            # existing gate

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)

# --- model-facing tool schemas (NOTE: no account_id — injected by dispatch) ---
_TOOLS = [
    {"type": "function", "function": {
        "name": "get_call_by_uuid",
        "description": "Voice-API CDR row for a call UUID (call_state, disconnect_reason, "
                       "direction, timestamps, carrier). Returns found:false if the UUID "
                       "is not a voice-API call for this account.",
        "parameters": {"type": "object", "properties": {"uuid": {"type": "string"}},
                       "required": ["uuid"]}}},
    {"type": "function", "function": {
        "name": "get_sip_trunk_call_detail",
        "description": "SIP trunk (SIP trunking) CDR detail for a call UUID: disconnect_reason, "
                       "disconnect_code, sip_code/sip_response, route_type, trunk routing, "
                       "timestamps. Use for SIP-trunk calls.",
        "parameters": {"type": "object", "properties": {"uuid": {"type": "string"}},
                       "required": ["uuid"]}}},
    {"type": "function", "function": {
        "name": "get_quality_metrics",
        "description": "Call-quality metrics (MOS, jitter, packet loss, suspected issues) "
                       "for a call UUID. Works for voice and sip_trunk calls.",
        "parameters": {"type": "object", "properties": {"uuid": {"type": "string"}},
                       "required": ["uuid"]}}},
    {"type": "function", "function": {
        "name": "get_calls_for_account",
        "description": "List recent calls in a time window for context (e.g. are other "
                       "calls failing the same way?). channel is 'voice' or 'sip_trunk'.",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
            "end": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
            "channel": {"type": "string", "enum": ["voice", "sip_trunk"]},
            "direction": {"type": "string"}, "disconnect_reason": {"type": "string"}},
            "required": ["start", "end", "channel"]}}},
]

_INVESTIGATE_SYSTEM = """\
You are investigating ONE customer call issue for a Plivo FDE. You have \
read-only tools that are ALREADY scoped to this customer's account — you cannot \
and must not specify an account; scoping is enforced for you.

Given a Call UUID, investigate thoroughly: a call is either a Voice-API call or \
a SIP trunk (SIP trunking) call, so check both get_call_by_uuid AND \
get_sip_trunk_call_detail, and pull get_quality_metrics. Optionally use \
get_calls_for_account for context.

If a tool returns found:false with reason "not found for this account", that \
call is simply not visible for THIS customer's account — do NOT speculate about \
it, do NOT try to look it up another way, and never imply it belongs elsewhere. \
Just note it.

Do not write SQL. Only the tool results are real — never invent field values. \
When you have gathered what the tools can show, stop and give a one-line summary."""


def _dispatch(account_id):
    """Return a dispatch(name, args) that injects the VERIFIED account_id.

    account_id comes from our verified-sender mapping, never from the model or
    the email. Any account_id in `args` is ignored.
    """
    def run(name: str, args: dict):
        uuid = args.get("uuid")
        try:
            if name == "get_call_by_uuid":
                return redshift_tools.get_call_by_uuid(uuid, account_id)
            if name == "get_sip_trunk_call_detail":
                return redshift_tools.get_sip_trunk_call_detail(uuid, account_id)
            if name == "get_quality_metrics":
                return redshift_tools.get_quality_metrics(uuid, account_id)
            if name == "get_calls_for_account":
                return redshift_tools.get_calls_for_account(
                    account_id, args.get("start"), args.get("end"),
                    channel=args.get("channel"), direction=args.get("direction"),
                    disconnect_reason=args.get("disconnect_reason"), limit=args.get("limit", 20))
        except Exception as e:  # never crash the loop on a bad tool call
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown tool {name}"}
    return run


# --- fact rendering (from tool RESULTS — code, never model prose) ------------
_SKIP_FIELDS = {"found", "channel", "source_table", "carrier_details", "raw_media_stats",
                "per_leg", "_gaps", "endpoint_config", "answer_url"}
# UUID-lookups that locate the SPECIFIC call (vs get_calls_for_account, context only).
_UUID_TOOLS = {"get_call_by_uuid", "get_sip_trunk_call_detail", "get_quality_metrics"}
_TOOL_CHANNEL = {"get_call_by_uuid": "voice CDR", "get_sip_trunk_call_detail": "sip_trunk",
                 "get_quality_metrics": "quality data"}


def _render_facts(trace: list[dict]) -> tuple[str, bool]:
    """Render GROUNDED FACTS from the tool trace.

    Returns (text, call_found). `call_found` is True only if a UUID-lookup
    positively located the SPECIFIC call (context rows from get_calls_for_account
    do not, on their own, make a case resolvable).

    A missed UUID-lookup is relabeled ("checked X — not there; found in Y") ONLY
    when a sibling lookup positively located the call — i.e. driven by the
    positive hit, NOT by reinterpreting the tool's flat "not found for this
    account" message. When nothing hit (genuine wrong-account / no data), the
    flat message is kept verbatim so existence is never disclosed.
    """
    hit_channels = sorted({s["result"].get("channel") for s in trace
                           if isinstance(s["result"], dict)
                           and s["result"].get("found") is True and s["result"].get("channel")})
    call_found = any(isinstance(s["result"], dict) and s["result"].get("found") is True
                     for s in trace if s["tool"] in _UUID_TOOLS)
    lines: list[str] = []
    for step in trace:
        tool, res = step["tool"], step["result"]
        if isinstance(res, list):  # get_calls_for_account — context only
            lines.append(f"• {tool}: {len(res)} call(s) in window (context)")
            continue
        if not isinstance(res, dict):
            continue
        if res.get("found") is False:
            if hit_channels:  # safe relabel — driven by the positive sibling hit
                lines.append(f"• checked {_TOOL_CHANNEL.get(tool, tool)} — not there; "
                             f"call found in {', '.join(hit_channels)}")
            else:             # nothing hit — preserve the flat, ambiguous message
                lines.append(f"• {tool}({step['args'].get('uuid', '')}): "
                             f"{res.get('reason', 'not found')}")
            continue
        if res.get("found") is True or tool == "get_calls_for_account":
            label = f"{tool} [{res.get('source_table', '')}]".strip()
            lines.append(f"• {label}:")
            for k, v in res.items():
                if k in _SKIP_FIELDS or v in (None, "", "null"):
                    continue
                lines.append(f"    {k} = {v}")
            if res.get("_gaps"):  # surface what's NOT in Redshift
                lines.append(f"    (gap) {res['_gaps']}")
    return ("\n".join(lines) if lines else "(no data returned by the tools)"), call_found


# --- internal-field stripping (customer draft ONLY; backstop) ----------------
def _strip_internal(text: str, account_id) -> str:
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[internal-ip]", text)
    text = re.sub(r"\b\S*\.zt\.plivo\.com\b", "[internal-trunk]", text, flags=re.I)
    text = re.sub(r"\b\S*\.plivops\.com\b", "[internal-host]", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{32}\b", "[hash]", text, flags=re.I)
    text = re.sub(rf"\b{re.escape(str(account_id))}\b", "[account]", text)
    return text


def _sender(thread: dict) -> str:
    msgs = thread.get("messages") or [{}]
    return msgs[-1].get("from", "") or ""


def _email_text(thread: dict) -> str:
    msgs = thread.get("messages") or []
    return "\n\n".join((m.get("body") or "") for m in msgs).strip()


# --- live Slack posting + pending-case store ---------------------------------
_PENDING_PREFIX = "debugcase:"  # kv_state key: debugcase:<#verify thread ts>


def _post(channel: str, text: str, thread_ts: str | None = None) -> dict:
    """Post a plain mrkdwn message to a channel (live). Returns the API response."""
    from slack_sdk import WebClient
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set (see .env).")
    resp = WebClient(token=token).chat_postMessage(
        channel=channel, text=text[:39000], thread_ts=thread_ts)
    return resp.data


def _finalize(email_text: str, facts: str, interpretation: str, case_title: str,
              account_id, verify_reply: str) -> tuple[str, str]:
    """Shared final step: integrate the reviewer's reply -> (final #debugging post,
    stripped customer draft). Used by both the dry-run simulation and live resume."""
    final = llm.debug_final_findings(facts, interpretation, verify_reply, email_text)
    final_post = (
        f"*{case_title} — final findings*\n\n"
        f"*Grounded facts (from internal tools):*\n{facts}\n\n"
        f"*Confirmed (with the reviewer's verification):*\n{final.get('final_interpretation', '')}")
    draft = _strip_internal(llm.debug_customer_draft(
        final.get("customer_safe_explanation", ""), email_text), account_id)
    return final_post, draft


# --- the orchestrator --------------------------------------------------------
def run_case(thread: dict, account_id, *, dry_run: bool = True, verify_reply: str | None = None) -> dict:
    """Run Slice-1 debugging for one thread + verified account_id.

    dry_run=True: compose everything, post/​send NOTHING (returns artifacts; an
    `verify_reply` lets the dry-run show the final findings + draft).
    dry_run=False: post to #debugging + #verify for real, persist the pending case
    keyed on the #verify thread ts, and return — the Socket Mode listener resumes
    via resume() when Nihal replies in that thread.
    """
    email_text = _email_text(thread)
    sender = _sender(thread)
    art: dict = {"account_id": account_id, "sender": sender, "dry_run": dry_run}

    # 1. detect: must carry a Call UUID
    m = _UUID_RE.search(email_text)
    if not m:
        art["terminal"] = "no Call UUID in the mail — not a Slice-1 case (route elsewhere)"
        return art
    uuid = m.group(0)
    art["call_uuid"] = uuid

    # 2. investigate (model-driven; account_id injected by dispatch)
    agent = llm.run_agent(
        _INVESTIGATE_SYSTEM,
        f"Customer email:\n{email_text}\n\nInvestigate Call UUID: {uuid}",
        _TOOLS, _dispatch(account_id),
    )
    art["trace"] = [{"tool": s["tool"], "args": s["args"]} for s in agent["trace"]]
    facts, have_data = _render_facts(agent["trace"])
    art["grounded_facts"] = facts

    # 3. interpret
    interp = llm.debug_interpret(facts, email_text) if have_data else {
        "resolved": False, "unresolved_reason": "the call was not visible in the call records",
        "case_title": f"Unresolved — {uuid}", "interpretation": "", "verify_leads": []}
    art["case_title"] = interp.get("case_title", f"Call {uuid}")

    # Terminal: couldn't resolve -> say so in #debugging, draft nothing.
    if not have_data or not interp.get("resolved"):
        art["terminal"] = "couldn't resolve"
        art["debugging_post"] = (
            f"*{art['case_title']}*\n"
            f"Customer: {sender}\n\n"
            f"*Grounded facts (from internal tools):*\n{facts}\n\n"
            f"*Status:* Could not resolve from call records — "
            f"{interp.get('unresolved_reason') or 'insufficient data'}. No draft produced.")
        if not dry_run:
            _post(DEBUG_CHANNEL, art["debugging_post"])
        return art

    # #debugging post (facts + interpretation, labeled) — report-only
    art["debugging_post"] = (
        f"*{art['case_title']}*\n"
        f"Customer: {sender}\n\n"
        f"*Grounded facts (from internal tools):*\n{facts}\n\n"
        f"*Interpretation (agent's read — hypothesis, not confirmed):*\n"
        f"{interp.get('interpretation', '')}")

    # #verify prompt: targeted verify leads ONLY (no findings dump)
    leads = interp.get("verify_leads") or []
    art["verify_prompt"] = (
        f"@the reviewer — verifying a call for a customer case (call `{uuid}`). "
        f"Can you confirm/dig into:\n" + "\n".join(f"  • {l}" for l in leads))

    if dry_run:
        # 4./5./6. simulate the wait so the dry-run can show final + draft.
        if verify_reply is None:
            art["status"] = "would post #debugging + #verify, then WAIT for the reviewer reply"
            return art
        art["verify_reply"] = verify_reply
        art["final_debugging_post"], art["customer_draft"] = _finalize(
            email_text, facts, interp.get("interpretation", ""), art["case_title"],
            account_id, verify_reply)
        art["customer_draft_destination"] = (
            f"approval channel {APPROVAL_CHANNEL or '(unset)'} — human gate (not auto-sent)")
        return art

    # LIVE: post #debugging + #verify, persist the pending case, then return.
    _post(DEBUG_CHANNEL, art["debugging_post"])
    verify_resp = _post(VERIFY_CHANNEL, art["verify_prompt"])
    verify_ts = verify_resp.get("ts")
    art["verify_ts"] = verify_ts
    db.kv_set(_PENDING_PREFIX + str(verify_ts), {
        "account_id": account_id, "uuid": uuid, "facts": facts,
        "interpretation": interp.get("interpretation", ""), "case_title": art["case_title"],
        "email_text": email_text, "sender": sender, "thread": thread})
    art["status"] = f"posted #debugging + #verify (thread {verify_ts}); waiting for the reviewer reply"
    return art


def resume(verify_thread_ts: str, verify_reply: str) -> dict | None:
    """Resume a pending case when Nihal replies in its #verify thread.

    Called by the Socket Mode listener (which enforces: right thread ts + a
    human message from Nihal, not the bot, not the reviewer). Posts final findings to
    #debugging and routes the customer draft through the EXISTING approval gate.
    """
    # REQUIRE non-empty text: an empty / image-only reply must NOT consume the case
    # (findings can't be grounded on it). Leave it pending; the catch-up nudges to
    # paste as text. This guard holds on the live-handler path too.
    if not (verify_reply and verify_reply.strip()):
        return {"branch": "empty_reply_ignored"}

    key = _PENDING_PREFIX + str(verify_thread_ts)
    # ATOMIC claim-and-clear: exactly one caller wins the pending case — whether
    # it's a live #verify reply or a reconnect catch-up firing for the same case.
    # Concurrent callers get None here and no-op (no double resume).
    pending = db.claim_pending_case(key)
    if not pending:
        return None
    try:
        final_post, draft = _finalize(
            pending["email_text"], pending["facts"], pending["interpretation"],
            pending["case_title"], pending["account_id"], verify_reply)
        _post(DEBUG_CHANNEL, final_post)

        # customer draft -> existing approval gate (draft record + Block Kit card).
        from app import slack_approval  # lazy: avoid import cycle with the listener
        thread = pending["thread"]
        classification = {"intent": "technical_support", "customer_name": None,
                          "summary": pending["case_title"]}
        ids = db.persist_processing(thread, classification, draft, source="debug",
                                    reply_context=thread.get("reply_context"), booking=None,
                                    artifact_key=db.build_artifact_key(
                                        thread.get("thread_id"), "debug", verify_thread_ts))
        slack_approval.post_draft_once(pending["email_text"], draft, ids["draft_id"])  # claim-then-post
    except Exception:
        db.kv_set(key, pending)  # restore so the case can be retried, then re-raise
        raise
    return {"final_debugging_post": final_post, "customer_draft": draft,
            "draft_id": ids["draft_id"]}


def _find_nihal_reply(thread_ts: str, channel: str = VERIFY_CHANNEL) -> tuple[str | None, bool]:
    """Read a case's thread (in `channel`) for a reply from Nihal (not the bot,
    not the reviewer). Returns (text, file_only): the reply TEXT if he replied with usable
    text, else (None, True) if he replied image/file-ONLY with no text (so the
    caller nudges 'paste as text' and does NOT consume the case), else (None, False)
    if he hasn't replied yet."""
    import os
    from slack_sdk import WebClient
    nihal = os.getenv("SLACK_NIHAL_USER_ID")
    client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    msgs = client.conversations_replies(channel=channel, ts=thread_ts).get("messages", [])
    file_only = False
    for m in msgs:
        if m.get("ts") == thread_ts or m.get("bot_id"):
            continue  # parent bot prompt / the reviewer
        subtype = m.get("subtype")
        if subtype and subtype != "file_share":
            continue  # system/join/etc — but a file_share IS a (text-less) reply
        if nihal and m.get("user") != nihal:
            continue  # only Nihal's reply resumes
        txt = (m.get("text") or "").strip()
        if txt:
            return txt, False                     # usable text reply
        if m.get("files") or subtype == "file_share":
            file_only = True                      # replied image/file-only, no text
    return None, file_only


def _nudge_paste_text(thread_ts: str, channel: str, prefix: str) -> None:
    """Nihal replied to a pending case with an image/file and no text — nudge once
    (dampened) to paste as text; leave the case PENDING (never consumed)."""
    key = prefix + str(thread_ts)
    pend = db.kv_get(key)
    if not pend or pend.get("nudged_filetext"):
        return
    db.kv_set(key, {**pend, "nudged_filetext": True})
    _post(channel, "⚠️ That reply was an image/file with no text — I can't read it. Please paste "
          "the answer as TEXT and I'll pick it up.", thread_ts=str(thread_ts))
    return None


def catch_up_pending_cases(max_age_seconds: int = 604800, max_cases: int = 20) -> int:
    """On (re)connect: for each recent pending case, check its #verify thread for a
    Nihal reply that landed while we were disconnected, and resume it.

    BOUNDED (only cases newer than max_age_seconds, capped at max_cases — one
    conversations.replies each) so a reconnect storm can't become an API-rate
    storm. Safe under a concurrent live event: resume() claims the case
    atomically, so the reply is processed exactly once.
    """
    resumed = 0
    # Scan BOTH kinds of Slack-waiting cases: the reviewer-verify (#verify) and
    # account-ask (#debugging). Customer-suspended cases (dbgcust) re-attach via
    # the ingest hook, not here.
    for prefix, channel, resume_fn in (
        (_PENDING_PREFIX, VERIFY_CHANNEL, resume),
        (_ACCT_PREFIX, DEBUG_CHANNEL, resume_account),
    ):
        for key in db.list_pending_case_keys(prefix, max_age_seconds, max_cases):
            ts = key[len(prefix):]
            try:
                reply, file_only = _find_nihal_reply(ts, channel)
                if reply and resume_fn(ts, reply):
                    resumed += 1
                    print(f"[catch-up] resumed {prefix}{ts}")
                elif file_only:
                    # image/file-only reply — nudge, do NOT consume the case.
                    _nudge_paste_text(ts, channel, prefix)
                    print(f"[catch-up] file-only reply on {prefix}{ts} — nudged, case kept")
            except Exception as e:
                print(f"[catch-up] error on {ts}: {type(e).__name__}: {e}")
    return resumed


# =====================================================================
# Slice 2 — account-id-first investigation (vague case, with/without UUID)
# =====================================================================
_ACCT_PREFIX = "dbgacct:"   # account-ask pending, waits in #debugging (distinct key space)
_CUST_PREFIX = "dbgcust:"   # customer-suspended pending, keyed on the Gmail thread_id


def _parse_account_id(text: str):
    """Nihal's #debugging reply is a lone number (copy-paste). First integer wins."""
    m = re.search(r"\b(\d{4,})\b", text or "")
    return int(m.group(1)) if m else None


def _extract_window(thread: dict) -> dict:
    """Investigation window from the email's time hints; default last 24h."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    w = llm.extract_debug_window(_email_text(thread), now.strftime("%Y-%m-%d %H:%M:%S"))
    if w.get("has_hint") and w.get("start") and w.get("end"):
        return {"start": w["start"], "end": w["end"], "has_hint": True}
    return {"start": (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            "end": now.strftime("%Y-%m-%d %H:%M:%S"), "has_hint": False}


# Known-NORMAL disconnect causes — excluded from the failure set. duration==0
# means "didn't connect", NOT "failed abnormally"; a heavy-normal-traffic account
# would otherwise flip a genuinely-clear case to AMBIGUOUS and wrongly email the
# customer. The excluded COUNT is surfaced in the #debugging header so an
# incomplete list here is visible, not silent.
_NORMAL_CAUSES = {"normal_clearing", "user_busy", "no_answer"}


def _is_normal_cause(cause: str | None) -> bool:
    c = (cause or "").strip().lower()
    return c in _NORMAL_CAUSES or "cancel" in c  # *_cancel / *_cancelled


def _histogram_verdict(rows: list[dict]) -> tuple[dict, str, str | None, int]:
    """Failure histogram (ABNORMAL disconnect_reason among calls that didn't connect) +
    dominance verdict. Clear iff >=3 failed AND top cause >= 2x the runner-up
    (single-cause degenerate case trivially clear). Known-normal causes are
    excluded from the failure set. Returns (hist, verdict, rep_uuid, n_excluded)."""
    from collections import Counter
    didnt_connect = [r for r in rows if r.get("duration") in (0, None)]
    n_excluded = sum(1 for r in didnt_connect if _is_normal_cause(r.get("disconnect_reason")))
    failed = [r for r in didnt_connect if not _is_normal_cause(r.get("disconnect_reason"))]
    hist = Counter((r.get("disconnect_reason") or "unknown") for r in failed)
    if not hist:
        return {}, "ambiguous", None, n_excluded
    ranked = hist.most_common()
    top_cause, top_n = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    clear = sum(hist.values()) >= 3 and (len(ranked) == 1 or top_n >= 2 * runner)
    rep = None
    if clear:
        cands = [r for r in failed
                 if (r.get("disconnect_reason") or "unknown") == top_cause and r.get("call_uuid")]
        cands.sort(key=lambda r: str(r.get("initiation_time") or r.get("start_time") or ""),
                   reverse=True)
        rep = cands[0]["call_uuid"] if cands else None
    return dict(hist), ("clear" if clear else "ambiguous"), rep, n_excluded


def investigate_account_wide(thread: dict, account_id, channel: str | None = None) -> dict:
    """Account-wide read-only investigation: window -> get_calls_for_account ->
    failure histogram + dominance verdict. account_id is injected here (verified
    case account), never taken from email content."""
    win = _extract_window(thread)
    channels = [channel] if channel else ["sip_trunk", "voice"]
    rows: list[dict] = []
    ok, errors = 0, []
    for ch in channels:
        try:
            rows += redshift_tools.get_calls_for_account(
                account_id, win["start"], win["end"], channel=ch, limit=500)
            ok += 1  # query SUCCEEDED (possibly empty)
        except Exception as e:
            errors.append(f"{ch}:{type(e).__name__}")
            print(f"[acct-wide] {ch} query error: {type(e).__name__}: {e}")
    if ok == 0:
        # EVERY data query failed — we could NOT investigate. This is NOT "no
        # failures found"; must not be mistaken for ambiguous / ask-the-customer.
        return {"account_id": account_id, "window": win, "n_rows": 0, "histogram": {},
                "n_failed": 0, "n_normal_excluded": 0, "verdict": "unavailable",
                "representative_uuid": None, "errors": errors}
    if not rows:
        # Queries SUCCEEDED but the account has NO traffic in the window — a
        # distinct third verdict. Asking the customer for a UUID here is
        # nonsensical; instead note "wrong account? wrong window?" to Nihal.
        return {"account_id": account_id, "window": win, "n_rows": 0, "histogram": {},
                "n_failed": 0, "n_normal_excluded": 0, "verdict": "no_data",
                "representative_uuid": None, "errors": errors}
    hist, verdict, rep, n_excluded = _histogram_verdict(rows)
    return {"account_id": account_id, "window": win, "n_rows": len(rows),
            "histogram": hist, "n_failed": sum(hist.values()),
            "n_normal_excluded": n_excluded, "verdict": verdict,
            "representative_uuid": rep, "errors": errors}


# ============================================================================
# Slice 3 — SMS/MDR debugging (same rails as Slice 2, minus the the reviewer step).
# Verification weight rests on facts-vs-interpretation labelling: interpretation
# is CODE-RENDERED and PATTERN-DESCRIPTIVE ONLY (never causal). carrier_code /
# error_code are opaque carrier DLR codes — a code->cause guess would be the
# 1078 fabrication class, so it is structurally impossible here (no LLM in the
# interpretation path, no code->meaning table).
# ============================================================================
_SMS_FAILED_STATES = {"failed", "undelivered"}
_SMS_DELIVERED_STATES = {"delivered", "read", "received"}   # success / excluded
_SMS_LIMBO_STATES = {"sent", "queued"}                      # submitted, no final DLR
_SMS_FAIL_MIN = 3    # min failed to call a failure pattern (mirrors voice)
_SMS_LIMBO_MIN = 3   # min limbo to call a "submitted-not-confirming" finding
_SIGNOFF = "Best regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo"


def _sms_buckets(hist: list[dict]) -> dict:
    """Fold the EXACT (state, carrier_code, country) aggregate into three buckets +
    the carrier_code histogram over FAILED rows + destination clustering. Because the
    aggregate has no row cap, these counts are complete — the verdict can't be
    flipped by recency truncation. Unknown states are counted as `other` (visible,
    never silently dropped)."""
    from collections import Counter
    failed = delivered = limbo = other = 0
    dlr: Counter = Counter()
    country: Counter = Counter()
    for r in hist:
        st = (r.get("delivery_state") or "").strip().lower()
        n = int(r.get("n") or 0)
        if st in _SMS_FAILED_STATES:
            failed += n
            dlr[(r.get("carrier_code") or "unknown")] += n
            country[(r.get("country_iso") or "??")] += n
        elif st in _SMS_DELIVERED_STATES:
            delivered += n
        elif st in _SMS_LIMBO_STATES:
            limbo += n
        else:
            other += n
    return {"failed": failed, "delivered": delivered, "limbo": limbo, "other": other,
            "total": failed + delivered + limbo + other,
            "dlr": dict(dlr), "country": dict(country)}


def _sms_dominance(dlr: dict) -> tuple[bool, str | None, int, int]:
    """failed_finding = >=3 failed AND top carrier_code >= 2x the runner-up (a lone
    code is trivially dominant). Returns (is_finding, top_code, top_n, runner_n)."""
    if not dlr:
        return False, None, 0, 0
    ranked = sorted(dlr.items(), key=lambda kv: -kv[1])
    top_code, top_n = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    is_finding = sum(dlr.values()) >= _SMS_FAIL_MIN and (len(ranked) == 1 or top_n >= 2 * runner)
    return is_finding, top_code, top_n, runner


def _sms_verdict(b: dict) -> dict:
    """Verdict from EXACT buckets. The two abnormal buckets are assessed
    INDEPENDENTLY — a small failed cluster never wins CLEAR while limbo dominates:
      failed_finding AND limbo_finding -> mixed  (report BOTH)
      limbo_finding only               -> limbo
      failed_finding only              -> clear
      neither                          -> ambiguous (mixed-no-dominance OR healthy)
    limbo_finding requires limbo >= FAIL threshold AND limbo >= failed (co-dominant)."""
    failed_finding, dom_code, dom_n, runner = _sms_dominance(b["dlr"])
    limbo_finding = b["limbo"] >= _SMS_LIMBO_MIN and b["limbo"] >= b["failed"]
    if failed_finding and limbo_finding:
        verdict = "mixed"
    elif limbo_finding:
        verdict = "limbo"
    elif failed_finding:
        verdict = "clear"
    else:
        verdict = "ambiguous"
    return {"verdict": verdict, "failed_finding": failed_finding,
            "limbo_finding": limbo_finding, "dom_code": dom_code,
            "dom_n": dom_n, "runner_n": runner}


def _sms_cap_note(rows: list[dict], limit: int) -> str:
    """Disclosure for any BOUNDED row query that is surfaced. The verdict never
    rides on such a query (it comes from the no-cap aggregate), but where row
    samples appear we say so rather than implying we saw everything."""
    return (f"  ⚠️ representative sample only — the row query hit its {limit}-row cap "
            f"(most recent first); counts/verdict come from the full-window aggregate, not this list."
            ) if len(rows) >= limit else ""


def investigate_sms_account_wide(thread: dict, account_id) -> dict:
    """Account-wide SMS investigation. Counts come from the EXACT aggregate
    (get_message_histogram, no row cap); a bounded row query is used ONLY to pick
    one representative failed UUID to drill. Verdicts: unavailable (query failed),
    no_data (0 rows), out_of_scope (0 SMS but WhatsApp/MMS present), then
    clear/limbo/mixed/ambiguous from _sms_verdict. account_id injected here."""
    win = _extract_window(thread)
    try:
        hist = redshift_tools.get_message_histogram(account_id, win["start"], win["end"])
    except Exception as e:
        print(f"[sms-acct-wide] histogram query error: {type(e).__name__}: {e}")
        return {"account_id": account_id, "window": win, "verdict": "unavailable",
                "buckets": {}, "errors": [f"histogram:{type(e).__name__}"],
                "representative_uuid": None}

    b = _sms_buckets(hist)
    if b["total"] == 0:
        # 0 outbound SMS — distinguish truly-empty from out-of-scope (WhatsApp/MMS).
        try:
            breakdown = redshift_tools.get_message_type_breakdown(account_id, win["start"], win["end"])
        except Exception:
            breakdown = []
        nonsms = {(r.get("message_type") or "?"): int(r["n"])
                  for r in breakdown if (r.get("message_type") or "").lower() != "sms"}
        verdict = "out_of_scope" if nonsms else "no_data"
        return {"account_id": account_id, "window": win, "verdict": verdict,
                "buckets": b, "type_breakdown": nonsms, "errors": [], "representative_uuid": None}

    v = _sms_verdict(b)
    rep, cap_note = None, ""
    if v["failed_finding"] and v["dom_code"]:
        # ONE representative failed message with the dominant code. Recency-biased
        # row query is fine here — we only need an example; the verdict is already
        # fixed by the exact aggregate above.
        try:
            LIMIT = 5
            cands = redshift_tools.get_messages_for_account(
                account_id, win["start"], win["end"], delivery_state=None,
                carrier_code=v["dom_code"], limit=LIMIT)
            cap_note = _sms_cap_note(cands, LIMIT)
            failed_cands = [c for c in cands
                            if (c.get("delivery_state") or "").lower() in _SMS_FAILED_STATES] or cands
            rep = failed_cands[0]["message_uuid"] if failed_cands else None
        except Exception as e:
            print(f"[sms-acct-wide] representative selection error: {type(e).__name__}: {e}")
    return {"account_id": account_id, "window": win, **v, "buckets": b,
            "errors": [], "representative_uuid": rep, "cap_note": cap_note}


def _sms_header(inv: dict) -> str:
    """Three-bucket header (your addition 1). Built from the EXACT aggregate — no
    truncation note here because there is no cap. `other` shown only if present."""
    b = inv.get("buckets", {})
    dlr = b.get("dlr", {})
    top = ", ".join(f"{c}×{n}" for c, n in sorted(dlr.items(), key=lambda kv: -kv[1])[:6]) or "(none failed)"
    seg = (f"{b.get('failed', 0)} failed, {b.get('delivered', 0)} delivered excluded, "
           f"{b.get('limbo', 0)} queued/sent-no-DLR")
    if b.get("other"):
        seg += f", {b['other']} other-state"
    return (f"_Account-wide SMS {inv['window']['start']}→{inv['window']['end']} "
            f"({seg}): carrier_code {top} — verdict: {inv['verdict'].upper()}_{inv.get('cap_note','')}")


def _sms_facts(row: dict) -> str:
    """Code-rendered grounded facts for ONE message (no LLM). Opaque codes as-is."""
    keys = ["message_uuid", "account_id", "delivery_state", "message_direction",
            "message_type", "country_iso", "to_number_redacted", "from_number_redacted",
            "carrier_code", "error_code", "units", "message_time"]
    lines = "\n".join(f"    {k} = {row.get(k)}" for k in keys if k in row)
    return ("get_message_by_uuid [messages_fact]:\n" + lines +
            "\n    (carrier_code / error_code are carrier DLR status codes, reported as-is; "
            "no code→cause mapping exists in this warehouse — not interpreted.)")


def _sms_interpretation(inv: dict) -> str:
    """PATTERN-DESCRIPTIVE, code-rendered. States what the exact aggregate shows —
    counts, dominant code, destination clustering — and nothing causal."""
    b = inv["buckets"]
    v = inv["verdict"]
    countries = ", ".join(f"{c} ({n})" for c, n in sorted(b.get("country", {}).items(),
                                                          key=lambda kv: -kv[1])[:3]) or "—"
    fail_line = (f"{b['failed']} of {b['total']} outbound SMS are failed/undelivered; "
                 f"dominant carrier code carrier_code={inv.get('dom_code')} on {inv.get('dom_n')} "
                 f"(runner-up {inv.get('runner_n')}). Top destinations: {countries}.")
    limbo_line = (f"{b['limbo']} outbound SMS are submitted/sent to the carrier with no final "
                  f"delivery receipt (states sent/queued) — 'submitted, not confirming', not a "
                  f"confirmed failure.")
    if v == "mixed":
        return ("Two concurrent patterns (what the data shows, not a cause):\n"
                f"  1) {fail_line}\n  2) {limbo_line}")
    if v == "clear":
        return (fail_line + " The code is a carrier DLR status, not a confirmed cause "
                "(we'd need the carrier's code reference to say why).")
    if v == "limbo":
        return limbo_line + f" Failed in the same window: {b['failed']}."
    if v == "out_of_scope":
        return ("No outbound SMS in this window; the account's messaging traffic here is "
                f"{inv.get('type_breakdown')} — out of scope for SMS debugging.")
    if v == "no_data":
        return "No outbound SMS for this account in the window (wrong account or wrong window?)."
    # ambiguous — mixed-no-dominance vs healthy
    if b["failed"] == 0:
        return (f"Window looks healthy: {b['delivered']} delivered, {b['failed']} failed. "
                "The failure the customer means isn't in this window.")
    return (f"{b['failed']} failed with no single dominant code (top: "
            f"{sorted(b.get('dlr', {}).items(), key=lambda kv: -kv[1])[:3]}); no clear pattern.")


def _sms_customer_message(inv: dict, customer_name: str | None) -> str:
    """Conservative, descriptive-only customer draft (code-templated so it cannot
    assert a cause). Clear/limbo/mixed state the observed pattern + promise to
    confirm the carrier-side reason; ambiguous ACKNOWLEDGES the data before asking."""
    hi = f"Hi {customer_name}," if customer_name else "Hi,"
    b = inv["buckets"]
    v = inv["verdict"]
    win = f"{inv['window']['start']} → {inv['window']['end']}"
    if v == "clear":
        body = (f"We looked into your SMS. In {win} we see {b['failed']} of {b['total']} outbound "
                f"messages failing, most with the same carrier status code ({inv.get('dom_code')}). "
                "We're confirming the carrier-side reason for that code and will follow up.")
    elif v == "limbo":
        body = (f"We looked into your SMS. In {win}, {b['limbo']} outbound messages show as submitted "
                "to the carrier but we don't yet have delivery confirmation (no final delivery "
                "receipt). We're checking the delivery status with the carrier and will follow up.")
    elif v == "mixed":
        body = (f"We looked into your SMS for {win}. Two things stand out: {b['failed']} messages "
                f"failed (most with carrier code {inv.get('dom_code')}), and separately {b['limbo']} "
                "are submitted but not yet confirming delivery. We're confirming both with the "
                "carrier and will follow up.")
    elif b["failed"] == 0:   # ambiguous — healthy window, acknowledge the data
        body = ("Your messages in this window show as delivered on our side — could you share a "
                "specific message UUID or the timestamp (with timezone) of one that failed, so we "
                "can pinpoint it?")
    else:                    # ambiguous — mixed failures, no pattern
        body = (f"We see {b['failed']} failures in this window but they don't share a single pattern. "
                "Could you share a specific message UUID or timestamp (with timezone) so we can "
                "pinpoint the one you're seeing?")
    return f"{hi}\n\n{body}\n\n{_SIGNOFF}"


def _slice1_artifacts(thread: dict, account_id, uuid: str, *, verify_reply=None, header="") -> dict:
    """The Slice-1 per-UUID rail as pure artifacts (no side effects): investigate
    (account_id injected) -> grounded facts -> interpretation -> #debugging +
    #verify; if verify_reply given, also final findings + stripped customer draft."""
    email_text = _email_text(thread)
    agent = llm.run_agent(_INVESTIGATE_SYSTEM,
                          f"Customer email:\n{email_text}\n\nInvestigate Call UUID: {uuid}",
                          _TOOLS, _dispatch(account_id))
    trace = [{"tool": s["tool"], "args": s["args"]} for s in agent["trace"]]
    facts, have_data = _render_facts(agent["trace"])
    interp = llm.debug_interpret(facts, email_text) if have_data else {
        "resolved": False, "unresolved_reason": "call not visible for this account",
        "case_title": f"Unresolved — {uuid}", "interpretation": "", "verify_leads": []}
    title = interp.get("case_title", f"Call {uuid}")
    art = {"call_uuid": uuid, "trace": trace, "grounded_facts": facts,
           "case_title": title, "interpretation": interp.get("interpretation", ""),
           "resolved": bool(interp.get("resolved") and have_data)}
    if not art["resolved"]:
        art["terminal"] = "couldn't resolve"
        art["debugging_post"] = (f"*{title}*\n{header}\n*Grounded facts (from internal tools):*\n"
                                 f"{facts}\n\n*Status:* could not resolve — "
                                 f"{interp.get('unresolved_reason') or 'insufficient data'}.")
        return art
    art["debugging_post"] = (f"*{title}*\n{header}\n*Grounded facts (from internal tools):*\n{facts}\n\n"
                             f"*Interpretation (agent's read — hypothesis, not confirmed):*\n"
                             f"{interp.get('interpretation','')}")
    leads = interp.get("verify_leads") or []
    art["verify_prompt"] = (f"@the reviewer — verifying call `{uuid}`. Can you confirm/dig into:\n"
                           + "\n".join(f"  • {l}" for l in leads))
    if verify_reply is not None:
        art["final_debugging_post"], art["customer_draft"] = _finalize(
            email_text, facts, interp.get("interpretation", ""), title, account_id, verify_reply)
    return art


def _histogram_header(aw: dict) -> str:
    """Render the histogram + branch for the #debugging post so a borderline
    dominance call is visible (and the threshold tunable from real cases)."""
    h = ", ".join(f"{c}×{n}" for c, n in sorted(aw["histogram"].items(), key=lambda kv: -kv[1])) or "(no abnormal failures)"
    # Partial-data note: if a channel query failed but another succeeded, the
    # verdict is from partial data and must SAY so (never a silent partial verdict).
    partial = ""
    errs = aw.get("errors") or []
    if errs:
        failed = ", ".join(e.split(":")[0] for e in errs)
        partial = f"  ⚠️ PARTIAL — {failed} query FAILED; verdict from the remaining channel(s) only."
    return (f"_Account-wide {aw['window']['start']}→{aw['window']['end']} "
            f"({aw['n_rows']} calls, {aw['n_failed']} failed, {aw.get('n_normal_excluded', 0)} normal excluded): "
            f"{h} — verdict: {aw['verdict'].upper()}_{partial}")


def reattach_investigate(thread: dict, account_id, reply_text: str) -> dict:
    """Re-attach a customer reply into the investigation. account_id is the
    SUSPENDED case's verified account — NEVER taken from the (untrusted) reply.
    A UUID in the reply -> Slice-1 rail; otherwise re-run account-wide."""
    m = _UUID_RE.search(reply_text or "")
    if m:
        return {"path": "uuid", **_slice1_artifacts(thread, account_id, m.group(0))}
    return {"path": "account_wide", **investigate_account_wide(thread, account_id)}


def _account_wide_live(thread: dict, account_id, claimed: dict, reply_text: str) -> dict:
    """LIVE account-wide re-attach: run the same four-verdict branch as
    resume_account, but keyed on the Gmail-thread dbgcust case (not the #debugging
    account-ask). clear → drill a representative call onto the Slice-1 rail;
    ambiguous → customer-ask draft + RE-SUSPEND a fresh dbgcust; no_data/unavailable
    → held note + keep the case for a later retry. Side effects (posts, drafts,
    suspends); raises on failure so the caller restores the claimed case."""
    thread_id = thread.get("thread_id")
    aw = investigate_account_wide(thread, account_id)
    header = _histogram_header(aw)

    def _hold(note: str) -> dict:
        # Held (unavailable / no_data): keep the case alive so a later customer
        # reply retries; note to Nihal; NO customer-ask, NO fresh suspend churn.
        db.kv_set(_CUST_PREFIX + str(thread_id), claimed)
        _post(DEBUG_CHANNEL, note)
        return {"branch": aw["verdict"]}

    if aw["verdict"] == "unavailable":
        return _hold(f"⚠️ Re-attach account-wide for {account_id}: couldn't reach the call data "
                     f"({', '.join(aw.get('errors', []))}) — HELD, case kept for retry; NOT asking the customer.")
    if aw["verdict"] == "no_data":
        return _hold(f"ℹ️ Re-attach account-wide for {account_id}: no calls in "
                     f"{aw['window']['start']}→{aw['window']['end']} — wrong account or window? "
                     f"HELD, case kept; NOT asking the customer.")
    if aw["verdict"] == "clear" and aw["representative_uuid"]:
        _post(DEBUG_CHANNEL, f"*Re-attached — account {account_id}*\n{header}\n"
                             f"*Branch:* PATTERN-CLEAR → drilling `{aw['representative_uuid']}`")
        return _run_slice1_live(thread, account_id, aw["representative_uuid"], header=header)

    # AMBIGUOUS again (we asked, they answered vaguely) → ask once more and
    # RE-SUSPEND a FRESH dbgcust on the same thread. The just-claimed key is now
    # empty (claim_pending_case DELETEd it), so this write can't collide.
    _post(DEBUG_CHANNEL, f"*Re-attached — account {account_id}*\n{header}\n"
                         f"*Branch:* AMBIGUOUS → asking the customer again (case re-suspended).")
    from app import slack_approval
    draft = _strip_internal(llm.debug_customer_ask(reply_text), account_id)
    ids = db.persist_processing(thread, {"intent": "technical_support", "customer_name": None,
                                         "summary": "debug: ask customer for specifics (re-attach)"},
                                draft, source="debug", reply_context=thread.get("reply_context"),
                                artifact_key=db.build_artifact_key(
                                    thread.get("thread_id"), "debug-ask", account_id))
    slack_approval.post_draft_once(reply_text, draft, ids["draft_id"])
    db.kv_set(_CUST_PREFIX + str(thread_id), {
        "account_id": account_id, "thread": thread, "email_text": reply_text,
        "asked": "uuid/timestamp/destination"})
    return {"branch": "ambiguous", "customer_draft": draft, "draft_id": ids["draft_id"],
            "suspended_thread": thread_id}


def _latest_uuid(thread: dict) -> str | None:
    """The UUID from the customer's LATEST message (their reply to our ask) — NOT
    the whole concatenated thread, which can carry a stale UUID from an earlier
    turn and misroute the drill (observed live: a voice UUID quoted throughout a
    thread's history hijacking an SMS re-attach → not_found). Ingest does not
    separate quoted text, so this is first-match WITHIN the latest message;
    residual: a customer who BOTTOM-posts (types below the quoted chain) could
    still surface a quoted UUID first. Top-posting (the common case) puts the new
    UUID first, which is what we want."""
    msgs = thread.get("messages") or []
    if not msgs:
        return None
    m = _UUID_RE.search(msgs[-1].get("body") or "")
    return m.group(0) if m else None


def _voice_uuid_exists(uuid: str, account_id) -> bool:
    """Cheap scoped existence pre-check for the voice re-attach: is this UUID a
    call for THIS account (voice CDR or sip_trunk)? Lets a not-found UUID fall
    through to account-wide instead of dead-ending (and stranding the case)."""
    return bool(redshift_tools.get_call_by_uuid(uuid, account_id).get("found")
                or redshift_tools.get_sip_trunk_call_detail(uuid, account_id).get("found"))


# --- SMS live continuation (Slice 3 — same rails as voice, NO the reviewer step) -----
def _sms_emit_uuid_finding(thread: dict, account_id, uuid: str, row: dict) -> dict:
    """One-message SMS finding for an ALREADY-FOUND row: #debugging facts +
    descriptive interpretation → conservative customer draft to the approval gate.
    (Existence/not-found + fall-through is handled by _sms_investigate_and_emit.)"""
    facts = _sms_facts(row)
    st = (row.get("delivery_state") or "").lower()
    descr = (f"delivery_state={row.get('delivery_state')}, carrier_code={row.get('carrier_code')} "
             f"(opaque carrier DLR code — not interpreted), to {row.get('country_iso')}. "
             + ("This is a failed/undelivered message — what the data shows, not a confirmed cause."
                if st in _SMS_FAILED_STATES else
                "This message is not in a failed state on our side."))
    _post(DEBUG_CHANNEL, f"*SMS {uuid}*\n*Grounded facts (from internal tools):*\n{facts}\n\n"
                         f"*Interpretation (pattern-descriptive — not a cause):*\n{descr}")
    hi = f"Hi {(thread.get('customer_name') or '')}," if thread.get("customer_name") else "Hi,"
    if st in _SMS_FAILED_STATES:
        body = (f"We looked into that message — on our side it shows as {row.get('delivery_state')} "
                f"with a carrier status code ({row.get('carrier_code')}). We're confirming the carrier-"
                "side reason for that code and will follow up.")
    else:
        body = (f"We looked into that message — on our side it shows as {row.get('delivery_state')}, "
                "which isn't a failure on our end. Could you share what you're seeing on your side "
                "(and the timestamp with timezone) so we can dig further?")
    draft = _strip_internal(f"{hi}\n\n{body}\n\n{_SIGNOFF}", account_id)
    from app import slack_approval
    ids = db.persist_processing(thread, {"intent": "technical_support", "customer_name": None,
                                         "summary": f"SMS debug — {uuid}"},
                                draft, source="debug", reply_context=thread.get("reply_context"),
                                artifact_key=db.build_artifact_key(
                                    thread.get("thread_id"), "debug-sms", f"msg-{uuid}"))
    slack_approval.post_draft_once(_email_text(thread), draft, ids["draft_id"])
    return {"branch": "resolved", "message_uuid": uuid, "draft_id": ids["draft_id"]}


def _sms_investigate_and_emit(thread: dict, account_id, uuid: str | None,
                              reply_text: str, *, on_hold) -> dict:
    """Shared SMS entry for both the account-ask resume and the re-attach paths.
    A found UUID → one-message finding. A UUID that is NOT found for this account
    → a VISIBLE #debugging miss note, then FALL THROUGH to account-wide (what an
    engineer would do with a bad UUID) — never a silent dead-end that consumes the
    reply with no outcome. No UUID → straight to account-wide. on_hold supplies
    the caller-specific hold (dbgacct restore vs dbgcust re-suspend)."""
    if uuid:
        row = redshift_tools.get_message_by_uuid(uuid, account_id)
        if row.get("found"):
            return _sms_emit_uuid_finding(thread, account_id, uuid, row)
        _post(DEBUG_CHANNEL, f"*SMS — {account_id}*\nmessage UUID `{uuid}` not found for this "
                             f"account — proceeding account-wide (wrong account, or a UUID-"
                             f"extraction issue?).")
    inv = investigate_sms_account_wide(thread, account_id)
    return _sms_dispatch_verdict(thread, account_id, inv, reply_text, on_hold=on_hold)


def _sms_emit_finding(thread: dict, account_id, inv: dict, reply_text: str) -> dict:
    """clear/limbo/mixed → #debugging finding (header + descriptive interpretation
    + a labelled representative example) + a conservative customer draft to the
    approval gate. Shared by the re-attach and account-ask paths."""
    header = _sms_header(inv)
    facts = ""
    if inv.get("representative_uuid"):
        row = redshift_tools.get_message_by_uuid(inv["representative_uuid"], account_id)
        if row.get("found"):
            facts = (f"\n*Representative failed message (one example — most recent with the "
                     f"dominant code carrier_code={inv.get('dom_code')}; the counts above are the "
                     f"full-window aggregate, not this single row):*\n" + _sms_facts(row))
    _post(DEBUG_CHANNEL, f"*Account-wide SMS — {account_id}*\n{header}\n"
                         f"*Interpretation (pattern-descriptive — not a cause):*\n"
                         f"{_sms_interpretation(inv)}{facts}")
    draft = _strip_internal(_sms_customer_message(inv, thread.get("customer_name")), account_id)
    from app import slack_approval
    ids = db.persist_processing(thread, {"intent": "technical_support", "customer_name": None,
                                         "summary": f"SMS debug ({inv['verdict']}) — {account_id}"},
                                draft, source="debug", reply_context=thread.get("reply_context"),
                                artifact_key=db.build_artifact_key(
                                    thread.get("thread_id"), "debug-sms", f"finding-{account_id}"))
    slack_approval.post_draft_once(reply_text, draft, ids["draft_id"])
    return {"branch": inv["verdict"], "draft_id": ids["draft_id"]}


def _sms_emit_customer_ask(thread: dict, account_id, inv: dict, reply_text: str) -> dict:
    """ambiguous → data-acknowledging customer-ask + SUSPEND a fresh SMS-tagged
    dbgcust on the Gmail thread (so the customer's reply re-attaches to the SMS
    rail). Shared by both paths."""
    thread_id = thread.get("thread_id")
    _post(DEBUG_CHANNEL, f"*Account-wide SMS — {account_id}*\n{_sms_header(inv)}\n"
                         f"*Branch:* AMBIGUOUS → asking the customer (case suspended).")
    draft = _strip_internal(_sms_customer_message(inv, thread.get("customer_name")), account_id)
    from app import slack_approval
    ids = db.persist_processing(thread, {"intent": "technical_support", "customer_name": None,
                                         "summary": f"SMS debug (ambiguous) — {account_id}"},
                                draft, source="debug", reply_context=thread.get("reply_context"),
                                artifact_key=db.build_artifact_key(
                                    thread.get("thread_id"), "debug-sms", f"ask-{account_id}"))
    slack_approval.post_draft_once(reply_text, draft, ids["draft_id"])
    db.kv_set(_CUST_PREFIX + str(thread_id), {
        "account_id": account_id, "channel": "sms", "thread": thread,
        "email_text": reply_text, "asked": "uuid/timestamp/destination"})
    return {"branch": "ambiguous", "draft_id": ids["draft_id"], "suspended_thread": thread_id}


def _sms_dispatch_verdict(thread: dict, account_id, inv: dict, reply_text: str, *, on_hold) -> dict:
    """Shared SMS verdict dispatch. Only the HOLD action differs between callers —
    the re-attach path re-suspends the dbgcust case; the account-ask path restores
    the dbgacct pending — so it is supplied via on_hold(note). Emission
    (finding / customer-ask) is identical, hence extracted."""
    v = inv["verdict"]
    if v == "unavailable":
        return on_hold(f"⚠️ SMS account-wide for {account_id}: couldn't reach the data "
                       f"({', '.join(inv.get('errors', []))}) — HELD for retry; NOT asking the customer.")
    if v == "no_data":
        return on_hold(f"ℹ️ SMS account-wide for {account_id}: no outbound SMS in "
                       f"{inv['window']['start']}→{inv['window']['end']} — wrong account or window? "
                       f"HELD; NOT asking the customer.")
    if v == "out_of_scope":
        return on_hold(f"ℹ️ SMS account-wide for {account_id}: no outbound SMS in-window; traffic is "
                       f"{inv.get('type_breakdown')} — out of scope for SMS debugging, HELD; NOT asking.")
    if v in ("clear", "limbo", "mixed"):
        return _sms_emit_finding(thread, account_id, inv, reply_text)
    return _sms_emit_customer_ask(thread, account_id, inv, reply_text)


def _sms_reattach_continue(thread: dict, account_id, claimed: dict, reply_text: str) -> dict:
    """SMS re-attach: take the UUID from the customer's LATEST message (not thread
    history); found → one-message rail, not-found/absent → visible miss note +
    account-wide. HOLD re-suspends the dbgcust so a later reply retries."""
    def _hold(note: str) -> dict:
        db.kv_set(_CUST_PREFIX + str(thread.get("thread_id")), claimed)
        _post(DEBUG_CHANNEL, note)
        return {"branch": "held"}
    uuid = _latest_uuid(thread)
    path = "uuid" if uuid else "account_wide"
    return {"path": path, "channel": "sms",
            **_sms_investigate_and_emit(thread, account_id, uuid, reply_text, on_hold=_hold)}


def _reattach_channel(claimed: dict) -> str:
    """Which continuation a suspended case routes to. A channel-ABSENT (legacy,
    pre-Slice-3) dbgcust case routes to VOICE — the default must stay voice so
    existing suspended cases keep working, never an SMS misroute."""
    return (claimed.get("channel") or "voice").lower()


def _reattach_continue(thread: dict, account_id, claimed: dict, reply_text: str) -> dict:
    """LIVE continuation of a re-attached customer reply, dispatched by the case's
    channel. voice (default / legacy channel-absent): path=uuid → Slice-1 rail
    (#verify + verify-pending → resume()), else account-wide four-verdict. sms
    (Slice 3, no the reviewer): path=uuid → one-message rail, else account-wide SMS
    verdict. Raises on failure so maybe_reattach_debug_case restores the case."""
    if _reattach_channel(claimed) == "sms":
        return _sms_reattach_continue(thread, account_id, claimed, reply_text)
    # Voice: UUID from the customer's LATEST message (not thread history — same
    # stale-UUID stranding wrinkle the SMS path had). If that UUID isn't a call
    # for this account, post a VISIBLE miss note and FALL THROUGH to account-wide
    # rather than dead-ending on "could not resolve" and consuming the case.
    uuid = _latest_uuid(thread)
    if uuid and _voice_uuid_exists(uuid, account_id):
        header = f"_Re-attached from customer reply — account {account_id}_"
        return {"path": "uuid", "channel": "voice",
                **_run_slice1_live(thread, account_id, uuid, header=header)}
    if uuid:
        _post(DEBUG_CHANNEL, f"*Re-attach — account {account_id}*\ncall UUID `{uuid}` not found for "
                             f"this account — proceeding account-wide (wrong account, or a UUID-"
                             f"extraction issue?).")
    return {"path": "account_wide", "channel": "voice",
            **_account_wide_live(thread, account_id, claimed, reply_text)}


def maybe_reattach_debug_case(thread: dict) -> bool:
    """Ingest hook: if this Gmail thread has a SUSPENDED customer debug case, route
    the new message back into the investigation (LIVE — onto the Slice-1 / account-
    wide rail) and return True (caller skips normal classification). Returns False
    if there's no suspended case.

    FAIL-OPEN before the claim: any lookup/claim error returns False (fall through
    to normal classification) — a broken hook must never drop mail or wedge ingest.

    RESTORE-ON-FAILURE after the claim: once claim_pending_case has consumed the
    dbgcust case, a throwing continuation must NOT eat the customer's reply — we
    restore the claimed case (same pattern as resume()) and still return True so
    the reply isn't also re-classified as fresh mail. Retry happens on the
    customer's next message; a #debugging note makes the held state visible.

    Does NOT create cases from fresh mail (the initial trigger stays invoke-only).
    """
    try:
        thread_id = thread.get("thread_id")
        if not thread_id:
            return False
        key = _CUST_PREFIX + str(thread_id)
        if not db.kv_get(key):
            return False
        claimed = db.claim_pending_case(key)   # atomic: exactly one re-attach wins
        if not claimed:
            return False
    except Exception as e:
        import traceback
        print(f"[debug-reattach] FAILED-OPEN (pre-claim) → normal classify (mail not dropped): "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # From here the dbgcust case is CLAIMED (consumed). Any failure must restore it.
    account_id = claimed["account_id"]       # from the suspended case, NOT the reply
    try:
        art = _reattach_continue(thread, account_id, claimed, _email_text(thread))
        print(f"[debug-reattach] thread={thread_id} path={art.get('path')} "
              f"branch={art.get('branch')} account={account_id}")
        return True
    except Exception as e:
        db.kv_set(key, claimed)   # restore — a failed continuation must not eat the reply
        import traceback
        print(f"[debug-reattach] continuation FAILED after claim → RESTORED {key} "
              f"(mail routed, not re-classified): {type(e).__name__}: {e}")
        traceback.print_exc()
        try:  # best-effort visibility; the restore already happened
            _post(DEBUG_CHANNEL, f"⚠️ Re-attach investigation errored for account {account_id} "
                                 f"(thread {thread_id}) — case restored, will retry on the next reply "
                                 f"({type(e).__name__}).")
        except Exception:
            pass
        return True


# --- Slice-2 LIVE entry + account resume -----------------------------------
def run_case_v2(thread: dict, *, dry_run: bool = True) -> dict:
    """Slice-2 invoke entry: ask Nihal for the account_id in #debugging, persist
    the account-ask pending case, and wait. (Auto-fire from real mail stays
    invoke-only — this is the manual kickoff.)"""
    email_text = _email_text(thread)
    sender = _sender(thread)
    uuid_m = _UUID_RE.search(email_text)
    ask = (f"*New debugging case* — {thread.get('subject') or '(no subject)'} — from {sender}"
           + (f"  (Call UUID in mail: `{uuid_m.group(0)}`)" if uuid_m else "  (no UUID in mail)")
           + "\nWhat's the *account_id* for this case? (reply with just the number)")
    art = {"dry_run": dry_run, "has_uuid": bool(uuid_m), "account_ask_post": ask}
    if dry_run:
        return art
    resp = _post(DEBUG_CHANNEL, ask)
    ts = resp.get("ts")
    db.kv_set(_ACCT_PREFIX + str(ts), {
        "thread": thread, "email_text": email_text, "sender": sender,
        "uuid": uuid_m.group(0) if uuid_m else None})
    art["debug_ts"] = ts
    art["status"] = f"posted account-ask to #debugging (thread {ts}); waiting for account_id"
    return art


def run_sms_case(thread: dict, *, dry_run: bool = True) -> dict:
    """Slice-3 invoke entry (SMS): ask Nihal for the account_id in #debugging and
    persist the account-ask pending TAGGED channel='sms', so resume_account routes
    it to the SMS rail (no the reviewer). Same invoke-only kickoff as run_case_v2."""
    email_text = _email_text(thread)
    sender = _sender(thread)
    uuid_m = _UUID_RE.search(email_text)
    ask = (f"*New SMS debugging case* — {thread.get('subject') or '(no subject)'} — from {sender}"
           + (f"  (message UUID in mail: `{uuid_m.group(0)}`)" if uuid_m else "  (no UUID in mail)")
           + "\nWhat's the *account_id* for this SMS case? (reply with just the number)")
    art = {"dry_run": dry_run, "channel": "sms", "has_uuid": bool(uuid_m), "account_ask_post": ask}
    if dry_run:
        return art
    resp = _post(DEBUG_CHANNEL, ask)
    ts = resp.get("ts")
    db.kv_set(_ACCT_PREFIX + str(ts), {
        "thread": thread, "email_text": email_text, "sender": sender,
        "uuid": uuid_m.group(0) if uuid_m else None, "channel": "sms"})
    art["debug_ts"] = ts
    art["status"] = f"posted SMS account-ask to #debugging (thread {ts}); waiting for account_id"
    return art


# --- Auto-trigger (Alex case): route debugging mail off the invoke-only path ---
_BOUNCE_RE = re.compile(r"\bnot\s+a\s+debug(ging)?\s+case\b", re.I)
_CHANNEL_TOKEN_RE = re.compile(r"\b(voice|sms)\b", re.I)
_VOICE_HINT_RE = re.compile(r"\b(calls?|voice|dial(?:ing|er)?|sip|ivr|ringing|hangup)\b", re.I)
_SMS_HINT_RE = re.compile(r"\b(sms|messages?|texts?|texting|whatsapp|mms|dlr|deliver(?:y|ed))\b", re.I)


def _channel_hint(text: str) -> str | None:
    """Cheap content hint: voice vs sms. None when ambiguous/none — the account-ask
    then asks Nihal to specify. Nearly free; ambiguous residue is one word in the
    reply. Never DEFAULTS to voice (that would confidently mis-investigate an SMS
    complaint)."""
    t = text or ""
    sms, voice = bool(_SMS_HINT_RE.search(t)), bool(_VOICE_HINT_RE.search(t))
    if sms and not voice:
        return "sms"
    if voice and not sms:
        return "voice"
    return None


def _probe_channel(uuid: str, account_id) -> str | None:
    """DEFINITIVE channel for a UUID: whichever account-scoped table has the row.
    Structured lookup, not a guess (same principle as pricing/account-injection)."""
    if redshift_tools.get_call_by_uuid(uuid, account_id).get("found"):
        return "voice"
    if redshift_tools.get_sip_trunk_call_detail(uuid, account_id).get("found"):
        return "voice"
    if redshift_tools.get_message_by_uuid(uuid, account_id).get("found"):
        return "sms"
    return None


def run_debug_case(thread: dict, *, channel_hint: str | None = None, dry_run: bool = True) -> dict:
    """Auto-trigger entry (channel-agnostic). Posts the account-ask to #debugging
    with a channel HINT (inferred), marks the case `auto` so resume_account resolves
    the channel (reply token > UUID probe > hint > ask) instead of defaulting, and
    tells Nihal he can bounce it ('not a debugging case'). Still asks for the
    account_id — no sender→account enrichment exists."""
    email_text = _email_text(thread)
    sender = _sender(thread)
    uuid = _latest_uuid(thread)
    hint = channel_hint or _channel_hint(email_text)
    ch_line = (f"Channel: investigating as *{hint.upper()}* — inferred; reply 'sms'/'voice' to redirect."
               if hint else
               "Channel: *unknown* — reply e.g. '<account_id> voice' or '<account_id> sms'.")
    ask = (f"*New debugging case (auto)* — {thread.get('subject') or '(no subject)'} — from {sender}\n"
           + (f"UUID in mail: `{uuid}`\n" if uuid else "No UUID in mail.\n")
           + ch_line + "\nReply the *account_id* (optionally + 'voice'/'sms'), "
           "or *'not a debugging case'* to send it back to normal handling.")
    art = {"dry_run": dry_run, "has_uuid": bool(uuid), "channel_hint": hint, "account_ask_post": ask}
    if dry_run:
        return art
    ts = _post(DEBUG_CHANNEL, ask).get("ts")
    db.kv_set(_ACCT_PREFIX + str(ts), {
        "thread": thread, "email_text": email_text, "sender": sender, "uuid": uuid,
        "channel_hint": hint, "auto": True})
    art["debug_ts"] = ts
    art["status"] = f"posted auto debugging account-ask to #debugging (thread {ts})"
    return art


def maybe_debug_autotrigger(thread: dict, classification: dict) -> bool:
    """Flag-gated (worker checks DEBUG_AUTODETECT) auto-trigger, called AFTER
    classify. Tier 1: a UUID in the customer's latest message → debugging. Tier 2:
    llm.is_debugging_case (a specific account-data-diagnosable traffic failure) →
    debugging. Returns True if it routed (posted the account-ask); the worker then
    discards the generic draft and returns. Any exception propagates to the worker's
    fail-open handler (fall through to normal drafting)."""
    if _latest_uuid(thread):                                   # Tier 1 — near-unambiguous
        run_debug_case(thread, dry_run=False)
        return True
    gate = llm.is_debugging_case(_email_text(thread), classification)   # Tier 2
    if gate.get("is_debugging"):
        run_debug_case(thread, channel_hint=gate.get("channel_hint"), dry_run=False)
        return True
    return False


def _run_slice1_live(thread: dict, account_id, uuid: str, *, header: str = "") -> dict:
    """Post the Slice-1 rail live: #debugging facts+interpretation, then #verify
    verify prompt + persist the the reviewer-pending case (resumed by resume())."""
    art = _slice1_artifacts(thread, account_id, uuid, header=header)
    _post(DEBUG_CHANNEL, art["debugging_post"])
    if not art.get("resolved"):
        return {"branch": "unresolved", "call_uuid": uuid}
    verify_ts = _post(VERIFY_CHANNEL, art["verify_prompt"]).get("ts")
    db.kv_set(_PENDING_PREFIX + str(verify_ts), {
        "account_id": account_id, "uuid": uuid, "facts": art["grounded_facts"],
        "interpretation": art.get("interpretation", ""), "case_title": art["case_title"],
        "email_text": _email_text(thread), "sender": _sender(thread), "thread": thread})
    return {"branch": "resolved", "call_uuid": uuid, "verify_ts": verify_ts}


def resume_account(debug_thread_ts, reply_text) -> dict | None:
    """Resume an account-ask: claim the pending case, parse the account_id (a lone
    number), then branch — UUID in the mail → Slice-1; else account-wide (clear →
    drill a representative call; ambiguous → customer-ask draft + suspend)."""
    key = _ACCT_PREFIX + str(debug_thread_ts)
    pending = db.claim_pending_case(key)
    if not pending:
        return None

    # Bounce (the fuzzy-boundary backstop): 'not a debugging case' -> send back to
    # normal drafting. Exactly ONE draft results — persist_processing's idempotency
    # anchor collapses any replay; the debug route persisted no draft to orphan.
    if _BOUNCE_RE.search(reply_text or ""):
        from app import worker
        worker.draft_and_post_normally(pending["thread"])
        _post(DEBUG_CHANNEL, "↩️ Bounced to normal handling — drafted as ordinary mail.",
              thread_ts=str(debug_thread_ts))
        return {"branch": "bounced_to_normal"}

    account_id = _parse_account_id(reply_text)
    if account_id is None:
        db.kv_set(key, pending)  # restore — don't lose the case; ask again
        _post(DEBUG_CHANNEL, "Couldn't read an account_id there — reply with just the number "
              "(or 'not a debugging case').", thread_ts=str(debug_thread_ts))
        return {"branch": "reask_account"}
    thread, uuid = pending["thread"], pending.get("uuid")

    # Resolve the channel: explicit reply token > invoke-preset channel > UUID probe
    # (definitive) > content hint. For an AUTO case that's still unknown, ASK (never
    # guess voice — that mis-investigates SMS complaints); invoke entries keep their
    # prior voice default.
    tok = _CHANNEL_TOKEN_RE.search(reply_text or "")
    channel = (tok.group(1).lower() if tok else None) or pending.get("channel")
    probed = None
    if channel is None and uuid:
        probed = _probe_channel(uuid, account_id)
        channel = probed
    if channel is None:
        channel = pending.get("channel_hint")
    if channel is None:
        if pending.get("auto"):
            db.kv_set(key, pending)  # keep the case; resolve channel first
            _post(DEBUG_CHANNEL, "Which channel? Reply '<account_id> voice' or "
                  "'<account_id> sms'.", thread_ts=str(debug_thread_ts))
            return {"branch": "reask_channel"}
        channel = "voice"  # invoke-entry default (unchanged)
    if probed and pending.get("channel_hint") and probed != pending["channel_hint"]:
        _post(DEBUG_CHANNEL, f"(UUID resolves to *{probed}* — overriding the inferred "
              f"{pending['channel_hint']} hint.)", thread_ts=str(debug_thread_ts))

    if channel == "sms":
        def _sms_hold(note: str) -> dict:
            # HOLD restores the dbgacct pending (re-paste / retry), dampened.
            already = bool(pending.get("held_warned"))
            db.kv_set(key, {**pending, "held_warned": True})
            if not already:
                _post(DEBUG_CHANNEL, note, thread_ts=str(debug_thread_ts))
            return {"branch": "held", "warned": not already}

        return _sms_investigate_and_emit(
            thread, account_id, uuid,
            pending.get("email_text") or _email_text(thread), on_hold=_sms_hold)

    if uuid:
        return _run_slice1_live(thread, account_id, uuid)

    aw = investigate_account_wide(thread, account_id)
    header = _histogram_header(aw)

    def _hold(note: str) -> dict:
        # Held verdicts (unavailable / no_data): note to Nihal, NO customer-ask,
        # NO suspend. Dampened — only the FIRST hold of a case warns; repeat holds
        # (a long outage, or a re-paste of the same wrong account) stay silent.
        # Pending is restored so a retry (outage) or a corrected re-paste works.
        already = bool(pending.get("held_warned"))
        db.kv_set(key, {**pending, "held_warned": True})
        if not already:
            _post(DEBUG_CHANNEL, note, thread_ts=str(debug_thread_ts))
        return {"branch": aw["verdict"], "warned": not already}

    if aw["verdict"] == "unavailable":
        return _hold(f"⚠️ Couldn't reach the call data for account {account_id} "
                     f"({', '.join(aw.get('errors', []))}) — HELD for retry, NOT asking the customer.")
    if aw["verdict"] == "no_data":
        return _hold(f"ℹ️ No calls for account {account_id} in "
                     f"{aw['window']['start']}→{aw['window']['end']} — wrong account or wrong window? "
                     f"HELD (re-paste a corrected account); NOT asking the customer.")
    if aw["verdict"] == "clear" and aw["representative_uuid"]:
        _post(DEBUG_CHANNEL, f"{header}\n*Branch:* PATTERN-CLEAR → drilling `{aw['representative_uuid']}`")
        return _run_slice1_live(thread, account_id, aw["representative_uuid"], header=header)

    # AMBIGUOUS → ask the customer for specifics, then suspend on the Gmail thread.
    _post(DEBUG_CHANNEL, f"*Account-wide — {pending.get('sender')}*\n{header}\n"
                         f"*Branch:* AMBIGUOUS → asking the customer for specifics (case suspended).")
    from app import slack_approval
    draft = _strip_internal(llm.debug_customer_ask(pending["email_text"]), account_id)
    ids = db.persist_processing(thread, {"intent": "technical_support", "customer_name": None,
                                         "summary": "debug: ask customer for specifics"},
                                draft, source="debug", reply_context=thread.get("reply_context"),
                                artifact_key=db.build_artifact_key(
                                    thread.get("thread_id"), "debug-ask", account_id))
    slack_approval.post_draft_once(pending["email_text"], draft, ids["draft_id"])
    thread_id = thread.get("thread_id")
    db.kv_set(_CUST_PREFIX + str(thread_id), {
        "account_id": account_id, "thread": thread, "email_text": pending["email_text"],
        "asked": "uuid/timestamp/destination"})
    return {"branch": "ambiguous", "customer_draft": draft, "draft_id": ids["draft_id"],
            "suspended_thread": thread_id}
