"""Meeting follow-up from Gemini meeting-notes notification mail (Phase 5-ish).

Flow: a `gemini-notes@google.com` "Notes: ..." mail arrives -> parse the notes
straight from the notification body (probe-proven: summary + topics + attributed
next steps are IN the text/plain body, no Docs/Drive scope needed) -> pick the
thread to reply on from the meeting's CALENDAR attendees (external only) -> draft
a follow-up grounded ONLY in the notes -> existing approval gate. Nothing auto-
sent; trigger is invoke-only until wired into the worker.

Hard safety properties (both enforced in CODE, not just the prompt):
- GROUNDED-OR-THIN: the draft is built only from the parsed notes; ungrounded or
  thin -> a code-assembled thin/grounded fallback. Never enriched.
- RECORDED-NOT-AGREED: next steps are relayed as what the notes RECORDED, never as
  asserted commitments. A commitment-phrasing guard (incl. first-person-plural
  "we agreed" / "as discussed and agreed" / "per our agreement") rejects the LLM
  draft to the fallback if it slips — a misattributed commitment is the 1078 class
  laundered through Gemini's machine attribution.

Notes-generation-failure ("Problem with the notes" from meetings-noreply) and
unparseable/empty notes and unmatchable meetings are VISIBLE holds, never silent.
"""
from __future__ import annotations

import re

from app import calendar_client, gmail_client, llm

NOTES_SENDER = "gemini-notes@google.com"
NOTES_FAILED_SENDER = "meetings-noreply@google.com"
MY_DOMAIN = "@plivo.com"

_QUOTE = "\"“”‘’'"           # straight + curly quotes
_TITLE_RE = re.compile(rf"Notes:\s*[{_QUOTE}](.+?)[{_QUOTE}]")
_TITLE_BODY_RE = re.compile(rf"Notes from\s*[{_QUOTE}](.+?)[{_QUOTE}]")
_DATE_RE = re.compile(r"[{q}]\s*(\w{{3}}\s+\d{{1,2}},\s*\d{{4}})".format(q=_QUOTE))
_GEN_RE = re.compile(r"auto-generated on (.+?) and may contain", re.I)
_STEP_RE = re.compile(r"^\s*\[(.+?)\]\s*(.+)$")

# Everything from the first footer marker on is boilerplate, dropped before parse.
_FOOTER_MARKERS = ("Meeting records", "Is the Next Steps section", "Not Useful Email",
                   "Was this summary", "Google LLC")
_HEADER_PREFIXES = ("Notes from", "These notes have been sent", "Open meeting notes",
                    "The content was auto-generated")

# Commitment-assertion phrasing — the ONE catastrophic failure, blocked in code
# regardless of the model. Includes first-person-plural forms the model reaches for.
_COMMITMENT_PATTERNS = [
    r"\byou agreed\b", r"\byou've agreed\b", r"\byou have agreed\b",
    r"\bwe agreed\b", r"\bwe've agreed\b", r"\bwe have agreed\b",
    r"\bas agreed\b", r"\bas discussed and agreed\b", r"\bas we agreed\b",
    r"\bper our agreement\b", r"\bour agreement\b", r"\bthe agreement we\b",
    r"\byou committed\b", r"\bwe committed\b", r"\byou promised\b",
    r"\bas promised\b", r"\byou said you'd\b", r"\byou said you would\b",
    r"\byou agreed to\b", r"\bwe both agreed\b",
]
_COMMITMENT_RE = re.compile("|".join(_COMMITMENT_PATTERNS), re.I)

_SIGNOFF = "Best regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo"


# --------------------------------------------------------------------------- #
# Parse (deterministic — no LLM, minimal fabrication surface)
# --------------------------------------------------------------------------- #
def _paragraphs(lines: list[str]) -> list[str]:
    """Group hard-wrapped lines into paragraphs (blank line separates), joining
    wrapped lines with a space and collapsing whitespace."""
    paras, cur = [], []
    for ln in lines:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


def _parse_steps(lines: list[str]) -> list[dict]:
    """`[Owner] action: detail` items; a step continues until the next `[` line."""
    steps, owner, buf = [], None, []
    def flush():
        if owner is not None and buf:
            steps.append({"owner": owner, "text": re.sub(r"\s+", " ", " ".join(buf)).strip()})
    for ln in lines:
        m = _STEP_RE.match(ln)
        if m:
            flush(); owner, buf = m.group(1).strip(), [m.group(2).strip()]
        elif ln.strip() and owner is not None:
            buf.append(ln.strip())
    flush()
    return steps


def _label_index(lines: list[str], label: str) -> int:
    for i, ln in enumerate(lines):
        if ln.strip().lower() == label.lower():
            return i
    return -1


def parse_notes(subject: str, body: str) -> dict:
    """Parse the notification body -> {title, date, generated_at, summary, topics,
    next_steps[]}. Robust to missing sections (drives 'thin')."""
    tm = _TITLE_RE.search(subject or "") or _TITLE_BODY_RE.search(body or "")
    title = tm.group(1).strip() if tm else (subject or "").strip()
    dm = _DATE_RE.search(subject or "")
    gm = _GEN_RE.search(body or "")

    cut = len(body)
    for mk in _FOOTER_MARKERS:
        i = body.find(mk)
        if i != -1:
            cut = min(cut, i)
    lines = body[:cut].splitlines()

    si, ni = _label_index(lines, "Summary"), _label_index(lines, "Suggested next steps")
    summary, topics, steps = "", "", []
    if si != -1:
        paras = _paragraphs(lines[si + 1:(ni if ni != -1 else len(lines))])
        if paras:
            summary, topics = paras[0], "\n".join(paras[1:])
    if ni != -1:
        steps = _parse_steps(lines[ni + 1:])
    return {"title": title, "date": dm.group(1) if dm else None,
            "generated_at": gm.group(1).strip() if gm else None,
            "summary": summary.strip(), "topics": topics.strip(), "next_steps": steps}


def is_thin(parsed: dict) -> bool:
    return not parsed.get("summary") and not parsed.get("next_steps") and not parsed.get("topics")


def render_notes(parsed: dict) -> str:
    """Grounding source + LLM input: the parsed notes as plain text."""
    out = [f"Meeting: {parsed.get('title') or '(untitled)'}"]
    if parsed.get("summary"):
        out.append(f"Summary: {parsed['summary']}")
    if parsed.get("topics"):
        out.append(f"Topics discussed:\n{parsed['topics']}")
    if parsed.get("next_steps"):
        out.append("Recorded next steps (as captured by the notes):")
        out += [f"- [{s['owner']}] {s['text']}" for s in parsed["next_steps"]]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Thread selection (routing metadata from CALENDAR attendees — not grounding)
# --------------------------------------------------------------------------- #
def _find_meeting_event(title: str, date_str: str | None, cal_service):
    """Match the meeting to a calendar event on its date; returns the event or None.
    Identity comes from the event's attendee list (ground truth), never guessed."""
    from datetime import datetime, timedelta
    day = None
    if date_str:
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                day = datetime.strptime(date_str, fmt); break
            except ValueError:
                pass
    if day is None:
        return None
    # IST day window, RFC3339
    tmin = f"{day:%Y-%m-%d}T00:00:00+05:30"
    tmax = f"{day:%Y-%m-%d}T23:59:59+05:30"
    resp = calendar_client._execute(cal_service.events().list(
        calendarId="primary", timeMin=tmin, timeMax=tmax, singleEvents=True,
        orderBy="startTime", maxResults=50))
    events = resp.get("items", [])

    def norm(s):
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())
    want = norm(title)
    exact = [e for e in events if norm(e.get("summary")) == want]
    if exact:
        return exact[0]
    partial = [e for e in events if want and (want in norm(e.get("summary"))
                                              or norm(e.get("summary")) in want)]
    return partial[0] if partial else None


def _external_attendees(event: dict, my_email: str) -> list[dict]:
    out = []
    for a in event.get("attendees", []) or []:
        email = (a.get("email") or "").lower()
        if not email or a.get("resource"):
            continue
        if email == (my_email or "").lower() or email.endswith(MY_DOMAIN):
            continue
        out.append({"email": email, "name": a.get("displayName")})
    return out


def _most_recent_thread_with(emails: list[str], service) -> dict | None:
    if not emails:
        return None
    q = "(" + " OR ".join(f"from:{e} OR to:{e}" for e in emails) + ")"
    r = gmail_client._execute(service.users().messages().list(userId="me", maxResults=1, q=q))
    msgs = r.get("messages") or []
    if not msgs:
        return None
    thread_id = msgs[0]["threadId"]
    thread = gmail_client.fetch_thread(thread_id, service=service)
    return {"thread_id": thread_id, "subject": thread.get("subject"),
            "reply_context": thread.get("reply_context")}


def select_thread(parsed: dict, *, service, cal_service, my_email: str) -> dict:
    """Pick the reply target. thread (most recent with an external attendee) ->
    fresh (any thread-selection MISS: no prior thread, no external attendee, or no
    calendar match). Every result carries a `reason` for the approval card, and a
    fresh miss says WHY. select_thread never holds — holds are reserved for
    notes-failed / unparseable notes; a thread-miss is a visible fresh draft the
    reviewer routes."""
    event = _find_meeting_event(parsed.get("title"), parsed.get("date"), cal_service)
    if not event:
        return {"mode": "fresh", "recipient_email": None, "recipient_name": None, "external": [],
                "reason": f"couldn't match a calendar event for '{parsed.get('title')}' on "
                          f"{parsed.get('date')} — fresh mail; set the recipient before sending."}
    external = _external_attendees(event, my_email)
    if not external:
        return {"mode": "fresh", "event_id": event.get("id"), "recipient_email": None,
                "recipient_name": None, "external": [],
                "reason": "meeting had no external attendee (internal-only) — fresh mail; "
                          "confirm a follow-up is actually wanted."}
    primary = external[0]
    name = primary.get("name")  # displayName preferred; None (not the email local-part) if absent
    hit = _most_recent_thread_with([a["email"] for a in external], service)
    if hit:
        return {"mode": "thread", "thread_id": hit["thread_id"], "subject": hit["subject"],
                "reply_context": hit.get("reply_context"),
                "recipient_email": primary["email"], "recipient_name": name, "external": external,
                "reason": f"replying on the most recent thread with external attendee "
                          f"{primary['email']} — \"{hit['subject']}\""}
    return {"mode": "fresh", "recipient_email": primary["email"], "recipient_name": name,
            "external": external,
            "reason": f"no prior thread with external attendee {primary['email']} — fresh mail"}


# --------------------------------------------------------------------------- #
# Draft (constrained LLM + groundedness gate + commitment guard, thin fallback)
# --------------------------------------------------------------------------- #
def commitment_violations(text: str) -> list[str]:
    return sorted(set(m.group(0) for m in _COMMITMENT_RE.finditer(text or "")))


def _grounded_fallback(parsed: dict, name: str | None) -> str:
    """Code-assembled draft straight from the notes — grounded by construction,
    used for thin notes AND when the LLM draft trips a gate. Relays next steps as
    RECORDED. Never enriches: no notes content -> a bare thanks."""
    hi = f"Hi {name}," if name else "Hi,"
    parts = ["Thanks for taking the time to meet — it was good speaking with you."]
    if parsed.get("summary"):
        parts.append(f"For a quick recap, the notes captured: {parsed['summary']}")
    steps = parsed.get("next_steps") or []
    if steps:
        bullets = "\n".join(f"- {s['text']} (noted for {s['owner']})" for s in steps)
        parts.append("The notes also captured these next steps:\n" + bullets)
    parts.append("Please let me know if anything needs correcting or if I can help further.")
    return f"{hi}\n\n" + "\n\n".join(parts) + f"\n\n{_SIGNOFF}"


def _strip_subject_line(text: str) -> str:
    """Drop a leading 'Subject: …' line (+ following blanks) the model sometimes
    prepends — for a reply-on-thread the subject is inherited, so it must not sit
    in the body."""
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("subject:"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def build_followup(parsed: dict, recipient_name: str | None) -> tuple[str, list[dict]]:
    """Grounded-or-thin follow-up. Returns (draft_text, flags)."""
    if is_thin(parsed):
        return _grounded_fallback(parsed, recipient_name), [
            {"type": "meeting_thin_notes", "text": "Notes were thin — a brief thanks was drafted; not enriched."}]

    notes_text = render_notes(parsed)
    draft = _strip_subject_line(llm.draft_meeting_followup(notes_text, recipient_name))

    # Guard 1 (deterministic): asserted-commitment phrasing -> fallback.
    bad = commitment_violations(draft)
    if bad:
        return _grounded_fallback(parsed, recipient_name), [
            {"type": "meeting_commitment_blocked",
             "text": f"LLM draft used commitment phrasing {bad}; replaced with a recorded-not-agreed "
                     f"fallback. Verify tone before sending."}]

    # Guard 2 (groundedness): every claim must trace to the notes, else fallback.
    grounded = llm.rag_groundedness(draft, [{"title": "Meeting notes", "content": notes_text}])
    if not grounded.get("grounded"):
        unsup = ", ".join(grounded.get("unsupported_claims", [])[:3])
        return _grounded_fallback(parsed, recipient_name), [
            {"type": "meeting_ungrounded",
             "text": f"LLM draft had claims not in the notes ({unsup}); replaced with a grounded fallback."}]

    from app import draft as draft_mod
    return draft, draft_mod.flag_unverified_specifics(draft)


# --------------------------------------------------------------------------- #
# Orchestration (invoke-only; dry_run returns artifacts, no posts)
# --------------------------------------------------------------------------- #
def handle_notes_mail(subject: str, body: str, *, service, cal_service, my_email: str,
                      dry_run: bool = True) -> dict:
    """Parse -> select thread -> grounded draft. dry_run returns artifacts only."""
    parsed = parse_notes(subject, body)
    if is_thin(parsed) and not parsed.get("summary"):
        # empty/unparseable body -> visible hold, never a silent skip/crash.
        return {"status": "hold", "reason": "notes empty or unparseable — manual follow-up.",
                "parsed": parsed}
    # select_thread returns thread | fresh (never hold — a thread-miss is a visible
    # fresh draft the reviewer routes; the only holds are notes-failed/unparseable).
    selection = select_thread(parsed, service=service, cal_service=cal_service, my_email=my_email)
    draft, flags = build_followup(parsed, selection.get("recipient_name"))
    # Surface WHICH thread was chosen and WHY on the approval card (first flag).
    flags = [{"type": "meeting_thread_selection",
              "text": f"Reply target [{selection['mode']}]: {selection['reason']}"}] + flags
    art = {"status": "drafted", "parsed": parsed, "selection": selection,
           "draft": draft, "flags": flags}
    if dry_run:
        return art
    # LIVE: reply-on-thread (or fresh) through the existing approval gate.
    from app import db, slack_approval
    reply_context = selection.get("reply_context") or {
        "to": selection["recipient_email"],
        "subject": f"Following up on our call — {parsed.get('title')}"}
    thread = {"thread_id": selection.get("thread_id"),
              "messages": [{"from": selection["recipient_email"], "body": body}],
              "reply_context": reply_context, "subject": selection.get("subject")}
    ids = db.persist_processing(
        thread, {"intent": "meeting_followup", "customer_name": selection.get("recipient_name"),
                 "summary": f"Post-meeting follow-up — {parsed.get('title')}"},
        draft, source="gmail", reply_context=reply_context,
        # Distinct artifact key so a follow-up never collides with a prior
        # customer-mail hold on the SAME reply thread; stable per meeting (title +
        # generated-at) so notes-mail replays still dedup.
        artifact_key=db.build_artifact_key(
            selection.get("thread_id") or "fresh", "meeting",
            f"{parsed.get('title')}|{parsed.get('generated_at')}"))
    slack_approval.post_draft_once(render_notes(parsed), draft, ids["draft_id"], flags=flags)
    art["draft_id"] = ids["draft_id"]
    return art


def note_notes_failed(subject: str) -> str:
    """The meetings-noreply 'Problem with the notes' case — a VISIBLE note, no draft."""
    tm = _TITLE_RE.search(subject or "")
    title = tm.group(1).strip() if tm else (subject or "")
    return f"⚠️ Gemini notes FAILED for meeting '{title}' — no follow-up drafted (notes unavailable)."


# --------------------------------------------------------------------------- #
# Detector + worker-entry (carve-out routes these BEFORE the is_automated skip)
# --------------------------------------------------------------------------- #
def is_notes_mail(from_hdr: str | None) -> bool:
    return NOTES_SENDER in (from_hdr or "").lower()


def is_notes_failed_mail(from_hdr: str | None, subject: str | None) -> bool:
    return (NOTES_FAILED_SENDER in (from_hdr or "").lower()
            and "problem with the notes" in (subject or "").lower())


def post_note(text: str) -> None:
    """Post a plain VISIBLE note (notes-failed / unparseable hold) to the approval
    channel — a hold must be seen, never a silent skip."""
    import os
    from slack_sdk import WebClient
    token, ch = os.getenv("SLACK_BOT_TOKEN"), os.getenv("SLACK_APPROVAL_CHANNEL")
    if not (token and ch):
        raise RuntimeError("SLACK_BOT_TOKEN / SLACK_APPROVAL_CHANNEL not set")
    WebClient(token=token).chat_postMessage(channel=ch, text=text[:39000])


def handle_gmail_message(mid: str, from_hdr: str | None, subject: str | None, *,
                         service, cal_service, my_email: str, dry_run: bool = False) -> dict:
    """Carve-out entry for one Gmail message already identified as meeting-related.
    'Problem with the notes' → visible note (no draft). gemini-notes → fetch body →
    handle_notes_mail; a hold outcome also posts a visible note. Returns an artifact."""
    if is_notes_failed_mail(from_hdr, subject):
        note = note_notes_failed(subject)
        if not dry_run:
            post_note(note)
        return {"status": "notes_failed", "note": note}

    full = gmail_client._execute(service.users().messages().get(userId="me", id=mid, format="full"))
    body = gmail_client._extract_body(full.get("payload", {}))
    art = handle_notes_mail(subject or "", body, service=service, cal_service=cal_service,
                            my_email=my_email, dry_run=dry_run)
    if art.get("status") == "hold" and not dry_run:
        title = (art.get("parsed") or {}).get("title")
        post_note(f"⚠️ Meeting follow-up HELD (meeting '{title}'): {art.get('reason')}")
    return art
