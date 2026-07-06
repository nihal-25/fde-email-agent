"""Gmail integration: pull-based ingest + threaded send (Milestone 5).

Design (per CLAUDE.md):
- PULL, not push. Gmail `users.watch` publishes change notifications to a
  Pub/Sub topic; we pull them from a subscription (app/worker.py). No public
  inbound endpoint.
- The watch expires (≤7 days); `ensure_watch()` renews it (call daily).
- Sending happens ONLY from the human-approval path. This module exposes a
  send function, but nothing here sends on its own.

Auth: OAuth installed-app flow. The client secret JSON (downloaded from Google
Cloud) and the resulting token are both gitignored. Scope is gmail.modify,
which covers reading threads, watch, and sending.

Thread parsing is validated against a REAL payload (see `dump-latest` CLI
below) rather than guessed — run it once after OAuth and we confirm the parser.

CLI helpers:
    python -m app.gmail_client auth          # one-time OAuth, writes token.json
    python -m app.gmail_client whoami        # print the authorized address
    python -m app.gmail_client watch         # register/renew the Pub/Sub watch
    python -m app.gmail_client stop           # stop the watch
    python -m app.gmail_client dump-latest    # print one raw message payload
    python -m app.gmail_client dump-query <q>  # same, for a Gmail search query
"""

from __future__ import annotations

import base64
import errno
import os
import re
import socket
import sys
import threading
import time
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

# httplib2 (under googleapiclient) is NOT thread-safe: concurrent calls on one
# service corrupt the shared SSL socket. Pub/Sub delivers notifications on
# multiple threads, so every Gmail API call goes through this lock.
_API_LOCK = threading.Lock()

# Local-socket errnos worth a retry (stale/broken persistent connection, the
# transient network churn seen on this laptop). EADDRNOTAVAIL = the Errno 49 blip.
_TRANSIENT_ERRNOS = {errno.EADDRNOTAVAIL, errno.ECONNRESET, errno.EPIPE, errno.ETIMEDOUT}


def is_transient_error(exc: Exception) -> bool:
    """True only for errors worth a retry / holding the ingest cursor — network
    blips and server-side 5xx.

    Deliberately NARROW: any error NOT recognized here is treated as PERMANENT by
    callers (skip + advance the cursor), so a permanent/unknown error can never
    wedge the pipeline by being mistaken for a retryable one.
    """
    if isinstance(exc, (BrokenPipeError, ConnectionError, TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS:
        return True
    from googleapiclient.errors import HttpError
    if isinstance(exc, HttpError):
        status = getattr(getattr(exc, "resp", None), "status", None) or getattr(exc, "status_code", None)
        try:
            return int(status) in (500, 502, 503, 504)
        except (TypeError, ValueError):
            return False
    return False


def _execute(request):
    """Execute a googleapiclient request under the global Gmail API lock, with a
    single retry on a transient error (a stale httplib2 socket reopens on retry)."""
    with _API_LOCK:
        try:
            return request.execute()
        except Exception as exc:
            if is_transient_error(exc):
                time.sleep(0.5)
                return request.execute()  # one retry; the stale connection reopens
            raise

# gmail.modify = read + modify + send + watch (no permanent delete).
# calendar.events + calendar.freebusy add Google Calendar read (free/busy) and
# event creation for the meeting-scheduling case. One token, shared by the Gmail
# and Calendar service builders (see get_credentials). Changing this list
# invalidates an existing token.json — re-run `python -m app.gmail_client auth`.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
]

CLIENT_SECRETS = os.getenv("GMAIL_OAUTH_CLIENT_SECRETS", "credentials.json")
TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")
PUBSUB_TOPIC = os.getenv("GMAIL_PUBSUB_TOPIC")

# KV keys for worker state (stored in Postgres via app.db).
KV_WATCH = "gmail_watch"          # {"history_id": str, "expiration_ms": int}
KV_LAST_HISTORY = "gmail_last_history_id"  # {"history_id": str}


# --- Auth -------------------------------------------------------------------

def get_credentials():
    """Load cached OAuth credentials, refreshing or running the flow as needed.

    Shared by the Gmail and Calendar service builders so there is exactly one
    token covering all SCOPES.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not os.path.exists(CLIENT_SECRETS):
            raise RuntimeError(
                f"OAuth client secrets not found at '{CLIENT_SECRETS}'. Download "
                "the OAuth client JSON from Google Cloud and set "
                "GMAIL_OAUTH_CLIENT_SECRETS."
            )
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
        # Opens a browser for consent; runs a tiny local server for the callback.
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return creds


def get_service():
    from googleapiclient.discovery import build

    account = get_credentials()
    return build("gmail", "v1", credentials=account, cache_discovery=False)


def get_profile_email(service=None) -> str:
    service = service or get_service()
    return _execute(service.users().getProfile(userId="me"))["emailAddress"]


# --- Watch (Pub/Sub) --------------------------------------------------------

def register_watch(service=None) -> dict:
    """Register/refresh the mailbox watch on the INBOX -> Pub/Sub topic."""
    if not PUBSUB_TOPIC:
        raise RuntimeError("GMAIL_PUBSUB_TOPIC is not set (see .env).")
    service = service or get_service()
    resp = _execute(service.users().watch(
        userId="me",
        body={
            "topicName": PUBSUB_TOPIC,
            "labelIds": ["INBOX"],
            "labelFilterBehavior": "INCLUDE",
        },
    ))
    # resp: {"historyId": "...", "expiration": "<ms epoch>"}
    from app import db

    db.kv_set(KV_WATCH, {"history_id": resp["historyId"], "expiration_ms": int(resp["expiration"])})
    # Seed the last-processed history id if we don't have one yet.
    if db.kv_get(KV_LAST_HISTORY) is None:
        db.kv_set(KV_LAST_HISTORY, {"history_id": resp["historyId"]})
    return resp


def stop_watch(service=None) -> None:
    service = service or get_service()
    _execute(service.users().stop(userId="me"))


def ensure_watch(service=None, *, now_ms: int | None = None, renew_within_ms: int = 24 * 3600 * 1000) -> dict:
    """Register the watch if missing or expiring within `renew_within_ms`.

    `now_ms` must be supplied by the caller (the worker), since wall-clock time
    is injected rather than read here. Returns the current watch state.
    """
    from app import db

    state = db.kv_get(KV_WATCH)
    if state is None:
        return register_watch(service)
    if now_ms is not None and state["expiration_ms"] - now_ms <= renew_within_ms:
        return register_watch(service)
    return state


# --- History (find new messages) --------------------------------------------

def get_message(message_id: str, service=None, *, fmt: str = "metadata",
                metadata_headers: list[str] | None = None) -> dict:
    """Fetch a single message (thread-safe via the API lock)."""
    service = service or get_service()
    kwargs = {"userId": "me", "id": message_id, "format": fmt}
    if metadata_headers is not None:
        kwargs["metadataHeaders"] = metadata_headers
    return _execute(service.users().messages().get(**kwargs))


def list_new_message_ids(start_history_id: str, service=None) -> tuple[list[str], str | None]:
    """Return (message_ids_added_to_INBOX, latest_history_id) since the given id."""
    service = service or get_service()
    message_ids: list[str] = []
    latest_history_id = start_history_id
    page_token = None
    while True:
        resp = _execute(service.users().history().list(
            userId="me",
            startHistoryId=start_history_id,
            historyTypes=["messageAdded"],
            labelId="INBOX",
            pageToken=page_token,
        ))
        for record in resp.get("history", []):
            for added in record.get("messagesAdded", []):
                msg = added.get("message", {})
                if "INBOX" in msg.get("labelIds", []):
                    message_ids.append(msg["id"])
        if "historyId" in resp:
            latest_history_id = resp["historyId"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    # De-dup while preserving order.
    seen = set()
    deduped = [m for m in message_ids if not (m in seen or seen.add(m))]
    return deduped, latest_history_id


# --- Automated-sender pre-filter --------------------------------------------
#
# The worker must NOT draft replies to bots/newsletters/bounces. We skip a
# message when it looks automated. Header signals (RFC 3834 Auto-Submitted,
# bulk Precedence, List-* headers) are authoritative; sender local-part hints
# and bulk Gmail categories are secondary. Real customer mail rarely carries
# any of these.

_AUTOMATED_LOCALPART_HINTS = (
    "no-reply", "noreply", "no_reply", "donotreply", "do-not-reply", "do_not_reply",
    "mailer-daemon", "postmaster", "bounce", "bounces", "notification", "notifications",
    "newsletter", "mailer", "automated", "auto-confirm", "no.reply",
    # Role / automation aliases (a relay or integration account, not a person).
    "automation", "integrations",
)
# Clearly non-customer Gmail categories. UPDATES is intentionally NOT skipped on
# label alone (it can hold real transactional mail) — header signals catch the
# automated ones.
_SKIP_CATEGORIES = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"}

# VERP / bulk-ESP envelope senders. Automated mail almost always sends from a
# bounce-handling envelope address even when the visible From name is human
# (e.g. "Fred from Fireflies.ai" <fred@fireflies.ai> with a Return-Path of
# bounces+1696684-...=plivo.com@send.fireflies.ai). Two near-universal tells:
#   * the recipient address is encoded into the local-part with '=' (VERP), and
#   * common ESP bounce prefixes (bounce/bounces/prvs/msprvs/fbl/...).
# Person-to-person mail never does this, so it generalizes across ESPs (SendGrid,
# Mailgun, SES, SparkPost, ...) without hardcoding any vendor.
_VERP_PREFIX_RE = re.compile(
    r"^(?:bounces?|prvs|msprvs\d*|fbl|sb|return|email[-_.]?bounces?)[-+=]", re.IGNORECASE
)


def _looks_like_verp(localpart: str) -> bool:
    if not localpart:
        return False
    # Recipient encoded into the envelope local-part (a@b.com -> ...=b.com).
    if "=" in localpart:
        return True
    return bool(_VERP_PREFIX_RE.match(localpart))


def _is_bounce_domain(domain: str) -> bool:
    """True if the envelope domain is a dedicated bounce/return-path subdomain,
    e.g. atlassian-bounces.atlassian.net, bounces.google.com, email.bounce.acme.io.
    Person-to-person mail's envelope domain is the real sending domain, never a
    bounce subdomain — so this generalizes across notification services."""
    return any("bounce" in label for label in domain.split(".") if label)


def _normalize_headers(headers) -> dict:
    """Accept Gmail's [{name,value}] list (or a dict) -> lowercased-name dict."""
    if isinstance(headers, dict):
        return {k.lower(): v for k, v in headers.items()}
    return {h.get("name", "").lower(): h.get("value", "") for h in (headers or [])}


def _localpart(address: str | None) -> str:
    if not address:
        return ""
    import email.utils

    _, addr = email.utils.parseaddr(address)
    return (addr.split("@", 1)[0] if addr else "").lower()


def _domain(address: str | None) -> str:
    if not address:
        return ""
    import email.utils

    _, addr = email.utils.parseaddr(address)
    return (addr.split("@", 1)[1] if "@" in addr else "").lower()


def is_automated(from_header: str | None, headers, label_ids=None) -> tuple[bool, str | None]:
    """Return (skip?, reason). reason is a short human-readable string."""
    h = _normalize_headers(headers)

    # 1) RFC 3834: Auto-Submitted present and not "no" => automated.
    auto = h.get("auto-submitted")
    if auto and auto.strip().lower() != "no":
        return True, f"Auto-Submitted: {auto.strip()}"

    # 2) Bulk precedence.
    prec = (h.get("precedence") or "").strip().lower()
    if prec in {"bulk", "list", "junk", "auto_reply"}:
        return True, f"Precedence: {prec}"

    # 3) Mailing-list / bulk headers.
    if "list-unsubscribe" in h or "list-id" in h:
        return True, "List-Unsubscribe/List-Id header present (bulk/list mail)"
    if "x-auto-response-suppress" in h:
        return True, "X-Auto-Response-Suppress present"

    # 3b) ESP feedback-loop id (AmazonSES, SendGrid, ...). Bulk/transactional
    #     senders stamp this for abuse feedback; person-to-person mail does not.
    if h.get("feedback-id") or h.get("feedback-ids"):
        return True, "Feedback-ID header present (bulk/ESP mail)"

    # 4) Sender local-part hints. Check the visible From/Reply-To AND the
    #    envelope sender (Return-Path / Sender): a friendly human From name
    #    often masks an automated envelope.
    for field in (from_header, h.get("reply-to"), h.get("return-path"), h.get("sender")):
        lp = _localpart(field)
        if any(hint in lp for hint in _AUTOMATED_LOCALPART_HINTS):
            return True, f"sender local-part looks automated: '{lp}'"

    # 5) VERP / bulk-ESP envelope sender (Return-Path / Sender). This is the
    #    signal that catches automated mail with a human-sounding From name.
    for field in (h.get("return-path"), h.get("sender")):
        lp = _localpart(field)
        if _looks_like_verp(lp):
            return True, f"VERP/bulk envelope sender: '{lp}'"

    # 5b) Envelope routed through a dedicated bounce subdomain (notification
    #     services: atlassian-bounces.atlassian.net, bounces.google.com, ...).
    for field in (h.get("return-path"), h.get("sender")):
        dom = _domain(field)
        if _is_bounce_domain(dom):
            return True, f"envelope via bounce subdomain: '{dom}'"

    # 6) Bulk Gmail categories.
    for lbl in (label_ids or []):
        if lbl in _SKIP_CATEGORIES:
            return True, f"Gmail category {lbl}"

    return False, None


# --- Parsing ----------------------------------------------------------------

def _header(headers: list[dict], name: str) -> str | None:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value")
    return None


def _b64url_decode(data: str) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode()).decode("utf-8", errors="replace")


def _extract_body(payload: dict) -> str:
    """Prefer text/plain; fall back to the first text part we find."""
    mime = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    if mime == "text/plain" and body_data:
        return _b64url_decode(body_data)
    # Recurse into parts.
    plain, anytext = None, None
    for part in payload.get("parts", []) or []:
        text = _extract_body(part)
        if not text:
            continue
        if part.get("mimeType") == "text/plain" and plain is None:
            plain = text
        elif anytext is None:
            anytext = text
    if plain is not None:
        return plain
    if anytext is not None:
        return anytext
    if body_data:  # single-part non-plain
        return _b64url_decode(body_data)
    return ""


def _parse_message(msg: dict) -> dict:
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "id": msg.get("id"),
        "message_id_header": _header(headers, "Message-ID"),
        "from": _header(headers, "From"),
        "to": _header(headers, "To"),
        "date": _header(headers, "Date"),
        "subject": _header(headers, "Subject"),
        "references": _header(headers, "References"),
        "label_ids": msg.get("labelIds", []),
        "body": _extract_body(payload).strip(),
    }


def fetch_thread(thread_id: str, service=None) -> dict:
    """Fetch a full Gmail thread and normalize it to our internal shape.

    Returns a dict matching samples/sample_email.json (subject + messages with
    from/date/body) PLUS a `reply_context` describing how to reply on-thread.
    """
    service = service or get_service()
    raw = _execute(service.users().threads().get(userId="me", id=thread_id, format="full"))
    parsed = [_parse_message(m) for m in raw.get("messages", [])]

    subject = next((m["subject"] for m in parsed if m.get("subject")), "(no subject)")
    messages = [
        {"from": m["from"], "to": m["to"], "date": m["date"], "body": m["body"]}
        for m in parsed
    ]

    # Reply context is derived from the LAST message in the thread. The
    # recipient is taken from our parsed data, never from model output.
    last = parsed[-1] if parsed else {}
    references = " ".join(
        x for x in [last.get("references"), last.get("message_id_header")] if x
    ).strip()
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    reply_context = {
        "gmail_thread_id": thread_id,
        "to": last.get("from"),
        "subject": reply_subject,
        "in_reply_to": last.get("message_id_header"),
        "references": references or None,
        "last_message_id": last.get("id"),
    }

    return {
        "thread_id": thread_id,
        "subject": subject,
        "messages": messages,
        "reply_context": reply_context,
    }


# --- Send (only from the human-approval path) -------------------------------

def send_reply(reply_context: dict, body_text: str, service=None) -> dict:
    """Send a threaded reply. Called ONLY after human approval (worker/slack).

    The recipient/subject/threading all come from `reply_context`, which our
    own ingest code populated from the verified inbound message.
    """
    if not reply_context or not reply_context.get("to"):
        raise RuntimeError("reply_context missing recipient; cannot send.")
    service = service or get_service()

    mime = MIMEText(body_text, "plain", "utf-8")
    mime["To"] = reply_context["to"]
    mime["Subject"] = reply_context.get("subject", "Re:")
    if reply_context.get("in_reply_to"):
        mime["In-Reply-To"] = reply_context["in_reply_to"]
    if reply_context.get("references"):
        mime["References"] = reply_context["references"]

    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    sent = _execute(service.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": reply_context.get("gmail_thread_id")},
    ))
    return sent  # {"id": ..., "threadId": ..., "labelIds": [...]}


# --- CLI helpers ------------------------------------------------------------

def _main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    if cmd == "auth":
        email = get_profile_email()
        print(f"Authorized as: {email}\nToken written to {TOKEN_FILE}")
    elif cmd == "whoami":
        print(get_profile_email())
    elif cmd == "watch":
        resp = register_watch()
        print(f"Watch registered. historyId={resp['historyId']} expiration_ms={resp['expiration']}")
    elif cmd == "stop":
        stop_watch()
        print("Watch stopped.")
    elif cmd in ("dump-latest", "dump-query"):
        import json

        service = get_service()
        if cmd == "dump-query":
            query = " ".join(argv[1:]).strip()
            if not query:
                print("Usage: dump-query <gmail search query>   e.g. dump-query from:fireflies.ai")
                return 2
            listing = _execute(service.users().messages().list(
                userId="me", maxResults=1, q=query))
        else:
            listing = _execute(service.users().messages().list(
                userId="me", maxResults=1, labelIds=["INBOX"]))
        if not listing.get("messages"):
            print("No matching messages found.")
            return 0
        mid = listing["messages"][0]["id"]
        msg = get_message(mid, service=service, fmt="full")

        out_path = "_real_message.json"
        with open(out_path, "w") as f:
            json.dump(msg, f, indent=2)

        # Print a structural skeleton only (no decoded body content).
        def skeleton(payload, depth=0):
            pad = "  " * depth
            has_data = bool(payload.get("body", {}).get("data"))
            size = payload.get("body", {}).get("size")
            print(f"{pad}- mimeType={payload.get('mimeType')} body.data={has_data} size={size}")
            for p in payload.get("parts", []) or []:
                skeleton(p, depth + 1)

        payload = msg.get("payload", {})
        header_names = [h.get("name") for h in payload.get("headers", [])]
        print(f"Full raw message written to {out_path} (gitignored).")
        print(f"id={msg.get('id')} threadId={msg.get('threadId')} labels={msg.get('labelIds')}")
        print("header names present:", header_names)
        # Values of headers relevant to the automated-sender pre-filter, so we
        # can diagnose is_automated() without exposing the body.
        diag = ("From", "Reply-To", "Sender", "Return-Path", "Auto-Submitted",
                "Precedence", "List-Id", "List-Unsubscribe", "X-Auto-Response-Suppress",
                "X-Mailer", "X-Autoreply", "Feedback-ID")
        diag_lower = {n.lower() for n in diag}
        print("automated-signal headers:")
        for h in payload.get("headers", []):
            if h.get("name", "").lower() in diag_lower:
                print(f"  {h.get('name')}: {h.get('value')}")
        print("payload structure:")
        skeleton(payload)
        parsed = _parse_message(msg)
        print("\n_parse_message extracted (body shown as length only):")
        print({**{k: parsed[k] for k in ('from','to','subject','date','message_id_header')},
               "body_chars": len(parsed.get("body") or "")})
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
