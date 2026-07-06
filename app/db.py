"""Persistence + audit trail for the FDE Email Agent (Milestone 3).

Three tables, mapping to the processing loop in CLAUDE.md:
- emails:    the raw ingested thread (what we received).
- drafts:    one row per generated reply, carrying its classification, the
             draft text, lifecycle status, any human edit, and the final
             sent text. This is the per-thread "state".
- audit_log: an append-only record of every event (email ingested,
             classified, draft generated, human decision, sent). Nothing is
             ever updated or deleted here — it is the source of truth for
             "what happened and when".

Backend is Postgres (see docker-compose.yml); connection comes from
DATABASE_URL. We use SQLAlchemy's portable JSON column type so the same models
also run on SQLite for quick local smoke tests.
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import JSON, DateTime
from pgvector.sqlalchemy import Vector

# Embedding dimension for the docs-RAG vector store. Must match the model used
# in app/llm.py (text-embedding-3-small -> 1536). Change both together.
EMBED_DIM = 1536

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://fde:fde@localhost:5432/fde_email_agent",
)

# Draft lifecycle states. Phase 1 only auto-reaches "drafted"; the rest are
# set by the human-approval step. There is deliberately NO "auto_send" state.
DRAFT_DRAFTED = "drafted"
DRAFT_APPROVED = "approved"
DRAFT_EDITED = "edited"
DRAFT_REJECTED = "rejected"
DRAFT_SENT = "sent"

# Terminal states: a draft here is finished and must not be re-decided. This is
# the structural backstop behind the "approval is structural" rule — a stale
# Slack card with live buttons cannot revive a rejected draft or re-send a sent
# one (which would push an unwanted reply to a customer).
_TERMINAL_STATUSES = frozenset({DRAFT_REJECTED, DRAFT_SENT})


class DecisionConflict(Exception):
    """Raised when a human decision targets an already-finalized draft."""

    def __init__(self, draft_id: int, current_status: str, attempted: str):
        self.draft_id = draft_id
        self.current_status = current_status
        self.attempted = attempted
        super().__init__(
            f"draft {draft_id} is already '{current_status}'; "
            f"'{attempted}' not applied"
        )


class Base(DeclarativeBase):
    pass


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="cli")  # cli | gmail
    # The full thread exactly as ingested (untrusted customer input).
    raw_thread: Mapped[dict] = mapped_column(JSON)
    # Everything needed to send a threaded reply later (Gmail thread id, the
    # message-id to reply to, References chain, recipient, subject). Populated
    # by gmail ingest; null for cli-sourced rows. The recipient here is set by
    # OUR code from the verified inbound message, never chosen by the model.
    reply_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    drafts: Mapped[list["Draft"]] = relationship(back_populates="email")


class Draft(Base):
    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), index=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classification: Mapped[dict] = mapped_column(JSON)
    draft_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default=DRAFT_DRAFTED)
    # Set later by the human-approval step (Milestone 4+).
    edited_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Slack approval message coordinates, captured when the card is posted, so a
    # later action (bulk reject, status sync) can update the exact message
    # without needing the channels:history scope to find it again.
    slack_channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    slack_ts: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Idempotency anchor: stable key for the delivery unit that produced this
    # draft — (thread_id, triggering message id), content-hash fallback. UNIQUE,
    # so a REPLAY (a redelivered notification reprocessing the SAME message)
    # collapses to one draft instead of duplicating. A genuinely NEW inbound
    # message has a different key -> a new draft (multi-turn preserved). NULLs
    # don't conflict, so pre-existing rows are unaffected.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)
    # Atomic posting claim (a lease): before posting the approval card, a poster
    # (normal path OR the orphaned-draft sweeper) must CAS-win this claim, so
    # exactly one posts and the sweeper can never race an in-flight post. Held as
    # a timestamp so a claim can be reclaimed after a lease if the claimer died
    # before recording slack_ts (otherwise a crash mid-post would orphan it).
    post_claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Present only on a "booking draft" (a meeting confirmation): the calendar
    # action this draft will perform on approval. Shape:
    #   {start, end, duration_min, attendee_email, title, timezone}
    # The attendee_email is injected by our code from the verified sender, never
    # chosen by the model. None for ordinary (non-booking) drafts.
    booking: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    email: Mapped["Email"] = relationship(back_populates="drafts")


class WebPricing(Base):
    """Per-country pricing parsed from plivo.com SSG pages (voice/whatsapp/
    numbers). We store the RENDERED figures verbatim (the ₹/$ the site shows),
    with the currency read from the page — never converted. `tables` is the
    parsed price table(s) as JSON (route rows -> rendered rate strings). One row
    per (channel, iso); atomic per-channel reload; imported_at drives the
    'current pricing as of <date>' stamp and the weekly change-diff."""

    __tablename__ = "web_pricing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)   # voice|whatsapp|numbers
    iso: Mapped[str] = mapped_column(String(8), index=True)
    country_name: Mapped[str] = mapped_column(Text)
    currency: Mapped[str] = mapped_column(String(8))               # INR | USD (from page)
    tables: Mapped[dict] = mapped_column(JSON)                     # parsed price tables
    source_url: Mapped[str] = mapped_column(Text)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SmsPricing(Base):
    """SMS outbound pricing, one row per sheet entry (country-level blended, plus
    the US route rows and the India IN-ILDO row). Reference data, exact lookup
    only — no vectors. `rate_usd` is stored as the EXACT string from the sheet so
    it is quoted verbatim (we never compute with it). Reloaded atomically on
    import. The India row loads as data but is suppressed by a code override."""

    __tablename__ = "sms_pricing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    iso: Mapped[str] = mapped_column(String(16), index=True)  # AE, US-10DLC, IN-ILDO, ...
    country_name: Mapped[str] = mapped_column(Text)           # sheet's Country name
    rate_usd: Mapped[str] = mapped_column(String(32))         # exact blended rate string
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class KVState(Base):
    """Tiny durable key/value store for worker state (e.g. Gmail watch
    expiration and the last processed historyId)."""

    __tablename__ = "kv_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SchedulingState(Base):
    """Per-thread meeting-scheduling negotiation state, keyed by Gmail thread id.

    Lets the agent tell a continuing negotiation ("yeah 4pm works") from a fresh
    request, and remember what's been proposed/agreed so it books the right slot
    once aligned. One row per thread; `state` is a JSON blob owned by
    app/scheduling.py (stage, proposed_slots, candidate_slot, duration, tz, ...).
    """

    __tablename__ = "scheduling_state"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    state: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    email_id: Mapped[int | None] = mapped_column(ForeignKey("emails.id"), nullable=True)
    draft_id: Mapped[int | None] = mapped_column(ForeignKey("drafts.id"), nullable=True)
    # e.g. "email_ingested", "classified", "draft_generated",
    #      "human_approved", "human_edited", "human_rejected", "email_sent".
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")  # system | human
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# --- Docs-RAG vector store (Phase 2 case 2) ---------------------------------
#
# In a SEPARATE declarative base so the SQLite-backed unit tests (which call
# Base.metadata.create_all) never try to build a `vector` column. These tables
# exist only on Postgres (pgvector); init_db creates them there.

class VectorBase(DeclarativeBase):
    pass


class DocChunk(VectorBase):
    """One retrievable chunk of Plivo documentation/support/GitHub content, with
    its embedding. Platform answers are drawn ONLY from rows in this table."""

    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(16), index=True)  # docs|support|github|fde_ratified
    url: Mapped[str] = mapped_column(Text, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo: Mapped[str | None] = mapped_column(String(128), nullable=True)  # github only
    content: Mapped[str] = mapped_column(Text)
    # Hash of the chunk content; refresh re-embeds only when this changes.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))
    # Learning-loop provenance + non-destructive revocation for ratified FDE facts.
    # active=false drops a chunk out of retrieval without deleting it (auditable);
    # origin_draft_id links a fde_ratified fact back to the edit it was captured from.
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    origin_draft_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class StyleRule(Base):
    """A drafting style/type rule LEARNED from Nihal's edits (the edit-learning
    loop, style side). Lifecycle: candidate -> ratified -> revoked. ONLY
    status='ratified' rows are injected into drafting prompts, so a candidate is
    inert until Nihal ratifies it and a bad rule is revoked (status='revoked')
    without a redeploy. `scope` is the NARROWEST the evidence supports — an intent
    name, or 'global' only with cross-intent evidence. `supersedes_id` chains a
    refinement to the version it replaces. `evidence_draft_ids` are the edits the
    rule was distilled from (shown on the ratify card)."""

    __tablename__ = "style_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(64), index=True)   # 'global' | intent name
    rule_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_draft_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ratified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --- Engine / session plumbing ----------------------------------------------

_engine = None
_SessionFactory: sessionmaker | None = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), future=True)
    return _SessionFactory()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call repeatedly.

    Also performs the one additive column migration introduced in Milestone 5
    (emails.reply_context) so an existing dev DB doesn't need to be dropped.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    # Lightweight additive migrations for already-created tables.
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE emails ADD COLUMN IF NOT EXISTS reply_context JSONB")
            )
            conn.execute(
                text("ALTER TABLE drafts ADD COLUMN IF NOT EXISTS slack_channel VARCHAR(32)")
            )
            conn.execute(
                text("ALTER TABLE drafts ADD COLUMN IF NOT EXISTS slack_ts VARCHAR(32)")
            )
            conn.execute(
                text("ALTER TABLE drafts ADD COLUMN IF NOT EXISTS booking JSONB")
            )
            # Edit-learning loop: fact provenance + non-destructive revocation.
            conn.execute(text(
                "ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE"))
            conn.execute(text(
                "ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS origin_draft_id INTEGER"))
        # Docs-RAG vector store: enable pgvector, create doc_chunks + its HNSW
        # cosine index. Extension must exist before the vector column is created.
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        VectorBase.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_hnsw "
                "ON doc_chunks USING hnsw (embedding vector_cosine_ops)"
            ))


# --- High-level helpers used by the CLI / worker -----------------------------

def record_audit(
    session: Session,
    event_type: str,
    *,
    thread_id: str | None = None,
    email_id: int | None = None,
    draft_id: int | None = None,
    actor: str = "system",
    payload: dict | None = None,
) -> AuditLog:
    """Append one immutable event to the audit log."""
    entry = AuditLog(
        event_type=event_type,
        thread_id=thread_id,
        email_id=email_id,
        draft_id=draft_id,
        actor=actor,
        payload=payload,
    )
    session.add(entry)
    return entry


def persist_processing(
    thread: dict,
    classification: dict,
    draft_text: str,
    *,
    source: str = "cli",
    reply_context: dict | None = None,
    booking: dict | None = None,
    artifact_key: str | None = None,
) -> dict:
    """Persist a single ingest→classify→draft pass with its audit trail.

    Writes the email, the draft, and three audit events in one transaction.
    `booking` (optional) attaches the calendar action a meeting-confirmation
    draft will perform on approval. Returns the new row ids.

    `artifact_key` (optional) OVERRIDES the trigger-derived idempotency anchor.
    The default anchor encodes only the triggering message, which is correct for
    the ONE primary reply per inbound mail. But a single mail can legitimately
    produce SEVERAL artifact kinds (a debugging answer, a meeting follow-up) that
    must NOT collapse into — or be collapsed by — that primary hold. Such callers
    pass a key that encodes ARTIFACT identity (see build_artifact_key): distinct
    from the hold (so it lands as its own card) yet stable per artifact (so its
    own replays still dedup).
    """
    from sqlalchemy.exc import IntegrityError

    thread_id = thread.get("thread_id") or thread.get("subject") or "unknown"
    intent = classification.get("intent")
    key = artifact_key or _idempotency_key(thread)

    with get_session() as session:
        # IDEMPOTENCY ANCHOR: a replay of the same delivery unit (same triggering
        # message) returns the existing draft instead of creating a duplicate.
        if key:
            existing = session.query(Draft).filter(Draft.idempotency_key == key).first()
            if existing:
                return {"email_id": existing.email_id, "draft_id": existing.id,
                        "thread_id": thread_id, "deduped": True}
        email = Email(
            thread_id=thread_id,
            subject=thread.get("subject"),
            source=source,
            raw_thread=thread,
            reply_context=reply_context,
        )
        session.add(email)
        session.flush()  # assigns email.id
        record_audit(
            session, "email_ingested", thread_id=thread_id, email_id=email.id,
            payload={"source": source, "message_count": len(thread.get("messages", []))},
        )
        record_audit(
            session, "classified", thread_id=thread_id, email_id=email.id,
            payload=classification,
        )

        draft = Draft(
            email_id=email.id,
            intent=intent,
            classification=classification,
            draft_text=draft_text,
            status=DRAFT_DRAFTED,
            booking=booking,
            idempotency_key=key,
        )
        session.add(draft)
        session.flush()  # assigns draft.id
        record_audit(
            session, "draft_generated", thread_id=thread_id,
            email_id=email.id, draft_id=draft.id,
            payload={"intent": intent, "draft_chars": len(draft_text),
                     "booking": bool(booking)},
        )

        try:
            session.commit()
        except IntegrityError:
            # A concurrent replay won the unique key first — return that draft.
            session.rollback()
            existing = session.query(Draft).filter(Draft.idempotency_key == key).first()
            if existing:
                return {"email_id": existing.email_id, "draft_id": existing.id,
                        "thread_id": thread_id, "deduped": True}
            raise
        return {"email_id": email.id, "draft_id": draft.id, "thread_id": thread_id}


def build_artifact_key(thread_id: str, kind: str, discriminator) -> str:
    """Dedup anchor for a NON-PRIMARY artifact: encodes artifact identity, not just
    trigger identity. `kind` names the artifact type (debug / debug-sms / meeting);
    `discriminator` makes it stable-per-artifact (e.g. the the reviewer thread ts, the
    account_id, the message UUID, the notes mail id). Distinct from the primary
    hold's trigger-derived key, so both coexist; identical across the SAME
    artifact's replays, so those still collapse."""
    return f"{thread_id}::{kind}::{discriminator}"


def _idempotency_key(thread: dict) -> str | None:
    """Stable key for the delivery unit = (thread_id, triggering message id).

    Falls back to a content hash when the latest message has no id (e.g. CLI /
    synthetic threads). A redelivered notification reprocessing the SAME latest
    message yields the SAME key -> dedup. A genuinely new inbound message yields
    a different key -> a new draft (multi-turn is preserved, NOT collapsed).
    """
    tid = thread.get("thread_id") or thread.get("subject") or "unknown"
    msgs = thread.get("messages") or []
    if not msgs:
        return None
    mid = msgs[-1].get("id")
    if mid:
        return f"{tid}::{mid}"
    import hashlib
    body = msgs[-1].get("body") or ""
    return f"{tid}::h{hashlib.sha256((str(tid) + '|' + body).encode()).hexdigest()[:16]}"


def claim_draft_for_posting(draft_id: int, lease_seconds: int = 120) -> bool:
    """Atomically claim a draft for posting its approval card. Returns True iff
    THIS caller won the claim (and must therefore be the one to post).

    Race-safe compare-and-set in a single conditional UPDATE: the winner is the
    only transaction whose WHERE still matches when it acquires the row lock. A
    draft is claimable iff it is not yet posted (slack_ts IS NULL) and is either
    unclaimed or its prior claim's lease has expired (so a claimer that died
    before recording slack_ts is recovered, never orphaned). Any poster — the
    normal path or the sweeper — calls this first; exactly one wins.
    """
    from sqlalchemy import text
    sql = text(
        "UPDATE drafts SET post_claimed_at = now() "
        "WHERE id = :id AND slack_ts IS NULL "
        "AND (post_claimed_at IS NULL OR post_claimed_at < now() - make_interval(secs => :lease)) "
        "RETURNING id"
    )
    with get_engine().begin() as conn:
        return conn.execute(sql, {"id": draft_id, "lease": lease_seconds}).first() is not None


def release_post_claim(draft_id: int) -> None:
    """Release a posting claim (only if the card still isn't posted) so the next
    attempt can re-claim IMMEDIATELY instead of waiting out the lease. Called when
    a claimed post fails — a failed post should retry promptly, not burn a lease."""
    from sqlalchemy import text
    with get_engine().begin() as conn:
        conn.execute(text("UPDATE drafts SET post_claimed_at = NULL "
                          "WHERE id = :id AND slack_ts IS NULL"), {"id": draft_id})


def find_unposted_drafts(grace_seconds: int = 120, limit: int = 50) -> list[dict]:
    """Drafts whose approval card never landed: slack_ts IS NULL, still live
    (status = drafted, not a terminal decision), and older than the grace period.

    The grace period means the normal post path has already had its chance, so the
    sweeper rarely races the common case — and the claim makes it safe when it
    does. Returns {draft_id, draft_text, raw_thread} for reposting.
    """
    from sqlalchemy import text
    with get_engine().begin() as conn:
        rows = conn.execute(text(
            "SELECT d.id AS draft_id, d.draft_text, e.raw_thread "
            "FROM drafts d JOIN emails e ON e.id = d.email_id "
            "WHERE d.slack_ts IS NULL AND d.status = :st "
            "AND d.created_at < now() - make_interval(secs => :g) "
            "ORDER BY d.created_at LIMIT :lim"),
            {"st": DRAFT_DRAFTED, "g": grace_seconds, "lim": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def claim_pending_case(key: str) -> dict | None:
    """Atomically claim (read-AND-clear) a kv_state entry: DELETE the row and
    return its prior value. Exactly one concurrent caller wins the row; every
    other caller gets None. This makes 'read pending + clear' one atomic step, so
    a reconnect catch-up scan and a live event can NEVER both resume the same
    case — whoever's DELETE hits the row first wins, the rest no-op.
    """
    from sqlalchemy import text
    with get_engine().begin() as conn:
        row = conn.execute(text("DELETE FROM kv_state WHERE key = :k RETURNING value"),
                           {"k": key}).first()
    return row[0] if row else None


def list_pending_case_keys(prefix: str, max_age_seconds: int, max_cases: int) -> list[str]:
    """Keys of recent, still-pending cases for a reconnect catch-up scan —
    BOUNDED so a reconnect storm can't become an API-rate storm: only cases
    touched within max_age_seconds, newest first, capped at max_cases. Skips
    already-cleared ({}) sentinels."""
    from sqlalchemy import text
    with get_engine().begin() as conn:
        rows = conn.execute(text(
            "SELECT key FROM kv_state WHERE key LIKE :p "
            "AND updated_at > now() - make_interval(secs => :age) "
            "AND value::text <> '{}' "
            "ORDER BY updated_at DESC LIMIT :lim"),
            {"p": prefix + "%", "age": max_age_seconds, "lim": max_cases},
        ).all()
    return [r[0] for r in rows]


def get_draft_view(draft_id: int) -> dict | None:
    """Read-only snapshot of a draft + its email, safe to use after the session
    closes (returns plain data, not ORM objects bound to a session)."""
    with get_session() as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            return None
        email = session.get(Email, draft.email_id)
        return {
            "draft_id": draft.id,
            "email_id": draft.email_id,
            "thread_id": email.thread_id if email else None,
            "subject": email.subject if email else None,
            "intent": draft.intent,
            "status": draft.status,
            "draft_text": draft.draft_text,
            "edited_text": draft.edited_text,
            "final_text": draft.final_text,
            "effective_text": draft.edited_text or draft.draft_text,
        }


def set_slack_message(draft_id: int, channel: str | None, ts: str | None) -> None:
    """Record where a draft's approval card was posted (channel + message ts),
    so it can be updated later without re-discovering it via channel history."""
    if not (channel and ts):
        return
    with get_session() as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            return
        draft.slack_channel = channel
        draft.slack_ts = ts
        session.commit()


def record_human_decision(
    draft_id: int,
    decision: str,
    *,
    actor: str = "human",
    edited_text: str | None = None,
) -> dict:
    """Apply a human approval decision to a draft and audit it, atomically.

    decision -> (status, audit event):
      "approve" -> approved   / human_approved
      "edit"    -> edited     / human_edited   (requires edited_text)
      "reject"  -> rejected   / human_rejected

    This is the ONLY path that moves a draft toward being sendable, and it is
    only ever called from a human-triggered handler — never by the model.
    Note: approval does NOT send. Actual sending (and final_text) lands in
    Milestone 5.
    """
    mapping = {
        "approve": (DRAFT_APPROVED, "human_approved"),
        "edit": (DRAFT_EDITED, "human_edited"),
        "reject": (DRAFT_REJECTED, "human_rejected"),
    }
    if decision not in mapping:
        raise ValueError(f"unknown decision: {decision!r}")
    new_status, event_type = mapping[decision]

    with get_session() as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            raise ValueError(f"no draft with id {draft_id}")
        # Structural guard: never let a finalized draft be re-decided. Clicking a
        # stale Approve button on a rejected draft must NOT send it.
        if draft.status in _TERMINAL_STATUSES:
            raise DecisionConflict(draft_id, draft.status, decision)
        email = session.get(Email, draft.email_id)
        thread_id = email.thread_id if email else None

        draft.status = new_status
        if decision == "edit":
            if not edited_text:
                raise ValueError("edit decision requires edited_text")
            draft.edited_text = edited_text

        record_audit(
            session, event_type,
            thread_id=thread_id, email_id=draft.email_id, draft_id=draft.id,
            actor=actor,
            payload={"decision": decision, "edited": bool(edited_text)},
        )
        session.commit()
        return {
            "draft_id": draft.id,
            "status": draft.status,
            "effective_text": draft.edited_text or draft.draft_text,
            "thread_id": thread_id,
        }


# --- KV state (worker bookkeeping) ------------------------------------------

def kv_get(key: str, default=None):
    with get_session() as session:
        row = session.get(KVState, key)
        return row.value if row else default


def kv_set(key: str, value: dict) -> None:
    with get_session() as session:
        row = session.get(KVState, key)
        if row is None:
            session.add(KVState(key=key, value=value))
        else:
            row.value = value
        session.commit()


# --- Scheduling state (per Gmail thread) ------------------------------------

def get_scheduling_state(thread_id: str) -> dict | None:
    """Return the meeting-negotiation state for a thread, or None if none yet."""
    with get_session() as session:
        row = session.get(SchedulingState, thread_id)
        return dict(row.state) if row else None


def set_scheduling_state(thread_id: str, state: dict) -> None:
    """Upsert the meeting-negotiation state for a thread."""
    with get_session() as session:
        row = session.get(SchedulingState, thread_id)
        if row is None:
            session.add(SchedulingState(thread_id=thread_id, state=state))
        else:
            row.state = state
        session.commit()


# --- SMS pricing (exact lookup; atomic reload) ------------------------------

def replace_sms_pricing(rows: list[dict]) -> int:
    """Atomically replace the entire sms_pricing table (truncate + reload in one
    transaction) so a bad/partial import never half-replaces good rate data.
    Each row: {iso, country_name, rate_usd}."""
    with get_session() as session:
        session.query(SmsPricing).delete()
        session.add_all([SmsPricing(iso=r["iso"], country_name=r["country_name"],
                                    rate_usd=r["rate_usd"]) for r in rows])
        session.commit()
    return len(rows)


def replace_web_pricing(channel: str, rows: list[dict]) -> int:
    """Atomically replace all web_pricing rows for one channel. Each row:
    {iso, country_name, currency, tables, source_url}."""
    with get_session() as session:
        session.query(WebPricing).filter(WebPricing.channel == channel).delete()
        session.add_all([WebPricing(channel=channel, iso=r["iso"], country_name=r["country_name"],
                                    currency=r["currency"], tables=r["tables"],
                                    source_url=r["source_url"]) for r in rows])
        session.commit()
    return len(rows)


def get_web_pricing(channel: str, iso: str) -> dict | None:
    with get_session() as session:
        r = session.query(WebPricing).filter(WebPricing.channel == channel,
                                             WebPricing.iso == iso).first()
        return {"channel": r.channel, "iso": r.iso, "country_name": r.country_name,
                "currency": r.currency, "tables": r.tables, "source_url": r.source_url,
                "imported_at": r.imported_at.isoformat() if r.imported_at else None} if r else None


def all_web_pricing(channel: str) -> list[dict]:
    with get_session() as session:
        rows = session.query(WebPricing).filter(WebPricing.channel == channel).all()
        return [{"iso": r.iso, "country_name": r.country_name, "currency": r.currency,
                 "tables": r.tables} for r in rows]


def get_sms_pricing_by_iso(iso: str) -> dict | None:
    with get_session() as session:
        row = session.query(SmsPricing).filter(SmsPricing.iso == iso).first()
        return {"iso": row.iso, "country_name": row.country_name, "rate_usd": row.rate_usd} if row else None


def all_sms_pricing() -> list[dict]:
    with get_session() as session:
        rows = session.query(SmsPricing).order_by(SmsPricing.iso).all()
        return [{"iso": r.iso, "country_name": r.country_name, "rate_usd": r.rate_usd} for r in rows]


# --- Docs-RAG store helpers -------------------------------------------------

def add_chunk(*, source_type: str, url: str, content: str, content_hash: str,
              embedding: list[float], title: str | None = None,
              heading: str | None = None, repo: str | None = None) -> int:
    """Insert one embedded doc chunk; returns its id. (Bulk ingest + refresh
    upsert-by-hash arrive in the crawler step.)"""
    with get_session() as session:
        c = DocChunk(source_type=source_type, url=url, title=title, heading=heading,
                     repo=repo, content=content, content_hash=content_hash, embedding=embedding)
        session.add(c)
        session.flush()
        cid = c.id
        session.commit()
        return cid


def count_chunks(source_type: str | None = None) -> int:
    with get_session() as session:
        q = session.query(func.count(DocChunk.id))
        if source_type:
            q = q.filter(DocChunk.source_type == source_type)
        return q.scalar()


def existing_chunk_hashes(url: str) -> dict[str, int]:
    """{content_hash: id} for chunks currently stored for a URL (refresh diff)."""
    with get_session() as session:
        rows = session.query(DocChunk.content_hash, DocChunk.id).filter(DocChunk.url == url).all()
        return {h: i for h, i in rows}


def add_chunks(chunks: list[dict]) -> int:
    """Bulk-insert embedded chunks (each dict carries an `embedding`)."""
    if not chunks:
        return 0
    cols = ("source_type", "url", "title", "heading", "repo", "content", "content_hash", "embedding")
    with get_session() as session:
        session.add_all([DocChunk(**{k: c.get(k) for k in cols}) for c in chunks])
        session.commit()
    return len(chunks)


def delete_chunks(ids: list[int]) -> int:
    if not ids:
        return 0
    with get_session() as session:
        n = session.query(DocChunk).filter(DocChunk.id.in_(ids)).delete(synchronize_session=False)
        session.commit()
        return n


def search_chunks(embedding: list[float], k: int = 8) -> list[dict]:
    """Raw cosine KNN over doc_chunks. Returns chunks with a `similarity` score
    (1 - cosine distance), nearest first. Confidence routing on these scores is
    added in the retrieval step; this is the low-level query only."""
    with get_session() as session:
        distance = DocChunk.embedding.cosine_distance(embedding).label("distance")
        rows = (session.query(DocChunk, distance)
                .filter(DocChunk.active.is_(True))   # revoked fde_ratified facts drop out
                .order_by(distance).limit(k).all())
        return [{
            "id": c.id, "source_type": c.source_type, "url": c.url, "title": c.title,
            "heading": c.heading, "repo": c.repo, "content": c.content,
            "similarity": 1.0 - float(d),
        } for c, d in rows]


# --- Edit-learning loop: fact capture (-> RAG corpus) + style rules ----------
FDE_FACT_SOURCE = "fde_ratified"


def add_fact_chunk(*, content: str, embedding: list[float], origin_draft_id: int | None,
                   title: str | None = None, url: str = "fde://ratified") -> int:
    """Insert a RATIFIED FDE fact as a retrievable, citable doc chunk. It inherits
    the full grounding/citation discipline (it's just another chunk); `source_type`
    marks provenance and `active` allows non-destructive revocation."""
    import hashlib
    ch = hashlib.sha256(content.encode()).hexdigest()
    with get_session() as session:
        c = DocChunk(source_type=FDE_FACT_SOURCE, url=url, title=title, heading=None,
                     repo=None, content=content, content_hash=ch, embedding=embedding,
                     active=True, origin_draft_id=origin_draft_id)
        session.add(c)
        session.flush()
        cid = c.id
        session.commit()
        return cid


def deactivate_fact_chunk(chunk_id: int) -> bool:
    """Revoke a ratified fact non-destructively (drops out of retrieval, kept for
    audit). Returns True if a row was flipped."""
    with get_session() as session:
        c = session.get(DocChunk, chunk_id)
        if not c:
            return False
        c.active = False
        session.commit()
        return True


def add_style_candidate(scope: str, rule_text: str, evidence_draft_ids: list) -> int:
    with get_session() as session:
        r = StyleRule(scope=scope, rule_text=rule_text, status="candidate",
                      evidence_draft_ids=list(evidence_draft_ids or []))
        session.add(r)
        session.flush()
        rid = r.id
        session.commit()
        return rid


def set_style_rule_status(rule_id: int, status: str) -> bool:
    """ratify | revoke | reject a rule. Only 'ratified' rules reach the prompts."""
    with get_session() as session:
        r = session.get(StyleRule, rule_id)
        if not r:
            return False
        r.status = status
        if status == "ratified":
            r.ratified_at = func.now()
        session.commit()
        return True


def get_active_style_rules(intent: str | None = None) -> list[dict]:
    """Ratified rules in scope: always 'global', plus the current intent. This is
    the ONLY reader the drafting prompts use — candidates/revoked never appear."""
    scopes = ["global"] + ([intent] if intent else [])
    with get_session() as session:
        rows = (session.query(StyleRule)
                .filter(StyleRule.status == "ratified", StyleRule.scope.in_(scopes))
                .order_by(StyleRule.id).all())
        return [{"id": r.id, "scope": r.scope, "rule_text": r.rule_text} for r in rows]


# --- Send-side (Milestone 5) ------------------------------------------------

def get_send_context(draft_id: int) -> dict | None:
    """Everything the approve handler needs to actually send a reply.

    Returns the effective text plus the email's reply_context (Gmail thread id,
    message-id to reply to, recipient, subject). recipient comes from our
    stored ingest data, never from the model.
    """
    with get_session() as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            return None
        email = session.get(Email, draft.email_id)
        return {
            "draft_id": draft.id,
            "email_id": draft.email_id,
            "status": draft.status,
            "effective_text": draft.edited_text or draft.draft_text,
            "reply_context": (email.reply_context if email else None),
            "booking": draft.booking,
        }


def mark_draft_sent(draft_id: int, final_text: str, provider_message_id: str | None) -> dict:
    """Record that an approved draft was actually sent. Sets final_text and
    status=sent, and writes the terminal `email_sent` audit event."""
    with get_session() as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            raise ValueError(f"no draft with id {draft_id}")
        email = session.get(Email, draft.email_id)
        thread_id = email.thread_id if email else None

        draft.status = DRAFT_SENT
        draft.final_text = final_text
        record_audit(
            session, "email_sent",
            thread_id=thread_id, email_id=draft.email_id, draft_id=draft.id,
            actor="system",
            payload={"provider_message_id": provider_message_id, "chars": len(final_text)},
        )
        session.commit()
        return {"draft_id": draft.id, "status": draft.status, "thread_id": thread_id}


def record_booking(draft_id: int, event: dict) -> None:
    """Audit a calendar event created on approval (the booking action)."""
    with get_session() as session:
        draft = session.get(Draft, draft_id)
        email = session.get(Email, draft.email_id) if draft else None
        thread_id = email.thread_id if email else None
        record_audit(
            session, "calendar_event_created",
            thread_id=thread_id,
            email_id=(draft.email_id if draft else None),
            draft_id=draft_id,
            actor="system",
            payload=event,
        )
        session.commit()
