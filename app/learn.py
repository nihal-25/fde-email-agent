"""Edit-learning loop — proposes lessons from Nihal's edits; NOTHING applies
without his ratification (same gate architecture as drafts).

TWO separate mechanisms, different consumers/lifecycles:
  (a) STYLE distillation  -> style_rules table -> injected into drafting prompts
      (versioned, revocable). Proposed at the NARROWEST scope the evidence supports.
  (b) FACT capture        -> doc_chunks(source_type='fde_ratified') -> RAG corpus
      (inherits grounding/citation discipline; non-destructive revoke via active=false).
      Every hold edited into an answer IS the missing corpus entry — closes over-holds.

Ratification is push (Slack #learning Ratify/Reject cards). Every card shows its
EVIDENCE inline (real before/after diff from the cited drafts); fact cards show the
VERBATIM text that will be embedded. Candidates are inert until ratified.
"""
from __future__ import annotations

from app import db, llm
from app.db import Draft, Email
from sqlalchemy import select


def _effective(d: Draft) -> str | None:
    return d.edited_text or d.final_text


def get_edit_deltas(limit: int = 50) -> list[dict]:
    """Recent (original, edited) pairs where the human CHANGED the draft. This is
    the only signal source; it accrues as edits are made via the Slack Edit modal."""
    out = []
    s = db.get_session()
    try:
        rows = s.execute(select(Draft).where(
            (Draft.edited_text.isnot(None)) | (Draft.final_text.isnot(None))
        ).order_by(Draft.id.desc()).limit(limit)).scalars().all()
        for d in rows:
            eff = _effective(d)
            if eff and eff.strip() and eff.strip() != (d.draft_text or "").strip():
                out.append({"draft_id": d.id, "intent": d.intent,
                            "original": d.draft_text, "edited": eff})
    finally:
        s.close()
    return out


_STYLE_MIN_EVIDENCE = 3   # a STYLE (pattern) claim needs >=3 in-scope edits — n=1/2
                          # over-generalization is structurally impossible, not just
                          # prompt-discouraged. Facts have no floor (one edit can carry one fact).


def _in_scope_evidence(scope: str, draft_ids: list[int]) -> list[int]:
    """The subset of cited edits that actually fall in the rule's scope: for an
    intent-scoped rule, edits of THAT intent; for 'global', any. Filtering here
    (not trusting the model's count) is what makes the floor structural."""
    s = db.get_session()
    try:
        keep = []
        for did in dict.fromkeys(draft_ids or []):        # de-dup, preserve order
            d = s.get(Draft, did)
            if d and (scope == "global" or d.intent == scope):
                keep.append(did)
        return keep
    finally:
        s.close()


def evidence_snippets(draft_ids: list[int]) -> list[dict]:
    """Before/after text for the cited drafts — shown inline on the ratify card so
    Nihal ratifies an inference against its source, not a bare distilled rule."""
    out = []
    s = db.get_session()
    try:
        for did in draft_ids or []:
            d = s.get(Draft, did)
            if d:
                out.append({"draft_id": did, "original": d.draft_text,
                            "edited": _effective(d) or ""})
    finally:
        s.close()
    return out


def propose_lessons(limit: int = 50, *, persist_style: bool = True) -> dict:
    """Distil STYLE candidates (persisted as StyleRule candidates so a card can
    ratify by id) and extract FACT candidates (carried verbatim on the card, only
    embedded into the corpus on ratify). Returns both for carding."""
    deltas = get_edit_deltas(limit)
    result = {"n_deltas": len(deltas), "style_candidates": [], "fact_candidates": []}
    if not deltas:
        return result

    # (a) style — one distillation over all deltas so cross-intent patterns are visible
    result["dropped_under_evidenced"] = []
    for r in (llm.distill_style_rules(deltas).get("rules") or []):
        scope = (r.get("scope") or "global").strip()
        rule_text = (r.get("rule_text") or "").strip()
        ev = [int(x) for x in (r.get("evidence_draft_ids") or []) if str(x).isdigit()]
        if not rule_text:
            continue
        # STRUCTURAL FLOOR: >=3 in-scope evidence edits or the candidate is never
        # proposed. A one-off edit cannot become a style rule.
        ev = _in_scope_evidence(scope, ev)
        if len(ev) < _STYLE_MIN_EVIDENCE:
            result["dropped_under_evidenced"].append(
                {"scope": scope, "rule_text": rule_text, "n_in_scope": len(ev)})
            continue
        cand = {"scope": scope, "rule_text": rule_text, "evidence_draft_ids": ev,
                "evidence": evidence_snippets(ev)}
        if persist_style:
            cand["rule_id"] = db.add_style_candidate(scope, rule_text, ev)
        result["style_candidates"].append(cand)

    # (b) facts — per delta; the edit's own thread is the context for general-vs-specific
    for d in deltas:
        for fact in (llm.extract_added_facts(d["original"], d["edited"]).get("facts") or []):
            fact = (fact or "").strip()
            if fact:
                result["fact_candidates"].append(
                    {"fact_text": fact, "origin_draft_id": d["draft_id"],
                     "evidence": evidence_snippets([d["draft_id"]])})
    return result


# --- ratify / reject application (called by the Slack handlers) --------------
def ratify_style(rule_id: int) -> bool:
    """Ratify a style candidate — now (and only now) it enters drafting prompts."""
    return db.set_style_rule_status(rule_id, "ratified")


def reject_style(rule_id: int) -> bool:
    return db.set_style_rule_status(rule_id, "revoked")


def revoke_style(rule_id: int) -> bool:
    """Remove a previously-ratified rule from the prompts (one status flip)."""
    return db.set_style_rule_status(rule_id, "revoked")


def ratify_fact(fact_text: str, origin_draft_id: int | None, title: str | None = None) -> int:
    """Embed the VERBATIM ratified fact and add it to the RAG corpus as a citable,
    revocable fde_ratified chunk. Returns the chunk id."""
    embedding = llm.embed([fact_text])[0]
    return db.add_fact_chunk(content=fact_text, embedding=embedding,
                             origin_draft_id=origin_draft_id, title=title)


def revoke_fact(chunk_id: int) -> bool:
    """Non-destructive: the fact stops being retrieved/cited but is kept for audit."""
    return db.deactivate_fact_chunk(chunk_id)
