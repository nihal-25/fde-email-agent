"""Edit-learning loop — the ratification gate + prompt injection + fact lifecycle,
by injection. Hard rule under test: nothing learned takes effect unratified.
DB-backed (the gate lives there); synthetic embeddings so it's deterministic.
"""
from unittest import mock

from app import db, llm, learn
from app.db import StyleRule, DocChunk, EMBED_DIM
from sqlalchemy import select


def _cleanup_rules(ids):
    s = db.get_session()
    try:
        for i in ids:
            r = s.get(StyleRule, i)
            if r:
                s.delete(r)
        s.commit()
    finally:
        s.close()


def _cleanup_chunk(cid):
    s = db.get_session()
    try:
        c = s.get(DocChunk, cid)
        if c:
            s.delete(c)
        s.commit()
    finally:
        s.close()


# ---------- (1) candidate is INERT; (2) ratified enters; (3) revoked leaves --
def test_style_rule_gate_candidate_ratified_revoked():
    rid = db.add_style_candidate("pricing_question", "Drop the opening pleasantry.", [1, 2])
    try:
        # candidate -> NOT active, NOT in prompt directives
        assert not db.get_active_style_rules("pricing_question")
        assert "pleasantry" not in llm.style_directives("pricing_question")

        # ratified -> active + injected
        assert learn.ratify_style(rid)
        active = db.get_active_style_rules("pricing_question")
        assert any(r["id"] == rid for r in active)
        assert "pleasantry" in llm.style_directives("pricing_question")

        # revoked -> leaves the prompt
        assert learn.revoke_style(rid)
        assert not db.get_active_style_rules("pricing_question")
        assert "pleasantry" not in llm.style_directives("pricing_question")
    finally:
        _cleanup_rules([rid])


def test_ratified_rule_actually_reaches_the_draft_prompt():
    rid = db.add_style_candidate("technical_support", "Answer in at most three sentences.", [3])
    learn.ratify_style(rid)
    captured = {}

    class _Resp:  # minimal shape for resp.choices[0].message.content
        choices = [type("C", (), {"message": type("M", (), {"content": "DRAFT"})()})()]

    def fake_create(**kw):
        captured["messages"] = kw["messages"]
        return _Resp()

    try:
        with mock.patch.object(llm, "_client", lambda: type("X", (), {
                "chat": type("Y", (), {"completions": type("Z", (), {"create": staticmethod(fake_create)})()})()})()):
            llm.draft_reply("customer thread text", "technical_support")
        user_msg = "".join(m["content"] for m in captured["messages"] if m["role"] == "user")
        assert "at most three sentences" in user_msg          # ratified rule reached the prompt
    finally:
        _cleanup_rules([rid])


# ---------- (narrowest scope) ----------------------------------------------
def test_scope_intent_rule_does_not_leak_to_other_intents():
    intent_rid = db.add_style_candidate("pricing_question", "PRICING-ONLY rule.", [1])
    global_rid = db.add_style_candidate("global", "GLOBAL rule.", [1, 9])
    learn.ratify_style(intent_rid)
    learn.ratify_style(global_rid)
    try:
        pricing = llm.style_directives("pricing_question")
        other = llm.style_directives("meeting_request")
        assert "PRICING-ONLY" in pricing and "GLOBAL" in pricing
        assert "PRICING-ONLY" not in other and "GLOBAL" in other   # intent rule stays scoped
    finally:
        _cleanup_rules([intent_rid, global_rid])


# ---------- (4) fact becomes citable, then flips out on revoke --------------
def test_fact_becomes_citable_then_revocable():
    vec = [1.0] + [0.0] * (EMBED_DIM - 1)     # distinctive; exact match => nearest
    cid = db.add_fact_chunk(content="RATIFIED TEST FACT: widgets ship in 2 days.",
                            embedding=vec, origin_draft_id=42)
    try:
        hits = db.search_chunks(vec, k=1)
        assert hits and hits[0]["id"] == cid and hits[0]["source_type"] == "fde_ratified"

        assert db.deactivate_fact_chunk(cid)                 # non-destructive revoke
        hits2 = db.search_chunks(vec, k=1)
        assert not any(h["id"] == cid for h in hits2)        # no longer retrievable/citable
        # kept for audit (row still exists, just inactive)
        s = db.get_session()
        try:
            row = s.get(DocChunk, cid)
            assert row is not None and row.active is False and row.origin_draft_id == 42
        finally:
            s.close()
    finally:
        _cleanup_chunk(cid)


# ---------- propose_lessons: style persists as CANDIDATE (inert), facts held -
def test_propose_lessons_persists_candidates_not_applied():
    deltas = [{"draft_id": 5, "intent": "pricing_question",
               "original": "Hi, warm regards ...", "edited": "Rate is $0.005."}]
    created = []
    real_add = db.add_style_candidate
    def spy_add(scope, rule_text, ev):
        rid = real_add(scope, rule_text, ev); created.append(rid); return rid
    with mock.patch.object(learn, "get_edit_deltas", return_value=deltas), \
         mock.patch.object(llm, "distill_style_rules",
                           return_value={"rules": [{"scope": "pricing_question",
                                                    "rule_text": "State the rate directly.",
                                                    "evidence_draft_ids": [5, 6, 7]}]}), \
         mock.patch.object(learn, "_in_scope_evidence", side_effect=lambda scope, ids: list(dict.fromkeys(ids))), \
         mock.patch.object(llm, "extract_added_facts",
                           return_value={"facts": ["The SMS rate to X is $0.005."]}), \
         mock.patch.object(db, "add_style_candidate", spy_add):
        res = learn.propose_lessons()
    try:
        assert res["style_candidates"] and res["fact_candidates"]
        # style candidate persisted but INERT (status candidate, not in prompts)
        assert not db.get_active_style_rules("pricing_question")
        # fact candidate is NOT yet in the corpus (only carded; enters on ratify)
        assert res["fact_candidates"][0]["fact_text"] == "The SMS rate to X is $0.005."
        # evidence is inline (real before/after), per the card requirement
        assert res["style_candidates"][0]["evidence"][0]["original"] is not None
    finally:
        _cleanup_rules(created)


# ---------- card requirements: evidence inline, scope shown, verbatim fact ---
def test_style_card_shows_scope_and_inline_evidence():
    import json as _json
    from app import slack_approval as sa
    cand = {"rule_id": 7, "scope": "pricing_question", "rule_text": "State the rate directly.",
            "evidence": [{"draft_id": 5, "original": "Hi, warm regards...", "edited": "Rate is $0.005."}]}
    blocks = sa.build_style_card_blocks(cand)
    blob = _json.dumps(blocks)
    assert "pricing_question" in blob                     # (b) scope shown
    assert "State the rate directly." in blob
    assert "warm regards" in blob and "Rate is $0.005." in blob   # (a) inline before/after
    ratify = [e for b in blocks if b.get("type") == "actions" for e in b["elements"]
              if e["action_id"] == sa.ACTION_STYLE_RATIFY]
    assert ratify and ratify[0]["value"] == "7"           # ratify rides the rule_id


def test_fact_card_shows_verbatim_text_and_valid_payload():
    import json as _json
    from app import slack_approval as sa
    fact = "Numbers must complete KYC before India SMS can be enabled."
    cand = {"fact_text": fact, "origin_draft_id": 42,
            "evidence": [{"draft_id": 42, "original": "...", "edited": f"...{fact}"}]}
    blocks = sa.build_fact_card_blocks(cand)
    assert fact in _json.dumps(blocks)                    # (c) VERBATIM text on the card
    ratify = [e for b in blocks if b.get("type") == "actions" for e in b["elements"]
              if e["action_id"] == sa.ACTION_FACT_RATIFY]
    payload = _json.loads(ratify[0]["value"])             # valid JSON (exercises json path)
    assert payload["fact_text"] == fact and payload["origin_draft_id"] == 42


# ---------- structural floor: a style rule needs >=3 in-scope evidence edits -
def test_style_floor_blocks_under_evidenced_candidates():
    deltas = [{"draft_id": 5, "intent": "pricing_question", "original": "a", "edited": "b"}]
    with mock.patch.object(learn, "get_edit_deltas", return_value=deltas), \
         mock.patch.object(llm, "distill_style_rules", return_value={"rules": [
             {"scope": "pricing_question", "rule_text": "TWO-edit claim.", "evidence_draft_ids": [5, 6]},
             {"scope": "pricing_question", "rule_text": "THREE-edit claim.", "evidence_draft_ids": [5, 6, 7]},
         ]}), \
         mock.patch.object(llm, "extract_added_facts", return_value={"facts": []}), \
         mock.patch.object(learn, "_in_scope_evidence", side_effect=lambda scope, ids: list(dict.fromkeys(ids))), \
         mock.patch.object(db, "add_style_candidate", return_value=99):
        res = learn.propose_lessons()
    texts = [c["rule_text"] for c in res["style_candidates"]]
    assert "THREE-edit claim." in texts                    # >=3 -> proposed
    assert "TWO-edit claim." not in texts                  # n=2 -> structurally impossible
    assert any(d["rule_text"] == "TWO-edit claim." for d in res["dropped_under_evidenced"])


def test_in_scope_evidence_filters_by_intent():
    from app.db import Draft, Email, AuditLog
    from sqlalchemy import select
    made = []
    for i, intent in enumerate(("pricing_question", "other")):
        th = {"thread_id": f"scope-test-{i}", "subject": "s",
              "messages": [{"id": f"m{i}", "body": "x"}], "reply_context": {"to": "c@x.com"}}
        made.append((intent, db.persist_processing(th, {"intent": intent}, "d", source="gmail")["draft_id"]))
    pricing_did = [d for it, d in made if it == "pricing_question"][0]
    all_ids = [d for _, d in made]
    try:
        assert learn._in_scope_evidence("pricing_question", all_ids) == [pricing_did]  # intent filters
        assert set(learn._in_scope_evidence("global", all_ids)) == set(all_ids)        # global keeps all
    finally:
        s = db.get_session()
        try:
            for it, did in made:
                d = s.get(Draft, did)
                for al in s.execute(select(AuditLog).where(
                        (AuditLog.draft_id == did) | (AuditLog.email_id == d.email_id))).scalars():
                    s.delete(al)
                eid = d.email_id
                s.delete(d)
                e = s.get(Email, eid)
                if e:
                    s.delete(e)
            s.commit()
        finally:
            s.close()
