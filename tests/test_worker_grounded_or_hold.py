"""Fabrication-surface gate: the four generic free-write intents
(account_billing, general_inquiry, feature_request, other) plus the two
doc-answer intents must route through the grounded-or-hold RAG stack, and a RAG
error must HOLD (fixed template) rather than fall back to the ungrounded generic
free-write. Regression anchor: draft 1081's hedged-but-ungrounded GST/tax claims.
"""
from unittest import mock

from app import worker, rag


# A sentinel the generic free-write path would produce; if it ever reaches the
# approval card for a grounded-or-hold intent, the fabrication gate leaked.
FREE_WRITE = "GENERIC FREE-WRITE: GST is typically added, exclusive of taxes."


def _run_process_thread(intent, *, rag_answer=None, rag_raises=None):
    """Drive process_thread with all external boundaries stubbed, returning the
    draft text handed to the approval card (post_draft_once)."""
    posted = {}

    def fake_post(thread_text, draft, draft_id, flags=None, booking=None):
        posted["draft"] = draft
        posted["flags"] = flags
        return {"ts": "1.1"}

    classification = {"intent": intent, "customer_name": "Chris",
                      "summary": "tax discrepancy", "key_points": ["taxes?"]}

    stubs = [
        mock.patch.object(worker.gmail_client, "fetch_thread",
                          lambda tid, service=None: {"messages": [{"body": "hi"}],
                                                     "reply_context": {"to": "cust@x.com"}}),
        mock.patch("app.debug_orchestrator.maybe_reattach_debug_case", lambda t: False),
        mock.patch.object(worker.llm, "classify", lambda t: classification),
        mock.patch.object(worker.llm, "draft_reply", lambda t, i: FREE_WRITE),
        mock.patch.object(worker.draft_mod, "flag_unverified_specifics", lambda d: []),
        mock.patch.object(worker.db, "persist_processing", lambda *a, **k: {"draft_id": 1}),
        mock.patch.object(worker.slack_approval, "post_draft_once", fake_post),
    ]
    if rag_raises is not None:
        stubs.append(mock.patch.object(worker.rag, "answer",
                                       mock.Mock(side_effect=rag_raises)))
    elif rag_answer is not None:
        stubs.append(mock.patch.object(worker.rag, "answer",
                                       mock.Mock(return_value=rag_answer)))

    import contextlib
    with contextlib.ExitStack() as es:
        for s in stubs:
            es.enter_context(s)
        worker.process_thread("T1", "me@plivo.com", service=None)
    return posted


GENERIC_INTENTS = ["account_billing", "general_inquiry", "feature_request", "other"]


def test_generic_intents_route_through_rag_not_free_write():
    grounded = rag.RagResult(rag.PATH_STRONG, "Grounded doc answer.", flags=[])
    for intent in GENERIC_INTENTS + ["platform_query", "technical_support"]:
        posted = _run_process_thread(intent, rag_answer=grounded)
        assert posted["draft"] == "Grounded doc answer.", intent
        assert FREE_WRITE not in posted["draft"], intent


def test_rag_error_holds_and_never_leaks_free_write():
    # On a RAG failure the gate must HOLD (fixed template), NOT keep the generic
    # free-write. This is the leak the fix closes.
    for intent in GENERIC_INTENTS:
        posted = _run_process_thread(intent, rag_raises=RuntimeError("boom"))
        assert FREE_WRITE not in posted["draft"], intent
        assert "GST" not in posted["draft"], intent
        # It is exactly the fixed holding template.
        assert posted["draft"] == rag.holding_reply("Chris", intent), intent
        assert any(f["type"] == "rag_error_hold" for f in posted["flags"]), intent


def test_hold_result_carries_no_ungrounded_claims():
    # When rag itself holds (its normal grounded-or-hold behaviour), the posted
    # draft is the fixed template with none of the free-write's claims.
    hold = rag.RagResult(rag.PATH_WEAK, rag.holding_reply("Chris", "account_billing"),
                         flags=[{"type": "rag_holding", "text": "held"}])
    posted = _run_process_thread("account_billing", rag_answer=hold)
    for claim in ("GST", "exclusive of taxes", "local taxes"):
        assert claim not in posted["draft"]
