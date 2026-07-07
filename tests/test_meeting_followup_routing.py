"""meeting_followup must NEVER free-write: the guarded builder is the intent's
ONLY entry, whether the notes-mail detector or the normal classifier emitted it.
A notes mail from a sender is_notes_mail() doesn't recognize would otherwise take
the unguarded classifier→draft_reply lane. Thin/absent notes → a VISIBLE,
un-enriched hold; an LLM commitment phrasing → the deterministic guard swaps in
the grounded fallback. A fabricated commitment to a customer is the failure mode
these guards exist to prevent.
"""
from unittest import mock

from app import worker, meeting_followup as mf

FREE_WRITE = "FREEWRITE SENTINEL: the notes captured a commitment we never made."

# A representative inline auto-notes body (parseable, NOT thin).
NOTES_BODY = """Notes from "Acme <> ExampleCo"

Open meeting notes

Summary
Team discussed automation APIs for dashboard purchasing, concurrency limits,
and compliance document processing.

Suggested next steps

[Jordan Lee] Share API documentation: Provide the API docs for purchasing
numbers and submitting compliance information.
[Sam Rivera] Request concurrency increase: Send an email when requesting
increases to account concurrency.
"""


def _route(thread, *, classification):
    posted = {}

    def fake_post(te, d, did, flags=None, booking=None):
        posted["draft"], posted["flags"] = d, flags or []
        return {"ts": "1.1"}

    with mock.patch.object(worker.db, "persist_processing", lambda *a, **k: {"draft_id": 1}), \
         mock.patch.object(worker.db, "set_scheduling_state", lambda *a, **k: None), \
         mock.patch.object(worker.slack_approval, "post_draft_once", fake_post):
        worker._route_and_post(thread, "T1", worker.render_thread(thread),
                               classification, FREE_WRITE, [], service=None)
    return posted


def test_thin_notes_mail_holds_never_freewrites():
    thread = {"subject": "Accepted: Acme <> ExampleCo @ Tue 4pm",
              "messages": [{"from": "c@example.com", "body": "Accepted: Acme <> ExampleCo @ Tue 4pm"}],
              "reply_context": {"to": "c@example.com"}}
    posted = _route(thread, classification={"intent": "meeting_followup", "customer_name": "Jordan"})
    assert posted["draft"] != FREE_WRITE
    assert "the notes captured" not in posted["draft"].lower()
    assert any(f["type"] in ("meeting_thin_notes", "meeting_followup_error") for f in posted["flags"])


def test_notes_mail_routes_through_guards_not_freewrite():
    thread = {"subject": "Notes from Acme <> ExampleCo",
              "messages": [{"from": "notes@acme.example", "body": NOTES_BODY}],
              "reply_context": {"to": "notes@acme.example"}}
    assert not mf.is_thin(mf.parse_notes(thread["subject"], NOTES_BODY))

    with mock.patch.object(mf.llm, "draft_meeting_followup",
                           return_value="Hi Sam, as we agreed, you committed to raising the concurrency limit."), \
         mock.patch.object(mf.llm, "rag_groundedness", return_value={"grounded": True}):
        posted = _route(thread, classification={"intent": "meeting_followup", "customer_name": "Sam"})

    assert posted["draft"] != FREE_WRITE
    assert "as we agreed" not in posted["draft"].lower()
    assert "you committed" not in posted["draft"].lower()
    assert any(f["type"] == "meeting_commitment_blocked" for f in posted["flags"])
    assert "noted for" in posted["draft"].lower()


def test_builder_error_holds_visibly_not_freewrite():
    thread = {"subject": "Notes", "messages": [{"from": "c@example.com", "body": NOTES_BODY}],
              "reply_context": {"to": "c@example.com"}}
    with mock.patch.object(mf, "build_followup", side_effect=RuntimeError("boom")):
        posted = _route(thread, classification={"intent": "meeting_followup", "customer_name": None})
    assert posted["draft"] != FREE_WRITE
    assert any(f["type"] == "meeting_followup_error" for f in posted["flags"])
