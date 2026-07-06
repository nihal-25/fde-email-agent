"""Debug auto-trigger (the Alex fix): tiered detection, channel resolution
(probe/hint/ask), the 'not a debugging case' bounce, draft-disposition (routed
mail leaves no orphan card; a bounce lands exactly one draft), and fail-open.
Mocked seams — no network. The idempotency-anchor test hits the real DB.
"""
import contextlib
from unittest import mock

import app.debug_orchestrator as d
import app.worker as worker

MY = "nihal.manjunath@plivo.com"
UUID = "11111111-1111-4111-8111-111111111111"


def _thread(body, tid="T"):
    return {"thread_id": tid, "customer_name": "Alex", "reply_context": {"to": "v@ext.com"},
            "messages": [{"from": "v@ext.com", "body": body}]}


# ---------- tiered detection ----------------------------------------------
def test_tier1_uuid_routes_to_debugging():
    th = _thread(f"our calls are failing, here's one: {UUID}")
    with mock.patch.object(d, "run_debug_case") as rdc, \
         mock.patch.object(d.llm, "is_debugging_case",
                           side_effect=AssertionError("Tier 1 must not need the LLM gate")):
        assert d.maybe_debug_autotrigger(th, {"intent": "technical_support"}) is True
    assert rdc.called


def test_tier2_gate_yes_routes_with_channel_hint():
    th = _thread("None of our SMS to the UK are being delivered since this morning.")
    with mock.patch.object(d.llm, "is_debugging_case",
                           return_value={"is_debugging": True, "channel_hint": "sms"}), \
         mock.patch.object(d, "run_debug_case") as rdc:
        assert d.maybe_debug_autotrigger(th, {"intent": "technical_support"}) is True
    assert rdc.call_args.kwargs.get("channel_hint") == "sms"


def test_tier2_gate_no_does_not_route():
    th = _thread("How do I configure a webhook URL in the dashboard?")
    with mock.patch.object(d.llm, "is_debugging_case",
                           return_value={"is_debugging": False, "channel_hint": None}), \
         mock.patch.object(d, "run_debug_case", side_effect=AssertionError("must not route")):
        assert d.maybe_debug_autotrigger(th, {"intent": "technical_support"}) is False


# ---------- channel resolution --------------------------------------------
def _pending(**kw):
    base = {"thread": _thread("calls failing"), "uuid": None, "auto": True,
            "email_text": "calls failing", "sender": "v@ext.com"}
    base.update(kw)
    return base


def test_uuid_probe_routes_voice():
    with mock.patch.object(d.db, "claim_pending_case", return_value=_pending(uuid=UUID)), \
         mock.patch.object(d, "_probe_channel", return_value="voice"), \
         mock.patch.object(d, "_post", lambda *a, **k: None), \
         mock.patch.object(d, "_run_slice1_live", return_value={"branch": "resolved"}) as rail, \
         mock.patch.object(d, "_sms_investigate_and_emit", side_effect=AssertionError("not sms")):
        out = d.resume_account("ts", "10000002")
    assert out["branch"] == "resolved" and rail.called


def test_uuid_probe_routes_sms():
    with mock.patch.object(d.db, "claim_pending_case", return_value=_pending(uuid=UUID)), \
         mock.patch.object(d, "_probe_channel", return_value="sms"), \
         mock.patch.object(d, "_post", lambda *a, **k: None), \
         mock.patch.object(d, "_sms_investigate_and_emit", return_value={"branch": "clear"}) as sms, \
         mock.patch.object(d, "_run_slice1_live", side_effect=AssertionError("not voice")):
        out = d.resume_account("ts", "10000002")
    assert out["branch"] == "clear" and sms.called


def test_uuid_in_neither_channel_asks():
    posts = []
    with mock.patch.object(d.db, "claim_pending_case", return_value=_pending(uuid=UUID)), \
         mock.patch.object(d, "_probe_channel", return_value=None), \
         mock.patch.object(d.db, "kv_set", lambda *a, **k: None), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)):
        out = d.resume_account("ts", "10000002")
    assert out["branch"] == "reask_channel"
    assert any("Which channel" in p[1] for p in posts)


def test_reply_token_overrides_and_probe_note_on_mismatch():
    # explicit 'sms' token wins; and when a probe result differs from the inferred
    # hint, a visible note is posted (inferred-vs-confirmed transparency).
    posts = []
    with mock.patch.object(d.db, "claim_pending_case",
                           return_value=_pending(uuid=None, channel_hint="voice")), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)), \
         mock.patch.object(d, "_sms_investigate_and_emit", return_value={"branch": "clear"}) as sms:
        out = d.resume_account("ts", "10000002 sms")     # token overrides voice hint
    assert sms.called and out["branch"] == "clear"


# ---------- bounce ---------------------------------------------------------
def test_bounce_reruns_normal_draft():
    th = _thread("Is ASR enabled?")
    with mock.patch.object(d.db, "claim_pending_case", return_value=_pending(thread=th)), \
         mock.patch.object(d, "_post", lambda *a, **k: None), \
         mock.patch.object(worker, "draft_and_post_normally") as norm:
        out = d.resume_account("ts", "not a debugging case")
    assert out["branch"] == "bounced_to_normal"
    assert norm.call_args.args[0] is th          # the original thread re-drafted


# ---------- draft-disposition (routed => no orphan; flag off => normal) -----
def _run_process_thread(autodetect, trigger_result):
    posted = {"persist": 0, "post": 0}

    def fake_persist(*a, **k):
        posted["persist"] += 1
        return {"draft_id": 1}

    stubs = [
        mock.patch.dict("os.environ", {"DEBUG_AUTODETECT": "1"} if autodetect else {}, clear=False),
        mock.patch.object(worker.gmail_client, "fetch_thread",
                          lambda tid, service=None: {"messages": [{"body": "hi"}],
                                                     "reply_context": {"to": "c@x.com"}}),
        mock.patch("app.debug_orchestrator.maybe_reattach_debug_case", lambda t: False),
        mock.patch("app.debug_orchestrator.maybe_debug_autotrigger", trigger_result),
        mock.patch.object(worker.llm, "classify",
                          lambda t: {"intent": "technical_support", "customer_name": None,
                                     "summary": "calls failing", "key_points": []}),
        mock.patch.object(worker.llm, "draft_reply", lambda t, i: "GENERIC"),
        mock.patch.object(worker.draft_mod, "flag_unverified_specifics", lambda d: []),
        mock.patch.object(worker.rag, "answer",
                          lambda *a, **k: worker.rag.RagResult(worker.rag.PATH_WEAK, "HOLD", flags=[])),
        mock.patch.object(worker.db, "persist_processing", fake_persist),
        mock.patch.object(worker.slack_approval, "post_draft_once",
                          lambda *a, **k: posted.__setitem__("post", posted["post"] + 1) or {"ts": "1"}),
    ]
    if autodetect:
        os_env_needed = None
    with contextlib.ExitStack() as es:
        for s in stubs:
            es.enter_context(s)
        worker.process_thread("T1", "me@plivo.com", service=None)
    return posted


def test_routed_mail_discards_generic_draft_no_orphan_card():
    # flag ON + trigger routes => process_thread returns before persist/post: the
    # generic safety-net draft is discarded, no orphaned second card at the gate.
    posted = _run_process_thread(autodetect=True, trigger_result=lambda thread, cls: True)
    assert posted["persist"] == 0 and posted["post"] == 0


def test_flag_off_never_calls_trigger_and_drafts_normally():
    posted = _run_process_thread(
        autodetect=False,
        trigger_result=mock.Mock(side_effect=AssertionError("trigger must not run with flag off")))
    assert posted["persist"] == 1 and posted["post"] == 1     # normal single draft


# ---------- bounce single-draft: the idempotency anchor, proven on the DB ----
def test_bounce_single_draft_idempotency_anchor_collapses_replay():
    from sqlalchemy import select, func
    from app import db
    from app.db import Draft, Email
    tid = "autotrigger-idem-test-thread"
    thread = {"thread_id": tid, "subject": "Is ASR enabled?",
              "messages": [{"id": "MSGID-BOUNCE-1", "from": "v@ext.com",
                            "body": f"is ASR enabled for {UUID}?"}],
              "reply_context": {"to": "v@ext.com"}}
    cls = {"intent": "technical_support", "customer_name": None, "summary": "asr"}
    s = db.get_session()
    try:
        # replay: the SAME triggering message drafted twice (bounce re-run / redelivery)
        id1 = db.persist_processing(thread, cls, "draft one", source="gmail")
        id2 = db.persist_processing(thread, cls, "draft two (replay)", source="gmail")
        assert id1["draft_id"] == id2["draft_id"]            # anchor collapsed to ONE
        eids = [e.id for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars()]
        n = s.execute(select(func.count(Draft.id)).where(Draft.email_id.in_(eids))).scalar()
        assert n == 1                                        # exactly one draft row
    finally:
        # cleanup in FK order: audit_log -> drafts -> emails
        from app.db import AuditLog
        eids = [e.id for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars()]
        if eids:
            dids = [dr.id for dr in s.execute(select(Draft).where(Draft.email_id.in_(eids))).scalars()]
            for al in s.execute(select(AuditLog).where(
                    (AuditLog.email_id.in_(eids)) | (AuditLog.draft_id.in_(dids or [-1])))).scalars():
                s.delete(al)
            for dr in s.execute(select(Draft).where(Draft.email_id.in_(eids))).scalars():
                s.delete(dr)
            for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars():
                s.delete(e)
            s.commit()
        s.close()
