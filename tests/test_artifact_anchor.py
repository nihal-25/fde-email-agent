"""The dedup-anchor fix: a non-primary artifact (debug answer, meeting follow-up)
must encode ARTIFACT identity, so it lands as its own card over a pre-existing
hold yet still collapses its own replays. Plus the empty/file-only reply guard:
a file_share reply must NOT consume the case. Injection tests — the first two hit
the real DB (the anchor lives there); the rest mock the seams.
"""
from unittest import mock

import app.debug_orchestrator as d
from app import db
from app.db import Draft, Email, AuditLog
from sqlalchemy import select, func


def _cleanup(tid):
    s = db.get_session()
    try:
        eids = [e.id for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars()]
        if eids:
            dids = [x.id for x in s.execute(select(Draft).where(Draft.email_id.in_(eids))).scalars()]
            for al in s.execute(select(AuditLog).where(
                    (AuditLog.email_id.in_(eids)) | (AuditLog.draft_id.in_(dids or [-1])))).scalars():
                s.delete(al)
            for x in s.execute(select(Draft).where(Draft.email_id.in_(eids))).scalars():
                s.delete(x)
            for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars():
                s.delete(e)
            s.commit()
    finally:
        s.close()


def test_build_artifact_key_is_distinct_from_trigger_anchor():
    thread = {"thread_id": "T", "messages": [{"body": "x"}]}   # no msg id -> content-hash
    assert db.build_artifact_key("T", "debug", "verify1") != db._idempotency_key(thread)


def test_debug_answer_lands_over_preexisting_hold():
    # THE Alex failure, by injection: a hold exists for the mail; the debug answer
    # for the SAME mail must land as its own card, NOT collapse into the hold.
    tid = "anchor-test-lands"
    thread = {"thread_id": tid, "subject": "ASR?", "messages": [{"body": "is ASR enabled?"}],
              "reply_context": {"to": "v@x.com"}}
    cls = {"intent": "technical_support", "customer_name": None, "summary": "asr"}
    try:
        hold = db.persist_processing(thread, cls, "GENERIC HOLD", source="gmail")   # trigger anchor
        answer = db.persist_processing(thread, cls, "REAL DEBUG ANSWER", source="debug",
                                       artifact_key=db.build_artifact_key(tid, "debug", "verify-777"))
        assert hold["draft_id"] != answer["draft_id"]        # landed, not collapsed
        assert not answer.get("deduped")
        s = db.get_session()
        try:
            eids = [e.id for e in s.execute(select(Email).where(Email.thread_id == tid)).scalars()]
            n = s.execute(select(func.count(Draft.id)).where(Draft.email_id.in_(eids))).scalar()
            assert n == 2                                    # both cards exist
        finally:
            s.close()
    finally:
        _cleanup(tid)


def test_debug_replay_still_collapses():
    tid = "anchor-test-replay"
    thread = {"thread_id": tid, "subject": "ASR?", "messages": [{"body": "is ASR enabled?"}],
              "reply_context": {"to": "v@x.com"}}
    cls = {"intent": "technical_support", "customer_name": None, "summary": "asr"}
    ak = db.build_artifact_key(tid, "debug", "verify-777")
    try:
        a1 = db.persist_processing(thread, cls, "ANSWER", source="debug", artifact_key=ak)
        a2 = db.persist_processing(thread, cls, "ANSWER (replay)", source="debug", artifact_key=ak)
        assert a1["draft_id"] == a2["draft_id"] and a2.get("deduped")   # same artifact -> collapse
    finally:
        _cleanup(tid)


# ---------- empty / file-only reply must NOT consume the case --------------
def test_resume_empty_reply_does_not_claim_or_consume():
    with mock.patch.object(d.db, "claim_pending_case",
                           side_effect=AssertionError("empty reply must not claim the case")):
        for bad in ("", "   ", None):
            assert d.resume("ts", bad) == {"branch": "empty_reply_ignored"}


def test_file_share_catchup_nudges_and_does_not_consume():
    sets, posts = {}, []
    def fake_keys(prefix, *a, **k):
        return [prefix + "TS1"] if prefix == d._PENDING_PREFIX else []
    with mock.patch.object(d.db, "list_pending_case_keys", fake_keys), \
         mock.patch.object(d, "_find_nihal_reply", return_value=(None, True)), \
         mock.patch.object(d, "resume", side_effect=AssertionError("file-only must NOT resume/consume")), \
         mock.patch.object(d, "resume_account", side_effect=AssertionError("no acct resume")), \
         mock.patch.object(d.db, "kv_get", return_value={"account_id": 1}), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.update({k: v})), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append((a, k))):
        n = d.catch_up_pending_cases()
    assert n == 0                                            # nothing consumed
    assert sets and list(sets.values())[0].get("nudged_filetext")   # dampen marker set
    assert any("paste" in a[1].lower() for a, k in posts)    # 'paste as text' nudge


def test_file_only_reply_is_dampened_second_tick_silent():
    posts = []
    def fake_keys(prefix, *a, **k):
        return [prefix + "TS1"] if prefix == d._PENDING_PREFIX else []
    with mock.patch.object(d.db, "list_pending_case_keys", fake_keys), \
         mock.patch.object(d, "_find_nihal_reply", return_value=(None, True)), \
         mock.patch.object(d, "resume", side_effect=AssertionError("no resume")), \
         mock.patch.object(d, "resume_account", side_effect=AssertionError("no acct")), \
         mock.patch.object(d.db, "kv_get", return_value={"account_id": 1, "nudged_filetext": True}), \
         mock.patch.object(d.db, "kv_set", side_effect=AssertionError("must not re-mark")), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)):
        d.catch_up_pending_cases()
    assert posts == []                                       # already nudged -> silent
