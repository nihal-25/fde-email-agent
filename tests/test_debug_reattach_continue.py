"""Slice-2 re-attach LIVE continuation (dry-run as artifacts): a re-attached
customer reply must land on the real rail — path=uuid onto Slice-1, path=
account_wide through the four-verdict branch — with restore-on-failure so a
throwing continuation never eats the reply, and a clean fresh re-suspend on a
repeat ambiguous. All side-effecting boundaries are mocked, so this is
deterministic and exercises exactly the branching/plumbing.
"""
from unittest import mock

import app.debug_orchestrator as d

THREAD = {"thread_id": "T-thread", "reply_context": {"to": "cust@x.com"},
          "messages": [{"from": "cust@x.com", "body": "still broken"}]}
CLAIMED = {"account_id": 10000001, "thread": {"thread_id": "T-thread"},
           "email_text": "orig", "asked": "uuid/timestamp/destination"}


def _aw(verdict, rep=None):
    return {"account_id": 10000001, "verdict": verdict, "representative_uuid": rep,
            "window": {"start": "s", "end": "e"}, "n_rows": 5, "n_failed": 5,
            "n_normal_excluded": 0, "histogram": {"x": 5}, "errors": []}


# ---- path=uuid → Slice-1 rail --------------------------------------------
def test_reattach_uuid_drills_slice1_rail():
    uuid = "11111111-1111-4111-8111-111111111111"
    # UUID now comes from the customer's LATEST message (not reply_text arg), and
    # only drills if it exists for the account (else falls through to account-wide).
    th = {**THREAD, "messages": [{"from": "cust@x.com", "body": f"here's the call {uuid}"}]}
    with mock.patch.object(d, "_voice_uuid_exists", return_value=True), \
         mock.patch.object(d, "_run_slice1_live",
                           return_value={"branch": "resolved", "verify_ts": "A1"}) as rail:
        art = d._reattach_continue(th, 10000001, CLAIMED, "reply")
    assert art["path"] == "uuid"
    rail.assert_called_once()
    assert rail.call_args.args[2] == uuid          # the customer's UUID (latest msg)
    assert rail.call_args.args[1] == 10000001       # account from the case


# ---- path=account_wide: four verdicts ------------------------------------
def test_account_wide_clear_drills_representative():
    rep = "22222222-2222-4222-8222-222222222222"
    posts = []
    with mock.patch.object(d, "investigate_account_wide", return_value=_aw("clear", rep)), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)), \
         mock.patch.object(d, "_run_slice1_live",
                           return_value={"branch": "resolved", "verify_ts": "A1"}) as rail:
        art = d._account_wide_live(THREAD, 10000001, CLAIMED, "some vague reply")
    assert art["branch"] == "resolved"
    rail.assert_called_once()
    assert rail.call_args.args[2] == rep
    assert any("PATTERN-CLEAR" in p[1] for p in posts)


def test_account_wide_ambiguous_reasks_and_resuspends_fresh():
    sets, posts, cards = [], [], []
    with mock.patch.object(d, "investigate_account_wide", return_value=_aw("ambiguous")), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)), \
         mock.patch.object(d.llm, "debug_customer_ask", lambda t: "Could you share the Call UUID?"), \
         mock.patch.object(d.db, "persist_processing", lambda *a, **k: {"draft_id": 7}), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.append((k, v))), \
         mock.patch("app.slack_approval.post_draft_once",
                    lambda *a, **k: cards.append(a) or {"ts": "1.1"}):
        art = d._account_wide_live(THREAD, 10000001, CLAIMED, "it just fails sometimes")
    assert art["branch"] == "ambiguous" and art["draft_id"] == 7
    assert len(cards) == 1                                   # customer-ask card posted
    # a FRESH dbgcust written on the same thread (the claimed key was already
    # DELETEd by claim, so this is a clean write, no collision)
    fresh = [v for k, v in sets if k == d._CUST_PREFIX + "T-thread"]
    assert fresh and fresh[-1]["account_id"] == 10000001
    assert fresh[-1]["email_text"] == "it just fails sometimes"


def test_account_wide_no_data_holds_no_customer_ask():
    sets, posts, cards = [], [], []
    with mock.patch.object(d, "investigate_account_wide", return_value=_aw("no_data")), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.append((k, v))), \
         mock.patch("app.slack_approval.post_draft_once",
                    lambda *a, **k: cards.append(a)):
        art = d._account_wide_live(THREAD, 10000001, CLAIMED, "vague")
    assert art["branch"] == "no_data"
    assert cards == []                                       # NEVER asks the customer
    # case kept alive for a later retry (re-suspend the claimed payload)
    assert any(k == d._CUST_PREFIX + "T-thread" and v == CLAIMED for k, v in sets)


def test_account_wide_unavailable_holds_no_customer_ask():
    sets, cards = [], []
    with mock.patch.object(d, "investigate_account_wide", return_value=_aw("unavailable")), \
         mock.patch.object(d, "_post", lambda *a, **k: None), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.append((k, v))), \
         mock.patch("app.slack_approval.post_draft_once", lambda *a, **k: cards.append(a)):
        art = d._account_wide_live(THREAD, 10000001, CLAIMED, "vague")
    assert art["branch"] == "unavailable"
    assert cards == []
    assert any(k == d._CUST_PREFIX + "T-thread" and v == CLAIMED for k, v in sets)


# ---- restore-on-failure: a throwing continuation must not eat the reply ---
def test_continuation_failure_restores_case_and_still_returns_true():
    sets = []
    with mock.patch.object(d.db, "kv_get", return_value=CLAIMED), \
         mock.patch.object(d.db, "claim_pending_case", return_value=CLAIMED), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.append((k, v))), \
         mock.patch.object(d, "_reattach_continue", side_effect=RuntimeError("redshift down")), \
         mock.patch.object(d, "_post", lambda *a, **k: None):
        out = d.maybe_reattach_debug_case(THREAD)
    assert out is True                                       # consumed → don't re-classify
    # the claimed case was RESTORED (not eaten)
    assert (d._CUST_PREFIX + "T-thread", CLAIMED) in sets


def test_no_case_returns_false():
    with mock.patch.object(d.db, "kv_get", return_value=None):
        assert d.maybe_reattach_debug_case(THREAD) is False


def test_preclaim_error_fails_open():
    with mock.patch.object(d.db, "kv_get", side_effect=RuntimeError("kv down")):
        assert d.maybe_reattach_debug_case(THREAD) is False
