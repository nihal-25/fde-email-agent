"""Slice-3 G2 — dry-run the SMS verdict/interp/draft SHAPES as artifacts, plus
the channel-tagged re-attach routing. All Redshift/Slack/DB boundaries mocked, so
this is deterministic. Covers: all five verdicts + mixed, out_of_scope vs no_data,
the recency-truncation requirement (aggregate decides, row-cap only disclosed),
the healthy-window data-acknowledging ask, and the channel-absent legacy → voice.
"""
from unittest import mock

import app.debug_orchestrator as d


def _hist(rows):
    """rows: list of (state, carrier_code, country, n) → aggregate shape."""
    return [{"delivery_state": s, "carrier_code": e, "country_iso": c, "n": n} for s, e, c, n in rows]


# ---------- verdict engine over EXACT aggregate buckets --------------------
def test_verdict_clear_dominant_code():
    b = d._sms_buckets(_hist([("failed", "451", "AU", 40), ("failed", "300", "US", 3),
                              ("delivered", "000", "US", 500)]))
    v = d._sms_verdict(b)
    assert v["verdict"] == "clear" and v["dom_code"] == "451" and v["failed_finding"]


def test_verdict_limbo_dominates():
    b = d._sms_buckets(_hist([("failed", "451", "AU", 2), ("sent", "", "US", 150),
                              ("queued", "", "US", 50), ("delivered", "000", "US", 10)]))
    v = d._sms_verdict(b)
    assert v["verdict"] == "limbo" and v["limbo_finding"] and not v["failed_finding"]


def test_verdict_mixed_reports_both_not_clear_on_the_six():
    # THE ordering-bug case: 6 failed (dominant code) + 200 limbo must be MIXED,
    # never CLEAR on the 6.
    b = d._sms_buckets(_hist([("failed", "451", "AU", 6), ("sent", "", "US", 200),
                              ("delivered", "000", "US", 20)]))
    v = d._sms_verdict(b)
    assert v["verdict"] == "mixed" and v["failed_finding"] and v["limbo_finding"]
    interp = d._sms_interpretation({"buckets": b, "verdict": "mixed", **v})
    assert "1)" in interp and "2)" in interp  # BOTH findings stated


def test_verdict_ambiguous_no_dominance():
    b = d._sms_buckets(_hist([("failed", "451", "AU", 3), ("failed", "300", "US", 3),
                              ("failed", "456", "BR", 2), ("delivered", "000", "US", 5)]))
    assert d._sms_verdict(b)["verdict"] == "ambiguous"


def test_verdict_ambiguous_healthy_window_acknowledges_data():
    b = d._sms_buckets(_hist([("delivered", "000", "US", 900), ("sent", "", "US", 2)]))
    inv = {"buckets": b, "window": {"start": "s", "end": "e"}, **d._sms_verdict(b)}
    assert inv["verdict"] == "ambiguous"
    msg = d._sms_customer_message(inv, "Priya")
    assert "show as delivered on our side" in msg and "failed" in msg  # acknowledges + asks


# ---------- no_data vs out_of_scope (addition 2) ---------------------------
def _thread():
    return {"thread_id": "T-sms", "customer_name": "Priya",
            "reply_context": {"to": "c@x.com"}, "messages": [{"body": "sms failing"}]}


def test_no_data_when_truly_empty():
    with mock.patch.object(d.redshift_tools, "get_message_histogram", return_value=[]), \
         mock.patch.object(d.redshift_tools, "get_message_type_breakdown", return_value=[]), \
         mock.patch.object(d, "_extract_window", return_value={"start": "s", "end": "e", "has_hint": False}):
        inv = d.investigate_sms_account_wide(_thread(), 10000002)
    assert inv["verdict"] == "no_data"


def test_out_of_scope_when_traffic_is_whatsapp():
    with mock.patch.object(d.redshift_tools, "get_message_histogram", return_value=[]), \
         mock.patch.object(d.redshift_tools, "get_message_type_breakdown",
                           return_value=[{"message_type": "whatsapp", "n": 120}]), \
         mock.patch.object(d, "_extract_window", return_value={"start": "s", "end": "e", "has_hint": False}):
        inv = d.investigate_sms_account_wide(_thread(), 10000002)
    assert inv["verdict"] == "out_of_scope" and inv["type_breakdown"] == {"whatsapp": 120}


def test_unavailable_when_query_errors():
    with mock.patch.object(d.redshift_tools, "get_message_histogram", side_effect=RuntimeError("boom")), \
         mock.patch.object(d, "_extract_window", return_value={"start": "s", "end": "e", "has_hint": False}):
        inv = d.investigate_sms_account_wide(_thread(), 10000002)
    assert inv["verdict"] == "unavailable" and inv["errors"]


# ---------- recency-truncation requirement (the upgraded observation) ------
def test_verdict_is_truncation_immune_and_cap_is_disclosed():
    # Exact aggregate: an EARLY failure burst (450) that a recency-capped row query
    # would hide behind healthy recent traffic. Verdict must be CLEAR from the
    # aggregate; the bounded representative query hitting its cap is DISCLOSED.
    big = _hist([("failed", "451", "AU", 450), ("delivered", "000", "US", 100000)])
    capped_rows = [{"message_uuid": f"u{i}", "delivery_state": "failed", "carrier_code": "451"}
                   for i in range(5)]  # len == LIMIT(5) → capped
    with mock.patch.object(d.redshift_tools, "get_message_histogram", return_value=big), \
         mock.patch.object(d.redshift_tools, "get_messages_for_account", return_value=capped_rows), \
         mock.patch.object(d, "_extract_window", return_value={"start": "s", "end": "e", "has_hint": False}):
        inv = d.investigate_sms_account_wide(_thread(), 10000002)
    assert inv["verdict"] == "clear" and inv["buckets"]["failed"] == 450  # exact, not capped
    assert inv["representative_uuid"] == "u0"
    assert "representative sample only" in inv["cap_note"]                # cap disclosed
    assert "representative sample only" in d._sms_header(inv)


# ---------- channel-tagged re-attach dispatch (incl legacy → voice) --------
def test_reattach_routes_sms_channel_to_sms_continuation():
    sms = mock.Mock(return_value={"path": "account_wide", "channel": "sms", "branch": "clear"})
    with mock.patch.object(d, "_sms_reattach_continue", sms):
        out = d._reattach_continue(_thread(), 10000002, {"account_id": 10000002, "channel": "sms"}, "vague")
    assert out["channel"] == "sms" and sms.called


def _thread_msgs(bodies):
    return {"thread_id": "T", "customer_name": "Priya", "reply_context": {"to": "c@x.com"},
            "messages": [{"from": "c@x.com", "body": b} for b in bodies]}


def test_reattach_channel_absent_legacy_routes_to_voice():
    # EXPLICIT assertion (change 2): a channel-absent legacy dbgcust must hit the
    # VOICE continuation, never SMS.
    assert d._reattach_channel({"account_id": 1}) == "voice"
    assert d._reattach_channel({"account_id": 1, "channel": "voice"}) == "voice"
    voice_uuid = "11111111-1111-4111-8111-111111111111"
    th = _thread_msgs([f"here is the failing call {voice_uuid}"])
    with mock.patch.object(d, "_voice_uuid_exists", return_value=True), \
         mock.patch.object(d, "_run_slice1_live", return_value={"branch": "resolved"}) as vrail, \
         mock.patch.object(d, "_sms_reattach_continue", side_effect=AssertionError("must not route to SMS")):
        out = d._reattach_continue(th, 555, {"account_id": 555}, "reply")
    assert out["channel"] == "voice" and vrail.called
    assert vrail.call_args.args[2] == voice_uuid


# ---------- G3 bug fix: latest-message UUID + not-found fall-through --------
def test_latest_uuid_ignores_stale_history():
    # The live bug: a stale UUID from an earlier turn (+ quoted chains) must NOT be
    # picked; the customer's LATEST message decides.
    stale = "11111111-1111-4111-8111-111111111111"
    fresh = "33333333-3333-4333-8333-333333333333"
    th = _thread_msgs([f"old context {stale}",
                       f"quoted {stale} {stale}",
                       f"here's the one that failed {fresh}\n\n> quoted history {stale}"])
    assert d._latest_uuid(th) == fresh          # latest message's first UUID, not history's
    assert d._latest_uuid(_thread_msgs(["no uuid here"])) is None


def test_sms_reattach_not_found_falls_through_to_account_wide():
    fresh = "33333333-3333-4333-8333-333333333333"
    th = _thread_msgs([f"failing msg {fresh}"])
    posts, dispatched = [], []
    with mock.patch.object(d.redshift_tools, "get_message_by_uuid", return_value={"found": False}), \
         mock.patch.object(d, "investigate_sms_account_wide", return_value={"verdict": "clear"}), \
         mock.patch.object(d, "_sms_dispatch_verdict",
                           lambda *a, **k: dispatched.append(True) or {"branch": "clear"}), \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)):
        out = d._sms_reattach_continue(th, 10000002, {"account_id": 10000002, "channel": "sms"}, "reply")
    assert out["path"] == "uuid" and dispatched            # fell through to account-wide
    assert any("not found for this" in p[1] and "account-wide" in p[1] for p in posts)  # VISIBLE miss


def test_voice_reattach_not_found_falls_through_symmetric():
    # Voice symmetry (change 3): a not-found voice UUID must also post a miss note
    # and fall through to account-wide, not dead-end + strand the case.
    stale = "11111111-1111-4111-8111-111111111111"
    th = _thread_msgs([f"failing call {stale}"])
    posts = []
    with mock.patch.object(d, "_voice_uuid_exists", return_value=False), \
         mock.patch.object(d, "_run_slice1_live", side_effect=AssertionError("must NOT drill a not-found uuid")), \
         mock.patch.object(d, "_account_wide_live", return_value={"branch": "clear"}) as aw, \
         mock.patch.object(d, "_post", lambda *a, **k: posts.append(a)):
        out = d._reattach_continue(th, 555, {"account_id": 555}, "reply")
    assert out["path"] == "account_wide" and out["channel"] == "voice" and aw.called
    assert any("not found for this" in p[1] and "account-wide" in p[1] for p in posts)


# ---------- G3 wiring: invoke entry + resume dispatch ----------------------
def test_run_sms_case_tags_channel_sms():
    art = d.run_sms_case(_thread(), dry_run=True)
    assert art["channel"] == "sms" and "SMS" in art["account_ask_post"]
    # live path persists the dbgacct pending TAGGED channel='sms'
    sets = {}
    with mock.patch.object(d, "_post", lambda *a, **k: {"ts": "9.9"}), \
         mock.patch.object(d.db, "kv_set", lambda k, v: sets.update({k: v})):
        d.run_sms_case(_thread(), dry_run=False)
    acct = sets[d._ACCT_PREFIX + "9.9"]
    assert acct["channel"] == "sms"


def test_resume_account_sms_channel_routes_to_sms_dispatch():
    pending = {"thread": _thread(), "uuid": None, "email_text": "sms failing", "channel": "sms"}
    with mock.patch.object(d.db, "claim_pending_case", return_value=pending), \
         mock.patch.object(d, "investigate_sms_account_wide",
                           return_value={"verdict": "clear", "window": {"start": "s", "end": "e"}}), \
         mock.patch.object(d, "_sms_dispatch_verdict",
                           return_value={"branch": "clear", "draft_id": 1}) as smsd, \
         mock.patch.object(d, "investigate_account_wide",
                           side_effect=AssertionError("must NOT use the voice path")):
        out = d.resume_account("ts1", "10000002")
    assert out["branch"] == "clear" and smsd.called


def test_resume_account_voice_unchanged_regression():
    # A channel-absent (voice) account-ask must still hit the voice account-wide
    # path — the one change to proven code must not disturb it.
    pending = {"thread": _thread(), "uuid": None, "email_text": "calls failing"}
    with mock.patch.object(d.db, "claim_pending_case", return_value=pending), \
         mock.patch.object(d, "investigate_account_wide",
                           return_value={"verdict": "no_data", "window": {"start": "s", "end": "e"},
                                         "histogram": {}, "n_rows": 0, "n_failed": 0,
                                         "n_normal_excluded": 0, "errors": [], "representative_uuid": None}), \
         mock.patch.object(d, "_sms_dispatch_verdict",
                           side_effect=AssertionError("must NOT use the SMS path")), \
         mock.patch.object(d, "_post", lambda *a, **k: {"ts": "x"}), \
         mock.patch.object(d.db, "kv_set", lambda *a, **k: None):
        out = d.resume_account("ts2", "10000001")
    assert out["branch"] == "no_data"
