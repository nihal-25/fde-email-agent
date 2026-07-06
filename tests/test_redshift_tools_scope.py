"""Account-scope invariant for the SMS/MDR tools (security-critical, CLAUDE.md).

These assert the scoping is enforced in CODE, not by prompt: every query binds
account_id, no unscoped fallback is ever issued, and a miss is a flat not-found
that leaks no row data and does not disclose existence under another account.
Redshift is mocked — no warehouse access, deterministic.
"""
from unittest import mock

from app import redshift_tools as rt

FAILED_ROW = {
    "account_id": 10000002, "sub_account_id": 0,
    "message_uuid": "33333333-3333-4333-8333-333333333333",
    "parent_message_uuid": None, "message_time": "2026-07-01 00:04:29",
    "delivery_state": "failed", "message_direction": "outbound", "message_type": "sms",
    "to_number_redacted": "61468601***", "from_number_redacted": "1855***",
    "country_iso": "AU", "carrier_code": "451", "error_code": 64, "units": 1,
    "carrier_id": "x", "tier": "t",
}


def test_get_message_by_uuid_binds_account_and_uuid():
    captured = {}

    def fake_query_one(sql, params):
        captured["sql"], captured["params"] = sql, params
        return FAILED_ROW

    with mock.patch.object(rt.redshift, "query_one", fake_query_one):
        out = rt.get_message_by_uuid("33333333-3333-4333-8333-333333333333", 10000002)

    assert out["found"] and out["channel"] == "sms"
    # account_id is in the WHERE, bound as a parameter (not interpolated)
    assert "account_id = %s" in captured["sql"]
    assert "message_uuid = %s" in captured["sql"]
    assert captured["params"] == ("33333333-3333-4333-8333-333333333333", 10000002)


def test_miss_is_flat_not_found_no_row_data_no_second_query():
    calls = []

    def fake_query_one(sql, params):
        calls.append((sql, params))
        return None  # no row for this (uuid, account)

    with mock.patch.object(rt.redshift, "query_one", fake_query_one):
        out = rt.get_message_by_uuid("33333333-3333-4333-8333-333333333333", 10000003)

    assert out == {"found": False, "reason": "not found for this account",
                   "message_uuid": "33333333-3333-4333-8333-333333333333", "account_id": 10000003}
    # exactly ONE scoped query — no unscoped existence-probe fallback
    assert len(calls) == 1 and "account_id = %s" in calls[0][0]
    # no message data leaked
    for leaky in ("delivery_state", "country_iso", "carrier_code", "to_number_redacted"):
        assert leaky not in out


def test_scope_guard_fires_on_mismatched_row():
    # Defense-in-depth: if the warehouse ever returned a foreign row, assert.
    bad = {**FAILED_ROW, "account_id": 99999999}
    with mock.patch.object(rt.redshift, "query_one", lambda s, p: bad):
        try:
            rt.get_message_by_uuid(FAILED_ROW["message_uuid"], 10000002)
            assert False, "expected scope-guard assertion"
        except AssertionError as e:
            assert "account-scope guard violated" in str(e)


def test_messages_for_account_always_scoped_and_defaults_outbound_sms():
    captured = {}

    def fake_query(sql, params):
        captured["sql"], captured["params"] = sql, params
        return []

    with mock.patch.object(rt.redshift, "query", fake_query):
        rt.get_messages_for_account(10000002, "2026-07-01 00:00:00", "2026-07-01 23:59:59")

    assert "account_id = %s" in captured["sql"]
    assert "message_type = %s" in captured["sql"] and "message_direction = %s" in captured["sql"]
    # account_id is first-bound; defaults are outbound sms
    assert captured["params"][0] == 10000002
    assert "sms" in captured["params"] and "outbound" in captured["params"]


def test_messages_for_account_all_types_for_out_of_scope_check():
    # message_type=None drops the type filter so a WhatsApp/MMS-only window can be
    # distinguished from a truly empty one (still account-scoped).
    captured = {}
    with mock.patch.object(rt.redshift, "query",
                           lambda s, p: captured.update(sql=s, params=p) or []):
        rt.get_messages_for_account(10000002, "s", "e", message_type=None, direction=None)
    assert "account_id = %s" in captured["sql"]
    assert "message_type = %s" not in captured["sql"]
    assert "message_direction = %s" not in captured["sql"]


def test_histogram_is_account_scoped_aggregate_no_row_limit():
    captured = {}
    with mock.patch.object(rt.redshift, "query",
                           lambda s, p: captured.update(sql=s, params=p) or []):
        rt.get_message_histogram(10000002, "2026-07-01 00:00:00", "2026-07-01 23:59:59")
    sql = captured["sql"]
    assert "account_id = %s" in sql and "group by" in sql.lower()
    assert "limit" not in sql.lower()          # NO row cap -> truncation-immune
    assert captured["params"][0] == 10000002


def test_type_breakdown_is_account_scoped():
    captured = {}
    with mock.patch.object(rt.redshift, "query",
                           lambda s, p: captured.update(sql=s, params=p) or []):
        rt.get_message_type_breakdown(10000002, "s", "e")
    assert "account_id = %s" in captured["sql"] and "group by message_type" in captured["sql"].lower()
    assert captured["params"][0] == 10000002
