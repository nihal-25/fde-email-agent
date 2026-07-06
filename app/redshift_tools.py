"""Account-scoped, read-only Redshift debugging tools (Phase 4).

Every function here is account-scoped BY CONSTRUCTION:
- `account_id` is a required argument (no default, no flag to disable it).
- It is always bound into the WHERE clause as a parameter, so the warehouse
  only ever returns rows for that account — another account's row never leaves
  Redshift.
- UUID lookups additionally assert the returned row's account_id equals the
  passed account_id (belt-and-suspenders) and return a distinct
  "not found for this account" when the UUID exists under a *different* account
  (the existence probe reads no row data, so nothing leaks).

The account_id is injected by our calling code (the verified sender's account),
never chosen by the model and never taken from untrusted email text.

Schema was verified live (see discovery probe), not guessed:
  voice CDR : calls_fact      (account_id bigint,  ts: start_time)
  sip_trunk  : trunk_calls_fact (account_id integer, ts: initiation_time)
  SMS  MDR  : messages_fact       (account_id integer, ts: message_time,
                                            uuid col: message_uuid)

Known gap (flagged, never fabricated): answer_url / endpoint-config do NOT exist
in this warehouse. For SIP trunk we expose the routing fields that DO exist
(trunk_domain, carrier_gateway, carrier_id, region, + parsed extra_data); the
answer_url / endpoint-config must be sourced elsewhere (app-config DB / the reviewer).
"""

from __future__ import annotations

import json

from app import redshift

# Verified table identifiers (hardcoded — never built from caller input).
_VOICE = "calls_fact"
_ZENTRUNK = "trunk_calls_fact"
_MDR = "messages_fact"  # SMS/messaging (Message Detail Records)

ANSWER_URL_GAP = (
    "answer_url / endpoint-config is not available in Redshift; source it from "
    "the app-config DB / the reviewer. Do not promise it from this tool."
)

# --- curated column sets (debugging-relevant subset, not all 138 cols) -------
_VOICE_COLS = (
    "account_id, subaccount_id, call_uuid, parent_call_uuid, request_uuid, "
    "session_uuid, call_direction, call_state, call_leg, disconnect_reason, "
    "disconnect_reason_name, disconnect_code, disconnect_source, "
    "to_number, from_number, from_iso, country_iso, start_time, answer_time, "
    "end_time, duration, bill_duration, ring_time, post_dial_delay, "
    "carrier_name, application_type, application_id, via_trunk, currency"
)
_ZT_COLS = (
    "account_id, subaccount_id, call_uuid, call_id, call_direction, "
    "disconnect_reason, disconnect_code, disconnect_initiator, to_number, from_number, "
    "from_country, to_country, initiation_time, answer_time, end_time, "
    "duration, bill_duration, ring_time, route_type, region, trunk_domain, "
    "carrier_gateway, carrier_id, transport_protocol, srtp, extra_data"
)
_VOICE_QUALITY_COLS = (
    "account_id, call_uuid, mos_avg, rtt_avg, in_jitter_avg, out_jitter_avg, "
    "in_fraction_lost_avg, out_fraction_lost_avg, self_leg_suspected_issues, "
    "all_legs_suspected_issues, plivo_rtcp_status, remote_rtcp_status, client_issues"
)
# SMS/MDR: debugging-relevant subset. carrier_code / error_code are OPAQUE
# carrier DLR status codes — reported as-is, never mapped to a cause (no code->
# meaning table exists; a guess would be the 1078 fabrication class). Numbers are
# already redacted at source (to/from_number_redacted).
_MDR_COLS = (
    "account_id, sub_account_id, message_uuid, parent_message_uuid, message_time, "
    "delivery_state, message_direction, message_type, to_number_redacted, "
    "from_number_redacted, country_iso, carrier_code, error_code, units, "
    "carrier_id, tier"
)
# Lighter projection for account-wide histogram scans.
_MDR_SUMMARY_COLS = (
    "message_uuid, message_time, delivery_state, message_direction, message_type, "
    "to_number_redacted, from_number_redacted, country_iso, carrier_code, "
    "error_code, units"
)


def _json(value) -> dict | None:
    """Best-effort parse of a JSON/super column into a dict; None on failure."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _scoped_lookup(table: str, columns: str, uuid: str, account_id, *,
                   uuid_col: str = "call_uuid"):
    """Fetch ONE row for (uuid, account_id), ALWAYS scoped by account_id.

    Returns the row, or None if no row matches BOTH the uuid and the account.
    `uuid_col` (a hardcoded constant, never caller input) is the id column —
    call_uuid for voice/sip_trunk, message_uuid for SMS.

    There is deliberately NO unscoped fallback / existence check: these tools
    never issue a query without the account_id filter (no full-table scan), and
    never reveal whether a UUID exists under some other account — that existence
    disclosure would itself be a leak. No-row is reported flatly as
    "not found for this account".
    """
    acct = int(account_id)  # numeric scope; reject non-numeric early
    row = redshift.query_one(
        f"select {columns} from {table} where {uuid_col} = %s and account_id = %s limit 1",
        (uuid, acct),
    )
    if row is not None:
        # The WHERE already guarantees this; assert to make the invariant loud.
        assert int(row["account_id"]) == acct, "account-scope guard violated"
    return row


def _not_found(uuid: str, account_id, *, uuid_key: str = "call_uuid") -> dict:
    # Flat message — we do NOT distinguish "wrong account" from "absent", because
    # confirming a UUID exists under another account would itself leak.
    return {"found": False, "reason": "not found for this account",
            uuid_key: uuid, "account_id": int(account_id)}


# --- public tools ------------------------------------------------------------
def get_call_by_uuid(uuid: str, account_id) -> dict:
    """Voice CDR row (calls_fact) for `uuid`, scoped to `account_id`."""
    row = _scoped_lookup(_VOICE, _VOICE_COLS, uuid, account_id)
    if row is None:
        return _not_found(uuid, account_id)
    return {"found": True, "channel": "voice", "source_table": _VOICE, **dict(row)}


def get_sip_trunk_call_detail(uuid: str, account_id) -> dict:
    """SIP trunk CDR detail (trunk_calls_fact) for `uuid`, scoped.

    Surfaces the failure-reason fields (disconnect_reason / disconnect_code /
    disconnect_initiator + parsed extra_data.hangup_data sipcode/sipresponse) and
    the routing/endpoint fields that exist (trunk_domain, carrier_gateway,
    carrier_id, region, + trunk_signaling_ip / carrier_details). answer_url and
    endpoint-config are flagged as a known gap — never fabricated.
    """
    row = _scoped_lookup(_ZENTRUNK, _ZT_COLS, uuid, account_id)
    if row is None:
        return _not_found(uuid, account_id)

    out = {"found": True, "channel": "sip_trunk", "source_table": _ZENTRUNK, **dict(row)}
    extra = _json(out.pop("extra_data", None)) or {}
    hangup_data = extra.get("hangup_data") or {}
    out["sip_code"] = hangup_data.get("sipcode")
    out["sip_response"] = hangup_data.get("sipresponse")
    call_stats = extra.get("call_stats") or {}
    out["trunk_signaling_ip"] = call_stats.get("trunk_signaling_ip")
    out["carrier_details"] = extra.get("carrier_details")
    # Known gap — surfaced explicitly so the orchestrator never promises these.
    out["answer_url"] = None
    out["endpoint_config"] = None
    out["_gaps"] = ANSWER_URL_GAP
    return out


# --- SMS / MDR tools (mirror the voice tools; account-scoped by construction) --
def get_message_by_uuid(uuid: str, account_id) -> dict:
    """SMS/MDR row (messages_fact) for `uuid`, scoped to `account_id`.

    Surfaces the outcome + failure fields that EXIST (delivery_state, carrier_code,
    error_code) plus destination/sender/route context. carrier_code /
    error_code are returned as OPAQUE codes — this tool never maps a code to
    a cause (that mapping does not exist here; a guess would be fabrication).
    """
    row = _scoped_lookup(_MDR, _MDR_COLS, uuid, account_id, uuid_col="message_uuid")
    if row is None:
        return _not_found(uuid, account_id, uuid_key="message_uuid")
    return {"found": True, "channel": "sms", "source_table": _MDR, **dict(row)}


def get_messages_for_account(account_id, start, end, *, direction: str | None = "outbound",
                             message_type: str | None = "sms", delivery_state=None,
                             carrier_code=None, limit: int = 500) -> list[dict]:
    """List messages for `account_id` in [start, end], scoped.

    Defaults to OUTBOUND SMS (the failure-analysis scope). Pass message_type=None
    to see ALL messaging traffic (used to tell a true no_data window apart from a
    WhatsApp/MMS out-of-scope one). Optional delivery_state / carrier_code narrow the
    result (used to pick a representative failed message matching the dominant
    code). account_id is ALWAYS filtered in-function; every value is bound as a
    parameter.

    NOTE: this is a bounded (ORDER BY message_time DESC LIMIT) query — recency-
    biased. It is used ONLY for representative-UUID selection and single-message
    facts, NEVER for verdict counts (those come from get_message_histogram, which
    has no row limit). Callers surfacing these rows must disclose the cap.
    """
    acct = int(account_id)
    sql = (f"select {_MDR_SUMMARY_COLS} from {_MDR} "
           f"where account_id = %s and message_time between %s and %s")
    params: list = [acct, start, end]
    if message_type is not None:
        sql += " and message_type = %s"
        params.append(message_type)
    if direction is not None:
        sql += " and message_direction = %s"
        params.append(direction)
    if delivery_state is not None:
        sql += " and delivery_state = %s"
        params.append(delivery_state)
    if carrier_code is not None:
        sql += " and carrier_code = %s"
        params.append(carrier_code)
    sql += " order by message_time desc limit %s"
    params.append(int(limit))
    return redshift.query(sql, tuple(params))


def get_message_histogram(account_id, start, end, *, direction: str | None = "outbound",
                          message_type: str | None = "sms") -> list[dict]:
    """EXACT counts for [account_id, window] grouped by (delivery_state, carrier_code,
    country_iso). Scans the FULL window with NO row limit, so the verdict it feeds
    is immune to recency truncation (an early failure burst can't be hidden behind
    healthy recent traffic). Row count is bounded by distinct combos, not message
    volume. account_id is ALWAYS filtered in-function."""
    acct = int(account_id)
    sql = (f"select delivery_state, carrier_code, country_iso, count(*) as n from {_MDR} "
           f"where account_id = %s and message_time between %s and %s")
    params: list = [acct, start, end]
    if message_type is not None:
        sql += " and message_type = %s"
        params.append(message_type)
    if direction is not None:
        sql += " and message_direction = %s"
        params.append(direction)
    sql += " group by delivery_state, carrier_code, country_iso"
    return redshift.query(sql, tuple(params))


def get_message_type_breakdown(account_id, start, end, *, direction: str | None = "outbound") -> list[dict]:
    """EXACT per-message_type counts for the window (all channels). Used ONLY when
    the SMS histogram is empty, to tell a truly no_data window apart from one whose
    traffic is WhatsApp/MMS (out of scope). account_id always filtered."""
    acct = int(account_id)
    sql = (f"select message_type, count(*) as n from {_MDR} "
           f"where account_id = %s and message_time between %s and %s")
    params: list = [acct, start, end]
    if direction is not None:
        sql += " and message_direction = %s"
        params.append(direction)
    sql += " group by message_type"
    return redshift.query(sql, tuple(params))


# channel -> (table, timestamp column, summary columns, supports call_state)
_CHANNELS = {
    "voice": (
        _VOICE, "start_time",
        "call_uuid, start_time, call_direction, call_state, disconnect_reason, "
        "to_number, from_number, duration, bill_duration, mos_avg",
        True,
    ),
    "sip_trunk": (
        _ZENTRUNK, "initiation_time",
        "call_uuid, initiation_time, call_direction, disconnect_reason, disconnect_code, "
        "to_number, from_number, duration, bill_duration, route_type",
        False,
    ),
}


def get_calls_for_account(account_id, start, end, *, channel, direction=None,
                          call_state=None, disconnect_reason=None, limit: int = 100) -> list[dict]:
    """List calls for `account_id` in [start, end] on `channel` (voice|sip_trunk).

    account_id is ALWAYS filtered in-function. Optional direction / call_state /
    disconnect_reason narrow the result; all values are bound as parameters.
    """
    if channel not in _CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {list(_CHANNELS)}")
    table, ts_col, cols, supports_state = _CHANNELS[channel]
    if call_state is not None and not supports_state:
        raise ValueError(f"call_state filter is not supported for channel {channel!r}")

    acct = int(account_id)
    sql = f"select {cols} from {table} where account_id = %s and {ts_col} between %s and %s"
    params: list = [acct, start, end]
    if direction is not None:
        sql += " and call_direction = %s"
        params.append(direction)
    if call_state is not None:
        sql += " and call_state = %s"
        params.append(call_state)
    if disconnect_reason is not None:
        sql += " and disconnect_reason = %s"
        params.append(disconnect_reason)
    sql += f" order by {ts_col} desc limit %s"
    params.append(int(limit))
    return redshift.query(sql, tuple(params))


def get_quality_metrics(uuid: str, account_id) -> dict:
    """Call-quality metrics for `uuid`, scoped to `account_id`.

    Voice-API calls carry structured columns (mos_avg, jitter, fraction_lost,
    rtt, suspected_issues). SIP trunk calls carry quality inside extra_data
    (call_stats.media_stats). Tries voice CDR first, then sip_trunk.
    """
    row = _scoped_lookup(_VOICE, _VOICE_QUALITY_COLS, uuid, account_id)
    if row is not None:
        out = {"found": True, "channel": "voice", "source_table": _VOICE, **dict(row)}
        out["suspected_issues"] = (row.get("self_leg_suspected_issues")
                                   or row.get("all_legs_suspected_issues") or "")
        return out

    # Not a voice-API call — try sip_trunk (quality lives in extra_data JSON).
    # Both lookups are account-scoped; no unscoped query is ever issued.
    zrow = _scoped_lookup(_ZENTRUNK, "account_id, call_uuid, extra_data", uuid, account_id)
    if zrow is not None:
        media = ((_json(zrow.get("extra_data")) or {}).get("call_stats") or {}).get("media_stats") or {}
        return {
            "found": True, "channel": "sip_trunk", "source_table": _ZENTRUNK,
            "account_id": int(account_id), "call_uuid": uuid,
            "mos_average": media.get("mos_average"),
            "jitter": media.get("mos_average_jitter"),
            "packet_loss": media.get("mos_average_packetloss"),
            "round_trip": media.get("mos_average_roundtrip"),
            "per_leg": {"A": media.get("callstats_A"), "B": media.get("callstats_B")},
            "raw_media_stats": media,
        }
    return _not_found(uuid, account_id)
