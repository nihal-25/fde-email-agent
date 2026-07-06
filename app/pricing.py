"""SMS pricing handler — country resolution + exact lookup (Phase 2 pricing).

Flow: the LLM extracts a country NAME only (never a rate). This module
DETERMINISTICALLY resolves name -> ISO -> sheet key, then looks up the exact
stored rate. Hard rules:

- NO fuzzy matching anywhere. Fuzzy is exactly what silently picks the wrong
  country for money ("Niger" -> "Nigeria"). We use exact alias/ISO/name matches
  only; anything that doesn't resolve to a single country is ambiguous/unresolved.
- Near-miss disambiguation: a genuinely ambiguous term (Congo, Korea, ...) NEVER
  picks one — it returns `ask_disambiguate` with the candidates. Distinct
  confusable names (Niger/Nigeria, Dominica/Dominican Republic) resolve EXACTLY.
- India override keys off the NORMALIZED country (IN), checked before any sheet
  lookup, so "India"/"+91"/"Indian numbers" all hit it.
- US is route-type: returns the three route rows, never one blended figure.
- Country valid but not in the sheet -> HOLD, never an approximate/near-by rate.

This file is the resolution + lookup layer only. Draft templating + worker
routing are wired after review.
"""

from __future__ import annotations

import re

from app import db, llm

# --- Curated alias table: normalized phrase -> ISO alpha-2 ------------------
# Customer phrasings pycountry doesn't match. Exact (post-normalization) only.
_ALIASES = {
    "uae": "AE", "u a e": "AE", "emirates": "AE", "the emirates": "AE",
    "usa": "US", "us": "US", "u s": "US", "u s a": "US", "america": "US",
    "the states": "US", "states": "US",
    "uk": "GB", "u k": "GB", "britain": "GB", "great britain": "GB", "england": "GB",
    "south korea": "KR", "north korea": "KP",
    "dr congo": "CD", "drc": "CD",
    "russia": "RU", "holland": "NL", "the netherlands": "NL",
    # Exact-intent forms pycountry misses (from the ambiguity audit):
    "macau": "MO",                       # pycountry only knows "Macao"
    "macedonia": "MK", "north macedonia": "MK",
    "republic of ireland": "IE",         # explicit -> the Republic
    "northern ireland": "GB",            # part of the UK -> GB pricing
}

# --- Ambiguous terms: normalized term -> candidate (label, ISO) list --------
# ALWAYS ask; NEVER pick. Checked BEFORE pycountry (whose exact lookup would
# silently resolve bare "Congo" -> CG).
#
# RULE for adding entries (the test is intent, not technical ambiguity):
#   - Disambiguate (put here) only when the candidates are genuinely CO-EQUAL in
#     likely customer intent — a real customer typing the bare word could
#     plausibly mean either (Congo, Korea, Guinea, Sudan).
#   - Resolve-to-obvious (leave to alias/pycountry) when there's a dominant
#     default AND the minority meaning has its own explicit name the customer
#     would use instead (Ireland->IE since "Northern Ireland" is said explicitly;
#     China->CN since "Hong Kong"/"Taiwan" are said explicitly).
_AMBIGUOUS = {
    "congo": [("Republic of the Congo", "CG"),
              ("Democratic Republic of the Congo", "CD")],
    "korea": [("South Korea", "KR"), ("North Korea", "KP")],
    "guinea": [("Guinea", "GN"), ("Equatorial Guinea", "GQ"),
               ("Guinea-Bissau", "GW"), ("Papua New Guinea", "PG")],
    "sudan": [("Sudan", "SD"), ("South Sudan", "SS")],
    "samoa": [("Samoa", "WS"), ("American Samoa", "AS")],
    "virgin islands": [("U.S. Virgin Islands", "VI"), ("British Virgin Islands", "VG")],
}

# --- Demonyms / short forms -> ISO (pycountry doesn't map these). Lets the
#     resolver itself handle "Indian numbers" -> India even if the LLM passes the
#     adjective through. Exact, not fuzzy.
_DEMONYMS = {
    "indian": "IN", "american": "US", "british": "GB", "australian": "AU",
    "emirati": "AE", "saudi": "SA", "filipino": "PH", "german": "DE",
    "french": "FR", "spanish": "ES", "brazilian": "BR", "canadian": "CA",
}
# Filler tokens stripped when a phrase doesn't resolve as-is ("send to india numbers").
_FILLER = {"send", "sending", "to", "a", "an", "the", "my", "our", "your", "for",
           "sms", "text", "texts", "message", "messages", "msg", "number", "numbers",
           "pricing", "price", "prices", "rate", "rates", "cost", "costs", "in", "on"}

# --- Calling-code backstop (LLM usually extracts the country name; this covers
#     a bare "+NN"). +1 / +7 are shared (NANP / RU+KZ) -> ambiguous, never pick.
_CALLING = {"91": "IN", "971": "AE", "44": "GB", "61": "AU", "65": "SG",
            "971": "AE", "92": "PK", "880": "BD", "234": "NG", "227": "NE"}
_CALLING_AMBIGUOUS = {"1": [("United States", "US"), ("Canada", "CA")],
                      "7": [("Russia", "RU"), ("Kazakhstan", "KZ")]}


def _norm(s: str | None) -> str:
    """Lowercase; strip punctuation/digits; collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", (s or "").lower())).strip()


def resolve_country(raw_name: str | None) -> dict:
    """Resolve an extracted country name/phrase to an ISO, or flag ambiguity.

    Returns:
      {"status": "resolved", "iso": "NE"}                         exactly one country
      {"status": "ambiguous", "candidates": [(label, iso), ...]}  must ask
      {"status": "unresolved"}                                    no confident match
    """
    raw = (raw_name or "").strip()
    # Phone prefix? ("+91", "0091", "91 ...")
    m = re.match(r"^\+?0*([0-9]{1,4})\b", raw)
    if m and (raw.startswith("+") or raw[:2] == "00"):
        code = m.group(1)
        if code in _CALLING_AMBIGUOUS:
            return {"status": "ambiguous", "candidates": _CALLING_AMBIGUOUS[code]}
        if code in _CALLING:
            return {"status": "resolved", "iso": _CALLING[code]}

    s = _norm(raw)
    if not s:
        return {"status": "unresolved"}
    hit = _resolve_norm(s, raw)
    if hit:
        return hit
    # Retry on a filler-stripped phrase ("send to india numbers" -> "india").
    stripped = " ".join(t for t in s.split() if t not in _FILLER)
    if stripped and stripped != s:
        hit = _resolve_norm(stripped, stripped)
        if hit:
            return hit
    return {"status": "unresolved"}


def _resolve_norm(s: str, raw_for_pycountry: str) -> dict | None:
    """Try ambiguous / alias / demonym / pycountry-EXACT on one normalized string.
    Returns a verdict dict or None (no match). Ambiguity is checked first so a
    bare 'congo' can never be silently resolved by pycountry."""
    if s in _AMBIGUOUS:
        return {"status": "ambiguous", "candidates": _AMBIGUOUS[s]}
    if s in _ALIASES:
        return {"status": "resolved", "iso": _ALIASES[s]}
    if s in _DEMONYMS:
        return {"status": "resolved", "iso": _DEMONYMS[s]}
    import pycountry
    try:
        return {"status": "resolved", "iso": pycountry.countries.lookup(raw_for_pycountry).alpha_2}
    except LookupError:
        return None


def lookup_sms_price(raw_name: str | None) -> dict:
    """Resolve country -> exact SMS pricing result. Pure data/logic; no LLM, no
    rate ever generated. Statuses the draft layer will template:
      quote | us_routes | india_unavailable | ask_disambiguate | hold_not_in_sheet | ask_country
    """
    r = resolve_country(raw_name)
    if r["status"] == "ambiguous":
        return {"status": "ask_disambiguate", "candidates": r["candidates"]}
    if r["status"] == "unresolved":
        return {"status": "ask_country"}

    iso = r["iso"]
    if iso == "IN":                      # India override — normalized-country keyed
        return {"status": "india_unavailable"}
    if iso == "US":                      # route-type, never one blended figure
        routes = [row for row in db.all_sms_pricing() if row["iso"].startswith("US")]
        return {"status": "us_routes", "routes": routes}

    row = db.get_sms_pricing_by_iso(iso)
    if row:
        return {"status": "quote", "iso": iso, "country": row["country_name"],
                "rate": row["rate_usd"], "currency": "USD"}
    return {"status": "hold_not_in_sheet", "iso": iso}   # valid country, not priced -> HOLD


# --- Draft (fully code-templated; LLM never writes the rate sentence) --------

_SIGNOFF = "Best regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo"
_RATE_NUM_RE = re.compile(r"\d*\.\d+")   # rate-shaped (decimal) numbers only


def _iso_display_name(iso: str) -> str:
    try:
        import pycountry
        return pycountry.countries.lookup(iso).name
    except LookupError:
        return iso


def _allowed_rates(result: dict) -> set[str]:
    """The exact rate strings this result is permitted to contain (for the
    numeric post-check). Non-quote statuses permit NO rate."""
    if result["status"] == "quote":
        return {result["rate"]}
    if result["status"] == "us_routes":
        return {r["rate_usd"] for r in result["routes"]}
    return set()


def _sms_body(result: dict) -> str:
    """The SMS body for one result (no greeting/sign-off). Reused by the single
    draft and the multi-quote per-line block."""
    st = result["status"]
    if st == "quote":
        return (f"Our outbound SMS rate to {result['country']} is "
                f"${result['rate']} {result['currency']} per message.")
    if st == "us_routes":
        by_iso = {r["iso"]: r["rate_usd"] for r in result["routes"]}
        return ("For the United States, outbound SMS pricing is by route type:\n"
                f"- 10DLC / Long code: ${by_iso.get('US-10DLC')} USD per message\n"
                f"- Short code: ${by_iso.get('US-SC')} USD per message\n"
                f"- Toll-free: ${by_iso.get('US-TF')} USD per message")
    if st == "india_unavailable":
        return "We don't currently offer SMS to India."
    if st == "ask_disambiguate":
        opts = " or ".join(label for label, _ in result["candidates"])
        return (f"Could you confirm which destination you mean — {opts}? "
                f"SMS rates differ by country, so I want to quote the right one.")
    if st == "hold_not_in_sheet":
        return (f"Let me confirm our current outbound SMS rate to "
                f"{_iso_display_name(result['iso'])} and get back to you shortly.")
    if st == "ask_channel":
        return ("Happy to help with pricing — which service (SMS, voice, WhatsApp, "
                "phone numbers) and which destination country are you asking about?")
    return ("Happy to help with SMS pricing — which destination country are "
            "you asking about? Rates are per-country.")


def draft_sms_pricing(result: dict, customer_name: str | None) -> str:
    """Full single SMS reply. Every rate is a code-supplied looked-up figure."""
    hi = f"Hi {customer_name}," if customer_name else "Hi,"
    return f"{hi}\n\n{_sms_body(result)}\n\n{_SIGNOFF}"


def _numeric_post_check(draft: str, allowed: set[str]) -> bool:
    """Defense-in-depth: every rate-shaped (decimal) number in the templated
    draft must be a looked-up figure. Should always pass (code built it); guards
    against a template regression ever shipping an unintended number."""
    return set(_RATE_NUM_RE.findall(draft)) <= {a for a in allowed if "." in a}


_WEB_CHANNELS = {"voice", "whatsapp", "numbers"}   # have SSG pages (web_pricing)
_CHANNEL_LABEL = {"voice": "voice calling", "whatsapp": "WhatsApp messaging",
                  "numbers": "phone number"}
# A rate token left OUTSIDE an allowed full string = a leaked/flattened figure.
_WEB_RATE_LEFT = re.compile(r"[₹$]\s?\d|\d[\d.,]*\s*/\s*(min|month|message|lookup|sec)", re.IGNORECASE)


def _primary_table(tables: list[dict], channel: str) -> dict | None:
    """The core price table to quote for a channel (scoped, not a dump):
    voice -> the Outbound/Inbound route table; whatsapp -> per-message categories;
    numbers -> the rental table. Falls back to the first price table."""
    want = {"voice": ("outbound", "inbound"), "whatsapp": ("per message",),
            "numbers": ("price",)}.get(channel, ())
    for t in tables:
        sec = t["section"].lower()
        if any(w in sec for w in want):
            return t
    return tables[0] if tables else None


def lookup_web_price(channel: str, raw_name: str | None) -> dict:
    """Resolve country -> exact web_pricing for a channel. NO India override here
    (India HAS voice/WhatsApp/numbers pricing, in INR). Statuses: web_quote /
    ask_disambiguate / ask_country / hold_not_covered."""
    r = resolve_country(raw_name)
    if r["status"] == "ambiguous":
        return {"status": "ask_disambiguate", "candidates": r["candidates"]}
    if r["status"] == "unresolved":
        return {"status": "ask_country"}
    row = db.get_web_pricing(channel, r["iso"])
    if not row:
        return {"status": "hold_not_covered", "channel": channel, "iso": r["iso"]}
    table = _primary_table(row["tables"], channel)
    if not table:
        return {"status": "hold_not_covered", "channel": channel, "iso": r["iso"]}
    return {"status": "web_quote", "channel": channel,
            "country": _iso_display_name(r["iso"]),   # from ISO, not the page H1
            "currency": row["currency"], "table": table,
            "as_of": (row.get("imported_at") or "")[:10]}


def _web_allowed_strings(table: dict) -> set[str]:
    """Every full rendered price string in the quoted table — the atomic units
    the post-check validates (qualifier inseparable from number)."""
    out = set()
    for row in table["rows"]:
        out.update(row["prices"].values())
    return out


def _web_post_check(draft: str, allowed: set[str]) -> bool:
    """Faithful-quote guard: remove each allowed FULL string once; if any rate
    token remains, a number leaked outside its qualifier (e.g. 'Starts at
    $0.0075/min' got flattened to '$0.0075/min') -> fail."""
    rem = draft
    for s in sorted(allowed, key=len, reverse=True):
        rem = rem.replace(s, " ")        # all occurrences (a price may repeat, e.g. local in==out)
    return not _WEB_RATE_LEFT.search(rem)


def _web_body(result: dict) -> str:
    """The web (voice/whatsapp/numbers) body for one result (no greeting/sign-off).
    Each price is inserted as the FULL rendered string (qualifier + number
    together); never splits out the number."""
    st = result["status"]
    if st == "web_quote":
        label = _CHANNEL_LABEL.get(result["channel"], result["channel"])
        lines = []
        for row in result["table"]["rows"]:
            prices = row["prices"]
            if len(prices) == 1:
                lines.append(f"- {row['label']}: {next(iter(prices.values()))}")
            else:
                parts = ", ".join(f"{col.lower()}: {val}" for col, val in prices.items())
                lines.append(f"- {row['label']} — {parts}")
        return (f"Here's our {label} pricing for {result['country']} "
                f"({result['currency']}, current as of {result['as_of']}):\n"
                + "\n".join(lines))
    if st == "hold_not_covered":
        return (f"Let me confirm our {_CHANNEL_LABEL.get(result.get('channel'),'')} "
                f"pricing for that destination and get back to you shortly.")
    if st == "channel_unsupported":
        return (f"Let me confirm our {result.get('channel','').upper()} pricing and "
                f"get back to you shortly.")
    if st == "ask_disambiguate":
        opts = " or ".join(label for label, _ in result["candidates"])
        return (f"Could you confirm which destination you mean — {opts}? "
                f"Pricing differs by country.")
    return ("Happy to help with pricing — which destination country are you "
            "asking about? Rates are per-country.")


def draft_web_pricing(result: dict, customer_name: str | None) -> str:
    """Full single web reply."""
    hi = f"Hi {customer_name}," if customer_name else "Hi,"
    return f"{hi}\n\n{_web_body(result)}\n\n{_SIGNOFF}"


_NON_SMS_CHANNELS = {"voice", "whatsapp", "rcs", "numbers"}
MAX_PAIRS = 6   # fan-out cap: beyond this, point to the page + ask to narrow (no wall)

_FLAG_MAP = {
    "quote": None, "us_routes": None, "web_quote": None,
    "india_unavailable": {"type": "pricing_india", "text": "India SMS marked unavailable (code rule)."},
    "ask_disambiguate": {"type": "pricing_disambiguate", "text": "Ambiguous destination — asked to clarify; verify before sending."},
    "hold_not_in_sheet": {"type": "pricing_hold", "text": "Destination not in the SMS price sheet — holding; needs a manual rate."},
    "hold_not_covered": {"type": "pricing_hold", "text": "No page for that destination/channel — holding; needs a manual rate."},
    "channel_unsupported": {"type": "pricing_channel_unsupported", "text": "RCS pricing not available on site — holding."},
    "ask_country": {"type": "pricing_ask", "text": "No destination given — asked which country."},
    "ask_channel": {"type": "pricing_ask", "text": "Channel unclear — asked which service + country."},
    "post_check_hold": {"type": "pricing_postcheck_hold", "text": "Post-check tripped (rate/qualifier mismatch) — held; needs manual quote."},
}


def _lookup_and_body(channel: str, country: str | None) -> tuple[dict, str]:
    """Run the EXISTING exact lookup for ONE (channel, country) and return
    (result, post-checked body). Identical lookup path as the single quote —
    only the assembly differs. India override applies per-(channel,country)
    (inside lookup_sms_price only). On post-check failure -> a safe hold body."""
    if channel == "sms":
        result = lookup_sms_price(country)
        body = _sms_body(result)
        ok = _numeric_post_check(body, _allowed_rates(result))
    elif channel in _WEB_CHANNELS:
        result = lookup_web_price(channel, country)
        body = _web_body(result)
        allowed = _web_allowed_strings(result["table"]) if result["status"] == "web_quote" else set()
        ok = _web_post_check(body, allowed)
    elif channel == "rcs":
        result = {"status": "channel_unsupported", "channel": "rcs"}
        body = _web_body(result)
        ok = True
    else:
        result, body, ok = {"status": "ask_channel"}, _sms_body({"status": "ask_channel"}), True
    if not ok:
        print(f"[pricing] post-check FAILED channel={channel} status={result.get('status')}; holding")
        result, body = {"status": "post_check_hold"}, "Let me confirm that pricing and get back to you shortly."
    return result, body


def handle(thread: dict, classification: dict) -> tuple[str, list]:
    """Route a pricing_question -> draft + flags. Each (channel, country) pair is
    resolved by the SAME exact lookup (SMS->sheet w/ India override; voice/
    WhatsApp/numbers->web_pricing; RCS->hold) and concatenated — every line is
    exact-or-hold, every number traces to a looked-up figure, each currency
    carried from its own source (mixed ₹/$ stay per-line labeled). NEVER asks the
    customer to pick when multiple are named; lists all (capped)."""
    from app.cli import render_thread

    name = (classification or {}).get("customer_name")
    hi = f"Hi {name}," if name else "Hi,"
    ext = llm.extract_pricing(render_thread(thread))
    channels = [c.lower() for c in (ext.get("channels") or []) if c]
    countries = [c for c in (ext.get("countries") or []) if c]

    if not countries:
        return f"{hi}\n\n{_sms_body({'status': 'ask_country'})}\n\n{_SIGNOFF}", [_FLAG_MAP['ask_country']]
    if not channels:
        return f"{hi}\n\n{_sms_body({'status': 'ask_channel'})}\n\n{_SIGNOFF}", [_FLAG_MAP['ask_channel']]

    pairs, seen = [], set()
    for ch in channels:
        for co in countries:
            k = (ch, co.strip().lower())
            if k not in seen:
                seen.add(k)
                pairs.append((ch, co))

    if len(pairs) > MAX_PAIRS:                       # don't generate a wall
        body = (f"That's {len(pairs)} service/destination combinations — to keep the "
                f"quote accurate I'd rather not guess at a long list. Could you narrow "
                f"to a few, or see the full rate card at https://www.plivo.com/pricing/?")
        return f"{hi}\n\n{body}\n\n{_SIGNOFF}", [{"type": "pricing_too_many",
                "text": f"{len(pairs)} combinations asked — pointed to pricing page / narrow."}]

    if len(pairs) == 1:
        result, body = _lookup_and_body(*pairs[0])
        f = _FLAG_MAP.get(result["status"])
        return f"{hi}\n\n{body}\n\n{_SIGNOFF}", ([f] if f else [])

    # Multi: same per-pair lookup, concatenated. Per-line honest (quote/hold/ask),
    # mixed currencies stay labeled per line.
    bodies, statuses = [], []
    for ch, co in pairs:
        result, body = _lookup_and_body(ch, co)
        bodies.append(body)
        statuses.append(result["status"])
    draft = f"{hi}\n\n" + "\n\n".join(bodies) + f"\n\n{_SIGNOFF}"
    needs_review = [s for s in statuses if s not in ("quote", "us_routes", "web_quote")]
    flags = [{"type": "pricing_multi",
              "text": f"Multi-quote ({len(pairs)} lines; verify each — mixed currencies possible)."
                      + (f" {len(needs_review)} line(s) hold/ask/India — check those." if needs_review else "")}]
    return draft, flags
