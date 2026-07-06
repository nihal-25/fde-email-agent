"""Re-runnable ingest of voice/WhatsApp/numbers pricing from plivo.com SSG pages.

Access path (decided): the per-country server-rendered pages carry the FINAL
figures + currency (₹ for India, $ for rest). We read those verbatim and never
run the site's convert/Exchange logic — quoting a computed money number is
exactly what this case avoids.

  voice    -> /voice/pricing/{cc}/
  whatsapp -> /whatsapp/pricing/{cc}/
  numbers  -> /virtual-phone-numbers/pricing/{cc}/
  (RCS has no SSG pages -> not ingested; the handler holds RCS.)

Country set is discovered from sitemap.xml (authoritative coverage: ~225 voice,
~140 whatsapp, ~225 numbers). Each page's PRICE tables are parsed (tables whose
header has a price column; definition/compliance tables skipped). Stored atomically
per channel. A change-diff vs the prior run surfaces moved rates for review.

CLI:
    python -m app.web_pricing_import all                 # full ingest (all channels)
    python -m app.web_pricing_import <channel>           # one channel
    python -m app.web_pricing_import sample IN US GB      # parse just these (eyeball), no store
"""

from __future__ import annotations

import re
import sys

from bs4 import BeautifulSoup

from app import db
from app.ingest import crawl

SITEMAP = "https://www.plivo.com/sitemap.xml"
CHANNELS = {
    "voice": "/voice/pricing/",
    "whatsapp": "/whatsapp/pricing/",
    "numbers": "/virtual-phone-numbers/pricing/",
}
_PRICE_HDR_RE = re.compile(r"price|outbound|inbound", re.IGNORECASE)
# Trailing UI/CTA text the SSG renders inside price cells — stripped so it never
# leaks into a quote. "Starts at" and "Not Supported" qualifiers are KEPT.
_CELL_CTA_RE = re.compile(r"\s*(View detailed.*|Learn more.*|See .*pricing.*)$", re.IGNORECASE)


def _clean_cell(val: str) -> str:
    return _CELL_CTA_RE.sub("", val).strip()
_RATE_CELL_RE = re.compile(r"[₹$]|not supported|free|/min|/month|/message|per message", re.IGNORECASE)
_CCY = {"₹": "INR", "$": "USD"}


def discover(fetcher: crawl.PoliteFetcher, channel: str) -> dict[str, str]:
    """ISO(lower) -> page URL for a channel, from the sitemap."""
    body = fetcher.get(SITEMAP)
    stem = CHANNELS[channel]
    urls = re.findall(r"<loc>([^<]*" + re.escape(stem) + r"[a-z]{2}/)</loc>", body)
    out = {}
    for u in urls:
        m = re.search(re.escape(stem) + r"([a-z]{2})/", u)
        if m:
            out[m.group(1)] = u
    return out


def parse_price_tables(html: str) -> tuple[list[dict], str | None, str | None]:
    """Return (tables, currency, country_name). Only tables with a price column
    are kept; cells are the verbatim rendered strings. Currency is read from the
    rate cells (₹->INR, $->USD)."""
    soup = BeautifulSoup(html, "html.parser")
    country = None
    h1 = soup.find("h1")
    if h1:
        country = h1.get_text(" ", strip=True)
    tables, ccy_syms = [], set()
    for tb in soup.find_all("table"):
        rows = tb.find_all("tr")
        if not rows:
            continue
        header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        if not header or not any(_PRICE_HDR_RE.search(h) for h in header):
            continue  # skip definition/compliance tables (no price column)
        parsed_rows = []
        for r in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])]
            if len(cells) < 2 or not cells[0]:
                continue
            label, prices = cells[0], {}
            for i, val in enumerate(cells[1:], start=1):
                col = header[i] if i < len(header) else f"col{i}"
                if _RATE_CELL_RE.search(val):
                    val = _clean_cell(val)
                    prices[col] = val
                    for sym in _CCY:
                        if sym in val:
                            ccy_syms.add(sym)
            if prices:
                parsed_rows.append({"label": label, "prices": prices})
        if parsed_rows:
            tables.append({"section": " | ".join(header), "rows": parsed_rows})
    currency = None
    if ccy_syms:
        currency = "INR" if "₹" in ccy_syms else _CCY.get(next(iter(ccy_syms)))
    return tables, currency, country


def _country_name(iso: str) -> str:
    try:
        import pycountry
        return pycountry.countries.lookup(iso).name
    except LookupError:
        return iso.upper()


def parse_country(fetcher, channel, iso, url) -> dict | None:
    html = fetcher.get(url)
    tables, currency, _h1 = parse_price_tables(html)  # H1 is the channel title, not the country
    if not tables or not currency:
        return None
    return {"iso": iso.upper(), "country_name": _country_name(iso),
            "currency": currency, "tables": tables, "source_url": url}


def _diff(channel: str, new_rows: list[dict]) -> list[str]:
    """Human-readable rate changes vs what's currently stored (for review)."""
    old = {r["iso"]: r for r in db.all_web_pricing(channel)}
    changes = []
    for r in new_rows:
        o = old.get(r["iso"])
        if o is None:
            changes.append(f"+ NEW {channel}/{r['iso']}")
        elif o["tables"] != r["tables"] or o["currency"] != r["currency"]:
            changes.append(f"~ CHANGED {channel}/{r['iso']} ({r['country_name']})")
    for iso in set(old) - {r["iso"] for r in new_rows}:
        changes.append(f"- REMOVED {channel}/{iso}")
    return changes


def ingest_channel(channel: str, fetcher=None, only: list[str] | None = None) -> dict:
    db.init_db()
    fetcher = fetcher or crawl.PoliteFetcher(min_interval=1.0)
    urls = discover(fetcher, channel)
    if only:
        urls = {cc: u for cc, u in urls.items() if cc.upper() in {x.upper() for x in only}}
    rows, errors = [], []
    for cc, u in urls.items():
        try:
            row = parse_country(fetcher, channel, cc, u)
            if row:
                rows.append(row)
            else:
                errors.append(f"{cc}: no price tables/currency parsed")
        except Exception as e:
            errors.append(f"{cc}: {type(e).__name__}: {str(e)[:60]}")
    changes = _diff(channel, rows)
    db.replace_web_pricing(channel, rows)
    return {"channel": channel, "discovered": len(urls), "loaded": len(rows),
            "errors": errors, "changes": changes}


# --- CLI ---------------------------------------------------------------------

def _print_parsed(channel, row):
    print(f"\n--- {channel.upper()} {row['iso']} ({row['country_name']}) currency={row['currency']} ---")
    for t in row["tables"]:
        print(f"  [{t['section']}]")
        for r in t["rows"]:
            print(f"     {r['label']:38s} {r['prices']}")


def main(argv):
    cmd = argv[0] if argv else "help"
    if cmd == "sample":
        only = argv[1:] or ["IN", "US", "GB"]
        f = crawl.PoliteFetcher(min_interval=1.0)
        for channel in CHANNELS:
            urls = discover(f, channel)
            for cc in only:
                u = urls.get(cc.lower())
                if not u:
                    print(f"\n--- {channel.upper()} {cc}: NO PAGE (would HOLD) ---"); continue
                row = parse_country(f, channel, cc.lower(), u)
                if row:
                    _print_parsed(channel, row)
                else:
                    print(f"\n--- {channel.upper()} {cc}: parsed nothing ---")
    elif cmd in CHANNELS or cmd == "all":
        chans = list(CHANNELS) if cmd == "all" else [cmd]
        for ch in chans:
            r = ingest_channel(ch)
            print(f"\n=== {ch}: discovered {r['discovered']} / loaded {r['loaded']} / errors {len(r['errors'])} ===")
            print(f"changes vs prior run: {len(r['changes'])}")
            for c in r["changes"][:30]:
                print("  ", c)
            for e in r["errors"][:10]:
                print("  err:", e)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
