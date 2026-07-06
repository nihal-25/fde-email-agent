"""Re-runnable SMS pricing import from SMS Master Pricing.xlsx.

Reads ONLY the "Final Summary Sheet" tab. Structure (verified against the file):
  row 0: blank | row 1: "New Prices Per Unit" | row 2: column headers |
  row 3+: data. Columns (0-indexed):
    1 Country | 2 Country ISO | 3 Operator Name |
    4 Outbound-MT (Country-level blended) | 5 Outbound-MT (per-operator) | 6 Inbound-MO

We keep ONLY rows where col 4 (country-level blended) is non-blank — that value
appears once per country (the first operator row); per-operator rows have col 4
blank. We load Country, ISO, and the blended rate as an EXACT string. Per-operator
/ Inbound columns and the whole of Sheet1 (market share / old rates / % change)
are never read, so internal columns cannot leak. Reload is atomic.

CLI:
    python -m app.pricing_import sms "SMS Master Pricing.xlsx"   # parse + load
    python -m app.pricing_import show                            # print loaded table
"""

from __future__ import annotations

import sys
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from app import db

SHEET = "Final Summary Sheet"
COL_COUNTRY, COL_ISO, COL_BLENDED = 1, 2, 4  # 0-indexed
DATA_FIRST_ROW = 4  # openpyxl 1-indexed; sheet 0-idx row 3


# All blended cells use Excel number_format '$0.0000' (4 dp), so the rate the
# sheet PRESENTS is the 4-dp value. We quote that (matches what a human sees and
# cleans float-formula noise like 0.051498999999999996 -> 0.0515). ROUND_HALF_UP
# mirrors Excel's display rounding.
_RATE_QUANTUM = Decimal("0.0001")


def _rate_str(value) -> str | None:
    """Normalize a blended-rate cell to the sheet's displayed 4-dp string, or
    None if not a valid positive number.

    To match Excel's $0.0000 display EXACTLY we round the actual stored DOUBLE
    half-away-from-zero — Decimal(float) gives the exact binary value, not its
    short repr. This matters only at knife-edge boundaries: e.g. Kiribati's
    0.16005 is stored as 0.16004999…, which Excel shows as 0.1600 (not 0.1601);
    Decimal(str(0.16005)) would wrongly round the literal up."""
    if value is None or value == "":
        return None
    try:
        d = Decimal(value) if isinstance(value, float) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if d <= 0:
        return None
    return format(d.quantize(_RATE_QUANTUM, rounding=ROUND_HALF_UP), "f")  # e.g. '0.0515'


def parse_sms_sheet(path: str) -> tuple[list[dict], list[str]]:
    """Return (rows, rejects). rows: [{iso, country_name, rate_usd}]. rejects:
    human-readable notes for rows dropped during validation."""
    import openpyxl

    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[SHEET]
    rows, rejects = [], []
    seen_iso: dict[str, str] = {}
    for i, r in enumerate(ws.iter_rows(min_row=DATA_FIRST_ROW, values_only=True), start=DATA_FIRST_ROW):
        country = r[COL_COUNTRY] if len(r) > COL_COUNTRY else None
        iso = r[COL_ISO] if len(r) > COL_ISO else None
        blended = r[COL_BLENDED] if len(r) > COL_BLENDED else None
        if blended in (None, ""):
            continue  # per-operator row (no country-level blended) -> skip silently
        country = (str(country).strip() if country else "")
        iso = (str(iso).strip() if iso else "")
        rate = _rate_str(blended)
        if not country or not iso:
            rejects.append(f"row {i}: missing country/iso (country={country!r} iso={iso!r})")
            continue
        if rate is None:
            rejects.append(f"row {i}: bad blended rate {blended!r} for {country}")
            continue
        if iso in seen_iso:
            rejects.append(f"row {i}: duplicate ISO {iso} ({country}); already had {seen_iso[iso]}")
            continue
        seen_iso[iso] = country
        rows.append({"iso": iso, "country_name": country, "rate_usd": rate})
    return rows, rejects


def import_sms(path: str) -> dict:
    db.init_db()
    rows, rejects = parse_sms_sheet(path)
    if not rows:
        raise RuntimeError(f"no valid rows parsed from {path!r} — refusing to wipe table")
    n = db.replace_sms_pricing(rows)
    return {"loaded": n, "rejects": rejects}


def _show():
    rows = db.all_sms_pricing()
    us = [r for r in rows if r["iso"].startswith("US")]
    india = [r for r in rows if r["iso"].startswith("IN") or "india" in r["country_name"].lower()]
    sample_isos = ["AD", "AE", "AU", "GB", "SA", "NG", "NE", "DM", "DO", "CG", "CD"]
    print(f"=== sms_pricing: {len(rows)} countries loaded ===")
    print("\n-- sample (incl. near-miss names) --")
    for iso in sample_isos:
        m = next((r for r in rows if r["iso"] == iso), None)
        if m:
            print(f"  {m['iso']:8s} {m['country_name'][:34]:34s} {m['rate_usd']} USD")
    print("\n-- US route rows (route-type, all quoted) --")
    for r in us:
        print(f"  {r['iso']:8s} {r['country_name'][:34]:34s} {r['rate_usd']} USD")
    print("\n-- India (present in sheet, SUPPRESSED by code override — never quoted) --")
    for r in india:
        print(f"  {r['iso']:8s} {r['country_name'][:34]:34s} {r['rate_usd']} USD   <-- OVERRIDDEN")


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    if cmd == "sms":
        if len(argv) < 2:
            print('usage: python -m app.pricing_import sms "<path.xlsx>"'); return 2
        res = import_sms(argv[1])
        print(f"loaded {res['loaded']} rows; {len(res['rejects'])} rejected")
        for x in res["rejects"][:20]:
            print("  reject:", x)
        print()
        _show()
    elif cmd == "show":
        _show()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
