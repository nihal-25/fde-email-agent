"""Reply drafting + honesty review (draft-time guardrail).

The model itself is instructed (in app/llm.py) never to assert a specific fact
it wasn't given. This module is the deterministic backstop: after a draft is
produced, `flag_unverified_specifics()` scans it for concrete claims — numbers
with units, money/rates, percentages, timelines, dates, and URLs — that are
NOT inside a [bracketed placeholder]. Those are surfaced to the human reviewer
(CLI output, and the Slack approval message) so a plausible-but-unsupported
specific can't slip through unnoticed.

This is a review aid, not a censor: it does not block approval. It also can't
know whether a specific IS supported by the thread — it simply highlights every
concrete claim so the human can check it. The model call stays in app/llm.py;
nothing here imports a model SDK.
"""

from __future__ import annotations

import re

from app import llm

# Spans like [timeframe] / [rate] are explicit placeholders — never flagged.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

# Each pattern captures a *specific* claim a reader would take as fact.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("link", re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)),
    ("money/rate", re.compile(r"[₹$€£]\s?\d[\d,]*(?:\.\d+)?")),
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?\s?%")),
    (
        "duration",
        re.compile(
            r"\b\d+(?:\s?[-–to]+\s?\d+)?\s*(?:business\s+)?"
            r"(?:hours?|hrs?|days?|weeks?|months?|years?|minutes?|mins?|seconds?|secs?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "timeline-phrase",
        re.compile(
            r"\b(?:a few (?:hours|days|minutes|weeks)"
            r"|a couple of (?:hours|days|weeks)"
            r"|within (?:a|an|the) (?:hour|day|week|month|next \w+)"
            r"|same[- ]day|next[- ]day|by (?:tomorrow|tonight|end of day|eod)"
            r"|in a (?:day|week|moment)|shortly)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "date",
        re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
            r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b",
            re.IGNORECASE,
        ),
    ),
]


def _bracket_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _BRACKET_RE.finditer(text)]


def _in_brackets(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _context(text: str, start: int, end: int, pad: int = 25) -> str:
    snippet = text[max(0, start - pad): min(len(text), end + pad)].replace("\n", " ")
    return snippet.strip()


def flag_unverified_specifics(draft_text: str) -> list[dict]:
    """Return a list of concrete claims that aren't bracketed placeholders.

    Each item: {"type": str, "text": matched value, "context": surrounding text}.
    De-duplicated by (type, matched text).
    """
    if not draft_text:
        return []
    spans = _bracket_spans(draft_text)
    findings: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(draft_text):
            if _in_brackets(m.start(), spans):
                continue
            value = m.group(0).strip()
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            findings.append({
                "type": kind,
                "text": value,
                "context": _context(draft_text, m.start(), m.end()),
            })
    return findings


def format_flags(findings: list[dict]) -> str:
    """Human-readable summary for CLI/logs."""
    if not findings:
        return "✅ No unverified specifics detected."
    lines = ["⚠️ Unverified specifics to review (verify each is real, or bracket/omit it):"]
    for f in findings:
        lines.append(f"  - [{f['type']}] \"{f['text']}\"  …{f['context']}…")
    return "\n".join(lines)


def generate(thread_text: str, intent: str) -> dict:
    """Draft a reply and run the honesty review in one call.

    Returns {"draft": str, "flags": list[dict]}.
    """
    draft_text = llm.draft_reply(thread_text, intent)
    return {"draft": draft_text, "flags": flag_unverified_specifics(draft_text)}
