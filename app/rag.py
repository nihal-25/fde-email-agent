"""Platform/product Q&A from Plivo docs — retrieval + confidence routing.

Hard rule (CLAUDE.md): platform answers come ONLY from retrieved Plivo content,
never the model's own knowledge, never invented. That rule is enforced by a
LAYERED stack in CODE — not by trusting any single prompt:

  1. score floor      : top1 cosine < PARTIAL_FLOOR -> holding reply (kills
                        off-topic cheaply, no LLM call).
  2. answerability gate: llm.rag_answerability — do the excerpts actually answer
                        THIS question? Conservative; uncertainty -> holding.
                        This is what catches adjacent-domain fabrication
                        (unsupported features, competitor questions) that score
                        alone cannot separate (calibration gap was -0.213).
  3. draft            : llm.rag_draft using ONLY retrieved chunks.
  4. groundedness gate: llm.rag_groundedness — every claim traceable to a chunk,
                        else downgrade to holding.
  5. honesty flagger  : draft.flag_unverified_specifics on whatever we keep.

Score only sets the confidence TIER (strong vs partial) and rejects off-topic;
it is NOT trusted to decide answerable. Gates 2 and 4 are LLM calls, so this is
a backstop stack, not a single guarantee — the layering is the enforcement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app import db, draft as draft_mod, llm

_URL_RE = re.compile(r"https?://[^\s)>\]}]+")
# A number token: digits with optional thousands-commas and optional decimal.
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Normalized number tokens in text, excluding any inside URLs (so doc
    article-ids / a draft's own links don't count). Commas stripped so
    1,600 == 1600; decimals kept."""
    text = _URL_RE.sub(" ", text or "")
    return {m.replace(",", "") for m in _NUM_RE.findall(text)}


def _numbers_grounded(draft_text: str, chunks: list[dict]) -> tuple[bool, set[str]]:
    """Deterministic numeric-grounding gate: every number in the draft must
    appear (normalized) in the retrieved chunk text. Numbers are the highest-risk
    factual specific and are verbatim-checkable, so we don't leave them to the
    probabilistic LLM gate. Returns (ok, missing_numbers)."""
    chunk_nums: set[str] = set()
    for c in chunks:
        chunk_nums |= _numbers(c.get("content", ""))
    missing = _numbers(draft_text) - chunk_nums
    return (not missing), missing

# Framing/connective vocabulary for the link-forward groundedness shortcut.
# A draft built ONLY from these words (+ the matched link's own topic words +
# a verbatim doc URL) makes no substantive factual claim, so it is grounded by
# construction. CAPABILITY VERBS are deliberately EXCLUDED (support/provide/
# send/enable/allow/configure/...): a real claim like "Plivo supports X" uses
# one, so it falls through to the LLM groundedness gate. Generic nouns and
# conversational glue are allowed (they can't assert a Plivo capability on their
# own). Any word outside the allowed set, or any digit (a spec is a claim), ->
# no shortcut, fall through. Keep this conservative.
_FRAMING_WORDS = frozenset("""
hi hello hey thanks thank you your a an the to for of on at in and or it is are
here heres here's that this these those guide guides doc docs documentation link
links page reaching out connect connecting connection with plivo below above
feel free ask asking if hit hitting any anything anytime question questions more
detail details be found see check steps step let me know happy glad help so as
well also walk through please refer referring covers explains how do i we i'm im
there getting started has have which what get stuck working agent agents app
apps account number numbers call calls message messages api sdk example examples
sample samples everything anything out up
""".split())


def _topic_tokens(chunks: list[dict]) -> set[str]:
    """Words from each matched chunk's title and URL slug — the topic of the
    grounding link, safe to echo in a link-forward reply (e.g. 'livekit')."""
    toks = set()
    for c in chunks:
        for field in (c.get("title") or "", (c.get("url") or "").rsplit("/", 2)[-1]):
            toks.update(re.findall(r"[a-zA-Z]{3,}", field.lower()))
    return toks


def _link_forward_grounded(draft_text: str, chunks: list[dict]) -> bool:
    """Deterministic groundedness for a pure link-forward reply: true iff the
    draft's only factual content is URL(s) that appear verbatim in the retrieved
    chunks, with the rest being framing / the link's own topic words / greeting /
    sign-off. Conservative: returns False (-> fall through to the LLM gate) on
    any doubt (unknown word, a digit/spec, or a URL not from the docs)."""
    urls = _URL_RE.findall(draft_text)
    if not urls:
        return False
    haystack = " ".join((c.get("url") or "") + " " + (c.get("content") or "") for c in chunks)
    for u in urls:
        if u.rstrip(".,);:'\"") not in haystack:  # any URL not from the docs -> no shortcut
            return False
    body = re.split(r"Best regards|Best,|Warm regards|Cheers", draft_text)[0]
    lines = body.splitlines()
    if lines and re.match(r"\s*(hi|hello|hey)\b", lines[0], re.I):
        lines = lines[1:]
    body = _URL_RE.sub(" ", " ".join(lines))
    if any(ch.isdigit() for ch in body):
        return False
    allowed = _FRAMING_WORDS | _topic_tokens(chunks)
    for w in re.findall(r"[a-zA-Z']+", body.lower()):
        if w not in allowed:
            return False
    return True

# Locked thresholds (see calibration checkpoint).
# PARTIAL_FLOOR lowered 0.55 -> 0.47: at 0.55 the floor was rejecting a CORRECT,
# rank-#1 grounded-limitation chunk (e.g. Vapi+India "not supported, TRAI" at
# ~0.51) before the validated answerability gate could decide. 0.47 sits +0.038
# above the off-topic (group E) ceiling of 0.432 and ~0.04 below the limitation
# band, so off-topic is still rejected cheaply while grounded-limitation answers
# reach the gate. This routes 3 adjacent-domain cases (Twilio-pricing 0.537,
# Amazon SNS 0.509, SendGrid 0.485) to the gate for the first time — the gate
# (explicit-only) must still HOLD them (re-validated two-directionally).
PARTIAL_FLOOR = 0.47   # below -> WEAK/holding (off-topic), no LLM call
STRONG_TOP1 = 0.62     # at/above (with corroboration) -> answer directly
STRONG_MEAN3 = 0.52    # corroboration: top-3 mean, so STRONG isn't one fluke chunk
RETRIEVE_K = 8
CONTEXT_CHUNKS = 6     # how many top chunks the gates/draft actually see

PATH_STRONG = "strong"
PATH_PARTIAL = "partial"
PATH_WEAK = "weak"


@dataclass
class RagResult:
    path: str                 # strong | partial | weak
    draft_text: str
    flags: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    reason: str = ""          # why this path (for logs/audit)
    top1: float = 0.0


_SIGNOFF = "Best regards,\nNihal Manjunath\nForward Deployed Engineer @ Plivo"


def _holding_reply(customer_name: str | None, intent: str) -> str:
    """A safe, fact-free holding reply (never fabricates to fill a gap). For
    technical_support it asks for the concrete debugging inputs (the Phase-1 ask
    minus any diagnosis); otherwise a generic confirm-and-follow-up."""
    hi = f"Hi {customer_name}," if customer_name else "Hi,"
    if intent == "technical_support":
        body = (
            "Thanks for flagging this — I want to get you an accurate answer rather "
            "than guess. To dig in, could you share:\n"
            "- the exact API request you're making (endpoint + parameters),\n"
            "- the full error response you're getting back, and\n"
            "- the approximate time(s) it happened.\n\n"
            "With those I'll investigate and follow up with what's going on and how to fix it."
        )
    else:
        body = ("Thanks for reaching out. Let me confirm the details on this and get "
                "back to you shortly with an accurate answer.")
    return f"{hi}\n\n{body}\n\n{_SIGNOFF}"


def holding_reply(customer_name: str | None, intent: str) -> str:
    """Public accessor for the safe, fact-free holding reply. Callers that hit a
    RAG error (and so have no RagResult) use this to hold cleanly rather than
    fall back to an ungrounded free-write draft — the grounded-or-hold guarantee
    must survive a RAG failure, not just a RAG hold."""
    return _holding_reply(customer_name, intent)


def _holding(customer_name, reason, top1, intent, *, extra_flag=None):
    flags = [{"type": "rag_holding",
              "text": f"No confident/grounded doc answer ({reason}); holding reply drafted — "
                      f"needs a manual answer."}]
    if extra_flag:
        flags.append(extra_flag)
    return RagResult(PATH_WEAK, _holding_reply(customer_name, intent), flags=flags,
                     reason=reason, top1=top1)


def _guide_url(chunks: list[dict]) -> str | None:
    """Pick the best 'guide' page to link for a how-to answer. Prefer docs/
    support pages (GitHub code files aren't guides), then the URL backed by the
    most retrieved chunks (tie-break by best similarity). Returns None if there's
    nothing linkable -> caller falls through to the LLM draft path."""
    pool = [c for c in chunks if c.get("source_type") in ("docs", "support")] or chunks
    agg: dict[str, list] = {}
    for c in pool:
        u = c.get("url")
        if not u:
            continue
        a = agg.setdefault(u, [0, 0.0])
        a[0] += 1
        a[1] = max(a[1], c.get("similarity", 0.0))
    if not agg:
        return None
    return sorted(agg.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[0][0]


def _howto_reply(customer_name: str | None, url: str) -> str:
    hi = f"Hi {customer_name}," if customer_name else "Hi,"
    return (f"{hi}\n\nHere's the guide for that: {url} — feel free to ask if you "
            f"hit anything.\n\n{_SIGNOFF}")


def _citations(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        u = c.get("url")
        if u and u not in seen:
            seen.add(u)
            out.append({"url": u, "title": c.get("title"), "source_type": c.get("source_type")})
    return out


def answer(query: str, *, customer_name: str | None = None,
           intent: str = "platform_query", embed=None, search=None) -> RagResult:
    """Run the layered RAG stack for one question. `query` is the
    retrieval/answer text (the customer's question). `intent` only shapes the
    holding-reply wording (technical_support asks for debugging inputs). The
    answer-or-hold rule is identical for all intents. embed/search injectable."""
    embed = embed or (lambda q: llm.embed([q])[0])
    search = search or (lambda v, k: db.search_chunks(v, k))

    chunks = search(embed(query), RETRIEVE_K)
    if not chunks:
        return _holding(customer_name, "no_chunks", 0.0, intent)

    top1 = chunks[0]["similarity"]
    mean3 = sum(c["similarity"] for c in chunks[:3]) / min(3, len(chunks))

    # 1) score floor — cheap off-topic rejection, no LLM call.
    if top1 < PARTIAL_FLOOR:
        return _holding(customer_name, f"weak_score(top1={top1:.3f})", top1, intent)

    ctx = chunks[:CONTEXT_CHUNKS]

    # 2) answerability gate (conservative; uncertainty -> holding).
    gate = llm.rag_answerability(query, ctx)
    if not gate.get("supported"):
        return _holding(customer_name, f"gate_unsupported:{gate.get('reason','')[:80]}", top1, intent)

    tier = PATH_STRONG if (top1 >= STRONG_TOP1 and mean3 >= STRONG_MEAN3) else PATH_PARTIAL

    # 3a) how-to -> deterministic link-forward reply built in code from the best
    #     guide URL. Grounded by construction (URL comes from a retrieved chunk
    #     the gate judged relevant): no LLM draft, no groundedness call, no
    #     variance. Falls through to the LLM path only if there's no linkable URL.
    if gate.get("question_type") == "howto":
        url = _guide_url(ctx)
        if url:
            flags = []
            if tier == PATH_PARTIAL:
                flags = [{"type": "rag_partial",
                          "text": "Guide linked from a PARTIAL doc match — confirm it's the right "
                                  "page before sending."}]
            return RagResult(tier, _howto_reply(customer_name, url), flags=flags,
                             citations=_citations(ctx), reason="howto_template", top1=top1)

    # 3b) factual (or how-to with no linkable URL): draft from retrieved chunks.
    draft_text = llm.rag_draft(query, ctx, gate.get("question_type", "factual"))

    # 4a) deterministic numeric-grounding gate — every number in the draft must
    #     appear verbatim in the chunks, else hold. Catches a derived/fabricated
    #     figure (e.g. "737") that the probabilistic LLM gate may miss. Runs
    #     BEFORE the LLM gate; both must pass.
    nums_ok, missing = _numbers_grounded(draft_text, ctx)
    if not nums_ok:
        return _holding(customer_name, "ungrounded_number", top1, intent,
                        extra_flag={"type": "rag_ungrounded_number",
                                    "text": f"Draft cited number(s) not found in docs: "
                                            f"{', '.join(sorted(missing))} — held."})

    # 4b) groundedness gate — every claim must trace to a chunk. A pure
    #    link-forward reply (only a verbatim doc URL + framing) is grounded by
    #    construction, so we skip the (nondeterministic) LLM call for it; any
    #    other shape falls through to the gate.
    if _link_forward_grounded(draft_text, ctx):
        grounded = {"grounded": True}
    else:
        grounded = llm.rag_groundedness(draft_text, ctx)
    if not grounded.get("grounded"):
        bad = ", ".join(grounded.get("unsupported_claims", [])[:3])
        return _holding(customer_name, "ungrounded", top1, intent,
                        extra_flag={"type": "rag_ungrounded",
                                    "text": f"Draft had unsupported claims, downgraded: {bad}"})

    # 5) honesty flagger + path flag.
    flags = draft_mod.flag_unverified_specifics(draft_text)
    if tier == PATH_PARTIAL:
        flags = [{"type": "rag_partial",
                  "text": "Answer is from a PARTIAL doc match — verify it against the cited source "
                          "before sending."}] + flags

    return RagResult(tier, draft_text, flags=flags, citations=_citations(ctx),
                     reason=gate.get("reason", ""), top1=top1)
