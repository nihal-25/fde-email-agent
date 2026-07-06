"""Markdown -> retrievable chunks, split on heading structure.

Keeps each chunk tied to its nearest heading + the page title/url, packs to a
target size with a little overlap, and strips the Mintlify per-page boilerplate
(the "Documentation Index / fetch llms.txt" blockquote prepended to every .md).
"""

from __future__ import annotations

import hashlib
import re

TARGET_CHARS = 2200   # ~550 tokens
OVERLAP_CHARS = 300
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _strip_boilerplate(md: str) -> str:
    """Remove the leading Mintlify blockquote that points at llms.txt."""
    lines = md.splitlines()
    i = 0
    # Skip a leading run of blockquote/blank lines if it mentions the doc index.
    lead = "\n".join(lines[:8]).lower()
    if "documentation index" in lead or "llms.txt" in lead:
        while i < len(lines) and (lines[i].lstrip().startswith(">") or not lines[i].strip()):
            i += 1
    return "\n".join(lines[i:]).strip()


def _first_h1(md: str) -> str | None:
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return None


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1].replace(".md", "")
    return slug.replace("-", " ").replace("_", " ").title()


def _split_big(text: str, target: int, overlap: int) -> list[str]:
    """Split an oversized block on paragraph boundaries, with tail overlap."""
    paras = re.split(r"\n\s*\n", text)
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > target:
            out.append(cur.strip())
            tail = cur[-overlap:]
            cur = (tail + "\n\n" + p)
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        out.append(cur.strip())
    return out


def chunk_code(content: str, url: str, *, repo: str, path: str,
               target_chars: int = 2400, overlap_chars: int = 200) -> list[dict]:
    """Chunk a source/example file (no markdown headings). Each chunk is tagged
    with its file path so retrieval cites which file/repo it came from."""
    header = f"File: {path} (repo: {repo})"
    pieces = (_split_big(content, target_chars, overlap_chars)
              if len(content) > target_chars * 1.5 else [content])
    out = []
    for p in pieces:
        body = f"{header}\n\n{p.strip()}"
        if len(p.strip()) < 20:
            continue
        out.append({
            "source_type": "github", "url": url, "repo": repo,
            "title": path, "heading": path, "content": body,
            "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        })
    return out


def chunk_markdown(md: str, url: str, *, source_type: str = "docs",
                   repo: str | None = None, title: str | None = None,
                   target_chars: int = TARGET_CHARS, overlap_chars: int = OVERLAP_CHARS) -> list[dict]:
    """Return a list of chunk dicts ready for embedding + storage."""
    md = _strip_boilerplate(md)
    title = title or _first_h1(md) or _title_from_url(url)

    # Break into (heading, body) blocks; the heading line stays in the body.
    blocks: list[tuple[str, list[str]]] = []
    cur_heading = title
    buf: list[str] = []
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf:
                blocks.append((cur_heading, buf))
            cur_heading = m.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        blocks.append((cur_heading, buf))

    # Pack blocks into target-sized chunks; split any oversized block.
    packed: list[tuple[str, str]] = []
    cur_text, cur_head = "", title
    for heading, body_lines in blocks:
        text = "\n".join(body_lines).strip()
        if not text:
            continue
        if len(text) > target_chars * 1.5:
            if cur_text:
                packed.append((cur_head, cur_text)); cur_text = ""
            for piece in _split_big(text, target_chars, overlap_chars):
                packed.append((heading, piece))
            continue
        if cur_text and len(cur_text) + len(text) > target_chars:
            packed.append((cur_head, cur_text)); cur_text = ""
        if not cur_text:
            cur_head = heading
        cur_text = (cur_text + "\n\n" + text) if cur_text else text
    if cur_text:
        packed.append((cur_head, cur_text))

    out = []
    for heading, content in packed:
        content = content.strip()
        if len(content) < 20:  # drop near-empty fragments
            continue
        out.append({
            "source_type": source_type, "url": url, "repo": repo,
            "title": title, "heading": heading, "content": content,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    return out
