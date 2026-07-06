"""support.plivo.com ingest via the Zendesk Help Center content API.

The help center exposes clean JSON at /api/v2/help_center/en-us/articles.json
(title + HTML body + html_url + section). We page through it, convert each
article body HTML -> markdown, and chunk it. legacy-support.plivo.com is NOT
touched (stale content, excluded by instruction). Same PoliteFetcher politeness.
"""

from __future__ import annotations

import json

from markdownify import markdownify

from app.ingest import crawl
from app.ingest.chunk import chunk_markdown

ARTICLES_API = "https://support.plivo.com/api/v2/help_center/en-us/articles.json?per_page=100"


def _iter_articles(fetcher: crawl.PoliteFetcher):
    """Yield published, current articles across all pages."""
    url = ARTICLES_API
    while url:
        data = json.loads(fetcher.get(url))
        for a in data.get("articles", []):
            if a.get("draft") or a.get("outdated"):
                continue  # skip unpublished / stale
            if not (a.get("body") or "").strip():
                continue
            yield a
        url = data.get("next_page")


def ingest_articles(fetcher: crawl.PoliteFetcher, *, max_articles: int | None = None):
    """Yield (article, chunks) for each support article."""
    count = 0
    for a in _iter_articles(fetcher):
        if max_articles and count >= max_articles:
            break
        title = a.get("title") or ""
        md = markdownify(a.get("body") or "", heading_style="ATX").strip()
        # Prepend the title as an H1 so chunks carry it like docs pages do.
        full = f"# {title}\n\n{md}" if title else md
        chunks = chunk_markdown(full, a["html_url"], source_type="support", title=title)
        count += 1
        yield a, chunks
