"""Polite fetching + docs.plivo.com discovery.

docs.plivo.com is a Mintlify site: every page has a clean markdown twin at
`<url>.md`, and /docs/sitemap.xml lists all pages. So we discover URLs from the
sitemap and fetch the `.md` of each — pristine content, no HTML/nav scraping.

Politeness: a single shared fetcher enforces a min interval between requests,
sends an identifying User-Agent, honors robots.txt where present, retries
5xx/429 with backoff, and raises if the site blocks us (so we stop, not hammer).
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from urllib.parse import urlparse

USER_AGENT = "PlivoFDEAgent-DocsIngest/0.1 (+internal RAG; contact nihal.manjunath@plivo.com)"
DOCS_SITEMAP = "https://docs.plivo.com/docs/sitemap.xml"

_ASSET_RE = re.compile(r"\.(png|jpe?g|svg|ico|css|js|woff2?|gif|webp|xml)$", re.IGNORECASE)


class Blocked(Exception):
    """Raised when the site refuses us (403 / persistent 429) — stop crawling."""


class PoliteFetcher:
    def __init__(self, min_interval: float = 1.0, timeout: float = 25.0, max_retries: int = 3):
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _robots_ok(self, url: str) -> bool:
        base = "{0.scheme}://{0.netloc}".format(urlparse(url))
        rp = self._robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._throttle()
                req = urllib.request.Request(base + "/robots.txt", headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    rp.parse(r.read().decode("utf-8", "replace").splitlines())
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    rp.allow_all = True  # no robots.txt -> allowed by convention
                else:
                    rp.allow_all = True
            except Exception:
                rp.allow_all = True
            self._robots[base] = rp
        return rp.can_fetch(USER_AGENT, url)

    def get(self, url: str, *, check_robots: bool = True, headers: dict | None = None) -> str:
        # check_robots=False for GitHub (api.github.com / raw.githubusercontent.com
        # are governed by API rate limits + ToS, not the website robots.txt, which
        # disallows all bots on the raw host). Plivo sites keep check_robots=True.
        if check_robots and not self._robots_ok(url):
            raise Blocked(f"robots.txt disallows {url}")
        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                if e.code in (403,):
                    raise Blocked(f"{e.code} on {url}")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise


def discover_doc_urls(fetcher: PoliteFetcher) -> list[str]:
    """Return all docs page URLs from the Mintlify sitemap (assets excluded)."""
    body = fetcher.get(DOCS_SITEMAP)
    locs = re.findall(r"<loc>([^<]+)</loc>", body)
    seen, pages = set(), []
    for u in locs:
        if _ASSET_RE.search(u) or u in seen:
            continue
        seen.add(u)
        pages.append(u)
    return pages


def fetch_doc_markdown(fetcher: PoliteFetcher, page_url: str) -> str:
    """Fetch the clean markdown twin of a docs page (Mintlify `<url>.md`)."""
    md_url = page_url if page_url.endswith(".md") else page_url + ".md"
    return fetcher.get(md_url)
