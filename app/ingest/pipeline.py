"""Docs ingest pipeline: discover -> fetch markdown -> chunk -> embed -> store.

Refresh-by-hash: per page, only NEW chunk hashes are embedded+inserted and
hashes that disappeared are deleted, so re-running is cheap.

CLI:
    python -m app.ingest.pipeline docs-test [N]   # dry run N pages, print samples
    python -m app.ingest.pipeline docs [N]        # full ingest/refresh (optional cap)
"""

from __future__ import annotations

import sys
import time

from app import db, llm
from app.ingest import crawl
from app.ingest.chunk import chunk_markdown

# text-embedding-3-small price (USD per 1M tokens) — for the cost estimate only.
_EMBED_USD_PER_1M = 0.02


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def run_docs_ingest(max_pages: int | None = None, *, dry_run: bool = False,
                    fetcher: crawl.PoliteFetcher | None = None) -> dict:
    fetcher = fetcher or crawl.PoliteFetcher(min_interval=1.0)
    t0 = time.monotonic()
    stats = {"pages": 0, "chunks_seen": 0, "chunks_new": 0, "chunks_deleted": 0,
             "tokens_embedded": 0, "errors": [], "blocked": False, "samples": []}

    try:
        urls = crawl.discover_doc_urls(fetcher)
    except crawl.Blocked as e:
        stats["blocked"] = True
        stats["errors"].append(f"discover: {e}")
        return stats
    if max_pages:
        urls = urls[:max_pages]
    stats["discovered"] = len(urls)

    for url in urls:
        try:
            md = crawl.fetch_doc_markdown(fetcher, url)
        except crawl.Blocked as e:
            stats["blocked"] = True
            stats["errors"].append(f"{url}: {e}")
            break  # stop immediately if the site blocks us
        except Exception as e:
            stats["errors"].append(f"{url}: {type(e).__name__}: {str(e)[:80]}")
            continue

        chunks = chunk_markdown(md, url, source_type="docs")
        stats["pages"] += 1
        stats["chunks_seen"] += len(chunks)

        if dry_run:
            if len(stats["samples"]) < 6 and chunks:
                # capture a couple of chunks per early page for eyeballing
                for c in chunks[:2]:
                    if len(stats["samples"]) < 6:
                        stats["samples"].append(c)
            continue

        existing = db.existing_chunk_hashes(url)
        new_hashes = {c["content_hash"] for c in chunks}
        to_insert = [c for c in chunks if c["content_hash"] not in existing]
        to_delete = [i for h, i in existing.items() if h not in new_hashes]

        if to_insert:
            vecs = llm.embed([c["content"] for c in to_insert])
            for c, v in zip(to_insert, vecs):
                c["embedding"] = v
                stats["tokens_embedded"] += _est_tokens(c["content"])
            db.add_chunks(to_insert)
            stats["chunks_new"] += len(to_insert)
        if to_delete:
            stats["chunks_deleted"] += db.delete_chunks(to_delete)

    stats["seconds"] = round(time.monotonic() - t0, 1)
    stats["est_cost_usd"] = round(stats["tokens_embedded"] / 1_000_000 * _EMBED_USD_PER_1M, 4)
    return stats


def _new_stats() -> dict:
    return {"chunks_seen": 0, "chunks_new": 0, "chunks_deleted": 0,
            "tokens_embedded": 0, "errors": [], "blocked": False, "samples": []}


def _finish(stats: dict, t0: float) -> dict:
    stats["seconds"] = round(time.monotonic() - t0, 1)
    stats["est_cost_usd"] = round(stats["tokens_embedded"] / 1_000_000 * _EMBED_USD_PER_1M, 4)
    return stats


def _capture_samples(stats: dict, chunks: list[dict], per: int = 1, cap: int = 6):
    for c in chunks[:per]:
        if len(stats["samples"]) < cap:
            stats["samples"].append(c)


def _upsert(url: str, chunks: list[dict], stats: dict):
    """Embed new chunks (by hash), insert, delete vanished — for one url."""
    if not chunks:
        return
    existing = db.existing_chunk_hashes(url)
    new_hashes = {c["content_hash"] for c in chunks}
    to_insert = [c for c in chunks if c["content_hash"] not in existing]
    to_delete = [i for h, i in existing.items() if h not in new_hashes]
    if to_insert:
        vecs = llm.embed([c["content"] for c in to_insert])
        for c, v in zip(to_insert, vecs):
            c["embedding"] = v
            stats["tokens_embedded"] += _est_tokens(c["content"])
        db.add_chunks(to_insert)
        stats["chunks_new"] += len(to_insert)
    if to_delete:
        stats["chunks_deleted"] += db.delete_chunks(to_delete)


def run_support_ingest(max_articles: int | None = None, *, dry_run: bool = False,
                       fetcher: crawl.PoliteFetcher | None = None) -> dict:
    from app.ingest import support
    fetcher = fetcher or crawl.PoliteFetcher(min_interval=1.0)
    t0 = time.monotonic()
    stats = _new_stats(); stats["articles"] = 0
    try:
        for article, chunks in support.ingest_articles(fetcher, max_articles=max_articles):
            stats["articles"] += 1
            stats["chunks_seen"] += len(chunks)
            _capture_samples(stats, chunks)
            if dry_run:
                continue
            _upsert(article["html_url"], chunks, stats)
    except crawl.Blocked as e:
        stats["blocked"] = True
        stats["errors"].append(str(e))
    except Exception as e:
        stats["errors"].append(f"{type(e).__name__}: {str(e)[:100]}")
    return _finish(stats, t0)


def run_github_ingest(*, dry_run: bool = False, repos=None,
                      fetcher: crawl.PoliteFetcher | None = None) -> dict:
    from app.ingest import github
    repos = repos if repos is not None else github.REPOS
    fetcher = fetcher or crawl.PoliteFetcher(min_interval=1.0)
    t0 = time.monotonic()
    stats = _new_stats(); stats["repos"] = 0; stats["files"] = 0
    for repo, mode in repos:
        try:
            for path, chunks in github.ingest_repo(fetcher, repo, mode):
                stats["files"] += 1
                stats["chunks_seen"] += len(chunks)
                _capture_samples(stats, chunks)
                if dry_run:
                    continue
                if chunks:
                    _upsert(chunks[0]["url"], chunks, stats)
            stats["repos"] += 1
        except crawl.Blocked as e:
            stats["blocked"] = True
            stats["errors"].append(f"{repo}: {e}")
            break  # stop on rate-limit/block; likely needs GITHUB_TOKEN
        except Exception as e:
            stats["errors"].append(f"{repo}: {type(e).__name__}: {str(e)[:80]}")
    return _finish(stats, t0)


def _print_samples(stats: dict):
    print("\n--- sample chunks ---")
    for i, c in enumerate(stats["samples"], 1):
        print(f"\n[{i}] {c['source_type']} url={c['url']}")
        print(f"    title={c['title']!r} heading={c['heading']!r} repo={c.get('repo')} chars={len(c['content'])}")
        for line in c["content"][:600].splitlines():
            print("    " + line)
        if len(c["content"]) > 600:
            print(f"    …(+{len(c['content'])-600} more chars)")


def _print_report(stats: dict, *, dry: bool):
    print("=" * 70)
    if dry:
        print(f"DRY RUN — pages fetched: {stats['pages']} / discovered {stats.get('discovered','?')}"
              f" · chunks: {stats['chunks_seen']} · {stats['seconds']}s")
        print(f"blocked={stats['blocked']} errors={len(stats['errors'])}")
        for e in stats["errors"][:5]:
            print("  err:", e)
        print("\n--- sample chunks (eyeball extraction/chunking quality) ---")
        for i, c in enumerate(stats["samples"], 1):
            print(f"\n[{i}] url={c['url']}")
            print(f"    title={c['title']!r}  heading={c['heading']!r}  chars={len(c['content'])}")
            body = c["content"]
            print("    ---")
            for line in body[:700].splitlines():
                print("    " + line)
            if len(body) > 700:
                print(f"    …(+{len(body)-700} more chars)")
    else:
        print(f"DOCS INGEST — pages: {stats['pages']} · chunks new: {stats['chunks_new']} "
              f"(deleted {stats['chunks_deleted']}, seen {stats['chunks_seen']})")
        print(f"tokens embedded: {stats['tokens_embedded']:,} · est cost: ${stats['est_cost_usd']} "
              f"· {stats['seconds']}s · blocked={stats['blocked']} · errors={len(stats['errors'])}")
        for e in stats["errors"][:8]:
            print("  err:", e)
        print(f"total chunks in store now: {db.count_chunks('docs')}")
    print("=" * 70)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else None
    if cmd == "docs-test":
        stats = run_docs_ingest(max_pages=n or 25, dry_run=True)
        _print_report(stats, dry=True)
    elif cmd == "docs":
        db.init_db()
        stats = run_docs_ingest(max_pages=n, dry_run=False)
        _print_report(stats, dry=False)
    elif cmd in ("support", "support-test"):
        dry = cmd.endswith("test")
        if not dry:
            db.init_db()
        stats = run_support_ingest(max_articles=(n or (5 if dry else None)), dry_run=dry)
        print("=" * 70)
        print(f"SUPPORT INGEST{' (dry)' if dry else ''} — articles: {stats['articles']} · "
              f"chunks new: {stats['chunks_new']} (seen {stats['chunks_seen']}, deleted {stats['chunks_deleted']})")
        print(f"tokens: {stats['tokens_embedded']:,} · est cost: ${stats['est_cost_usd']} · "
              f"{stats['seconds']}s · blocked={stats['blocked']} · errors={len(stats['errors'])}")
        for e in stats["errors"][:8]:
            print("  err:", e)
        if not dry:
            print(f"support chunks in store: {db.count_chunks('support')}")
        _print_samples(stats)
        print("=" * 70)
    elif cmd in ("github", "github-test"):
        dry = cmd.endswith("test")
        if not dry:
            db.init_db()
        stats = run_github_ingest(dry_run=dry)
        print("=" * 70)
        print(f"GITHUB INGEST{' (dry)' if dry else ''} — repos: {stats['repos']} · files: {stats['files']} · "
              f"chunks new: {stats['chunks_new']} (seen {stats['chunks_seen']}, deleted {stats['chunks_deleted']})")
        print(f"tokens: {stats['tokens_embedded']:,} · est cost: ${stats['est_cost_usd']} · "
              f"{stats['seconds']}s · blocked={stats['blocked']} · errors={len(stats['errors'])}")
        for e in stats["errors"][:12]:
            print("  err:", e)
        if not dry:
            print(f"github chunks in store: {db.count_chunks('github')}")
        _print_samples(stats)
        print("=" * 70)
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
