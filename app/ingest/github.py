"""Official Plivo GitHub repos -> chunks (READMEs, docs/, examples/).

Per repo: one API call for the default branch + one for the recursive tree, then
fetch wanted files via raw.githubusercontent.com (CDN — does not count against the
API rate limit). Uses GITHUB_TOKEN if set to lift the 60/hr unauth API limit.
"""

from __future__ import annotations

import json
import os

from app.ingest import crawl
from app.ingest.chunk import chunk_code, chunk_markdown

ORG = "plivo"

# Approved set. mode controls which files we ingest:
#   "sdk"     -> README + docs/ + examples/ (the SDK source tree is skipped)
#   "samples" -> whole repo (all doc+code files): these repos ARE sample/guide
#                collections, often organized by topic at the root (sms/, api/…),
#                so restricting to examples/ would miss the samples.
#   "readme"  -> README only
# Browser/mobile client SDKs and demo integrations are excluded for v1.
REPOS: list[tuple[str, str]] = [
    ("plivo-python", "sdk"), ("plivo-node", "sdk"), ("plivo-php", "sdk"),
    ("plivo-ruby", "sdk"), ("plivo-java", "sdk"), ("plivo-go", "sdk"),
    ("plivo-dotnet", "sdk"),
    ("plivo-examples-python", "samples"), ("plivo-examples-node", "samples"),
    ("plivo-examples-php", "samples"), ("plivo-examples-ruby", "samples"),
    ("plivo-examples-java", "samples"), ("plivo-examples-dotnet", "samples"),
    ("plivo-stream-sdk-python", "sdk"), ("plivo-stream-sdk-node", "sdk"),
    ("plivo-stream-sdk-java", "sdk"),
    ("plivo-agentstack-python", "sdk"), ("plivo-agentstack-node", "sdk"),
    ("plivo-agentstack-go", "sdk"),
    ("plivo-audiostream-integration-guides", "samples"),
    ("python-agents-examples", "samples"), ("plivo-streaming-examples", "samples"),
    ("plivo-cli", "readme"),
]

DOC_EXT = {".md", ".mdx", ".rst", ".txt"}
CODE_EXT = {".py", ".js", ".ts", ".php", ".go", ".rb", ".java", ".cs", ".sh"}
_SKIP = ("node_modules/", "vendor/", "/dist/", "/build/", ".min.",
         "package-lock.json", "composer.lock", "yarn.lock", "/test/", "/tests/", "/.github/")
MAX_FILES_PER_REPO = 80
MAX_FILE_BYTES = 120_000


def _api_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    return "." + name.rsplit(".", 1)[-1] if "." in name else ""


def _want(path: str, mode: str) -> bool:
    low = path.lower()
    if any(s in low for s in _SKIP):
        return False
    name = low.rsplit("/", 1)[-1]
    if name.startswith("readme"):
        return True
    if mode == "readme":
        return False
    ext = _ext(path)
    if mode == "samples":
        # whole repo is samples/guides -> any doc or code file
        return ext in (DOC_EXT | CODE_EXT)
    # mode == "sdk": only curated subtrees
    if low.startswith("docs/") and ext in DOC_EXT:
        return True
    if (low.startswith(("examples/", "example/", "samples/", "quickstart"))
            and ext in (DOC_EXT | CODE_EXT)):
        return True
    return False


def ingest_repo(fetcher: crawl.PoliteFetcher, repo: str, mode: str = "sdk"):
    """Yield (path, chunks) for the wanted files in one repo."""
    hdrs = _api_headers()
    meta = json.loads(fetcher.get(f"https://api.github.com/repos/{ORG}/{repo}",
                                  check_robots=False, headers=hdrs))
    branch = meta.get("default_branch", "main")
    tree = json.loads(fetcher.get(
        f"https://api.github.com/repos/{ORG}/{repo}/git/trees/{branch}?recursive=1",
        check_robots=False, headers=hdrs)).get("tree", [])

    picked = [b for b in tree if b.get("type") == "blob"
              and (b.get("size") or 0) <= MAX_FILE_BYTES and _want(b["path"], mode)]
    picked = picked[:MAX_FILES_PER_REPO]

    for blob in picked:
        path = blob["path"]
        raw = fetcher.get(f"https://raw.githubusercontent.com/{ORG}/{repo}/{branch}/{path}",
                          check_robots=False)
        html_url = f"https://github.com/{ORG}/{repo}/blob/{branch}/{path}"
        if _ext(path) in DOC_EXT:
            chunks = chunk_markdown(raw, html_url, source_type="github", repo=repo, title=path)
        else:
            chunks = chunk_code(raw, html_url, repo=repo, path=path)
        yield path, chunks
