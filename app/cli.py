"""Local CLI for the FDE Email Agent (Milestone 2).

Pipe a sample email thread file in, get a classification + drafted reply out —
with zero external setup (no Gmail, no Slack). This proves the "brain"
(classify -> draft) end to end.

Usage:
    python -m app.cli samples/sample_email.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import db, draft as draft_mod, llm


def load_thread(path: str) -> dict:
    """Load a thread JSON file (simplified Gmail-like shape)."""
    data = json.loads(Path(path).read_text())
    if "messages" not in data or not isinstance(data["messages"], list):
        raise ValueError(f"{path}: expected a 'messages' array in the thread JSON")
    return data


def render_thread(thread: dict) -> str:
    """Flatten a thread into plain text, oldest message first.

    This is the single textual representation we hand to the model. When Gmail
    ingest lands (Milestone 5), it will produce a thread dict of this same
    shape so this renderer keeps working unchanged.
    """
    subject = thread.get("subject", "(no subject)")
    lines = [f"Subject: {subject}", ""]
    for msg in thread["messages"]:
        lines.append(f"From: {msg.get('from', '(unknown)')}")
        lines.append(f"Date: {msg.get('date', '(no date)')}")
        lines.append("")
        lines.append((msg.get("body", "")).strip())
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
    return "\n".join(lines).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FDE Email Agent — local CLI")
    parser.add_argument("thread_file", help="Path to a thread JSON file (see samples/)")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip persistence (don't write to Postgres). Useful with no DB running.",
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Post the draft to Slack for approval (requires SLACK_* env vars and "
             "a running listener: python -m app.slack_approval).",
    )
    args = parser.parse_args(argv)

    if args.slack and args.no_db:
        parser.error("--slack requires the DB (it needs a draft_id); drop --no-db.")

    thread = load_thread(args.thread_file)
    thread_text = render_thread(thread)

    print("=" * 70)
    print(f"Subject: {thread.get('subject', '(no subject)')}")
    print(f"Messages in thread: {len(thread['messages'])}")
    print("=" * 70)

    print("\nClassifying...\n")
    classification = llm.classify(thread_text)

    print("--- CLASSIFICATION ---")
    print(f"Intent:    {classification.get('intent')}")
    print(f"Urgency:   {classification.get('urgency')}")
    print(f"Customer:  {classification.get('customer_name')}  ({classification.get('company')})")
    print(f"Summary:   {classification.get('summary')}")
    key_points = classification.get("key_points") or []
    if key_points:
        print("Key points:")
        for kp in key_points:
            print(f"  - {kp}")

    print("\nDrafting reply...\n")
    draft = llm.draft_reply(thread_text, classification.get("intent", "other"))
    flags = draft_mod.flag_unverified_specifics(draft)

    print("--- DRAFTED REPLY (not sent — for human review) ---")
    print(draft)

    print("\n--- HONESTY REVIEW ---")
    print(draft_mod.format_flags(flags))

    if args.no_db:
        print("\n" + "=" * 70)
        print("Nothing was sent. This is a draft only. (DB write skipped: --no-db)")
        return 0

    print("\nPersisting to DB (email + draft + audit trail)...")
    db.init_db()
    ids = db.persist_processing(thread, classification, draft, source="cli")

    print("--- PERSISTED ---")
    print(f"email_id={ids['email_id']}  draft_id={ids['draft_id']}  thread_id={ids['thread_id']}")

    if args.slack:
        from app import slack_approval

        print("\nPosting draft to Slack for approval...")
        resp = slack_approval.post_draft_for_approval(thread_text, draft, ids["draft_id"], flags=flags)
        print(f"--- POSTED TO SLACK --- channel={resp.get('channel')} ts={resp.get('ts')}")
        print("Approve/Edit/Reject from Slack (listener must be running).")

    print("\n" + "=" * 70)
    print("Nothing was sent. This is a draft only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
