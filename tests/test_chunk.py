"""Tests for app/ingest/chunk.chunk_markdown (pure; no network)."""

import unittest

from app.ingest.chunk import chunk_code, chunk_markdown

BOILERPLATE = (
    "> ## Documentation Index\n"
    "> Fetch the complete documentation index at: https://plivo.com/docs/llms.txt\n"
    "> Use this file to discover all available pages before exploring further.\n\n"
)
PAGE = BOILERPLATE + (
    "# Send an SMS\n\n"
    "> Send a text message using the Messages API\n\n"
    "Use the Messages API to send an SMS.\n\n"
    "## Parameters\n\n"
    '<ParamField body="src" type="string">The sender number.</ParamField>\n\n'
    "## Example\n\n"
    "```python\nclient.messages.create(src='1', dst='2', text='hi')\n```\n"
)


class TestChunkMarkdown(unittest.TestCase):
    def test_strips_llms_boilerplate(self):
        chunks = chunk_markdown(PAGE, "https://plivo.com/docs/messaging/send")
        alltext = "\n".join(c["content"] for c in chunks)
        self.assertNotIn("Documentation Index", alltext)
        self.assertNotIn("llms.txt", alltext)

    def test_title_and_headings(self):
        chunks = chunk_markdown(PAGE, "https://plivo.com/docs/messaging/send")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["title"], "Send an SMS")
        headings = {c["heading"] for c in chunks}
        self.assertTrue(headings & {"Send an SMS", "Parameters", "Example"})

    def test_keeps_mdx_and_code(self):
        chunks = chunk_markdown(PAGE, "https://plivo.com/docs/messaging/send")
        alltext = "\n".join(c["content"] for c in chunks)
        self.assertIn("<ParamField", alltext)        # MDX structure kept
        self.assertIn("client.messages.create", alltext)  # code kept

    def test_metadata_and_hash(self):
        chunks = chunk_markdown(PAGE, "https://plivo.com/docs/messaging/send")
        c = chunks[0]
        self.assertEqual(c["source_type"], "docs")
        self.assertEqual(c["url"], "https://plivo.com/docs/messaging/send")
        self.assertEqual(len(c["content_hash"]), 64)  # sha256 hex
        # deterministic
        again = chunk_markdown(PAGE, "https://plivo.com/docs/messaging/send")
        self.assertEqual([x["content_hash"] for x in chunks], [x["content_hash"] for x in again])

    def test_title_fallback_from_url(self):
        chunks = chunk_markdown("Some text with no heading at all.", "https://x/docs/sip-trunking/setup-guide")
        self.assertEqual(chunks[0]["title"], "Setup Guide")


class TestChunkCode(unittest.TestCase):
    def test_tags_file_path_and_repo(self):
        code = "import plivo\nclient = plivo.RestClient('id', 'token')\nclient.messages.create()\n"
        chunks = chunk_code(code, "https://github.com/plivo/plivo-examples-python/blob/master/sms/send.py",
                            repo="plivo-examples-python", path="sms/send.py")
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c["source_type"], "github")
        self.assertEqual(c["repo"], "plivo-examples-python")
        self.assertEqual(c["heading"], "sms/send.py")
        self.assertIn("File: sms/send.py", c["content"])
        self.assertIn("client.messages.create", c["content"])


if __name__ == "__main__":
    unittest.main()
