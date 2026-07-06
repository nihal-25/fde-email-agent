"""Tests for app/pricing.resolve_country — the country normalization +
disambiguation logic (pure, no DB/LLM). This is the highest-stakes path: a wrong
resolution = a real rate for the WRONG country. Verifies near-misses resolve
EXACTLY, ambiguous terms ASK (never silently pick), and India phrasings normalize
to IN.
"""

import unittest

from app import pricing


class TestResolveCountry(unittest.TestCase):
    def _iso(self, name):
        r = pricing.resolve_country(name)
        self.assertEqual(r["status"], "resolved", f"{name!r} -> {r}")
        return r["iso"]

    def test_near_miss_pairs_resolve_distinctly(self):
        self.assertEqual(self._iso("Niger"), "NE")
        self.assertEqual(self._iso("Nigeria"), "NG")
        self.assertEqual(self._iso("Dominica"), "DM")
        self.assertEqual(self._iso("Dominican Republic"), "DO")

    def test_ambiguous_terms_ask_never_pick(self):
        for term, n in [("Congo", 2), ("Korea", 2), ("Guinea", 4), ("Sudan", 2)]:
            r = pricing.resolve_country(term)
            self.assertEqual(r["status"], "ambiguous", f"{term} -> {r}")
            self.assertEqual(len(r["candidates"]), n)

    def test_shared_calling_code_is_ambiguous(self):
        self.assertEqual(pricing.resolve_country("+1")["status"], "ambiguous")   # US/Canada
        self.assertEqual(pricing.resolve_country("+91")["iso"], "IN")            # unambiguous

    def test_india_phrasings_normalize_to_in(self):
        for phrase in ["India", "Indian numbers", "send to india",
                       "send to indian numbers", "+91"]:
            self.assertEqual(self._iso(phrase), "IN", phrase)

    def test_aliases(self):
        self.assertEqual(self._iso("UAE"), "AE")
        self.assertEqual(self._iso("the Emirates"), "AE")
        self.assertEqual(self._iso("Britain"), "GB")
        self.assertEqual(self._iso("Saudi"), "SA")
        self.assertEqual(self._iso("US"), "US")

    def test_unknown_is_unresolved(self):
        self.assertEqual(pricing.resolve_country("Wakanda")["status"], "unresolved")
        self.assertEqual(pricing.resolve_country("")["status"], "unresolved")

    def test_no_fuzzy_collapse(self):
        # "Niger" must NOT fuzzy-resolve to Nigeria; exact only.
        self.assertNotEqual(self._iso("Niger"), "NG")


if __name__ == "__main__":
    unittest.main()
