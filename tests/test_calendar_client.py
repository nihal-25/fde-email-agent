"""Tests for app/calendar_client.py — slot math + event body, all offline.

A fake Calendar service stands in for the Google API, so no network/OAuth.

Run: .venv/bin/python -m unittest tests.test_calendar_client
"""

import unittest
from datetime import datetime, timedelta

from app import calendar_client as cc
from app.calendar_client import IST


def ist(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=IST)


class FakeService:
    """Implements just the chained call shapes calendar_client uses."""

    def __init__(self, busy=None, event=None):
        self._busy = busy or []  # list of (start_dt, end_dt)
        self._event = event or {"id": "evt123", "htmlLink": "https://cal/evt123",
                                "hangoutLink": "https://meet.google.com/abc-defg-hij"}
        self.inserted = None
        self.patched = None
        self.last_query = None

    def freebusy(self):
        outer = self

        class _FB:
            def query(self, body):
                outer.last_query = body
                cal_id = body["items"][0]["id"]
                busy = [{"start": s.isoformat(), "end": e.isoformat()} for s, e in outer._busy]

                class _R:
                    def execute(self_inner):
                        return {"calendars": {cal_id: {"busy": busy}}}

                return _R()

        return _FB()

    def events(self):
        outer = self

        class _EV:
            def insert(self, calendarId, body, sendUpdates, conferenceDataVersion=None):
                outer.inserted = {"calendarId": calendarId, "body": body, "sendUpdates": sendUpdates,
                                  "conferenceDataVersion": conferenceDataVersion}

                class _R:
                    def execute(self_inner):
                        return outer._event

                return _R()

            def patch(self, calendarId, eventId, body, sendUpdates, conferenceDataVersion=None):
                outer.patched = {"calendarId": calendarId, "eventId": eventId, "body": body,
                                 "sendUpdates": sendUpdates, "conferenceDataVersion": conferenceDataVersion}

                class _R:
                    def execute(self_inner):
                        return outer._event

                return _R()

        return _EV()


def _monday(base: datetime) -> datetime:
    """The Monday 00:00 of base's week (deterministic weekday handling)."""
    m = base - timedelta(days=base.weekday())
    return m.replace(hour=0, minute=0, second=0, microsecond=0)


class TestFreeBusy(unittest.TestCase):
    def test_get_freebusy_parses_intervals(self):
        svc = FakeService(busy=[(ist(2026, 6, 22, 11), ist(2026, 6, 22, 12))])
        busy = cc.get_freebusy(ist(2026, 6, 22, 9), ist(2026, 6, 22, 18), service=svc)
        self.assertEqual(busy, [(ist(2026, 6, 22, 11), ist(2026, 6, 22, 12))])

    def test_is_free_true_when_no_overlap(self):
        svc = FakeService(busy=[(ist(2026, 6, 22, 11), ist(2026, 6, 22, 12))])
        self.assertTrue(cc.is_free(ist(2026, 6, 22, 15), ist(2026, 6, 22, 15, 30), service=svc))

    def test_is_free_false_when_overlap(self):
        svc = FakeService(busy=[(ist(2026, 6, 22, 11), ist(2026, 6, 22, 12))])
        # 11:30-12:00 overlaps the 11-12 busy block.
        self.assertFalse(cc.is_free(ist(2026, 6, 22, 11, 30), ist(2026, 6, 22, 12), service=svc))

    def test_is_free_adjacent_is_free(self):
        svc = FakeService(busy=[(ist(2026, 6, 22, 11), ist(2026, 6, 22, 12))])
        # 12:00-12:30 starts exactly when busy ends — not an overlap.
        self.assertTrue(cc.is_free(ist(2026, 6, 22, 12), ist(2026, 6, 22, 12, 30), service=svc))


class TestFindFreeSlots(unittest.TestCase):
    def test_skips_busy_and_caps_count(self):
        mon = _monday(ist(2026, 6, 22, 0))
        svc = FakeService(busy=[(mon.replace(hour=11), mon.replace(hour=12))])
        slots = cc.find_free_slots(mon.replace(hour=9), mon.replace(hour=18),
                                   duration_min=30, max_slots=3, service=svc)
        self.assertEqual(len(slots), 3)
        # Work starts 10:30, so the first slot is 10:30 (not 09:00).
        self.assertEqual(slots[0], (mon.replace(hour=10, minute=30), mon.replace(hour=11)))
        # Each slot is 30 min and none overlaps the 11-12 block.
        for s, e in slots:
            self.assertEqual(e - s, timedelta(minutes=30))
            self.assertFalse(s < mon.replace(hour=12) and e > mon.replace(hour=11))

    def test_excludes_past_via_window_start(self):
        mon = _monday(ist(2026, 6, 22, 0))
        svc = FakeService(busy=[])
        # Start mid-morning at 10:15 -> first slot rounds up to 10:30, not 09:00.
        slots = cc.find_free_slots(mon.replace(hour=10, minute=15), mon.replace(hour=18),
                                   duration_min=30, max_slots=2, service=svc)
        self.assertEqual(slots[0][0], mon.replace(hour=10, minute=30))

    def test_skips_weekend(self):
        mon = _monday(ist(2026, 6, 22, 0))
        sat = mon + timedelta(days=5)
        svc = FakeService(busy=[])
        slots = cc.find_free_slots(sat.replace(hour=9), sat.replace(hour=18),
                                   duration_min=30, service=svc)
        self.assertEqual(slots, [])

    def test_respects_working_hours_end(self):
        mon = _monday(ist(2026, 6, 22, 0))
        svc = FakeService(busy=[])
        slots = cc.find_free_slots(mon.replace(hour=17, minute=15), mon.replace(hour=23),
                                   duration_min=30, max_slots=5, service=svc)
        # 17:30-18:00 and 18:00-18:30 fit before the 18:30 work-end; 18:30+ does not.
        self.assertEqual(slots, [(mon.replace(hour=17, minute=30), mon.replace(hour=18)),
                                 (mon.replace(hour=18), mon.replace(hour=18, minute=30))])


class TestWorkingHours(unittest.TestCase):
    def setUp(self):
        self.mon = _monday(ist(2026, 6, 22, 0))

    def test_inside_hours(self):
        self.assertTrue(cc.within_working_hours(self.mon.replace(hour=15),
                                                self.mon.replace(hour=15, minute=30)))

    def test_before_open_is_out(self):
        # 10:00-10:30 starts before the 10:30 open.
        self.assertFalse(cc.within_working_hours(self.mon.replace(hour=10),
                                                 self.mon.replace(hour=10, minute=30)))

    def test_after_close_is_out(self):
        # 18:15-18:45 ends after the 18:30 close.
        self.assertFalse(cc.within_working_hours(self.mon.replace(hour=18, minute=15),
                                                 self.mon.replace(hour=18, minute=45)))

    def test_boundaries_inclusive(self):
        self.assertTrue(cc.within_working_hours(self.mon.replace(hour=10, minute=30),
                                                self.mon.replace(hour=11)))          # opens exactly
        self.assertTrue(cc.within_working_hours(self.mon.replace(hour=18),
                                                self.mon.replace(hour=18, minute=30)))  # closes exactly

    def test_weekend_is_out(self):
        sat = self.mon + timedelta(days=5)
        self.assertFalse(cc.within_working_hours(sat.replace(hour=12),
                                                 sat.replace(hour=12, minute=30)))


class TestCreateEvent(unittest.TestCase):
    def test_builds_event_with_attendee_and_invite(self):
        svc = FakeService()
        out = cc.create_event("Plivo <> Acme", ist(2026, 6, 22, 15), ist(2026, 6, 22, 15, 30),
                              "priya@acme.io", description="intro", service=svc)
        self.assertEqual(out["event_id"], "evt123")
        body = svc.inserted["body"]
        self.assertEqual(body["attendees"], [{"email": "priya@acme.io"}])
        self.assertEqual(body["start"]["timeZone"], "Asia/Kolkata")
        self.assertIn("+05:30", body["start"]["dateTime"])
        self.assertEqual(svc.inserted["sendUpdates"], "all")
        # Google Meet is always attached.
        self.assertEqual(svc.inserted["conferenceDataVersion"], 1)
        self.assertEqual(
            body["conferenceData"]["createRequest"]["conferenceSolutionKey"]["type"],
            "hangoutsMeet")
        self.assertTrue(body["conferenceData"]["createRequest"]["requestId"])
        self.assertEqual(out["meet_link"], "https://meet.google.com/abc-defg-hij")

    def test_meet_link_from_conference_entrypoint(self):
        # When hangoutLink is absent, the video entry point is used.
        svc = FakeService(event={"id": "e2", "htmlLink": "h", "conferenceData": {
            "entryPoints": [{"entryPointType": "video", "uri": "https://meet.google.com/xyz"}]}})
        out = cc.create_event("t", ist(2026, 6, 22, 15), ist(2026, 6, 22, 15, 30),
                              "a@b.com", service=svc)
        self.assertEqual(out["meet_link"], "https://meet.google.com/xyz")

    def test_requires_attendee(self):
        with self.assertRaises(RuntimeError):
            cc.create_event("x", ist(2026, 6, 22, 15), ist(2026, 6, 22, 15, 30), "", service=FakeService())


class TestUpdateEvent(unittest.TestCase):
    def test_patches_times_and_preserves_meet(self):
        svc = FakeService()
        out = cc.update_event("evt-1", ist(2026, 6, 22, 17), ist(2026, 6, 22, 17, 30), service=svc)
        self.assertEqual(svc.patched["eventId"], "evt-1")
        self.assertIn("+05:30", svc.patched["body"]["start"]["dateTime"])
        self.assertEqual(svc.patched["sendUpdates"], "all")
        self.assertEqual(svc.patched["conferenceDataVersion"], 1)  # keeps Meet
        self.assertEqual(out["meet_link"], "https://meet.google.com/abc-defg-hij")
        self.assertIsNone(svc.inserted)  # never created a new event

    def test_requires_event_id(self):
        with self.assertRaises(RuntimeError):
            cc.update_event("", ist(2026, 6, 22, 17), ist(2026, 6, 22, 17, 30), service=FakeService())


if __name__ == "__main__":
    unittest.main()
