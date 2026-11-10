import os
import tempfile
import unittest

from app.services import audit


class AuditStatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_audit_db_path = audit.settings.audit_db_path
        audit.settings.audit_db_path = os.path.join(self.tmp.name, "audit.db")
        audit.init_db()
        for question, source in [
            ("How do I connect to the VPN?", "private_kb"),
            ("Latest Teams outage guidance", "web_search"),
            ("Latest Teams outage guidance", "web_search"),
            ("Something unknown", "insufficient_evidence"),
        ]:
            audit.write_audit(question, source, ["trace"])

    def tearDown(self):
        audit.settings.audit_db_path = self.original_audit_db_path
        self.tmp.cleanup()

    def test_source_breakdown(self):
        counts = {row["source"]: row["count"] for row in audit.source_breakdown()}
        self.assertEqual(counts, {"web_search": 2, "private_kb": 1, "insufficient_evidence": 1})

    def test_kb_gaps_groups_repeated_questions(self):
        gaps = audit.kb_gaps()
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["question"], "Latest Teams outage guidance")
        self.assertEqual(gaps[0]["count"], 2)
        self.assertTrue(gaps[0]["last_seen"])

    def test_recent_and_daily(self):
        self.assertEqual(len(audit.recent_queries()), 4)
        self.assertEqual(audit.daily_counts()[0]["count"], 4)

    def test_limit_is_applied_to_gap_and_recent_results(self):
        self.assertEqual(len(audit.kb_gaps(limit=1)), 1)
        recent = audit.recent_queries(limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["question"], "Something unknown")


if __name__ == "__main__":
    unittest.main()
