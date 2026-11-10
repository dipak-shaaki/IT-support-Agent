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
        for question, source, answer in [
            ("How do I connect to the VPN?", "private_kb", "Install NovaSecure."),
            ("Latest Teams outage guidance", "web_search", "Check the status page."),
            ("Latest Teams outage guidance", "web_search", "Check the status page."),
            (
                "Something unknown",
                "private_kb",
                "That information is not documented in the company knowledge base.",
            ),
        ]:
            audit.write_audit(question, source, ["trace"], answer)

    def tearDown(self):
        audit.settings.audit_db_path = self.original_audit_db_path
        self.tmp.cleanup()

    def test_source_breakdown(self):
        counts = {row["source"]: row["count"] for row in audit.source_breakdown()}
        self.assertEqual(counts, {"web_search": 2, "private_kb": 2})

    def test_kb_gaps_groups_repeated_questions(self):
        gaps = audit.kb_gaps()
        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["question"], "Latest Teams outage guidance")
        self.assertEqual(gaps[0]["count"], 2)
        self.assertTrue(gaps[0]["last_seen"])

    def test_undocumented_private_kb_answers_are_counted_as_gaps(self):
        questions = {gap["question"] for gap in audit.kb_gaps()}

        self.assertIn("Something unknown", questions)
        self.assertNotIn("How do I connect to the VPN?", questions)
        self.assertEqual(audit.gap_total(), 3)

    def test_recent_and_daily(self):
        self.assertEqual(len(audit.recent_queries()), 4)
        self.assertEqual(audit.daily_counts()[0]["count"], 4)

    def test_limit_is_applied_to_gap_and_recent_results(self):
        self.assertEqual(len(audit.kb_gaps(limit=1)), 1)
        recent = audit.recent_queries(limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["question"], "Something unknown")


class AuditMigrationTest(unittest.TestCase):
    def test_init_db_adds_the_gap_column_to_an_existing_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = audit.settings.audit_db_path
            audit.settings.audit_db_path = os.path.join(tmp, "old.db")
            try:
                con = audit._connect()
                con.execute(
                    """CREATE TABLE query_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        question TEXT NOT NULL,
                        source_used TEXT NOT NULL,
                        trace_json TEXT NOT NULL
                    )"""
                )
                con.commit()
                con.close()

                audit.init_db()
                audit.write_audit(
                    "Undocumented",
                    "private_kb",
                    [],
                    "That information is not documented in the company knowledge base.",
                )

                self.assertEqual(audit.gap_total(), 1)
            finally:
                audit.settings.audit_db_path = original


if __name__ == "__main__":
    unittest.main()
