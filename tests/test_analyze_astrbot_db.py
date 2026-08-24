import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_astrbot_db import analyze_database, iter_user_messages

ROOT = Path(__file__).resolve().parents[1]


class OfflineDatabaseAnalysisTests(unittest.TestCase):
    def test_conversation_extractor_accepts_only_user_roles(self):
        payload = [
            {"role": "system", "content": "secret system prompt"},
            {"role": "assistant", "content": "assistant answer"},
            {"role": "user", "content": "洛克王国 洛克王国"},
            {"role": "tool", "content": "tool output"},
        ]
        messages = list(iter_user_messages(payload, source_table="conversations"))
        self.assertEqual(messages, [("洛克王国 洛克王国", ["text"])])

    def test_report_is_aggregate_only_and_excludes_identity_and_raw_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "data.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE conversations (content TEXT)")
            connection.execute("CREATE TABLE platform_message_history (content TEXT)")
            connection.execute(
                "INSERT INTO conversations(content) VALUES (?)",
                (
                    json.dumps(
                        [
                            {"role": "assistant", "content": "PRIVATE_ASSISTANT"},
                            {
                                "role": "user",
                                "content": "洛克王国 PRIVATE_RAW 12345678901",
                            },
                        ],
                        ensure_ascii=False,
                    ),
                ),
            )
            history = {
                "sender_id": "987654321",
                "group_id": "123456789",
                "message": [
                    {"type": "Plain", "text": "洛克王国 wiki"},
                    {"type": "Image", "url": "https://private.example/id"},
                ],
            }
            connection.execute(
                "INSERT INTO platform_message_history(content) VALUES (?)",
                (json.dumps(history, ensure_ascii=False),),
            )
            connection.execute(
                "INSERT INTO platform_message_history(content) VALUES (?)",
                (json.dumps(history, ensure_ascii=False),),
            )
            connection.commit()
            connection.close()

            report = analyze_database(
                database,
                market_snapshot=ROOT / "data" / "market_snapshot.json",
                taxonomy_path=ROOT / "data" / "plugin_taxonomy.json",
            )
            encoded = json.dumps(report, ensure_ascii=False)
            self.assertTrue(report["$meta"]["aggregate_only"])
            self.assertTrue(report["$meta"]["database_opened_read_only"])
            self.assertEqual(report["user_messages_analyzed"], 3)
            self.assertEqual(report["cross_table_duplicates_skipped"], 0)
            self.assertFalse(report["sample_sufficient"])
            self.assertNotIn("PRIVATE_RAW", encoded)
            self.assertNotIn("PRIVATE_ASSISTANT", encoded)
            self.assertNotIn("987654321", encoded)
            self.assertNotIn("123456789", encoded)
            self.assertNotIn("private.example", encoded)
            self.assertEqual(report["topics"], [])

    def test_read_only_database_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "data.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE conversations (content TEXT)")
            connection.execute("CREATE TABLE platform_message_history (content TEXT)")
            connection.commit()
            before = database.read_bytes()
            connection.close()
            analyze_database(
                database,
                market_snapshot=ROOT / "data" / "market_snapshot.json",
                taxonomy_path=ROOT / "data" / "plugin_taxonomy.json",
            )
            self.assertEqual(database.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
