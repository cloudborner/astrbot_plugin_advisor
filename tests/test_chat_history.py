import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from advisor.chat_history import (
    HistoryFetchResult,
    HistoryImportState,
    HistoryUnavailableError,
    OneBotHistoryProvider,
    normalize_history_message,
    provider_for_event,
    write_history_export,
)


def _message(seq, *, text=None, message_id=None, sender="10001"):
    return {
        "message_id": message_id or f"message-{seq}",
        "message_seq": seq,
        "time": 1_700_000_000 + seq,
        "group_id": "123456789",
        "user_id": sender,
        "sender": {"user_id": sender, "card": f"成员{sender}"},
        "message": [{"type": "text", "data": {"text": text or f"消息{seq}"}}],
        "raw_message": text or f"消息{seq}",
    }


class ChatHistoryTests(unittest.TestCase):
    def test_normalize_message_keeps_useful_segments_but_omits_base64(self):
        raw = _message(9, text="ignored")
        raw["message"] = [
            {"type": "text", "data": {"text": "查看这个"}},
            {
                "type": "image",
                "data": {"file": "base64://secret-content", "url": "https://x/img"},
            },
            {"type": "file", "data": {"name": "report.pdf", "file_id": "f1"}},
        ]

        item = normalize_history_message(raw, fallback_group_id="123456789")

        self.assertIsNotNone(item)
        self.assertEqual(item.sequence, 9)
        self.assertEqual(item.sender_name, "成员10001")
        self.assertEqual(item.text, "查看这个[图片][文件:report.pdf]")
        self.assertEqual(item.segments[1]["data"]["file"], "[base64 data omitted]")
        self.assertIn("image", item.component_types)
        self.assertIn("file", item.component_types)

    def test_llbot_and_napcat_common_action_paginates_and_deduplicates(self):
        calls = []

        async def call_action(action, **params):
            calls.append((action, params))
            if action == "get_version_info":
                return {"app_name": "NapCat.OneBot", "app_version": "4.18"}
            cursor = params.get("message_seq")
            if cursor is None:
                return {"messages": [_message(5), _message(4), _message(3)]}
            if cursor == 2:
                # Include one duplicate to cover adapters that retain the anchor.
                return {"data": {"messages": [_message(3), _message(2), _message(1)]}}
            return {"messages": []}

        provider = OneBotHistoryProvider(call_action, page_size=10)
        result = asyncio.run(provider.fetch_group_history(group_id="123456789", limit=5))

        self.assertEqual(result.provider, "NapCat / OneBot")
        self.assertEqual([item.sequence for item in result.messages], [1, 2, 3, 4, 5])
        history_calls = [item for item in calls if item[0] == "get_group_msg_history"]
        self.assertEqual(history_calls[0][1]["reverseOrder"], False)
        self.assertNotIn("message_seq", history_calls[0][1])
        self.assertEqual(history_calls[1][1]["message_seq"], 2)

    def test_provider_stops_safely_when_sequence_does_not_move(self):
        async def call_action(action, **_params):
            if action == "get_version_info":
                return {"app_name": "LLBot"}
            return {"messages": [_message(10)]}

        result = asyncio.run(
            OneBotHistoryProvider(call_action, page_size=10).fetch_group_history(
                group_id="123456789", limit=20
            )
        )

        self.assertEqual(result.provider, "LLBot / OneBot")
        self.assertEqual(len(result.messages), 1)
        self.assertIn("没有继续向前", result.warning)

    def test_unsupported_platform_has_clear_capability_error(self):
        with self.assertRaises(HistoryUnavailableError):
            provider_for_event(object())

    def test_import_state_is_idempotent_and_contains_no_plain_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "history_import_state.json")
            state = HistoryImportState(path, salt="test-salt", max_seen_per_group=100)
            state.mark(
                platform="aiocqhttp",
                group_id="123456789",
                stable_key="id:secret-message-id",
            )
            state.save()

            text = path.read_text(encoding="utf-8")
            self.assertNotIn("123456789", text)
            self.assertNotIn("secret-message-id", text)
            loaded = HistoryImportState(path, salt="test-salt", max_seen_per_group=100)
            self.assertTrue(
                loaded.contains(
                    platform="aiocqhttp",
                    group_id="123456789",
                    stable_key="id:secret-message-id",
                )
            )

    def test_json_jsonl_and_text_exports_are_atomic_and_readable(self):
        messages = tuple(
            normalize_history_message(_message(seq), fallback_group_id="123456789")
            for seq in (1, 2)
        )
        result = HistoryFetchResult(
            messages=messages,
            provider="LLBot / OneBot",
            requested=2,
            reached_limit=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            export_dir = Path(directory)
            json_path = write_history_export(
                export_dir,
                group_id="123456789",
                result=result,
                export_format="json",
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["message_count"], 2)
            self.assertEqual(payload["messages"][0]["message_seq"], 1)

            jsonl_path = write_history_export(
                export_dir,
                group_id="123456789",
                result=result,
                export_format="jsonl",
            )
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["type"] for row in rows], ["metadata", "message", "message"])

            txt_path = write_history_export(
                export_dir,
                group_id="123456789",
                result=result,
                export_format="txt",
            )
            text = txt_path.read_text(encoding="utf-8")
            self.assertIn("成员10001(10001): 消息1", text)
            self.assertFalse(list(export_dir.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
