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
    history_message_from_event,
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

    def test_legacy_cq_image_is_structured_and_emoji_is_excluded(self):
        message = normalize_history_message(
            {
                "message_id": "cq-1",
                "group_id": "123456",
                "user_id": "10001",
                "message": (
                    "看看这张图"
                    "[CQ:image,file=normal.jpg,url=https://example.com/a.jpg]"
                    "[CQ:image,file=face.jpg,url=https://example.com/face.jpg,sub_type=1]"
                ),
            },
            fallback_group_id="123456",
        )
        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.semantic_text, "看看这张图")
        self.assertEqual(message.image_references, ("https://example.com/a.jpg",))

    def test_component_metadata_is_counted_without_polluting_semantic_text(self):
        message = normalize_history_message(
            {
                "message_id": "components-1",
                "group_id": "123456",
                "user_id": "10001",
                "message": [
                    {"type": "text", "data": {"text": "/jm 处理 https://example.com/a"}},
                    {"type": "video", "data": {"file": "clip.mp4"}},
                    {"type": "file", "data": {"name": "notes.txt"}},
                    {"type": "reply", "data": {"id": "42"}},
                ],
            },
            fallback_group_id="123456",
        )
        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.command_texts, ("jm",))
        self.assertEqual(message.video_count, 1)
        self.assertEqual(message.file_count, 1)
        self.assertEqual(message.reply_count, 1)
        self.assertEqual(message.link_count, 1)
        self.assertNotIn("视频", message.semantic_text)
    def test_semantic_text_excludes_platform_labels_and_keeps_image_reference(self):
        message = normalize_history_message(
            {
                "message_id": "semantic-1",
                "time": 1_700_000_000,
                "group_id": "123456789",
                "user_id": "10001",
                "message": [
                    {"type": "text", "data": {"text": "用户真的需要图片识别"}},
                    {
                        "type": "image",
                        "data": {"url": "https://example.com/picture.jpg"},
                    },
                    {"type": "forward", "data": {"id": "forward-1"}},
                    {"type": "json", "data": {"content": "{}"}},
                ],
            },
            fallback_group_id="123456789",
        )
        self.assertIsNotNone(message)
        self.assertEqual(message.semantic_text, "用户真的需要图片识别")
        self.assertEqual(
            message.image_references,
            ("https://example.com/picture.jpg",),
        )
        self.assertIsNotNone(message.occurred_at)
        self.assertIn("[图片]", message.text)
        self.assertIn("[合并转发]", message.text)
        self.assertIn("[卡片]", message.text)

    def test_live_event_preserves_text_and_image_components(self):
        class Plain:
            text = "请分析这张图片"

        class Image:
            url = "https://example.com/live.jpg"
            file = ""
            path = ""

        class Event:
            def get_group_id(self):
                return "123456789"

            def get_messages(self):
                return [Plain(), Image()]

            def get_message_str(self):
                return "请分析这张图片[图片]"

            def get_sender_id(self):
                return "10001"

        message = history_message_from_event(Event())
        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.semantic_text, "请分析这张图片")
        self.assertEqual(message.image_references, ("https://example.com/live.jpg",))

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
