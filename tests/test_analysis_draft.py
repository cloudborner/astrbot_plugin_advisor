import unittest

from advisor.analysis_draft import AnalysisDraftStore
from advisor.chat_history import normalize_history_message
from advisor.phrase_extraction import ExtractedPhrase


def history_message(seq, text, *, sender="10001", images=(), reply_to=""):
    segments = [{"type": "text", "data": {"text": text}}]
    segments.extend({"type": "image", "data": {"url": url}} for url in images)
    if reply_to:
        segments.append({"type": "reply", "data": {"id": reply_to}})
    return normalize_history_message(
        {
            "message_id": f"m-{seq}",
            "message_seq": seq,
            "time": 1_700_000_000 + seq,
            "group_id": "123456789",
            "user_id": sender,
            "message": segments,
        },
        fallback_group_id="123456789",
    )


class AnalysisDraftTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.store = AnalysisDraftStore(
            ttl_seconds=60,
            clock=lambda: self.now,
        )
        messages = [
            history_message(1, "图片识别讨论", images=("https://example.com/a.jpg",)),
            history_message(
                2,
                "继续讨论",
                sender="10002",
                images=("https://example.com/a.jpg", "https://example.com/b.jpg"),
                reply_to="m-1",
            ),
        ]
        self.draft = self.store.create(
            owner_id="3297718367",
            platform="aiocqhttp",
            group_id="123456789",
            messages=[item for item in messages if item is not None],
            phrases=[
                ExtractedPhrase("图片识别", 4, ("消息0001",)),
                ExtractedPhrase("群聊总结", 2, ("消息0002",)),
                ExtractedPhrase("jm", 1, ("消息0002",), "command"),
            ],
        )

    def test_image_references_are_deduplicated_and_context_linked(self):
        self.assertEqual(len(self.draft.images), 2)
        self.assertEqual(self.draft.images[0].message_evidence_id, "消息0001")
        self.assertEqual(self.draft.images[1].message_evidence_id, "消息0002")
        self.assertEqual(self.draft.messages[0].sender_alias, "用户001")
        self.assertEqual(self.draft.messages[1].sender_alias, "用户002")
        self.assertEqual(self.draft.messages[1].reply_to_evidence_id, "消息0001")
        self.assertEqual(self.draft.messages[1].source_platform, "aiocqhttp")
        self.assertEqual(self.draft.messages[1].message_type, "group")
        self.assertFalse(self.draft.messages[1].is_bot)

    def test_modify_delete_and_paging_keep_stable_indices(self):
        changed = self.draft.modify_phrase(2, "对话整理")
        self.assertEqual(changed.index, 2)
        self.assertTrue(changed.edited)
        deleted = self.draft.delete_phrase(1)
        self.assertEqual(deleted.index, 1)
        page, pages = self.draft.visible_phrases(page=1, page_size=1)
        self.assertEqual(pages, 2)
        self.assertEqual(page[0].index, 2)
        self.assertEqual(self.draft.phrase_at(3).index, 3)

    def test_duplicate_edit_labels_merge_only_in_model_payload(self):
        self.draft.modify_phrase(2, "图片识别")
        payload = self.draft.model_phrase_payload()
        merged = next(item for item in payload if item["phrase"] == "图片识别")
        self.assertEqual(merged["count"], 6)
        self.assertTrue(merged["user_edited"])
        self.assertEqual(self.draft.phrase_at(1).index, 1)
        self.assertEqual(self.draft.phrase_at(2).index, 2)

    def test_expiry_and_cancel_remove_raw_draft(self):
        self.assertIs(self.store.get("3297718367"), self.draft)
        self.now = 161.0
        self.assertIsNone(self.store.get("3297718367"))
        replacement = self.store.create(
            owner_id="3297718367",
            platform="aiocqhttp",
            group_id="123456789",
            messages=[],
            phrases=[],
        )
        self.assertIs(self.store.pop("3297718367"), replacement)
        self.assertIsNone(self.store.get("3297718367"))

    def test_new_draft_replaces_same_owners_previous_group_draft(self):
        self.now = 101.0
        second = self.store.create(
            owner_id="3297718367",
            platform="aiocqhttp",
            group_id="987654321",
            messages=[],
            phrases=[],
        )

        self.assertIsNone(
            self.store.get(
                "3297718367", platform="aiocqhttp", group_id="123456789"
            )
        )
        self.assertIs(
            self.store.get(
                "3297718367",
                platform="aiocqhttp",
                group_id="987654321",
            ),
            second,
        )
        self.assertIs(self.store.get("3297718367"), second)
        self.assertIs(
            self.store.pop(
                "3297718367",
                platform="aiocqhttp",
                group_id="987654321",
            ),
            second,
        )
        self.assertIsNone(self.store.get("3297718367"))

    def test_global_message_budget_evicts_oldest_other_owner(self):
        store = AnalysisDraftStore(
            ttl_seconds=60,
            max_total_messages=2,
            clock=lambda: self.now,
        )
        first = store.create(
            owner_id="owner-1",
            platform="aiocqhttp",
            group_id="group-1",
            messages=[history_message(1, "第一条"), history_message(2, "第二条")],
            phrases=[],
        )
        self.now = 101.0
        second = store.create(
            owner_id="owner-2",
            platform="aiocqhttp",
            group_id="group-2",
            messages=[history_message(3, "第三条")],
            phrases=[],
        )

        self.assertIsNone(store.get(first.owner_id))
        self.assertIs(store.get(second.owner_id), second)

    def test_draft_message_text_is_bounded_before_storage(self):
        store = AnalysisDraftStore(
            ttl_seconds=60,
            max_message_chars=100,
            clock=lambda: self.now,
        )
        draft = store.create(
            owner_id="owner",
            platform="aiocqhttp",
            group_id="group",
            messages=[history_message(1, "长" * 500)],
            phrases=[],
        )

        self.assertEqual(len(draft.messages[0].text), 100)

    def test_invalid_phrase_operations_are_rejected(self):
        with self.assertRaises(KeyError):
            self.draft.modify_phrase(99, "不存在")
        with self.assertRaises(ValueError):
            self.draft.modify_phrase(1, "")
        self.draft.delete_phrase(1)
        with self.assertRaises(KeyError):
            self.draft.delete_phrase(1)


if __name__ == "__main__":
    unittest.main()
