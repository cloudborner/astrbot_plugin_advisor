import unittest

from advisor.phrase_extraction import (
    PhraseSource,
    clean_semantic_text,
    extract_phrases,
)


class PhraseExtractionTests(unittest.TestCase):
    def test_short_latin_domain_alias_does_not_match_inside_other_words(self):
        rows = extract_phrases(
            [
                PhraseSource(
                    evidence_id="消息0001",
                    text="normal format information",
                )
            ],
            known_phrases=("rm", "robomaster"),
        )

        self.assertNotIn("rm", {item.text for item in rows})

    def test_transport_labels_and_identifiers_do_not_become_phrases(self):
        sources = [
            PhraseSource(
                "消息0001",
                "[合并转发][图片][卡片] 智能车调试 https://example.com/a 123456789",
            ),
            PhraseSource("消息0002", "智能车比赛 [动画表情]"),
        ]
        rows = extract_phrases(sources, known_phrases=("智能车",))
        values = {item.text: item.count for item in rows}
        self.assertEqual(values.get("智能车"), 2)
        for polluted in (
            "合并转发",
            "合并转",
            "并转发",
            "图片",
            "卡片",
            "动画表情",
            "链接",
            "编号",
        ):
            self.assertNotIn(polluted, values)

    def test_user_authored_image_word_is_preserved(self):
        rows = extract_phrases(
            [
                PhraseSource("消息0001", "需要图片识别功能"),
                PhraseSource("消息0002", "图片识别真的很有用"),
            ],
            known_phrases=("图片识别",),
        )
        self.assertEqual({item.text: item.count for item in rows}.get("图片识别"), 2)

    def test_command_and_evidence_are_preserved(self):
        rows = extract_phrases(
            [
                PhraseSource("消息0001", "/jm 123"),
                PhraseSource("消息0002", "大家又在用 /jm"),
            ]
        )
        command = next(item for item in rows if item.text == "jm")
        self.assertEqual(command.kind, "command")
        self.assertEqual(command.count, 2)
        self.assertEqual(command.evidence_ids, ("消息0001", "消息0002"))

    def test_invalid_blacklist_regex_is_ignored_safely(self):
        rows = extract_phrases(
            [PhraseSource("消息0001", "图片识别")],
            known_phrases=("图片识别",),
            blacklist_regexes=("(",),
        )
        self.assertIn("图片识别", {item.text for item in rows})

    def test_clean_text_keeps_placeholders_for_model_but_removes_secrets(self):
        text = clean_semantic_text(
            "联系 test@example.com，打开 https://example.com/path，编号123456789"
        )
        self.assertNotIn("test@example.com", text)
        self.assertNotIn("https://example.com/path", text)
        self.assertNotIn("123456789", text)
        self.assertIn("[邮箱]", text)
        self.assertIn("[链接]", text)
        self.assertIn("[编号]", text)

    def test_no_fixed_width_chinese_fragments_are_generated(self):
        rows = extract_phrases(
            [
                PhraseSource("消息0001", "今天晚上一起去打游戏吧"),
                PhraseSource("消息0002", "有没有人知道这个怎么弄"),
            ]
        )
        values = {item.text for item in rows}
        for fragment in (
            "今天晚上一起",
            "去打游戏吧",
            "有没有人知道",
            "这个怎么弄",
            "是什",
            "有没",
            "来不",
            "你不",
        ):
            self.assertNotIn(fragment, values)

    def test_blacklist_literal_is_removed_before_segmentation(self):
        rows = extract_phrases(
            [
                PhraseSource(
                    "消息0001",
                    "请使用最新版手机QQ查看详情，图片识别功能很好用",
                )
            ],
            known_phrases=("图片识别",),
        )
        values = {item.text for item in rows}
        self.assertIn("图片识别", values)
        for polluted in ("最新版", "手机", "qq", "查看", "详情"):
            self.assertNotIn(polluted, values)

    def test_repeated_complete_clause_is_kept_without_sliding_substrings(self):
        rows = extract_phrases(
            [
                PhraseSource("消息0001", "我们需要整理比赛报名资料"),
                PhraseSource("消息0002", "我们需要整理比赛报名资料"),
            ]
        )
        values = {item.text: item.count for item in rows}
        self.assertEqual(values.get("我们需要整理比赛报名资料"), 2)
        self.assertNotIn("需要整理比赛报", values)


if __name__ == "__main__":
    unittest.main()
