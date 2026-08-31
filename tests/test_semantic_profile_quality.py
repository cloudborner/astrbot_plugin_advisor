import unittest

from scripts.audit_semantic_profiles import audit_documents


class SemanticProfileQualityTests(unittest.TestCase):
    def test_reports_review_candidates_for_each_quality_risk(self):
        semantic = {
            "profiles": {
                "owner/repo": {
                    "summary": "一键搞定群聊处理，省钱又方便。",
                    "capabilities": [
                        {
                            "name": "状态与测试指令",
                            "evidence_refs": ["config:_conf_schema.json:timeout"],
                        }
                    ],
                    "use_cases": [],
                    "limitations": [{"text": "需要安装浏览器", "evidence_refs": ["readme:README.md"]}],
                }
            }
        }
        evidence = {
            "profiles": {
                "owner/repo": {
                    "evidence": {
                        "dependencies": ["playwright"],
                        "config_items": [
                            {
                                "file": "_conf_schema.json",
                                "key": "timeout",
                                "description": "请求超时时间",
                            },
                            {
                                "file": "_conf_schema.json",
                                "key": "ffmpeg_path",
                                "description": "ffmpeg 可执行文件路径",
                            },
                            {
                                "file": "_conf_schema.json",
                                "key": "account.cookie",
                                "description": "账号 Cookie",
                            },
                        ],
                    }
                }
            }
        }
        report = audit_documents(semantic, evidence)
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("promotional_language", codes)
        self.assertIn("pseudo_capability", codes)
        self.assertIn("unscoped_requirement", codes)
        self.assertIn("missing_prerequisite", codes)
        self.assertIn("missing_credential_condition", codes)
        self.assertIn("possible_evidence_mismatch", codes)

    def test_scoped_requirement_and_covered_prerequisite_are_not_reported(self):
        semantic = {
            "profiles": {
                "owner/repo": {
                    "summary": "新闻推送与玩家资料查询插件。",
                    "capabilities": [],
                    "use_cases": [],
                    "limitations": [
                        {
                            "text": "玩家资料查询需要 Cookie，刷新头像需要 Playwright/Chromium。",
                            "evidence_refs": ["readme:README.md"],
                        }
                    ],
                }
            }
        }
        evidence = {
            "profiles": {
                "owner/repo": {
                    "evidence": {
                        "dependencies": ["playwright"],
                        "config_items": [
                            {
                                "file": "_conf_schema.json",
                                "key": "account.cookie",
                                "description": "账号 Cookie",
                            }
                        ],
                    }
                }
            }
        }
        report = audit_documents(semantic, evidence)
        codes = {finding["code"] for finding in report["findings"]}
        self.assertNotIn("unscoped_requirement", codes)
        self.assertNotIn("missing_prerequisite", codes)
        self.assertNotIn("missing_credential_condition", codes)


if __name__ == "__main__":
    unittest.main()
