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
                    "limitations": [
                        {"text": "需要安装浏览器", "evidence_refs": ["readme:README.md"]}
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

    def test_generic_browser_and_login_wording_cover_runtime_conditions(self):
        semantic = {
            "profiles": {
                "owner/repo": {
                    "summary": "登录后可查询账号资料。",
                    "capabilities": [],
                    "use_cases": [],
                    "limitations": [
                        {
                            "text": "图片卡片功能需要本机浏览器环境。",
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
                        "dependencies": ["playwright", "selenium"],
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
        codes = {finding["code"] for finding in audit_documents(semantic, evidence)["findings"]}
        self.assertNotIn("missing_prerequisite", codes)
        self.assertNotIn("missing_credential_condition", codes)

    def test_python_library_is_not_treated_as_operator_prerequisite(self):
        semantic = {
            "profiles": {
                "owner/repo": {
                    "summary": "群聊记忆检索插件。",
                    "capabilities": [],
                    "use_cases": [],
                    "limitations": [],
                }
            }
        }
        evidence = {
            "profiles": {
                "owner/repo": {"evidence": {"dependencies": ["faiss-cpu"], "config_items": []}}
            }
        }
        codes = {finding["code"] for finding in audit_documents(semantic, evidence)["findings"]}
        self.assertNotIn("missing_prerequisite", codes)

    def test_known_unused_browser_dependency_is_exempt(self):
        semantic = {
            "profiles": {
                "SakuraMikku/astrbot_plugin_hardwareinfo": {
                    "summary": "查询主机硬件信息。",
                    "capabilities": [],
                    "use_cases": [],
                    "limitations": [],
                }
            }
        }
        evidence = {
            "profiles": {
                "SakuraMikku/astrbot_plugin_hardwareinfo": {
                    "evidence": {"dependencies": ["selenium"], "config_items": []}
                }
            }
        }
        codes = {finding["code"] for finding in audit_documents(semantic, evidence)["findings"]}
        self.assertNotIn("missing_prerequisite", codes)


if __name__ == "__main__":
    unittest.main()
