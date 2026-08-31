import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_extracted_sources import expected_dir_names
from scripts.build_capability_index import build_document as build_capability_document
from scripts.build_full_plugin_index import plan_document, safe_delete_archive
from scripts.download_sources import DownloadItem
from scripts.extract_plugin_function_evidence import build_document as build_function_document
from scripts.validate_capability_index import validate_document

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class FullSourcePipelineTests(unittest.TestCase):
    def test_default_branch_directory_can_be_mapped_without_commit_sha(self):
        self.assertEqual(
            expected_dir_names("https://github.com/owner/repo", ""),
            {"owner__repo__default"},
        )
        self.assertIn(
            "owner__repo__default",
            expected_dir_names("https://github.com/owner/repo", "a" * 40),
        )

    def test_function_evidence_reads_commands_readme_config_and_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source_extracted"
            plugin_root = source_root / "owner__repo__default"
            plugin_root.mkdir(parents=True)
            (plugin_root / "README.md").write_text(
                "# 群聊知识助手\n\n把群聊内容整理成摘要并提供历史检索。\n\n## 知识库检索\n",
                encoding="utf-8",
            )
            (plugin_root / "main.py").write_text(
                "@filter.command('总结')\n"
                "async def summary(event):\n"
                "    '''总结最近的群聊消息'''\n"
                "    return None\n",
                encoding="utf-8",
            )
            write_json(
                plugin_root / "_conf_schema.json",
                {"general": {"items": {"history": {"description": "设置读取的历史消息数量"}}}},
            )
            market = {
                "$meta": {"generated_at": "2026-01-01T00:00:00Z"},
                "plugins": {
                    "owner/repo": {
                        "plugin_id": "owner/repo",
                        "author": "owner",
                        "name": "repo",
                        "display_name": "群聊助手",
                        "version": "1.0",
                        "repo": "https://github.com/owner/repo",
                        "desc": "群聊辅助工具",
                        "category": "管理",
                        "tags": ["总结"],
                    }
                },
            }
            market_path = root / "market.json"
            manifest_path = source_root / "pipeline_manifest.json"
            resources_path = root / "resources.json"
            write_json(market_path, market)
            write_json(
                manifest_path,
                {
                    "plugins": {
                        "owner/repo": {
                            "status": "complete",
                            "directory": plugin_root.name,
                        }
                    }
                },
            )
            write_json(
                resources_path,
                {
                    "profiles": {
                        "owner/repo": {
                            "features": ["remote_llm", "storage"],
                            "overall_level": "L1",
                            "dependencies": ["httpx"],
                        }
                    }
                },
            )
            document = build_function_document(
                market_path, source_root, manifest_path, resources_path
            )
            profile = document["profiles"]["owner/repo"]
            self.assertIn("总结", profile["capabilities"])
            self.assertIn("知识库检索", profile["capabilities"])
            self.assertIn("远程大模型调用", profile["capabilities"])
            self.assertEqual(profile["evidence"]["commands"][0]["line"], 2)
            self.assertIn("设置读取的历史消息数量", profile["use_cases"])
            self.assertFalse(document["$meta"]["plugin_code_executed"])

    def test_source_evidence_merges_into_capability_index_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path = root / "market.json"
            evidence_path = root / "evidence.json"
            market = {
                "$meta": {"generated_at": "2026-01-01T00:00:00Z", "market_version": "x"},
                "plugins": {
                    "owner/repo": {
                        "plugin_id": "owner/repo",
                        "author": "owner",
                        "name": "repo",
                        "version": "1.0",
                        "repo": "https://github.com/owner/repo",
                        "desc": "普通工具",
                        "category": "其他",
                        "tags": [],
                    }
                },
            }
            write_json(market_path, market)
            write_json(
                evidence_path,
                {
                    "$meta": {"plugin_code_executed": False},
                    "profiles": {
                        "owner/repo": {
                            "version": "1.0",
                            "summary": "自动整理群聊并检索历史知识",
                            "capabilities": ["群聊总结", "知识库检索"],
                            "aliases": ["总结"],
                            "use_cases": ["查询过去讨论过的内容"],
                            "limitations": [],
                            "sources": ["source_readme", "source_commands"],
                            "confidence": 0.85,
                        }
                    },
                },
            )
            document = build_capability_document(
                market_path, ROOT / "data" / "plugin_taxonomy.json", evidence_path
            )
            result = validate_document(
                document, market["plugins"], require_source_count=1
            )
            self.assertEqual(result["source_profiles"], 1)
            self.assertTrue(document["$meta"]["source_code_downloaded"])
            self.assertIn("知识库检索", document["profiles"]["owner/repo"]["capabilities"])

    def test_reviewed_semantic_profile_replaces_noisy_source_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path = root / "market.json"
            evidence_path = root / "evidence.json"
            semantic_path = root / "semantic.json"
            market = {
                "$meta": {"generated_at": "2026-01-01T00:00:00Z", "market_version": "x"},
                "plugins": {
                    "owner/repo": {
                        "plugin_id": "owner/repo",
                        "author": "owner",
                        "name": "repo",
                        "version": "1.0",
                        "repo": "https://github.com/owner/repo",
                        "desc": "普通工具",
                        "category": "其他",
                        "tags": [],
                    }
                },
            }
            evidence = {
                "$meta": {"plugin_code_executed": False},
                "profiles": {
                    "owner/repo": {
                        "version": "1.0",
                        "source_digest": "digest-1",
                        "summary": "源码自动提取的冗长描述",
                        "capabilities": ["安装方法", "帮助说明", "群聊总结"],
                        "aliases": ["旧别名"],
                        "use_cases": ["旧用法"],
                        "limitations": [],
                        "sources": ["market_metadata", "source_readme"],
                        "confidence": 0.7,
                        "evidence": {"readme_file": "README.md"},
                    }
                },
            }
            semantic = {
                "$meta": {"schema_version": 3},
                "profiles": {
                    "owner/repo": {
                        "plugin_id": "owner/repo",
                        "version": "1.0",
                        "source_digest": "digest-1",
                        "summary": "群聊总结插件：自动收集聊天消息并提炼主要话题和待办事项，可保存结果供成员稍后查阅。",
                        "capabilities": [
                            {"name": "群聊内容总结", "evidence_refs": ["readme:README.md"]}
                        ],
                        "aliases": ["群总结"],
                        "use_cases": [
                            {"text": "错过聊天后查看要点", "evidence_refs": ["readme:README.md"]},
                            {"text": "整理讨论中的待办事项", "evidence_refs": ["readme:README.md"]},
                        ],
                        "limitations": [],
                        "uncertainties": [],
                        "confidence": 0.82,
                    }
                },
                "failures": {},
            }
            write_json(market_path, market)
            write_json(evidence_path, evidence)
            write_json(semantic_path, semantic)
            document = build_capability_document(
                market_path,
                ROOT / "data" / "plugin_taxonomy.json",
                evidence_path,
                semantic_path,
            )
            profile = document["profiles"]["owner/repo"]
            self.assertEqual(profile["summary"], semantic["profiles"]["owner/repo"]["summary"])
            self.assertIn("群聊内容总结", profile["capabilities"])
            self.assertNotIn("安装方法", profile["capabilities"])
            self.assertEqual(profile["aliases"], ["群总结"])
            self.assertIn("source_llm_reviewed", profile["sources"])
            self.assertEqual(document["$meta"]["semantic_profile_count"], 1)

            semantic["profiles"]["owner/repo"]["capabilities"][0][
                "evidence_refs"
            ] = ["config:_conf_schema.json:unknown"]
            write_json(semantic_path, semantic)
            with self.assertRaisesRegex(ValueError, "evidence is invalid"):
                build_capability_document(
                    market_path,
                    ROOT / "data" / "plugin_taxonomy.json",
                    evidence_path,
                    semantic_path,
                )

    def test_plan_reports_commit_and_default_branch_counts(self):
        items = [
            DownloadItem("a/a", "A", "https://github.com/a/a", "a", "a", "a" * 40, "commit", "a.tar.gz", "https://example/a"),
            DownloadItem("b/b", "B", "https://github.com/b/b", "b", "b", "HEAD", "default_branch", "b.tar.gz", "https://example/b"),
        ]
        plan = plan_document(items, 2)
        self.assertEqual(plan["$meta"]["fixed_commit_count"], 1)
        self.assertEqual(plan["$meta"]["default_branch_count"], 1)

    def test_archive_delete_is_restricted_to_exact_archive_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "plugin.tar.gz"
            archive.write_bytes(b"archive")
            safe_delete_archive(archive, root)
            self.assertFalse(archive.exists())
            outside = root.parent / "outside.tar.gz"
            with self.assertRaises(ValueError):
                safe_delete_archive(outside, root)


if __name__ == "__main__":
    unittest.main()
