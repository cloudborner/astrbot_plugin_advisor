import copy
import importlib.util
import json
import unittest
from pathlib import Path

from advisor.index import load_index
from scripts.validate_index import validate_quality

ROOT = Path(__file__).resolve().parents[1]


class DataQualityTests(unittest.TestCase):
    def test_bundled_index_is_self_consistent(self):
        index = load_index(ROOT / "data" / "resource_profiles.json")
        market = json.loads(
            (ROOT / "data" / "market_snapshot.json").read_text(encoding="utf-8")
        )["plugins"]
        result = validate_quality(
            index,
            market_plugins=market,
            minimum_profiles=1000,
            minimum_github_ratio=0.95,
        )
        self.assertEqual(result["source_code_downloaded"], False)
        expected_github = sum(
            1
            for item in index["profiles"].values()
            if str(item.get("evidence_level")).startswith("github_")
        )
        self.assertEqual(result["github_commit_bound"], expected_github)
        self.assertEqual(result["market_version_bound"], len(market))
        self.assertEqual(index["$meta"].get("commit_sha_kind"), "github_commit_oid")
        self.assertEqual(
            index["$meta"].get("commit_binding_api"),
            "github_list_commits_metadata",
        )

    def test_quality_rejects_version_mismatch(self):
        index = load_index(ROOT / "data" / "resource_profiles.json")
        market = json.loads(
            (ROOT / "data" / "market_snapshot.json").read_text(encoding="utf-8")
        )["plugins"]
        changed = copy.deepcopy(index)
        plugin_id = next(iter(changed["profiles"]))
        changed["profiles"][plugin_id]["version"] = "definitely-not-market-version"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            validate_quality(changed, market_plugins=market)

    def test_quality_rejects_github_evidence_without_commit(self):
        index = load_index(ROOT / "data" / "resource_profiles.json")
        changed = copy.deepcopy(index)
        profile = next(
            item
            for item in changed["profiles"].values()
            if str(item.get("evidence_level")).startswith("github_")
        )
        profile["commit_sha"] = ""
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            validate_quality(changed)

    def test_quality_rejects_tree_sha_mislabeled_as_commit_binding(self):
        index = load_index(ROOT / "data" / "resource_profiles.json")
        changed = copy.deepcopy(index)
        changed["$meta"]["commit_sha_kind"] = "github_tree_oid"
        with self.assertRaisesRegex(ValueError, "commit_sha_kind"):
            validate_quality(changed)

    def test_quality_rejects_source_detail_commit_endpoint(self):
        index = load_index(ROOT / "data" / "resource_profiles.json")
        changed = copy.deepcopy(index)
        changed["$meta"]["commit_binding_api"] = "github_commit_detail"
        with self.assertRaisesRegex(ValueError, "commit_binding_api"):
            validate_quality(changed)

    @unittest.skipUnless(
        importlib.util.find_spec("jsonschema"), "jsonschema not installed"
    )
    def test_bundled_index_matches_json_schema(self):
        import jsonschema

        schema = json.loads(
            (ROOT / "schemas" / "resource_index.schema.json").read_text(
                encoding="utf-8"
            )
        )
        index = json.loads(
            (ROOT / "data" / "resource_profiles.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(index))
        self.assertEqual(errors, [], errors[:3])


if __name__ == "__main__":
    unittest.main()
