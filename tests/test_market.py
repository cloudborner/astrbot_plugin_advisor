import unittest
from unittest.mock import patch

from advisor.market import (
    ApiError,
    GitHubClient,
    GitHubObservation,
    load_market,
    parse_github_repo,
)
from advisor.models import MAX_COUNTER_VALUE, MAX_PLATFORMS, MAX_TAGS, PluginRecord


class MarketTests(unittest.TestCase):
    def test_untrusted_market_fields_are_strictly_bounded(self):
        record = PluginRecord.from_market(
            "owner/plugin",
            {
                "author": "a" * 1_000,
                "name": "n" * 1_000,
                "desc": "d" * 100_000,
                "tags": "must-not-expand",
                "support_platforms": {"bad": "shape"},
                "stars": "not-a-number",
                "download_count": 10**100,
            },
        )
        self.assertLessEqual(len(record.author), 128)
        self.assertLessEqual(len(record.name), 128)
        self.assertLessEqual(len(record.desc), 4_000)
        self.assertEqual(record.tags, [])
        self.assertEqual(record.support_platforms, [])
        self.assertEqual(record.stars, 0)
        self.assertEqual(record.download_count, MAX_COUNTER_VALUE)

        bounded = PluginRecord.from_market(
            "owner/plugin",
            {
                "tags": ["x" * 1_000] * (MAX_TAGS + 100),
                "support_platforms": ["y" * 1_000] * (MAX_PLATFORMS + 100),
            },
        )
        self.assertEqual(len(bounded.tags), MAX_TAGS)
        self.assertTrue(all(len(item) == 100 for item in bounded.tags))
        self.assertEqual(len(bounded.support_platforms), MAX_PLATFORMS)
        self.assertTrue(all(len(item) == 64 for item in bounded.support_platforms))

    @patch("advisor.network_safety.socket.getaddrinfo")
    @patch("advisor.network_safety.PUBLIC_HTTPS_OPENER.open")
    def test_market_rejects_hostname_resolving_to_private_ip(self, opener, resolve):
        resolve.return_value = [(2, 1, 6, "", ("169.254.169.254", 443))]
        with self.assertRaisesRegex(ValueError, "non-global"):
            load_market("https://market.example/plugins.json", max_retries=0)
        opener.assert_not_called()

    @patch("advisor.market.MAX_MARKET_PLUGINS", 1)
    @patch("advisor.market._safe_json_request")
    def test_market_rejects_excess_record_count(self, request):
        request.return_value = (
            {
                "$meta": {"version": "1"},
                "owner/one": {},
                "owner/two": {},
            },
            {},
        )
        with self.assertRaisesRegex(ApiError, "exceeds"):
            load_market(max_retries=0)

    def test_parse_repo_and_branch(self):
        self.assertEqual(
            parse_github_repo("https://github.com/owner/repo/tree/dev"),
            ("owner", "repo", "dev"),
        )

    def test_reject_non_github_url(self):
        with self.assertRaises(ValueError):
            parse_github_repo("https://example.com/owner/repo")

    def test_reject_path_injection(self):
        with self.assertRaises(ValueError):
            parse_github_repo("https://github.com/owner%20bad/repo")

    @patch("advisor.market.time.sleep", return_value=None)
    @patch("advisor.market._safe_json_request")
    def test_market_download_retries_transient_disconnect(self, request, _sleep):
        request.side_effect = [
            ApiError("incomplete response"),
            (
                {
                    "$meta": {"version": "1"},
                    "owner/plugin": {
                        "author": "owner",
                        "name": "plugin",
                        "repo": "https://github.com/owner/plugin",
                    },
                },
                {},
            ),
        ]
        metadata, records = load_market()
        self.assertEqual(metadata["version"], "1")
        self.assertEqual([item.plugin_id for item in records], ["owner/plugin"])
        self.assertEqual(request.call_count, 2)

    @patch("advisor.market.time.sleep", return_value=None)
    @patch("advisor.market.time.monotonic", side_effect=[0.0, 0.0, 1.0])
    @patch("advisor.market._safe_json_request", side_effect=ApiError("network"))
    def test_market_download_has_absolute_deadline(self, request, _monotonic, _sleep):
        with self.assertRaisesRegex(ApiError, "deadline"):
            load_market(max_retries=6, deadline_seconds=0.5)
        self.assertEqual(request.call_count, 1)

    def test_observation_binds_commit_separately_from_tree(self):
        client = GitHubClient(min_interval=0)
        commit_sha = "a" * 40
        tree_sha = "b" * 40
        with patch.object(
            client,
            "_get",
            side_effect=[
                (
                    [{"sha": commit_sha, "commit": {"tree": {"sha": tree_sha}}}],
                    {"x-ratelimit-remaining": "4999"},
                ),
                ({"sha": tree_sha, "tree": []}, {"x-ratelimit-remaining": "4998"}),
                ({"sbom": {"packages": []}}, {"x-ratelimit-remaining": "4997"}),
            ],
        ) as request:
            result = client.observe("https://github.com/owner/repo")
        self.assertTrue(result.commit_ok)
        self.assertEqual(result.commit_sha, commit_sha)
        self.assertEqual(result.tree_sha, tree_sha)
        self.assertEqual(result.commit_api, "list_commits_metadata")
        self.assertNotEqual(result.commit_sha, result.tree_sha)
        self.assertEqual(request.call_count, 3)
        self.assertIn("/commits?per_page=1", request.call_args_list[0].args[0])

    def test_legacy_cached_tree_only_needs_commit_request(self):
        client = GitHubClient(min_interval=0)
        cached = GitHubObservation(
            commit_sha="b" * 40,
            tree=[{"path": "main.py", "type": "blob", "size": 10}],
            packages=[],
            tree_ok=True,
            sbom_ok=False,
            errors=["sbom:404"],
        )
        with patch.object(
            client,
            "_get",
            return_value=(
                [{"sha": "a" * 40, "commit": {"tree": {"sha": "c" * 40}}}],
                {"x-ratelimit-remaining": "4999"},
            ),
        ) as request:
            result = client.observe("https://github.com/owner/repo", cached)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result.commit_sha, "a" * 40)
        self.assertEqual(result.tree_sha, "c" * 40)
        self.assertEqual(result.commit_api, "list_commits_metadata")
        self.assertTrue(result.tree_ok)
        self.assertIn("sbom:404", result.errors)

    def test_observation_can_defer_sbom_to_preserve_quota(self):
        client = GitHubClient(min_interval=0)
        with patch.object(
            client,
            "_get",
            side_effect=[
                (
                    [{"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}}],
                    {"x-ratelimit-remaining": "4999"},
                ),
                ({"sha": "b" * 40, "tree": []}, {"x-ratelimit-remaining": "4998"}),
            ],
        ) as request:
            first = client.observe("https://github.com/owner/repo", include_sbom=False)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(first.commit_ok)
        self.assertTrue(first.tree_ok)
        self.assertFalse(first.sbom_ok)

        with patch.object(
            client,
            "_get",
            return_value=(
                {"sbom": {"packages": [{"name": "aiohttp"}]}},
                {"x-ratelimit-remaining": "4997"},
            ),
        ) as request:
            completed = client.observe("https://github.com/owner/repo", first)
        self.assertEqual(request.call_count, 1)
        self.assertTrue(completed.sbom_ok)
        self.assertEqual(completed.packages, ["aiohttp"])


if __name__ == "__main__":
    unittest.main()
