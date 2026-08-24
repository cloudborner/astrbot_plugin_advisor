import unittest
from unittest.mock import patch

from advisor.market import ApiError, GitHubClient


class RateLimitTests(unittest.TestCase):
    @patch("advisor.market._safe_json_request")
    def test_expired_absolute_deadline_makes_no_request(self, request):
        client = GitHubClient(max_retries=8, min_interval=0)
        with self.assertRaisesRegex(ApiError, "deadline exceeded"):
            client._get("/repos/a/b", deadline=0.0)
        request.assert_not_called()

    @patch("advisor.market.time.sleep", return_value=None)
    @patch("advisor.market._safe_json_request")
    def test_rate_limited_request_retries(self, request, _sleep):
        request.side_effect = [
            ApiError("limited", status=429, rate_limited=True),
            ({"ok": True}, {}),
        ]
        client = GitHubClient(max_retries=1, min_interval=0)
        value, _ = client._get("/test")
        self.assertTrue(value["ok"])
        self.assertEqual(request.call_count, 2)

    @patch("advisor.market._safe_json_request")
    def test_normal_403_does_not_retry(self, request):
        request.side_effect = ApiError("forbidden", status=403, rate_limited=False)
        client = GitHubClient(max_retries=4, min_interval=0)
        with self.assertRaises(ApiError):
            client._get("/test")
        self.assertEqual(request.call_count, 1)

    @patch("advisor.market.time.sleep", return_value=None)
    @patch("advisor.market._safe_json_request")
    def test_transient_network_error_retries(self, request, _sleep):
        request.side_effect = [
            ApiError("connection reset"),
            ({"ok": True}, {}),
        ]
        client = GitHubClient(max_retries=1, min_interval=0)
        value, _ = client._get("/test")
        self.assertTrue(value["ok"])
        self.assertEqual(request.call_count, 2)

    def test_rate_status_extracts_only_safe_counters(self):
        client = GitHubClient(token="secret")
        payload = {
            "resources": {
                "core": {"limit": 5000, "used": 12, "remaining": 4988, "reset": 42}
            }
        }
        with patch.object(client, "_get", return_value=(payload, {})):
            self.assertEqual(
                client.rate_status(),
                {"limit": 5000, "used": 12, "remaining": 4988, "reset": 42},
            )


if __name__ == "__main__":
    unittest.main()
