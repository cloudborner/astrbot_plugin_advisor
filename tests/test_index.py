import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from advisor.index import atomic_write_json, load_index, profile_is_current, sha256_hex


class IndexTests(unittest.TestCase):
    def test_profile_stale_when_market_record_is_newer_than_scan(self):
        profile = SimpleNamespace(
            version="1.0",
            commit_sha="abc",
            scanned_at="2026-08-20T00:00:00+00:00",
        )
        self.assertFalse(
            profile_is_current(
                profile,
                version="1.0",
                record_updated_at="2026-08-21T00:00:00Z",
            )
        )
        self.assertTrue(
            profile_is_current(
                profile,
                version="1.0",
                record_updated_at="2026-08-19T00:00:00Z",
            )
        )

    def test_profile_stale_when_commit_differs(self):
        profile = SimpleNamespace(
            version="1.0",
            commit_sha="abc",
            scanned_at="2026-08-20T00:00:00+00:00",
        )
        self.assertFalse(profile_is_current(profile, version="1.0", commit_sha="def"))

    def test_checksum_validation(self):
        profiles = {"owner/plugin": {"plugin_id": "owner/plugin"}}
        index = {
            "$meta": {"schema_version": 1, "profiles_sha256": sha256_hex(profiles)},
            "profiles": profiles,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            atomic_write_json(path, index)
            self.assertEqual(load_index(path)["profiles"], profiles)
            index["profiles"]["owner/plugin"]["version"] = "changed"
            atomic_write_json(path, index)
            with self.assertRaises(ValueError):
                load_index(path)


if __name__ == "__main__":
    unittest.main()
