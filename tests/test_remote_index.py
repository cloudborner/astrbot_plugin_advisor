import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from advisor.index import sha256_hex
from advisor.remote_index import (
    _RejectRedirectHandler,
    _validate_public_https_url,
    signature_payload,
    update_from_manifest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


class RemoteIndexTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        self.public_pem = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.manifest = {
            "schema_version": 1,
            "key_id": "test",
            "index_url": "resource_profiles.json",
            "sha256": "a" * 64,
            "signed_at": "2026-08-23T00:00:00+00:00",
        }
        self.manifest["signature"] = base64.b64encode(
            self.private.sign(signature_payload(self.manifest))
        ).decode("ascii")

    def test_valid_signature(self):
        verify_manifest(self.manifest, self.public_pem)

    def test_tampered_manifest_fails(self):
        self.manifest["sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            verify_manifest(self.manifest, self.public_pem)

    def test_bundled_manifest_matches_bundled_index(self):
        manifest = json.loads(
            (ROOT / "data" / "resource_profiles.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        verify_manifest(manifest, (ROOT / "data" / "index_public_key.pem").read_bytes())
        self.assertEqual(
            manifest["sha256"],
            hashlib.sha256(
                (ROOT / "data" / "resource_profiles.json").read_bytes()
            ).hexdigest(),
        )

    def test_rejects_local_and_private_index_urls(self):
        for url in (
            "http://example.com/index.json",
            "https://localhost/index.json",
            "https://127.0.0.1/index.json",
            "https://10.0.0.1/index.json",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_public_https_url(url)

    def test_remote_index_redirects_are_rejected(self):
        handler = _RejectRedirectHandler()
        with self.assertRaisesRegex(ValueError, "redirects are not allowed"):
            handler.redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "https://127.0.0.1/private",
            )

    @staticmethod
    def _profile(plugin_id="owner/plugin", *, score=1):
        dimensions = (
            "idle_memory",
            "peak_memory",
            "idle_cpu",
            "peak_cpu",
            "disk",
            "network",
        )
        return {
            "plugin_id": plugin_id,
            "version": "1.0",
            "commit_sha": "a" * 40,
            "levels": {key: f"L{score}" for key in dimensions},
            "scores": {key: score for key in dimensions},
            "features": [],
            "external_processes": [],
            "background_tasks": "unknown",
            "evidence": [],
            "unknowns": [],
            "confidence": 0.5,
            "evidence_level": "github_tree",
            "scanned_at": "2026-08-24T00:00:00+00:00",
        }

    def _signed_payloads(
        self, *, generated_at="2026-08-24T00:00:00+00:00", profiles=None
    ):
        profiles = (
            profiles if profiles is not None else {"owner/plugin": self._profile()}
        )
        index = {
            "$meta": {
                "schema_version": 1,
                "generated_at": generated_at,
                "profile_count": len(profiles),
                "profiles_sha256": sha256_hex(profiles),
                "source_code_downloaded": False,
                "commit_sha_kind": "github_commit_oid",
                "commit_binding_api": "github_list_commits_metadata",
            },
            "profiles": profiles,
        }
        index_bytes = (json.dumps(index, sort_keys=True) + "\n").encode()
        manifest = dict(self.manifest)
        manifest["sha256"] = hashlib.sha256(index_bytes).hexdigest()
        manifest["signature"] = base64.b64encode(
            self.private.sign(signature_payload(manifest))
        ).decode("ascii")
        return json.dumps(manifest).encode(), index_bytes

    def test_update_is_atomic_and_keeps_backup(self):
        manifest_bytes, index_bytes = self._signed_payloads()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(b"old-index")
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with patch(
                "advisor.remote_index._download",
                side_effect=[manifest_bytes, index_bytes],
            ):
                loaded = update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(set(loaded["profiles"]), {"owner/plugin"})
            self.assertEqual(destination.read_bytes(), index_bytes)
            self.assertEqual(
                destination.with_suffix(".json.bak").read_bytes(), b"old-index"
            )

    def test_checksum_failure_preserves_current_index(self):
        manifest_bytes, index_bytes = self._signed_payloads()
        damaged = index_bytes + b"damage"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(b"current-index")
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[manifest_bytes, damaged],
                ),
                self.assertRaisesRegex(ValueError, "checksum mismatch"),
            ):
                update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(destination.read_bytes(), b"current-index")
            self.assertFalse(destination.with_suffix(".json.bak").exists())

    def test_empty_signed_index_is_rejected_without_replacement(self):
        manifest_bytes, index_bytes = self._signed_payloads(profiles={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(b"current-index")
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[manifest_bytes, index_bytes],
                ),
                self.assertRaisesRegex(ValueError, "empty|minimum coverage"),
            ):
                update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(destination.read_bytes(), b"current-index")

    def test_stale_signed_index_is_rejected(self):
        old_manifest, old_bytes = self._signed_payloads(
            generated_at="2026-08-25T00:00:00+00:00"
        )
        new_manifest, replay_bytes = self._signed_payloads(
            generated_at="2026-08-24T00:00:00+00:00"
        )
        del old_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(old_bytes)
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[new_manifest, replay_bytes],
                ),
                self.assertRaisesRegex(ValueError, "stale|replayed"),
            ):
                update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(destination.read_bytes(), old_bytes)

    def test_first_update_rejects_index_older_than_bundled_baseline(self):
        _bundled_manifest, bundled_bytes = self._signed_payloads(
            generated_at="2026-08-25T00:00:00+00:00"
        )
        candidate_manifest, candidate_bytes = self._signed_payloads(
            generated_at="2026-08-24T00:00:00+00:00"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            bundled = root / "bundled.json"
            bundled.write_bytes(bundled_bytes)
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[candidate_manifest, candidate_bytes],
                ),
                self.assertRaisesRegex(ValueError, "stale|replayed"),
            ):
                update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                    baseline_path=bundled,
                )
            self.assertFalse(destination.exists())

    def test_identical_signed_index_is_a_noop(self):
        manifest_bytes, index_bytes = self._signed_payloads()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(index_bytes)
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[manifest_bytes, index_bytes],
                ),
                patch("advisor.remote_index.os.replace") as replace,
            ):
                loaded = update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(loaded["$meta"]["profile_count"], 1)
            replace.assert_not_called()
            self.assertFalse(destination.with_suffix(".json.bak").exists())

    def test_invalid_profile_semantics_are_rejected(self):
        broken = self._profile()
        broken["scores"]["peak_memory"] = 4
        manifest_bytes, index_bytes = self._signed_payloads(
            profiles={"owner/plugin": broken}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "resource_profiles.json"
            destination.write_bytes(b"current-index")
            public_key = root / "public.pem"
            public_key.write_bytes(self.public_pem)
            with (
                patch(
                    "advisor.remote_index._download",
                    side_effect=[manifest_bytes, index_bytes],
                ),
                self.assertRaisesRegex(ValueError, "level/score mismatch"),
            ):
                update_from_manifest(
                    "https://example.com/manifest.json",
                    destination=destination,
                    public_key_path=public_key,
                )
            self.assertEqual(destination.read_bytes(), b"current-index")


if __name__ == "__main__":
    unittest.main()
