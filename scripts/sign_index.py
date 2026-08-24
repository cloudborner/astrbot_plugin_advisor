from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.index import atomic_write_json, load_index  # noqa: E402
from advisor.remote_index import signature_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign an advisor resource index")
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data" / "resource_profiles.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "resource_profiles.manifest.json",
    )
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument(
        "--public-key", type=Path, default=ROOT / "data" / "index_public_key.pem"
    )
    parser.add_argument("--index-url", default="resource_profiles.json")
    parser.add_argument("--key-id", default="advisor-index-2026-01")
    parser.add_argument("--generate-key", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_index(args.index)
    if args.private_key.exists():
        private_key = serialization.load_pem_private_key(
            args.private_key.read_bytes(), password=None
        )
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("private key must be Ed25519")
    elif args.generate_key:
        private_key = Ed25519PrivateKey.generate()
        args.private_key.parent.mkdir(parents=True, exist_ok=True)
        args.private_key.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    else:
        raise FileNotFoundError("private key does not exist; pass --generate-key once")

    args.public_key.parent.mkdir(parents=True, exist_ok=True)
    args.public_key.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    digest = hashlib.sha256(args.index.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "key_id": args.key_id,
        "index_url": args.index_url,
        "sha256": digest,
        "signed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    manifest["signature"] = base64.b64encode(
        private_key.sign(signature_payload(manifest))
    ).decode("ascii")
    atomic_write_json(args.manifest, manifest)
    print(
        json.dumps(
            {"manifest": str(args.manifest), "sha256": digest, "key_id": args.key_id}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
