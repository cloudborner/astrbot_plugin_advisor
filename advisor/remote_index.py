from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .index import (
    MAX_INDEX_BYTES,
    canonical_json_bytes,
    index_generated_at,
    load_index,
    sha256_hex,
    validate_index_semantics,
)
from .network_safety import (
    PUBLIC_HTTPS_OPENER,
    RejectRedirectHandler,
    validate_public_https_url,
)

MAX_MANIFEST_BYTES = 64 * 1024
_RejectRedirectHandler = RejectRedirectHandler


def signature_payload(manifest: dict[str, Any]) -> bytes:
    signed = {
        "schema_version": int(manifest.get("schema_version") or 0),
        "key_id": str(manifest.get("key_id") or ""),
        "index_url": str(manifest.get("index_url") or ""),
        "sha256": str(manifest.get("sha256") or ""),
        "signed_at": str(manifest.get("signed_at") or ""),
    }
    return canonical_json_bytes(signed)


def _validate_public_https_url(url: str) -> str:
    return validate_public_https_url(url, label="remote index URL")


def _download(url: str, *, max_bytes: int, timeout: float) -> bytes:
    _validate_public_https_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "astrbot-plugin-advisor"},
        method="GET",
    )
    with PUBLIC_HTTPS_OPENER.open(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError("remote payload is too large")
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("remote payload exceeded size limit")
        return payload


def verify_manifest(manifest: dict[str, Any], public_key_pem: bytes) -> None:
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("unsupported remote index manifest schema")
    digest = str(manifest.get("sha256") or "")
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.lower()
    ):
        raise ValueError("invalid index sha256")
    signature = base64.b64decode(str(manifest.get("signature") or ""), validate=True)
    public_key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("resource index key must be Ed25519")
    try:
        public_key.verify(signature, signature_payload(manifest))
    except InvalidSignature as exc:
        raise ValueError("resource index signature verification failed") from exc


def update_from_manifest(
    manifest_url: str,
    *,
    destination: Path,
    public_key_path: Path,
    baseline_path: Path | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    manifest_bytes = _download(
        manifest_url, max_bytes=MAX_MANIFEST_BYTES, timeout=timeout
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("remote index manifest must be an object")
    verify_manifest(manifest, public_key_path.read_bytes())
    index_url = urllib.parse.urljoin(manifest_url, str(manifest["index_url"]))
    index_bytes = _download(index_url, max_bytes=MAX_INDEX_BYTES, timeout=timeout)
    actual = hashlib.sha256(index_bytes).hexdigest()
    if actual != str(manifest["sha256"]).lower():
        raise ValueError("downloaded resource index checksum mismatch")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(index_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_index(temp_path)
        validate_index_semantics(loaded)
        baselines: list[tuple[Path, dict[str, Any]]] = []
        for path in dict.fromkeys(
            item for item in (destination, baseline_path) if item is not None
        ):
            if not path.exists():
                continue
            try:
                candidate = load_index(path)
                validate_index_semantics(candidate)
                baselines.append((path, candidate))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        baseline = (
            max(baselines, key=lambda item: index_generated_at(item[1]))
            if baselines
            else None
        )
        if baseline is not None:
            old_meta = baseline[1]["$meta"]
            new_meta = loaded["$meta"]
            same_semantic_index = (
                new_meta.get("generated_at") == old_meta.get("generated_at")
                and sha256_hex(loaded["profiles"])
                == sha256_hex(baseline[1]["profiles"])
                and new_meta.get("profile_count") == old_meta.get("profile_count")
            )
            if same_semantic_index:
                return loaded
            validate_index_semantics(loaded, baseline=baseline[1])
        backup = destination.with_suffix(destination.suffix + ".bak")
        if destination.exists():
            shutil.copy2(destination, backup)
        os.replace(temp_path, destination)
        return loaded
    finally:
        if temp_path.exists():
            temp_path.unlink()
