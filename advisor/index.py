from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ResourceProfile

INDEX_SCHEMA_VERSION = 1
MAX_INDEX_BYTES = 32 * 1024 * 1024
MIN_REMOTE_COVERAGE_RATIO = 0.90


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_index(path: Path, *, max_bytes: int = MAX_INDEX_BYTES) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"resource index exceeds {max_bytes} bytes")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("$meta"), dict):
        raise ValueError("invalid resource index root")
    if int(raw["$meta"].get("schema_version") or 0) != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported resource index schema")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("resource index profiles must be an object")
    expected = raw["$meta"].get("profiles_sha256")
    if expected and expected != sha256_hex(profiles):
        raise ValueError("resource index checksum mismatch")
    return raw


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def index_generated_at(index: dict[str, Any]) -> datetime:
    meta = index.get("$meta")
    if not isinstance(meta, dict):
        raise ValueError("invalid resource index metadata")
    return _parse_timestamp(meta.get("generated_at"), field="generated_at")


def validate_index_semantics(
    index: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    minimum_profiles: int = 1,
) -> None:
    """Validate signed-index meaning, not just its JSON checksum.

    This is intentionally dependency-free so it can run inside AstrBot before
    replacing the last known-good index.
    """
    profiles = index.get("profiles")
    meta = index.get("$meta")
    if not isinstance(profiles, dict) or not isinstance(meta, dict):
        raise ValueError("invalid resource index structure")
    if len(profiles) < max(1, int(minimum_profiles)):
        raise ValueError("remote resource index is empty or below minimum coverage")
    if meta.get("profile_count") != len(profiles):
        raise ValueError("profile_count does not match profiles object")
    source_downloaded = meta.get("source_code_downloaded")
    if source_downloaded is True:
        if (
            meta.get("scan_mode")
            != "local_source_static_read_only_with_metadata_fallback"
            or meta.get("plugin_code_executed") is not False
            or meta.get("network_used") is not False
        ):
            raise ValueError("source-derived index must declare read-only, offline, non-executing scan mode")
    elif source_downloaded is not False:
        raise ValueError("source_code_downloaded must be a boolean")
    generated = _parse_timestamp(meta.get("generated_at"), field="generated_at")

    if baseline is not None:
        old_profiles = baseline.get("profiles")
        old_meta = baseline.get("$meta")
        if isinstance(old_profiles, dict) and old_profiles:
            minimum_coverage = max(
                1, int(len(old_profiles) * MIN_REMOTE_COVERAGE_RATIO)
            )
            if len(profiles) < minimum_coverage:
                raise ValueError("remote resource index coverage dropped significantly")
        if isinstance(old_meta, dict) and old_meta.get("generated_at"):
            old_generated = _parse_timestamp(
                old_meta.get("generated_at"), field="baseline generated_at"
            )
            if generated <= old_generated:
                raise ValueError("remote resource index is stale or replayed")

    dimensions = {
        "idle_memory",
        "peak_memory",
        "idle_cpu",
        "peak_cpu",
        "disk",
        "network",
    }
    required = {
        "plugin_id",
        "version",
        "commit_sha",
        "levels",
        "scores",
        "features",
        "external_processes",
        "background_tasks",
        "evidence",
        "unknowns",
        "confidence",
        "evidence_level",
        "scanned_at",
    }
    github_count = 0
    for plugin_id, profile in profiles.items():
        if not isinstance(plugin_id, str) or not plugin_id:
            raise ValueError("profile key must be a non-empty string")
        if not isinstance(profile, dict) or set(profile) != required:
            raise ValueError(f"invalid profile fields for {plugin_id}")
        if profile.get("plugin_id") != plugin_id:
            raise ValueError(f"profile key/id mismatch for {plugin_id}")
        levels = profile.get("levels")
        scores = profile.get("scores")
        if not isinstance(levels, dict) or set(levels) != dimensions:
            raise ValueError(f"invalid levels for {plugin_id}")
        if not isinstance(scores, dict) or set(scores) != dimensions:
            raise ValueError(f"invalid scores for {plugin_id}")
        for dimension in dimensions:
            score = scores[dimension]
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= 4
            ):
                raise ValueError(f"invalid score for {plugin_id}:{dimension}")
            if levels[dimension] != f"L{score}":
                raise ValueError(f"level/score mismatch for {plugin_id}:{dimension}")
        for array_field in ("features", "external_processes", "evidence", "unknowns"):
            value = profile[array_field]
            if (
                not isinstance(value, list)
                or len(value) > 100
                or any(not isinstance(item, str) or len(item) > 2048 for item in value)
            ):
                raise ValueError(f"invalid {array_field} for {plugin_id}")
        if profile["background_tasks"] not in {"yes", "likely", "no", "unknown"}:
            raise ValueError(f"invalid background_tasks for {plugin_id}")
        confidence = profile["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"invalid confidence for {plugin_id}")
        _parse_timestamp(profile["scanned_at"], field=f"scanned_at for {plugin_id}")
        evidence_level = profile["evidence_level"]
        if not isinstance(evidence_level, str) or not evidence_level:
            raise ValueError(f"invalid evidence_level for {plugin_id}")
        if evidence_level.startswith("github_"):
            github_count += 1
            if not re.fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}", str(profile["commit_sha"])
            ):
                raise ValueError(f"invalid GitHub commit for {plugin_id}")
    if github_count:
        if meta.get("commit_sha_kind") != "github_commit_oid":
            raise ValueError("GitHub profiles require commit_sha_kind")
        if meta.get("commit_binding_api") != "github_list_commits_metadata":
            raise ValueError("GitHub profiles require commit_binding_api")


def get_profile(index: dict[str, Any], plugin_id: str) -> ResourceProfile | None:
    raw = index.get("profiles", {}).get(plugin_id)
    if not isinstance(raw, dict):
        return None
    return ResourceProfile.from_dict(raw)


def profile_is_current(
    profile: ResourceProfile,
    *,
    version: str,
    commit_sha: str = "",
    record_updated_at: str = "",
) -> bool:
    if profile.version != version:
        return False
    if commit_sha and profile.commit_sha != commit_sha:
        return False
    if record_updated_at and profile.scanned_at:
        try:
            record_time = datetime.fromisoformat(
                record_updated_at.replace("Z", "+00:00")
            )
            scan_time = datetime.fromisoformat(
                profile.scanned_at.replace("Z", "+00:00")
            )
            if record_time.tzinfo is None:
                record_time = record_time.replace(tzinfo=UTC)
            if scan_time.tzinfo is None:
                scan_time = scan_time.replace(tzinfo=UTC)
            if record_time.astimezone(UTC) > scan_time.astimezone(UTC):
                return False
        except ValueError:
            # Malformed market timestamps must not invalidate an otherwise bound profile.
            pass
    return True
