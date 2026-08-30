from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.capabilities import CapabilityIndex  # noqa: E402


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_document(
    document: dict[str, Any],
    market_plugins: dict[str, Any],
    *,
    require_source_count: int = 0,
) -> dict[str, Any]:
    parsed = CapabilityIndex.from_dict(document)
    profiles = document.get("profiles")
    meta = document.get("$meta")
    if not isinstance(profiles, dict) or not isinstance(meta, dict):
        raise ValueError("capability index has an invalid root schema")
    if set(profiles) != set(market_plugins):
        missing = sorted(set(market_plugins) - set(profiles))
        extra = sorted(set(profiles) - set(market_plugins))
        raise ValueError(f"market coverage mismatch; missing={missing[:3]} extra={extra[:3]}")
    version_mismatches = [
        plugin_id
        for plugin_id, profile in profiles.items()
        if str(profile.get("version") or "")
        != str((market_plugins.get(plugin_id) or {}).get("version") or "")
    ]
    if version_mismatches:
        raise ValueError(f"profile/market version mismatch: {version_mismatches[:3]}")
    canonical = json.dumps(
        profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if meta.get("profiles_sha256") != digest:
        raise ValueError("profiles_sha256 does not match profiles")
    source_names = {
        "source_readme",
        "source_commands",
        "source_config_schema",
        "source_resource_static",
    }
    source_profiles = sum(
        bool(source_names.intersection(str(item) for item in profile.get("sources", [])))
        for profile in profiles.values()
        if isinstance(profile, dict)
    )
    if source_profiles < max(0, require_source_count):
        raise ValueError(
            f"only {source_profiles} source-derived profiles; expected at least {require_source_count}"
        )
    if bool(meta.get("source_code_downloaded")) != bool(source_profiles):
        raise ValueError("source_code_downloaded does not match source-derived profiles")
    if source_profiles and meta.get("plugin_code_executed") is not False:
        raise ValueError("source-derived index lacks the no-execution guarantee")
    source_counts = Counter(
        source
        for profile in parsed.profiles.values()
        for source in profile.sources
        if source.startswith("source_")
    )
    empty_aliases = sum(not profile.aliases for profile in parsed.profiles.values())
    return {
        "profiles": len(parsed.profiles),
        "source_profiles": source_profiles,
        "source_evidence": dict(sorted(source_counts.items())),
        "profiles_without_aliases": empty_aliases,
        "sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the plugin capability index")
    parser.add_argument(
        "index", nargs="?", type=Path, default=ROOT / "data" / "plugin_capabilities.json"
    )
    parser.add_argument(
        "--market", type=Path, default=ROOT / "data" / "market_snapshot.json"
    )
    parser.add_argument("--require-source-count", type=int, default=0)
    args = parser.parse_args()
    try:
        document = load_object(args.index)
        market = load_object(args.market).get("plugins")
        if not isinstance(market, dict):
            raise ValueError("market snapshot plugins must be an object")
        result = validate_document(
            document, market, require_source_count=max(0, args.require_source_count)
        )
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
