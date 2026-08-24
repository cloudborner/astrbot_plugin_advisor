from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.index import load_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an advisor resource index")
    parser.add_argument(
        "index", nargs="?", type=Path, default=ROOT / "data" / "resource_profiles.json"
    )
    parser.add_argument("--minimum-profiles", type=int, default=1)
    parser.add_argument("--minimum-github-ratio", type=float, default=0.0)
    parser.add_argument(
        "--market",
        type=Path,
        default=ROOT / "data" / "market_snapshot.json",
        help="market snapshot used to verify plugin_id/version binding",
    )
    return parser.parse_args()


def validate_quality(
    index: dict,
    *,
    market_plugins: dict | None = None,
    minimum_profiles: int = 1,
    minimum_github_ratio: float = 0.0,
) -> dict:
    profiles = index["profiles"]
    if len(profiles) < minimum_profiles:
        raise ValueError(
            f"only {len(profiles)} profiles; expected at least {minimum_profiles}"
        )
    if index["$meta"].get("profile_count") != len(profiles):
        raise ValueError("profile_count does not match profiles object")
    evidence = Counter()
    peak_memory = Counter()
    invalid_ids = []
    missing_commits = []
    missing_market = []
    version_mismatches = []
    for plugin_id, profile in profiles.items():
        if profile.get("plugin_id") != plugin_id:
            invalid_ids.append(plugin_id)
        evidence_level = str(profile.get("evidence_level") or "unknown")
        evidence[evidence_level] += 1
        peak_memory[str(profile.get("levels", {}).get("peak_memory") or "unknown")] += 1
        if evidence_level.startswith("github_") and not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", str(profile.get("commit_sha") or "")
        ):
            missing_commits.append(plugin_id)
        if market_plugins is not None:
            market_record = market_plugins.get(plugin_id)
            if not isinstance(market_record, dict):
                missing_market.append(plugin_id)
            elif str(profile.get("version") or "") != str(
                market_record.get("version") or ""
            ):
                version_mismatches.append(plugin_id)
    if invalid_ids:
        raise ValueError(f"profile key/id mismatch: {invalid_ids[:3]}")
    if missing_commits:
        raise ValueError(
            f"GitHub evidence without a valid commit SHA: {missing_commits[:3]}"
        )
    if missing_market:
        raise ValueError(f"profiles missing from market snapshot: {missing_market[:3]}")
    if version_mismatches:
        raise ValueError(f"profile/market version mismatch: {version_mismatches[:3]}")
    if market_plugins is not None:
        missing_profiles = sorted(set(market_plugins) - set(profiles))
        if missing_profiles:
            raise ValueError(f"market plugins missing profiles: {missing_profiles[:3]}")
    source_downloaded = index["$meta"].get("source_code_downloaded")
    if source_downloaded is True:
        if (
            index["$meta"].get("scan_mode")
            != "local_source_static_read_only_with_metadata_fallback"
            or index["$meta"].get("plugin_code_executed") is not False
            or index["$meta"].get("network_used") is not False
        ):
            raise ValueError("source-derived index lacks read-only scan guarantees")
    elif source_downloaded is not False:
        raise ValueError("source_code_downloaded must be a boolean")
    declared_evidence = index["$meta"].get("evidence_counts")
    if declared_evidence is not None and declared_evidence != dict(evidence):
        raise ValueError("evidence_counts does not match profile evidence")
    github_count = sum(
        count for name, count in evidence.items() if name.startswith("github_")
    )
    if github_count and index["$meta"].get("commit_sha_kind") != "github_commit_oid":
        raise ValueError(
            "GitHub evidence must declare commit_sha_kind=github_commit_oid"
        )
    if (
        github_count
        and index["$meta"].get("commit_binding_api") != "github_list_commits_metadata"
    ):
        raise ValueError(
            "GitHub evidence must declare commit_binding_api=github_list_commits_metadata"
        )
    github_ratio = github_count / len(profiles)
    if github_ratio < minimum_github_ratio:
        raise ValueError(
            f"GitHub evidence ratio {github_ratio:.3f} below {minimum_github_ratio:.3f}"
        )
    return {
        "profiles": len(profiles),
        "scan_mode": index["$meta"].get("scan_mode"),
        "github_ratio": round(github_ratio, 4),
        "github_commit_bound": github_count,
        "market_version_bound": len(profiles) if market_plugins is not None else None,
        "evidence": dict(sorted(evidence.items())),
        "peak_memory": dict(sorted(peak_memory.items())),
        "source_code_downloaded": index["$meta"].get("source_code_downloaded"),
    }


def main() -> int:
    args = parse_args()
    try:
        index = load_index(args.index)
        market_raw = json.loads(args.market.read_text(encoding="utf-8"))
        market_plugins = (
            market_raw.get("plugins") if isinstance(market_raw, dict) else None
        )
        if not isinstance(market_plugins, dict):
            raise ValueError("market snapshot plugins must be an object")
        result = validate_quality(
            index,
            market_plugins=market_plugins,
            minimum_profiles=max(0, args.minimum_profiles),
            minimum_github_ratio=max(0.0, min(1.0, args.minimum_github_ratio)),
        )
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
