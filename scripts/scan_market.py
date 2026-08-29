from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.index import atomic_write_json, sha256_hex  # noqa: E402
from advisor.market import (  # noqa: E402
    DEFAULT_MARKET_URL,
    GitHubClient,
    GitHubObservation,
    load_market,
)
from advisor.resource_rules import build_resource_profile, load_rules  # noqa: E402
from scripts.build_capability_index import (  # noqa: E402
    build_document,
    write_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AstrBot plugin resource profiles"
    )
    parser.add_argument("--market-url", default=DEFAULT_MARKET_URL)
    parser.add_argument(
        "--rules", type=Path, default=ROOT / "data" / "resource_rules.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "resource_profiles.json"
    )
    parser.add_argument(
        "--market-output", type=Path, default=ROOT / "data" / "market_snapshot.json"
    )
    parser.add_argument(
        "--capability-output",
        type=Path,
        default=ROOT / "data" / "plugin_capabilities.json",
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / ".cache" / "github_observations.json"
    )
    parser.add_argument("--mode", choices=("market", "github"), default="market")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-interval", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-unauthenticated", action="store_true")
    return parser.parse_args()


def observation_to_dict(value: GitHubObservation) -> dict[str, Any]:
    return {
        "commit_sha": value.commit_sha,
        "commit_ok": value.commit_ok,
        "commit_api": value.commit_api,
        "tree_sha": value.tree_sha,
        "tree": value.tree,
        "packages": value.packages,
        "tree_ok": value.tree_ok,
        "sbom_ok": value.sbom_ok,
        "errors": value.errors,
        "rate_remaining": value.rate_remaining,
    }


def observation_from_dict(value: dict[str, Any]) -> GitHubObservation:
    return GitHubObservation(
        commit_sha=str(value.get("commit_sha") or ""),
        tree=[x for x in value.get("tree") or [] if isinstance(x, dict)],
        packages=[str(x) for x in value.get("packages") or []],
        tree_ok=bool(value.get("tree_ok")),
        sbom_ok=bool(value.get("sbom_ok")),
        errors=[str(x) for x in value.get("errors") or []],
        rate_remaining=value.get("rate_remaining"),
        tree_sha=(
            str(value.get("tree_sha") or "")
            if str(value.get("tree_sha") or "") != str(value.get("commit_sha") or "")
            else ""
        ),
        commit_ok=bool(value.get("commit_ok")),
        commit_api=str(value.get("commit_api") or ""),
    )


def estimated_observation_requests(cached: GitHubObservation | None) -> int:
    if cached is None:
        return 3
    count = int(
        not cached.commit_ok or cached.commit_api != "list_commits_metadata"
    ) + int(not cached.tree_ok)
    permanent_sbom_miss = any(error.startswith("sbom:404") for error in cached.errors)
    if not cached.sbom_ok and not permanent_sbom_miss:
        count += 1
    return count


def mandatory_observation_requests(cached: GitHubObservation | None) -> int:
    """Requests required to bind a profile to a commit and inspect its tree."""
    if cached is None:
        return 2
    return int(
        not cached.commit_ok or cached.commit_api != "list_commits_metadata"
    ) + int(not cached.tree_ok)


def select_checkpoint_batch(
    pending: list[tuple[Any, GitHubObservation | None]],
    *,
    remaining: int,
    reserve: int = 50,
) -> tuple[list[tuple[Any, GitHubObservation | None]], int]:
    """Select a deterministic mandatory phase that fits the current quota.

    Zero-cost cached rows are included.  The caller persists each result but
    must never publish a partial index when ``deferred`` is non-zero.
    """
    budget = max(0, int(remaining) - max(0, int(reserve)))
    selected: list[tuple[Any, GitHubObservation | None]] = []
    deferred = 0
    for item in pending:
        cost = mandatory_observation_requests(item[1])
        if cost <= budget:
            selected.append(item)
            budget -= cost
        else:
            deferred += 1
    return selected, deferred


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    args = parse_args()
    metadata, records = load_market(args.market_url)
    records.sort(key=lambda item: item.plugin_id.lower())
    if args.limit > 0:
        records = records[: args.limit]
    rules = load_rules(args.rules)
    client = GitHubClient(
        token=os.getenv("GITHUB_TOKEN", ""),
        min_interval=args.min_interval,
    )
    if (
        args.mode == "github"
        and not client.authenticated
        and not args.allow_unauthenticated
    ):
        print(
            "GitHub mode requires GITHUB_TOKEN or --allow-unauthenticated",
            file=sys.stderr,
        )
        return 2

    cache = load_cache(args.cache)
    observations: dict[str, GitHubObservation] = {}
    pending: list[tuple[Any, GitHubObservation | None]] = []
    if args.mode == "github":
        for record in records:
            cached = cache.get(record.plugin_id)
            if (
                isinstance(cached, dict)
                and cached.get("version") == record.version
                and cached.get("repo") == record.repo
                and cached.get("updated_at") == record.updated_at
                and isinstance(cached.get("observation"), dict)
            ):
                cached_observation = observation_from_dict(cached["observation"])
                transient = any(
                    error.startswith(
                        ("tree:403", "tree:429", "sbom:403", "sbom:429", "fatal:")
                    )
                    for error in cached_observation.errors
                )
                if (
                    cached_observation.tree_ok
                    and cached_observation.commit_ok
                    and cached_observation.commit_api == "list_commits_metadata"
                    and not transient
                ):
                    observations[record.plugin_id] = cached_observation
                else:
                    pending.append((record, cached_observation))
            else:
                pending.append((record, None))

        rate = client.rate_status()
        estimated_requests = (
            sum(estimated_observation_requests(cached) for _record, cached in pending)
            + 50
        )
        mandatory_requests = (
            sum(mandatory_observation_requests(cached) for _record, cached in pending)
            + 50
        )
        print(
            json.dumps(
                {
                    "github_rate_limit": rate["limit"],
                    "github_rate_remaining": rate["remaining"],
                    "estimated_requests": estimated_requests,
                    "mandatory_requests": mandatory_requests,
                    "cached_current": len(observations),
                    "pending": len(pending),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        if rate["limit"] < 5000:
            print(
                "GitHub did not accept the token as authenticated (core limit is below 5000)",
                file=sys.stderr,
                flush=True,
            )
            return 4
        mandatory_batch, deferred = select_checkpoint_batch(
            pending, remaining=rate["remaining"]
        )

        workers = max(1, min(4, args.workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    client.observe,
                    record.repo,
                    cached,
                    include_sbom=False,
                ): (record, cached)
                for record, cached in mandatory_batch
            }
            for index, future in enumerate(as_completed(futures), start=1):
                record, cached = futures[future]
                try:
                    observation = future.result()
                except Exception as exc:  # one repository must not abort the market
                    observation = cached or GitHubObservation(
                        "", [], [], False, False, []
                    )
                    observation.commit_sha = ""
                    observation.commit_ok = False
                    observation.commit_api = ""
                    observation.errors.append(f"fatal:{type(exc).__name__}")
                observations[record.plugin_id] = observation
                cache[record.plugin_id] = {
                    "version": record.version,
                    "repo": record.repo,
                    "updated_at": record.updated_at,
                    "observation": observation_to_dict(observation),
                }
                if index % 25 == 0:
                    atomic_write_json(args.cache, cache)
                    print(
                        f"mandatory scanned {index}/{len(mandatory_batch)}",
                        file=sys.stderr,
                        flush=True,
                    )
        atomic_write_json(args.cache, cache)
        if deferred:
            print(
                f"Checkpoint saved; {deferred} repositories deferred until quota reset. "
                "No index was overwritten.",
                file=sys.stderr,
                flush=True,
            )
            return 5

        # SBOM is valuable but optional static evidence.  A fresh classic PAT
        # cannot fit 3 requests for every market plugin, so consume only the
        # post-tree quota while keeping a 50-request safety reserve.
        sbom_candidates = []
        for record in records:
            observation = observations.get(record.plugin_id)
            if observation is None or observation.sbom_ok:
                continue
            if any(error.startswith("sbom:404") for error in observation.errors):
                continue
            sbom_candidates.append((record, observation))
        post_rate = client.rate_status()
        sbom_budget = max(0, int(post_rate["remaining"]) - 50)
        sbom_batch = sbom_candidates[:sbom_budget]
        for index, (record, observation) in enumerate(sbom_batch, start=1):
            try:
                refreshed = client.observe(record.repo, observation, include_sbom=True)
            except Exception as exc:
                refreshed = observation
                refreshed.errors.append(f"sbom:fatal:{type(exc).__name__}")
            observations[record.plugin_id] = refreshed
            cache[record.plugin_id] = {
                "version": record.version,
                "repo": record.repo,
                "updated_at": record.updated_at,
                "observation": observation_to_dict(refreshed),
            }
            if index % 25 == 0:
                atomic_write_json(args.cache, cache)
                print(
                    f"optional SBOM scanned {index}/{len(sbom_batch)}",
                    file=sys.stderr,
                    flush=True,
                )
        atomic_write_json(args.cache, cache)

    profiles = {}
    evidence_counts: dict[str, int] = {}
    for record in records:
        profile = build_resource_profile(
            record, rules, observations.get(record.plugin_id)
        )
        profiles[record.plugin_id] = profile.to_dict()
        evidence_counts[profile.evidence_level] = (
            evidence_counts.get(profile.evidence_level, 0) + 1
        )

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    invalid_commits = [
        plugin_id
        for plugin_id, profile in profiles.items()
        if str(profile.get("evidence_level") or "").startswith("github_")
        and not GitHubClient._valid_object_id(str(profile.get("commit_sha") or ""))
    ]
    if invalid_commits:
        print(
            f"GitHub evidence lacks verified commit SHA for {invalid_commits[:5]}; "
            "no index was overwritten",
            file=sys.stderr,
        )
        return 6
    index = {
        "$meta": {
            "schema_version": 1,
            "generated_at": generated_at,
            "market_url": args.market_url,
            "market_schema_version": metadata.get("schema_version"),
            "market_version": metadata.get("version", ""),
            "profile_count": len(profiles),
            "scan_mode": args.mode,
            "commit_sha_kind": "github_commit_oid"
            if args.mode == "github"
            else "unavailable",
            "commit_binding_api": (
                "github_list_commits_metadata"
                if args.mode == "github"
                else "unavailable"
            ),
            "evidence_counts": evidence_counts,
            "profiles_sha256": sha256_hex(profiles),
            "source_code_downloaded": False,
        },
        "profiles": profiles,
    }
    market_snapshot = {
        "$meta": {
            "schema_version": 1,
            "generated_at": generated_at,
            "source": args.market_url,
            "market_version": metadata.get("version", ""),
        },
        "plugins": {record.plugin_id: record.to_dict() for record in records},
    }
    atomic_write_json(args.market_output, market_snapshot)
    atomic_write_json(args.output, index)
    capability_document = build_document(args.market_output)
    write_document(capability_document, args.capability_output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "market_output": str(args.market_output),
                "capability_output": str(args.capability_output),
                "capability_profiles": capability_document["$meta"]["profile_count"],
                "profiles": len(profiles),
                "evidence": evidence_counts,
                "sha256": index["$meta"]["profiles_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
