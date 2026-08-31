#!/usr/bin/env python3
"""Download, safely scan, merge, and validate the complete plugin market."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.capabilities import CapabilityIndex  # noqa: E402
from advisor.index import atomic_write_json, load_index  # noqa: E402
from scripts import download_sources  # noqa: E402
from scripts.analyze_extracted_sources import (  # noqa: E402
    load_json,
    runtime_index,
)
from scripts.analyze_extracted_sources import (  # noqa: E402
    run as analyze_resources,
)
from scripts.audit_semantic_profiles import audit_documents  # noqa: E402
from scripts.build_capability_index import (  # noqa: E402
    build_document as build_capability_document,
)
from scripts.build_capability_index import (  # noqa: E402
    write_document as write_capability_document,
)
from scripts.download_sources import (  # noqa: E402
    DownloadItem,
    archive_name,
    build_items,
    download_one,
)
from scripts.extract_plugin_function_evidence import (  # noqa: E402
    build_document as build_function_document,
)
from scripts.extract_sources import extract_archive  # noqa: E402
from scripts.validate_capability_index import validate_document  # noqa: E402
from scripts.validate_index import validate_quality  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the market, resume source downloads, safely extract, delete archives, "
            "scan functions/resources, build indexes, and validate all outputs"
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", action="store_true", help="only calculate the download plan")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="development-only item limit")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-archive-mib", type=int, default=512)
    parser.add_argument("--max-plugin-mib", type=int, default=1024)
    parser.add_argument("--max-total-gib", type=int, default=24)
    parser.add_argument("--minimum-free-gib", type=int, default=5)
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("ADVISOR_SOURCE_PROXY", ""),
        help="optional explicit HTTP(S) proxy; system proxy settings are otherwise ignored",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep successfully extracted archives (default deletes them)",
    )
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def configure_proxy(proxy_url: str) -> None:
    if not proxy_url:
        return
    parsed = urllib.parse.urlparse(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--proxy-url must use http or https")
    download_sources.DIRECT_OPENER = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    )


def plan_document(items: list[DownloadItem], market_count: int) -> dict[str, Any]:
    fixed = sum(item.ref_kind == "commit" for item in items)
    return {
        "$meta": {
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "market_record_count": market_count,
            "public_repository_count": len(items),
            "fixed_commit_count": fixed,
            "default_branch_count": len(items) - fixed,
        },
        "plugins": {
            item.plugin_id: {
                "plugin_id": item.plugin_id,
                "display_name": item.display_name,
                "repo": item.repo,
                "ref": item.ref,
                "ref_kind": item.ref_kind,
                "archive_name": item.archive_name,
                "archive_url": item.archive_url,
            }
            for item in items
        },
    }


def safe_delete_archive(path: Path, archive_root: Path) -> None:
    resolved = path.resolve()
    root = archive_root.resolve()
    if resolved.parent != root or not resolved.name.endswith(".tar.gz"):
        raise ValueError(f"refusing to delete archive outside the archive root: {resolved}")
    resolved.unlink()


def acquire_one(
    item: DownloadItem,
    archive_root: Path,
    source_root: Path,
    *,
    retries: int,
    max_archive_bytes: int,
    max_plugin_bytes: int,
    keep_archives: bool,
) -> dict[str, Any]:
    effective_item = item
    result = download_one(
        effective_item,
        archive_root,
        None,
        max_archive_bytes,
        retries,
    )
    if (
        result.status == "failed"
        and item.ref_kind == "commit"
        and "404" in result.error
    ):
        effective_item = DownloadItem(
            plugin_id=item.plugin_id,
            display_name=item.display_name,
            repo=item.repo,
            owner=item.owner,
            name=item.name,
            ref="HEAD",
            ref_kind="default_branch_fallback",
            archive_name=archive_name(item.owner, item.name, "HEAD"),
            archive_url=(
                f"https://codeload.github.com/{item.owner}/{item.name}/tar.gz/HEAD"
            ),
        )
        result = download_one(
            effective_item,
            archive_root,
            None,
            max_archive_bytes,
            retries,
        )
    archive = archive_root / effective_item.archive_name
    directory = effective_item.archive_name.removesuffix(".tar.gz")
    destination = source_root / directory
    record = asdict(result)
    record["directory"] = directory
    record["repo"] = item.repo
    record["planned_ref"] = item.ref
    record["used_default_branch_fallback"] = effective_item is not item
    if result.status not in {"complete", "skipped_existing"}:
        record["stage"] = "download"
        return record
    try:
        extracted = extract_archive(archive, destination, max_bytes=max_plugin_bytes)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "stage": "extract",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return record
    record.update(extracted)
    record["stage"] = "complete"
    record["download_size_bytes"] = result.size_bytes
    record["download_sha256"] = result.sha256
    record["archive_deleted"] = False
    if not keep_archives:
        safe_delete_archive(archive, archive_root)
        record["archive_deleted"] = True
    return record


def acquire_sources(
    items: list[DownloadItem],
    archive_root: Path,
    source_root: Path,
    manifest_path: Path,
    *,
    workers: int,
    retries: int,
    max_archive_bytes: int,
    max_plugin_bytes: int,
    max_total_bytes: int,
    minimum_free_bytes: int,
    keep_archives: bool,
) -> dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    old = load_json(manifest_path, {})
    records = old.get("plugins", {}) if isinstance(old, dict) else {}
    records = dict(records) if isinstance(records, dict) else {}
    item_ids = {item.plugin_id for item in items}
    records = {key: value for key, value in records.items() if key in item_ids}
    pending: list[DownloadItem] = []
    for item in items:
        record = records.get(item.plugin_id)
        directory = str(record.get("directory") or "") if isinstance(record, dict) else ""
        if (
            isinstance(record, dict)
            and record.get("status") == "complete"
            and (
                record.get("ref") == item.ref
                or record.get("planned_ref") == item.ref
            )
            and directory
            and (source_root / directory).is_dir()
        ):
            continue
        pending.append(item)
    total_expanded = sum(
        int(record.get("expanded_bytes") or 0)
        for record in records.values()
        if isinstance(record, dict) and record.get("status") == "complete"
    )

    def write_manifest() -> None:
        counts: dict[str, int] = {}
        for value in records.values():
            status = str(value.get("status") or "unknown") if isinstance(value, dict) else "invalid"
            counts[status] = counts.get(status, 0) + 1
        atomic_write_json(
            manifest_path,
            {
                "$meta": {
                    "schema_version": 1,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_code_executed": False,
                    "archive_deleted_after_success": not keep_archives,
                    "planned_count": len(items),
                    "counts": counts,
                    "expanded_bytes": total_expanded,
                },
                "plugins": records,
            },
        )

    print(
        json.dumps(
            {
                "stage": "acquire",
                "planned": len(items),
                "resumed_complete": len(items) - len(pending),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    completed_this_run = 0
    failed_this_run = 0
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[dict[str, Any]], DownloadItem] = {}

        def submit_next() -> bool:
            if total_expanded >= max_total_bytes:
                return False
            if shutil.disk_usage(source_root).free < minimum_free_bytes:
                return False
            try:
                item = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(
                acquire_one,
                item,
                archive_root,
                source_root,
                retries=retries,
                max_archive_bytes=max_archive_bytes,
                max_plugin_bytes=max_plugin_bytes,
                keep_archives=keep_archives,
            )
            futures[future] = item
            return True

        for _ in range(workers):
            submit_next()
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                item = futures.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # one repository must not abort the manifest
                    record = {
                        "plugin_id": item.plugin_id,
                        "repo": item.repo,
                        "ref": item.ref,
                        "ref_kind": item.ref_kind,
                        "archive_name": item.archive_name,
                        "directory": item.archive_name.removesuffix(".tar.gz"),
                        "status": "failed",
                        "stage": "worker",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                previous = records.get(item.plugin_id)
                previous_bytes = (
                    int(previous.get("expanded_bytes") or 0)
                    if isinstance(previous, dict) and previous.get("status") == "complete"
                    else 0
                )
                records[item.plugin_id] = record
                if record.get("status") == "complete":
                    total_expanded += int(record.get("expanded_bytes") or 0) - previous_bytes
                    completed_this_run += 1
                else:
                    failed_this_run += 1
                write_manifest()
                done = len(items) - len(pending) + completed_this_run + failed_this_run
                print(
                    json.dumps(
                        {
                            "stage": "acquire",
                            "done": done,
                            "total": len(items),
                            "plugin_id": item.plugin_id,
                            "status": record.get("status"),
                            "failure_stage": record.get("stage")
                            if record.get("status") != "complete"
                            else "",
                            "expanded_mib": record.get("expanded_mib", 0),
                            "error": record.get("error", ""),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                submit_next()
    write_manifest()
    complete = sum(
        isinstance(value, dict)
        and value.get("status") == "complete"
        and (source_root / str(value.get("directory") or "")).is_dir()
        for value in records.values()
    )
    return {
        "planned": len(items),
        "complete": complete,
        "failed": len(items) - complete,
        "expanded_bytes": total_expanded,
        "manifest": records,
    }


def write_resource_outputs(root: Path, source_root: Path) -> dict[str, Any]:
    args = type(
        "ResourceArgs",
        (),
        {
            "root": str(root),
            "source": str(source_root),
            "profiles": str(root / "data" / "source_resource_profiles.json"),
            "queue": str(root / "data" / "source_resource_review_queue.json"),
        },
    )()
    profiles, queue = analyze_resources(args)
    fallback = load_json(root / "data" / "resource_profiles.json", {})
    runtime = runtime_index(profiles, fallback)
    atomic_write_json(root / "data" / "source_resource_profiles.json", profiles)
    atomic_write_json(root / "data" / "source_resource_review_queue.json", queue)
    atomic_write_json(root / "data" / "source_resource_index.json", runtime)
    return {"profiles": profiles, "runtime": runtime, "queue": queue}


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16")
    root = args.root.resolve()
    market_path = root / "data" / "market_snapshot.json"
    resource_fallback_path = root / "data" / "resource_profiles.json"
    market = load_object(market_path)
    market_plugins = market.get("plugins")
    if not isinstance(market_plugins, dict):
        raise SystemExit("market snapshot plugins must be an object")
    fallback = load_object(resource_fallback_path)
    items = build_items(fallback, market)
    if args.limit > 0:
        items = items[: args.limit]
    archive_root = root / "source_archives"
    source_root = root / "source_extracted"
    plan_path = archive_root / "download_plan.json"
    manifest_path = source_root / "pipeline_manifest.json"
    archive_root.mkdir(parents=True, exist_ok=True)
    plan = plan_document(items, len(market_plugins))
    atomic_write_json(plan_path, plan)
    print(json.dumps(plan["$meta"], ensure_ascii=False), flush=True)
    if args.plan:
        return 0
    if args.limit:
        raise SystemExit("--limit is only supported with --plan; full outputs require the complete market")
    configure_proxy(args.proxy_url)
    acquisition = acquire_sources(
        items,
        archive_root,
        source_root,
        manifest_path,
        workers=args.workers,
        retries=max(0, args.retries),
        max_archive_bytes=max(1, args.max_archive_mib) * 1024 * 1024,
        max_plugin_bytes=max(1, args.max_plugin_mib) * 1024 * 1024,
        max_total_bytes=max(1, args.max_total_gib) * 1024 * 1024 * 1024,
        minimum_free_bytes=max(1, args.minimum_free_gib) * 1024 * 1024 * 1024,
        keep_archives=args.keep_archives,
    )
    print(json.dumps({"stage": "resources", "status": "started"}), flush=True)
    resource_outputs = write_resource_outputs(root, source_root)
    print(json.dumps({"stage": "functions", "status": "started"}), flush=True)
    function_document = build_function_document(
        market_path,
        source_root,
        manifest_path,
        root / "data" / "source_resource_profiles.json",
    )
    function_path = root / "data" / "source_function_evidence.json"
    atomic_write_json(function_path, function_document)
    semantic_path = root / "data" / "source_function_llm_profiles_v3_reviewed.json"
    semantic_quality: dict[str, Any] | None = None
    if semantic_path.exists():
        semantic_quality = audit_documents(load_object(semantic_path), function_document)
        atomic_write_json(
            root / "artifacts" / "semantic_profile_quality_report.json",
            semantic_quality,
        )
    print(json.dumps({"stage": "capability_index", "status": "started"}), flush=True)
    capability_document = build_capability_document(
        market_path,
        root / "data" / "plugin_taxonomy.json",
        function_path,
        semantic_path,
    )
    capability_path = root / "data" / "plugin_capabilities.json"
    write_capability_document(capability_document, capability_path)
    CapabilityIndex.from_file(capability_path)
    resource_index = load_index(root / "data" / "source_resource_index.json")
    resource_validation = validate_quality(
        resource_index,
        market_plugins=market_plugins,
        minimum_profiles=len(market_plugins),
    )
    capability_validation = validate_document(
        capability_document,
        market_plugins,
        require_source_count=acquisition["complete"],
    )
    summary = {
        "stage": "complete",
        "market_records": len(market_plugins),
        "planned_repositories": len(items),
        "downloaded_and_extracted": acquisition["complete"],
        "failed_repositories": acquisition["failed"],
        "archives_deleted_after_success": not args.keep_archives,
        "resource_profiles": resource_outputs["profiles"]["$meta"]["profile_count"],
        "function_profiles": function_document["$meta"]["profile_count"],
        "capability_profiles": capability_validation["profiles"],
        "semantic_quality": (
            {
                "affected_profiles": semantic_quality["$meta"][
                    "affected_profile_count"
                ],
                "findings": semantic_quality["$meta"]["finding_count"],
                "counts": semantic_quality["counts"],
            }
            if semantic_quality is not None
            else None
        ),
        "resource_validation": resource_validation,
        "capability_validation": capability_validation,
    }
    atomic_write_json(root / "artifacts" / "full_source_pipeline_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if acquisition["failed"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
