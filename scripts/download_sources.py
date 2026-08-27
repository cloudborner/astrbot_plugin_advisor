from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = DEFAULT_ROOT / "data" / "resource_profiles.json"
DEFAULT_MARKET = DEFAULT_ROOT / "data" / "market_snapshot.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "source_archives"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "download_manifest.json"
SHA_RE = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class DownloadItem:
    plugin_id: str
    display_name: str
    repo: str
    owner: str
    name: str
    ref: str
    ref_kind: str
    archive_name: str
    archive_url: str


@dataclass
class DownloadResult:
    plugin_id: str
    status: str
    archive_name: str
    archive_url: str
    ref: str
    ref_kind: str
    size_bytes: int = 0
    sha256: str = ""
    error: str = ""
    attempts: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download plugin source archives without extracting or executing them"
    )
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-archive-mib", type=int, default=512)
    parser.add_argument("--max-total-gib", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_repo(repo: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(repo)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("repository is not a public GitHub HTTPS URL")
    parts = [urllib.parse.unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("repository URL must contain owner and repository")
    owner, name = parts
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", name.removesuffix(".git")
    ):
        raise ValueError("unsafe GitHub owner or repository name")
    return owner, name.removesuffix(".git")


def archive_name(owner: str, name: str, ref: str) -> str:
    short_ref = ref[:12] if SHA_RE.fullmatch(ref) else "default"
    return f"{owner}__{name}__{short_ref}.tar.gz"


def build_items(profiles: dict[str, Any], market: dict[str, Any]) -> list[DownloadItem]:
    profile_values = profiles.get("profiles")
    market_values = market.get("plugins")
    if not isinstance(profile_values, dict) or not isinstance(market_values, dict):
        raise ValueError("profiles or market JSON has an invalid schema")

    items: list[DownloadItem] = []
    seen: set[str] = set()
    for plugin_id, market_value in market_values.items():
        if not isinstance(market_value, dict):
            continue
        repo = str(market_value.get("repo") or "")
        try:
            owner, name = parse_repo(repo)
        except ValueError:
            continue
        profile = profile_values.get(plugin_id)
        profile = profile if isinstance(profile, dict) else {}
        raw_ref = str(profile.get("commit_sha") or "").lower()
        ref = raw_ref if SHA_RE.fullmatch(raw_ref) else "HEAD"
        ref_kind = "commit" if ref != "HEAD" else "default_branch"
        key = f"{owner}/{name}/{ref}"
        if key in seen:
            continue
        seen.add(key)
        filename = archive_name(owner, name, ref)
        items.append(
            DownloadItem(
                plugin_id=str(plugin_id),
                display_name=str(market_value.get("display_name") or market_value.get("name") or plugin_id),
                repo=repo,
                owner=owner,
                name=name,
                ref=ref,
                ref_kind=ref_kind,
                archive_name=filename,
                archive_url=f"https://codeload.github.com/{owner}/{name}/tar.gz/{urllib.parse.quote(ref, safe='')}",
            )
        )
    return sorted(items, key=lambda item: item.plugin_id.casefold())


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    records = value.get("plugins")
    return records if isinstance(records, dict) else {}


def download_one(
    item: DownloadItem,
    output: Path,
    existing: dict[str, Any] | None,
    max_archive_bytes: int,
    retries: int,
) -> DownloadResult:
    destination = output / item.archive_name
    if destination.exists() and not existing:
        digest = hashlib.sha256()
        size = 0
        with destination.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                if size > max_archive_bytes:
                    return DownloadResult(
                        plugin_id=item.plugin_id,
                        status="failed",
                        archive_name=item.archive_name,
                        archive_url=item.archive_url,
                        ref=item.ref,
                        ref_kind=item.ref_kind,
                        error=f"existing archive exceeds {max_archive_bytes} byte limit",
                    )
                digest.update(block)
        return DownloadResult(
            plugin_id=item.plugin_id,
            status="skipped_existing",
            archive_name=item.archive_name,
            archive_url=item.archive_url,
            ref=item.ref,
            ref_kind=item.ref_kind,
            size_bytes=size,
            sha256=digest.hexdigest(),
        )
    if (
        existing
        and existing.get("status") == "complete"
        and existing.get("ref") == item.ref
        and destination.exists()
    ):
        expected_hash = str(existing.get("sha256") or "")
        expected_size = int(existing.get("size_bytes") or 0)
        if expected_hash and expected_size == destination.stat().st_size:
            digest = hashlib.sha256()
            with destination.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() == expected_hash:
                return DownloadResult(
                    plugin_id=item.plugin_id,
                    status="skipped_existing",
                    archive_name=item.archive_name,
                    archive_url=item.archive_url,
                    ref=item.ref,
                    ref_kind=item.ref_kind,
                    size_bytes=expected_size,
                    sha256=expected_hash,
                )

    part = destination.with_suffix(destination.suffix + ".part")
    last_error = ""
    for attempt in range(1, max(0, retries) + 2):
        try:
            if part.exists():
                part.unlink()
            request = urllib.request.Request(
                item.archive_url,
                headers={"User-Agent": "astrbot-plugin-advisor-source-fetch/1.0"},
                method="GET",
            )
            digest = hashlib.sha256()
            total = 0
            with DIRECT_OPENER.open(request, timeout=60) as response, part.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > max_archive_bytes:
                        raise ValueError(f"archive exceeds {max_archive_bytes} byte limit")
                    handle.write(block)
                    digest.update(block)
            os.replace(part, destination)
            return DownloadResult(
                plugin_id=item.plugin_id,
                status="complete",
                archive_name=item.archive_name,
                archive_url=item.archive_url,
                ref=item.ref,
                ref_kind=item.ref_kind,
                size_bytes=total,
                sha256=digest.hexdigest(),
                attempts=attempt,
            )
        except (
            OSError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            http.client.IncompleteRead,
            TimeoutError,
            ValueError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if part.exists():
                part.unlink()
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                408,
                425,
                429,
            } and not 500 <= exc.code <= 599:
                break
            if attempt <= retries:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    return DownloadResult(
        plugin_id=item.plugin_id,
        status="failed",
        archive_name=item.archive_name,
        archive_url=item.archive_url,
        ref=item.ref,
        ref_kind=item.ref_kind,
        error=last_error,
        attempts=max(0, retries) + 1,
    )


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be between 1 and 16")
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = read_json(args.profiles)
    market = read_json(args.market)
    items = build_items(profiles, market)
    manifest = load_manifest(manifest_path)
    max_archive_bytes = max(1, args.max_archive_mib) * 1024 * 1024
    max_total_bytes = max(1, args.max_total_gib) * 1024 * 1024 * 1024
    if not items:
        raise SystemExit("no public GitHub plugins found")

    print(json.dumps({"plugins": len(items), "output": str(output), "manifest": str(manifest_path)}), flush=True)
    completed_bytes = sum(
        int(value.get("size_bytes") or 0)
        for value in manifest.values()
        if isinstance(value, dict) and value.get("status") in {"complete", "skipped_existing"}
    )
    results: dict[str, dict[str, Any]] = {}
    for plugin_id, value in manifest.items():
        if isinstance(value, dict):
            results[plugin_id] = value
    counts = {"complete": 0, "skipped_existing": 0, "failed": 0}
    pending = [item for item in items if not (
        isinstance(manifest.get(item.plugin_id), dict)
        and manifest[item.plugin_id].get("status") == "complete"
        and manifest[item.plugin_id].get("ref") == item.ref
        and (output / item.archive_name).exists()
    )]
    for item in items:
        if item not in pending:
            counts["skipped_existing"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                item,
                output,
                manifest.get(item.plugin_id),
                max_archive_bytes,
                max(0, args.retries),
            ): item
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive: one plugin must not stop the batch
                result = DownloadResult(
                    plugin_id=item.plugin_id,
                    status="failed",
                    archive_name=item.archive_name,
                    archive_url=item.archive_url,
                    ref=item.ref,
                    ref_kind=item.ref_kind,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if result.status in {"complete", "skipped_existing"}:
                completed_bytes += result.size_bytes
            if completed_bytes > max_total_bytes:
                raise SystemExit(f"download safety limit exceeded: {completed_bytes} bytes")
            counts[result.status] = counts.get(result.status, 0) + 1
            results[result.plugin_id] = asdict(result)
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_code_executed": False,
                    "plugins": results,
                },
            )
            print(
                json.dumps(
                    {"done": index, "total_pending": len(pending), "plugin_id": item.plugin_id, "status": result.status, "size_mib": round(result.size_bytes / 1024 / 1024, 3), "error": result.error},
                    ensure_ascii=True,
                ),
                flush=True,
            )
    print(json.dumps({"summary": counts, "bytes": completed_bytes, "gib": round(completed_bytes / 1024 / 1024 / 1024, 3)}), flush=True)
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; manifest is resumable", file=sys.stderr)
        raise SystemExit(130)
