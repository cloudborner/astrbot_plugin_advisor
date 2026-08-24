from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "source_archives"
DEFAULT_OUTPUT = ROOT / "source_extracted"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "extraction_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely extract downloaded plugin source archives"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-plugin-mib", type=int, default=1024)
    parser.add_argument("--max-total-gib", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    if any(not part or part == "." for part in path.parts):
        return None
    return path


def flattened_path(member_path: PurePosixPath, top_level: str | None) -> Path:
    parts = list(member_path.parts)
    if top_level and parts and parts[0] == top_level:
        parts = parts[1:]
    return Path(*parts) if parts else Path(".")


def extract_archive(
    archive: Path,
    destination: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    temporary = destination.parent / f".{destination.name}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    extracted_bytes = 0
    file_count = 0
    skipped_links = 0
    skipped_special = 0
    top_level: str | None = None
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            names = [safe_member_path(member.name) for member in bundle.getmembers()]
            valid = [path for path in names if path is not None and path.parts]
            first_parts = {path.parts[0] for path in valid}
            if len(first_parts) == 1:
                top_level = next(iter(first_parts))
            for member, member_path in zip(bundle.getmembers(), names, strict=True):
                if member_path is None:
                    raise ValueError(f"unsafe archive path: {member.name!r}")
                relative = flattened_path(member_path, top_level)
                if relative == Path("."):
                    continue
                target = (temporary / relative).resolve()
                root = temporary.resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"archive path escapes destination: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.issym() or member.islnk():
                    skipped_links += 1
                    continue
                if not member.isfile():
                    skipped_special += 1
                    continue
                if member.size < 0 or extracted_bytes + member.size > max_bytes:
                    raise ValueError(f"expanded archive exceeds {max_bytes} byte limit")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                with source, target.open("wb") as handle:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        extracted_bytes += len(block)
                        if extracted_bytes > max_bytes:
                            raise ValueError(f"expanded archive exceeds {max_bytes} byte limit")
                file_count += 1
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        try:
            os.replace(temporary, destination)
        except PermissionError:
            # Some Windows scanners briefly hold a newly-created directory and
            # reject the rename.  Copy only into the known destination as a safe
            # fallback; the destination was checked to be absent above.
            shutil.copytree(temporary, destination)
            shutil.rmtree(temporary)
        return {
            "status": "complete",
            "archive": archive.name,
            "directory": destination.name,
            "expanded_bytes": extracted_bytes,
            "expanded_mib": round(extracted_bytes / 1024 / 1024, 3),
            "file_count": file_count,
            "skipped_links": skipped_links,
            "skipped_special": skipped_special,
            "top_level_stripped": top_level or "",
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(input_dir.glob("*.tar.gz"), key=lambda path: path.name.casefold())
    if not archives:
        raise SystemExit("no .tar.gz archives found")
    existing = read_json(manifest_path) if manifest_path.exists() else {}
    records = existing.get("archives") if isinstance(existing.get("archives"), dict) else {}
    records = dict(records)
    max_plugin_bytes = max(1, args.max_plugin_mib) * 1024 * 1024
    max_total_bytes = max(1, args.max_total_gib) * 1024 * 1024 * 1024
    total_expanded = sum(
        int(value.get("expanded_bytes") or 0)
        for value in records.values()
        if isinstance(value, dict) and value.get("status") == "complete"
    )
    counts = {"complete": 0, "skipped_existing": 0, "failed": 0}
    print(json.dumps({"archives": len(archives), "output": str(output_dir)}), flush=True)
    for index, archive in enumerate(archives, start=1):
        directory_name = archive.name.removesuffix(".tar.gz")
        destination = output_dir / directory_name
        old = records.get(archive.name)
        if (
            isinstance(old, dict)
            and old.get("status") == "complete"
            and destination.is_dir()
        ):
            counts["skipped_existing"] += 1
            print(json.dumps({"done": index, "total": len(archives), "archive": archive.name, "status": "skipped_existing"}), flush=True)
            continue
        try:
            result = extract_archive(
                archive,
                destination,
                max_bytes=max_plugin_bytes,
            )
            total_expanded += int(result["expanded_bytes"])
            if total_expanded > max_total_bytes:
                raise ValueError(f"total expanded size exceeds {max_total_bytes} byte limit")
            result["archive_sha256"] = sha256_file(archive)
            counts["complete"] += 1
        except Exception as exc:
            result = {
                "status": "failed",
                "archive": archive.name,
                "directory": directory_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            counts["failed"] += 1
        records[archive.name] = result
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_code_executed": False,
                "archives": records,
            },
        )
        print(json.dumps({"done": index, "total": len(archives), "archive": archive.name, "status": result["status"], "error": result.get("error", "")}), flush=True)
    print(json.dumps({"summary": counts, "expanded_gib": round(total_expanded / 1024 / 1024 / 1024, 3)}), flush=True)
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
