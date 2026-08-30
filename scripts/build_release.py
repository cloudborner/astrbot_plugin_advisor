from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "astrbot_plugin_advisor"
EXCLUDED_PARTS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".test-data",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "docs",
    "scripts",
    "schemas",
    "source_archives",
    "source_extracted",
    "tests",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
EXCLUDED_FILES = {
    "CHANGELOG.md",
    "pytest.ini",
    "ruff.toml",
    "advisor/remote_index.py",
    "data/index_public_key.pem",
    "data/resource_profiles.json",
    "data/resource_profiles.manifest.json",
    "data/source_resource_profiles.json",
    "data/source_resource_review_queue.json",
    "data/source_function_evidence.json",
}
REQUIRED_FILES = {
    "__init__.py",
    "main.py",
    "metadata.yaml",
    "_conf_schema.json",
    "requirements.txt",
    "README.md",
}


def release_version() -> str:
    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version:\s*['\"]?([^'\"\s]+)", metadata)
    if match is None:
        raise ValueError("metadata.yaml does not contain a version")
    return match.group(1)


def package_files() -> list[Path]:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FILES
        and path.suffix.casefold() not in EXCLUDED_SUFFIXES
    ]
    relative_names = {path.relative_to(ROOT).as_posix() for path in files}
    missing = REQUIRED_FILES - relative_names
    if missing:
        raise ValueError(f"release is missing required files: {sorted(missing)}")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def build_release(output_dir: Path) -> tuple[Path, Path]:
    target_dir = Path(output_dir).resolve()
    if target_dir == ROOT or ROOT in target_dir.parents:
        raise ValueError("release output must be outside the plugin source directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / f"{PACKAGE_ROOT}-{release_version()}.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as bundle:
        for source in package_files():
            relative = source.relative_to(ROOT).as_posix()
            destination = f"{PACKAGE_ROOT}/{relative}"
            info = zipfile.ZipInfo(destination, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, source.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean AstrBot plugin archive")
    parser.add_argument("output", type=Path, help="output directory outside the plugin tree")
    arguments = parser.parse_args()
    archive, checksum = build_release(arguments.output)
    print(archive)
    print(checksum)


if __name__ == "__main__":
    main()
