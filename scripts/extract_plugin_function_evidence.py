#!/usr/bin/env python3
"""Extract bounded functional evidence without importing or executing plugins."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.capabilities import normalize_summary  # noqa: E402
from advisor.index import atomic_write_json  # noqa: E402

MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILES = 400
EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "vendor",
    "third_party",
    "__pycache__",
    "tests",
    "test",
    "examples",
    "fixtures",
}
README_SUFFIXES = {".md", ".rst", ".txt"}
COMMAND_DECORATORS = {
    "command",
    "command_group",
    "llm_tool",
    "regex",
    "event_message_type",
}
GENERIC_HEADINGS = {
    "安装",
    "配置",
    "使用",
    "用法",
    "说明",
    "更新",
    "更新日志",
    "许可证",
    "致谢",
    "install",
    "installation",
    "configuration",
    "usage",
    "changelog",
    "license",
    "thanks",
}
RESOURCE_LABELS = {
    "remote_llm": "远程大模型调用",
    "local_model": "本地模型推理",
    "model_dependency": "模型运行依赖",
    "browser": "浏览器自动化",
    "pdf": "PDF处理",
    "image_processing": "图片处理",
    "audio_processing": "音频处理",
    "storage": "持久化存储",
    "vector_store": "向量检索",
    "runtime_download": "运行时下载",
    "large_upload": "大文件上传",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def bounded_text(value: object, maximum: int) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()[:maximum]


def unique(values: Iterable[object], *, limit: int, maximum: int = 80) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = bounded_text(value, maximum)
        key = text.casefold()
        if len(text) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def iter_source_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    emitted = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name.casefold() not in EXCLUDED_DIRS and not name.startswith(".")
        ]
        for name in sorted(files, key=str.casefold):
            path = Path(base) / name
            if path.suffix.casefold() not in suffixes:
                continue
            yield path
            emitted += 1
            if emitted >= MAX_SOURCE_FILES:
                return


def read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        data = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in data:
        return ""
    return data.decode("utf-8", errors="replace")


def _clean_markdown(value: str) -> str:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", value)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[`*_>#|~-]+", " ", text)
    return bounded_text(text, 500)


def find_readme(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.stem.casefold().startswith("readme")
        and path.suffix.casefold() in README_SUFFIXES
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.name), path.name.casefold()))[0]


def readme_evidence(root: Path) -> tuple[str, list[str], str]:
    path = find_readme(root)
    if path is None:
        return "", [], ""
    text = read_text(path)
    if not text:
        return "", [], path.name
    headings: list[str] = []
    paragraphs: list[str] = []
    in_fence = False
    current: list[str] = []
    for raw in text.splitlines()[:1500]:
        stripped = raw.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if match:
            heading = _clean_markdown(match.group(1))
            if heading.casefold() not in GENERIC_HEADINGS:
                headings.append(heading)
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if (
            stripped.startswith(("![", "[![", "<img", "<div", "<p align", "<!--"))
            or re.match(r"^[-*+]\s+", stripped)
            or re.match(r"^\d+[.)]\s+", stripped)
        ):
            continue
        cleaned = _clean_markdown(stripped)
        if cleaned:
            current.append(cleaned)
    if current:
        paragraphs.append(" ".join(current))
    summary = ""
    for paragraph in paragraphs:
        candidate = normalize_summary(paragraph)
        if 20 <= len(candidate) <= 240 and not candidate.startswith(("http://", "https://")):
            summary = candidate
            break
    return summary, unique(headings, limit=12), path.name


def dotted_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bounded_text(node.value, 80)
    return ""


def command_evidence(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    commands: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in iter_source_files(root, {".py", ".pyi"}):
        text = read_text(path)
        if not text:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, ValueError, MemoryError) as exc:
            parse_errors.append(
                f"{path.relative_to(root).as_posix()}: {exc.__class__.__name__}"
            )
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = bounded_text(ast.get_docstring(node) or "", 160)
            for decorator in node.decorator_list:
                call = decorator if isinstance(decorator, ast.Call) else None
                target = call.func if call else decorator
                kind = dotted_name(target).rsplit(".", 1)[-1]
                if kind not in COMMAND_DECORATORS:
                    continue
                label = literal_string(call.args[0]) if call and call.args else ""
                if not label and kind == "llm_tool":
                    label = node.name.replace("_", " ")
                if not label:
                    continue
                key = (kind, label.casefold())
                if key in seen:
                    continue
                seen.add(key)
                commands.append(
                    {
                        "name": label,
                        "kind": kind,
                        "description": doc,
                        "file": path.relative_to(root).as_posix(),
                        "line": int(getattr(node, "lineno", 1) or 1),
                    }
                )
                if len(commands) >= 30:
                    return commands, unique(parse_errors, limit=12, maximum=160)
    return commands, unique(parse_errors, limit=12, maximum=160)


def _walk_config(node: object, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        description = bounded_text(
            node.get("description") or node.get("hint") or node.get("help"), 160
        )
        if prefix and description:
            yield prefix, description
        items = node.get("items")
        if isinstance(items, dict):
            for key, value in items.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                yield from _walk_config(value, child)
        for key, value in node.items():
            if key in {"items", "description", "hint", "help", "options"}:
                continue
            if isinstance(value, dict):
                child = f"{prefix}.{key}" if prefix else str(key)
                yield from _walk_config(value, child)


def config_evidence(root: Path) -> list[dict[str, str]]:
    paths = sorted(
        {
            *root.glob("*_conf_schema.json"),
            *root.glob("**/_conf_schema.json"),
        },
        key=lambda path: path.as_posix().casefold(),
    )[:8]
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            raw = json.loads(read_text(path))
        except (json.JSONDecodeError, TypeError):
            continue
        for key, description in _walk_config(raw):
            folded = f"{key}\0{description}".casefold()
            if folded in seen:
                continue
            seen.add(folded)
            result.append(
                {
                    "key": bounded_text(key, 100),
                    "description": description,
                    "file": path.relative_to(root).as_posix(),
                }
            )
            if len(result) >= 30:
                return result
    return result


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    for path in iter_source_files(
        root, {".py", ".pyi", ".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml"}
    ):
        try:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            path.stat()
        except OSError:
            continue
        digest.update(relative)
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            continue
        count += 1
    digest.update(str(count).encode("ascii"))
    return digest.hexdigest()


def build_profile(
    plugin_id: str,
    market: dict[str, Any],
    source_dir: Path,
    resource: dict[str, Any],
) -> dict[str, Any]:
    readme_summary, headings, readme_file = readme_evidence(source_dir)
    commands, parse_errors = command_evidence(source_dir)
    configs = config_evidence(source_dir)
    command_names = [item["name"] for item in commands if item["kind"] != "regex"]
    command_descriptions = [item["description"] for item in commands if item["description"]]
    config_descriptions = [item["description"] for item in configs]
    resource_features = [str(item) for item in resource.get("features", [])]
    resource_terms = [RESOURCE_LABELS[item] for item in resource_features if item in RESOURCE_LABELS]
    market_terms = [*(market.get("tags") or []), market.get("category")]
    capabilities = unique(
        [*market_terms, *command_names, *headings, *resource_terms], limit=20
    )
    aliases = unique(
        [market.get("display_name"), market.get("name"), *command_names], limit=20
    )
    use_cases = unique([*command_descriptions, *config_descriptions], limit=12)
    market_summary = normalize_summary(market.get("short_desc") or market.get("desc"))
    functional_bits: list[str] = []
    if command_names:
        functional_bits.append("提供“" + "、".join(command_names[:6]) + "”等命令")
    if resource_terms:
        functional_bits.append("涉及" + "、".join(resource_terms[:4]))
    functional_summary = "；".join(functional_bits)
    summary = normalize_summary("；".join(x for x in (market_summary, readme_summary, functional_summary) if x))
    if not summary:
        summary = bounded_text(market.get("display_name") or market.get("name") or plugin_id, 240)
    evidence_kinds = ["market_metadata"]
    if readme_summary or headings:
        evidence_kinds.append("source_readme")
    if commands:
        evidence_kinds.append("source_commands")
    if configs:
        evidence_kinds.append("source_config_schema")
    if resource:
        evidence_kinds.append("source_resource_static")
    confidence = 0.5
    confidence += 0.12 if readme_summary else 0
    confidence += 0.15 if commands else 0
    confidence += 0.08 if configs else 0
    confidence += 0.08 if resource else 0
    return {
        "plugin_id": plugin_id,
        "version": bounded_text(market.get("version"), 64),
        "repo": bounded_text(market.get("repo"), 500),
        "source_dir": source_dir.name,
        "source_digest": source_digest(source_dir),
        "summary": summary,
        "capabilities": capabilities or ["其他"],
        "aliases": aliases,
        "use_cases": use_cases,
        "limitations": unique(parse_errors, limit=8, maximum=160),
        "sources": evidence_kinds,
        "confidence": round(min(0.95, confidence), 4),
        "evidence": {
            "readme_file": readme_file,
            "readme_summary": readme_summary,
            "readme_headings": headings,
            "commands": commands,
            "config_items": configs,
            "resource_features": resource_features[:40],
            "resource_level": bounded_text(resource.get("overall_level"), 8),
            "dependencies": [str(item)[:100] for item in resource.get("dependencies", [])[:80]],
        },
    }


def build_document(
    market_path: Path,
    source_root: Path,
    extraction_manifest_path: Path,
    resource_profiles_path: Path,
) -> dict[str, Any]:
    market_doc = load_json(market_path, {})
    market_plugins = market_doc.get("plugins", {}) if isinstance(market_doc, dict) else {}
    if not isinstance(market_plugins, dict):
        raise ValueError("market snapshot plugins must be an object")
    extraction = load_json(extraction_manifest_path, {})
    records = extraction.get("plugins", {}) if isinstance(extraction, dict) else {}
    if not isinstance(records, dict):
        raise ValueError("pipeline extraction manifest plugins must be an object")
    resources_doc = load_json(resource_profiles_path, {})
    resources = resources_doc.get("profiles", {}) if isinstance(resources_doc, dict) else {}
    resources = resources if isinstance(resources, dict) else {}
    profiles: dict[str, Any] = {}
    missing: list[str] = []
    for plugin_id, market in sorted(market_plugins.items(), key=lambda item: item[0].casefold()):
        if not isinstance(market, dict):
            continue
        record = records.get(plugin_id)
        directory = str(record.get("directory") or "") if isinstance(record, dict) else ""
        source_dir = source_root / directory
        if not directory or not source_dir.is_dir():
            missing.append(plugin_id)
            continue
        resource = resources.get(plugin_id)
        resource = resource if isinstance(resource, dict) else {}
        profiles[plugin_id] = build_profile(plugin_id, market, source_dir, resource)
        if len(profiles) % 250 == 0:
            print(f"function evidence {len(profiles)}/{len(market_plugins)}", file=sys.stderr)
    canonical = json.dumps(
        profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "$meta": {
            "schema_version": 1,
            "generated_at": now_iso(),
            "scan_mode": "local_source_static_read_only",
            "source_code_downloaded": True,
            "plugin_code_executed": False,
            "network_used": False,
            "market_count": len(market_plugins),
            "profile_count": len(profiles),
            "missing_source_count": len(missing),
            "missing_source_plugin_ids": missing,
            "profiles_sha256": hashlib.sha256(canonical).hexdigest(),
        },
        "profiles": profiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract functional and resource evidence from local plugin sources"
    )
    parser.add_argument("--market", type=Path, default=ROOT / "data" / "market_snapshot.json")
    parser.add_argument("--source", type=Path, default=ROOT / "source_extracted")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "source_extracted" / "pipeline_manifest.json"
    )
    parser.add_argument(
        "--resources", type=Path, default=ROOT / "data" / "source_resource_profiles.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "source_function_evidence.json"
    )
    args = parser.parse_args()
    document = build_document(
        args.market.resolve(),
        args.source.resolve(),
        args.manifest.resolve(),
        args.resources.resolve(),
    )
    atomic_write_json(args.output.resolve(), document)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "profiles": document["$meta"]["profile_count"],
                "missing": document["$meta"]["missing_source_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
