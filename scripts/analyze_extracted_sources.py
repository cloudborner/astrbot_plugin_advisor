#!/usr/bin/env python3
"""Read-only static resource estimator for extracted AstrBot plugins.

The scanner deliberately never imports, installs, or executes plugin code.  It
uses only stdlib parsing (AST plus small dependency-file readers) and local
market/cache snapshots.  The output is intentionally an estimate, not a
runtime benchmark.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "source-resource-profile-v1"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_REACHABLE = 20  # maximum emitted paths; the import graph is analysed more deeply
MAX_IMPORT_GRAPH = 250
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "dist", "build", "target",
    "coverage", "htmlcov", "__pycache__", ".pytest_cache", ".mypy_cache",
    "tests", "test", "examples", "example", "docs", "doc", "changelogs",
    "benchmarks", "benchmark", "fixtures", "fixture", "vendor", "third_party",
}
EXCLUDED_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}
SOURCE_EXTS = {".py", ".pyi", ".sh", ".bash", ".js", ".mjs", ".cjs", ".ts", ".java", ".rs"}
DEP_NAMES = {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "package.json", "Cargo.toml"}
MODEL_IMPORTS = {"torch", "tensorflow", "transformers", "sentence_transformers", "onnxruntime", "paddle", "paddleocr", "whisper", "faster_whisper", "funasr", "ultralytics", "llama_cpp", "llama_cpp_python", "diffusers", "keras"}
MODEL_FILE_HINTS = (".onnx", ".safetensors", ".gguf", ".ckpt", ".pth", ".pt")
REMOTE_HINTS = {"request_llm", "get_llm", "llm_response", "provider", "embedding_provider", "text_embedding", "astrbot.core.provider"}
DOWNLOAD_CALL_HINTS = {"download", "fetch_media", "save_file", "save_video", "save_audio", "youtube_dl", "youtubedl", "jmcomic", "jmdownloader"}
PROCESS_HINTS = {"ffmpeg", "ffprobe", "node", "java", "cargo", "rustc", "aria2", "yt-dlp", "yt_dlp", "gallery-dl"}
DOWNLOADER_PROCESSES = {"aria2", "yt-dlp", "yt_dlp", "gallery-dl"}
BROWSER_IMPORTS = {"playwright", "selenium", "pyppeteer"}
BROWSER_LAUNCH_SUFFIXES = ("chromium.launch", "firefox.launch", "webkit.launch", "launch_persistent_context", "connect_over_cdp", "pyppeteer.launch")
MEDIA_FILE_HINTS = (".pdf", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp3", ".wav", ".flac", ".mp4", ".mkv", ".webm", ".zip", ".tar", ".gz")
CACHE_NAME_HINTS = ("cache", "history", "records", "memory", "messages")
CACHE_SCHEMA_HINTS = ("schema", "field", "column", "type_cache", "validator")
CACHE_GROWTH_METHODS = ("append", "extend", "add", "update", "setdefault")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def repo_parts(repo: str) -> tuple[str, str] | None:
    if not isinstance(repo, str):
        return None
    m = re.search(r"github\.com[/:]([^/]+)/([^/#]+)", repo)
    if not m:
        m = re.match(r"^([^/]+)/([^/]+)$", repo)
    if not m:
        return None
    return m.group(1), m.group(2).removesuffix(".git")


def expected_dir_names(repo: str, sha: str) -> set[str]:
    parts = repo_parts(repo)
    if not parts or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{7,64}", sha):
        return set()
    owner, name = parts
    return {f"{owner}__{name}__{sha[:12].lower()}", f"{owner}__{name}__{sha[:12]}"}


def make_observation_indexes(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    market_doc = load_json(root / "data" / "market_snapshot.json", {})
    market = market_doc.get("plugins", {}) if isinstance(market_doc, dict) else {}
    old_doc = load_json(root / "data" / "resource_profiles.json", {})
    old = old_doc.get("profiles", {}) if isinstance(old_doc, dict) else {}
    obs_doc = load_json(root / ".cache" / "github_observations.json", {})
    observations: dict[str, dict[str, Any]] = {}
    candidate: dict[str, list[str]] = defaultdict(list)
    for pid, rec in market.items() if isinstance(market, dict) else []:
        rec = rec if isinstance(rec, dict) else {}
        cache = obs_doc.get(pid, {}) if isinstance(obs_doc, dict) else {}
        cache = cache if isinstance(cache, dict) else {}
        obs = cache.get("observation", {}) if isinstance(cache.get("observation", {}), dict) else {}
        old_rec = old.get(pid, {}) if isinstance(old, dict) else {}
        old_rec = old_rec if isinstance(old_rec, dict) else {}
        repo = cache.get("repo") or rec.get("repo") or old_rec.get("repo")
        sha = obs.get("commit_sha") or old_rec.get("commit_sha")
        item = {
            "plugin_id": pid,
            "repo": repo or "",
            "commit_sha": sha or "",
            "version": cache.get("version") or rec.get("version") or old_rec.get("version") or "",
            "download_count": int(rec.get("download_count") or 0),
            "stars": int(rec.get("stars") or 0),
        }
        observations[pid] = item
        for name in expected_dir_names(item["repo"], item["commit_sha"]):
            candidate[name].append(pid)
    meta = {"market_count": len(market) if isinstance(market, dict) else 0, "old_count": len(old) if isinstance(old, dict) else 0}
    return observations, candidate, {"market": market, "old": old, "meta": meta}


def map_sources(source_root: Path, observations: dict[str, dict[str, Any]], candidates: dict[str, list[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dirs = [p for p in source_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    mapped: list[dict[str, Any]] = []
    unmapped: list[str] = []
    conflicts: dict[str, list[str]] = {}
    plugin_dirs: dict[str, list[str]] = defaultdict(list)
    for path in sorted(dirs, key=lambda p: p.name.lower()):
        ids = sorted(set(candidates.get(path.name, [])))
        if len(ids) == 1:
            pid = ids[0]
            item = dict(observations[pid])
            item["source_dir"] = path
            mapped.append(item)
            plugin_dirs[pid].append(path.name)
        elif len(ids) > 1:
            conflicts[path.name] = ids
        else:
            unmapped.append(path.name)
    duplicates = {pid: names for pid, names in plugin_dirs.items() if len(names) != 1}
    missing = sorted(set(observations) - set(plugin_dirs))
    summary = {
        "source_dir_count": len(dirs),
        "mapped_dir_count": len(mapped),
        "unmapped_dirs": unmapped,
        "conflicts": conflicts,
        "duplicate_plugin_mappings": duplicates,
        "market_records_without_source": missing,
    }
    return mapped, summary


def excluded(rel: Path) -> bool:
    parts = {p.lower() for p in rel.parts}
    return bool(parts & EXCLUDED_DIRS) or rel.name.lower() in EXCLUDED_NAMES


def iter_files(root: Path) -> Iterable[Path]:
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in files:
            path = Path(base) / name
            rel = path.relative_to(root)
            if excluded(rel):
                continue
            yield path


def read_text_limited(path: Path) -> tuple[str | None, str | None]:
    try:
        size = path.stat().st_size
        if size > MAX_TEXT_BYTES:
            return None, "file exceeds safe AST size limit"
        data = path.read_bytes()
        if b"\x00" in data:
            return None, "binary file skipped"
        return data.decode("utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"read error: {exc.__class__.__name__}"


def module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    bits = list(rel.parts)
    if bits and bits[-1] == "__init__":
        bits.pop()
    return ".".join(bits)


def local_module_path(root: Path, mod: str, level: int, current: Path) -> Path | None:
    if level:
        base = current.parent
        for _ in range(level - 1):
            base = base.parent
        tail = mod.split(".") if mod else []
        candidate = base.joinpath(*tail)
    else:
        candidate = root.joinpath(*mod.split("."))
    for p in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if p.exists() and p.is_file() and not excluded(p.relative_to(root)):
            return p
    return None


def const_strings(node: ast.AST) -> list[str]:
    out: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            out.append(item.value)
    return out


def dotted_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def target_name(node: ast.AST | None) -> str:
    """Return a stable dotted collection name for assignments and mutations."""
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    return dotted_name(node)


def normalized_var(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def call_strings(node: ast.Call, bindings: dict[str, list[str]]) -> list[str]:
    """Resolve literal command arguments, including simple local list variables."""
    values = const_strings(node)
    for arg in node.args:
        raw = arg.value if isinstance(arg, ast.Starred) else arg
        if isinstance(raw, ast.Name):
            values.extend(bindings.get(raw.id, []))
    return values


def enclosing_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    cur = node
    while cur in parents and not isinstance(parents[cur], (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        cur = parents[cur]
    return parents.get(cur, cur)


def model_call_risk(name: str, import_roots: set[str]) -> int:
    """Return 0/3/4 only for an actual model-loading call with runtime context."""
    last = name.rsplit(".", 1)[-1]
    deps = import_roots & MODEL_IMPORTS
    if last == "inferencesession" and "onnxruntime" in deps:
        return 3
    if last == "from_pretrained" and deps:
        receiver = name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        if any(token in receiver for token in ("tokenizer", "processor", "featureextractor", "config")):
            return 0
        if any(token in receiver for token in ("causallm", "conditionalgeneration", "diffusion", "stable", "whisper", "cosyvoice", "gpt", "llama", "vlmodel")):
            return 4
        return 3
    if (name.endswith("torch.load") or last in {"load_state_dict", "load_checkpoint"}) and deps:
        return 3
    if last == "pipeline" and deps & {"transformers", "diffusers"}:
        return 4 if "diffusers" in deps else 3
    if last in {"sentencetransformer", "paddleocr", "yolo", "automodel"} and deps:
        return 3
    if last in {"initialize_model", "init_model"}:
        return 4
    if last in {"whispermodel", "llama"} and deps:
        return 4
    if last == "load_model":
        receiver = name.rsplit(".", 1)[0] if "." in name else ""
        explicit_receiver = any(token in receiver.split(".") for token in ("model", "seg", "inference", "classifier", "encoder", "whisper"))
        if deps or explicit_receiver:
            if any(token in receiver for token in ("whisper", "cosyvoice", "gpt", "sovits", "llama", "llm", "diffusion")):
                return 4
            return 3
    return 0


def owning_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.ClassDef | None:
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.ClassDef):
            return cur
    return None


def is_plugin_class(node: ast.ClassDef | None) -> bool:
    if node is None:
        return False
    bases = {dotted_name(base).lower().rsplit(".", 1)[-1] for base in node.bases}
    decorators = {
        dotted_name(dec.func if isinstance(dec, ast.Call) else dec).lower()
        for dec in node.decorator_list
    }
    return bool(bases & {"star", "plugin", "astrbotplugin"}) or any("register" in name for name in decorators)


def line_for(node: ast.AST) -> int:
    return int(getattr(node, "lineno", 1) or 1)


def evidence(kind: str, path: Path, node: ast.AST | None, symbol: str, detail: str, strength: str = "medium") -> dict[str, Any]:
    return {"kind": kind, "file": path.as_posix(), "line": line_for(node) if node else 1, "symbol": symbol or "module", "detail": detail[:160], "strength": strength}


def level(score: int) -> str:
    return f"L{max(0, min(4, int(score)))}"


def parse_dependencies(root: Path) -> list[str]:
    names: set[str] = set()
    for path in iter_files(root):
        if path.name not in DEP_NAMES:
            continue
        text, _ = read_text_limited(path)
        if not text:
            continue
        if path.name == "package.json":
            try:
                data = json.loads(text)
                for sec in ("dependencies", "optionalDependencies"):
                    names.update(str(x) for x in (data.get(sec, {}) or {}).keys())
            except json.JSONDecodeError:
                pass
        elif path.name == "pyproject.toml" and tomllib is not None:
            try:
                data = tomllib.loads(text)
                project = data.get("project", {}) or {}
                names.update(str(x).split("[", 1)[0].split("=", 1)[0].strip() for x in project.get("dependencies", []) or [])
                poetry = (data.get("tool", {}) or {}).get("poetry", {}) or {}
                for key in (poetry.get("dependencies", {}) or {}):
                    if str(key).lower() != "python":
                        names.add(str(key))
            except (ValueError, TypeError):
                pass
        elif path.name == "setup.cfg":
            in_runtime = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    in_runtime = stripped.lower() == "[options]"
                elif in_runtime and ("install_requires" in stripped or (line.startswith(" ") and stripped)):
                    m = re.match(r"([A-Za-z0-9_.-]+)", stripped)
                    if m:
                        names.add(m.group(1))
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "-e ", "--")):
                    continue
                m = re.match(r"([A-Za-z0-9_.-]+)", line)
                if m:
                    names.add(m.group(1))
    return sorted(names)[:80]


class SourceAnalyzer:
    def __init__(self, root: Path, plugin: dict[str, Any]):
        self.root = root
        self.plugin = plugin
        self.evidence: list[dict[str, Any]] = []
        self.unknowns: list[str] = []
        self.ast_files: dict[Path, ast.AST] = {}
        self.reachable: list[Path] = []
        self.imports: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.strings: list[tuple[Path, ast.AST, str]] = []
        self.local_model = False
        self.local_model_risk = 0
        self.model_file_reference = False
        self.model_dependency = False
        self.remote_llm = False
        self.processes: set[str] = set()
        self.downloads: set[str] = set()
        self.background_count = 0
        self.background_unbounded = False
        self.hard_unbounded = False
        self.background_cleanup = False
        self.cpu_task_risk = 0
        self.sem_limits: list[int] = []
        self.sem_found = False
        self.worker_limits: list[int] = []
        self.full_reads = 0
        self.small_full_reads = 0
        self.streaming_reads = 0
        self.cache_unbounded = False
        self.cache_bounded = False
        self.cache_present = False
        self.db_present = False
        self.vector_present = False
        self.browser_present = False
        self.browser_dependency = False
        self.image_present = False
        self.audio_present = False
        self.pdf_present = False
        self.disk_download = False
        self.size_limit_signals: set[str] = set()
        self.upload_signals: set[str] = set()
        self.config_files: list[str] = []

    def collect_python(self) -> list[Path]:
        files = [p for p in iter_files(self.root) if p.suffix.lower() in {".py", ".pyi"}]
        if not files:
            self.unknowns.append("没有可解析的 Python 源文件")
            return []
        for path in files:
            text, err = read_text_limited(path)
            if err:
                self.unknowns.append(f"{path.relative_to(self.root).as_posix()}: {err}")
                continue
            try:
                self.ast_files[path] = ast.parse(text or "", filename=str(path))
            except (SyntaxError, ValueError, MemoryError) as exc:
                self.unknowns.append(f"{path.relative_to(self.root).as_posix()}: AST {exc.__class__.__name__}")
        return files

    def build_reachable(self) -> None:
        files = list(self.ast_files)
        mains = [p for p in files if p.name == "main.py"]
        if not mains:
            mains = sorted(files, key=lambda p: (len(p.relative_to(self.root).parts), p.as_posix()))[:1]
        queue: deque[Path] = deque(mains)
        seen: set[Path] = set()
        while queue and len(seen) < MAX_IMPORT_GRAPH:
            path = queue.popleft()
            if path in seen or path not in self.ast_files:
                continue
            seen.add(path)
            tree = self.ast_files[path]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports[alias.name.split(".")[0]] += 1
                        nxt = local_module_path(self.root, alias.name, 0, path)
                        if nxt and nxt not in seen:
                            queue.append(nxt)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod:
                        self.imports[mod.split(".")[0]] += 1
                    for alias in node.names:
                        self.imports[alias.name.split(".")[0]] += 1
                    nxt = local_module_path(self.root, mod, node.level, path)
                    if nxt and nxt not in seen:
                        queue.append(nxt)
        if not seen:
            self.unknowns.append("入口文件无法解析，回退扫描全部非测试 Python 文件")
            self.reachable = files
            return
        self.reachable = sorted(seen, key=lambda p: p.as_posix())
        if queue:
            self.unknowns.append("本地导入图超过可达文件上限")

    def add(self, ev: dict[str, Any]) -> None:
        key = (ev["kind"], ev["file"], ev["line"], ev["symbol"], ev["detail"])
        if not any((x["kind"], x["file"], x["line"], x["symbol"], x["detail"]) == key for x in self.evidence):
            self.evidence.append(ev)

    def inspect_runtime_configs(self) -> None:
        """Read only small runtime config manifests; never interpret YAML or execute it."""
        config_names = {"_conf_schema.json", "config.json", "config.yaml", "config.yml", "metadata.yaml", "metadata.yml"}
        for path in iter_files(self.root):
            if path.name.lower() not in config_names:
                continue
            text, err = read_text_limited(path)
            if err or text is None:
                continue
            rel = path.relative_to(self.root).as_posix()
            self.config_files.append(rel)
            low = text.lower()
            for key in ("max_concurrent", "max_concurrency", "max_workers", "semaphore", "concurrency_limit", "rate_limit", "size_limit", "soft_limit_mb", "max_file_size"):
                if key in low:
                    self.add(evidence("config", path, None, key, f"运行配置包含资源边界 {key}", "medium"))
                    if "concurr" in key or "semaphore" in key or "worker" in key:
                        self.sem_found = True
                    if "limit" in key or "size" in key:
                        self.size_limit_signals.add(key)
            if any(k in low for k in ("upload", "send_file", "send_video", "send_document")):
                self.upload_signals.add("config_upload")
            if any(ext in low for ext in MODEL_FILE_HINTS):
                self.model_file_reference = True
                self.add(evidence("model_reference", path, None, "model_file", "运行配置引用模型权重文件，但未证明运行时加载", "weak"))

    def inspect(self) -> None:
        for path in self.reachable:
            tree = self.ast_files.get(path)
            if not tree:
                continue
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            imports = {a.name.lower() for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            imports.update({(n.module or "").lower() for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})
            imports.update({a.name.lower() for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names})
            import_roots = {name.split(".")[0] for name in imports if name}
            tree_literals = {value.lower() for value in const_strings(tree)}
            ytdlp_ffmpeg_config = bool(tree_literals & {"ffmpeg_location", "merge_output_format", "postprocessors", "ffmpegpostprocessor"})

            bindings: dict[str, list[str]] = {}
            mutated: set[str] = set()
            cache_bound_signal = False
            for candidate in ast.walk(tree):
                if isinstance(candidate, (ast.Name, ast.Attribute, ast.FunctionDef, ast.AsyncFunctionDef)):
                    raw_name = candidate.id if isinstance(candidate, ast.Name) else candidate.attr if isinstance(candidate, ast.Attribute) else candidate.name
                    low_name = raw_name.lower()
                    if "cache" in low_name and any(x in low_name for x in ("limit", "max", "ttl", "capacity", "cleanup", "evict")):
                        cache_bound_signal = True
                if isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                    value = candidate.value
                    targets = candidate.targets if isinstance(candidate, ast.Assign) else [candidate.target]
                    literal_values = const_strings(value) if value is not None else []
                    for target in targets:
                        if isinstance(target, ast.Name) and literal_values:
                            bindings[target.id] = literal_values
                        if isinstance(target, ast.Subscript):
                            mutated.add(normalized_var(target_name(target)))
                        raw = target_name(target)
                        low = normalized_var(raw)
                        if "cache" in low and any(x in low for x in ("limit", "max", "ttl", "capacity")):
                            cache_bound_signal = True
                elif isinstance(candidate, ast.AugAssign) and isinstance(candidate.target, ast.Subscript):
                    mutated.add(normalized_var(target_name(candidate.target)))
                elif isinstance(candidate, ast.Call):
                    called = dotted_name(candidate.func).lower()
                    last = called.rsplit(".", 1)[-1]
                    if last in CACHE_GROWTH_METHODS and "." in called:
                        mutated.add(normalized_var(called.rsplit(".", 1)[0]))
                    if last in {"ttlcache", "ttl_cache"} or (last == "lru_cache" and any(
                        kw.arg == "maxsize" and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
                        for kw in candidate.keywords
                    )):
                        cache_bound_signal = True

            for name in imports:
                root = name.split(".")[0]
                if root in MODEL_IMPORTS:
                    self.model_dependency = True
                    self.add(evidence("model_dependency", path, tree, root, f"可达代码导入模型运行时依赖 {root}，但未单独证明权重加载", "medium"))
                if root in {"sqlite3", "sqlalchemy", "peewee", "redis", "faiss", "chromadb", "lancedb"} or "faiss" in name or "sqlite" in name:
                    self.db_present |= root in {"sqlite3", "sqlalchemy", "peewee", "redis"} or "sqlite" in name
                    self.vector_present |= root in {"faiss", "chromadb", "lancedb"} or "faiss" in name or "chromadb" in name
                    self.add(evidence("storage", path, tree, root, f"可达代码使用持久化/向量存储 {root}", "medium"))
                if root in BROWSER_IMPORTS:
                    self.browser_dependency = True
                    self.add(evidence("browser_dependency", path, tree, root, f"可达代码导入浏览器依赖 {root}，尚未证明启动浏览器", "weak"))
                if root in {"pil", "cv2", "imageio", "fitz", "pypdf", "pdf2image"}:
                    self.image_present |= root in {"pil", "cv2", "imageio"}
                    self.pdf_present |= root in {"fitz", "pypdf", "pdf2image"}
                    self.add(evidence("media", path, tree, root, f"可达代码处理图像或 PDF {root}", "medium"))
                if root in {"asyncio", "threading", "concurrent", "apscheduler", "schedule"}:
                    self.add(evidence("background", path, tree, root, f"可达代码使用异步/线程调度 {root}", "medium"))
                    if root in {"apscheduler", "schedule"}:
                        self.background_count += 1
                        self.add(evidence("background", path, tree, root, f"可达代码使用定时调度器 {root}", "medium"))
                if root in {"requests", "httpx", "aiohttp", "urllib"}:
                    self.remote_llm |= False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = dotted_name(node.func).lower()
                    last = name.rsplit(".", 1)[-1]
                    self.calls[name] += 1
                    if any(h in name for h in REMOTE_HINTS):
                        self.remote_llm = True
                        self.add(evidence("remote_api", path, node, name, "调用 AstrBot/远程模型接口，不按本地模型计", "strong"))
                    model_risk = model_call_risk(name, import_roots)
                    if model_risk:
                        self.local_model = True
                        self.local_model_risk = max(self.local_model_risk, model_risk)
                        self.add(evidence("local_model", path, node, name, f"运行时调用本地模型加载入口（风险 L{model_risk}）", "strong"))

                    process_call = (
                        (name.startswith("subprocess.") and last in {"run", "popen", "call", "check_call", "check_output"})
                        or name.startswith("asyncio.create_subprocess")
                        or ("multiprocessing" in import_roots and last == "process")
                        or name in {"os.system", "os.popen"}
                        or name.startswith("os.spawn")
                    )
                    process_values = call_strings(node, bindings) if process_call else []
                    matched_processes: set[str] = set()
                    if process_call:
                        self.processes.add("subprocess")
                        self.add(evidence("external_process", path, node, name, "启动外部进程", "strong"))
                        for text in process_values:
                            low_text = text.strip().lower()
                            for hint in PROCESS_HINTS:
                                if low_text == hint or re.search(rf"(?<![a-z0-9_-]){re.escape(hint)}(?![a-z0-9_-])", low_text):
                                    matched_processes.add(hint)
                                    self.processes.add(hint)
                                    self.add(evidence("external_process", path, node, hint, f"外部进程参数包含 {hint}", "strong"))
                                    if hint in DOWNLOADER_PROCESSES:
                                        self.downloads.add("yt-dlp" if hint == "yt_dlp" else hint)
                                        self.disk_download = True
                        rel_low = path.relative_to(self.root).as_posix().lower()
                        if not matched_processes and any(token in rel_low for token in ("gpt_sovits", "gpt-sovits", "comfyui_start", "model_server", "inference_server")):
                            self.local_model = True
                            self.local_model_risk = max(self.local_model_risk, 4)
                            self.add(evidence("local_model", path, node, name, "启动本地模型服务子进程", "strong"))
                    if process_call and (any(token in name for token in ("pip.install", "pip_install", "pipinstall")) or any("pip install" in x.lower() for x in process_values)):
                        self.processes.add("pip install")
                        self.downloads.add("pip-install")
                        self.disk_download = True
                        self.add(evidence("external_process", path, node, "pip install", "运行时安装或升级依赖", "strong"))

                    if any(suffix in name for suffix in BROWSER_LAUNCH_SUFFIXES) or (
                        self.browser_dependency and last in {"chrome", "firefox", "edge", "safari"} and "webdriver" in name
                    ):
                        self.browser_present = True
                        self.processes.add("browser")
                        self.add(evidence("external_process", path, node, name, "运行时启动或连接浏览器进程", "strong"))

                    if any(token in name for token in DOWNLOAD_CALL_HINTS) or "yt_dlp" in name or "yt-dlp" in name:
                        if "jmcomic" in name or "jmdownloader" in name:
                            download_name = "jmcomic"
                        elif "yt_dlp" in name or "youtube" in name:
                            download_name = "yt-dlp"
                        else:
                            download_name = "download"
                        self.downloads.add(download_name)
                        self.disk_download = True
                        self.add(evidence("download", path, node, name, "运行时调用下载或媒体落盘入口", "strong"))
                        if download_name == "yt-dlp" and ytdlp_ffmpeg_config:
                            self.processes.add("ffmpeg")
                            self.add(evidence("external_process", path, node, "ffmpeg", "yt-dlp 下载配置启用 FFmpeg 合并或后处理", "strong"))

                    if any(token in name for token in ("export_pdf", "create_pdf", "generate_pdf", "convert_to_pdf", "images_to_pdf")):
                        self.pdf_present = True
                        self.add(evidence("media", path, node, name, "运行时生成或转换 PDF", "strong"))

                    if "audiosegment" in name or any(token in name for token in ("detect_leading_silence", "audioop.", "wave.writeframes")):
                        self.audio_present = True
                        self.add(evidence("media", path, node, name, "运行时解码、裁剪或写入音频", "strong"))

                    if name.endswith("create_task") or name.endswith("ensure_future") or name.endswith("run_forever"):
                        self.background_count += 1
                        self.add(evidence("background", path, node, name, "创建异步后台任务或常驻循环", "medium"))
                    if name.endswith(("to_thread", "run_in_executor")):
                        scope = enclosing_scope(node, parents)
                        scope_name = getattr(scope, "name", "").lower()
                        arg_names = " ".join(dotted_name(arg).lower() for arg in node.args)
                        task_context = " ".join((scope_name, arg_names, path.relative_to(self.root).as_posix().lower()))
                        if any(token in task_context for token in ("rebuild_index", "rebuild_graph", "graph_rebuild", "index_validator_rebuild", "transcode", "render_video", "generate_video")):
                            self.cpu_task_risk = max(self.cpu_task_risk, 3)
                            self.add(evidence("peak_cpu", path, node, name, "后台线程执行索引/图重建或重型媒体任务", "strong"))
                        elif any(token in task_context for token in ("compact", "render", "encode", "decode", "parse", "analy", "transform", "tokeniz", "process", "generate", "export")):
                            self.cpu_task_risk = max(self.cpu_task_risk, 2)
                            self.add(evidence("peak_cpu", path, node, name, "后台线程执行压缩、解析、渲染或转换任务", "medium"))
                    if name.endswith("gather"):
                        self.add(evidence("concurrency", path, node, name, "并行聚合多个协程，需结合上游任务数判断峰值", "medium"))
                    if any(token in name for token in ("send_file", "send_video", "send_document", "upload", "upload_file")):
                        self.upload_signals.add(name)
                        self.add(evidence("upload", path, node, name, "运行时上传文件或媒体", "medium"))
                    if name.endswith("thread") or name.endswith("threading.thread") or name.endswith("threadpoolexecutor") or name.endswith("processpoolexecutor"):
                        self.background_count += 1
                        self.add(evidence("background", path, node, name, "创建线程/进程池后台工作", "medium"))
                    if name.endswith("semaphore"):
                        self.sem_found = True
                        value = None
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                            value = int(node.args[0].value)
                        if value is not None and value > 0:
                            self.sem_limits.append(value)
                            self.add(evidence("concurrency", path, node, name, f"使用有界 Semaphore({value})", "strong"))
                    if name.endswith("threadpoolexecutor") or name.endswith("processpoolexecutor"):
                        for kw in node.keywords:
                            if kw.arg == "max_workers" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                                self.worker_limits.append(int(kw.value.value))
                    if name.endswith("read_bytes") or name.endswith("read_text") or name.endswith("read"):
                        if not node.args and not node.keywords:
                            receiver = name.rsplit(".", 1)[0] if "." in name else name
                            scope_text = " ".join(const_strings(enclosing_scope(node, parents))).lower()
                            materialized_media = any(token in receiver for token in ("response", "resp", "upload", "media", "image", "audio", "video", "pdf", "archive", "blob")) or any(
                                suffix in scope_text for suffix in MEDIA_FILE_HINTS
                            )
                            if materialized_media:
                                self.full_reads += 1
                                self.add(evidence("peak_memory", path, node, name, "一次性读取媒体、归档或响应内容，可能形成峰值内存", "strong"))
                            else:
                                self.small_full_reads += 1
                        else:
                            self.streaming_reads += 1
                    if "iter_content" in name or "aiter_bytes" in name or "copyfileobj" in name:
                        self.streaming_reads += 1
                    if last in {"ttlcache", "ttl_cache"}:
                        self.cache_present = True
                        self.cache_bounded = True
                        self.add(evidence("cache", path, node, name, "使用带 TTL/容量语义的缓存", "strong"))
                    elif last == "lru_cache":
                        self.cache_present = True
                        maxsize_none = any(kw.arg == "maxsize" and isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in node.keywords)
                        if maxsize_none:
                            self.cache_unbounded = True
                            self.add(evidence("cache", path, node, name, "lru_cache(maxsize=None) 可无界增长", "strong"))
                        else:
                            self.cache_bounded = True
                            self.add(evidence("cache", path, node, name, "使用有界 LRU 缓存", "strong"))
                    elif last == "cache" and name in {"cache", "functools.cache"}:
                        self.cache_present = True
                        self.cache_unbounded = True
                        self.add(evidence("cache", path, node, name, "functools.cache 无容量上限", "strong"))
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    value_name = dotted_name(value).lower()
                    if value_name.endswith(("response.content", "resp.content")):
                        scope = enclosing_scope(node, parents)
                        scope_name = getattr(scope, "name", "").lower()
                        scope_text = " ".join(const_strings(scope)).lower()
                        if any(token in scope_name or token in scope_text for token in ("audio", "video", "image", "media", "pdf", "download", ".wav", ".mp3", ".mp4")):
                            self.full_reads += 1
                            self.add(evidence("peak_memory", path, node, value_name, "远程媒体响应以完整字节串驻留内存", "strong"))
                    for target in targets:
                        var = target_name(target)
                        low = normalized_var(var)
                        lexical_long_lived = isinstance(parents.get(node), (ast.Module, ast.ClassDef))
                        plugin_instance = var.startswith(("self.", "cls.")) and is_plugin_class(owning_class(node, parents))
                        long_lived = lexical_long_lived or plugin_instance
                        fixed_constant = isinstance(target, ast.Name) and target.id.isupper()
                        cache_like = any(x in low for x in CACHE_NAME_HINTS)
                        collection = isinstance(value, (ast.Dict, ast.List, ast.Set))
                        if long_lived and cache_like and collection and not fixed_constant:
                            self.cache_present = True
                            if any(x in low for x in CACHE_SCHEMA_HINTS):
                                self.cache_bounded = True
                                self.add(evidence("cache", path, node, var, f"{var} 是结构/字段缓存，不按运行消息无界增长计", "medium"))
                            elif low in mutated:
                                if cache_bound_signal:
                                    self.cache_bounded = True
                                    self.add(evidence("cache", path, node, var, f"长期集合 {var} 存在写入且检测到 TTL/容量上限", "strong"))
                                else:
                                    self.cache_unbounded = True
                                    self.add(evidence("cache", path, node, var, f"长期集合 {var} 存在运行时写入且未见容量上限", "strong"))
                            else:
                                self.add(evidence("cache", path, node, var, f"长期集合 {var} 未发现运行时增长写入", "weak"))
                        if any(x in low for x in ("limit", "max_size", "soft_limit", "part_size", "max_file")):
                            self.size_limit_signals.add(var)
                            self.add(evidence("download_limit", path, node, var, "代码变量体现下载或文件大小上限", "medium"))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value
                    low = text.lower()
                    self.strings.append((path, node, text))
                    if any(ext in low for ext in MODEL_FILE_HINTS):
                        self.model_file_reference = True
                        self.add(evidence("model_reference", path, node, "model_file", "可达代码引用模型文件，但未单独证明运行时加载", "weak"))
                    if any(x in low for x in ("size_limit", "max_size", "soft_limit", "part_size", "max_file_size")):
                        self.size_limit_signals.add(text[:80])
                        self.add(evidence("download_limit", path, node, "size_limit", "代码或配置包含下载/文件大小限制", "medium"))
            for node in ast.walk(tree):
                if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = any(isinstance(child, ast.Break) for child in ast.walk(node))
                    has_deadline = any(isinstance(child, ast.Name) and child.id.lower() in {"deadline", "timeout", "retry_timeout", "max_retries", "attempt"} for child in ast.walk(node)) and any(isinstance(child, (ast.Raise, ast.Return)) for child in ast.walk(node))
                    loop_calls = {dotted_name(child.func).lower() for child in ast.walk(node) if isinstance(child, ast.Call)}
                    has_pause = any(call.endswith(("sleep", "wait", "recv", "receive")) for call in loop_calls) or any(
                        call.endswith(".get") and any(token in call for token in ("queue", "channel", "mailbox"))
                        for call in loop_calls
                    )
                    spawns_tasks = any(call.endswith(("create_task", "ensure_future")) for call in loop_calls)
                    scope = enclosing_scope(node, parents)
                    scope_name = getattr(scope, "name", "").lower()
                    background_named = any(token in scope_name for token in ("loop", "worker", "watch", "poll", "schedule", "monitor", "heartbeat", "renew", "consume"))
                    has_await = any(isinstance(child, ast.Await) for child in ast.walk(node))
                    if not (has_pause or spawns_tasks or background_named or has_await):
                        continue
                    self.background_count += 1
                    if not (has_break or has_deadline or has_pause) or (spawns_tasks and not (self.sem_found or self.worker_limits)):
                        self.hard_unbounded = True
                        self.background_unbounded = True
                    detail = "while True 循环；结合退出、阻塞等待和任务派生判断是否无界"
                    self.add(evidence("background", path, node, "while True", detail, "strong" if self.hard_unbounded else "medium"))
            if self.background_count:
                names = {dotted_name(n.func).lower() for n in ast.walk(tree) if isinstance(n, ast.Call)}
                if any(x.endswith(("unlink", "remove", "rmtree", "cleanup", "close", "cancel")) for x in names):
                    self.background_cleanup = True
        # A background task is not automatically unbounded: sleeping schedulers and
        # singleton lifecycle tasks are common. Only a tight/unbounded producer loop
        # is elevated here.
        self.background_unbounded = self.hard_unbounded

    def scores(self) -> tuple[dict[str, int], dict[str, Any]]:
        size = sum(p.stat().st_size for p in iter_files(self.root) if p.is_file())
        s = {k: 0 for k in ("idle_memory", "peak_memory", "idle_cpu", "peak_cpu", "disk", "network")}
        if size >= 1024**3:
            s["disk"] = 4
        elif size >= 200 * 1024**2:
            s["disk"] = 3
        elif size >= 40 * 1024**2:
            s["disk"] = 2
        elif size >= 8 * 1024**2:
            s["disk"] = 1
        if s["disk"]:
            self.add(evidence("disk", self.root, None, "source_size", f"解压后可扫描文件总量约 {size} bytes", "medium"))
        if self.local_model:
            if self.local_model_risk >= 4:
                s.update(idle_memory=max(s["idle_memory"], 3), peak_memory=max(s["peak_memory"], 4), idle_cpu=max(s["idle_cpu"], 2), peak_cpu=max(s["peak_cpu"], 4), disk=max(s["disk"], 3))
            else:
                s.update(idle_memory=max(s["idle_memory"], 2), peak_memory=max(s["peak_memory"], 3), idle_cpu=max(s["idle_cpu"], 1), peak_cpu=max(s["peak_cpu"], 3), disk=max(s["disk"], 2))
        if self.browser_present:
            s.update(idle_memory=max(s["idle_memory"], 2), peak_memory=max(s["peak_memory"], 3), peak_cpu=max(s["peak_cpu"], 3), disk=max(s["disk"], 1), network=max(s["network"], 2))
        if self.processes:
            lightweight = {"subprocess", "ffprobe", "yt-dlp", "yt_dlp", "aria2", "gallery-dl", "browser"}
            if "ffmpeg" in self.processes:
                s["peak_cpu"] = max(s["peak_cpu"], 3)
            elif self.processes - lightweight:
                s["peak_cpu"] = max(s["peak_cpu"], 2)
            else:
                s["peak_cpu"] = max(s["peak_cpu"], 1)
        if self.downloads or self.disk_download:
            s["network"] = max(s["network"], 3)
            s["disk"] = max(s["disk"], 2)
        if self.remote_llm:
            s["network"] = max(s["network"], 1)
        if self.cpu_task_risk:
            s["peak_cpu"] = max(s["peak_cpu"], self.cpu_task_risk)
        if self.full_reads:
            s["peak_memory"] = max(s["peak_memory"], 2)
        if self.pdf_present:
            s["peak_memory"] = max(s["peak_memory"], 2)
            s["peak_cpu"] = max(s["peak_cpu"], 2)
        elif self.image_present or self.audio_present:
            s["peak_memory"] = max(s["peak_memory"], 1)
            s["peak_cpu"] = max(s["peak_cpu"], 2)
        if self.db_present or self.vector_present:
            s["idle_memory"] = max(s["idle_memory"], 1)
            s["peak_memory"] = max(s["peak_memory"], 2 if self.vector_present else 1)
            s["disk"] = max(s["disk"], 1)
        if self.cache_unbounded:
            s["idle_memory"] = max(s["idle_memory"], 2)
            s["peak_memory"] = max(s["peak_memory"], 3)
        elif self.cache_bounded:
            s["idle_memory"] = max(s["idle_memory"], 1)
            s["peak_memory"] = max(s["peak_memory"], 1)
        if self.background_count:
            s["idle_cpu"] = max(s["idle_cpu"], 3 if self.background_unbounded else 1)
            s["idle_memory"] = max(s["idle_memory"], 2 if self.background_unbounded else 1)
            if self.background_unbounded:
                s["peak_cpu"] = max(s["peak_cpu"], 3)
        # Make sure every nonzero dimension has at least one relevant evidence item.
        kinds_for = {
            "idle_memory": {"idle_memory", "local_model", "external_process", "storage", "cache", "background", "disk"},
            "peak_memory": {"peak_memory", "local_model", "external_process", "cache", "media", "storage"},
            "idle_cpu": {"idle_cpu", "local_model", "background"},
            "peak_cpu": {"peak_cpu", "local_model", "external_process", "media", "download", "peak_memory", "background"},
            "disk": {"disk", "download", "storage", "local_model", "media"},
            "network": {"network", "remote_api", "download", "external_process", "local_model", "storage"},
        }
        for dim, val in s.items():
            if val and not any(ev["kind"] in kinds_for[dim] for ev in self.evidence):
                self.add(evidence(dim, self.root, None, dim, "静态规则命中，但缺少更细粒度运行时证据", "weak"))
        return s, {"size": size}

    def profile(self) -> dict[str, Any]:
        self.collect_python()
        self.build_reachable()
        self.inspect_runtime_configs()
        self.inspect()
        scores, extra = self.scores()
        confidence = 0.85
        if self.unknowns:
            confidence = 0.65
        if not self.ast_files or not self.reachable or any("AST" in x or "入口" in x for x in self.unknowns):
            confidence = 0.40
        if self.model_file_reference and self.model_dependency and not self.local_model:
            confidence = min(confidence, 0.65)
        if not any((self.local_model, self.model_dependency, self.remote_llm, self.processes, self.downloads, self.background_count, self.cpu_task_risk, self.cache_present, self.db_present, self.vector_present, self.browser_present, self.full_reads, self.pdf_present, self.image_present, self.audio_present)) and not any(scores.values()):
            # A quiet source is not proof of a high-confidence zero-risk result.
            confidence = min(confidence, 0.65)
        for ev in self.evidence:
            raw_file = Path(ev["file"])
            try:
                ev["file"] = "." if raw_file == self.root else raw_file.relative_to(self.root).as_posix()
            except ValueError:
                ev["file"] = raw_file.as_posix()
        self.evidence.sort(key=lambda e: (e["file"], e["line"], e["kind"], e["symbol"]))
        relevant = {
            "idle_memory": {"local_model", "external_process", "storage", "cache", "background", "disk"},
            "peak_memory": {"local_model", "external_process", "peak_memory", "cache", "media", "storage"},
            "idle_cpu": {"local_model", "background"},
            "peak_cpu": {"peak_cpu", "local_model", "external_process", "media", "download", "peak_memory"},
            "disk": {"disk", "download", "storage", "local_model", "media"},
            "network": {"remote_api", "download", "external_process", "local_model", "storage"},
        }
        selected: list[dict[str, Any]] = []
        priority_kinds = []
        if self.local_model:
            priority_kinds.append("local_model")
        if self.browser_present or self.processes:
            priority_kinds.append("external_process")
        if self.cpu_task_risk:
            priority_kinds.append("peak_cpu")
        if self.cache_unbounded:
            priority_kinds.append("cache")
        if self.background_unbounded:
            priority_kinds.append("background")
        if self.full_reads:
            priority_kinds.append("peak_memory")
        strength_order = {"strong": 0, "medium": 1, "weak": 2}
        for kind in priority_kinds:
            candidates = [ev for ev in self.evidence if ev["kind"] == kind]
            if candidates:
                best = min(
                    candidates,
                    key=lambda ev: (
                        0 if kind == "local_model" and "L4" in ev.get("detail", "") else 1,
                        strength_order.get(ev.get("strength", "medium"), 1),
                        ev["file"],
                        ev["line"],
                    ),
                )
                if best not in selected:
                    selected.append(best)
        for dim, value in scores.items():
            if not value:
                continue
            for ev in self.evidence:
                if ev["kind"] in relevant[dim] and ev not in selected:
                    selected.append(ev)
                    break
        for ev in self.evidence:
            if ev not in selected:
                selected.append(ev)
            if len(selected) >= 8:
                break
        profile = {
            "plugin_id": self.plugin["plugin_id"],
            "repo": self.plugin.get("repo", ""),
            "version": self.plugin.get("version", ""),
            "commit_sha": self.plugin.get("commit_sha", ""),
            "source_dir": self.root.name,
            "source_file_count": sum(1 for _ in iter_files(self.root)),
            "source_size_bytes": extra["size"],
            "reachable_python_file_count": len(self.reachable),
            "reachable_python_files": [p.relative_to(self.root).as_posix() for p in self.reachable[:MAX_REACHABLE]],
            "dependencies": parse_dependencies(self.root),
            "features": sorted(set(self.downloads) | ({"local_model"} if self.local_model else set()) | ({"model_dependency"} if self.model_dependency and not self.local_model else set()) | ({"model_file_reference"} if self.model_file_reference and not self.local_model else set()) | ({"remote_llm"} if self.remote_llm else set()) | ({"cpu_task"} if self.cpu_task_risk else set()) | ({"browser"} if self.browser_present else set()) | ({"browser_dependency"} if self.browser_dependency and not self.browser_present else set()) | ({"pdf"} if self.pdf_present else set()) | ({"image_processing"} if self.image_present else set()) | ({"audio_processing"} if self.audio_present else set()) | ({"storage"} if self.db_present else set()) | ({"vector_store"} if self.vector_present else set()) | ({"download_limit"} if self.size_limit_signals else set()) | ({"large_upload"} if self.upload_signals else set()))[:40],
            "levels": {k: level(v) for k, v in scores.items()},
            "scores": scores,
            "overall_level": level(max(scores["peak_memory"], scores["peak_cpu"])),
            "background_tasks": {"count": self.background_count, "bounded": not self.background_unbounded if self.background_count else True, "cleanup_detected": self.background_cleanup},
            "concurrency": {"bounded": not self.background_unbounded if self.background_count else True, "limit": min(self.sem_limits + self.worker_limits) if self.sem_limits or self.worker_limits else None, "limits": sorted(set(self.sem_limits + self.worker_limits))[:10], "unbounded_signals": self.background_unbounded},
            "cache": {"present": self.cache_present, "bounded": not self.cache_unbounded if self.cache_present else True, "unbounded_growth_possible": self.cache_unbounded},
            "external_processes": sorted(self.processes),
            "external_process_details": [{"name": ev["symbol"], "file": ev["file"], "line": ev["line"], "possibly_resident": bool(self.background_count or "server" in ev["symbol"].lower() or "popen" in ev["symbol"].lower())} for ev in self.evidence if ev["kind"] == "external_process"][:12],
            "runtime_downloads": sorted(self.downloads),
            "evidence": selected[:8],
            "unknowns": self.unknowns[:20],
            "confidence": confidence,
            "manual_review": False,
            "auto_resolved_low_confidence": confidence == 0.40,
        }
        return profile


def old_overall(old: dict[str, Any]) -> int:
    scores = old.get("scores", {}) if isinstance(old, dict) else {}
    return max([int(scores.get(k, 0) or 0) for k in ("peak_memory", "peak_cpu")] or [0])


def review_queue(profiles: dict[str, dict[str, Any]], old: dict[str, Any], observations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pid, p in profiles.items():
        score = p["scores"]
        old_p = old.get(pid, {}) if isinstance(old, dict) else {}
        delta = max(score["peak_memory"], score["peak_cpu"]) - old_overall(old_p)
        risk = (4 - p["confidence"]) * 5 + score["peak_memory"] * 3 + score["peak_cpu"] * 3 + score["network"] + score["disk"] + max(0, delta) * 2
        if p["confidence"] < 0.85 or p["overall_level"] in {"L3", "L4"} or p["runtime_downloads"] or p["external_processes"]:
            m = observations.get(pid, {})
            reasons = list(p["unknowns"][:2]) + p["external_processes"] + p["runtime_downloads"]
            if "local_model" in p["features"]:
                reasons.append("local_model")
            if "browser" in p["features"]:
                reasons.append("browser")
            if "cpu_task" in p["features"]:
                reasons.append("cpu_task")
            if p["cache"]["unbounded_growth_possible"]:
                reasons.append("unbounded_cache")
            if p["concurrency"]["unbounded_signals"]:
                reasons.append("unbounded_loop")
            if p["overall_level"] in {"L3", "L4"}:
                reasons.append(f"overall={p['overall_level']}")
            if p["confidence"] < 0.85:
                reasons.append(f"confidence={p['confidence']:.2f}")
            rows.append({"plugin_id": pid, "overall_level": p["overall_level"], "confidence": p["confidence"], "reason": sorted(set(reasons))[:6], "download_count": m.get("download_count", 0), "stars": m.get("stars", 0), "priority": round(risk + (m.get("download_count", 0) / 100000) + (m.get("stars", 0) / 1000), 4)})
    rows.sort(key=lambda x: (-x["priority"], x["plugin_id"]))
    return rows[:30]


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(args.root).resolve()
    source_root = Path(args.source).resolve()
    observations, candidates, docs = make_observation_indexes(root)
    mapped, mapping = map_sources(source_root, observations, candidates)
    profiles: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(mapped, 1):
        profile = SourceAnalyzer(item.pop("source_dir"), item).profile()
        profiles[profile["plugin_id"]] = profile
        if idx % 250 == 0:
            print(f"scanned {idx}/{len(mapped)}", file=sys.stderr)
    old = docs["old"]
    queue = review_queue(profiles, old, observations)
    queued_ids = {item["plugin_id"] for item in queue}
    for pid, profile in profiles.items():
        profile["manual_review"] = pid in queued_ids
    old_diffs = Counter()
    old_keyword_false_positive_count = 0
    for pid, p in profiles.items():
        before = old_overall(old.get(pid, {})) if isinstance(old, dict) else 0
        after = int(p["overall_level"][1:])
        old_diffs["upgraded" if after > before else "downgraded" if after < before else "unchanged"] += 1
        old_features = old.get(pid, {}).get("features", []) if isinstance(old, dict) and isinstance(old.get(pid, {}), dict) else []
        old_text = " ".join(str(x).lower() for x in old_features)
        if any(token in old_text for token in ("transformers", "local_model", "本地模型")) and "local_model" not in p["features"]:
            old_keyword_false_positive_count += 1
    signal_counts = {
        "local_model": sum("local_model" in p["features"] for p in profiles.values()),
        "browser": sum("browser" in p["features"] for p in profiles.values()),
        "ffmpeg": sum("ffmpeg" in p["external_processes"] for p in profiles.values()),
        "downloaders": sum(bool(p["runtime_downloads"]) for p in profiles.values()),
        "vector_store": sum("vector_store" in p["features"] for p in profiles.values()),
        "background_tasks": sum(int(p["background_tasks"]["count"]) > 0 for p in profiles.values()),
        "bounded_cache": sum(bool(p["cache"]["present"] and p["cache"]["bounded"]) for p in profiles.values()),
        "unbounded_cache": sum(bool(p["cache"]["unbounded_growth_possible"]) for p in profiles.values()),
        "bounded_concurrency": sum(bool(p["background_tasks"]["count"] and p["concurrency"]["bounded"]) for p in profiles.values()),
        "unbounded_concurrency": sum(bool(p["concurrency"]["unbounded_signals"]) for p in profiles.values()),
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "scan_mode": "local_source_static_read_only",
        "network_used": False,
        "plugin_code_executed": False,
        "source_root": str(source_root),
        "profile_count": len(profiles),
        "mapping": mapping,
        "old_profile_diff": dict(old_diffs),
        "old_keyword_false_positive_count": old_keyword_false_positive_count,
        "signal_counts": signal_counts,
        "manual_review_queue_count": len(queue),
    }
    out = {"$meta": meta, "profiles": profiles}
    queue_doc = {"$meta": {"schema_version": SCHEMA_VERSION, "generated_at": meta["generated_at"], "max_items": 30, "ordering": "risk first, then local market usage for ties"}, "items": queue}
    return out, queue_doc


def runtime_index(source_doc: dict[str, Any], fallback_doc: dict[str, Any]) -> dict[str, Any]:
    """Convert source-analysis records to the advisor's runtime index contract.

    Source-derived profiles take precedence. Market entries without an extracted
    source directory retain their previously validated metadata-only profile, so
    the runtime index remains complete without asking an LLM to fill every gap.
    """
    generated_at = str(source_doc.get("$meta", {}).get("generated_at") or now_iso())
    source_profiles = source_doc.get("profiles", {})
    old_profiles = fallback_doc.get("profiles", {}) if isinstance(fallback_doc, dict) else {}
    runtime_profiles: dict[str, dict[str, Any]] = {}
    for pid, raw in source_profiles.items() if isinstance(source_profiles, dict) else []:
        if not isinstance(raw, dict):
            continue
        task = raw.get("background_tasks", {}) if isinstance(raw.get("background_tasks"), dict) else {}
        if raw.get("concurrency", {}).get("unbounded_signals"):
            background = "yes"
        elif int(task.get("count") or 0) > 0:
            background = "likely"
        else:
            background = "no"
        evidence_rows = []
        for item in raw.get("evidence", []) if isinstance(raw.get("evidence"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence_rows.append(
                f"{item.get('kind', 'signal')}:{item.get('file', '.')}:{int(item.get('line') or 1)} {item.get('detail', '')}"[:2048]
            )
        runtime_profiles[pid] = {
            "plugin_id": pid,
            "version": str(raw.get("version") or ""),
            "commit_sha": str(raw.get("commit_sha") or ""),
            "levels": dict(raw.get("levels") or {}),
            "scores": dict(raw.get("scores") or {}),
            "features": [str(x) for x in raw.get("features", [])][:100],
            "external_processes": [str(x) for x in raw.get("external_processes", [])][:100],
            "background_tasks": background,
            "evidence": evidence_rows[:100],
            "unknowns": [str(x)[:2048] for x in raw.get("unknowns", [])][:100],
            "confidence": float(raw.get("confidence") or 0.0),
            "evidence_level": "local_source_static_ast",
            "scanned_at": generated_at,
        }
    fallback_count = 0
    if isinstance(old_profiles, dict):
        for pid, raw in old_profiles.items():
            if pid not in runtime_profiles and isinstance(raw, dict):
                runtime_profiles[pid] = raw
                fallback_count += 1
    evidence_counts = dict(Counter(str(p.get("evidence_level") or "unknown") for p in runtime_profiles.values()))
    canonical = json.dumps(runtime_profiles, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    meta = {
        "schema_version": 1,
        "generated_at": generated_at,
        "profile_count": len(runtime_profiles),
        "profiles_sha256": hashlib.sha256(canonical).hexdigest(),
        "scan_mode": "local_source_static_read_only_with_metadata_fallback",
        "source_code_downloaded": True,
        "plugin_code_executed": False,
        "network_used": False,
        "source_static_profile_count": len(runtime_profiles) - fallback_count,
        "metadata_fallback_profile_count": fallback_count,
        "evidence_counts": evidence_counts,
    }
    if any(name.startswith("github_") for name in evidence_counts):
        meta["commit_sha_kind"] = "github_commit_oid"
        meta["commit_binding_api"] = "github_list_commits_metadata"
    return {"$meta": meta, "profiles": runtime_profiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--source", default="")
    ap.add_argument("--profiles", default="")
    ap.add_argument("--queue", default="")
    ap.add_argument("--runtime-index", default="")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    args.source = args.source or str(root / "source_extracted")
    args.profiles = args.profiles or str(root / "data" / "source_resource_profiles.json")
    args.queue = args.queue or str(root / "data" / "source_resource_review_queue.json")
    args.runtime_index = args.runtime_index or str(root / "data" / "source_resource_index.json")
    profiles, queue = run(args)
    fallback = load_json(root / "data" / "resource_profiles.json", {})
    runtime = runtime_index(profiles, fallback)
    Path(args.profiles).parent.mkdir(parents=True, exist_ok=True)
    Path(args.profiles).write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.queue).write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.runtime_index).write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {profiles['$meta']['profile_count']} source profiles, {runtime['$meta']['profile_count']} runtime profiles; review queue {len(queue['items'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
